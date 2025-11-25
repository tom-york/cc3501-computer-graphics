import os.path
import numpy as np
import pyglet
import pyglet.gl as GL
import click
from pathlib import Path
import grafica.transformations as tr
from grafica.utils import load_pipeline
from grafica.scenegraph import Scenegraph


@click.command()
@click.option("--width", default=800, help="Ancho de la ventana")
@click.option("--height", default=600, help="Alto de la ventana")
def main(width, height):
    # Crear ventana y pipeline
    window = pyglet.window.Window(width, height, "Cubo Rubik")
    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    # Crear grafo de escena
    graph = Scenegraph("rubik")
    graph.load_and_register_mesh('cube', "assets/cube.off", fix_normals=True)
    graph.register_pipeline('basic_pipeline', pipeline)

    # ================================================================
    # CONSTRUCCIÓN DEL CUBO
    # ================================================================

    # Cada cubo deberia tener distintos colores por cara, por lo que se crea esta funcion para no tener
    # que recurrir a texturas mas complicadas, usando "stickers" para los colores
    def create_cubie(graph, name, pos, size=0.1, body_color=None, sticker_colors=None):
        """Create one cubie at pos=(ix,iy,iz) with body and stickers.
       pos is world translation (x,y,z). size is cube scale (edge length).
       sticker_colors is dict mapping face keys ['+X','-X','+Y','-Y','+Z','-Z'] to 3-float arrays.
        """
        sx = size
        half = sx / 2.0
        eps = 0.006  # sticker offset outward from body
        body_col = body_color if body_color is not None else np.array([0.08, 0.08, 0.08], dtype=np.float32)
        # body transforms
        graph.add_transform(f"{name}_rot", tr.identity())
        graph.add_transform(f"{name}_geom", tr.scale(sx, sx, sx))
        graph.add_transform(f"{name}_trans", tr.translate(*pos))
        graph.add_mesh_instance(f"{name}_mesh", "cube", "basic_pipeline", color=body_col)

        sticker_plate_scale = (sx * 0.9, sx * 0.9, sx * 0.04)

        faces = {
        "+X": (half + eps, 0.0, 0.0), # (1,0,0)
        "-X": (-half - eps, 0.0, 0.0), # (-1,0,0)
        "+Y": (0.0, half + eps, 0.0), # (0,1,0)
        "-Y": (0.0, -half - eps, 0.0), # (0,-1,0)
        "+Z": (0.0, 0.0, half + eps), # (0,0,1)
        "-Z": (0.0, 0.0, -half - eps), # (0,0,-1)
        }
        for key, offset in faces.items():
            color = None if sticker_colors is None else sticker_colors.get(key, None)
            if color is None:
                continue
            # sticker node names
            s_rot = f"{name}_sticker_{key}_rot"
            s_geom = f"{name}_sticker_{key}_geom"
            s_trans = f"{name}_sticker_{key}_trans"
            s_mesh = f"{name}_sticker_{key}_mesh"

            graph.add_transform(s_rot, tr.identity())
            graph.add_transform(s_geom, tr.scale(*sticker_plate_scale))
            graph.add_transform(s_trans, tr.translate(*offset))
            graph.add_mesh_instance(s_mesh, "cube", "basic_pipeline", color=np.array(color, dtype=np.float32))
        return {
            "rot": f"{name}_rot",
            "geom": f"{name}_geom",
            "trans": f"{name}_trans",
            "mesh": f"{name}_mesh",
            "stickers": [f"{name}_sticker_{k}_mesh" for k in sticker_colors.keys()] if sticker_colors else []
        }
    
    cube_size = 0.2
    spacing = 0.02

    WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    YELLOW = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ORANGE = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    BLUE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    BLACK = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    cube_array = []

    # Crear cubos iterativamente
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            for iz in [-1, 0, 1]:
                pos = (
                    ix * (cube_size + spacing),
                    iy * (cube_size + spacing),
                    iz * (cube_size + spacing),
                )
                name = f"cubie_{ix+1}_{iy+1}_{iz+1}"
                # Definir colores de stickers segun posicion
                sticker_colors = {}
                if ix == 1:  sticker_colors["+X"] = BLUE
                if ix == -1: sticker_colors["-X"] = GREEN
                if iy == 1:  sticker_colors["+Y"] = WHITE
                if iy == -1: sticker_colors["-Y"] = YELLOW
                if iz == 1:  sticker_colors["+Z"] = RED
                if iz == -1: sticker_colors["-Z"] = ORANGE

                create_cubie(graph, name, pos, cube_size, BLACK, sticker_colors)

    # ================================================================
    # JERARQUÍA (Padre -> Hijo)
    # ================================================================

    # Generar matriz para llevar cuenta de la estructura del cubo
    cube_state = np.empty((3, 3, 3), dtype=object)

    # Estructura central
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            for iz in [-1, 0, 1]:
                name = f"cubie_{ix+1}_{iy+1}_{iz+1}"
                cube_state[ix+1, iy+1, iz+1] = name
                # Conectar nodos del cubo al grafo
                graph.add_edge("rubik", f"{name}_rot")
                graph.add_edge(f"{name}_rot", f"{name}_geom")
                graph.add_edge(f"{name}_geom", f"{name}_trans")
                graph.add_edge(f"{name}_trans", f"{name}_mesh")

                # Conectar stickers
                for key in ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]:
                    sticker_mesh_name = f"{name}_sticker_{key}_mesh"
                    if sticker_mesh_name in graph.nodes:
                        graph.add_edge(f"{name}_trans", f"{name}_sticker_{key}_rot")
                        graph.add_edge(f"{name}_sticker_{key}_rot", f"{name}_sticker_{key}_geom")
                        graph.add_edge(f"{name}_sticker_{key}_geom", f"{name}_sticker_{key}_trans")
                        graph.add_edge(f"{name}_sticker_{key}_trans", sticker_mesh_name)

    # ================================================================
    
    def first_pose(graph):
        """"Pose saludando"""
        graph.add_transform("left_upper_arm_rot", tr.rotationZ(-2.5))
        graph.add_transform("left_shoulder", tr.translate(-0.245, 0.35, 0.0))
        graph.add_transform("right_shoulder", tr.translate(0.245, 0.18, 0.0))

        graph.add_transform("right_upper_arm_rot", tr.rotationZ(0))


        graph.add_transform("left_lower_arm_rot", tr.rotationZ(-0.5))
        graph.add_transform("left_elbow", tr.translate(-0.05, -0.25, 0.0))

        graph.add_transform("right_lower_arm_rot", tr.rotationZ(0))
        

        graph.add_transform("left_upper_leg_rot", tr.rotationZ(0))
        graph.add_transform("left_hip", tr.translate(-0.115, -0.58, 0.0))

        graph.add_transform("right_upper_leg_rot", tr.rotationZ(0))
        graph.add_transform("right_hip", tr.translate(0.115, -0.58, 0.0))

        graph.add_transform("left_lower_leg_rot", tr.rotationZ(0))
        graph.add_transform("right_lower_leg_rot", tr.rotationZ(0))
        return tr.lookAt(np.array([0.0, 0.5, 3.0]), 
                         np.array([0.0, 0.0, 0.0]), 
                         np.array([0.0, 1.0, 0.0]))


    def second_pose(graph):
        """Pose caminando"""
        graph.add_transform("left_upper_arm_rot", tr.rotationX(-0.75))
        graph.add_transform("left_shoulder", tr.translate(-0.245, 0.18, 0.0))

        graph.add_transform("right_upper_arm_rot", tr.rotationX(0.8))
        graph.add_transform("right_shoulder", tr.translate(0.245, 0.18, 0.0))

        graph.add_transform("left_lower_arm_rot", tr.rotationZ(0))
        graph.add_transform("left_elbow", tr.translate(0.0, -0.3, 0.0))

        graph.add_transform("right_lower_arm_rot", tr.rotationZ(0))


        graph.add_transform("left_upper_leg_rot", tr.rotationX(-0.5))
        graph.add_transform("left_hip", tr.translate(-0.115, -0.50, 0.12))


        graph.add_transform("right_upper_leg_rot", tr.rotationX(0.5))
        graph.add_transform("right_hip", tr.translate(0.115, -0.50, -0.128))


        graph.add_transform("left_lower_leg_rot", tr.rotationZ(0))
        graph.add_transform("right_lower_leg_rot", tr.rotationZ(0))
        return tr.lookAt(np.array([2.0, 1.0, 3.0]), 
                         np.array([0.0, 0.0, 0.0]), 
                         np.array([0.0, 1.0, 0.0]))


    def third_pose(graph):
        graph.add_transform("left_upper_arm_rot", tr.rotationZ(-10))
        graph.add_transform("right_upper_arm_rot", tr.rotationZ(10))
        graph.add_transform("left_lower_arm_rot", tr.rotationZ(20))
        graph.add_transform("right_lower_arm_rot", tr.rotationZ(-20))
        graph.add_transform("left_upper_leg_rot", tr.rotationZ(-15))
        graph.add_transform("right_upper_leg_rot", tr.rotationZ(15))
        graph.add_transform("left_lower_leg_rot", tr.rotationZ(5))
        graph.add_transform("right_lower_leg_rot", tr.rotationZ(-5))
        return tr.lookAt(np.array([-2.0, 1.0, 3.0]), 
                         np.array([0.0, 0.5, 0.0]), 
                         np.array([0.0, 1.0, 0.0]))


    poses = [first_pose, second_pose, third_pose]
    current_pose = 0




    # ================================================================
    # RENDER LOOP
    # ================================================================
    @window.event
    def on_draw():
        window.clear()
        GL.glEnable(GL.GL_DEPTH_TEST)

        view = poses[current_pose](graph)
        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        graph.register_view_transform(view)
        graph.set_global_attributes(projection=projection)
        graph.render()

    @window.event
    def on_key_press(symbol, modifiers):
        from pyglet.window import key
        nonlocal current_pose
        if symbol == key.SPACE:
            current_pose = (current_pose + 1) % len(poses)



    pyglet.app.run()


if __name__ == "__main__":
    main()
