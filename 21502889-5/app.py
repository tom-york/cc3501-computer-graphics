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
        half = size / 2.0
        eps = 0.016  # sticker offset outward from body
        body_col = body_color if body_color is not None else np.array([0.08, 0.08, 0.08], dtype=np.float32)
        # body transforms
        graph.add_transform(f"{name}_rot_face", tr.identity())
        graph.add_transform(f"{name}_rot", tr.identity())
        graph.add_transform(f"{name}_geom", tr.scale(size, size, size))
        graph.add_transform(f"{name}_trans", tr.translate(*pos))
        graph.add_mesh_instance(f"{name}_mesh", "cube", "basic_pipeline", color=body_col)

        sticker_plate_scale = (size * 0.9, size * 0.9, size * 0.04)

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
            if key == "+X":
                graph.add_transform(s_rot, tr.rotationY(-np.pi / 2))
            if key == "+Y":
                graph.add_transform(s_rot, tr.rotationX(np.pi / 2))
            if key == "Z":
                graph.add_transform(s_rot, tr.identity())
            if key == "-X":
                graph.add_transform(s_rot, tr.rotationY(np.pi / 2))
            if key == "-Y":
                graph.add_transform(s_rot, tr.rotationX(-np.pi / 2))
            if key == "-Z":
                graph.add_transform(s_rot, tr.rotationY(np.pi))
            graph.add_transform(s_geom, tr.scale(*sticker_plate_scale))
            graph.add_transform(s_trans, tr.translate(*offset))
            graph.add_mesh_instance(s_mesh, "cube", "basic_pipeline", color=np.array(color, dtype=np.float32))
        return {
            "rot": f"{name}_rot",
            "rot_face": f"{name}_rot_face",
            "geom": f"{name}_geom",
            "trans": f"{name}_trans",
            "mesh": f"{name}_mesh",
            "stickers": [(f"{name}_sticker_{k}_mesh", k) for k in sticker_colors.keys()] if sticker_colors else []
        }
    
    cube_size = 0.2
    spacing = 0.1

    WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    YELLOW = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ORANGE = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    BLUE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    BLACK = np.array([0.06, 0.06, 0.06], dtype=np.float32)

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

                cube_array.append(((ix, iy, iz), (create_cubie(graph, name, pos, cube_size, BLACK, sticker_colors))))

    # ================================================================
    # JERARQUÍA (Padre -> Hijo)
    # ================================================================

    # Generar matriz para llevar cuenta de la estructura del cubo
    cube_state = np.empty((3, 3, 3), dtype=object)

    cube_rot_angles = {f"cubie_{ix+1}_{iy+1}_{iz+1}": 0.0
                        for ix in [-1, 0, 1]
                        for iy in [-1, 0, 1]
                        for iz in [-1, 0, 1]}

    # Estructura central
    for ((x, y, z), cube) in cube_array:
        graph.add_edge("rubik", cube["rot_face"])
        graph.add_edge(cube["rot_face"], cube["trans"])
        graph.add_edge(cube["trans"], cube["rot"])
        graph.add_edge(cube["rot"], cube["geom"])
        graph.add_edge(cube["geom"], cube["mesh"])
        cube_state[x+1, y+1, z+1] = f"cubie_{x+1}_{y+1}_{z+1}"
        for (sticker_mesh, key) in cube["stickers"]:
            # Extraer el nombre base del sticker
            sticker_base = sticker_mesh.replace("_mesh", "")
            s_rot = f"{sticker_base}_rot"
            s_geom = f"{sticker_base}_geom"
            s_trans = f"{sticker_base}_trans"
            
            # Conectar sticker a la transformación del cubie
            graph.add_edge(cube["rot"], s_trans)
            graph.add_edge(s_trans, s_rot)
            graph.add_edge(s_rot, s_geom)
            graph.add_edge(s_geom, sticker_mesh)


    def x_rotate_layer_90(graph, layer_index, direction=1):
        """Rotate layer at x=layer_index by 90 degrees clockwise."""
        nonlocal cube_state, cube_rot_angles
        temp_matrix = cube_state.copy()
        angle = (np.pi / 2) * direction
        for iy in [-1, 0, 1]:
            for iz in [-1, 0, 1]:
                name = cube_state[layer_index, iy+1, iz+1]
                cube_rot_angles[name] = (cube_rot_angles.get(name) + angle) % (2 * np.pi)
                graph.add_transform(f"{name}_rot_face", tr.rotationX(cube_rot_angles[name]))
                # Update cube_state matrix
                new_iy = -iz
                new_iz = iy
                temp_matrix[layer_index, new_iy+1, new_iz+1] = name
        cube_state = temp_matrix

    def y_rotate_layer_90(graph, layer_index, direction=1):
        """Rotate layer at y=layer_index by 90 degrees clockwise."""
        nonlocal cube_state, cube_rot_angles
        temp_matrix = cube_state.copy()
        angle = (np.pi / 2) * direction
        for ix in [-1, 0, 1]:
            for iz in [-1, 0, 1]:
                name = cube_state[ix+1, layer_index, iz+1]
                cube_rot_angles[name] = (cube_rot_angles.get(name) + angle) % (2 * np.pi)
                graph.add_transform(f"{name}_rot_face", tr.rotationY(cube_rot_angles[name]))
                # Update cube_state matrix
                new_ix = iz
                new_iz = -ix
                temp_matrix[new_ix+1, layer_index, new_iz+1] = name
        cube_state = temp_matrix

    def z_rotate_layer_90(graph, layer_index, direction=1):
        """Rotate layer at z=layer_index by 90 degrees clockwise."""
        nonlocal cube_state, cube_rot_angles
        temp_matrix = cube_state.copy()
        angle = (np.pi / 2) * direction
        for ix in [-1, 0, 1]:
            for iy in [-1, 0, 1]:
                name = cube_state[ix+1, iy+1, layer_index]
                cube_rot_angles[name] = (cube_rot_angles.get(name) + angle) % (2 * np.pi)
                graph.add_transform(f"{name}_rot_face", tr.rotationZ(cube_rot_angles[name]))
                # Update cube_state matrix
                new_ix = -iy
                new_iy = ix
                temp_matrix[new_ix+1, new_iy+1, layer_index] = name
        cube_state = temp_matrix
    

    # ================================================================
    # RENDER LOOP
    # ================================================================

    camera_distance = 1.5
    camera_theta = np.pi / 4  # ángulo horizontal
    camera_phi = np.pi / 4    # ángulo vertical
    mouse_pressed = False
    last_mouse_x = 0
    last_mouse_y = 0

    @window.event
    def on_key_press(symbol, modifiers):
        shift = modifiers & pyglet.window.key.MOD_SHIFT

        if symbol == pyglet.window.key.Q:
            x_rotate_layer_90(graph, 0, direction=1 if not shift else -1)
        elif symbol == pyglet.window.key.W:
            x_rotate_layer_90(graph, 1, direction=1 if not shift else -1)
        elif symbol == pyglet.window.key.E:
            x_rotate_layer_90(graph, 2, direction=1 if not shift else -1)
        
        elif symbol == pyglet.window.key.A:
            y_rotate_layer_90(graph, 0, direction=1 if not shift else -1)
        elif symbol == pyglet.window.key.S:
            y_rotate_layer_90(graph, 1, direction=1 if not shift else -1)
        elif symbol == pyglet.window.key.D:
            y_rotate_layer_90(graph, 2, direction=1 if not shift else -1)

        elif symbol == pyglet.window.key.Z:
            z_rotate_layer_90(graph, 0, direction=1 if not shift else -1)
        elif symbol == pyglet.window.key.X:
            z_rotate_layer_90(graph, 1, direction=1 if not shift else -1)
        elif symbol == pyglet.window.key.C:
            z_rotate_layer_90(graph, 2, direction=1 if not shift else -1)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        nonlocal mouse_pressed, last_mouse_x, last_mouse_y
        if button == pyglet.window.mouse.LEFT:
            mouse_pressed = True
            last_mouse_x = x
            last_mouse_y = y

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        nonlocal mouse_pressed
        if button == pyglet.window.mouse.LEFT:
            mouse_pressed = False

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        nonlocal camera_theta, camera_phi
        if mouse_pressed:
            camera_theta += dx * 0.01
            camera_phi = np.clip(camera_phi + dy * 0.01, 0.1, np.pi - 0.1)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        nonlocal camera_distance
        camera_distance = np.clip(camera_distance - scroll_y * 0.1, 0.5, 5.0)

    @window.event
    def on_draw():
        window.clear()
        GL.glEnable(GL.GL_DEPTH_TEST)

        eye_x = camera_distance * np.sin(camera_phi) * np.cos(camera_theta)
        eye_y = camera_distance * np.cos(camera_phi)
        eye_z = camera_distance * np.sin(camera_phi) * np.sin(camera_theta)
        
        eye = np.array([eye_x, eye_y, eye_z])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        
        view = tr.lookAt(eye, center, up)
        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        graph.register_view_transform(view)
        graph.set_global_attributes(projection=projection)
        graph.render()
    pyglet.app.run()


if __name__ == "__main__":
    main()
