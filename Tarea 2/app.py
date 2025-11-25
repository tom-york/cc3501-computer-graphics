import os.path
import numpy as np
import pyglet
import pyglet.gl as GL
import click
from pathlib import Path
import grafica.transformations as tr
from grafica.utils import load_pipeline
from grafica.scenegraph import Scenegraph
from pyglet.window import key

@click.command()
@click.option("--width", default=800, help="Ancho de la ventana")
@click.option("--height", default=600, help="Alto de la ventana")
def main(width, height):
    # Crear ventana y pipeline
    window = pyglet.window.Window(width, height, "Personaje articulado")
    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    # Crear grafo de escena
    graph = Scenegraph("personaje")
    graph.load_and_register_mesh('cube', "assets/cube.off", fix_normals=True)
    graph.register_pipeline('basic_pipeline', pipeline)

    # ================================================================
    # CONSTRUCCIÓN DEL PERSONAJE
    # ================================================================

    # -------------------
    # TORSO (base central)
    # -------------------
    graph.add_transform("torso_rotation", tr.identity())
    graph.add_transform("torso_geometry", tr.scale(0.3, 0.6, 0.15))
    graph.add_transform("torso_translation", tr.translate(0.0, 0.0, 0.0))
    graph.add_mesh_instance("torso_mesh", "cube", "basic_pipeline",
                            color=np.array([0.8, 0.4, 0.2], dtype=np.float32))

    # -------------------
    # CABEZA
    # -------------------
    graph.add_transform("head_translation", tr.translate(0.0, 0.5, 0.0))
    graph.add_transform("head_geometry", tr.scale(0.25, 0.25, 0.25))
    graph.add_mesh_instance("head_mesh", "cube", "basic_pipeline",
                            color=np.array([1.0, 0.8, 0.6], dtype=np.float32))

    # ================================================================
    # BRAZOS (izquierdo y derecho, cada uno con 2 segmentos)
    # ================================================================
    # --- Brazo izquierdo superior ---
    graph.add_transform("left_shoulder", tr.translate(-0.245, 0.18, 0.0))
    graph.add_transform("left_upper_arm_rot", tr.rotationZ(0))
    graph.add_transform("left_upper_arm_geom", tr.scale(0.1, 0.3, 0.1))
    graph.add_mesh_instance("left_upper_arm_mesh", "cube", "basic_pipeline",
                            color=np.array([0.0, 0.6, 1.0], dtype=np.float32))

    # --- Brazo izquierdo inferior (antebrazo) ---
    graph.add_transform("left_elbow", tr.translate(0.0, -0.3, 0.0))
    graph.add_transform("left_lower_arm_rot", tr.rotationZ(0))
    graph.add_transform("left_lower_arm_geom", tr.scale(0.08, 0.25, 0.08))
    graph.add_mesh_instance("left_lower_arm_mesh", "cube", "basic_pipeline",
                            color=np.array([0.2, 0.8, 1.0], dtype=np.float32))

    # --- Brazo derecho superior ---
    graph.add_transform("right_shoulder",tr.translate(0.245, 0.18, 0.0))
    graph.add_transform("right_upper_arm_rot", tr.rotationZ(0))
    graph.add_transform("right_upper_arm_geom", tr.scale(0.1, 0.3, 0.1))
    graph.add_mesh_instance("right_upper_arm_mesh", "cube", "basic_pipeline",
                            color=np.array([0.0, 0.6, 1.0], dtype=np.float32))

    # --- Brazo derecho inferior (antebrazo) ---
    graph.add_transform("right_elbow", tr.translate(0.0, -0.3, 0.0))
    graph.add_transform("right_lower_arm_rot", tr.rotationZ(0))
    graph.add_transform("right_lower_arm_geom", tr.scale(0.08, 0.25, 0.08))
    graph.add_mesh_instance("right_lower_arm_mesh", "cube", "basic_pipeline",
                            color=np.array([0.2, 0.8, 1.0], dtype=np.float32))

    # ================================================================
    # PIERNAS (izquierda y derecha, cada una con 2 segmentos)
    # ================================================================
    # --- Pierna izquierda superior ---
    graph.add_transform("left_hip", tr.translate(-0.115, -0.58, 0.0))
    graph.add_transform("left_upper_leg_rot", tr.rotationZ(0))
    graph.add_transform("left_upper_leg_geom", tr.scale(0.12, 0.35, 0.12))
    graph.add_mesh_instance("left_upper_leg_mesh", "cube", "basic_pipeline",
                            color=np.array([0.4, 0.2, 0.8], dtype=np.float32))

    # --- Pierna izquierda inferior ---
    graph.add_transform("left_knee", tr.translate(0.0, -0.35, 0.0))
    graph.add_transform("left_lower_leg_rot", tr.rotationZ(0))
    graph.add_transform("left_lower_leg_geom", tr.scale(0.1, 0.3, 0.1))
    graph.add_mesh_instance("left_lower_leg_mesh", "cube", "basic_pipeline",
                            color=np.array([0.6, 0.4, 1.0], dtype=np.float32))

    # --- Pierna derecha superior ---
    graph.add_transform("right_hip", tr.translate(0.115, -0.58, 0.0))
    graph.add_transform("right_upper_leg_rot", tr.rotationZ(0))
    graph.add_transform("right_upper_leg_geom", tr.scale(0.12, 0.35, 0.12))
    graph.add_mesh_instance("right_upper_leg_mesh", "cube", "basic_pipeline",
                            color=np.array([0.4, 0.2, 0.8], dtype=np.float32))

    # --- Pierna derecha inferior ---
    graph.add_transform("right_knee", tr.translate(0.0, -0.35, 0.0))
    graph.add_transform("right_lower_leg_rot", tr.rotationZ(0))
    graph.add_transform("right_lower_leg_geom", tr.scale(0.1, 0.3, 0.1))
    graph.add_mesh_instance("right_lower_leg_mesh", "cube", "basic_pipeline",
                            color=np.array([0.6, 0.4, 1.0], dtype=np.float32))

    # ================================================================
    # JERARQUÍA (Padre -> Hijo)
    # ================================================================

    # Estructura central
    graph.add_edge("personaje", "torso_translation")
    graph.add_edge("torso_translation", "torso_rotation")
    graph.add_edge("torso_rotation", "torso_geometry")
    graph.add_edge("torso_geometry", "torso_mesh")

    # Cabeza
    graph.add_edge("torso_rotation", "head_translation")
    graph.add_edge("head_translation", "head_geometry")
    graph.add_edge("head_geometry", "head_mesh")

    # Brazo izquierdo
    graph.add_edge("torso_rotation", "left_shoulder")
    graph.add_edge("left_shoulder", "left_upper_arm_rot")
    graph.add_edge("left_upper_arm_rot", "left_upper_arm_geom")
    graph.add_edge("left_upper_arm_geom", "left_upper_arm_mesh")

    graph.add_edge("left_upper_arm_rot", "left_elbow")
    graph.add_edge("left_elbow", "left_lower_arm_rot")
    graph.add_edge("left_lower_arm_rot", "left_lower_arm_geom")
    graph.add_edge("left_lower_arm_geom", "left_lower_arm_mesh")

    # Brazo derecho
    graph.add_edge("torso_rotation", "right_shoulder")
    graph.add_edge("right_shoulder", "right_upper_arm_rot")
    graph.add_edge("right_upper_arm_rot", "right_upper_arm_geom")
    graph.add_edge("right_upper_arm_geom", "right_upper_arm_mesh")

    graph.add_edge("right_upper_arm_rot", "right_elbow")
    graph.add_edge("right_elbow", "right_lower_arm_rot")
    graph.add_edge("right_lower_arm_rot", "right_lower_arm_geom")
    graph.add_edge("right_lower_arm_geom", "right_lower_arm_mesh")

    # Pierna izquierda
    graph.add_edge("torso_rotation", "left_hip")
    graph.add_edge("left_hip", "left_upper_leg_rot")
    graph.add_edge("left_upper_leg_rot", "left_upper_leg_geom")
    graph.add_edge("left_upper_leg_geom", "left_upper_leg_mesh")

    graph.add_edge("left_upper_leg_rot", "left_knee")
    graph.add_edge("left_knee", "left_lower_leg_rot")
    graph.add_edge("left_lower_leg_rot", "left_lower_leg_geom")
    graph.add_edge("left_lower_leg_geom", "left_lower_leg_mesh")

    # Pierna derecha
    graph.add_edge("torso_rotation", "right_hip")
    graph.add_edge("right_hip", "right_upper_leg_rot")
    graph.add_edge("right_upper_leg_rot", "right_upper_leg_geom")
    graph.add_edge("right_upper_leg_geom", "right_upper_leg_mesh")
    
    graph.add_edge("right_upper_leg_rot", "right_knee")
    graph.add_edge("right_knee", "right_lower_leg_rot")
    graph.add_edge("right_lower_leg_rot", "right_lower_leg_geom")
    graph.add_edge("right_lower_leg_geom", "right_lower_leg_mesh")

    # ================================================================
    projection = tr.perspective(60, float(width) / float(height), 0.1, 20)

    def set_initial_pose(graph):
        nonlocal projection
        projection = tr.perspective(60, float(width) / float(height), 0.1, 20)
        graph.add_transform("left_upper_arm_rot", tr.rotationZ(0))
        graph.add_transform("right_upper_arm_rot", tr.rotationZ(0))
        
        graph.add_transform("left_shoulder", tr.translate(-0.245, 0.18, 0.0))
        graph.add_transform("right_shoulder", tr.translate(0.245, 0.18, 0.0))

        graph.add_transform("left_lower_arm_rot", tr.rotationZ(0))
        graph.add_transform("left_elbow", tr.translate(0.0, -0.3, 0.0))

        graph.add_transform("right_lower_arm_rot", tr.rotationZ(0))
        graph.add_transform("right_elbow", tr.translate(0.0, -0.3, 0.0))

        graph.add_transform("left_upper_leg_rot", tr.rotationZ(0))
        graph.add_transform("left_hip", tr.translate(-0.115, -0.58, 0.0))

        graph.add_transform("right_upper_leg_rot", tr.rotationZ(0))
        graph.add_transform("right_hip", tr.translate(0.115, -0.58, 0.0))

        graph.add_transform("left_lower_leg_rot", tr.rotationZ(0))
        graph.add_transform("right_lower_leg_rot", tr.rotationZ(0))

    def first_pose(graph):
        nonlocal projection
        set_initial_pose(graph)
        # Pose saludando
        graph.add_transform("left_upper_arm_rot", tr.rotationZ(-2.5))
        graph.add_transform("left_shoulder", tr.translate(-0.245, 0.35, 0.0))

        graph.add_transform("left_lower_arm_rot", tr.rotationZ(-0.5))
        graph.add_transform("left_elbow", tr.translate(-0.05, -0.25, 0.0))

        projection = tr.perspective(110, float(width) / float(height), 0.1, 20)
        return tr.lookAt(np.array([0.0, 1.0, 0.7]), 
                         np.array([0.0, 0.0, 0.0]), 
                         np.array([0.0, 1.0, 0.0]))


    def second_pose(graph):
        nonlocal projection
        set_initial_pose(graph)
        # Pose caminando
        graph.add_transform("left_upper_arm_rot", tr.rotationX(-0.75))

        graph.add_transform("right_upper_arm_rot", tr.rotationX(0.8))

        graph.add_transform("left_upper_leg_rot", tr.rotationX(-0.5))
        graph.add_transform("left_hip", tr.translate(-0.115, -0.50, 0.12))

        graph.add_transform("right_upper_leg_rot", tr.rotationX(0.5))
        graph.add_transform("right_hip", tr.translate(0.115, -0.50, -0.128))

        projection = tr.perspective(30, float(width) / float(height), 0.1, 20)
        return tr.lookAt(np.array([2.0, 3.0, 3.5]), 
                         np.array([0.0, 0.0, 0.0]), 
                         np.array([0.0, 1.0, 0.0]))


    def third_pose(graph):
        nonlocal projection
        set_initial_pose(graph)
        # Pose de estrella
        graph.add_transform("left_upper_arm_rot", tr.rotationZ(-2.1))
        graph.add_transform("left_shoulder", tr.translate(-0.35, 0.35, 0.0))

        graph.add_transform("right_upper_arm_rot", tr.rotationZ(2.1))
        graph.add_transform("right_shoulder", tr.translate(0.35, 0.35, 0.0))

        graph.add_transform("left_upper_leg_rot", tr.rotationZ(-0.7))
        graph.add_transform("right_hip", tr.translate(0.235, -0.50, 0.0))

        graph.add_transform("right_upper_leg_rot", tr.rotationZ(0.7))
        graph.add_transform("left_hip", tr.translate(-0.235, -0.50, 0.0))

        projection = tr.perspective(90, float(width) / float(height), 0.1, 20)
        return tr.lookAt(np.array([1.0, 0.8, 1.5]), 
                         np.array([0.0, 0.0, 0.0]), 
                         np.array([0.5, -0.5, 0.0]))


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

        graph.register_view_transform(view)
        graph.set_global_attributes(projection=projection)
        graph.render()

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal current_pose
        if symbol == key.SPACE:
            current_pose = (current_pose + 1) % len(poses)



    pyglet.app.run()


if __name__ == "__main__":
    main()
