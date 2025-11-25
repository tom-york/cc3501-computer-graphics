import pyglet
import click
import random
import numpy as np

from pathlib import Path
from collections import deque
from OpenGL import GL

from grafica.particle import Particle
from grafica.utils import load_pipeline


@click.command("lorenz", short_help='Simulacion del atractor de Lorenz')
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=600)
@click.option("--sigma", type=float, default=10.0)
@click.option("--rho", type=float, default=28.0)
@click.option("--beta", type=float, default=8/3)
@click.option("--n_particles", type=int, default=3)

def tarea(width, height, sigma, rho, beta, n_particles):
    win = pyglet.window.Window(width, height)

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    pipeline.use()
    GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)


    # Coleccion de particulas
    particles = deque(maxlen=n_particles)
    trail = deque(maxlen=70000)
    particle_data = None

    # Semilla para reproducibilidad
    random.seed(42)

    # Vista inicial (0 = xy, 1 = xz, 2 = yz)
    view = 0

    time = 0.0

    # Anadir limites para evitar que las particulas diverjan demasiado
    boundary = 50.0
    dt = 0.005


    def lorenz_attractor(coordinates):
        x, y, z = coordinates
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz], dtype=np.float32)

    def calculate_lorenz_rk4(particle, dt):
        p = particle.position
        k1 = lorenz_attractor(p)
        k2 = lorenz_attractor(p + 0.5 * dt * k1)
        k3 = lorenz_attractor(p + 0.5 * dt * k2)
        k4 = lorenz_attractor(p + dt * k3)
        new_position = p + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Reiniciar particula si se sale de los limites
        if np.any(np.isnan(new_position)) or np.any(np.abs(new_position) > boundary):
            new_position = create_initial_position()
        particle.velocity = k1
        particle.position = new_position

    # metodo de Euler
    def calculate_lorenz_euler(particle, dt):
        x, y, z = particle.position
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        new_position = particle.position + np.array([dx, dy, dz], dtype=np.float32) * dt
        # Reiniciar particula si se sale de los limites
        if np.any(np.isnan(new_position)) or np.any(np.abs(new_position) > boundary):
            new_position = create_initial_position()
        particle.velocity = np.array([dx, dy, dz], dtype=np.float32)
        particle.position = new_position
    
    def create_initial_position():
        return np.array([
            random.uniform(-15, 15), 
            random.uniform(-20, 20), 
            random.uniform(5, 40)
        ], dtype=np.float32)
    
    def create_particle():
        position = create_initial_position()
        return Particle(position)

    def transform_view(position_array):
        scaled_array = position_array.copy()

        # Centrar el eje Z
        scaled_array[:, 2] = (scaled_array[:, 2] - 25.0)

        if view == 0:
            xy = scaled_array[:, :2]
            return xy
        elif view == 1:
            xz = scaled_array[:, [0, 2]]
            return xz
        else:
            yz = scaled_array[:, 1:]
            return yz
        
    def calculate_trayectory(particle):
        if view == 0:
            angle_rad = np.arctan2(particle.velocity[1], particle.velocity[0])
        elif view == 1:
            angle_rad = np.arctan2(particle.velocity[2], particle.velocity[0])
        else:
            angle_rad = np.arctan2(particle.velocity[2], particle.velocity[1])
        angle_deg = np.degrees(angle_rad) % 360.0
        return angle_deg
    
    def update_particle_system(dt):
        # Incrementar tiempo global
        nonlocal time, particle_data, particles
        time += dt

        # Emitir nuevas partículas si no se ha alcanzado el límite
        available_n_particles = n_particles - len(particles)
        if available_n_particles:
            for _ in range(available_n_particles):
                particles.append(create_particle())

        # Actualizar todas las partículas
        for particle in list(particles):
            calculate_lorenz_rk4(particle, dt)
            angle = calculate_trayectory(particle)
            trail.append([particle.position.copy(), angle])
        
        # Actualizar datos en GPU
        if particle_data is not None:
            particle_data.delete()
            particle_data = None
        
        num_particles = len(particles)
        
        # Preparar datos para renderizar
        if num_particles > 0:
            # Llenar datos
            positions_2d = transform_view(np.array([p[0] for p in trail], dtype=np.float32))
            angles = np.array([p[1] for p in trail], dtype=np.float32)

            # Saturacion
            x = positions_2d[:,0]
            y = positions_2d[:,1]

            grid_size = 800

            # Crear histograma de densidad
            H, xedges, yedges = np.histogram2d(x, y, bins=grid_size, range=[[-boundary,boundary],[-boundary,boundary]])

            x_idx = np.searchsorted(xedges, x) - 1
            y_idx = np.searchsorted(yedges, y) - 1

            x_idx = np.clip(x_idx, 0, grid_size-1)
            y_idx = np.clip(y_idx, 0, grid_size-1)

            densities = H[x_idx, y_idx]
            densities = np.log1p(densities) # suavizar las densidades
            densities /= densities.max()
            #densities = np.power(densities, 0.6)

            # Crear escala para ajustar la vista
            scale = 30.0
            positions = positions_2d / scale

            # Crear vertex_list
            particle_data = pipeline.vertex_list(
                len(trail),
                pyglet.gl.GL_POINTS,
                position=("f", positions.flatten()),
                hue=("f", angles.flatten()),
                saturation=("f", densities.flatten())
            )

    @win.event
    def on_draw():
        win.clear()
        nonlocal particle_data
        if particle_data:
            particle_data.draw(pyglet.gl.GL_POINTS)

    @win.event
    def on_key_press(symbol, modifiers):
        nonlocal view, particle_data
        if symbol == pyglet.window.key.SPACE:
            view = (view + 1) % 3
            print(f"View changed to {['XY', 'XZ', 'YZ'][view]}")
            # limpiar datos en caso de...
            if particle_data is not None:
                particle_data.delete()
                particle_data = None
        
    pyglet.clock.schedule_interval(update_particle_system, dt)
    pyglet.app.run()
