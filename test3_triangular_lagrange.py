import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Necesario para 3D

# ==========================================
# 1. FÍSICA Y MATEMÁTICAS (Intactas)
# ==========================================
def reduced_lagrange_equations(t, state, mu, k):
    rho, v_rho, phi = state
    # Ecuaciones reducidas del PDF
    phi_dot = k / rho**2
    rho_ddot = (rho * phi_dot**2) - (mu / rho**2)
    return [v_rho, rho_ddot, phi_dot]

def get_positions_from_s3(s3, m1, m2, m3):
    M = m1 + m2 + m3
    # Relaciones geométricas del triángulo equilátero (Complejos)
    s1 = s3 * np.exp(1j * 2 * np.pi / 3)
    s2 = s3 * np.exp(-1j * 2 * np.pi / 3)
    # Calcular posiciones absolutas
    x1 = (m3 * s2 - m2 * s3) / M
    x2 = (m1 * s3 - m3 * s1) / M
    x3 = (m2 * s1 - m1 * s2) / M
    return x1, x2, x3

# ==========================================
# 2. CONFIGURACIÓN Y SOLVER
# ==========================================
if __name__ == "__main__":
    # Parámetros físicos
    G = 1.0
    m1, m2, m3 = 1.0, 1.0, 1.0
    mu = G * (m1 + m2 + m3)

    # Condiciones de la Órbita
    eccentricity = 0.5
    a = 10.0

    # Condiciones Iniciales (Periastro)
    rho_0 = a * (1 - eccentricity)
    v_rho_0 = 0.0
    phi_0 = 0.0
    k_const = np.sqrt(mu * a * (1 - eccentricity**2))

    # Solver
    T_period = 2 * np.pi * np.sqrt(a**3 / mu)
    t_eval = np.linspace(0, 2 * T_period, 1000)

    sol = solve_ivp(
        reduced_lagrange_equations, 
        (0, 2 * T_period), 
        [rho_0, v_rho_0, phi_0], 
        args=(mu, k_const), 
        rtol=1e-9, atol=1e-9,
        t_eval=t_eval
    )

    rho_t = sol.y[0]
    phi_t = sol.y[2]
    s3_t = rho_t * np.exp(1j * phi_t)

    # Obtener trayectorias complejas (Plano XY)
    x1_c, x2_c, x3_c = get_positions_from_s3(s3_t, m1, m2, m3)

    # Generar coordenada Z (Ceros para movimiento plano en 3D)
    z1 = np.zeros_like(x1_c.real)
    z2 = np.zeros_like(x2_c.real)
    z3 = np.zeros_like(x3_c.real)

    # Agrupar datos para facilitar el acceso en el bucle
    all_pos_3d = [
        (x1_c.real, x1_c.imag, z1),
        (x2_c.real, x2_c.imag, z2),
        (x3_c.real, x3_c.imag, z3)
    ]

    # ==========================================
    # 3. VISUALIZACIÓN 3D
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- Configuración de Ejes Fijos (SIN ZOOM) ---
    limit = a * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    ax.set_xlabel("X (Real)")
    ax.set_ylabel("Y (Imag)")
    ax.set_zlabel("Z")
    ax.set_title(f"Solución Triangular Lagrange (e={eccentricity})\nDinámica Reducida")

    # --- Definición de Colores PERSONALIZADOS ---
    color_c1 = '#E06C9F'  # Rosa/Magenta Pastel
    color_c2 = '#5DADE2'  # Azul Celeste
    color_c3 = '#A9DFBF'  # Verde Pálido / Lima suave
    
    colors = [color_c1, color_c2, color_c3]
    labels = ['Masa 1', 'Masa 2', 'Masa 3']
    
    lines = []
    points = []
    
    # Crear objetos iniciales con sus colores asignados
    for i in range(3):
        # Estela (Línea)
        l, = ax.plot([], [], [], '-', color=colors[i], alpha=0.8, lw=2, label=labels[i])
        lines.append(l)
        # Masa (Punto)
        p, = ax.plot([], [], [], 'o', color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        points.append(p)

    # Triángulo conector (Negro)
    triangle, = ax.plot([], [], [], 'k-', lw=1, alpha=0.5)
    
    # Centro de masas (Negro)
    ax.plot([0], [0], [0], 'k+', markersize=10, label='CM') 

    # Añadir leyenda
    ax.legend(loc='upper right')

    def update(frame):
        # 1. Obtener coordenadas del fotograma actual
        xs = [pos[0][frame] for pos in all_pos_3d]
        ys = [pos[1][frame] for pos in all_pos_3d]
        zs = [pos[2][frame] for pos in all_pos_3d]

        # 2. Actualizar el Triángulo (Geometría)
        # Conecta 1->2->3->1
        triangle.set_data(xs + [xs[0]], ys + [ys[0]])
        triangle.set_3d_properties(zs + [zs[0]])

        # 3. Actualizar Masas y Estelas
        for i in range(3):
            # Posición actual (Punto)
            points[i].set_data([xs[i]], [ys[i]])
            points[i].set_3d_properties([zs[i]])

            # Historia (Estela)
            # Usamos slicing sobre los arrays pre-calculados (sin error de índice)
            hist_x = all_pos_3d[i][0][:frame]
            hist_y = all_pos_3d[i][1][:frame]
            hist_z = all_pos_3d[i][2][:frame]

            lines[i].set_data(hist_x, hist_y)
            lines[i].set_3d_properties(hist_z)

        return points + lines + [triangle]

    # Vista de cámara estática (elevación 30 grados, azimut 45)
    ax.view_init(elev=30, azim=45)

    ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)
    plt.show()