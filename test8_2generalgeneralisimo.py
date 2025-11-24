import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


masses = np.array([6, 6, 6]) 
G_const = 1.0



q_initial = np.array([
    [-3.0, 0.0,              0.0],  # Cuerpo 1
    [ 3.0, 0.0,              0.0],  # Cuerpo 2
    [ 0.0, 3 * np.sqrt(3),   0.0]   # Cuerpo 3 
])

v_initial = np.array([
    [ np.sqrt(3)/2, -0.5, 0.0], 
    [ 0.0,           1.0, 0.0], 
    [ -np.sqrt(3)/2, -0.5, 0.0] 
])

# Momentos y Estado Inicial
p_initial = (v_initial.T * masses).T 
initial_state = np.concatenate([q_initial.flatten(), p_initial.flatten()])

# Tiempo 
T_end = 86.875231 
t_span = (0, T_end)
t_eval = np.linspace(0, T_end, 4000)

# ==========================================
# 2. SOLVER
# ==========================================
def hamiltonian_equations(t, state, G, m):
    q = state[:9].reshape(3, 3) 
    p = state[9:].reshape(3, 3)
    dqdt = np.zeros_like(q)
    dpdt = np.zeros_like(p)
    
    for i in range(3):
        dqdt[i] = p[i] / m[i]
    
    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = q[j] - q[i]
                dist = np.linalg.norm(r_vec)
                dpdt[i] += G * m[i] * m[j] * r_vec / (dist**3)
    
    return np.concatenate([dqdt.flatten(), dpdt.flatten()])

print("Calculando solución II.16.B...")
solution = solve_ivp(
    hamiltonian_equations, t_span, initial_state, 
    method='RK45', t_eval=t_eval, args=(G_const, masses),
    rtol=1e-9, atol=1e-9
)
q_sol = solution.y[:9, :].T 

x1, y1, z1 = q_sol[:, 0], q_sol[:, 1], q_sol[:, 2]
x2, y2, z2 = q_sol[:, 3], q_sol[:, 4], q_sol[:, 5]
x3, y3, z3 = q_sol[:, 6], q_sol[:, 7], q_sol[:, 8]

# ==========================================
# 3. CLASE PARA EL ZOOM
# ==========================================
class ZoomHandler:
    def __init__(self, ax, base_scale=1.1):
        self.ax = ax
        self.base_scale = base_scale
        self.xlim = ax.get_xlim()
        self.ylim = ax.get_ylim()
        self.zlim = ax.get_zlim()

    def zoom(self, event):
        if event.button == 'up':
            scale_factor = 1 / self.base_scale 
        elif event.button == 'down':
            scale_factor = self.base_scale
        else:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        cur_zlim = self.ax.get_zlim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        new_depth = (cur_zlim[1] - cur_zlim[0]) * scale_factor

        x_center = (cur_xlim[1] + cur_xlim[0]) / 2
        y_center = (cur_ylim[1] + cur_ylim[0]) / 2
        z_center = (cur_zlim[1] + cur_zlim[0]) / 2

        self.ax.set_xlim([x_center - new_width / 2, x_center + new_width / 2])
        self.ax.set_ylim([y_center - new_height / 2, y_center + new_height / 2])
        self.ax.set_zlim([z_center - new_depth / 2, z_center + new_depth / 2])
        self.ax.figure.canvas.draw_idle()

# ==========================================
# 4. VISUALIZACIÓN
# ==========================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# --- TÍTULO GENERAL ---
ax.set_title('Simulación del Problema de los Tres Cuerpos ', fontsize=16, pad=20)

# Configuración de límites
max_range = 1.5 
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range/2, max_range/2])
ax.set_xlabel('X')
ax.set_ylabel('Y')

# --- MANEJADOR DE ZOOM ---
zoom_handler = ZoomHandler(ax)
fig.canvas.mpl_connect('scroll_event', zoom_handler.zoom)

# --- PALETA DE COLORES ---
color_c1 = '#E06C9F'  # Rosa/Magenta Pastel (Cuerpo 1)
color_c2 = '#5DADE2'  # Azul Celeste (Cuerpo 2)
color_c3 = '#A9DFBF'  # Verde Pálido / Lima suave (Cuerpo 3 - Borde exterior)

# Dibujar Rieles
ax.plot(x1, y1, z1, color=color_c1, lw=0.8, alpha=0.4)
ax.plot(x2, y2, z2, color=color_c2, lw=0.8, alpha=0.4)
ax.plot(x3, y3, z3, color=color_c3, lw=0.8, alpha=0.4)

# Elementos móviles
mass1, = ax.plot([], [], [], 'o', color=color_c1, ms=10, markeredgecolor='white', label='M1 ')
mass2, = ax.plot([], [], [], 'o', color=color_c2, ms=10, markeredgecolor='white', label='M2 ')
mass3, = ax.plot([], [], [], 'o', color=color_c3, ms=10, markeredgecolor='white', label='M3')

# Estelas
trail1, = ax.plot([], [], [], '-', color=color_c1, lw=2)
trail2, = ax.plot([], [], [], '-', color=color_c2, lw=2)
trail3, = ax.plot([], [], [], '-', color=color_c3, lw=2)

# --- LEYENDA DE CONDICIONES INICIALES ---
info_text = (
    r"$\bf{Condiciones\ Iniciales\ (V.1.A):}$" + "\n" +
    f"M1 : $m={masses[0]}$, $q={np.round(q_initial[0], 2)}$, $v={np.round(v_initial[0], 2)}$\n"
    f"M2 : $m={masses[1]}$, $q={np.round(q_initial[1], 2)}$, $v={np.round(v_initial[1], 2)}$\n"
    f"M3 : $m={masses[2]}$, $q={np.round(q_initial[2], 2)}$, $v={np.round(v_initial[2], 2)}$"
)

ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend(loc='upper right')

speed_step = 8 
trail_length = 150

def animate(i):
    mass1.set_data([x1[i]], [y1[i]])
    mass1.set_3d_properties([z1[i]])
    
    mass2.set_data([x2[i]], [y2[i]])
    mass2.set_3d_properties([z2[i]])
    
    mass3.set_data([x3[i]], [y3[i]])
    mass3.set_3d_properties([z3[i]])
    
    start = max(0, i - trail_length)
    
    trail1.set_data(x1[start:i], y1[start:i])
    trail1.set_3d_properties(z1[start:i])
    
    trail2.set_data(x2[start:i], y2[start:i])
    trail2.set_3d_properties(z2[start:i])
    
    trail3.set_data(x3[start:i], y3[start:i])
    trail3.set_3d_properties(z3[start:i])
    
    return mass1, mass2, mass3, trail1, trail2, trail3

ani = FuncAnimation(
    fig, animate, 
    frames=range(0, len(t_eval), speed_step), 
    interval=1,
    blit=False 
)

plt.show()