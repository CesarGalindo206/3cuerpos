import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. FÍSICA: Cálculo Analítico de Euler
# ==========================================

def solve_euler_quintic(m1, m2, m3):
    C5 = m1 + m2
    C4 = 3*m1 + 2*m2
    C3 = 3*m1 + m2
    C2 = -(m2 + 3*m3)
    C1 = -(2*m2 + 3*m3)
    C0 = -(m2 + m3)
    
    coeffs = [C5, C4, C3, C2, C1, C0]
    roots = np.roots(coeffs)
    
    real_roots = roots[np.isreal(roots)].real
    lambda_sol = real_roots[real_roots > 0][0]
    return lambda_sol

def get_euler_collinear_ics(e, a_semimajor, m1, m2, m3, G):
    lam = solve_euler_quintic(m1, m2, m3)
    
    r12_norm = 1.0
    x_temp = np.array([-r12_norm, 0.0, lam * r12_norm])
    
    x_cm_temp = (m1*x_temp[0] + m2*x_temp[1] + m3*x_temp[2]) / (m1 + m2 + m3)
    x_centered = x_temp - x_cm_temp
    
    r12 = np.abs(x_temp[1] - x_temp[0])
    r13 = np.abs(x_temp[2] - x_temp[0])
    
    acc_g_1 = G * m2 / r12**2 + G * m3 / r13**2
    r1_cm = np.abs(x_centered[0])
    
    omega_sq = acc_g_1 / r1_cm
    
    scale_pos = a_semimajor * (1 - e)
    q_final_x = x_centered * scale_pos
    
    r12_real = np.abs(q_final_x[1] - q_final_x[0])
    r13_real = np.abs(q_final_x[2] - q_final_x[0])
    r1_cm_real = np.abs(q_final_x[0])
    
    acc_real = G * m2 / r12_real**2 + G * m3 / r13_real**2
    omega_real = np.sqrt(acc_real / r1_cm_real)
    
    velocity_factor = np.sqrt(1 + e)
    
    q_initial = np.zeros((3, 3))
    v_initial = np.zeros((3, 3))
    
    for i in range(3):
        q_initial[i] = [q_final_x[i], 0.0, 0.0]
        vy = omega_real * q_final_x[i] * velocity_factor
        v_initial[i] = [0.0, vy, 0.0]
        
    return q_initial, v_initial

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
                dpdt[i] += G * m[i] * m[j] * r_vec / (dist**3 + 1e-12)
    
    return np.concatenate([dqdt.flatten(), dpdt.flatten()])

# ==========================================
# 2. CONFIGURACIÓN DE SIMULACIÓN
# ==========================================
G_const = 1.0
masses = np.array([10.0, 2.0, 5.0]) 
excentricidad = 0.4
a_semimajor = 5.0

q_initial, v_initial = get_euler_collinear_ics(excentricidad, a_semimajor, 
                                              masses[0], masses[1], masses[2], G_const)

p_initial = (v_initial.T * masses).T 
initial_state = np.concatenate([q_initial.flatten(), p_initial.flatten()])

T_estimate = 2 * np.pi * np.sqrt(a_semimajor**3 / (G_const * np.sum(masses))) * 2.5
t_span = (0, T_estimate)
t_eval = np.linspace(0, T_estimate, 4000)

print(f"Lambda (Euler): {solve_euler_quintic(masses[0], masses[1], masses[2]):.4f}")
print("Calculando simulación...")

solution = solve_ivp(
    hamiltonian_equations, t_span, initial_state, 
    method='RK45', t_eval=t_eval, args=(G_const, masses),
    rtol=1e-11, atol=1e-11
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
        if event.button == 'up': scale_factor = 1 / self.base_scale 
        elif event.button == 'down': scale_factor = self.base_scale
        else: return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        cur_zlim = self.ax.get_zlim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        new_depth = (cur_zlim[1] - cur_zlim[0]) * scale_factor

        x_center = (cur_xlim[1] + cur_xlim[0]) / 2
        y_center = (cur_ylim[1] + cur_ylim[0]) / 2
        z_center = (cur_zlim[1] + cur_zlim[0]) / 2

        self.ax.set_xlim([x_center - new_width/2, x_center + new_width/2])
        self.ax.set_ylim([y_center - new_height/2, y_center + new_height/2])
        self.ax.set_zlim([z_center - new_depth/2, z_center + new_depth/2])
        self.ax.figure.canvas.draw_idle()

# ==========================================
# 4. VISUALIZACIÓN (Corregida)
# ==========================================

# Función auxiliar para formatear listas numéricas limpias (sin np.float64)
def fmt_list(arr):
    """Convierte un array numpy a una lista de floats estándar de Python redondeados."""
    return [round(float(x), 2) for x in arr]

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection='3d')

# --- TÍTULO CORREGIDO ---
# Convertimos 'masses' a lista de floats nativos para que se imprima bonito
masses_list = [float(m) for m in masses]
ax.set_title(f'Solución Colineal de Euler (Analítica)\n(e={excentricidad}, Masas={masses_list})', fontsize=14, pad=20)

limit = a_semimajor * 2.0
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit/2, limit/2])
ax.set_xlabel('X')
ax.set_ylabel('Y')

zoom_handler = ZoomHandler(ax)
fig.canvas.mpl_connect('scroll_event', zoom_handler.zoom)

# --- PALETA DE COLORES ---
color_c1 = '#E06C9F'  # Rosa/Magenta Pastel
color_c2 = '#5DADE2'  # Azul Celeste
color_c3 = '#A9DFBF'  # Verde Pálido

# Elementos gráficos
line_connector, = ax.plot([], [], [], 'k-', lw=0.8, alpha=0.4, label='Eje Colineal')

ax.plot(x1, y1, z1, color=color_c1, lw=0.8, alpha=0.3)
ax.plot(x2, y2, z2, color=color_c2, lw=0.8, alpha=0.3)
ax.plot(x3, y3, z3, color=color_c3, lw=0.8, alpha=0.3)

sizes = [10 + (m/np.max(masses))*10 for m in masses]
mass1, = ax.plot([], [], [], 'o', color=color_c1, ms=sizes[0], markeredgecolor='white', label='M1')
mass2, = ax.plot([], [], [], 'o', color=color_c2, ms=sizes[1], markeredgecolor='white', label='M2')
mass3, = ax.plot([], [], [], 'o', color=color_c3, ms=sizes[2], markeredgecolor='white', label='M3')

trail1, = ax.plot([], [], [], '-', color=color_c1, lw=2)
trail2, = ax.plot([], [], [], '-', color=color_c2, lw=2)
trail3, = ax.plot([], [], [], '-', color=color_c3, lw=2)

# --- LEYENDA CORREGIDA ---
# Usamos fmt_list para limpiar los vectores
info_text = (
    r"$\bf{C\acute{a}lculo\ Anal\acute{i}tico\ (Euler):}$" + "\n" +
    f"M1 ($m={masses_list[0]}$): $q={fmt_list(q_initial[0])}$, $v={fmt_list(v_initial[0])}$\n"
    f"M2 ($m={masses_list[1]}$): $q={fmt_list(q_initial[1])}$, $v={fmt_list(v_initial[1])}$\n"
    f"M3 ($m={masses_list[2]}$): $q={fmt_list(q_initial[2])}$, $v={fmt_list(v_initial[2])}$"
)

ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.legend(loc='upper right')

speed_step = 5
trail_length = 120

def animate(i):
    xs = [x1[i], x2[i], x3[i]]
    ys = [y1[i], y2[i], y3[i]]
    zs = [z1[i], z2[i], z3[i]]
    
    line_connector.set_data(xs, ys)
    line_connector.set_3d_properties(zs)

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
    
    return mass1, mass2, mass3, trail1, trail2, trail3, line_connector

ani = FuncAnimation(fig, animate, frames=range(0, len(t_eval), speed_step), interval=1, blit=False)
plt.show()