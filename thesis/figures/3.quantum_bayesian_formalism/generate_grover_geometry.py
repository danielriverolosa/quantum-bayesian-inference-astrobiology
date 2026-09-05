import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Configuración de estilo
plt.style.use('default')
fig, ax = plt.subplots(figsize=(8, 8))

# Ocultar ejes estándar y dibujar cruz en origen
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.axis('equal')
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)

# Ejes (Estados Base)
ax.annotate(r'$|\Psi_0\rangle$ (Background Subspace)', xy=(1.2, 0), xytext=(1.25, -0.05),
            fontsize=12, ha='center')
ax.annotate(r'$|\Psi_1\rangle$ (Target Subspace)', xy=(0, 1.2), xytext=(-0.1, 1.25),
            fontsize=12, ha='center')

# Dibujar el arco del círculo unitario
theta = np.linspace(0, np.pi/2, 100)
x_circ = np.cos(theta)
y_circ = np.sin(theta)
ax.plot(x_circ, y_circ, 'k--', alpha=0.3)

# Ángulo inicial theta
theta_0 = np.pi / 8  # 22.5 grados para que se vea claro

# Vector de Estado Inicial |Psi>
x_psi = np.cos(theta_0)
y_psi = np.sin(theta_0)
ax.annotate('', xy=(x_psi, y_psi), xytext=(0, 0),
            arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0, width=2, headwidth=8))
ax.text(x_psi + 0.05, y_psi + 0.05, r'$|\Psi\rangle = \mathcal{A}|0\rangle^{\otimes n}$', color='blue', fontsize=14, fontweight='bold')

# Reflexión 1: S_chi (sobre el eje |Psi_0>)
y_ref = -np.sin(theta_0)
ax.annotate('', xy=(x_psi, y_ref), xytext=(0, 0),
            arrowprops=dict(facecolor='gray', edgecolor='gray', shrink=0, width=1, headwidth=6, linestyle='dashed'))
ax.text(x_psi + 0.05, y_ref - 0.05, r'$S_\chi|\Psi\rangle$', color='gray', fontsize=12)

# Reflexión 2: S_0 (sobre el vector inicial |Psi>) -> Rotación total de 2*theta
theta_1 = 3 * theta_0
x_q = np.cos(theta_1)
y_q = np.sin(theta_1)
ax.annotate('', xy=(x_q, y_q), xytext=(0, 0),
            arrowprops=dict(facecolor='red', edgecolor='red', shrink=0, width=2, headwidth=8))
ax.text(x_q + 0.05, y_q + 0.05, r'$\mathcal{Q}|\Psi\rangle$', color='red', fontsize=14, fontweight='bold')

# Arcos de ángulos
# Arco theta
arc1 = patches.Arc((0,0), 0.5, 0.5, angle=0, theta1=0, theta2=np.degrees(theta_0), color='blue', linewidth=1.5)
ax.add_patch(arc1)
ax.text(0.28, 0.08, r'$\theta$', color='blue', fontsize=14)

# Arco 2*theta (Rotación de Grover)
arc2 = patches.Arc((0,0), 0.7, 0.7, angle=0, theta1=np.degrees(theta_0), theta2=np.degrees(theta_1), color='red', linewidth=1.5)
ax.add_patch(arc2)
ax.text(0.25, 0.35, r'$2\theta$', color='red', fontsize=14)

plt.title(r'Geometric Representation of the Grover Operator $\mathcal{Q}$', pad=20, fontsize=16, fontweight='bold')
plt.grid(False)

# Save figure to both thesis and figures directories
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
targets = [
    os.path.join(repo_root, 'thesis', 'figures', '3.quantum_bayesian_formalism', 'figure_grover_geometry.png'),
    os.path.join(repo_root, 'figures', 'results', 'figure_grover_geometry.png')
]

for t in targets:
    os.makedirs(os.path.dirname(t), exist_ok=True)
    plt.savefig(t, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {t}")
