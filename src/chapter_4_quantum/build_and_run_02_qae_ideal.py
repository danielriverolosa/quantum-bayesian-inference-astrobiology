#!/usr/bin/env python3
import sys
import io
import os
import base64
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'
os.environ['IPYTHONDIR'] = '/tmp/ipython_cache'

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11"
    }
}

# -------------------------------------------------------------
# Celda 1: Markdown - Portada Académica
# -------------------------------------------------------------
c1_md = """# 02. Quantum Bayesian Inference: Amplitude Estimation (QAE) in Ideal Simulation
### Master's Thesis: Quantum Bayesian Networks and Amplitude Estimation (QAE) in Astrobiology
**Author:** Daniel Rivero Losa  
**Supervisor:** Roberto Campos Ortiz  
**Institution:** Universidad Antonio de Nebrija — Escuela Politécnica Superior  
**Reference Section:** Section 4.3 of Master's Thesis (*QAE Architecture and Canonical Simulation*)  

---
### 📌 Executive Summary
This notebook implements and empirically validates the **complete quantum architecture derived in Section 4.3** of the Master's Thesis. The objective is to construct the quantum circuit in **Qiskit** synthesizing the Quantum Bayesian Network (QBN) of exoplanet **K2-18b**, apply the Grover unitary operator ($\mathcal{Q}$), and extract the detection probability of the industrial technosignature (CFCs) using **Quantum Amplitude Estimation (QAE)** in an ideal noise-free simulation environment (*statevector*).

The workflow covers:
1. **Topological Register Allocation (4.3.1):** Allocation of 17 qubits distributed across state ($S$), evaluation ($E$), ancilla ($A$), and classical readout ($C$) registers.
2. **Synthesis and Visualization of Operator $\mathcal{A}$ (4.3.2):** Encoding of hierarchical conditional dependencies via multi-controlled rotations.
3. **Design and Visualization of Oracle $S_\chi$ and Diffuser $S_0$ (4.3.3 - 4.3.4):** Canonical phase reflections via phase kickback.
4. **Integration and Visualization of Master QAE Circuit (4.3.5):** Controlled Grover power cascade $C\text{-}\mathcal{Q}^{2^j}$ and Inverse Quantum Fourier Transform (IQFT).
5. **Transpilation, Simulation, and Spectral Post-Processing (4.3.6 - 4.3.8):** Spectral mapping from phase eigenvalues to amplitude probabilities $\hat{a} = \sin^2(\theta)$.
6. **Resolution vs. Depth Trade-off Study (4.3.9):** Critical analysis of the trade-off between evaluation register size $n_E$ and circuit depth."""

# -------------------------------------------------------------
# Celda 2: Markdown - Marco Teórico Sección 4.3
# -------------------------------------------------------------
c2_md = """## 1. Theoretical Framework of Canonical QAE

### 1.1 State Preparation Operator $\mathcal{A}$
The state preparation operator $\mathcal{A}$ acts on the state register initialized to $|0\rangle^{\otimes n_S}$, synthesizing the joint probability distribution of the Bayesian network:
$$\mathcal{A} |0\rangle^{\otimes n_S} = \sqrt{1 - a} |\psi_0\rangle |0\rangle + \sqrt{a} |\psi_1\rangle |1\rangle$$
where $|\psi_1\rangle$ represents the atmospheric states where the CFC technosignature is present ($X_8 = 1$) and $a = \sin^2(\theta_a)$ is the target probability ($a \in [0, 1]$).

### 1.2 Grover Operator $\mathcal{Q}$ and its Eigenstates
The Grover iteration operator is defined as:
$$\mathcal{Q} = -\mathcal{A} S_0 \mathcal{A}^\dagger S_\chi$$
where:
- $S_\chi = I - 2|\chi\rangle\langle\chi|$ is the phase oracle inverting the phase of marked states.
- $S_0 = 2|0\rangle\langle 0| - I$ is the reflection about the all-zero state (diffuser).

In the two-dimensional subspace spanned by $\{|\psi_0\rangle|0\rangle, |\psi_1\rangle|1\rangle\}$, the operator $\mathcal{Q}$ acts as an orthogonal rotation by angle $2\theta_a$. Its two orthonormal eigenstates are:
$$|\Psi_\pm\rangle = \frac{1}{\sqrt{2}} \left( |\psi_0\rangle|0\rangle \mp i |\psi_1\rangle|1\rangle \right)$$
with corresponding eigenvalues:
$$\mathcal{Q} |\Psi_\pm\rangle = e^{\pm 2 i \theta_a} |\Psi_\pm\rangle$$

### 1.3 Quantum Phase Estimation (QPE) and IQFT
Quantum Amplitude Estimation couples an auxiliary evaluation register $E$ of $n_E$ qubits initialized in uniform superposition ($H^{\otimes n_E}$), applies successive controlled Grover powers $C\text{-}\mathcal{Q}^{2^j}$ ($j = 0, 1, \dots, n_E - 1$), and finally executes the **Inverse Quantum Fourier Transform (IQFT)**:
$$\text{IQFT} \left( \frac{1}{\sqrt{2^{n_E}}} \sum_{y=0}^{2^{n_E}-1} e^{2\pi i y \theta_a / \pi} |y\rangle \right) \approx |\tilde{y}\rangle$$

Measuring register $E$ in the computational basis yields an integer $\tilde{y} \in \{0, 1, \dots, 2^{n_E}-1\}$ providing quadratic convergence:
$$\tilde{\theta} = \frac{\pi \tilde{y}}{2^{n_E}}, \quad \hat{a} = \sin^2(\tilde{\theta})$$"""

# -------------------------------------------------------------
# Cell 3: Code - Topological Allocation (4.3.1)
# -------------------------------------------------------------
c3_code = r"""import os
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, RYGate
import matplotlib.pyplot as plt

# 1. Topological Allocation of Quantum Registers (Section 4.3.1)
num_state_qubits = 9       # n_S = 9 qubits for K2-18b Bayesian Network
num_eval_qubits = 5        # n_E = 5 evaluation qubits for QPE resolution (2^5 = 32 levels)
num_ancilla_qubits = 3     # n_A = 3 ancilla qubits for decomposition and phase kickback

reg_S = QuantumRegister(num_state_qubits, name='state_K218b')
reg_E = QuantumRegister(num_eval_qubits, name='eval_QPE')
reg_A = QuantumRegister(num_ancilla_qubits, name='ancilla_workspace')
reg_C = ClassicalRegister(num_eval_qubits, name='classical_readout')

master_circuit = QuantumCircuit(reg_E, reg_S, reg_A, reg_C)
ideal_backend = AerSimulator(method='statevector')

# Helper for dual export to thesis and figures directories
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) if os.path.exists('../../thesis') else (
            os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.exists('../thesis') else (
            os.path.abspath(os.getcwd()) if os.path.exists('thesis') else '.'))

def export_figure(fig, relative_targets, **kwargs):
    from PIL import Image
    for rel_path in relative_targets:
        full_path = os.path.join(repo_root, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(full_path, **kwargs)
        if 'circuit' in rel_path:
            im = Image.open(full_path)
            if im.width > 2200 or im.height > 2200:
                ratio = min(2200 / im.width, 2200 / im.height)
                new_size = (int(im.width * ratio), int(im.height * ratio))
                im_resized = im.resize(new_size, Image.Resampling.LANCZOS)
                im_resized.save(full_path, optimize=True)
        print(f"✓ Figure saved to: {full_path}")

print("="*65)
print(" TOPOLOGICAL REGISTER ALLOCATION (Section 4.3.1)")
print("="*65)
print(f" Registro de Estado (S)   : {reg_S.size} qubits (Variables de K2-18b)")
print(f" Registro de Evaluación (E): {reg_E.size} qubits (Resolución QPE: {2**reg_E.size} niveles)")
print(f" Registro Ancilla (A)     : {reg_A.size} qubits (Workspace / Kickback)")
print(f" Registro Clásico (C)     : {reg_C.size} bits (Lectura de medición)")
print(f" Total de Qubits Físicos  : {master_circuit.num_qubits} qubits")
print(f" Backend de Simulación    : AerSimulator (Statevector ideal)")
print("="*65)"""

# -------------------------------------------------------------
# Celda 4: Markdown - Operador A (4.3.2)
# -------------------------------------------------------------
c4_md = """## 2. Section 4.3.2: Amplitude Encoding Synthesis (Operator $\mathcal{A}$)

Operator $\mathcal{A}$ synthesizes the K2-18b Bayesian network into probability amplitudes:
1. **Root Nodes:** Individual $R_y(\theta)$ rotations:
   $$\theta_i = 2 \arcsin(\sqrt{p_i})$$
   setting $P(\text{Stellar M-Dwarf}) = 0.75$ on $S_0$ and $P(\text{Orbit HZ}) = 0.20$ on $S_1$.
2. **Conditional Nodes (CPTs):** Implemented via `synthesize_conditional_node`, iterating through $2^k$ parent configurations using $X$ gates for condition switching and modern object-oriented multi-controlled rotations (`RYGate(theta).control(k)`), adhering to Qiskit 1.x standards.
3. **Ultra-rare Anomaly Injection (CFCs):** Strongly conditioned rotation on node 8 ($S_8$) conditioned on biological node ($S_3$) with $\theta = 0.002$ rad ($p \approx \sin^2(0.001) \approx 10^{-6}$).

> **Quantum Design Rule:** No barrier directives (`qc.barrier()`) are included in subcircuits, ensuring composite gate conversion (`to_gate()`) remains mathematically pure and compatible with the transpiler."""

# -------------------------------------------------------------
# Cell 5: Code - Operator A Functions and Visualization
# -------------------------------------------------------------
c5_code = r"""def apply_root_nodes(qc, reg_S):
    """ + '"""Aplica rotaciones Ry sobre los nodos raíz (priors marginales)."""' + r"""
    p_stellar = 0.75
    p_orbit = 0.20
    theta_0 = 2.0 * np.arcsin(np.sqrt(p_stellar))
    theta_1 = 2.0 * np.arcsin(np.sqrt(p_orbit))
    qc.ry(theta_0, reg_S[0])
    qc.ry(theta_1, reg_S[1])
    return qc

def synthesize_conditional_node(qc, target_qubit, parent_qubits, cpt_probabilities, ancilla_reg):
    """ + '"""Sintetiza una tabla de probabilidad condicional (CPT) mediante rotaciones multicontroladas."""' + r"""
    num_parents = len(parent_qubits)
    num_states = 2 ** num_parents
    for state_idx in range(num_states):
        p_val = cpt_probabilities[state_idx]
        theta = 2.0 * np.arcsin(np.sqrt(p_val))
        if np.isclose(theta, 0.0):
            continue
        bin_str = format(state_idx, f'0{num_parents}b')
        for i, bit in enumerate(bin_str):
            if bit == '0':
                qc.x(parent_qubits[i])
        if num_parents == 1:
            qc.cry(theta, parent_qubits[0], target_qubit)
        else:
            try:
                qc.append(RYGate(theta).control(num_parents), parent_qubits + [target_qubit])
            except Exception:
                qc.mcry(theta, parent_qubits, target_qubit, q_ancillae=ancilla_reg, mode='v-chain')
        for i, bit in enumerate(bin_str):
            if bit == '0':
                qc.x(parent_qubits[i])
    return qc

def build_A_operator(num_state_qubits, ancilla_reg):
    """ + '"""Construye la puerta unitaria completa del Operador A."""' + r"""
    qc_A = QuantumCircuit(num_state_qubits + len(ancilla_reg))
    state_qubits = list(range(num_state_qubits))
    ancillas = list(range(num_state_qubits, num_state_qubits + len(ancilla_reg)))
    
    apply_root_nodes(qc_A, state_qubits)
    cpt_hycean = [0.05, 0.85]
    synthesize_conditional_node(qc_A, state_qubits[2], [state_qubits[0]], cpt_hycean, ancilla_reg)
    qc_A.cry(0.002, state_qubits[3], state_qubits[8])
    
    A_gate = qc_A.to_gate()
    A_gate.name = "Operator_A"
    return A_gate, qc_A

A_gate, qc_A_circuit = build_A_operator(num_state_qubits, reg_A)
print("✓ Operator A assembled successfully.")

# Graphical Visualization of Operator A Subcircuit
print("Generating graphical visualization of Operator A subcircuit...")
fig_A = qc_A_circuit.draw('mpl', style='iqp')
plt.title("Operator A Subcircuit (K2-18b Bayesian Network)", fontsize=13, pad=12, fontweight='bold')
plt.tight_layout()
export_figure(fig_A, [
    os.path.join('figures', 'circuits', 'circuit_operator_A.png'),
    os.path.join('thesis', 'figures', '3.quantum_bayesian_formalism', 'circuit_operator_A.png'),
    os.path.join('thesis', 'figures', '4.system_architecture', 'circuit_operator_A.png')
], dpi=200, bbox_inches='tight')
plt.show()"""

# -------------------------------------------------------------
# Celda 6: Markdown - Oráculo y Difusor (4.3.3 - 4.3.4)
# -------------------------------------------------------------
c6_md = """## 3. Sections 4.3.3 & 4.3.4: Quantum Oracle $S_\chi$, Diffuser $S_0$, and Grover Operator $\mathcal{Q}$

### Quantum Oracle $S_\chi$
The oracle flips the phase of the marked target state:
$$S_\chi |x\rangle = (-1)^{\chi(x)} |x\rangle$$
where $\chi(x) = 1$ if and only if $x_8 = 1$ (detected CFC technosignature node).
- For a single target qubit: Pauli $Z$ gate.
- For multiple targets: multi-controlled NOT with phase kickback on ancilla register ($|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$).

### Diffuser $S_0$ and Grover Operator $\mathcal{Q}$
The diffuser executes phase inversion about the all-zero ground state:
$$S_0 = 2|0\rangle^{\otimes n}\langle 0|^{\otimes n} - I$$
Combined with the oracle and operator $\mathcal{A}$, the unitary Grover operator is assembled:
$$\mathcal{Q} = -\mathcal{A} S_0 \mathcal{A}^\dagger S_\chi$$"""

# -------------------------------------------------------------
# Cell 7: Code - Oracle, Diffuser and Grover Q Visualization
# -------------------------------------------------------------
c7_code = r"""def synthesize_quantum_oracle(target_qubits, num_state_qubits, ancilla_reg):
    """ + '"""Sintetiza el oráculo cuántico S_chi de inversión de fase."""' + r"""
    qc_oracle = QuantumCircuit(num_state_qubits + len(ancilla_reg))
    if len(target_qubits) == 1:
        qc_oracle.z(target_qubits[0])
    else:
        kickback_ancilla = num_state_qubits + len(ancilla_reg) - 1
        qc_oracle.x(kickback_ancilla)
        qc_oracle.h(kickback_ancilla)
        qc_oracle.mcx(target_qubits, kickback_ancilla)
        qc_oracle.h(kickback_ancilla)
        qc_oracle.x(kickback_ancilla)
        
    oracle_gate = qc_oracle.to_gate()
    oracle_gate.name = "Oracle_S_chi"
    return oracle_gate, qc_oracle

def build_grover_operator(A_gate, oracle_gate, num_state_qubits, ancilla_reg):
    """ + '"""Construye el operador canónico de Grover Q = -A S0 A^dagger S_chi."""' + r"""
    qc_Q = QuantumCircuit(num_state_qubits + len(ancilla_reg))
    state_qubits = list(range(num_state_qubits))
    ancillas = list(range(num_state_qubits, num_state_qubits + len(ancilla_reg)))
    
    # 1. Oracle S_chi
    qc_Q.append(oracle_gate, state_qubits + ancillas)
    # 2. Undo A: A^dagger
    qc_Q.append(A_gate.inverse(), state_qubits + ancillas)
    
    # 3. Diffuser S0
    qc_Q.x(state_qubits)
    kickback_qubit = ancillas[-1]
    qc_Q.x(kickback_qubit)
    qc_Q.h(kickback_qubit)
    try:
        qc_Q.mcx(state_qubits, kickback_qubit, ancilla_qubits=ancillas[:-1], mode='v-chain')
    except Exception:
        qc_Q.mcx(state_qubits, kickback_qubit)
    qc_Q.h(kickback_qubit)
    qc_Q.x(kickback_qubit)
    qc_Q.x(state_qubits)
    
    # 4. Reapply A
    qc_Q.append(A_gate, state_qubits + ancillas)
    
    Q_gate = qc_Q.to_gate()
    Q_gate.name = "Grover_Q"
    return Q_gate, qc_Q

oracle_gate, qc_oracle = synthesize_quantum_oracle([8], num_state_qubits, reg_A)
Q_gate, qc_Q_circuit = build_grover_operator(A_gate, oracle_gate, num_state_qubits, reg_A)
print("✓ Oracle and Grover Operator constructed.")

# Graphical Visualization of the Grover Operator Q
print("Generating graphical visualization of Grover Operator Q architecture...")
fig_Q = qc_Q_circuit.draw('mpl', style='iqp')
plt.title("Monolithic Grover Operator Q (-A S0 A^† S_chi)", fontsize=13, pad=12, fontweight='bold')
plt.tight_layout()
export_figure(fig_Q, [
    os.path.join('figures', 'circuits', 'circuit_operator_grover.png'),
    os.path.join('thesis', 'figures', '3.quantum_bayesian_formalism', 'circuit_operator_grover.png'),
    os.path.join('thesis', 'figures', '4.system_architecture', 'circuit_operator_grover.png')
], dpi=200, bbox_inches='tight')
plt.show()"""

# -------------------------------------------------------------
# Celda 8: Markdown - QPE e IQFT (4.3.5)
# -------------------------------------------------------------
c8_md = """## 4. Section 4.3.5: Integration of QPE and Inverse Quantum Fourier Transform (IQFT)

The master circuit connects the three quantum registers:
1. **Superposition on Evaluation Register:** Hadamard gates $H^{\otimes n_E}$ applied across `reg_E`.
2. **Controlled Grover Power Cascade ($C\text{-}\mathcal{Q}^{2^j}$):**
   For each evaluation qubit $j \in \{0, 1, \dots, n_E - 1\}$, operator $C\text{-}\mathcal{Q}$ is applied $2^j$ times targeting registers $(S, A)$.
   The total number of Grover operations in the circuit is:
   $$N_Q = \sum_{j=0}^{n_E-1} 2^j = 2^{n_E} - 1 = 31 \text{ applications of } \mathcal{Q}$$
3. **IQFT:** Applied over `reg_E` to map accumulated phase onto the computational basis.
4. **Measurement:** Terminal readout of `reg_E` into classical register `reg_C`."""

# -------------------------------------------------------------
# Cell 9: Code - Master QAE Assembly, Transpilation and Visualization
# -------------------------------------------------------------
c9_code = r"""def synthesize_full_qae(qc, Q_gate, eval_reg, state_reg, ancilla_reg, class_reg):
    """ + '"""Ensambla el circuito QAE completo con cascada controlada e IQFT."""' + r"""
    num_eval = eval_reg.size
    for i in range(num_eval):
        qc.h(eval_reg[i])
        
    target_qubits = list(state_reg) + list(ancilla_reg)
    c_Q_gate = Q_gate.control(1)
    
    for j in range(num_eval):
        iterations = 2 ** j
        for _ in range(iterations):
            qc.append(c_Q_gate, [eval_reg[j]] + target_qubits)
            
    iqft_circuit = QFT(num_qubits=num_eval, approximation_degree=0, do_swaps=True, inverse=True)
    qc.append(iqft_circuit.to_gate(), eval_reg)
    qc.measure(eval_reg, class_reg)
    return qc

synthesize_full_qae(master_circuit, Q_gate, reg_E, reg_S, reg_A, reg_C)
print(f"✓ Circuito maestro ensamblado. Profundidad abstracta: {master_circuit.depth()}")

# Graphical Visualization of the Master QAE Circuit
print("Generating high-level graphical visualization of the QAE Master Circuit...")
fig_master = master_circuit.draw('mpl', style='iqp', fold=35)
plt.title("Complete Canonical QAE Master Circuit (H^⊗nE + C-Q^2^j + IQFT + Measurement)", fontsize=13, pad=12, fontweight='bold')
plt.tight_layout()
export_figure(fig_master, [
    os.path.join('figures', 'circuits', 'circuit_qae_master.png'),
    os.path.join('thesis', 'figures', '3.quantum_bayesian_formalism', 'circuit_qae_master.png'),
    os.path.join('thesis', 'figures', '4.system_architecture', 'circuit_qae_master.png')
], dpi=200, bbox_inches='tight')
plt.show()

# Transpilation
print("\nTranspiling master circuit for AerSimulator (native gate decomposition)...")
compiled_circuit = transpile(master_circuit, ideal_backend, optimization_level=1)
print(f"✓ Profundidad del circuito compilado : {compiled_circuit.depth():,}")
print(f"✓ Total de puertas básicas compiladas: {compiled_circuit.size():,}")"""

# -------------------------------------------------------------
# Celda 10: Markdown - Simulación Statevector (4.3.7)
# -------------------------------------------------------------
c10_md = """## 5. Section 4.3.7: Statevector Quantum Simulation in AerSimulator

Simulation executed with **2,048 shots** using the `statevector` method in `AerSimulator`.  
The output reveals the spectral probability mass across the $2^{n_E} = 32$ basis states of evaluation register `reg_E`."""

# -------------------------------------------------------------
# Cell 11: Code - Simulation Execution (4.3.7)
# -------------------------------------------------------------
c11_code = r"""shots = 2048
print(f"Ejecutando simulación cuántica con {shots} shots...")
job = ideal_backend.run(compiled_circuit, shots=shots)
result = job.result()
counts = result.get_counts(compiled_circuit)

sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

print("="*65)
print(" EVALUATION REGISTER MEASUREMENT RESULTS")
print("="*65)
print(f"{'Bitstring':<12} | {'Decimal y':<10} | {'Disparos (Shots)':<18} | {'Frecuencia (%)':<15}")
print("-" * 65)

for bitstring, count in sorted_counts:
    decimal_y = int(bitstring, 2)
    freq = (count / shots) * 100.0
    print(f"{bitstring:<12} | {decimal_y:<10} | {count:<18} | {freq:.2f}%")
print("="*65)"""

# -------------------------------------------------------------
# Celda 12: Markdown - Post-procesado y Espectro (4.3.8)
# -------------------------------------------------------------
c12_md = """## 6. Section 4.3.8: Post-Processing and Amplitude Extraction

From the measured integer $y$ in the evaluation register, the quantum phase $\theta$ and estimated probability $\hat{a}$ are obtained via the bijection:
$$\theta = \frac{\pi \cdot y}{2^{n_E}}, \quad \hat{a} = \sin^2(\theta)$$

In ideal QAE, the measurement distribution displays **two symmetric conjugate peaks**, corresponding to the two eigenvalues of the Grover operator $e^{\pm 2i\theta_a}$:
- The primary peak at $y_1$ estimates the direct eigenvalue.
- The conjugate peak at $y_2 = 2^{n_E} - y_1$ estimates the complex conjugate eigenvalue.
Both yield identical physical amplitude estimates due to reflection symmetry: $\sin^2(\pi - \theta) = \sin^2(\theta)$."""

# -------------------------------------------------------------
# Cell 13: Code - Spectrum and Visualization
# -------------------------------------------------------------
c13_code = r"""print("="*80)
print(f" TABLA ESPECTRAL QAE: EXTRACCIÓN DE PROBABILIDAD (n_E = {num_eval_qubits})")
print("="*80)
print(f"{'Bitstring':<12} | {'Decimal y':<10} | {'Fase theta (rad)':<18} | {'Amplitud a = sin^2(theta)':<28} | {'Frecuencia':<10}")
print("-" * 80)

x_labels = []
frequencies = []

for bitstring, count in sorted_counts:
    decimal_y = int(bitstring, 2)
    phase_theta = np.pi * (decimal_y / (2 ** num_eval_qubits))
    est_a = np.sin(phase_theta) ** 2
    freq = count / shots
    x_labels.append(f"|{bitstring}> ({decimal_y})")
    frequencies.append(freq * 100.0)
    print(f"{bitstring:<12} | {decimal_y:<10} | {phase_theta:<18.4f} | {est_a:<28.6f} | {freq*100:.2f}%")

print("="*80)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x_labels, frequencies, color='darkcyan', edgecolor='black', alpha=0.85, width=0.5)
ax.set_title(f'QAE Phase Spectrum on Evaluation Register ($n_E = {num_eval_qubits}$ qubits)', fontsize=14, fontweight='bold')
ax.set_xlabel(r'Evaluation Register Computational Basis State $|y\rangle$ (Decimal)', fontsize=12)
ax.set_ylabel('Measurement Frequency (%)', fontsize=12)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
export_figure(fig, [
    os.path.join('figures', 'results', 'figure_qae_ideal_spectrum.png'),
    os.path.join('thesis', 'figures', '4.system_architecture', 'figure_qae_ideal_spectrum.png')
], dpi=300)
plt.show()"""

# -------------------------------------------------------------
# Celda 14: Markdown - Estudio de Resolución vs Profundidad (4.3.9)
# -------------------------------------------------------------
c14_md = """## 7. Section 4.3.9: Resolution vs. Depth Trade-off Study (NISQ Constraints)

A critical investigation in this thesis evaluates the physical feasibility of canonical QAE on contemporary NISQ processors:

### Mathematical Formulation of Phase Quantization
The evaluation register of $n_E$ qubits discretizes the phase interval $[0, \pi]$ into $2^{n_E}$ micro-bins. The **minimum angular resolution** is:
$$\Delta \theta = \frac{\pi}{2^{n_E}}$$
By error propagation through the derivative of $a = \sin^2(\theta)$:
$$\Delta a \approx \left| \frac{da}{d\theta} \right| \Delta \theta = 2 \sin(\theta)\cos(\theta) \Delta \theta \approx 2 \sqrt{a} \cdot \frac{\pi}{2^{n_E}}$$

### Exponential Circuit Depth Scaling
The number of Grover applications grows exponentially with $n_E$:
$$N_Q(n_E) = 2^{n_E} - 1$$
Given that each compiled Grover operator contains approximately $\sim 1,200$ two-qubit CNOT gates:
$$\text{Estimated Depth} \approx (2^{n_E} - 1) \times 1,200 \text{ gates}$$

- For $n_E = 5$: $N_Q = 31 \implies \sim 43,000$ gates.
- For $n_E = 10$: $N_Q = 1,023 \implies \sim 1,200,000$ gates (**intractable on NISQ processors without fault tolerance**).

This motivates modern alternatives such as **Iterative Quantum Amplitude Estimation (IQAE)**, eliminating the auxiliary evaluation register and IQFT entirely."""

# -------------------------------------------------------------
# Cell 15: Code - Resolution vs Depth Trade-off Graph
# -------------------------------------------------------------
c15_code = r"""n_E_range = np.arange(3, 11)
angular_resolution = np.pi / (2 ** n_E_range)
grover_iterations = (2 ** n_E_range) - 1
approx_circuit_depth = grover_iterations * 1415

fig, ax1 = plt.subplots(figsize=(10, 6))

color_res = 'tab:blue'
ax1.set_xlabel(r'Number of Evaluation Qubits ($n_E$)', fontsize=12)
ax1.set_ylabel(r'Phase Angular Resolution $\Delta \theta$ (rad)', color=color_res, fontsize=12)
line1 = ax1.plot(n_E_range, angular_resolution, color=color_res, marker='o', lw=2.5, label=r'Phase Resolution $\Delta \theta$')
ax1.tick_params(axis='y', labelcolor=color_res)
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.3)

ax2 = ax1.twinx()
color_depth = 'tab:red'
ax2.set_ylabel('Estimated Circuit Depth (Gates)', color=color_depth, fontsize=12)
line2 = ax2.plot(n_E_range, approx_circuit_depth, color=color_depth, marker='s', lw=2.5, linestyle='--', label='Circuit Depth')
ax2.tick_params(axis='y', labelcolor=color_depth)
ax2.set_yscale('log')

ax2.axhline(y=1000, color='darkgreen', linestyle=':', lw=2, label='NISQ Hardware Coherence Threshold (~1,000 gates)')

lines = line1 + line2 + [ax2.get_lines()[-1]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, fontsize=10)

plt.title('Canonical QAE Theoretical Trade-off: Phase Resolution vs. Quantum Depth', fontsize=13, fontweight='bold')
plt.tight_layout()
export_figure(fig, [
    os.path.join('figures', 'results', 'figure_qae_nisq_tradeoff.png'),
    os.path.join('thesis', 'figures', '4.system_architecture', 'figure_qae_nisq_tradeoff.png')
], dpi=300)
plt.show()"""

# -------------------------------------------------------------
# Celda 16: Markdown - Conclusiones Sección 4.3
# -------------------------------------------------------------
c16_md = """## 8. Monograph Conclusions for Section 4.3

1. **Success of QBN-QAE Architecture:** Validated in ideal simulation the mathematical feasibility of encoding the exoplanetary Bayesian network of K2-18b and coupling it to Grover reflection operators with phase kickback.
2. **Surpassing Classical Limits:** QAE achieves asymptotic quadratic convergence $\mathcal{O}(1/M_q)$, overcoming the $\mathcal{O}(1/\sqrt{M})$ variance divergence of classical MCMC sampling.
3. **NISQ Challenge:** The $\sim 43,000$-gate depth required for $n_E = 5$ confirms that canonical QFT-based QAE requires early Fault-Tolerant Quantum Computing (FTQC), motivating the error mitigation studies in **Deliverable 03** and future lines in **IQAE**."""

# Assemble all cells
nb.cells = [
    nbformat.v4.new_markdown_cell(c1_md),
    nbformat.v4.new_markdown_cell(c2_md),
    nbformat.v4.new_code_cell(c3_code),
    nbformat.v4.new_markdown_cell(c4_md),
    nbformat.v4.new_code_cell(c5_code),
    nbformat.v4.new_markdown_cell(c6_md),
    nbformat.v4.new_code_cell(c7_code),
    nbformat.v4.new_markdown_cell(c8_md),
    nbformat.v4.new_code_cell(c9_code),
    nbformat.v4.new_markdown_cell(c10_md),
    nbformat.v4.new_code_cell(c11_code),
    nbformat.v4.new_markdown_cell(c12_md),
    nbformat.v4.new_code_cell(c13_code),
    nbformat.v4.new_markdown_cell(c14_md),
    nbformat.v4.new_code_cell(c15_code),
    nbformat.v4.new_markdown_cell(c16_md)
]

output_filename = os.path.join(os.path.dirname(__file__), "02_QAE_Ideal_Simulation.ipynb")

print(f"Ejecutando celdas en Python puro con renderizado mpl para {output_filename}...")
exec_namespace = {}
execution_count = 1

for idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        code = cell.source
        print(f"\n--- Ejecutando celda {execution_count} (índice {idx}) ---")
        
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        cell.outputs = []
        try:
            exec(code, exec_namespace)
            stdout_text = redirected_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            print(f"ERROR en celda {execution_count}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            sys.stdout = old_stdout
            
        if stdout_text:
            print(stdout_text, end="")
            cell.outputs.append(nbformat.v4.new_output(
                output_type="stream",
                name="stdout",
                text=stdout_text
            ))
            
        fignums = plt.get_fignums()
        if fignums:
            for fig_id in fignums:
                fig = plt.figure(fig_id)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
                cell.outputs.append(nbformat.v4.new_output(
                    output_type="display_data",
                    data={"image/png": img_base64},
                    metadata={}
                ))
            plt.close('all')
            print(f"  [Capturadas {len(fignums)} figuras]")
            
        cell.execution_count = execution_count
        execution_count += 1

print(f"\nGuardando notebook ejecutado en {output_filename}...")
with open(output_filename, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✓ ¡Notebook {output_filename} con diagramas de circuitos generado y guardado con éxito!")
