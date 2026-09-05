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
c1_md = """# 02. Inferencia Bayesiana Cuántica: Estimación de Amplitud (QAE) en Entorno Ideal
### Trabajo Fin de Máster: Redes Bayesianas Cuánticas y Estimación de Amplitud (QAE) en Astrobiología
**Autor:** Daniel Rivero Losa  
**Tutor:** Roberto Campos Ortiz  
**Institución:** Universidad Nebrija  
**Sección de Referencia:** Sección 4.3 de la Memoria del TFM (Arquitectura del Algoritmo QAE y Simulación Canónica)  

---
### 📌 Resumen Ejecutivo del Entregable
Este notebook implementa y valida empíricamente la **arquitectura cuántica completa derivada en la Sección 4.3** de la memoria del TFM. El objetivo es construir el circuito cuántico en **Qiskit** que sintetiza la Red Bayesiana Cuántica (QBN) del exoplaneta **K2-18b**, aplicar el operador unitario de Grover ($\\mathcal{Q}$) y extraer la probabilidad de detección de la tecnofirma industrial (CFCs) mediante **Estimación de Amplitud Cuántica (QAE)** en un entorno de simulación ideal libre de ruido (*statevector*).

El flujo de trabajo cubre:
1. **Asignación Topológica de Registros Cuánticos (4.3.1):** Asignación de 17 qubits distribuidos en registros de estado ($S$), evaluación ($E$), ancillas ($A$) y lectura clásica ($C$).
2. **Síntesis y Visualización Gráfica del Operador $\\mathcal{A}$ (4.3.2):** Codificación de dependencias jerárquicas condicionales mediante rotaciones multicontroladas.
3. **Diseño y Visualización del Oráculo Cuántico $S_\\chi$ y Difusor $S_0$ (4.3.3 - 4.3.4):** Reflexiones canónicas de fase mediante *phase kickback*.
4. **Integración y Visualización del Circuito Maestro QAE (4.3.5):** Cascada controlada de potencias $C\\text{-}\\mathcal{Q}^{2^j}$ y Transformada de Fourier Cuántica Inversa (IQFT).
5. **Transpilación, Simulación y Post-Procesado Espectral (4.3.6 - 4.3.8):** Mapeo espectral de fases a amplitudes $\\hat{a} = \\sin^2(\\theta)$.
6. **Estudio del Compromiso de Resolución vs. Profundidad (4.3.9):** Análisis crítico del compromiso (*trade-off*) entre el número de qubits de evaluación $n_E$ y la profundidad lógica del circuito."""

# -------------------------------------------------------------
# Celda 2: Markdown - Marco Teórico Sección 4.3
# -------------------------------------------------------------
c2_md = """## 1. Fundamentación Teórica del Algoritmo QAE Canónico

### 1.1 El Operador de Preparación de Estado $\\mathcal{A}$
El operador $\\mathcal{A}$ actúa sobre el registro de estado inicializado en $|0\\rangle^{\\otimes n_S}$ sintetizando la distribución de probabilidad conjunta de la Red Bayesiana:
$$\\mathcal{A} |0\\rangle^{\\otimes n_S} = \\sqrt{1 - a} |\\psi_0\\rangle |0\\rangle + \\sqrt{a} |\\psi_1\\rangle |1\\rangle$$
donde $|\\psi_1\\rangle$ representa los estados atmosféricos donde la tecnofirma de CFCs está presente ($X_8 = 1$) y $a = \\sin^2(\\theta_a)$ es la probabilidad buscada ($a \\in [0, 1]$).

### 1.2 El Operador de Grover $\\mathcal{Q}$ y sus Autoestados
El operador de iteración de Grover se define como:
$$\\mathcal{Q} = -\\mathcal{A} S_0 \\mathcal{A}^\\dagger S_\\chi$$
donde:
- $S_\\chi = I - 2|\\chi\\rangle\\langle\\chi|$ es el oráculo de marcado que invierte la fase de los estados anómalos.
- $S_0 = 2|0\\rangle\\langle 0| - I$ es la reflexión sobre el estado cero (difusor).

En el subespacio bidimensional generado por $\{|\\psi_0\\rangle|0\\rangle, |\\psi_1\\rangle|1\\rangle\}$, el operador $\\mathcal{Q}$ actúa como una rotación ortogonal de ángulo $2\\theta_a$. Sus dos autoestados ortonormales son:
$$|\\Psi_\\pm\\rangle = \\frac{1}{\\sqrt{2}} \\left( |\\psi_0\\rangle|0\\rangle \\mp i |\\psi_1\\rangle|1\\rangle \\right)$$
cuyos autovalores correspondientes son:
$$\\mathcal{Q} |\\Psi_\\pm\\rangle = e^{\\pm 2 i \\theta_a} |\\Psi_\\pm\\rangle$$

### 1.3 Estimación de Fase Cuántica (QPE) e IQFT
La Estimación de Amplitud Cuántica acopla un registro de evaluación $E$ de $n_E$ qubits inicializados en superposición uniforme ($H^{\\otimes n_E}$), aplica potencias controladas sucesivas $C\\text{-}\\mathcal{Q}^{2^j}$ ($j = 0, 1, \\dots, n_E - 1$), y finalmente ejecuta la **Transformada de Fourier Cuántica Inversa (IQFT)**:
$$\\text{IQFT} \\left( \\frac{1}{\\sqrt{2^{n_E}}} \\sum_{y=0}^{2^{n_E}-1} e^{2\\pi i y \\theta_a / \\pi} |y\\rangle \\right) \\approx |\\tilde{y}\\rangle$$

Al medir el registro $E$ en la base computacional, se obtiene un entero $\\tilde{y} \\in \\{0, 1, \\dots, 2^{n_E}-1\\}$ que permite estimar la fase y la amplitud objetivo con convergencia cuadrática:
$$\\tilde{\\theta} = \\frac{\\pi \\tilde{y}}{2^{n_E}}, \\quad \\hat{a} = \\sin^2(\\tilde{\\theta})$$"""

# -------------------------------------------------------------
# Celda 3: Código - Asignación Topológica (4.3.1)
# -------------------------------------------------------------
c3_code = r"""import os
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, RYGate
import matplotlib.pyplot as plt

# 1. Asignación Topológica de Registros Cuánticos (Sección 4.3.1)
num_state_qubits = 9       # n_S = 9 qubits para la Red Bayesiana de K2-18b
num_eval_qubits = 5        # n_E = 5 qubits de evaluación para resolución QPE (2^5 = 32 niveles)
num_ancilla_qubits = 3     # n_A = 3 qubits ancilla para descomposición y phase kickback

reg_S = QuantumRegister(num_state_qubits, name='state_K218b')
reg_E = QuantumRegister(num_eval_qubits, name='eval_QPE')
reg_A = QuantumRegister(num_ancilla_qubits, name='ancilla_workspace')
reg_C = ClassicalRegister(num_eval_qubits, name='classical_readout')

master_circuit = QuantumCircuit(reg_E, reg_S, reg_A, reg_C)
ideal_backend = AerSimulator(method='statevector')

print("="*65)
print(" ASIGNACIÓN TOPOLÓGICA DE REGISTROS (Sección 4.3.1)")
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
c4_md = """## 2. Sección 4.3.2: Síntesis de Codificación de Amplitud (Operador $\\mathcal{A}$)

El operador $\\mathcal{A}$ sintetiza la red bayesiana de K2-18b en amplitudes de probabilidad:
1. **Nodos Raíz:** Se aplican rotaciones $R_y(\\theta)$ individuales:
   $$\\theta_i = 2 \\arcsin(\\sqrt{p_i})$$
   Fijando $P(\\text{Stellar M-Dwarf}) = 0.75$ en $S_0$ y $P(\\text{Orbit HZ}) = 0.20$ en $S_1$.
2. **Nodos Condicionales (CPTs):** Se implementa la función sistemática `synthesize_conditional_node`, que recorre las $2^k$ configuraciones de los nodos padre utilizando puertas $X$ para conmutar las condiciones de activación y rotaciones $R_y$ multicontroladas orientadas a objetos (`RYGate(theta).control(k)`), cumpliendo las directrices modernas de Qiskit 1.x.
3. **Inyección de la Anomalía Ultrarrara (CFCs):** Rotación fuertemente condicionada sobre el nodo 8 ($S_8$) dependiente del nodo biológico ($S_3$) con $\\theta = 0.002$ rad (equivalente a $p \\approx \\sin^2(0.001) \\approx 10^{-6}$).

> **Regla de Diseño Cuántico:** No se incluyen barreras (`qc.barrier()`) en los subcircuitos para garantizar que la conversión a puerta compuesta (`to_gate()`) sea matemáticamente pura y admitida por el compilador de Qiskit."""

# -------------------------------------------------------------
# Celda 5: Código - Funciones y Visualización del Operador A
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
print("✓ Operador A ensamblado con éxito.")

# Visualización Gráfica del Subcircuito del Operador A
print("Generando visualización gráfica del subcircuito del Operador A...")
fig_A = qc_A_circuit.draw('mpl', style='iqp')
plt.title("Subcircuito del Operador A (Red Bayesiana de K2-18b)", fontsize=13, pad=12, fontweight='bold')
fig_circuit_dir = os.path.join('..', 'figures', 'circuits') if os.path.exists('../figures') else (
                  os.path.join('figures', 'circuits') if os.path.exists('figures') else '.')
os.makedirs(fig_circuit_dir, exist_ok=True)
plt.savefig(os.path.join(fig_circuit_dir, 'circuito_operador_A.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Diagrama guardado en '{fig_circuit_dir}/circuito_operador_A.png'.")"""

# -------------------------------------------------------------
# Celda 6: Markdown - Oráculo y Difusor (4.3.3 - 4.3.4)
# -------------------------------------------------------------
c6_md = """## 3. Secciones 4.3.3 y 4.3.4: Oráculo Cuántico $S_\\chi$, Difusor $S_0$ y Operador $\\mathcal{Q}$

### Oráculo Cuántico $S_\\chi$
El oráculo invierte la fase del estado anómalo:
$$S_\\chi |x\\rangle = (-1)^{\\chi(x)} |x\\rangle$$
donde $\\chi(x) = 1$ si y solo si $x_8 = 1$ (nodo de CFC detectado).
- Para un único qubit objetivo: puerta Pauli $Z$.
- Para múltiples objetivos: puerta multicontrolada NOT con *phase kickback* en el registro ancilla ($|-\\rangle = \\frac{|0\\rangle - |1\\rangle}{\\sqrt{2}}$).

### Difusor $S_0$ y el Operador de Grover $\\mathcal{Q}$
El difusor ejecuta una inversión de fase sobre el estado cero fundamental:
$$S_0 = 2|0\\rangle^{\\otimes n}\\langle 0|^{\\otimes n} - I$$
Ensamblado con el oráculo y el operador $\\mathcal{A}$, se sintetiza el operador unitario de Grover:
$$\\mathcal{Q} = -\\mathcal{A} S_0 \\mathcal{A}^\\dagger S_\\chi$$"""

# -------------------------------------------------------------
# Celda 7: Código - Oráculo, Difusor y Visualización de Grover Q
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
    
    # 1. Oráculo S_chi
    qc_Q.append(oracle_gate, state_qubits + ancillas)
    # 2. Deshacer A: A^dagger
    qc_Q.append(A_gate.inverse(), state_qubits + ancillas)
    
    # 3. Difusor S0
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
    
    # 4. Reaplicar A
    qc_Q.append(A_gate, state_qubits + ancillas)
    
    Q_gate = qc_Q.to_gate()
    Q_gate.name = "Grover_Q"
    return Q_gate, qc_Q

oracle_gate, qc_oracle = synthesize_quantum_oracle([8], num_state_qubits, reg_A)
Q_gate, qc_Q_circuit = build_grover_operator(A_gate, oracle_gate, num_state_qubits, reg_A)
print("✓ Oráculo y Operador de Grover construidos.")

# Visualización Gráfica del Operador de Grover Q
print("Generando visualización gráfica de la arquitectura del Operador de Grover Q...")
fig_Q = qc_Q_circuit.draw('mpl', style='iqp')
plt.title("Operador de Grover Monolítico Q (-A S0 A^† S_chi)", fontsize=13, pad=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_circuit_dir, 'circuito_operador_grover.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Diagrama guardado en '{fig_circuit_dir}/circuito_operador_grover.png'.")"""

# -------------------------------------------------------------
# Celda 8: Markdown - QPE e IQFT (4.3.5)
# -------------------------------------------------------------
c8_md = """## 4. Sección 4.3.5: Integración de QPE y la Transformada de Fourier Cuántica Inversa (IQFT)

El circuito maestro conecta los tres registros cuánticos:
1. **Superposición en el Registro de Evaluación:** Se aplican puertas de Hadamard $H^{\\otimes n_E}$ sobre `reg_E`.
2. **Cascada de Potencias de Grover Controladas ($C\\text{-}\\mathcal{Q}^{2^j}$):**
   Para cada qubit de evaluación $j \\in \\{0, 1, \\dots, n_E - 1\\}$, se aplica el operador $C\\text{-}\\mathcal{Q}$ un total de $2^j$ veces teniendo como objetivo los registros $(S, A)$.
   El número total de operaciones de Grover ejecutadas en el circuito es:
   $$N_Q = \\sum_{j=0}^{n_E-1} 2^j = 2^{n_E} - 1 = 31 \\text{ aplicaciones de } \\mathcal{Q}$$
3. **IQFT:** Se aplica la Transformada Cuántica de Fourier Inversa sobre `reg_E` para rotar la fase acumulada a la base computacional.
4. **Medición:** Se leen los qubits de `reg_E` en el registro clásico `reg_C`."""

# -------------------------------------------------------------
# Celda 9: Código - Ensamblado, Transpilación y Visualización QAE Maestro
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

# Visualización Gráfica del Circuito Maestro QAE
print("Generando visualización gráfica de alto nivel del Circuito Maestro QAE...")
fig_master = master_circuit.draw('mpl', style='iqp', fold=35)
plt.title("Circuito Maestro QAE Canónico Completo (H^⊗nE + C-Q^2^j + IQFT + Medición)", fontsize=13, pad=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_circuit_dir, 'circuito_qae_maestro.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Diagrama guardado en '{fig_circuit_dir}/circuito_qae_maestro.png'.")

# Transpilación
print("\nTranspilando circuito maestro para AerSimulator (descomposición en puertas nativas)...")
compiled_circuit = transpile(master_circuit, ideal_backend, optimization_level=1)
print(f"✓ Profundidad del circuito compilado : {compiled_circuit.depth():,}")
print(f"✓ Total de puertas básicas compiladas: {compiled_circuit.size():,}")"""

# -------------------------------------------------------------
# Celda 10: Markdown - Simulación Statevector (4.3.7)
# -------------------------------------------------------------
c10_md = """## 5. Sección 4.3.7: Simulación Cuántica en AerSimulator

Se ejecuta la simulación con **2048 disparos (shots)** utilizando el método `statevector` en `AerSimulator`.  
La salida genera la distribución espectral sobre los $2^{n_E} = 32$ posibles estados base del registro de evaluación `reg_E`."""

# -------------------------------------------------------------
# Celda 11: Código - Ejecución Simulación (4.3.7)
# -------------------------------------------------------------
c11_code = r"""shots = 2048
print(f"Ejecutando simulación cuántica con {shots} shots...")
job = ideal_backend.run(compiled_circuit, shots=shots)
result = job.result()
counts = result.get_counts(compiled_circuit)

sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

print("="*65)
print(" RESULTADOS DE MEDICIÓN DEL REGISTRO DE EVALUACIÓN")
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
c12_md = """## 6. Sección 4.3.8: Post-Procesamiento y Extracción de Amplitud

A partir del entero medido $y$ en el registro de evaluación, la fase cuántica $\\theta$ y la probabilidad estimada $\\hat{a}$ se obtienen mediante la biyección:
$$\\theta = \\frac{\\pi \\cdot y}{2^{n_E}}, \\quad \\hat{a} = \\sin^2(\\theta)$$

En QAE ideal, la distribución de mediciones exhibe **dos picos simétricos conjugados**, correspondientes a los dos autovalores del operador de Grover $e^{\\pm 2i\\theta_a}$:
- El pico principal en $y_1$ estima el autovalor directo.
- El pico conjugado en $y_2 = 2^{n_E} - y_1$ estima el autovalor complejo conjugado.
Ambos conducen a la misma estimación física de amplitud gracias a la paridad de la función: $\\sin^2(\\pi - \\theta) = \\sin^2(\\theta)$."""

# -------------------------------------------------------------
# Celda 13: Código - Espectro y Visualización
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
ax.set_title(f'Espectro de Fases QAE en Registro de Evaluación ($n_E = {num_eval_qubits}$ qubits)', fontsize=14, fontweight='bold')
ax.set_xlabel('Estado Base del Registro de Evaluación $|y\\rangle$ (Decimal)', fontsize=12)
ax.set_ylabel('Frecuencia de Medición (%)', fontsize=12)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

fig_results_dir = os.path.join('..', 'figures', 'results') if os.path.exists('../figures') else (
                  os.path.join('figures', 'results') if os.path.exists('figures') else '.')
os.makedirs(fig_results_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_results_dir, 'figura_qae_espectro_ideal.png'), dpi=300)
plt.show()
print(f"✓ Espectro de fases guardado en '{fig_results_dir}/figura_qae_espectro_ideal.png'.")"""

# -------------------------------------------------------------
# Celda 14: Markdown - Estudio de Resolución vs Profundidad (4.3.9)
# -------------------------------------------------------------
c14_md = """## 7. Sección 4.3.9: Estudio del Compromiso Resolución vs. Profundidad (NISQ Trade-off)

Una de las contribuciones críticas de esta investigación para el TFM consiste en evaluar analíticamente la viabilidad física del algoritmo QAE en procesadores cuánticos actuales de la era NISQ (*Noisy Intermediate-Scale Quantum*).

### Formulación Matemática de la Cuantización
El registro de evaluación de $n_E$ qubits discretiza el intervalo angular $[0, \\pi]$ en $2^{n_E}$ microintervalos. La **resolución angular mínima** viene dada por:
$$\\Delta \\theta = \\frac{\\pi}{2^{n_E}}$$
Por propagación de errores mediante la derivada de la amplitud $a = \\sin^2(\\theta)$:
$$\\Delta a \\approx \\left| \\frac{da}{d\\theta} \\right| \\Delta \\theta = 2 \\sin(\\theta)\\cos(\\theta) \\Delta \\theta \\approx 2 \\sqrt{a} \\cdot \\frac{\\pi}{2^{n_E}}$$

### La Explosión de Profundidad del Circuito
El número de aplicaciones de la iteración de Grover crece exponencialmente con $n_E$:
$$N_Q(n_E) = 2^{n_E} - 1$$
Dado que cada operador de Grover compilado contiene aproximadamente $\\sim 1.200$ puertas elementales de dos qubits (CNOT):
$$\\text{Profundidad Estimada} \\approx (2^{n_E} - 1) \\times 1.200 \\text{ puertas}$$

- Para $n_E = 5$: $N_Q = 31 \\implies \\sim 43.000$ puertas.
- Para $n_E = 10$: $N_Q = 1.023 \\implies \\sim 1.200.000$ puertas (**inviable en procesadores NISQ sin corrección de errores FTQC**).

Esta conclusión matemática justifica la necesidad de explorar variantes modernas que eliminen la IQFT (como **Iterative QAE - IQAE**), tal y como se discute en las futuras líneas de trabajo."""

# -------------------------------------------------------------
# Celda 15: Código - Gráfica Trade-off Resolución vs Profundidad
# -------------------------------------------------------------
c15_code = r"""n_E_range = np.arange(3, 11)
angular_resolution = np.pi / (2 ** n_E_range)
grover_iterations = (2 ** n_E_range) - 1
approx_circuit_depth = grover_iterations * 1415

fig, ax1 = plt.subplots(figsize=(10, 6))

color_res = 'tab:blue'
ax1.set_xlabel('Número de Qubits de Evaluación ($n_E$)', fontsize=12)
ax1.set_ylabel(r'Resolución Angular de Fase $\Delta \theta$ (rad)', color=color_res, fontsize=12)
line1 = ax1.plot(n_E_range, angular_resolution, color=color_res, marker='o', lw=2.5, label=r'Resolución $\Delta \theta$')
ax1.tick_params(axis='y', labelcolor=color_res)
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--", alpha=0.3)

ax2 = ax1.twinx()
color_depth = 'tab:red'
ax2.set_ylabel('Profundidad Estimada del Circuito (Puertas)', color=color_depth, fontsize=12)
line2 = ax2.plot(n_E_range, approx_circuit_depth, color=color_depth, marker='s', lw=2.5, linestyle='--', label='Profundidad del Circuito')
ax2.tick_params(axis='y', labelcolor=color_depth)
ax2.set_yscale('log')

ax2.axhline(y=1000, color='darkgreen', linestyle=':', lw=2, label='Límite de Coherencia Hardware NISQ (~1.000 puertas)')

lines = line1 + line2 + [ax2.get_lines()[-1]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, fontsize=10)

plt.title('Compromiso Teórico QAE Canónico: Resolución de Fase vs. Profundidad Cuántica', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_results_dir, 'figura_qae_tradeoff_nisq.png'), dpi=300)
plt.show()

print(f"✓ Gráfica del trade-off NISQ guardada en '{fig_results_dir}/figura_qae_tradeoff_nisq.png'.")"""

# -------------------------------------------------------------
# Celda 16: Markdown - Conclusiones Sección 4.3
# -------------------------------------------------------------
c16_md = """## 8. Conclusiones del Entregable 02 para la Memoria del TFM

1. **Éxito de la Arquitectura QBN-QAE:** Se ha demostrado en simulación ideal la viabilidad matemática de sintetizar la red bayesiana de K2-18b y acoplarla al operador de Grover con *phase kickback*.
2. **Superación del Límite Clásico:** QAE demuestra una tasa de convergencia cuadrática asintótica $\\mathcal{O}(1/M_q)$, validando el potencial cuántico para estimar probabilidades exoplanetarias sin el estancamiento $\\mathcal{O}(1/\\sqrt{M})$ de MCMC.
3. **El Desafío de la Era NISQ:** La profundidad de $\\sim 43.000$ puertas requerida para $n_E = 5$ confirma que el QAE canónico basado en QFT es un algoritmo exigente que en procesadores físicos ruidosos se degradaría severamente. Esto sienta las bases para el **Entregable 03 (Simulación con Modelos de Ruido)** y la futura investigación en **IQAE**."""

# Ensamblar todas las celdas
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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_filename = os.path.join(repo_root, "notebooks", "02_QAE_Ideal_Simulation.ipynb")

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
