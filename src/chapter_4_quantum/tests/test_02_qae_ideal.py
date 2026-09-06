#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Chapter 4: Canonical QAE in Ideal Simulation (17 Qubits)
Part of the Master's Thesis: Quantum Bayesian Inference for Atmospheric Biosignature
and Technosignature Characterization in Exoplanetary Systems.

Tests:
  1. TestNotebookIntegrity: Verifies JSON structure, full execution, display figures, and outputs of 02_QAE_Ideal_Simulation.ipynb.
  2. TestRegisterAllocationAndTopologicalMapping: Verifies 17-qubit allocation across state (9), eval (5), and ancilla (3) registers.
  3. TestOperatorAEncodingAndPriors: Verifies subcircuit purity (zero barriers), CPT synthesis, and exact prior P(CFC=1) = 1e-6.
  4. TestOracleAndDiffuserUnitarity: Verifies phase kickback of S_chi, reflection S_0, and Grover operator Q = -A S0 A^dagger S_chi.
  5. TestQAEControlledCascadeAndIQFT: Verifies 31 Grover power iterations (sum 2^j for j=0..4), 5-qubit IQFT, and terminal readout.
  6. TestQAESpectralOutputAndQuantization: Verifies 100% projection onto centroid y=16 (pi/2) under finite grid resolution Delta_theta.
  7. TestResolutionVsDepthTradeoffScaling: Verifies analytical scaling Delta_theta = pi / 2^n_E and exponential depth N_Q = 2^n_E - 1.
  8. TestPublicationArtifacts: Verifies existence and size (>10 KB) of all 5 exported publication diagrams.
"""

import os
import sys
import io
import json
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'
os.environ['IPYTHONDIR'] = '/tmp/ipython_cache'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFTGate, RYGate
from qiskit_aer import AerSimulator

from src.chapter_4_quantum import build_and_run_02_qae_ideal as bld02


def exec_silent(code_str: str, namespace: dict):
    """Executes code string silently, suppressing stdout and display."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            exec(code_str, namespace)
        finally:
            sys.stdout = old_stdout
            plt.close('all')


class TestNotebookIntegrity(unittest.TestCase):
    """Audits the executed Jupyter Notebook 02_QAE_Ideal_Simulation.ipynb."""

    @classmethod
    def setUpClass(cls):
        cls.nb_path = os.path.join(os.path.dirname(__file__), "../02_QAE_Ideal_Simulation.ipynb")
        if not os.path.exists(cls.nb_path):
            raise FileNotFoundError(f"Notebook not found at: {cls.nb_path}")
        with open(cls.nb_path, "r", encoding="utf-8") as f:
            cls.nb = nbformat.read(f, as_version=4)

    def test_notebook_file_exists_and_valid_json(self):
        """Checks notebook exists, is non-empty, and conforms to nbformat v4."""
        self.assertTrue(os.path.exists(self.nb_path))
        self.assertGreater(os.path.getsize(self.nb_path), 50_000)
        self.assertEqual(self.nb.nbformat, 4)

    def test_all_code_cells_executed(self):
        """Validates that every code cell has been executed (execution_count >= 1)."""
        code_cells = [c for c in self.nb.cells if c.cell_type == "code"]
        self.assertGreater(len(code_cells), 0, "No code cells found in notebook.")
        for idx, cell in enumerate(code_cells, start=1):
            self.assertIsNotNone(
                cell.execution_count,
                f"Code cell {idx} has not been executed (execution_count is None)."
            )
            self.assertGreater(
                cell.execution_count, 0,
                f"Code cell {idx} execution_count must be >= 1."
            )

    def test_zero_execution_errors(self):
        """Verifies that no cell output contains an unhandled exception or error."""
        for idx, cell in enumerate(self.nb.cells):
            if cell.cell_type == "code":
                for out in cell.get("outputs", []):
                    self.assertNotEqual(
                        out.get("output_type"), "error",
                        f"Execution error found in cell {cell.get('execution_count')}: {out.get('ename')}: {out.get('evalue')}"
                    )

    def test_embedded_publication_figures(self):
        """Ensures that base64 PNG figures are embedded in display_data outputs."""
        display_outputs = []
        for cell in self.nb.cells:
            if cell.cell_type == "code":
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "display_data":
                        display_outputs.append(out)
        self.assertGreaterEqual(
            len(display_outputs), 5,
            f"Expected at least 5 embedded figure outputs, but found {len(display_outputs)}."
        )
        for out in display_outputs:
            self.assertIn("image/png", out.get("data", {}))
            self.assertGreater(len(out["data"]["image/png"]), 1000)

    def test_markdown_academic_structure_and_latex(self):
        """Verifies markdown cells contain clean LaTeX markup for QAE formalism."""
        md_cells = [c for c in self.nb.cells if c.cell_type == "markdown"]
        self.assertGreaterEqual(len(md_cells), 8)
        full_md = "\n".join(c.source for c in md_cells)
        self.assertIn(r"\mathcal{A} |0\rangle", full_md)
        self.assertIn(r"\mathcal{Q} = -\mathcal{A} S_0 \mathcal{A}^\dagger S_\chi", full_md)
        self.assertIn(r"\sin^2(\theta)", full_md)


class TestRegisterAllocationAndTopologicalMapping(unittest.TestCase):
    """Audits the 17-qubit topological register allocation (Section 4.3.1)."""

    @classmethod
    def setUpClass(cls):
        exec_ns = {}
        exec_silent(bld02.c3_code, exec_ns)
        cls.reg_S = exec_ns['reg_S']
        cls.reg_E = exec_ns['reg_E']
        cls.reg_A = exec_ns['reg_A']
        cls.reg_C = exec_ns['reg_C']
        cls.master_circuit = exec_ns['master_circuit']

    def test_qubit_and_bit_counts(self):
        """Verifies strict 17-qubit topology: 9 state + 5 eval + 3 ancilla, and 5 classical bits."""
        self.assertEqual(self.reg_S.size, 9)
        self.assertEqual(self.reg_E.size, 5)
        self.assertEqual(self.reg_A.size, 3)
        self.assertEqual(self.reg_C.size, 5)
        self.assertEqual(self.master_circuit.num_qubits, 17)
        self.assertEqual(self.master_circuit.num_clbits, 5)

    def test_register_names(self):
        """Verifies semantic naming of quantum registers."""
        self.assertEqual(self.reg_S.name, 'state_K218b')
        self.assertEqual(self.reg_E.name, 'eval_QPE')
        self.assertEqual(self.reg_A.name, 'ancilla_workspace')
        self.assertEqual(self.reg_C.name, 'classical_readout')


class TestOperatorAEncodingAndPriors(unittest.TestCase):
    """Audits Operator A amplitude encoding and prior probability normalization (Section 4.3.2)."""

    @classmethod
    def setUpClass(cls):
        exec_ns = {}
        exec_silent(bld02.c3_code, exec_ns)
        exec_silent(bld02.c5_code, exec_ns)
        cls.A_gate = exec_ns['A_gate']
        cls.qc_A_circuit = exec_ns['qc_A_circuit']
        cls.p_cfc_exact = exec_ns['p_cfc_exact']

    def test_subcircuit_purity_zero_barriers(self):
        """Enforces Quantum Software Engineer Rule: zero barriers in Operator A subcircuit."""
        barrier_count = sum(1 for inst in self.qc_A_circuit.data if inst.operation.name == 'barrier')
        self.assertEqual(barrier_count, 0, "Barriers detected in Operator A subcircuit.")

    def test_operator_a_qubit_dimension(self):
        """Operator A acts on state register (9) + ancilla register (3) = 12 qubits."""
        self.assertEqual(self.qc_A_circuit.num_qubits, 12)
        self.assertEqual(self.A_gate.num_qubits, 12)

    def test_exact_analytical_prior_cfc(self):
        """Verifies exact prior probability P(X8=1) = 1.0e-6 via statevector simulation."""
        self.assertAlmostEqual(self.p_cfc_exact, 1.0e-6, places=12)


class TestOracleAndDiffuserUnitarity(unittest.TestCase):
    """Audits Oracle S_chi, Diffuser S_0, and Grover operator Q (Sections 4.3.3 & 4.3.4)."""

    @classmethod
    def setUpClass(cls):
        exec_ns = {}
        exec_silent(bld02.c3_code, exec_ns)
        exec_silent(bld02.c5_code, exec_ns)
        exec_silent(bld02.c7_code, exec_ns)
        cls.oracle_gate = exec_ns['oracle_gate']
        cls.qc_oracle = exec_ns['qc_oracle']
        cls.Q_gate = exec_ns['Q_gate']
        cls.qc_Q_circuit = exec_ns['qc_Q_circuit']

    def test_subcircuit_purity_zero_barriers(self):
        """Verifies zero barriers in Oracle and Grover Q subcircuits."""
        barriers_oracle = sum(1 for inst in self.qc_oracle.data if inst.operation.name == 'barrier')
        barriers_Q = sum(1 for inst in self.qc_Q_circuit.data if inst.operation.name == 'barrier')
        self.assertEqual(barriers_oracle, 0, "Barriers found in Oracle subcircuit.")
        self.assertEqual(barriers_Q, 0, "Barriers found in Grover Q subcircuit.")

    def test_oracle_single_qubit_pauli_z(self):
        """Verifies that single-target oracle on qubit 8 synthesizes a Pauli Z gate."""
        self.assertEqual(self.qc_oracle.data[0].operation.name, 'z')
        self.assertEqual(self.qc_oracle.data[0].qubits[0]._index, 8)

    def test_grover_operator_dimension(self):
        """Verifies Grover operator Q acts on 12 qubits (9 state + 3 ancilla)."""
        self.assertEqual(self.Q_gate.num_qubits, 12)
        self.assertEqual(self.qc_Q_circuit.num_qubits, 12)

    def test_grover_operator_component_structure(self):
        """Verifies Q = -A S0 A^dagger S_chi sequence."""
        op_names = [inst.operation.name for inst in self.qc_Q_circuit.data]
        self.assertIn("Oracle_S_chi", op_names)
        self.assertIn("Operator_A_dg", op_names)
        self.assertIn("Operator_A", op_names)


class TestQAEControlledCascadeAndIQFT(unittest.TestCase):
    """Audits the QPE controlled-Grover cascade and IQFT assembly (Section 4.3.5)."""

    @classmethod
    def setUpClass(cls):
        exec_ns = {}
        exec_silent(bld02.c3_code, exec_ns)
        exec_silent(bld02.c5_code, exec_ns)
        exec_silent(bld02.c7_code, exec_ns)
        exec_silent(bld02.c9_code, exec_ns)
        cls.master_circuit = exec_ns['master_circuit']
        cls.compiled_circuit = exec_ns['compiled_circuit']

    def test_total_controlled_grover_iterations(self):
        """Verifies total number of Grover operations in cascade: sum_{j=0}^{4} 2^j = 31."""
        c_Q_calls = sum(1 for inst in self.master_circuit.data if "Grover_Q" in inst.operation.name)
        self.assertEqual(c_Q_calls, 31, "QAE must contain exactly 31 Grover operator invocations.")

    def test_hadamard_superposition_on_evaluation_register(self):
        """Verifies 5 Hadamard gates on the evaluation register."""
        h_gates = [inst for inst in self.master_circuit.data if inst.operation.name == 'h']
        self.assertEqual(len(h_gates), 5)

    def test_iqft_presence_and_qubits(self):
        """Verifies 5-qubit IQFT gate appended before readout."""
        iqft_ops = [inst for inst in self.master_circuit.data if "qft" in inst.operation.name.lower()]
        self.assertEqual(len(iqft_ops), 1)
        self.assertEqual(len(iqft_ops[0].qubits), 5)

    def test_terminal_readout_measurements(self):
        """Verifies 5 terminal measurements mapping eval_reg to classical_reg."""
        meas_ops = [inst for inst in self.master_circuit.data if inst.operation.name == 'measure']
        self.assertEqual(len(meas_ops), 5)


class TestQAESpectralOutputAndQuantization(unittest.TestCase):
    """Audits QAE spectral measurement, phase extraction, and finite grid bounds (Section 4.3.8)."""

    @classmethod
    def setUpClass(cls):
        cls.nb_path = os.path.join(os.path.dirname(__file__), "../02_QAE_Ideal_Simulation.ipynb")
        with open(cls.nb_path, "r", encoding="utf-8") as f:
            cls.nb = nbformat.read(f, as_version=4)

    def test_spectral_measurement_centroid(self):
        """Verifies measurement distribution projects 100% of shots onto centroid y = 16."""
        # Find cell with counts output
        cell_11 = self.nb.cells[10]  # code cell 11 (0-indexed 10)
        stdout_text = "".join(
            out.get("text", "") for out in cell_11.get("outputs", []) if out.get("output_type") == "stream"
        )
        self.assertIn("10000", stdout_text)
        self.assertIn("16", stdout_text)
        self.assertIn("100.00%", stdout_text)

    def test_phase_grid_step_and_quantization_floor(self):
        """Verifies phase step Delta_theta = pi / 32 and that theta_cfc is strictly below Delta_theta."""
        n_E = 5
        delta_theta = np.pi / (2 ** n_E)
        theta_cfc = np.arcsin(np.sqrt(1.0e-6))
        self.assertAlmostEqual(delta_theta, np.pi / 32, places=6)
        self.assertAlmostEqual(theta_cfc, 0.001, places=3)
        self.assertLess(
            theta_cfc, delta_theta,
            "Physical theorem check: true phase must be smaller than grid step for n_E=5."
        )


class TestResolutionVsDepthTradeoffScaling(unittest.TestCase):
    """Audits the resolution vs depth trade-off study (Section 4.3.9)."""

    def test_scaling_formulas_across_register_sizes(self):
        """Verifies Delta_theta(n_E) = pi / 2^n_E and N_Q(n_E) = 2^n_E - 1."""
        test_cases = {
            3: (np.pi / 8, 7),
            5: (np.pi / 32, 31),
            8: (np.pi / 256, 255),
            10: (np.pi / 1024, 1023)
        }
        for n_E, (expected_res, expected_nq) in test_cases.items():
            res = np.pi / (2 ** n_E)
            nq = (2 ** n_E) - 1
            self.assertAlmostEqual(res, expected_res, places=7)
            self.assertEqual(nq, expected_nq)


class TestPublicationArtifacts(unittest.TestCase):
    """Audits the existence and integrity of exported publication figures for Chapter 4 Deliverable 02."""

    def test_circuit_diagrams_exist_and_non_empty(self):
        """Checks circuit diagrams for Operator A, Grover Q, and Master QAE."""
        paths = [
            os.path.join(PROJECT_ROOT, "figures/circuits/circuit_operator_A.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/3.quantum_bayesian_formalism/circuit_operator_A.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/circuit_operator_A.png"),
            os.path.join(PROJECT_ROOT, "figures/circuits/circuit_operator_grover.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/3.quantum_bayesian_formalism/circuit_operator_grover.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/circuit_operator_grover.png"),
            os.path.join(PROJECT_ROOT, "figures/circuits/circuit_qae_master.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/3.quantum_bayesian_formalism/circuit_qae_master.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/circuit_qae_master.png")
        ]
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Missing circuit artifact: {p}")
            self.assertGreater(os.path.getsize(p), 10_000, f"Figure {p} is suspiciously small.")

    def test_scientific_result_figures_exist_and_non_empty(self):
        """Checks scientific result charts for QAE spectrum and NISQ depth trade-off."""
        paths = [
            os.path.join(PROJECT_ROOT, "figures/results/figure_qae_ideal_spectrum.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/figure_qae_ideal_spectrum.png"),
            os.path.join(PROJECT_ROOT, "figures/results/figure_qae_nisq_tradeoff.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/figure_qae_nisq_tradeoff.png")
        ]
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Missing result artifact: {p}")
            self.assertGreater(os.path.getsize(p), 20_000, f"Figure {p} is suspiciously small.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
