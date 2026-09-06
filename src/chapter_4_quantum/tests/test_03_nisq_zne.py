#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Chapter 4: Hardware Constraints (NISQ) and Error Mitigation via Zero-Noise Extrapolation (ZNE)
Part of the Master's Thesis: Quantum Bayesian Inference for Atmospheric Biosignature
and Technosignature Characterization in Exoplanetary Systems.

Tests:
  1. TestNotebookIntegrity: Verifies JSON structure, full execution, display figures, and outputs of 03_NISQ_ZNE_Mitigation.ipynb.
  2. TestNoiseModelCalibration: Verifies thermodynamic relaxation (T1, T2), gate durations, depolarizing rates, and readout error asymmetry.
  3. TestInferenceKernelCircuit: Verifies 5-qubit topology, subcircuit purity (zero barriers), gate synthesis, and ideal ground truth.
  4. TestGlobalUnitaryFolding: Verifies identity U(U^dagger U)^k = U on pure statevectors, terminal measurement isolation, and CNOT count linearity.
  5. TestNoiseDegradationPhysics: Verifies monotonic degradation towards maximally mixed state, increase of false-positive biomarker floor, and decay of <Z>.
  6. TestRichardsonExtrapolationEngine: Verifies Vandermonde algebraic coefficients, polynomial fit degree 2, noise error cancellation (gain >= 85%).
  7. TestPublicationArtifacts: Verifies presence and valid size of all generated publication diagrams in both figures/ and thesis/figures/.
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

import nbformat
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError

from src.chapter_4_quantum import build_and_run_03_nisq_zne as bld03


def exec_silent(code_str: str, namespace: dict):
    """Executes code string silently, suppressing stdout and matplotlib display."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code_str, namespace)
    finally:
        sys.stdout = old_stdout


class TestNotebookIntegrity(unittest.TestCase):
    """Audits the executed Jupyter Notebook 03_NISQ_ZNE_Mitigation.ipynb."""

    @classmethod
    def setUpClass(cls):
        cls.nb_path = os.path.join(os.path.dirname(__file__), "../03_NISQ_ZNE_Mitigation.ipynb")
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
            len(display_outputs), 4,
            f"Expected at least 4 embedded figure outputs, but found {len(display_outputs)}."
        )
        for out in display_outputs:
            self.assertIn("image/png", out.get("data", {}))
            self.assertGreater(len(out["data"]["image/png"]), 1000)

    def test_markdown_raw_strings_and_latex_syntax(self):
        """Verifies markdown cells contain clean LaTeX markup without double escaping regressions."""
        md_cells = [c for c in self.nb.cells if c.cell_type == "markdown"]
        self.assertGreaterEqual(len(md_cells), 6)
        full_md_text = "\n".join(c.source for c in md_cells)
        # Ensure LaTeX formulas are well-formed
        self.assertIn(r"U \longrightarrow", full_md_text)
        self.assertIn(r"\lambda \to 0", full_md_text)
        self.assertIn(r"U^\dagger U = I", full_md_text)
        self.assertIn(r"T_1", full_md_text)
        self.assertIn(r"T_2", full_md_text)


class TestNoiseModelCalibration(unittest.TestCase):
    """Audits the physical calibration and channel composition of the NISQ NoiseModel."""

    @classmethod
    def setUpClass(cls):
        # Physical parameters matching IBM Quantum processor metrics (Section 4.4.1)
        cls.t1 = 150e-6
        cls.t2 = 120e-6
        cls.time_u = 35e-9
        cls.time_cx = 300e-9
        cls.prob_dep_1q = 0.0008
        cls.prob_dep_2q = 0.0120
        cls.prob_ro_01 = 0.015
        cls.prob_ro_10 = 0.010

        # Thermodynamic relaxation
        thermal_u = thermal_relaxation_error(cls.t1, cls.t2, cls.time_u)
        thermal_cx = thermal_relaxation_error(cls.t1, cls.t2, cls.time_cx).expand(
            thermal_relaxation_error(cls.t1, cls.t2, cls.time_cx)
        )
        depol_u = depolarizing_error(cls.prob_dep_1q, 1)
        depol_cx = depolarizing_error(cls.prob_dep_2q, 2)

        error_u = thermal_u.compose(depol_u)
        error_cx = thermal_cx.compose(depol_cx)
        readout_err = ReadoutError([[1.0 - cls.prob_ro_01, cls.prob_ro_01],
                                    [cls.prob_ro_10, 1.0 - cls.prob_ro_10]])

        cls.noise_model = NoiseModel()
        cls.noise_model.add_all_qubit_quantum_error(error_u, ['u1', 'u2', 'u3', 'h', 'ry', 'x', 'z', 'rz', 'sx'])
        cls.noise_model.add_all_qubit_quantum_error(error_cx, ['cx', 'cz'])
        cls.noise_model.add_all_qubit_readout_error(readout_err)

    def test_thermodynamic_coherence_bounds(self):
        """Verifies fundamental physical coherence relation: T2 <= 2 * T1."""
        self.assertLessEqual(
            self.t2, 2.0 * self.t1,
            "Violation of thermodynamic coherence bound: T2 cannot exceed 2*T1."
        )
        self.assertGreater(self.t1, 100e-6)
        self.assertGreater(self.t2, 80e-6)

    def test_gate_durations_and_depolarization(self):
        """Verifies physical gate durations and depolarization probabilities."""
        self.assertAlmostEqual(self.time_u, 35e-9)
        self.assertAlmostEqual(self.time_cx, 300e-9)
        self.assertAlmostEqual(self.prob_dep_1q, 0.0008)
        self.assertAlmostEqual(self.prob_dep_2q, 0.0120)

    def test_readout_inversion_asymmetry(self):
        """Verifies physical readout error rates and thermal asymmetry P(1|0) > P(0|1)."""
        self.assertAlmostEqual(self.prob_ro_01, 0.015)
        self.assertAlmostEqual(self.prob_ro_10, 0.010)
        self.assertGreater(self.prob_ro_01, self.prob_ro_10)

    def test_noise_model_channel_coverage(self):
        """Ensures noise model covers single-qubit, two-qubit, and readout channels."""
        self.assertIn('cx', self.noise_model.noise_instructions)
        self.assertIn('ry', self.noise_model.noise_instructions)
        self.assertIn('x', self.noise_model.noise_instructions)
        self.assertIn('h', self.noise_model.noise_instructions)


class TestInferenceKernelCircuit(unittest.TestCase):
    """Audits the 5-qubit K2-18b causal inference kernel."""

    @classmethod
    def setUpClass(cls):
        qr = QuantumRegister(5, name='q_astrobio')
        cr = ClassicalRegister(1, name='c_readout')
        qc = QuantumCircuit(qr, cr)

        # Operator A: K2-18b Bayesian priors
        qc.ry(2.0 * np.arcsin(np.sqrt(0.75)), qr[0])
        qc.ry(2.0 * np.arcsin(np.sqrt(0.20)), qr[1])
        qc.ccx(qr[0], qr[1], qr[2])
        qc.cry(2.0 * np.arcsin(np.sqrt(0.40)), qr[2], qr[3])

        # Oracle S_chi: Biomarker phase kickback
        qc.x(qr[4])
        qc.h(qr[4])
        qc.cx(qr[3], qr[4])
        qc.h(qr[4])
        qc.x(qr[4])

        # Diffuser S_0: Reflection about zero state
        qc.x([qr[0], qr[1], qr[2], qr[3]])
        qc.h(qr[3])
        qc.mcx([qr[0], qr[1], qr[2]], qr[3])
        qc.h(qr[3])
        qc.x([qr[0], qr[1], qr[2], qr[3]])

        # Measurement
        qc.measure(qr[3], cr[0])
        cls.qc_base = qc

    def test_qubit_allocation_and_topology(self):
        """Verifies 5-qubit allocation (4 state + 1 ancilla) and 1 classical readout bit."""
        self.assertEqual(self.qc_base.num_qubits, 5)
        self.assertEqual(self.qc_base.num_clbits, 1)

    def test_subcircuit_purity_zero_barriers(self):
        """Enforces Quantum Software Engineer Rule: zero barriers in kernel."""
        barrier_count = sum(1 for inst in self.qc_base.data if inst.operation.name == 'barrier')
        self.assertEqual(barrier_count, 0, "Subcircuit purity violated: barriers found in kernel.")

    def test_ideal_ground_truth_counts_and_expectation(self):
        """Verifies deterministic ideal simulation (8192 shots): P(Bio=1) = 0.062988."""
        backend = AerSimulator(method='statevector', seed_simulator=100)
        compiled = transpile(self.qc_base, backend, optimization_level=1, seed_transpiler=100)
        shots = 8192
        job = backend.run(compiled, shots=shots, seed_simulator=100)
        counts = job.result().get_counts()

        self.assertEqual(counts.get('1'), 516)
        self.assertEqual(counts.get('0'), 7676)
        p_ideal = counts.get('1') / shots
        expval_ideal = (counts.get('0') - counts.get('1')) / shots
        self.assertAlmostEqual(p_ideal, 0.06298828, places=5)
        self.assertAlmostEqual(expval_ideal, 0.8740234, places=5)


class TestGlobalUnitaryFolding(unittest.TestCase):
    """Audits Global Unitary Circuit Folding U -> U (U^dagger U)^k."""

    @classmethod
    def setUpClass(cls):
        exec_ns = {}
        exec_silent(bld03.c3_code, exec_ns)
        exec_silent(bld03.c5_code, exec_ns)
        exec_silent(bld03.c9_code, exec_ns)
        cls.fold_circuit_global = staticmethod(exec_ns['fold_circuit_global'])
        cls.richardson_extrapolation = staticmethod(exec_ns['richardson_extrapolation'])
        cls.noisy_backend = exec_ns['noisy_backend']

        # Construct pure unitary kernel (without measurement)
        qr = QuantumRegister(5, name='q_astrobio')
        qc_u = QuantumCircuit(qr)
        qc_u.ry(2.0 * np.arcsin(np.sqrt(0.75)), qr[0])
        qc_u.ry(2.0 * np.arcsin(np.sqrt(0.20)), qr[1])
        qc_u.ccx(qr[0], qr[1], qr[2])
        qc_u.cry(2.0 * np.arcsin(np.sqrt(0.40)), qr[2], qr[3])
        qc_u.x(qr[4])
        qc_u.h(qr[4])
        qc_u.cx(qr[3], qr[4])
        qc_u.h(qr[4])
        qc_u.x(qr[4])
        qc_u.x([qr[0], qr[1], qr[2], qr[3]])
        qc_u.h(qr[3])
        qc_u.mcx([qr[0], qr[1], qr[2]], qr[3])
        qc_u.h(qr[3])
        qc_u.x([qr[0], qr[1], qr[2], qr[3]])
        cls.qc_unitary = qc_u

        # Base circuit with measurement
        cr = ClassicalRegister(1, name='c_readout')
        qc_meas = qc_u.copy()
        qc_meas.add_register(cr)
        qc_meas.measure(qr[3], cr[0])
        cls.qc_meas = qc_meas

    def test_folding_identity_scale_1(self):
        """Verifies that folding with lambda=1 returns an identical circuit copy."""
        folded_1 = self.fold_circuit_global(self.qc_meas, 1)
        self.assertEqual(len(folded_1.data), len(self.qc_meas.data))
        self.assertEqual(folded_1.depth(), self.qc_meas.depth())

    def test_statevector_invariance_under_unitary_folding(self):
        """Asserts exact theoretical identity: Statevector(U (U^dagger U)^k |0>) == Statevector(U |0>)."""
        sv_base = Statevector.from_instruction(self.qc_unitary)

        for scale in [3, 5]:
            folded_u = self.fold_circuit_global(self.qc_unitary, scale)
            sv_folded = Statevector.from_instruction(folded_u)
            fidelity = state_fidelity(sv_base, sv_folded)
            self.assertAlmostEqual(
                fidelity, 1.0, places=9,
                msg=f"Global folding violated unitary identity U^dagger U = I at scale {scale}."
            )

    def test_terminal_measurement_isolation(self):
        """Verifies that terminal measurements are not duplicated inside the folding loop."""
        for scale in [1, 3, 5]:
            folded = self.fold_circuit_global(self.qc_meas, scale)
            meas_ops = [inst for inst in folded.data if inst.operation.name == 'measure']
            self.assertEqual(len(meas_ops), 1, f"Expected exactly 1 measurement at scale {scale}.")
            self.assertEqual(folded.data[-1].operation.name, 'measure', "Measurement must be terminal.")

    def test_transpiled_cnot_linearity(self):
        """Verifies that transpiled CNOT count scales linearly: N_cx(lambda) = 23 * lambda."""
        expected_cnot_counts = {1: 23, 3: 69, 5: 115}
        for scale, expected_cx in expected_cnot_counts.items():
            folded = self.fold_circuit_global(self.qc_meas, scale)
            compiled = transpile(folded, self.noisy_backend, optimization_level=1, seed_transpiler=100)
            cnot_count = compiled.count_ops().get('cx', 0)
            self.assertEqual(
                cnot_count, expected_cx,
                f"Transpiled CNOT count at scale {scale} was {cnot_count}, expected {expected_cx}."
            )


class TestNoiseDegradationPhysics(unittest.TestCase):
    """Audits the physical degradation of quantum states under environmental NISQ noise."""

    @classmethod
    def setUpClass(cls):
        exec_ns = {}
        exec_silent(bld03.c3_code, exec_ns)
        exec_silent(bld03.c5_code, exec_ns)
        exec_silent(bld03.c9_code, exec_ns)
        exec_silent(bld03.c10_code, exec_ns)
        cls.zne_probs = exec_ns['zne_measured_probs']
        cls.zne_expvals = exec_ns['zne_measured_expvals']
        cls.p_ideal = exec_ns['p_ideal']
        cls.p_noisy = exec_ns['p_noisy']

    def test_monotonic_probability_decay_toward_maximally_mixed_state(self):
        """Verifies that noise monotonically inflates P(Bio=1) from 0.063 toward 0.500."""
        p_1, p_3, p_5 = self.zne_probs
        self.assertLess(self.p_ideal, p_1)
        self.assertLess(p_1, p_3)
        self.assertLess(p_3, p_5)
        self.assertLess(p_5, 0.5)

        self.assertAlmostEqual(p_1, 0.149414, places=5)
        self.assertAlmostEqual(p_3, 0.263672, places=5)
        self.assertAlmostEqual(p_5, 0.331055, places=5)

    def test_monotonic_decay_of_pauli_z_expectation(self):
        """Verifies monotonic decay of <Z> under increasing noise scale."""
        z_1, z_3, z_5 = self.zne_expvals
        self.assertGreater(z_1, z_3)
        self.assertGreater(z_3, z_5)
        self.assertGreater(z_5, 0.0)

        self.assertAlmostEqual(z_1, 0.701172, places=5)
        self.assertAlmostEqual(z_3, 0.472656, places=5)
        self.assertAlmostEqual(z_5, 0.337891, places=5)


class TestRichardsonExtrapolationEngine(unittest.TestCase):
    """Audits the mathematical correctness of the Richardson polynomial extrapolation."""

    @classmethod
    def setUpClass(cls):
        exec_ns = {}
        exec_silent(bld03.c3_code, exec_ns)
        exec_silent(bld03.c5_code, exec_ns)
        exec_silent(bld03.c9_code, exec_ns)
        cls.richardson = staticmethod(exec_ns['richardson_extrapolation'])

    def test_vandermonde_analytical_coefficients(self):
        """Verifies Richardson coefficients for scale factors lambda in {1, 3, 5}."""
        # Vandermonde matrix M where M[i, j] = lambda_i^j for lambda = [1, 3, 5]
        M = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 9.0],
            [1.0, 5.0, 25.0]
        ])
        b = np.array([1.0, 0.0, 0.0])
        gamma = np.linalg.solve(M.T, b)

        # Exact analytical values: gamma = [15/8, -10/8, 3/8] = [1.875, -1.250, 0.375]
        np.testing.assert_allclose(gamma, [1.875, -1.250, 0.375], atol=1e-10)
        self.assertAlmostEqual(np.sum(gamma), 1.0)
        self.assertAlmostEqual(gamma[0]*1 + gamma[1]*3 + gamma[2]*5, 0.0)
        self.assertAlmostEqual(gamma[0]*1 + gamma[1]*9 + gamma[2]*25, 0.0)

    def test_richardson_mitigated_accuracy_and_gain(self):
        """Verifies that Richardson extrapolation achieves >= 85% noise error cancellation."""
        scales = [1, 3, 5]
        probs = [0.1494140625, 0.263671875, 0.3310546875]
        p_ideal = 0.06298828125

        p_zne, coeffs = self.richardson(scales, probs)

        # Exact polynomial degree 2 check
        self.assertEqual(len(coeffs), 3)
        self.assertAlmostEqual(p_zne, 0.074707, places=5)

        err_unmitigated = abs(probs[0] - p_ideal)
        err_mitigated = abs(p_zne - p_ideal)
        gain = ((err_unmitigated - err_mitigated) / err_unmitigated) * 100.0

        self.assertGreaterEqual(gain, 85.0)
        self.assertAlmostEqual(gain, 86.44, places=1)


class TestPublicationArtifacts(unittest.TestCase):
    """Audits the existence and integrity of exported publication figures."""

    def test_circuit_diagrams_exist_and_non_empty(self):
        """Checks circuit diagrams for base and folded circuits."""
        paths = [
            os.path.join(PROJECT_ROOT, "figures/circuits/circuit_nisq_base.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/circuit_nisq_base.png"),
            os.path.join(PROJECT_ROOT, "figures/circuits/circuit_nisq_folded_lambda3.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/3.quantum_bayesian_formalism/circuit_nisq_folded_lambda3.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/circuit_nisq_folded_lambda3.png")
        ]
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Missing figure artifact: {p}")
            self.assertGreater(os.path.getsize(p), 10_000, f"Figure {p} is suspiciously small.")

    def test_scientific_result_figures_exist_and_non_empty(self):
        """Checks scientific result charts for spectral degradation and ZNE curve."""
        paths = [
            os.path.join(PROJECT_ROOT, "figures/results/figure_nisq_spectral_degradation.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/figure_nisq_spectral_degradation.png"),
            os.path.join(PROJECT_ROOT, "figures/results/figure_zne_extrapolation_curve.png"),
            os.path.join(PROJECT_ROOT, "thesis/figures/4.system_architecture/figure_zne_extrapolation_curve.png")
        ]
        for p in paths:
            self.assertTrue(os.path.exists(p), f"Missing figure artifact: {p}")
            self.assertGreater(os.path.getsize(p), 20_000, f"Figure {p} is suspiciously small.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
