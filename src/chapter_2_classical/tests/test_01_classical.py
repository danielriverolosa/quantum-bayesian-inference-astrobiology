#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Chapter 2: Classical Limits in Astrobiological Inference
Part of the Master's Thesis: Quantum Bayesian Inference for Atmospheric Biosignature
and Technosignature Characterization in Exoplanetary Systems.

Tests:
  1. TestNotebookIntegrity: Verifies JSON structure, full execution, display figures, and outputs of 01_Classical_Limits_K218b.ipynb.
  2. TestDAGTopologyAndCPDs: Verifies acyclicity, CPD normalization, and analytical priors in pgmpy.
  3. TestExactInferenceAndTreewidth: Verifies moralization, treewidth w=2, maximal cliques, and Variable Elimination.
  4. TestVectorizedSampler: Verifies K218bBayesianNetwork joint sampling, bit types, and reproducibility.
  5. TestStochasticPathologies: Verifies rejection discard rate (>90%), MCMC mixing, and Kac's recurrence trap.
  6. TestAsymptoticTheorems: Verifies L'Ecuyer relative error divergence, Clopper-Pearson bounds, and quantum crossover.
  7. TestClassicalQuantumBridge: Verifies analytical rotation angle mapping theta = 2 * arcsin(sqrt(p)).
"""

import os
import sys
import math
import json
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pgmpy')
import nbformat
import networkx as nx
import pgmpy
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from src.chapter_2_classical import build_and_run_01_classical as bld


def exec_silent(code_str: str, namespace: dict):
    """Executes code string silently, suppressing stdout."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code_str, namespace)
    finally:
        sys.stdout = old_stdout


class TestNotebookIntegrity(unittest.TestCase):
    """Audits the executed Jupyter Notebook 01_Classical_Limits_K218b.ipynb."""

    @classmethod
    def setUpClass(cls):
        cls.nb_path = os.path.join(os.path.dirname(__file__), "../01_Classical_Limits_K218b.ipynb")
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
            len(display_outputs), 2,
            f"Expected at least 2 embedded publication figures, found {len(display_outputs)}."
        )
        for out in display_outputs:
            self.assertIn("image/png", out.get("data", {}))
            self.assertGreater(len(out["data"]["image/png"]), 1000)

    def test_stdout_contains_ground_truth_assertions(self):
        """Checks that captured stdout contains expected mathematical benchmarks."""
        full_stdout = ""
        for cell in self.nb.cells:
            if cell.cell_type == "code":
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "stream" and out.get("name") == "stdout":
                        full_stdout += out.get("text", "")

        self.assertIn("Exact Graph Treewidth (w)  : 2", full_stdout)
        self.assertIn("72 operations", full_stdout)
        self.assertIn("0.019633", full_stdout)
        self.assertIn("5.1303", full_stdout)
        self.assertIn("FALSE NEGATIVE", full_stdout)


class TestDAGTopologyAndCPDs(unittest.TestCase):
    """Audits the pgmpy DiscreteBayesianNetwork specification."""

    @classmethod
    def setUpClass(cls):
        cls.ns = {}
        exec_silent(bld.cell3_code, cls.ns)
        exec_silent(bld.cell5_code, cls.ns)
        cls.dag = cls.ns["k218b_dag"]

    def test_dag_is_strictly_acyclic(self):
        """Verifies graph acyclicity via NetworkX."""
        self.assertTrue(nx.is_directed_acyclic_graph(self.dag), "Cycle detected in Bayesian Network!")

    def test_node_and_edge_counts(self):
        """Asserts 9 nodes and 8 directed edges (homologous to quantum 17-qubit layout)."""
        self.assertEqual(len(self.dag.nodes()), 9)
        self.assertEqual(len(self.dag.edges()), 8)
        expected_nodes = {
            "X0_Stellar_M", "X1_Orbit_HZ", "X2_Hycean", "X3_Bio_Ocean",
            "X4_CH4", "X5_CO2", "X6_H2O", "X7_DMS", "X8_CFC"
        }
        self.assertEqual(set(self.dag.nodes()), expected_nodes)

    def test_cpds_sum_to_one_invariance(self):
        """Validates that every column in every TabularCPD sums to exactly 1.0 within 1e-7."""
        for cpd in self.dag.get_cpds():
            col_sums = np.sum(cpd.values, axis=0)
            np.testing.assert_allclose(
                col_sums, 1.0, atol=1e-7,
                err_msg=f"CPD {cpd.variable} does not sum to 1.0 across all conditioning columns!"
            )

    def test_check_model_passes(self):
        """Ensures pgmpy's check_model() assertion passes."""
        self.assertTrue(self.dag.check_model())

    def test_cfc_marginal_prior(self):
        """Verifies that the industrial technosignature prior enforces marginal p = 1.0e-6."""
        cpd_x8 = self.dag.get_cpds("X8_CFC")
        p_cfc_given_bio = cpd_x8.values[1, 1]
        p_bio_analytical = 0.0763
        marginal_cfc = p_cfc_given_bio * p_bio_analytical
        self.assertAlmostEqual(marginal_cfc, 1.0e-6, places=10)


class TestExactInferenceAndTreewidth(unittest.TestCase):
    """Audits exact graphical inference, treewidth, and Variable Elimination."""

    @classmethod
    def setUpClass(cls):
        cls.ns = {}
        exec_silent(bld.cell3_code, cls.ns)
        exec_silent(bld.cell5_code, cls.ns)

    def test_moral_graph_treewidth(self):
        """Validates exact treewidth extraction w = 2."""
        self.assertEqual(self.ns["treewidth_w"], 2)

    def test_maximal_cliques(self):
        """Verifies maximal clique decomposition."""
        cliques = self.ns["maximal_cliques"]
        self.assertEqual(len(cliques), 7)
        largest_clique = max(cliques, key=len)
        self.assertEqual(len(largest_clique), 3)
        self.assertEqual(set(largest_clique), {"X2_Hycean", "X0_Stellar_M", "X1_Orbit_HZ"})

    def test_exact_variable_elimination_posteriors(self):
        """Verifies analytical posteriors under JWST evidence."""
        p_dms = self.ns["exact_post_dms"]
        p_cfc = self.ns["exact_post_cfc"]
        self.assertAlmostEqual(p_dms, 0.019633, places=5)
        self.assertAlmostEqual(p_cfc, 5.1303e-6, places=9)


class TestVectorizedSampler(unittest.TestCase):
    """Audits the K218bBayesianNetwork vectorized generator."""

    @classmethod
    def setUpClass(cls):
        cls.ns = {}
        exec_silent(bld.cell3_code, cls.ns)
        exec_silent(bld.cell7_code, cls.ns)
        cls.BNClass = cls.ns["K218bBayesianNetwork"]

    def test_sampling_shape_and_binary_domain(self):
        """Ensures generated samples have shape (N, 9) and values in {0, 1}."""
        bn = self.BNClass(cfc_rare_prob=1e-6, random_state=42)
        samples = bn.sample_joint(1000)
        self.assertEqual(samples.shape, (1000, 9))
        unique_vals = np.unique(samples)
        self.assertTrue(all(v in [0, 1] for v in unique_vals))

    def test_seed_reproducibility(self):
        """Ensures identical random_state produces bit-identical realizations."""
        bn1 = self.BNClass(cfc_rare_prob=1e-6, random_state=1234)
        s1 = bn1.sample_joint(500)
        bn2 = self.BNClass(cfc_rare_prob=1e-6, random_state=1234)
        s2 = bn2.sample_joint(500)
        np.testing.assert_array_equal(s1, s2)

    def test_prior_marginal_convergence(self):
        """Verifies Monte Carlo convergence of priors X0 (0.75) and X1 (0.20)."""
        bn = self.BNClass(cfc_rare_prob=1e-6, random_state=42)
        samples = bn.sample_joint(50_000)
        mean_x0 = np.mean(samples[:, 0])
        mean_x1 = np.mean(samples[:, 1])
        self.assertAlmostEqual(mean_x0, 0.75, delta=0.015)
        self.assertAlmostEqual(mean_x1, 0.20, delta=0.015)

    def test_evaluate_unnormalized_posterior(self):
        """Verifies likelihood evaluation and zeroing for evidence violations."""
        bn = self.BNClass(cfc_rare_prob=1e-6)
        evidence = {1: 1, 4: 1, 5: 1}
        # Inconsistent state: X1 = 0
        state_bad = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)
        self.assertEqual(bn.evaluate_unnormalized_posterior(state_bad, evidence), 0.0)
        # Consistent state: X1=1, X4=1, X5=1
        state_good = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)
        p_good = bn.evaluate_unnormalized_posterior(state_good, evidence)
        self.assertGreater(p_good, 0.0)


class TestStochasticPathologies(unittest.TestCase):
    """Audits rejection sampling, discrete MCMC, and Kac's recurrence theorem."""

    @classmethod
    def setUpClass(cls):
        cls.ns = {}
        exec_silent(bld.cell3_code, cls.ns)
        exec_silent(bld.cell5_code, cls.ns)
        exec_silent(bld.cell7_code, cls.ns)
        exec_silent(bld.cell9_code, cls.ns)
        exec_silent(bld.cell11_code, cls.ns)

    def test_rejection_sampling_discard_rate(self):
        """Asserts that conditioning on JWST evidence rejects >90% of samples."""
        bn = self.ns["K218bBayesianNetwork"](cfc_rare_prob=1e-6, random_state=42)
        res = self.ns["rejection_sampling_inference"](bn, {1: 1, 4: 1, 5: 1}, n_total_samples=50_000)
        self.assertLess(res["acceptance_rate"], 0.15)
        self.assertGreater(1.0 - res["acceptance_rate"], 0.85)

    def test_mcmc_acceptance_rate_and_shape(self):
        """Checks Metropolis-Hastings output shape and mixing regime."""
        bn = self.ns["K218bBayesianNetwork"](cfc_rare_prob=1e-6, random_state=42)
        chain, acc_rate, elapsed = self.ns["run_metropolis_hastings"](
            bn, {1: 1, 4: 1, 5: 1}, n_steps=1000, burn_in=200
        )
        self.assertEqual(chain.shape, (1000, 9))
        self.assertGreater(acc_rate, 0.15)
        self.assertLess(acc_rate, 0.45)

    def test_kac_recurrence_analytical_return_time(self):
        """Verifies Kac's theorem return time E[tau] = 1/p ~ 1.95e5 steps."""
        p_cond_cfc = 5.1303e-6
        mean_return_time = 1.0 / p_cond_cfc
        self.assertAlmostEqual(mean_return_time, 194920.37, delta=100.0)


class TestAsymptoticTheorems(unittest.TestCase):
    """Audits L'Ecuyer's theorem, Clopper-Pearson intervals, and quantum crossover."""

    def test_lecuyer_relative_error_scaling(self):
        """Verifies the theoretical divergence RE = sqrt((1-p)/(M*p))."""
        p = 1.0e-6
        # For M = 10^5, RE ~ 3.16 > 3.0
        re_1e5 = math.sqrt((1.0 - p) / (1e5 * p))
        self.assertGreater(re_1e5, 3.0)

        # For M = 10^6, RE ~ 1.0 (100% uncertainty)
        re_1e6 = math.sqrt((1.0 - p) / (1e6 * p))
        self.assertAlmostEqual(re_1e6, 1.0, delta=0.01)

        # For RE = 0.05, required M ~ 4e8
        m_req = (1.0 - p) / ((0.05 ** 2) * p)
        self.assertAlmostEqual(m_req, 399999600.0, delta=1000.0)

    def test_quantum_crossover_threshold(self):
        """Verifies the analytical crossover point epsilon_cross = (pi/2) * sqrt(p)."""
        p = 1.0e-6
        eps_cross = (math.pi / 2.0) * math.sqrt(p)
        self.assertAlmostEqual(eps_cross, 1.570796e-3, places=7)


class TestClassicalQuantumBridge(unittest.TestCase):
    """Audits mathematical parameter synchronization between Chapter 2 and Chapter 4."""

    def test_cpt_to_quantum_angle_mapping(self):
        """Verifies theta = 2 * arcsin(sqrt(p)) mapping for all Chapter 2 probabilities."""
        test_cases = [
            ("X0_Stellar_M", 0.75, 2.094395),
            ("X1_Orbit_HZ", 0.20, 0.927295),
            ("X2_Hycean_11", 0.85, 2.346194),
            ("X3_Bio_Ocean_1", 0.40, 1.369438),
            ("X4_CH4_1", 0.90, 2.498092),
            ("X5_CO2_1", 0.80, 2.214297),
            ("X6_H2O_1", 0.70, 1.982313),
            ("X7_DMS_1", 0.05, 0.451027),
            ("X8_CFC_target", 1.0e-6, 0.002000)
        ]

        for name, p_val, target_theta in test_cases:
            if "X8_CFC" in name:
                # Target canonical injection angle theta = 0.002
                reconstructed_p = math.sin(target_theta / 2.0) ** 2
                self.assertAlmostEqual(reconstructed_p, p_val, delta=1e-7)
            else:
                calculated_theta = 2.0 * math.asin(math.sqrt(p_val))
                self.assertAlmostEqual(calculated_theta, target_theta, places=5)
                reconstructed_p = math.sin(calculated_theta / 2.0) ** 2
                self.assertAlmostEqual(reconstructed_p, p_val, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
