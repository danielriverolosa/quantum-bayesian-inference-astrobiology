[![Download full thesis](https://img.shields.io/badge/Download%20full%20thesis-2EA44F?style=for-the-badge&logo=github)](https://github.com/danielriverolosa/tfm/releases/tag/latest)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Qiskit 1.0+](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg?style=for-the-badge&logo=qiskit)](https://qiskit.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)

# Quantum Bayesian Inference for Atmospheric Biosignature and Technosignature Characterization in Exoplanetary Systems

**Author:** Daniel Rivero Losa  
**Supervisor:** Roberto Campos Ortiz  
**Institution:** Universidad Antonio de Nebrija — Escuela Politécnica Superior  
**Degree:** Master's Degree in Quantum Computing (Academic Year 2025/2026)  

---

## Executive Summary

This repository contains the complete code, simulation data, and LaTeX manuscript for a Master's Thesis investigating and resolving the fundamental computational bottlenecks inherent to probabilistic astrobiological inference. With next-generation observatories—principally the James Webb Space Telescope (JWST) and the impending Habitable Worlds Observatory (HWO)—generating highly coupled, multivariate spectroscopic absorption telemetry, classical probabilistic graphical models (Bayesian Networks) face insurmountable algorithmic ceilings:

1. **Exact Inference Spatial Intractability:** Variable Elimination and Junction Tree algorithms suffer from exponential memory explosion bounded by $\mathcal{O}(N \cdot d^{w+1})$ induced by high treewidth $w$ in dense photochemical networks.
2. **Stochastic Sampling Variance Divergence:** Markov Chain Monte Carlo (MCMC) and Gibbs sampling are constrained to a slow $\mathcal{O}(1/\sqrt{M})$ asymptotic convergence rate, suffering severe variance divergence when sampling ultra-rare anomalies such as industrial chlorofluorocarbons (CFCs, $p \approx 10^{-6}$).

To overcome these ceilings, this work establishes an end-to-end **Quantum Bayesian Network (QBN)** architecture coupled with **Quantum Amplitude Estimation (QAE)** in Qiskit:
* **Linear Space Complexity:** Amplitude Encoding embeds the $2^n$-dimensional joint probability distribution within $n$ physical qubits, establishing a strict $\mathcal{O}(N)$ spatial footprint.
* **Quadratic Asymptotic Speedup:** By cascading Grover iteration operators ($\mathcal{Q}$) and applying the Inverse Quantum Fourier Transform (IQFT), the algorithm achieves deterministic probability convergence scaling as $\mathcal{O}(1/M)$.
* **NISQ Physical Viability:** Circuit transpilation metrics (44,853 operations, 5,115 CNOTs) are benchmarked against thermal relaxation and depolarization noise models, and mitigated via **Zero-Noise Extrapolation (ZNE)** using Mitiq.

---

## Project Structure

The repository is organized into the following main directories and files, matching the official Nebrija Master's Thesis standard:

- **`src/`**: Contains all source code, simulation pipelines, and analysis notebooks.
  - `src/chapter_2_classical/`: Classical baseline implementations (pgmpy Bayesian networks, Junction Tree spatial stress tests, and MCMC Gibbs sampling).
  - `src/chapter_4_quantum/`: Qiskit quantum circuit synthesis (17-qubit topological register allocation, $\mathcal{A}$ operator, oracle $S_\chi$, Grover diffuser $S_0$, QPE/IQFT, and ZNE noise mitigation).
- **`thesis/`**: Contains the full LaTeX source code of the thesis monograph.
  - `main.tex`: Master document formatted for two-sided book printing ($18.5 \times 25$ cm).
  - `main_digital.tex`: Screen-optimized digital edition with wide margins for marginal notes (`\notatfm`).
  - `main_overleaf.tex`: Selective compilation wrapper for Overleaf.
  - `cover.tex`, `cover_front.tex`, `cover_back.tex`: Full physical book cover and abstract back cover.
  - `frontmatter/`: Formatting macros (`format.tex`), official title pages (`titlepage.tex`, `titlepage_digital.tex`), and dedication.
  - `chapters/`: Modular chapter sources (`1.introduction.tex`, `2.classical_limits.tex`, `3.quantum_bayesian_formalism.tex`, `4.system_architecture.tex`, `structure.tex`, `acknowledgments.tex`).
  - `figures/`: High-resolution figures categorized by chapter.
  - `references.bib`: BibLaTeX bibliography database with interactive tooltip annotations.
- **`data/`**: Stores exoplanetary atmospheric prior distributions (JWST transmission spectra of K2-18b) and simulation sampling logs.
- **`figures/`**: Generated plots, convergence comparison curves, and transpiled circuit schematics.
- **`papers/`**: Academic literature and reference papers.
- **`thesis_pdf/`**: Contains compiled digital and print-ready PDF editions of the thesis.
- **`requirements.txt`**: Complete Python dependencies.
- **`README.md`**: Project documentation and reproduction guide.

---

## Getting Started

### 1. Environment Setup

It is strongly recommended to use a clean virtual environment (conda or venv) with **Python 3.10+** (Python 3.11 recommended):

```bash
# Clone the repository
git clone https://github.com/danielriverolosa/tfm.git
cd tfm

# Create and activate conda environment
conda create -n tfm_qae python=3.11 -y
conda activate tfm_qae

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Classical Benchmarks (Chapter 2)

Execute the classical probabilistic stress test (pgmpy and MCMC):

```bash
# Python script execution
python src/chapter_2_classical/build_and_run_01_classical.py

# Or explore interactively in Jupyter
jupyter notebook src/chapter_2_classical/01_Classical_Limits_K218b.ipynb
```

### 3. Running Quantum Circuit Simulations (Chapter 4)

Execute the 17-qubit canonical QAE simulation on Qiskit Aer:

```bash
# Ideal simulation (QPE + IQFT)
python src/chapter_4_quantum/build_and_run_02_qae_ideal.py

# NISQ Noise Injection and Zero-Noise Extrapolation (ZNE)
python src/chapter_4_quantum/build_and_run_03_nisq_zne.py
```

Or explore the interactive notebooks in `src/chapter_4_quantum/`.

---

## License

This project and its accompanying manuscript are released under the terms of the **GNU General Public License v3.0 (GPLv3)**. See the `LICENSE` file for details.
