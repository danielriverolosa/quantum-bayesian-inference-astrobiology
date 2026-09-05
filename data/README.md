# Scientific Data Repository & Observational Telemetry

This directory houses the foundational datasets, astrophysical priors, spectroscopic observations, and empirical simulation logs supporting the Master's Thesis:

> **Title:** *Quantum Bayesian Inference for Atmospheric Biosignature and Technosignature Characterization in Exoplanetary Systems*  
> **Author:** Daniel Rivero Losa  
> **Institution:** Universidad Antonio de Nebrija (2025/2026)

---

## 📂 Dataset Inventory

### 1. [`k218b_cpt_priors.json`](./k218b_cpt_priors.json)
**Machine-readable specification of the 9-node Astrobiological Directed Acyclic Graph (DAG) and Conditional Probability Tables (CPTs).**

* **Schema Format:** Structured JSON containing metadata, DAG adjacency edges, discrete state spaces, full combinatorial CPT matrices, and analytical marginals.
* **Nodes Encoded:**
  * $X_0$: Stellar host classification (M2.5V dwarf star, prior $P(X_0=1) = 0.75$).
  * $X_1$: Orbital habitable zone (insolation $S = 1368\,\mathrm{W/m^2}$, prior $P(X_1=1) = 0.20$).
  * $X_2$: Planetary structural regime (Hycean sub-Neptune, conditioned on $X_0, X_1$).
  * $X_3$: Biological ocean mantle (marine biosphere sustaining biogeochemical cycles, conditioned on $X_2$).
  * $X_4$: Atmospheric methane ($\mathrm{CH}_4$) spectral absorption at $1.4, 2.3, 3.3\,\mu\mathrm{m}$ ($>5\sigma$ detection).
  * $X_5$: Atmospheric carbon dioxide ($\mathrm{CO}_2$) absorption at $2.7, 4.3\,\mu\mathrm{m}$ ($\sim 3\sigma$ detection).
  * $X_6$: Tropospheric water vapor ($\mathrm{H}_2\mathrm{O}$) absorption at $1.4\,\mu\mathrm{m}$.
  * $X_7$: Volatile organosulfur biosignature (dimethyl sulfide, DMS, $\mathrm{CH}_3\mathrm{SCH}_3$).
  * $X_8$: Synthetic industrial halocarbon technosignature (CFC-12, $\mathrm{CF}_2\mathrm{Cl}_2$, prior $P(X_8=1) = 10^{-6}$).
* **Analytical Ground Truth:**
  * Joint Evidence: $P(\mathbf{e}) = P(X_4=1, X_5=1, X_6=1) = 0.095804$ ($9.58\%$).
  * Biosignature Posterior: $P(X_7=1 \mid \mathbf{e}) = 0.019630$ ($1.96\%$).
  * Technosignature Posterior: $P(X_8=1 \mid \mathbf{e}) = 5.130310 \times 10^{-6}$.
* **Thesis Cross-Reference:** Appendix A (`thesis/chapters/6.appendix_cpts.tex`) and Section 4.1.

---

### 2. [`k218b_jwst_transmission_spectrum.csv`](./k218b_jwst_transmission_spectrum.csv)
**Atmospheric transmission spectroscopy telemetry for exoplanet K2-18b observed by the James Webb Space Telescope (JWST).**

* **Observation Provenance:** Calibrated spectroscopic observations from JWST Cycle 1 GO 2722 program (*Madhusudhan et al. 2023*, ApJL, 956:L13; *Benneke et al. 2019*, ApJL, 887:L14).
* **Column Definitions:**
  * `wavelength_um`: Central spectral bin wavelength in micrometers ($\mu\mathrm{m}$), spanning $0.950\,\mu\mathrm{m}$ to $5.100\,\mu\mathrm{m}$.
  * `transit_depth_ppm`: Relative transit depth $(R_p/R_*)^2$ expressed in parts-per-million (ppm). Baseline transit depth is $\sim 2700 - 2875\,\mathrm{ppm}$.
  * `transit_depth_error_ppm`: Observational $1\sigma$ measurement uncertainty in ppm ($\sim 24 - 42\,\mathrm{ppm}$).
  * `instrument`: Observing spectrograph mode (`JWST_NIRISS_SOSS` across $0.9 - 2.5\,\mu\mathrm{m}$ or `JWST_NIRSpec_G395H` across $2.7 - 5.1\,\mu\mathrm{m}$).
  * `primary_feature`: Primary molecular cross-section absorption feature ($\mathrm{CH}_4$, $\mathrm{CO}_2$, $\mathrm{H}_2\mathrm{O}$, candidate $\mathrm{DMS}$ band, or continuum).
  * `signal_significance`: Statistical significance of the molecular absorption excess relative to flat continuum models.
* **Thesis Cross-Reference:** Chapter 1, Section 1.1, and Figure 1.1 (`thesis/figures/1.introduction/spectrum_k218b.png`).

---

### 3. [`classical_vs_quantum_benchmarks.csv`](./classical_vs_quantum_benchmarks.csv)
**Empirical simulation logs contrasting classical exact and stochastic inference against quantum algorithms.**

* **Benchmark Content:**
  * **Classical Exact Inference:** Junction Tree / Variable Elimination clique potential evaluations ($w = 2$, exact precision).
  * **Classical Rejection Sampling:** $2 \times 10^6$ forward sampling trials demonstrating the $90.42\%$ rejection ceiling and L'Ecuyer variance divergence on CFCs (1 detection, relative error $>95\%$).
  * **Classical MCMC (Metropolis-Hastings Gibbs):** $100{,}000$ transitions demonstrating Kac's recurrence trap ($0$ visits to CFC due to $\mathbb{E}[\tau] \approx 10^6$ steps).
  * **Canonical Quantum Amplitude Estimation (QAE):** 17-qubit statevector simulation ($n_E = 5$, 2,048 shots) detailing Fourier eigenvalue concentration ($80.22\%$ in $\ket{00000}$ and $19.78\%$ in $\ket{10000}$).
  * **NISQ Hardware Noise & ZNE:** 5-qubit kernel under calibrated IBM transmon decoherence ($T_1 = 150\,\mu\mathrm{s}, T_2 = 120\,\mu\mathrm{s}$) and unitary folding factors $\lambda \in \{1, 3, 5\}$, achieving $86.44\%$ error mitigation via 2nd-order Richardson extrapolation.
* **Thesis Cross-Reference:** Chapter 2, Chapter 4 (Sections 4.2–4.4), Chapter 5, and Appendix B.
