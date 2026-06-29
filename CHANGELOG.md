# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.39.0] - 2026-06-29

### Added
- **Dynamic Versioning**: Transitioned to `setuptools_scm` for dynamic version management, ensuring synchronization across the package.
- **Interactive Tutorials**: Added distance geometry and Ramachandran explorer tutorials to the documentation.
- **Modern Badges**: Updated documentation with modern Shields.io and Colab badges.

### Fixed
- **Colab Notebook Compatibility**: Resolved `numpy<2` conflicts and py3Dmol WebGL context exhaustion issues to stabilize Jupyter Notebook execution in Google Colab.
- **Biophysical Validation**: Corrected Histidine tautomer recognition and $i, i+4$ salt bridge formation logic.
- **Type Checking**: Resolved extensive `mypy` typing errors related to Matplotlib, Numba, and array shape warnings.

## [1.38.1] - 2026-06-07

### Added
- Added interactive Electrostatic Surfaces & pH Titration Lab Jupyter Notebook tutorial.

### Fixed
- Resolved `NamedTemporaryFile` permission issues causing test failures on Windows.
- Resolved HuggingFace cache loading `TypeError` on Windows by explicitly setting `HF_HOME`.

### Security
- Removed compromised `polyfill.io` CDN script from MkDocs configuration to resolve supply-chain vulnerability.

## [1.38.0] - 2026-06-02

### Added
- **New Visual Lab Tutorials**: Added five new substantive 'Lab' experiences (Mirror World, Molecular Machine, Prion Chameleon, etc.) with enhanced visualisations and interactive mechanics.
- **Modern JAX Stack Support**: Updated core dependencies to support the latest JAX/NVIDIA stack and lifted version caps for broader environment compatibility.

### Changed
- **SAXS Module Extraction**: Extracted heavy SAXS simulation logic into the standalone `synth-saxs` package. The `synth_pdb.saxs` module is now a compatibility shim, reducing the core package footprint while maintaining backward compatibility.
- **Numpy 2.0 Compatibility**: Extensive codebase updates to ensure full compatibility with Numpy 2.x while maintaining support for 1.x.

### Fixed
- **RDC Q-Factor Edge Cases**: Refactored Q-factor calculation to correctly return `nan` for empty datasets, preventing false "perfect agreement" signals in automated pipelines.
- **CI Robustness**: Resolved several systemic CI failures related to environment-specific paths and library warnings.
- **Type Safety**: Expanded type annotations across geometry and biophysics modules to ensure strict `mypy` compliance.

## [1.37.0] - 2026-05-14

### Added
- **Modern Formats Support**: Native support for mmCIF (PDBx) and BinaryCIF (BCIF) output via the `--format` flag.
- **Massive Protein Support**: Ability to generate structures with over 99,999 atoms and 62 chains using the CIF format.
- **BinaryCIF Export**: High-performance binary format for AI pipelines and web-based visualization (Mol*).
- **Interactive Tutorials**: Restored and improved interactive Jupyter Notebooks for NeRF geometry and protein quality assessment.
- **ASCII-Only Standard**: Package-wide migration to ASCII-safe strings to prevent encoding crashes in restricted terminal environments.

### Fixed
- **Core Performance**: Vectorized B-factor and occupancy calculations using NumPy, resulting in >10x speedup for large protein generation.
- **Validator Optimization**: Significant performance improvements to `PDBValidator` for large structures.
- **Reproducibility**: Restored strict seed propagation for all generation modes and formats.

## [1.36.0] - 2026-05-04

### Added
- **Enhanced Prompt Transparency**: The CLI now explicitly logs interpreted command-line arguments when using the `--prompt` interface, providing full visibility into how the LLM maps natural language to biophysical parameters.
- **Intelligent CLI Logging**: Implemented a custom `CLIFormatter` that keeps standard output clean while explicitly labeling `WARNING:` and `ERROR:` messages for improved diagnostic visibility.
- **Robust Prompt Documentation**: Added a "Tips for Reliable Generation" section to the Prompt-to-Protein guide, covering SLM hallucination management and clinical phrasing best practices.
- **Explicit GPU Control**: New `--platform` and `--precision` flags for OpenMM-driven energy minimization, allowing users to force `CUDA`, `Metal`, or `OpenCL` platforms and choose between `single`, `mixed`, or `double` precision for optimized performance.
- **Physics Progress Tracking**: Integrated `LoggingMinimizationReporter` to provide real-time energy convergence feedback during minimization at the `DEBUG` level.
- **Hardware Acceleration Benchmark**: New `scripts/benchmark_hardware.py` utility to quantify performance gains across different compute platforms (CPU vs GPU).

### Fixed
- **Metal Coordination Realism**: Refactored the Zinc Finger detection logic to require ligands from 4 unique residues (chain + residue ID mapping). This eliminates false-positive coordination sites caused by "double-counting" multiple atoms (e.g., His NE2/ND1) from the same residue.
- **CLI Logging Stream**: Standardized logging output to `stderr` to ensure compatibility with shell piping and automated test suites.
- **Memory Management**: Implemented caching for functional OpenMM platforms and improved cleanup in `simulate_trajectory` to resolve intermittent memory leaks and redundant initialization overhead.
- **Dependency Stability**: Explicitly excluded buggy OpenMM versions (8.5.0, 8.5.1) to protect against known upstream memory leaks.

## [1.35.0] - 2026-05-02

### Added
- **Circular Dichroism (CD) Simulator**: New `synth_pdb.cd_simulator` module for generating synthetic Far-UV CD spectra from 3D structures.
- Interactive CD Lab tutorial in `examples/interactive_tutorials/`.

## [1.34.2] - 2026-04-28

### Fixed
- **BMRB Validation Pipeline**: Updated `PDBValidationAPI` to use the unified PDBe v2 API, resolving 404 errors during metadata retrieval. Added a compatibility shim to maintain support for legacy field names in existing tutorials.

## [1.34.1] - 2026-04-27

### Added
- **Updated Gallery Images**: Enhanced visual snapshots in the examples gallery for better structural clarity.
- **Version Synchronization**: Unified version metadata across documentation header, PyPI badges, citation guides, and Streamlit app.

## [1.34.0] - 2026-04-27

### Added
- **High-Fidelity SAXS Visualization**: Integrated Standard, Kratky, and Guinier plots into the CLI and core library for biophysical assessment of protein folding and dimensions.
- **Scientific Rigor Suite**: New automated validation against peer-reviewed benchmarks (Waasmaier & Kirfel 1995 for form factors; Kratky & Porod 1949 for folding signatures).
- **Multi-Modal Dataset Factory**: New `scripts/build_multimodal_dataset.py` for high-throughput generation of synchronized PDB, MRC, SAXS, and NMR data for AI training.
- **Robust Sequence Parsing**: Support for 4-letter D-amino acid shorthands (DALA, dALA) and Selenocysteine (SEC/U).
- **Interactive Cryo-EM & SAXS Lab**: New tutorial showcasing ensemble-averaged density and scattering simulation.
- **Embedded Gallery Visuals**: Added snapshots for all major structural types in the examples documentation.

### Fixed
- **CLI Robustness**: Fixed `docking` mode to support on-the-fly generation from sequence strings.
- **PDB Compliance**: Updated atom formatting for 2-character elements (like Selenium) to meet rigorous PDB standards.
- **Browser Automation**: Mocked `webbrowser` in test suites to prevent unintended popup windows during CI runs.

## [1.33.0] - 2026-04-26

### Added
- **Comprehensive API Documentation**: 11 new modules fully documented including `cryo_em`, `saxs`, `docking`, `cofactors`, and `special_chemistry`.
- **New Interactive Tutorials**:
    - `cryo_em_saxs_lab`: Visualizing resolution and conformational heterogeneity.
    - `bmrb_validation`: Programmatic validation against experimental NMR data.
- **Multimodal Science Guide**: New guide explaining integrated scientific workflows.
- **Enhanced Ensemble Support**: Added `PeptideGenerator.generate_ensemble()` for high-performance vectorized generation.

### Fixed
- **Memory Leaks**: Resolved critical memory leaks in OpenMM physics engine and `BatchedGenerator` template caching.
- **Tutorial Modernization**: Updated `gfp_molecular_forge` and `neural_nmr_pipeline` to use the latest APIs.
- **Test Integrity**: Standardized documentation integrity tests and added coverage for new features.

## [1.32.0] - 2026-04-24

### Added
- **AI Mode: Structure Clustering**: Implemented RMSD-based clustering for protein conformational ensembles.
    - New CLI flag: `--mode ai --ai-op cluster`
    - Supports `--input-pattern` (glob) and `--n-clusters`.
    - Automatically superimposes structures to a reference before clustering.
    - Exports representative "medoid" structures for each cluster to a user-defined directory.
- **Improved Testing**: Added `tests/test_clustering.py` for full-workflow CLI validation.

## [1.31.0] - 2026-04-24

### Added
- **GNN Quality Filter Calibration**: New sensitivity suite in `tests/test_gnn_calibration.py` validating that the GNN model correctly responds to coordinate noise, steric clashes and backbone distortions.
- **Comprehensive Testing Suite**: Added massive coverage expansion for evolution kernels, packing algorithms, scoring functions and geometric edge cases (RMSD/superposition).
- **Non-Standard Residue Support**: Enhanced chemical shift and RDC predictors to gracefully handle non-standard residues (PTMs, D-amino acids) while maintaining pedagogical documentation density.
- **Local Notebook Validation**: Added scripts and environment setups (`scripts/setup_colab_venv.sh`, `test_notebooks_local.sh`) to verify tutorial notebooks in a production-like local environment.

### Fixed
- **D-Amino Acid Stereochemistry**: Corrected critical bugs in D-amino acid generation, ensuring biophysically accurate backbone mirroring and side-chain placement.
- **Chiral NMR Inversion**: Implemented a dual-pass coordinate inversion strategy for D-amino acid chemical shifts, ensuring predicted values respect chiral symmetry.
- **Physics Engine Stability**: Resolved memory leaks and ensured consistent implicit solvent behavior across the `EnergyMinimizer` pipeline.
- **Generator Parity**: Guaranteed 1:1 structural parity between `BatchedGenerator` and the standard `PeptideGenerator`.
- **CLI Robustness**: Strengthened argument validation in `main.py` and added regression tests for CLI flag collisions.
- **PTM Template Mismatch**: Fixed an issue where PTM residue templates (SEP/TPO/PTR) were incorrectly handled during physics preprocessing.

### Changed
- **Documentation Density**: Significantly increased inline "Educational Notes" in `coupling.py`, `msa.py` and `physics.py`.
- **CI Modernization**: Added Numpy 2.3.5 to the automated test matrix; refactored the J-Coupling shim for structural TDD.

## [1.30.0] - 2026-04-22

### Added
- **New Documentation Pages**:
  - `docs/api/rdc.md`: Full API reference for `synth_pdb.rdc` — Saupe-matrix RDC back-calculation, Q-factor interpretation table, alignment media comparison and complete workflow example.
  - `docs/api/ensemble.md`: Full API reference for `synth_pdb.ensemble` — `DAOPCalculator`, `EnsembleStatistics`, and `QualityAssessment` with Tejero 2013 quality thresholds.
  - `docs/science/ensemble-analysis.md`: Science background page explaining DAOP circular statistics, coordinate RMSD/RMSF, medoid selection, and IDP ensemble caveats.
- **Documentation Expansions**:
  - `docs/getting-started/first-structure.md`: Rewrote stub into a full 5-step walkthrough (generate → visualise → validate → minimise → Python API).
  - `docs/physics.md`: Added `simulate_trajectory()` section with Kabsch-aligned RMSF example and autodoc integration.
  - `docs/nmr.md`: Expanded from 26-line stub to comprehensive coverage of all NMR observables (NOE/NEF, chemical shifts, relaxation, RPF scores, RDCs, J-couplings).
- **README Additions**:
  - Added 6 new feature blocks: RDC module, NMR Ensemble Analysis, MSA Co-Evolution, PLM Embeddings, GNN Quality Scorer, and Engh & Huber / Top2018 validation details.
  - Updated project structure tree to reflect current 30+ module layout including `geometry/`, `ensemble/`, `quality/gnn/`, `rdc.py`, `msa.py`, `plm.py`.
  - Added 20 new glossary entries (BMRB, DAOP, DCA, Engh & Huber, ESM-2/PLM, GNN, IDR/IDP, Kauzmann, Magic Step, MCMC, Orientogram, pLDDT, Potts Model, PPII, PRE, Q-factor, RDC, RMSF, Saupe Matrix, Top2018).
  - Added 11 new references (Engh & Huber 1991, Hyberts 1992, Tejero 2013, Tjandra & Bax 1997, Saupe 1968, Morcos 2011, Lin 2023, Jumper 2021, Ruff & Pappu 2021, Clore & Iwahara 2009, Cornilescu 1998).

### Fixed
- **Docs site badges**: Replaced broken `Open in Colab` badge (pointed to non-existent `demo.ipynb`) with correct link to `examples/interactive_tutorials/virtual_nmr_spectrometer.ipynb`.
- **PyPI badge**: Replaced unreliable `badge.fury.io` endpoint with live `shields.io/pypi/v` badge.
- **mkdocs.yml**: Added `api/rdc.md` and `api/ensemble.md` to the API Reference nav; added `science/ensemble-analysis.md` to the Scientific Background nav.

## [1.29.0] - 2026-04-20

### Added
- **Aggressive Scientific Validation Framework**: Implemented a comprehensive, evidence-based quality filter in `PDBValidator` that proactively defends generated models using peer-reviewed biophysical standards.
  - **Engh & Huber Z-Scores**: Integrated backbone geometry validation against the landmark Engh & Huber (1991) standard deviations for bond lengths and angles.
  - **Dunbrack Rotamer Quality**: Added sidechain packaging assessment based on the Dunbrack Rotamer Library, ensuring models follow natural PDB conformational probabilities.
  - **SASA-based Burial Validation**: Implemented `calculate_residue_sasa()` using the Shrake-Rupley algorithm (via `biotite`) and a "Burial Ratio" metric to verify hydrophobic core formation (Kauzmann 1959).
  - **Modern Geometric Standards**: Upgraded Ramachandran validation logic to align with the high-resolution Top2018 dataset (~15,000 chains).
- **Structural Integrity Report**: New `get_quality_report()` method providing a multi-layered defense (Geometry + Physics + Biophysics) for structural plausibility.
- **NMR Integrations**: New validation tests against **mirror-image Protein G (BMRB 36464)** and **AI-refined Influenza NS1B-CTD (BMRB 51544)**.

### Fixed
- **Backbone Superimposition**: Corrected a critical regression where 4-atom superimposition caused backbone distortions. Restored stable 3-atom transformation (N, CA, C) for consistent global frame orientation.
- **D-Amino Acid Chirality**: Fixed torsion negation logic to correctly handle mirror-image proteins via sidechain reflection while maintaining standard L-conformation backbone traces.
- **CI/CD Cleanup**: Resolved numerous `mypy` and `ruff` issues across the source package; implemented warning filters to suppress external library noise (e.g., `biotite` deprecations).

## [1.28.0] - 2026-04-04

### Added
- **New Ensemble Module** (`synth_pdb.ensemble`): A new subpackage housing
  NMR ensemble analysis algorithms, extracted from `pdbstat-python` and
  elevated to a general-purpose, scientifically validated library.
- **`DAOPCalculator`** (`synth_pdb.ensemble.daop`): Implements the circular
  order-parameter formula from Hyberts et al. (1992) *Protein Science* 1:736
  for quantifying dihedral angle consistency across an NMR ensemble.
  Includes `calculate_order_parameter`, `find_well_defined_residues`
  (PDBStat S(φ)+S(ψ) ≥ 1.8 convention), and `calculate_backbone_daop`.
- **`EnsembleStatistics`** (`synth_pdb.ensemble.statistics`): Typed dataclass
  replacing plain `Dict[str, float]` for ensemble quality metrics (pairwise
  RMSD, RMSF, medoid, well-defined residues) with `precision` and
  `overall_quality` computed properties and `to_dict`/`from_dict` round-trips.
  Quality thresholds from Tejero et al. (2013) *J Biomol NMR* 56:337.
- **`QualityAssessment`** (`synth_pdb.ensemble.statistics`): Lightweight typed
  dataclass for the precision/overall-quality assessment pair.

---

## [1.27.0] - 2026-03-30

### Added
- **New Geometry Module** (`synth_pdb.geometry`): Modernized core protein geometry logic based upon `pdbstat-python`, including symmetry-aware RMSD calculation and rigid-body superposition.
- **Improved J-Coupling Prediction**: Updated `tests/test_coupling.py` to support `NaN` key exclusions in the latest `synth-nmr` predictor.
- **Extensive Geometry Testing**: Added edge-case kernels tests for RMSD, dihedral angles and Kabsch superposition.

### Fixed
- **Geometric Strain Formula**: Corrected calculation of omega dihedral deviation from 180° for consistent residue strain reporting.
- **Generator Robustness**: Resolved systemic `TypeError` regressions caused by `pytest-cov` environment interference with `numpy._NoValueType`.

### Changed
- **Dependency Update**: Bumped `synth-nmr` dependency to `>=0.9.0` for improved physical validation and J-coupling support.
- **CI/CD Stabilization**: Hard-pinned documentation dependencies (`mkdocstrings`, `mkdocstrings-python`, and `pygments`) to ensure build site reproducibility and resolve `NoneType` errors in the MkDocs pipeline.

---

## [1.26.0] - 2026-03-29

### Added
- **MSA Modernization**: Completely overhauled `synth_pdb.msa` to simulate biologically grounded evolutionary constraints.
  - Added $O(1)$ Delta-Energy evaluation (`calculate_delta_energy`) yielding a ~500x acceleration in MSA generation via Metropolis-Hastings sampling.
  - Implemented the "Magic Step" coupled mutations allowing the MCMC simulation to simultaneously mutate contacting residues (defaults to 20% proposal rate).
  - Added SASA Selective Pressure via `rel_sasa` enforcing geometric isolation of the hydrophobic core.
  - Expanded Potts Model `J_ij` coupling to factor in Electrostatic Salt Bridges (rewards) and Repulsions (penalties).
- **Expanded Physics Testing Suite**: Added dedicated `test_physics_expansion.py` boosting the physics engine coverage layout (fixing systemic segmentation faults driven by template caching errors).
- **Formalized Docstrings**: Embedded the biological rationale of Hydrophobic Collapse and "Magic Steps" directly into the algorithm source code as dense "Educational Notes" and created a dedicated MkDocs "Co-Evolution" background page.

### Fixed
- **Type Checking Strictness**: Fixed an unhandled `mypy` typing coercion causing the energy matrices to leak generalized `Any` objects.

## [1.25.0] - 2026-03-10

### Changed

- **Dependency Update**: Required `synth-nmr>=0.8.0` to incorporate the latest NMR simulation engine improvements and physical validator updates.

## [1.24.0] - 2026-03-08

### Added

- **Synthetic MSA Generation with Co-Evolution** (`synth_pdb/msa.py`): New generative workflow simulating Markov Chain Monte Carlo (MCMC) evolution over a 3D structural Potts Model, enabling zero-shot generation of deep multiple sequence alignments to test DCA/AlphaFold inputs.
- **MSA Mutual Information Tutorial** (`examples/interactive_tutorials/coevolution_msa_factory.ipynb`): New interactive Jupyter notebook visually demonstrating how the physics-based MCMC sampler enforces co-evolution signals that recover the 3D contact map.
- **High-Density Physics Documentation Suite**: Massively expanded the inline educational docstrings across `synth_pdb/physics.py` (>60% line density), exposing the physical rationale behind NVT Langevin dynamics, Amber forcefields, implicit solvents, and metadata restoration strategies.

## [1.23.0] - 2026-03-07

### Added

- **IDP Conformational Ensembles Tutorial** (`examples/interactive_tutorials/idp_ensemble_validation.ipynb`): New interactive notebook validating synthetic IDP ensembles mathematically and visually against $1/r^6$ Paramagnetic Relaxation Enhancement (PRE) NMR phenomena.
- **AlphaFold pLDDT vs NMR S² Tutorial** (`examples/interactive_tutorials/alphafold_vs_nmr_dynamics.ipynb`): New interactive notebook modeling an unstructured loop to demonstrate the inverse correlation between static AI prediction confidence (pLDDT) and physical NMR flexibility.
- **IDP Theoretical Physics Documentation** (`docs/science/idp-dynamics.md`): Added comprehensive biophysical theory documentation explaining how the `BatchedGenerator` successfully handles intrinsically disordered regions compared to rigid crystalline tools.

### Fixed

- **Py3Dmol Widget Crash on Local Jupyter**: Replaced IPyWidgets `HBox` layout with Py3Dmol's native `viewergrid=(1,2)` API across tutorials. This stops localized browser javascript injection blocks from breaking dual-viewer structures (fixing "3Dmol.js failed to load" spam).
- **GitHub Actions Formatting CI**: Ran `ruff check --fix .` across the codebase, fixing un-sorted imports and line-length breakages introduced by `physics.py` trajectory logging features, restoring a completely 'green' `test.yml` CI workflow.

---

## [1.22.0] - 2026-03-05

### Added

- **Residual Dipolar Coupling (RDC) module** (`synth_pdb/rdc.py`): New `calculate_rdcs()`
  function implementing the Saupe-matrix formalism for computing backbone N–H RDCs given
  an alignment tensor defined by axial component `Da` and rhombicity `R`.
- **RDC tutorial notebook** (`examples/interactive_tutorials/rdc_alignment_explorer.ipynb`):
  Standalone Colab-compatible notebook demonstrating RDC calculation, Q-factor
  validation against published ubiquitin data (1D3Z), and interactive alignment-tensor
  exploration.
- **Virtual NMR Spectrometer — RDC section** (`virtual_nmr_spectrometer.ipynb` cells
  13–14): Interactive RDC fingerprint panel with `ipywidgets` sliders for `Da` and `R`,
  and a `py3Dmol` 3D view coloured by RDC sign and magnitude.
- **Virtual NMR Spectrometer — Shift Predictor Comparison** (cells 15–16): Side-by-side
  HSQC scatter plots (empirical vs SHIFTX2), per-residue Δδ(¹H) bar chart, and a
  summary statistics table with literature RMSD benchmarks. Graceful fallback when
  SHIFTX2 is not installed.

### Fixed

- **`synth_pdb/physics.py` — `EnergyMinimizer`**: Eliminated a spurious
  `WARNING: Forcefield does not support implicitSolvent` log message that fired on
  every run. Root cause: the OBC2 implicit-solvent XML was correctly loaded at
  construction time, but `_create_system_robust` also passed `implicitSolvent=app.OBC2`
  as a `createSystem()` kwarg — which modern OpenMM rejects as unused. Fix: clear
  `self.implicit_solvent_enum` after the XML is appended so the kwarg path is never
  exercised.
- **`synth_pdb/relaxation.py` (via `synth-nmr>=0.7.2`)**: Heteronuclear NOE values
  were identical for every residue (a flat horizontal line) because S² cancelled
  exactly in the ratio cross-relaxation/R₁ when `tau_f=0`. Fix in `synth-nmr 0.7.2`:
  a per-residue fast internal motion timescale `tau_f = (1−S²)×500 ps + 50 ps` now
  breaks the cancellation. Flexible termini correctly give low/negative HetNOE; rigid
  helices give HetNOE ≈ 0.5–0.8. Ref: Lipari & Szabo (1982) *J Am Chem Soc* 104:4546.
- **`virtual_nmr_spectrometer.ipynb` Colab setup**: Added explicit
  `pip install synth-nmr>=0.7.2` to the setup cell (was silently skipped because
  `synth-pdb` was installed with `--no-deps`), fixing a `ModuleNotFoundError: No module named 'synth_nmr'` crash on every fresh Colab runtime.
- **`virtual_nmr_spectrometer.ipynb` Neural Shift cell**: Added a `torch_geometric`
  availability check with a clear install hint (`pip install synth-nmr[ml]`). Previously
  the cell crashed with `ImportError: torch_geometric is required` when the ML extras
  were absent (the default in Colab).

### Changed

- **Dependency**: Tightened `synth-nmr` lower bound from `>=0.6.1` to `>=0.7.2` to
  ensure the HetNOE bug fix and all NMR accuracy improvements are always present.

---

## [1.20.0] - 2026-02-21


### Changed
- **Numpy Compatibility**: Relaxed numpy dependency pin to `<3.0.0` to resolve binary incompatibility errors (`numpy.dtype size changed`) in Google Colab and other environments using `numpy 2.x`.
- **Dependency Update**: Bumped `synth-nmr` dependency to `>=0.6.1` to incorporate upstream numpy compatibility fixes.

---

## [1.19.1] - 2026-02-19

### Fixed
- **PLM Tutorial Bug**: Fixed a `TypeError` in `examples/ml_integration/plm_embeddings.ipynb` where an obsolete `ss_type` argument was used instead of `conformation` for `generate_pdb_content()`.

### Changed
- **Educational Enhancements**: Significantly expanded the PLM tutorial notebook (`plm_embeddings.ipynb`) with plain-language explanations of Protein Language Models, embeddings, similarity metrics, and UMAP clustering for chemists and biologists without machine learning backgrounds.

---

## [1.19.0] - 2026-02-19

### Added
- **PLM Embeddings**: Integrated ESM-2 protein language model support via `synth_pdb.quality.plm`. Generates per-residue and pooled embeddings from generated structures, enabling zero-shot quality scoring and downstream ML tasks.
- **PLM Tutorial**: Added `examples/ml_integration/plm_embeddings.ipynb` Colab-compatible notebook demonstrating ESM-2 embedding extraction and visualization.
- New optional dependency group `[plm]` (`torch>=2.0.0`, `transformers>=4.30.0`).

---

## [1.18.0] - 2026-02-19

### Added
- **GNN Quality Scorer**: New `synth_pdb.quality.gnn` module with a Graph Neural Network model for protein structure quality assessment. Nodes represent residues; edges encode sequence proximity and spatial contacts.
- **GNN Training Script**: `scripts/train_gnn_quality.py` for training the GNN on labelled structure datasets.
- **Random Forest Baseline**: Included alongside GNN as an interpretable quality-filter baseline (`synth_pdb.quality.rf_model`).
- New optional dependency group `[gnn]` (`torch>=2.0.0`, `torch_geometric>=2.4.0`, `scikit-learn`, `joblib`).

### Changed
- `[ai]` optional-dependency group now covers the Random Forest model; `[gnn]` covers the full GNN stack.

---

## [1.17.0] - 2026-02-11

### Changed
- **NMR Package Integration**: NMR functionality now provided by the [`synth-nmr`](https://github.com/elkins/synth-nmr) package
- Maintained 100% backward compatibility via compatibility shims
- All existing code continues to work without changes

### Added
- Dependency on `synth-nmr>=0.1.0`

### Removed
- ~1,200 lines of duplicate NMR code (now imported from synth-nmr)

## [1.15.0] - 2026-02-02
### Added
- **ML Handover Notebooks**: Added zero-copy handover examples for **JAX**, **MLX**, and **PyTorch** in `examples/ml_loading/`.
- **Vectorized Batch Generation**: Exposed `BatchedGenerator` and `BatchedPeptide` via `synth_pdb.generator` for high-performance AI training pipelines.
- **Salt Bridge Consolidation**: Unified salt bridge force parameters to prevent global parameter conflicts in complex structures.

### Fixed
- **Cyclic Peptide Physics**: Refined covalent ring closure using a surgical linear-to-cyclic conversion strategy, bypassing OpenMM template matching limitations.
- **Physics Preprocessing**: Resolved an `UnboundLocalError` in the simulation engine that caused crashes in specific edge-case topologies.
- **Notebook Robustness**: Added graceful dependency checks and precision-safe assertions (`assert_allclose`) to ML handover notebooks.
- **Test Stability**: Suppressed verbose Numba debug logging and fixed mock assertion failures in the physics test suite.

## [1.14.0] - 2026-01-31
### Added
- **D-Amino Acid Support**: Support for generating and validating peptides with D-amino acids using the `D-` prefix in sequences.
- **PDB Compatibility**: Automatic conversion of D-amino acids to standard 3-letter codes (e.g., `DAL`, `DPH`).
- **Educational Enhancements**: Detailed comments explaining chiral mirroring and stereochemistry.
- **New Tests**: Comprehensive TDD suite for D-amino acid generation and validation.


## [1.13.1] - 2026-01-30

### Added
- **EGF Generation Example**: Added a new example script `examples/generate_egf.py` demonstrating the generation of a complex 53-residue protein with disarmament minimization and synthetic NMR data.

### Fixed
- **Validator Stability**: Fixed a critical `TypeError` in `validator.py` where terminal caps or incomplete backbone atoms could cause a crash during bond angle validation.
- **Regression Testing**: Added automated regression tests for the validator crash to prevent future regressions.

## [1.13.0] - 2026-01-30

### Added
- **Cyclic Peptide Support**: Implemented head-to-tail macrocyclization with automated terminal atom removal (OXT/H1-3) and physics-based bond closure.
- **Numba JIT Acceleration**: Integrated `@njit` compilation for NeRF geometry engines, Lipari-Szabo spectral density, and Ring Current calculations, achieving **50-100x speedups**.
- **Visual Connectivity**: Automated `CONECT` record generation for cyclic bonds and disulfide bridges to ensure seamless representation in the 3D viewer.
- **Educational References**: Added seminal scientific citations for macrocyclization (Horton, Craik) and deep-dive biophysical commentary to the codebase and README.

### Fixed
- **Proline Minimization**: Resolved a bug where Proline residues in cyclic peptides caused OpenMM template errors by stripping illegal amide hydrogens.
- **Metadata Persistence**: Fixed an issue where PTM residue names (SEP, TPO, PTR) were lost during the minimization-to-assembly pipeline.

## [1.12.0] - 2026-01-29

### Added
- **Beta-Turn Geometries**: Implemented physics-based construction for Type I, II, I', II', and VIII beta-turns. Added `--structure` CLI argument (e.g., `'3-6:typeII'`) for precise loop modeling.
- **J-Coupling Prediction**: Added generation of $^3J_{H_NH_\alpha}$ scalar couplings using the Karplus equation ($A \cos^2\phi + B \cos\phi + C$). Output available via `main.py` (CSV export).
- **Cis-Proline Isomerization**: Added `--cis-proline-frequency` to simulate biologically realistic non-canonical conformations (~5% frequency).
- **Post-Translational Modifications (PTMs)**: Added `--phosphorylation-rate` to simulate Ser/Thr/Tyr phosphorylation, converting residues to SEP/TPO/PTR for downstream MD/NMR analysis.
- **Performance**: Vectorized geometry kernels and improved OpenMM platform selection (CUDA/Metal preference with CPU fallback).

### Fixed
- **CLI Regressions**: Fixed `AttributeError` caused by missing CLI arguments for new biophysics features.
- **Variable Scoping**: Resolved `NameError` in `generator.py` related to rotamer selection aliases.
## [1.11.0] - 2026-01-29

### Added
- **Pre-Proline Backbone Realism**: Implemented specific conformational sampling for residues preceding Proline (favoring Extended/Beta, restricting Alpha). This significantly reduces steric clashes.
- **Biophysical Efficiency**: Validated that improving backbone realism reduces energy minimization time by **>60%** (2.42s -> 0.91s) by providing physically sound starting structures.
- **Advanced Chemical Shifts**: Added **Ring Current Effects** (Haigh-Mallion point-dipole model) to chemical shift prediction. Protons above aromatic rings are now correctly shielded, and in-plane protons deshielded.
- **SASA-Modulated Relaxation**: Implemented Solvent Accessible Surface Area (SASA) calculation to modulate Order Parameters ($S^2$). Buried residues are now modeled as more rigid than exposed ones.
- **SSBOND Robustness**: Enhanced disulfide bond detection with strict 1-to-1 pairing logic and a defined capture radius (8.0 Å) to prevent multi-bond artifacts in dense structures.

### Fixed
- **SSBOND Regression**: Fixed an issue where single Cysteines could form multiple disulfide bonds.
- **SASA Calculations**: Added robust handling for `NaN` values in SASA calculation for small/mock structures.

## [1.10.0] - 2026-01-28

### Added
- **Full Rotamer Library**: Expanded backbone-dependent rotamer library to support **All 20 Standard Amino Acids** (previously limited). Includes charged (ARG, LYS, GLU, ASP) and aromatic residues with biophysically accurate probabilities.
- **Side-Chain Validation**: Implemented `validate_side_chain_rotamers()` in `PDBValidator`. It now checks if generated Chi1/Chi2 angles conform to the library distributions (with configurable tolerance).
- **Chirality Validation**: Added `validate_chirality()` to ensure L-amino acid stereochemistry (checking improper dihedrals).
- **Validation Integration**: Updated CLI (`main.py`) to run the full suite of validation checks (including rotamers and chirality) whenever `--validate`, `--best-of-N`, or `--guarantee-valid` is used.
- **Educational Notes**: Added extensive comments explaining Rotamer libraries, Staggered conformations, and Validation logic.

### Changed
- **CLI Robustness**: Refactored `main.py` validation calls to use `validator.validate_all()`, ensuring no checks are silently skipped in the future.
- **Tests**: Replaced incomplete mocks with robust TDD cases for rotamer violations.
- **Project Config**: Updated `pyproject.toml` to fix `setuptools` deprecation warnings (retaining backward compatibility).

### Fixed
- **Missing Validation**: Fixed an issue where `main.py` was selectively running only some validation checks, ignoring newly added ones.

## [1.9.0] - 2026-01-27

### Added
- **Feature**: Metal Ion Coordination (Zinc detection and injection).
- **Feature**: Disulfide Bond detection (SSBOND records).
- **Feature**: Salt Bridge stabilization in Energy Minimization.

## [1.8.0] - 2026-01-20

### Added
- **Feature**: NEF (NMR Exchange Format) IO support.
- **Feature**: Chemical Shift Prediction (SPARTA+ style logic).
