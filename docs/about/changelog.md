# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - `docs/api/rdc.md`: Full API reference for `synth_pdb.rdc` — Saupe-matrix RDC back-calculation, Q-factor interpretation table, alignment media comparison, and complete workflow example.
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
- **Docs site badges**: Replaced broken `Open in Colab` badge (pointed to non-existent `demo.ipynb`) with correct link to `../../examples/interactive_tutorials/virtual_nmr_spectrometer.ipynb`.
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
- **MSA Mutual Information Tutorial** (../../examples/interactive_tutorials/coevolution_msa_factory.ipynb): New interactive Jupyter notebook visually demonstrating how the physics-based MCMC sampler enforces co-evolution signals that recover the 3D contact map.
- **High-Density Physics Documentation Suite**: Massively expanded the inline educational docstrings across `synth_pdb/physics.py` (>60% line density), exposing the physical rationale behind NVT Langevin dynamics, Amber forcefields, implicit solvents, and metadata restoration strategies.
