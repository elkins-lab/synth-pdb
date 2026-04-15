# synth-nmr Integration Audit

## Current State

Seven `synth_pdb/` modules are **pure re-export shims** — they exist solely for backward
compatibility and contain no logic of their own:

| synth-pdb shim | Delegates to |
|---|---|
| `chemical_shifts.py` | `synth_nmr.chemical_shifts` |
| `coupling.py` | `synth_nmr.coupling` |
| `j_coupling.py` | `synth_nmr` (top-level) |
| `nef_io.py` | `synth_nmr.nef_io` |
| `nmr.py` | `synth_nmr` (top-level) |
| `relaxation.py` | `synth_nmr.relaxation` |
| `structure_utils.py` | `synth_nmr.structure_utils` |

---

## 1. New synth-nmr Features Not Yet Leveraged

### A. RDC Calculations (`synth_nmr.rdc`)
`calculate_rdcs(structure, Da, R)` computes backbone N-H Residual Dipolar Couplings.

**Opportunity:** Add a `synth_pdb/rdc.py` shim (mirrors the existing pattern exactly) and wire
it into `main.py` alongside chemical shifts and NOEs. RDCs are a standard NMR output that users
would naturally expect from a "generate PDB + NMR data" tool.

**Effort:** ~30 lines — one shim + one `--output-rdcs` flag in `main.py`.

---

### B. MD Trajectory / Ensemble NMR (`synth_nmr.trajectory`)
`TrajectoryEnsemble`, `load_trajectory()`, `ensemble_average_shifts()`,
`ensemble_average_noes()`, `ensemble_average_rdcs()`, `compute_s2_from_trajectory()`.

**Opportunity:** synth-pdb generates single-conformation PDB files. The trajectory module
bridges to MD ensembles. Two possible integrations:

1. **A `--ensemble N` flag** in `main.py`: generate N structures (already possible via
   `generate_pdb_content()` in a loop), wrap them as a `TrajectoryEnsemble`, and output
   ensemble-averaged NMR observables — more physically realistic than single-structure values.
2. **A new `synth_pdb/trajectory.py` shim** to expose the ensemble API to downstream
   synth-pdb users who want to load their own MD data alongside synthetic structures.

**Effort:** Medium — the ensemble loop logic is straightforward; the main work is plumbing
the CLI flag and output format.

---

### C. SHIFTX2 / Empirical Shift Predictor (`synth_nmr.chemical_shifts`)
synth-nmr's `predict_chemical_shifts()` now **prefers SHIFTX2** when available and falls
back to SPARTA+. synth-pdb's `main.py` calls `predict_chemical_shifts()` → it already
benefits automatically. However:

- synth-pdb has no way to expose `predict_empirical_shifts()` (the SPARTA+ path directly)
- There's no CLI flag to select the predictor
- The SHIFTX2 binary path is not configurable from the synth-pdb CLI

**Opportunity:** Thread `--shift-predictor [shiftx2|empirical]` through to the
`predict_chemical_shifts(use_shiftx2=...)` call in `main.py`.

**Effort:** Small — one flag, one kwarg pass-through.

---

### D. Neural Shift Predictor (`synth_nmr.neural_shifts`)
A PyTorch MLP+GNN that predicts chemical shift *residuals* on top of the empirical baseline.
Requires `torch` (optional). Currently completely disconnected from synth-pdb.

**Opportunity:** Expose via a `synth_pdb/neural_shifts.py` shim and an optional flag
`--shift-predictor neural`. The neural predictor could be a differentiable path for users
integrating with ML pipelines (e.g., the GFP Forge tutorial).

**Effort:** Small shim; the main challenge is documentation around optional `torch` dependency.

---

### E. BMRB Data Pipeline (`synth_nmr.data_pipeline`)
Downloads and parses experimental BMRB shift data alongside real PDB structures.

**Opportunity:** This is already a synth-nmr internal — no shim needed. But synth-pdb's
`dataset.py` (which generates ML training datasets) could optionally produce
`(synthetic_structure, experimental_shifts)` pairs by calling `load_matched_dataset()`,
giving users a ready-made heterogeneous training set.

**Effort:** Medium — adding a `--include-experimental` flag to dataset generation.

---

## 2. Duplicate / Redundant Code

### A. `synth_pdb/j_coupling.py` duplicates `synth_pdb/coupling.py`
Both re-export `calculate_hn_ha_coupling` from `synth_nmr`. `j_coupling.py` is a strict
subset of `coupling.py`. The shim was needed historically before `coupling.py` existed.

**Recommendation:** Deprecate `j_coupling.py`. Add a deprecation warning that redirects to
`synth_pdb.coupling`.

---

### B. `synth_pdb/plm.py` duplicates the pattern from `synth_nmr.neural_shifts`
`plm.py` implements its own lazy-import ESM-2 embedder with identical guard patterns to
`neural_shifts.py`. The two don't overlap in functionality (PLM embeddings vs. shift
prediction) but share the same design pattern and could share a common `_lazy_torch_import()`
utility.

**Recommendation:** Extract the lazy torch guard into a small shared helper in
`synth_pdb/_torch_utils.py`. Low priority.

---

## 3. Priority Summary

| Item | Impact | Effort | Priority |
|---|---|---|---|
| Add `rdc.py` shim + CLI flag | High — standard NMR output | Low | ⭐⭐⭐ High |
| SHIFTX2 selector CLI flag | Medium | Low | ⭐⭐⭐ High |
| Deprecate `j_coupling.py` | Low risk cleanup | Trivial | ⭐⭐ Medium |
| Ensemble NMR (`--ensemble N`) | High — unique feature | Medium | ⭐⭐ Medium |
| Neural shift predictor shim | Niche, ML-focused | Low | ⭐ Low |
| BMRB dataset integration | Research-facing | Medium | ⭐ Low |
| Shared `_lazy_torch_import` | Code cleanliness | Low | ⭐ Low |

---

## 4. Future RPF and Validation Enhancements (Proposed)

### A. CLI Integration for RPF Scores
**Opportunity:** Automatically report RPF scores in `main.py` when a user provides a restraint file (e.g., `.nbl` or `.tbl`). This would provide immediate feedback on structural agreement with experimental data.
**Effort:** Small — add a `--restraints` flag and call `calculate_rpf_score`.

### B. Structural Validator Integration
**Opportunity:** Add a `validate_nmr_restraints(restraints)` method to the `PDBValidator` class in `synth_pdb/validator.py`.
**Value:** Allows "Restraint Satisfaction" to be a pass/fail criterion in automated decoy-generation and structural biology pipelines.

### C. Quality Classifier Features
**Opportunity:** Include the **F-measure** as a feature in `extract_quality_features` (in `synth_pdb/quality/features.py`).
**Value:** A model that satisfies its NMR restraints is statistically much more likely to be a "correct" fold, improving the quality filter's accuracy.

### D. Ensemble-Average RPF
**Opportunity:** Extend `GeometryAnalyzer` in `synth_pdb/analysis.py` to calculate an **Ensemble-RPF**, reporting mean and variance of the F-measure across NMR ensembles.
**Value:** Identifies which members of an ensemble are most representative of the data.
