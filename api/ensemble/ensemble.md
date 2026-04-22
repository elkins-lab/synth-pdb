# ensemble Module

The `synth_pdb.ensemble` subpackage provides tools for **quantitative analysis of NMR structure bundles** — the multi-model ensembles produced by NMR structure calculation software (CYANA, XPLOR-NIH, CNS) that represent the conformational space consistent with experimental restraints.

```python
from synth_pdb.ensemble import DAOPCalculator, EnsembleStatistics
```

> [!NOTE]
> Added in **v1.28.0**. These tools mirror the analysis performed by **PDBStat** (Tejero et al. 2013) and are validated against the published NMR benchmarks in the `tests/test_scientific_validation.py` suite.

---

## Background

An NMR structure ensemble is not a single model but a **bundle of conformers** — typically 10–40 structures — each consistent with the experimental data (NOE distances, dihedral restraints, RDCs). The spread of the bundle is the key indicator of precision:

- **Tight bundle** → well-determined, rigid region
- **Loose bundle** → flexible, disordered, or poorly-restrained region

Two complementary metrics characterise ensemble quality:

| Metric | What it measures |
|--------|-----------------|
| **RMSD** | Pairwise coordinate deviation (Å) — precision of atomic positions |
| **DAOP** | Dihedral Angle Order Parameter — backbone angular consistency (0–1) |

---

## `DAOPCalculator`

::: synth_pdb.ensemble.daop.DAOPCalculator
    options:
      show_root_heading: true
      show_source: false
      members: [calculate, find_well_defined_residues]

### What is DAOP?

The **Dihedral Angle Order Parameter** (Hyberts et al. 1992) measures the circular variance of backbone dihedral angles (φ, ψ) across all models in an ensemble:

$$S(\alpha) = \left| \frac{1}{N} \sum_{k=1}^{N} e^{i\alpha_k} \right|$$

- **S = 1.0** → all models have identical dihedral angles (perfectly rigid)
- **S = 0.0** → angles are uniformly distributed (completely disordered)

### Well-Defined Residues (PDBStat Convention)

A residue is considered **well-defined** if it satisfies:

$$S(\phi) + S(\psi) \geq 1.8$$

This is the threshold used by PDBStat and the BMRB deposition system. Well-defined residues form the core of the folded structure and are used for global RMSD calculations.

### Usage

```python
from synth_pdb.ensemble import DAOPCalculator

# coords_list: List[np.ndarray] — one (N_atoms, 3) array per model
# residue_ids: List[int] — residue numbers matching the atom arrays
calc = DAOPCalculator(coords_list, residue_ids)

daop_per_residue = calc.calculate()
# Returns: Dict[int, Dict[str, float]]
# {residue_id: {"phi": S_phi, "psi": S_psi, "combined": S_phi + S_psi}}

well_defined = calc.find_well_defined_residues(threshold=1.8)
# Returns: List[int] — residue IDs where S(φ)+S(ψ) ≥ threshold
print(f"Well-defined residues: {well_defined}")
```

---

## `EnsembleStatistics`

::: synth_pdb.ensemble.statistics.EnsembleStatistics
    options:
      show_root_heading: true
      show_source: false

### Usage

```python
from synth_pdb.ensemble import EnsembleStatistics

stats = EnsembleStatistics(coords_list, residue_ids=residue_ids)

# Pairwise RMSD matrix (Å)
rmsd_matrix = stats.pairwise_rmsd          # np.ndarray (N_models × N_models)
print(f"Mean pairwise RMSD: {rmsd_matrix.mean():.2f} Å")

# RMSF per residue (Å) — after Kabsch alignment
rmsf = stats.rmsf                          # np.ndarray (N_residues,)

# Representative structure (medoid = lowest mean RMSD to all others)
medoid_idx = stats.medoid_index
print(f"Representative model: #{medoid_idx + 1}")

# Well-defined residues
print(f"Well-defined: {stats.well_defined_residues}")

# Overall quality assessment
qa = stats.quality_assessment              # QualityAssessment dataclass
print(f"Quality: {qa.overall_quality}")
```

---

## `QualityAssessment`

::: synth_pdb.ensemble.statistics.QualityAssessment
    options:
      show_root_heading: true
      show_source: false

### Quality Thresholds (Tejero et al. 2013)

| Field | Excellent | Acceptable | Poor |
|-------|-----------|------------|------|
| `mean_pairwise_rmsd_backbone` | < 0.5 Å | 0.5–1.5 Å | > 1.5 Å |
| `mean_pairwise_rmsd_heavy` | < 1.0 Å | 1.0–2.5 Å | > 2.5 Å |
| `fraction_well_defined` | > 0.85 | 0.60–0.85 | < 0.60 |
| `overall_quality` | `"excellent"` | `"acceptable"` | `"poor"` |

---

## Complete Example: Evaluating a Synthetic NMR Ensemble

```python
import numpy as np
from synth_pdb import generate_structure
from synth_pdb.ensemble import DAOPCalculator, EnsembleStatistics
from synth_pdb.geometry.superposition import find_medoid

# --- 1. Generate a 20-model synthetic ensemble ---
sequence = "LKELEKELEKELEKELEKELEKEL"
models = []
for _ in range(20):
    pdb_text = generate_structure(sequence=sequence, conformation="alpha")
    # Parse backbone Cα coordinates (simplified — use biotite in practice)
    # coords shape: (N_residues, 3)
    coords = parse_ca_coords(pdb_text)   # your parser here
    models.append(coords)

residue_ids = list(range(1, len(models[0]) + 1))

# --- 2. Dihedral Angle Order Parameters ---
daop = DAOPCalculator(models, residue_ids)
daop_scores = daop.calculate()
well_def = daop.find_well_defined_residues(threshold=1.8)
print(f"Well-defined residues: {well_def}")

# --- 3. Full ensemble statistics ---
stats = EnsembleStatistics(models, residue_ids=residue_ids)
print(f"Backbone RMSD:  {stats.quality_assessment.mean_pairwise_rmsd_backbone:.2f} Å")
print(f"Heavy-atom RMSD: {stats.quality_assessment.mean_pairwise_rmsd_heavy:.2f} Å")
print(f"Overall quality: {stats.quality_assessment.overall_quality}")
print(f"Representative model: #{stats.medoid_index + 1}")
```

---

## References

1. **Hyberts, S. G. et al. (1992).** The solution structure of eglin c based on measurements of many NOEs and coupling constants. *Protein Science*, 1(6), 736–751. *(Introduces DAOP)*
2. **Tejero, R. et al. (2013).** PDBStat: a universal restraint converter and restraint quality analyzer for protein NMR structures. *J. Biomol. NMR*, 56(4), 337–351. *(Defines quality thresholds)*

## See Also

- [Tutorial: IDP Conformational Ensembles](../tutorials/idp_ensemble_validation.ipynb)
- [API: geometry module](geometry.md) — Kabsch superposition used internally
- [Glossary: DAOP, RMSF, S²](https://github.com/elkins/synth-pdb#glossary-of-scientific-terms--acronyms)
