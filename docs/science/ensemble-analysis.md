# NMR Ensemble Analysis

An NMR structure "ensemble" is not a single model — it is a **bundle of conformers**, typically 10–40 structures, each consistent with the experimental restraints (NOE distances, dihedral angles, RDCs). The spread of the bundle is the primary indicator of **precision**: tight bundles indicate well-determined, rigid regions; loose bundles indicate disorder or sparse restraint coverage.

This page explains the scientific foundations of the ensemble analysis tools in `synth_pdb.ensemble`.

---

## Why Ensembles?

X-ray crystallography produces a single, time-averaged structure. NMR is unique in that it inherently produces a **probability distribution** of structures. The ensemble:

1. Represents the genuine conformational heterogeneity of the molecule in solution
2. Encodes uncertainty — regions with few restraints have higher spread
3. Provides a direct view of internal dynamics on the µs–ms timescale

The two key quantities characterising an NMR ensemble are:

| Quantity | Unit | What it measures |
|----------|------|-----------------|
| **Pairwise RMSD** | Å | Coordinate precision — how similar are the models geometrically? |
| **DAOP** (S(φ)+S(ψ)) | dimensionless (0–2) | Dihedral precision — how consistent are the backbone angles? |

---

## Dihedral Angle Order Parameter (DAOP)

Introduced by Hyberts et al. (1992), the DAOP measures the **circular variance** of backbone dihedral angles across all ensemble members.

For a dihedral angle $\alpha$ measured in $N$ models:

$$S(\alpha) = \left| \frac{1}{N} \sum_{k=1}^{N} e^{i\alpha_k} \right|$$

This is the **length of the mean resultant vector** on the unit circle, ranging from 0 (completely disordered) to 1 (perfectly ordered).

$$S(\phi) + S(\psi) \begin{cases} \geq 1.8 & \text{well-defined residue} \\ < 1.8 & \text{disordered / poorly restrained} \end{cases}$$

The threshold of 1.8 is the **PDBStat convention** used by the BMRB deposition system (Tejero et al. 2013).

---

## Coordinate RMSD

The **pairwise RMSD** matrix (size $N \times N$ for an $N$-model ensemble) encodes all structural differences. Key derived quantities:

- **Mean pairwise RMSD**: overall precision of the ensemble
- **RMSF per residue**: standard deviation of each residue's position after superposition — directly comparable to B-factors and to MD RMSF
- **Medoid**: the model with the lowest mean RMSD to all other models — the most "representative" structure

All RMSD calculations in `synth_pdb.ensemble` use **Kabsch superposition** (via `synth_pdb.geometry.superposition`) to remove rigid-body translation and rotation before computing coordinate differences.

---

## Quality Thresholds

The following thresholds are taken from the PDBStat paper (Tejero et al. 2013) and are used by `QualityAssessment`:

| Metric | Excellent | Acceptable | Poor |
|--------|-----------|------------|------|
| Backbone RMSD | < 0.5 Å | 0.5–1.5 Å | > 1.5 Å |
| Heavy-atom RMSD | < 1.0 Å | 1.0–2.5 Å | > 2.5 Å |
| Fraction well-defined | > 85% | 60–85% | < 60% |

---

## IDP Ensembles

For **Intrinsically Disordered Proteins (IDPs)**, these thresholds do not apply — a "good" IDP ensemble should have **high RMSD and low DAOP** in the disordered regions by design. The spread reflects genuine structural disorder, not poor data.

Validation of IDP ensembles requires comparing ensemble-averaged observables (PRE rates, SAXS profiles, chemical shifts) to experiment rather than applying rigid RMSD thresholds.

See [Science: Intrinsically Disordered Proteins](idp-dynamics.md) and the [IDP Ensemble tutorial](../../examples/interactive_tutorials/idp_ensemble_validation.ipynb).

---

## References

1. **Hyberts, S. G. et al. (1992).** The solution structure of eglin c based on measurements of many NOEs and coupling constants. *Protein Science*, 1(6), 736–751.
2. **Tejero, R. et al. (2013).** PDBStat: a universal restraint converter and restraint quality analyzer for protein NMR structures. *J. Biomol. NMR*, 56(4), 337–351.
3. **Clore, G. M. & Iwahara, J. (2009).** Theory, practice and applications of paramagnetic relaxation enhancement for the characterization of transient low-population states of biological macromolecules. *Chem. Rev.*, 109(9), 4108–4139.

## See Also

- [API: ensemble module](../api/ensemble.md) — `DAOPCalculator`, `EnsembleStatistics`, `QualityAssessment`
- [Tutorial: IDP Conformational Ensembles](../../examples/interactive_tutorials/idp_ensemble_validation.ipynb)
- [API: geometry module](../api/geometry.md) — Kabsch superposition
