# NMR Observables

`synth-pdb` generates a complete suite of synthetic NMR observables for testing assignment software, structure calculation pipelines, and AI/ML models — all physically grounded in the real NMR experiment.

> [!NOTE]
> As of v1.16.0, detailed NMR prediction is implemented in the separate [`synth-nmr`](https://github.com/elkins/synth-nmr) package. `synth-pdb` provides the CLI flags and Python shims described here, delegating computation to `synth-nmr ≥ 0.9.0`.

---

## NOE Restraints (`--gen-nef`)

Generates synthetic distance restraints based on inter-proton geometry.

**Physics**: The NOE intensity scales as $r^{-6}$, so only proton pairs closer than a cutoff (default 5.0 Å) produce observable cross-peaks.

```bash
synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha --gen-nef
# Output: my_structure_restraints.nef
```

Output is in **NMR Exchange Format (NEF)**, importable directly into CCPNMR Analysis V3, CYANA, and CS-Rosetta.

---

## Chemical Shifts (`--gen-shifts`)

Generates predicted backbone and sidechain chemical shifts ($\delta$) using a **SPARTA-lite** implementation:

1. **Base value**: Random coil shifts (Wishart et al. 1995)
2. **Secondary structure offset**: Based on local φ/ψ angles
3. **Ring-current corrections**: From aromatic residues (Haigh & Mallion 1979)

```bash
synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha --gen-shifts
# Output: my_structure_shifts.nef
```

Secondary structure signatures:

| Nucleus | α-Helix offset | β-Sheet offset |
|---------|--------------|---------------|
| $C_\alpha$ | +3.1 ppm | −1.5 ppm |
| $C_\beta$ | −0.5 ppm | +1.2 ppm |
| $H^\alpha$ | −0.4 ppm | +0.8 ppm |
| $^{15}$N | −1.5 ppm | +1.2 ppm |

Use the **Chemical Shift Index (CSI)** analysis to verify secondary structure is correctly encoded in the generated shifts — exactly as you would for a real BMRB entry.

---

## Relaxation Data (`--gen-relax`)

Simulates $R_1$, $R_2$, and heteronuclear NOE based on the **Lipari-Szabo model-free formalism**:

$$R_1 = \frac{d^2}{4} \left[ S^2 J(\omega_H - \omega_N) + \ldots \right]$$

$$S^2 = 1 - \frac{B\text{-factor}}{B_{\max}}$$

```bash
synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha \
  --gen-relax --tumbling-time 5.0
# Output: my_structure_relaxation.csv
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--tumbling-time` | auto | Global correlation time τ_c (ns) |
| `--field-strength` | 600 | ¹H Larmor frequency (MHz) |

---

## RPF Scores (`synth_pdb.nmr.calculate_rpf_score`)

The **RPF score** (Huang et al. 2005) evaluates how well a structural model explains a set of NOE-derived distance restraints:

| Metric | Measures |
|--------|---------|
| **R** (Recall) | Fraction of observed NOEs explained by the structure |
| **P** (Precision) | Fraction of structure-predicted contacts with NOE evidence |
| **F** | Harmonic mean of R and P (overall goodness-of-fit) |
| **DP** | Discriminant Power — separates correct from incorrect folds |

```python
from synth_pdb.nmr import calculate_rpf_score

score = calculate_rpf_score(
    pdb_content=pdb_text,
    noe_file="restraints.nef",
    distance_cutoff=5.0,
)
print(f"R={score.recall:.3f}  P={score.precision:.3f}  F={score.f_score:.3f}  DP={score.dp:.3f}")
```

**Interpretation** (BMRB thresholds):

| Score | Meaning |
|-------|---------|
| F > 0.70 | Good structural model |
| F 0.50–0.70 | Moderate — review restraint completeness |
| F < 0.50 | Poor — significant structural errors |

See the [RPF Validation tutorial](tutorials/nmr_validation_rpf.ipynb) for an interactive demonstration.

---

## Residual Dipolar Couplings (RDC)

See the dedicated **[RDC module documentation](api/rdc.md)** and the [RDC Alignment Explorer tutorial](tutorials/rdc_alignment_explorer.ipynb).

---

## J-Couplings (`--gen-couplings`)

Predicts scalar $^3J_\text{HN-HA}$ couplings via the **Karplus equation**:

$$^3J(\theta) = A\cos^2\theta + B\cos\theta + C$$

```bash
synth-pdb --sequence "LKELEKELEKELEKEL" --gen-couplings
# Output: my_structure_couplings.csv
```

| Secondary structure | $^3J_\text{HN-HA}$ |
|--------------------|-------------------|
| α-Helix | < 6 Hz |
| β-Sheet | > 8 Hz |
| Random coil | ~7 Hz |

---

## See Also

- [API: nmr module](api/nmr.md) — `calculate_rpf_score`, `read_restraint_file`
- [API: rdc module](api/rdc.md) — RDC back-calculation and Q-factor
- [API: relaxation module](api/relaxation.md) — $R_1$, $R_2$, NOE equations
- [Tutorial: Virtual NMR Spectrometer](tutorials/virtual_nmr_spectrometer.ipynb)
- [Guide: For NMR Spectroscopists](guides/nmr-spectroscopists.md)
