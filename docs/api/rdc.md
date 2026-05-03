# rdc Module

The `synth_pdb.rdc` module provides tools for computing and validating **Residual Dipolar Couplings (RDCs)** — a powerful class of NMR observables that encode the orientation of bond vectors relative to a global molecular alignment frame.

> [!NOTE]
> Core RDC back-calculation is delegated to the `synth-nmr` package (`from synth_nmr.rdc import calculate_rdcs`). This module re-exports that engine and adds structural validation tools (Q-factor, file I/O) that live within the `synth-pdb` ecosystem.

## Background

In solution NMR, rapid isotropic tumbling averages magnetic dipole–dipole interactions to zero. When a protein is placed in an **anisotropic alignment medium** (liquid crystals, phage particles, strained gels), a small **residual** coupling survives. For a backbone N–H bond vector, this coupling is:

$$D(\theta, \phi) = D_a \left[ (3\cos^2\theta - 1) + \frac{3}{2} R \sin^2\theta \cos(2\phi) \right]$$

where:

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Polar angle of the N–H vector relative to the principal alignment axis (Z) |
| $\phi$ | Azimuthal angle in the XY plane of the tensor |
| $D_a$ | Axial component of the alignment tensor (Hz); typical protein values: 5–25 Hz |
| $R$ | Rhombicity, $0 \leq R \leq 2/3$; $R=0$ = axially symmetric, $R=2/3$ = maximum |

**Why RDCs matter**: Unlike NOEs (purely local distances), RDCs encode **global fold topology** — the orientation of each bond vector in a shared molecular frame. Combined with NOEs, they dramatically improve the accuracy of NMR structure determination.

See [Science: NMR Theory](../science/nmr-theory.md) and the [RDC Alignment Explorer tutorial](../../examples/interactive_tutorials/rdc_alignment_explorer.ipynb) for interactive demonstrations.

---

## API Reference

### `calculate_rdcs`

Back-calculates RDC values from a 3D structure given an alignment tensor. Re-exported from `synth-nmr`.

```python
from synth_pdb.rdc import calculate_rdcs

rdcs = calculate_rdcs(
    structure,          # biotite AtomArray
    da=15.0,            # Axial component (Hz)
    r=0.1,              # Rhombicity (dimensionless, 0 ≤ R ≤ 2/3)
    euler_angles=(0.0, 0.0, 0.0),  # Rotation into alignment tensor PAS (radians)
)
# Returns: List[Dict] with keys: res_id, atom_1, atom_2, rdc_calc (Hz)
```

---

### `calculate_rdc_q_factor`

::: synth_pdb.rdc.calculate_rdc_q_factor
    options:
      show_root_heading: true
      show_source: false

#### Q-factor Interpretation

| Q-factor | Interpretation |
|----------|---------------|
| `< 0.20` | Excellent agreement — high-quality NMR structure |
| `0.20–0.35` | Good agreement — minor local geometry errors or dynamics |
| `0.35–0.50` | Moderate — investigate alignment tensor or dynamics |
| `> 0.50` | Poor — likely misfolding, wrong assignments, or tensor mismatch |

```python
import numpy as np
from synth_pdb.rdc import calculate_rdcs, calculate_rdc_q_factor

# Simulate RDCs from a synthetic structure
rdcs_calc = calculate_rdcs(structure, da=15.0, r=0.1)
d_calc = np.array([r["rdc_calc"] for r in rdcs_calc])

# Load experimental RDCs (e.g. from a .rdc file)
from synth_pdb.rdc import read_rdc_file
rdcs_obs = read_rdc_file("ubiquitin_nh.rdc")
d_obs = np.array([r["value"] for r in rdcs_obs])

q = calculate_rdc_q_factor(d_obs, d_calc)
print(f"Q-factor: {q:.3f}")  # e.g. Q-factor: 0.142
```

---

### `read_rdc_file`

::: synth_pdb.rdc.read_rdc_file
    options:
      show_root_heading: true
      show_source: false

#### Expected File Format

```text
# N-H RDCs for Ubiquitin in Pf1 phage (Hz)
# Res1  Atom1  Res2  Atom2  Value
1       N      1     HN     -12.4
2       N      2     HN       5.2
3       N      3     HN      18.7
```

Lines starting with `#` are treated as comments. Five whitespace-separated columns are required: `res1 atom1 res2 atom2 value_hz`.

---

## Complete Workflow Example

```python
import numpy as np
from synth_pdb import generate_structure
from synth_pdb.rdc import calculate_rdcs, calculate_rdc_q_factor, read_rdc_file

# 1. Generate a synthetic alpha-helical structure
pdb_text = generate_structure(sequence="LKELEKELEKELEKELEKELEKEL", conformation="alpha")

# 2. Parse into AtomArray (requires biotite)
import biotite.structure.io.pdb as pdbio, io
f = pdbio.PDBFile.read(io.StringIO(pdb_text))
structure = pdbio.get_structure(f, model=1)

# 3. Back-calculate backbone N-H RDCs
#    Da = 15 Hz, R = 0.06 (axially symmetric bicelle alignment)
rdcs_calc = calculate_rdcs(structure, da=15.0, r=0.06)

# 4. Validate against experimental data
rdcs_obs  = read_rdc_file("experimental.rdc")
d_obs  = np.array([r["value"]    for r in rdcs_obs])
d_calc = np.array([r["rdc_calc"] for r in rdcs_calc])

q = calculate_rdc_q_factor(d_obs, d_calc)
print(f"Q = {q:.3f}  ({'PASS' if q < 0.25 else 'REVIEW'})")
```

---

## The Alignment Tensor (Saupe Matrix)

The alignment is fully described by the **Saupe matrix** — a traceless, symmetric 3×3 tensor parameterised by two scalars:

- **`Da`** (axial component) — controls the overall magnitude of the RDCs. Larger `|Da|` → larger couplings.
- **`R`** (rhombicity) — controls the asymmetry. `R = 0` gives an axially symmetric tensor; `R = 2/3` is maximum rhombicity.

Use the interactive [RDC Alignment Explorer](../../examples/interactive_tutorials/rdc_alignment_explorer.ipynb) tutorial to visually understand how `Da` and `R` affect the RDC pattern across a helix or sheet.

---

## Alignment Media

| Medium | Alignment mechanism | Best for |
|--------|--------------------|---------:|
| Bicelles (DMPC/DHPC) | Steric | General proteins |
| Pf1 phage | Electrostatic | High-pI proteins |
| Polyacrylamide gel | Mechanical compression | Any pH/salt |
| Liquid crystals (C12E5) | Steric + electrostatic | Small proteins |

---

## References

1. **Tjandra, N. & Bax, A. (1997).** Direct measurement of distances and angles in biomolecules by NMR in a dilute liquid crystalline medium. *Science*, 278, 1111–1114.
2. **Cornilescu, G. et al. (1998).** Validation of protein structures derived from NMR data using the Q-factor. *J. Am. Chem. Soc.*, 120, 6836–6837.
3. **Prestegard, J.H. et al. (2000).** NMR structures using field-oriented media and residual dipolar couplings. *Q. Rev. Biophys.*, 33, 371–424.

## See Also

- [Science: NMR Theory](../science/nmr-theory.md)
- [Tutorial: RDC Alignment Explorer](../../examples/interactive_tutorials/rdc_alignment_explorer.ipynb)
- [Tutorial: Ubiquitin RDC Validation](../../examples/interactive_tutorials/ubiquitin_rdc_validation.ipynb)
- [API: nmr module](nmr.md)
