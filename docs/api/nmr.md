# NMR

API reference for the `synth_pdb` NMR-data modules.

## `synth_pdb.rdc` — Residual Dipolar Couplings

### Overview

Re-exports `calculate_rdcs` from [`synth-nmr`](https://github.com/elkins/synth-nmr).
RDCs encode the orientation of backbone N–H bond vectors relative to a global alignment
frame and are therefore complementary to local-distance NOE restraints.

**Physics behind RDCs** — see [NMR Theory: Residual Dipolar Couplings](../science/nmr-theory.md#residual-dipolar-couplings).

### `calculate_rdcs(structure, Da, R)`

```python
from synth_pdb.rdc import calculate_rdcs
rdcs = calculate_rdcs(structure, Da=10.0, R=0.1)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `biotite.AtomArray` | — | Protein structure containing backbone N and H atoms |
| `Da` | `float` | `10.0` | Axial component of the alignment tensor in Hz (typical: 5–25 Hz) |
| `R` | `float` | `0.1` | Rhombicity of the alignment tensor, $0 \leq R \leq 2/3$ |

**Returns:** `dict[int, float]` — mapping `res_id → RDC (Hz)`. Proline residues and any
residue missing a backbone amide H are automatically excluded.

### Example

```python
import biotite.structure.io.pdb as pdbio
from synth_pdb.rdc import calculate_rdcs

structure = pdbio.get_structure(pdbio.PDBFile.read("my_peptide.pdb"), model=1)
rdcs = calculate_rdcs(structure, Da=15.0, R=0.2)

for res_id, rdc_val in sorted(rdcs.items()):
    print(f"Res {res_id}: {rdc_val:+.2f} Hz")
```

---

## `synth_pdb.chemical_shifts` — Chemical Shift Prediction

Re-exports `predict_chemical_shifts` and `predict_empirical_shifts` from
[`synth-nmr`](https://github.com/elkins/synth-nmr).

### `predict_chemical_shifts(structure, use_shiftx2=True)`

Predicts backbone and CB chemical shifts. Selects the predictor backend via
`use_shiftx2`:

| `use_shiftx2` | Backend | Notes |
|---|---|---|
| `True` (default) | SHIFTX2 (Han et al., 2011) | Requires the external binary; falls back to empirical if missing |
| `False` | SPARTA+-style empirical | Always available; no external dependencies |

---

## `synth_pdb.coupling` — J-Couplings

Re-exports `calculate_hn_ha_coupling` and `predict_couplings_from_structure`
(Karplus equation for ${}^3J(\text{HN,HA})$ couplings).
