# `coupling` — J-Coupling Prediction

The `synth_pdb.coupling` module predicts **³J(HN–Hα) scalar coupling constants** for protein structures using the Karplus equation. These couplings are sensitive to the backbone φ torsion angle and are a key restraint in NMR structure validation.

## Scientific Background

### The Karplus Equation

The relationship between the ³J coupling and the dihedral angle is given by:

$$^3J(\theta) = A \cos^2\theta + B\cos\theta + C$$

where θ = φ − 60° for L-amino acids (H–N–Cα–H dihedral). The `synth_nmr` engine uses the **Vuister & Bax (1993)** parameterisation:

| Coefficient | Value |
|-------------|-------|
| A | 6.51 Hz |
| B | −1.76 Hz |
| C | 1.60 Hz |

### Secondary Structure Sensitivity

| Secondary Structure | φ (°) | ³J (Hz) |
|--------------------|-------|----------|
| α-helix | −60 | ~4.1 |
| β-strand | −135 | ~9.4 |
| PPII | −75 | ~6.1 |
| Extended | ±180 | ~4.1 |

### Proline Residues

Proline (PRO) and D-proline (DPR) form a secondary amine in the peptide bond and therefore **lack an amide proton**. The ³J(HN–Hα) coupling is physically undefined for these residues. `synth_pdb.coupling` automatically excludes them from the output.

### D-Amino Acids

For D-amino acids the stereochemical inversion means:

$$J_D(\phi) = J_L(-\phi)$$

This correction is applied automatically to all D-residue codes (DAL, DAR, … DVA).

## API Reference

::: synth_pdb.coupling
    handler: python
    options:
      members:
        - calculate_hn_ha_coupling
        - predict_couplings_from_structure

## Usage Examples

### Low-level: single φ angle

```python
from synth_pdb.coupling import calculate_hn_ha_coupling

j_helix = calculate_hn_ha_coupling(-60.0)   # → ~3.9 Hz
j_sheet = calculate_hn_ha_coupling(-135.0)  # → ~8.9 Hz

print(f"Helix: {j_helix:.2f} Hz,  Sheet: {j_sheet:.2f} Hz")
```

### Structure-level: all residues

```python
import io
import biotite.structure.io.pdb as pdb_io
from synth_pdb.generator import generate_pdb_content
from synth_pdb.coupling import predict_couplings_from_structure

pdb_content = generate_pdb_content(sequence_str="LKELEKELELK")
structure = pdb_io.PDBFile.read(io.StringIO(pdb_content)).get_structure(model=1)

couplings = predict_couplings_from_structure(structure)

# Iterate over all chains and residues
for chain_id, residues in couplings.items():
    for res_id, j_val in residues.items():
        print(f"Chain {chain_id}, Residue {res_id}: {j_val:.2f} Hz")
```

### Validation against experimental data

```python
import numpy as np
from synth_pdb.coupling import predict_couplings_from_structure

# Experimental 3J couplings for Ubiquitin (Vuister & Bax 1993, Table 1)
exp_j = {2: 5.5, 3: 7.2, 4: 6.8, 5: 5.1, 7: 8.3, 8: 4.9}

couplings = predict_couplings_from_structure(structure)
chain_A = couplings.get("A", {})

obs, pred = [], []
for res_id, exp_val in exp_j.items():
    if res_id in chain_A:
        obs.append(exp_val)
        pred.append(chain_A[res_id])

mae = float(np.mean(np.abs(np.array(obs) - np.array(pred))))
print(f"MAE vs experimental: {mae:.2f} Hz  (literature target < 2.0 Hz)")
```

## References

- Vuister, G.W. & Bax, A. (1993). Quantitative J correlation: a new approach for measuring homonuclear three-bond J(HN–Hα) coupling constants in ¹⁵N-enriched proteins. *J Am Chem Soc*, 115, 7772–7777. [DOI: 10.1021/ja00070a024](https://doi.org/10.1021/ja00070a024)

- Karplus, M. (1963). Vicinal proton coupling in nuclear magnetic resonance. *J Am Chem Soc*, 85, 2870–2871. [DOI: 10.1021/ja00901a059](https://doi.org/10.1021/ja00901a059)
