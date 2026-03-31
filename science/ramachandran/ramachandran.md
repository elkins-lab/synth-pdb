# Ramachandran Plots

The Ramachandran plot is a fundamental tool for understanding and validating protein backbone conformations.

## Backbone Dihedral Angles

The protein backbone consists of a repeating sequence of N, Cα, and C atoms. The conformation of each residue is primarily determined by two dihedral (torsion) angles:

1.  **Phi (φ):** Rotation about the N–Cα bond.
2.  **Psi (ψ):** Rotation about the Cα–C bond.
3.  **Omega (ω):** Rotation about the C–N (peptide) bond. This is typically fixed near 180° (trans) or 0° (cis).

## Theoretical Basis

A Ramachandran plot (φ vs. ψ) displays the energetically allowed regions for backbone dihedral angles. Some combinations of φ and ψ are prohibited due to steric clashes between the carbonyl oxygen (O) and the amide hydrogen (H), or between the side-chain atoms and the backbone.

### Allowed Regions

-   **Alpha-Helix:** φ ≈ -57°, ψ ≈ -47°.
-   **Beta-Sheet:** φ ≈ -135°, ψ ≈ 135°.
-   **Polyproline II (PPII):** φ ≈ -75°, ψ ≈ 145°.

## Implementation in `synth-pdb`

`synth-pdb` uses Ramachandran principles in two ways:

1.  **Structure Generation:** When using the `--conformation` or `--structure` flags, the tool uses preset φ and ψ angles (defined in `synth_pdb/data.py`) to build the backbone.
2.  **Validation:** The `--validate` flag calculates the φ and ψ angles of the generated structure and checks if they fall within the allowed regions.

### Presets

| Conformation | Phi (°) | Psi (°) |
| :--- | :--- | :--- |
| `alpha` | -57.0 | -47.0 |
| `beta` | -135.0 | 135.0 |
| `ppii` | -75.0 | 145.0 |
| `extended` | -120.0 | 120.0 |

### Special Cases

-   **Glycine (GLY):** Lacks a side chain, so it has much more conformational freedom. Its Ramachandran plot is broader and more symmetric.
-   **Proline (PRO):** Its side chain is cyclized back to the backbone Nitrogen, making it the most restricted residue. It is often a "structure breaker."
-   **Cis-Proline:** Proline is unique in having a relatively high frequency (~5%) of the `cis` peptide bond (ω ≈ 0°), which `synth-pdb` can simulate via the `--cis-proline-frequency` flag.
