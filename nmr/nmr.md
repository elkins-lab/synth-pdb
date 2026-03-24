# For NMR Spectroscopists

## Note for Spectroscopists

If you are coming from an NMR background (XPLOR-NIH, CYANA, CNS):

- **Structure Calculation vs. Generation**: `synth-pdb` mimics the *final stage* of an NMR structure calculation: Geometry Regularization (minimization in implicit solvent).
- **Proton Detection**: Unlike X-ray, NMR relies on 1H spins. That's why we explicitly add hydrogens before minimization.

## Synthetic NMR Data

### NOE Restraints (`--gen-nef`)
Generates distance restraints based on geometric proximity.
- **Criteria**: H-H distance < cutoff (default 5.0 Å)
- **Format**: NMR Exchange Format (NEF)

### Chemical Shifts (`--gen-shifts`)
Generates predicted chemical shifts ($\delta$) using **SPARTA-lite**:
- **Helix**: $C_\alpha$ +3.1 ppm
- **Sheet**: $C_\alpha$ -1.5 ppm

### Relaxation Data (`--gen-relax`)
Simulates $R_1$, $R_2$, and NOE based on Model-Free formalism.
- **Order Parameter ($S^2$)**: Derived from local flexibility.
- **Tumbling Time ($\tau_c$)**: Configurable via `--tumbling-time`.
