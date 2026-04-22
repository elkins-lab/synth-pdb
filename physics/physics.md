# Physics Engine

`synth-pdb` uses **OpenMM** as its physics backend, providing GPU-accelerated energy minimization and molecular dynamics simulation. This page documents the physics capabilities and their CLI/Python interfaces.

## Understanding Energy Minimization

**Energy Minimization** moves atoms "downhill" on the potential energy surface to find the nearest stable configuration.

```text
      High Energy
      (Unstable)
          |
         / \       Forces push atoms "downhill"
        /   \     (L-BFGS Gradient Descent)
       /     \
      /       \___
     /            \
    /              \__ Low Energy
   /                  (Stable / Minimized)
```

`synth-pdb` defaults to **Implicit Solvent (OBC2)** — simulating water's screening effect without the cost of thousands of explicit water molecules. This gives realistic results in 1–5 seconds on CPU.

```bash
synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha --minimize
```

---

## Explicit Solvent (Water Box)

For advanced MD, `synth-pdb` supports generating and simulating **Explicit Solvent**:

```bash
# Generate a peptide padded by 1.2 nm of TIP3P water
synth-pdb --sequence ALA-PRO-GLY --minimize \
  --solvent explicit \
  --solvent-padding 1.2
```

> [!TIP]
> **Stripping Solvent:**
> By default, `synth-pdb` strips the thousands of generated `HOH` (water) atoms from the final `.pdb` output to keep file sizes clean for downstream AI pipelines. Use `--keep-solvent` to export the full water box.

---

## MD Trajectory Simulation

Use `simulate_trajectory()` to run a short molecular dynamics trajectory and obtain per-frame coordinates for RMSF analysis:

```python
from synth_pdb import generate_structure
from synth_pdb.physics import simulate_trajectory
from synth_pdb.geometry.superposition import kabsch_superposition
import numpy as np

# 1. Generate a starting structure
pdb_text = generate_structure(sequence="LKELEKELEKELEKEL", conformation="alpha")

# 2. Run MD — returns a list of coordinate arrays, one per saved frame
frames = simulate_trajectory(
    pdb_content=pdb_text,
    n_steps=10_000,          # MD steps
    temperature=300,         # Kelvin
    save_interval=100,       # Save coordinates every N steps
)
# frames: List[np.ndarray], each shape (N_atoms, 3)

# 3. Compute Kabsch-aligned RMSF per residue
reference = frames[0]
aligned_frames = [kabsch_superposition(f, reference)[0] for f in frames[1:]]
coords_stack = np.stack(aligned_frames)          # (n_frames, N_atoms, 3)
rmsf = np.sqrt(np.mean(np.var(coords_stack, axis=0).sum(axis=-1)))
print(f"Mean RMSF: {rmsf:.2f} Å")
```

> [!NOTE]
> `simulate_trajectory()` requires OpenMM. Install with `pip install synth-pdb[physics]`.

---

## The Generation Pipeline

```text
[User] → [Generator] → [Geometry Builder] → [Sidechain Packer] → [Energy Minimizer] → [PDB File]
             ^                  |                    |                      |
             |              (N-CA-C-O)           (Rotamers)             (OpenMM)
             |                                       |                      |
             +---------------------------------------+----------------------+
```

## API Reference

::: synth_pdb.physics.simulate_trajectory
    options:
      show_root_heading: true
      show_source: false

## See Also

- [Science: Energy Minimization](science/energy-minimization.md)
- [Tutorial: AlphaFold Confidence vs NMR S²](tutorials/alphafold_vs_nmr_dynamics.ipynb)
- [API: geometry module](api/geometry.md) — Kabsch superposition for RMSF
