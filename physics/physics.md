# Physics Engine

`synth-pdb` uses **OpenMM** as its physics backend, providing GPU-accelerated energy minimization and molecular dynamics simulation. This page documents the physics capabilities and their CLI/Python interfaces.

> [!CAUTION]
> **OpenMM Version Warning**: Due to known upstream memory leaks in **OpenMM 8.5.0** and **8.5.1**, these versions are explicitly excluded from `synth-pdb` dependencies. We recommend using **OpenMM 8.4.x** or older for stable long-running generations.

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

## Advanced Force Fields: AMOEBA

While `synth-pdb` defaults to the **AMBER14** fixed-charge force field for speed and compatibility, OpenMM supports advanced polarizable force fields like **AMOEBA** (Atomic Multipole Optimized Energetics for Biomolecular Applications).

### What makes AMOEBA different?
1. **Polarizability:** Unlike fixed-charge models where atom charges are static, AMOEBA allows electronic distributions to respond to their environment via induced dipoles.
2. **Multipoles:** It uses explicit bond dipoles, quadrupoles, and octupoles instead of just point charges at atom centers.
3. **Accuracy:** It is significantly more accurate for water-protein interactions and transition metal coordination.

> [!WARNING]
> **Performance Trade-off:** AMOEBA simulations are typically **10x–100x slower** than AMBER. Supporting AMOEBA in `synth-pdb` would require a specialized pipeline and explicit solvent, which is currently on our long-term roadmap.

---

## Minimization Progress Reporting

Advanced users can monitor the energy minimization process in real-time. This is useful for identifying structural "explosions" or slow convergence in complex macrocycles.

### Enabling Real-Time Logs
To see the energy decay for every 50 iterations of the L-BFGS optimizer, set the log level to `DEBUG`:

```python
import logging
logging.getLogger("synth_pdb.physics").setLevel(logging.DEBUG)
```

**Log Output Format:**
`DEBUG:synth_pdb.physics:Physics: Minimization Iteration 150 | Energy: -4521.4302 kJ/mol`

---

## Hardware Acceleration (CUDA / Metal)

By default, `synth-pdb` automatically detects the fastest available hardware (CUDA > Metal > OpenCL > CPU). Power users can explicitly override this targeting for performance tuning or benchmarking.

### Explicit Platform Control
Use the `--platform` and `--precision` flags to control the hardware backend and numerical precision.

```bash
# Apple Silicon Mac: Force Metal platform with mixed precision
synth-pdb --sequence ALA-GLY-PRO --minimize --platform Metal --precision mixed

# NVIDIA GPU: Force CUDA platform
synth-pdb --sequence ALA-GLY-PRO --minimize --platform CUDA --precision mixed

# Force CPU (Reference) for high-precision debugging
synth-pdb --sequence ALA-GLY-PRO --minimize --platform CPU --precision double
```

**Supported Platforms:** `CUDA`, `Metal`, `OpenCL`, `CPU`, `Reference`
**Supported Precisions:** `single`, `mixed`, `double`

> [!TIP]
> **Mixed Precision:** Most molecular simulations use "mixed" precision, which uses single precision for calculations and double precision for accumulation. This provides a 2x-4x speedup on GPUs with negligible impact on accuracy.

---

### Educational Insight: The L-BFGS Algorithm
OpenMM's `minimizeEnergy` uses the **L-BFGS** algorithm. It is a quasi-Newton method that approximates the curvature of the energy landscape without calculating the full Hessian matrix, making it ideal for systems with thousands of atoms.

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
