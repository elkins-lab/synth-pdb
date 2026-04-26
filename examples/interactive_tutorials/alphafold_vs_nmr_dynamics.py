#!/usr/bin/env python
"""AlphaFold pLDDT vs. NMR S² — companion script.

Mirrors the notebook logic without py3Dmol (no interactive viewer in plain
Python). Run this as a smoke test for the full tutorial pipeline.
"""

import io
import os
import sys

import biotite.structure.io.pdb as bpdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# ── Repo path setup (local dev) ───────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "../../"))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Standard imports after path manipulation to satisfy E402
from synth_pdb.generator import generate_pdb_content  # noqa: E402
from synth_pdb.geometry.superposition import kabsch_superposition  # noqa: E402
from synth_pdb.physics import simulate_trajectory  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend for script execution

# ── Constants ─────────────────────────────────────────────────────────────────
SEQUENCE = "MEAAAQAAAAEAAAK" + "GSGSGSGSSGGSGSG" + "MAAAAQAAAAEAAAK"
STRUCTURE_DEF = "1-15:alpha, 16-30:random, 31-45:alpha"
HELIX1 = slice(0, 15)
LOOP = slice(15, 30)
HELIX2 = slice(30, 45)

# ── Step 1: Generate structure ────────────────────────────────────────────────
print("🧬 Generating initial structure and minimizing energy...")
initial_pdb = generate_pdb_content(
    sequence_str=SEQUENCE,
    structure=STRUCTURE_DEF,
    optimize_sidechains=True,
    minimize_energy=True,
)
print("✅ Baseline structure ready!")

# ── Step 2: MD simulation ─────────────────────────────────────────────────────
print("🔥 Running MD simulation (10–30 s on CPU)...")
trajectory_pdbs = simulate_trajectory(
    pdb_content=initial_pdb,
    temperature_kelvin=350.0,
    steps=5000,
    report_interval=50,
)
print(f"✅ Simulation complete! Captured {len(trajectory_pdbs)} trajectory frames.")

# ── Step 3: Kabsch-aligned RMSF ───────────────────────────────────────────────


def calculate_ca_rmsf_aligned(trajectory_list: list[str]) -> npt.NDArray[np.float64]:
    """Per-residue Cα RMSF with Kabsch alignment of every frame.

    Algorithm:
      1. Extract Cα coordinates → shape (F, N, 3).
      2. Compute coarse mean structure.
      3. Kabsch-align each frame to the mean.
      4. Recompute refined mean of aligned frames.
      5. RMSF = sqrt(mean squared deviation from aligned mean).
    """
    ca_coords_list = []
    for frame in trajectory_list:
        struct = bpdb.PDBFile.read(io.StringIO(frame)).get_structure(model=1)
        ca = struct[struct.atom_name == "CA"]
        ca_coords_list.append(ca.coord)

    ca_coords = np.array(ca_coords_list, dtype=np.float64)  # (F, N, 3)

    # Pass 1 — coarse mean
    mean_coords = ca_coords.mean(axis=0)

    # Kabsch-align every frame to the coarse mean
    aligned = np.empty_like(ca_coords)
    for i, frame_coords in enumerate(ca_coords):
        rot_matrix, translation = kabsch_superposition(frame_coords, mean_coords)
        aligned[i] = (rot_matrix @ frame_coords.T).T + translation

    # Pass 2 — refined mean
    aligned_mean = aligned.mean(axis=0)

    sq_dev = np.sum((aligned - aligned_mean) ** 2, axis=2)  # (F, N)
    rmsf: npt.NDArray[np.float64] = np.sqrt(sq_dev.mean(axis=0))  # (N,)
    return rmsf


rmsf_profile = calculate_ca_rmsf_aligned(trajectory_pdbs)
print(f"✅ RMSF calculation complete ({len(rmsf_profile)} residues).")
print(f"   Mean RMSF helix 1 : {rmsf_profile[HELIX1].mean():.3f} Å")
print(f"   Mean RMSF loop    : {rmsf_profile[LOOP].mean():.3f} Å")
print(f"   Mean RMSF helix 2 : {rmsf_profile[HELIX2].mean():.3f} Å")

# ── Step 4: S² from RMSF ─────────────────────────────────────────────────────
rmsf_max = rmsf_profile.max() + 1e-9
s2_profile = np.clip(1.0 - (rmsf_profile / rmsf_max) ** 2, 0.0, 1.0)

# Simulated AlphaFold pLDDT from known secondary structure
rng = np.random.default_rng(42)
plddt = np.empty(45)
plddt[HELIX1] = rng.normal(90, 4, 15)
plddt[LOOP] = rng.normal(45, 7, 15)
plddt[HELIX2] = rng.normal(90, 4, 15)
plddt = np.clip(plddt, 0, 100)

# ── Step 5: Plot ─────────────────────────────────────────────────────────────
residues = np.arange(1, 46)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.suptitle(
    "The Three-Way Correlation: MD Flexibility, NMR S², AlphaFold pLDDT",
    fontsize=13,
    fontweight="bold",
)

for ax in (ax1, ax2):
    ax.axvspan(1, 15, color="#3b82f6", alpha=0.10)
    ax.axvspan(16, 30, color="#f97316", alpha=0.10)
    ax.axvspan(31, 45, color="#22c55e", alpha=0.10)
    ax.set_xlim(1, 45)
    ax.grid(alpha=0.20)

ax1.plot(residues, s2_profile, "s-", color="#60a5fa", lw=2, ms=5, label="Synthetic S²")
ax1.axhline(0.85, color="#3b82f6", ls="--", lw=1, alpha=0.7, label="Helix benchmark (S²=0.85)")
ax1.axhline(0.45, color="#f97316", ls="--", lw=1, alpha=0.7, label="Loop benchmark (S²=0.45)")
ax1.set_ylabel("Order Parameter S²")
ax1.set_ylim(0, 1.15)
ax1.legend(fontsize=9, loc="lower right")

ax2.plot(residues, plddt, "o-", color="#f59e0b", lw=2, ms=5, label="Simulated pLDDT")
ax2.axhline(70, color="gray", ls=":", lw=1.2, alpha=0.8, label="pLDDT=70 threshold")
ax2.set_xlabel("Residue Number")
ax2.set_ylabel("pLDDT Score")
ax2.set_ylim(0, 110)
ax2.legend(fontsize=9, loc="lower right")

plt.tight_layout()
out_path = os.path.join(_here, "alphafold_vs_nmr_dynamics_output.png")
plt.savefig(out_path, dpi=150)
print(f"📊 Plot saved to {out_path}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n💡 CONCLUSION:")
print("=" * 60)
print(f"  Helix mean S²  : {s2_profile[HELIX1].mean():.2f} (rigid — high pLDDT expected)")
print(f"  Loop  mean S²  : {s2_profile[LOOP].mean():.2f}   (flexible — low pLDDT expected)")
print()
print("  Low pLDDT ≠ prediction failure.")
print("  AlphaFold correctly flags flexible regions as low-confidence.")
print("  Low pLDDT = low S² = high RMSF.")
