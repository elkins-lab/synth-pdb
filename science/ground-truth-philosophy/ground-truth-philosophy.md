# Ground Truth Philosophy: Geometry vs. Physics

This document outlines the philosophical and scientific rationale behind how `synth-pdb` defines "Ground Truth" for protein structures, particularly in the context of training and benchmarking AI models.

## The Central Tension: Idealism vs. Realism

In structural biology, there is a fundamental tension between two types of "Truth":

1.  **Geometric Truth (The Ideal)**: A structure defined by precise mathematical internal coordinates (torsions, bond lengths, angles).
2.  **Physical Truth (The Reality)**: A structure that has reached an energy minimum within a specific forcefield and solvent environment.

`synth-pdb` prioritizes **Geometric Truth** as the primary "Ground Truth" label.

## Why Non-Minimized Structures are "Truth"

For a structural biologist, a non-minimized structure with steric clashes looks "wrong." However, for a Machine Learning engineer, this structure is the most "honest" label possible.

### 1. Mathematical Determinism
The `synth-pdb` generator is a deterministic mathematical engine. When you request an alpha helix, it places atoms exactly at the Ramachandran centers (e.g., $\phi = -60^\circ, \psi = -45^\circ$). 
*   **The Signal**: The "Ground Truth" coordinates represent the **intent** of the generation.
*   **The Noise**: Steric clashes are "noise" resulting from the high-dimensional complexity of protein packing.

If we minimize the structure before calling it "truth," we allow a physics engine (like OpenMM) to rewrite our intent. We are no longer testing if an AI can learn to build an alpha helix; we are testing if it can learn the specific preferences of the **Amber14 forcefield**.

### 2. Forcefield Agnosticism
Physics engines are models, not reality. Every forcefield (Amber, CHARMM, OPLS) has its own biases. 
*   If we define ground truth as the output of an Amber14 minimization, our dataset is "biased" toward Amber14. 
*   An AI trained on this data might perform poorly on experimental data that doesn't perfectly match Amber's specific energetic curves.

By using the **NeRF-generated ideal geometry** as ground truth, we provide a stable, forcefield-independent reference point.

### 3. Measuring "Conformational Strain"
By preserving the non-minimized state as the reference, we can calculate a critical metric: **Strain-RMSD**.
*   **Formula**: $RMSD(GroundTruth, Minimized)$
*   **Interpretation**: If the RMSD is low (< 0.5 Å), the "ideal" geometric fold was physically plausible. If the RMSD is high (> 2.0 Å), the intended fold was physically impossible (a "clash-trap").

## The "Best of N" Strategy: A Bridge to Plausibility

To satisfy both the Geometer and the Physicist, `synth-pdb` recommends a **Best of N** selection strategy rather than a **Minimize-to-Relax** strategy.

### The Algorithm
Instead of taking one structure and moving its atoms to resolve clashes, we:
1.  Generate $N$ independent versions of the structure using small stochastic "drifts" in torsion angles.
2.  Calculate the number of steric violations for each.
3.  **Select the one with zero or minimum violations.**

### Why this is superior for AI
-   **No Warp**: The selected structure still uses "ideal" bond lengths and angles. No atoms were "pushed" by a forcefield.
-   **Physically Valid**: The structure is clash-free and exists in a low-energy state.
-   **Clean Labels**: The AI learns to associate a sequence with a physically plausible *but still mathematically ideal* conformation.

## Practical Recommendations for ML Research

| Use Case | Recommended Ground Truth | Rationale |
| :--- | :--- | :--- |
| **Pre-training** | **Non-minimized** | Maximizes signal/noise ratio for geometric patterns. |
| **Fine-tuning** | **Best of N (Selected)** | Introduces physical plausibility without forcefield bias. |
| **Physical Audit** | **Minimized** | Use as a secondary check to see if the AI's "ideal" prediction is physically stable. |

## Scientific Conclusion

In `synth-pdb`, the **Ground Truth** is the **Source of Intent**. We treat the structure as a "Label" that an AI should strive to recover. While minimization is a powerful tool for refinement, it should be treated as an **analytical step** rather than a **definition of truth**.

---

## See Also
- [NeRF Geometry](nerf-geometry.md) - The math behind the "Ideal" structure.
- [Energy Minimization](energy-minimization.md) - How to relax structures using OpenMM.
- [Validation Suite](../api/validator.md) - How we measure steric violations.
