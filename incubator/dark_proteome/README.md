# The Dark Proteome & Pathological Models

This directory contains examples of "Dark Proteome" proteins—intrinsically disordered regions (IDRs), orphans, and highly flexible ensembles—as well as a collection of "Pathological Models" designed to test the limits of structural biology software, visualization tools, and physics engines.

## 💀 The Hall of Pathological Perversions

The following models represent intentional violations of biophysical laws, geometric constraints, and chemical sanity:

1.  **`mobius_loop_pathological.pdb`**: A cyclic peptide generated with high internal strain to simulate a Moebius-like topology, forcing the backbone into a self-intersecting circular path.
2.  **`trp_singularity.pdb`**: A 20-residue poly-tryptophan chain built with zero clash refinement. It features a physically impossible density of aromatic rings packed into the same space.
3.  **`mirror_nightmare.pdb`**: The "Anti-Protein." Composed entirely of D-amino acids, it represents a complete chiral inversion of a standard peptide, creating a "mirror image" world of structural biology.
4.  **`chiral_chimera.pdb`**: A sequence of alternating L and D amino acids. This destroys the standard regularity of the protein backbone, making it impossible to form stable helices or sheets.
5.  **`amyloid_nightmare.pdb`**: A multi-chain stack of beta-sheets where the inter-chain distances have been reduced to values that would lead to immediate atomic fusion in a real system.
6.  **`steric_singularity.pdb`**: A 50-residue chain pathologically compressed into a 5Å sphere. It violates every known Van der Waals radius and creates a coordinate density "black hole."
7.  **`fractal_spaghetti.pdb`**: A 200-residue random walk that uses discontinuous fractal-like coordinates, causing the chain to "teleport" across space every 10 residues.
8.  **`entropic_abyss_ensemble.pdb`**: A 10-model ensemble where each model is a completely different, maximally high-entropy random state for the same sequence, representing total conformational chaos.
9.  **`infinity_coil.pdb`**: A 100-residue minimized chain featuring alternating regions of alpha-helix and beta-sheet, pushed to the limits of sequence-structure compatibility.
10. **`flatland_protein.pdb`**: The "2D Protein." Every Z-coordinate has been zeroed out, forcibly collapsing a 3D folded structure into a perfectly flat, impossible 2-dimensional plane with massive overlaps.
11. **`proline_alpha_helix.pdb`**: A poly-proline chain hard-coded into an ideal alpha-helical conformation. Because proline sidechains are rigid rings that lack amide hydrogens for H-bonding, this creates catastrophic, unfixable steric clashes.
12. **`exploded_protein.pdb`**: The "Macro-Protein." All inter-atomic distances have been scaled up by exactly 10x, creating a structure that looks like a protein but possesses giant, 15-Angstrom "covalent" bonds.
13. **`eclipsed_chain.pdb`**: The "Ramachandran Violation." Every single $\phi$ and $\psi$ dihedral angle is forced to exactly $0.0^\circ$, creating a maximally eclipsed geometry that represents a global peak of steric repulsion.
14. **`z_fighting_chimera.pdb`**: The "Z-Fighting Nightmare." A two-chain complex where Chain A (poly-Alanine) and Chain B (poly-Tryptophan) share the *exact same* spatial coordinates, resulting in an impossible molecular superposition.
15. **`black_hole_protein.pdb`**: The "Black Hole Protein." A 50-residue sequence where every single atomic coordinate is exactly `(0.000, 0.000, 0.000)`, simulating a total spatial collapse.
16. **`spaghetti_knot.pdb`**: The "Spaghetti Knot." A 20-chain complex where every single chain is independently generated and then centered at the exact origin, causing all 20 chains to inextricably overlap in a tangled mass.
17. **`overflow_protein.pdb`**: The "Coordinate Overflow." Coordinates are forced past the 9999.999 limit (`9999999.9`), intentionally breaking the strict PDB `8.3f` fixed-width column format to crash legacy parsers.
18. **`antimatter_protein.pdb`**: The "Antimatter Protein." All atomic occupancies are set to `-1.00` and all B-factors to `-99.99`, breaking standard biophysical assumptions about disorder and probabilities.
19. **`schrodingers_residue.pdb`**: "Schrodinger's Residue." Two overlapping `altloc` records (`A` and `B`) share the exact same spatial coordinates, but record `A` is Carbon and record `B` is Uranium. A true structural superposition.

## 🧬 Biological Dark Proteome Examples

These models represent real-world biological complexity found in disordered or non-standard systems:

*   **`alpha_syn_demo.pdb`**: A model of Alpha-synuclein, a classic intrinsically disordered protein (IDP) involved in Parkinson's disease.
*   **`tdp43_disordered.pdb`**: The low-complexity domain of TDP-43, known for its role in liquid-liquid phase separation (LLPS) and ALS.
*   **`ubiquitin_standard_candle.pdb`**: A high-resolution reference model used as a "standard candle" for NMR and Cryo-EM validation.
*   **`apela_orphan.pdb`**: A model of the Apela peptide, an "orphan" ligand with high conformational flexibility.
*   **`xcl1_hallucination_alpha.pdb` / `xcl1_hallucination_beta.pdb`**: Metamorphic protein states (XCL1) that can switch between entirely different folds.
*   **`ybea_knot_attempt.pdb`**: An exploration of knotted protein topologies.

## 🛠️ Scripts & Tools

*   **`cryo_em_demo.py`**: Script to generate synthetic Cryo-EM maps from these ensembles.
*   **`ensemble_demo.py`**: Demonstrates how to load and analyze multi-model ensembles in `synth-pdb`.
