# Visualization Examples

This document demonstrates the different visualization options available in `synth-pdb`.

## 1. Browser-Based 3D Viewer

The easiest way to visualize a generated structure is to use the `--visualize` flag:

```bash
python -m synth_pdb.main --length 30 --conformation alpha --visualize
```

This will automatically open your default web browser and display the structure in a 3D viewer powered by [3Dmol.js](https://3dmol.csb.pitt.edu/).

**Features of the 3D Viewer:**
-   **Interactive:** Rotate, zoom, and pan using your mouse.
-   **Style Overrides:** Switch between Cartoon, Stick, Sphere, and Line representations.
-   **Coloring:** Choose from Spectrum, Chain, or Secondary Structure color schemes.
-   **NMR Restraints:** If generated with NOE restraints (`--gen-nef`), the viewer can display them as red lines between the atoms.
-   **Hydrogen Bonds:** Shows backbone hydrogen bonds as dashed lines.
-   **SSBOND Visualization:** Displays disulfide bonds as thick yellow cylinders.
-   **PTM Support:** Highlights post-translational modifications (like phosphorylation) with orange markers.

## 2. PyMOL Scripting

For professional-quality rendering and detailed analysis, `synth-pdb` can generate PyMOL scripts (.pml files).

```bash
python -m synth_pdb.main \
  --input-pdb structure.pdb \
  --input-nef data.nef \
  --mode pymol \
  --output-pml view_restraints.pml
```

Then open the script in PyMOL:
```bash
pymol view_restraints.pml
```

The script will automatically:
-   Load your PDB structure and NEF data.
-   Style the protein as a cartoon and sticks.
-   Visualize NOE restraints as distance cylinders (colored by violation if applicable).
-   Set up standard views for analysis.

## 3. High-Fidelity Solvent Visualization

If you run energy minimization with explicit solvent, you can choose to keep the water molecules in the final PDB:

```bash
python -m synth_pdb.main \
  --sequence "MEELQK" \
  --minimize \
  --solvent explicit \
  --keep-solvent \
  --output structure_with_water.pdb
```

This is particularly useful for studying solvation shells and water-mediated hydrogen bonds.

## 4. Contact Maps

You can also export contact maps as an alternative "2D visualization" of structural proximity:

```bash
python -m synth_pdb.main \
  --sequence "ALA-GLY-SER-THR-VAL" \
  --export-constraints contacts.csv \
  --constraint-format csv \
  --constraint-cutoff 8.0
```

The CSV file will contain pairs of residues that are within the 8.0 Angstrom cutoff, which can be plotted as a heatmap using libraries like `matplotlib`.
