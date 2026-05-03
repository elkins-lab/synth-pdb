# Your First Structure

This page walks you through the complete workflow for generating, validating, and visualising your first protein structure with `synth-pdb` — from a single command to an interactive 3D view in your browser.

## Prerequisites

Make sure `synth-pdb` is installed:

```bash
pip install synth-pdb
```

For energy minimization you also need OpenMM:

```bash
pip install synth-pdb[physics]
```

---

## Step 1: Generate a Structure

The quickest way to create a structure is by specifying a **sequence** and a **conformation**:

=== "Alpha Helix"

    ```bash
    synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha --output helix.pdb
    ```

=== "Beta Sheet"

    ```bash
    synth-pdb --sequence "VTVTVTVTVTVTVT" --conformation beta --output sheet.pdb
    ```

=== "Random Length"

    ```bash
    synth-pdb --length 30 --conformation random --output random.pdb
    ```

The output is a full-atom PDB file with backbone + side-chain heavy atoms and hydrogens.

---

## Step 2: Visualise in Your Browser

Add `--visualize` to open an interactive 3D view immediately after generation:

```bash
synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha --visualize
```

This launches a **py3Dmol** viewer in your default browser, coloured by B-factor (flexibility).

---

## Step 3: Validate the Structure

`synth-pdb` runs validation automatically and prints a report. To see a detailed quality report:

```bash
synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha --output helix.pdb --validate
```

The report checks:

| Check | What it validates |
|-------|------------------|
| **Bond lengths** | Engh & Huber Z-scores vs. ideal geometry |
| **Ramachandran** | φ/ψ angles against Top2018 high-res dataset |
| **Rotamers** | Side-chain χ angles vs. Dunbrack library |
| **Steric clashes** | van der Waals overlap between non-bonded atoms |
| **Planarity** | Peptide ω angle deviation |

---

## Step 4: Energy Minimization (Optional)

For physically refined coordinates, add `--minimize`:

```bash
synth-pdb --sequence "LKELEKELEKELEKEL" --conformation alpha \
  --minimize --output helix_minimized.pdb
```

This runs **OpenMM L-BFGS minimization** with implicit OBC2 solvent and the AMBER force field. Typical runtime: 1–5 seconds on CPU.

---

## Step 5: Use the Python API

Everything available on the CLI is also accessible from Python:

```python
from synth_pdb import generate_structure, validate_structure

# Generate
pdb_text = generate_structure(
    sequence="LKELEKELEKELEKEL",
    conformation="alpha",
    minimize=True,
)

# Validate
report = validate_structure(pdb_text)
print(f"Ramachandran favoured: {report['ramachandran_favoured']:.1%}")
print(f"Rotamer outliers:      {report['rotamer_outliers']:.1%}")
print(f"Clashscore:            {report['clashscore']:.1f}")
```

---

## What's in the PDB File?

```text
ATOM      1  N   LEU A   1       2.345   1.234   0.000  1.00 15.23           N
ATOM      2  CA  LEU A   1       3.812   1.234   0.000  1.00 15.23           C
...
```

| Column | Field | Notes |
|--------|-------|-------|
| 7–11 | Serial | Atom number |
| 13–16 | Name | Atom name (N, CA, C, O, …) |
| 18–20 | Res name | 3-letter amino acid code |
| 23–26 | Res seq | Residue number |
| 31–54 | X, Y, Z | Cartesian coordinates (Å) |
| 61–66 | B-factor | Temperature factor — proxies for local flexibility |

The **B-factor** column in `synth-pdb` output encodes a physics-derived flexibility profile: high for terminal residues and loop regions, low for the hydrophobic core.

---

## Next Steps

- [Quick Start](quickstart.md) — more examples in 5 minutes
- [CLI Reference](../guides/cli-reference.md) — all command-line options
- [Interactive Tutorials](../../examples/interactive_tutorials/virtual_nmr_spectrometer.ipynb) — open in Google Colab
- [API Reference](../api/overview.md) — Python API documentation
