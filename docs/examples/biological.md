# Biologically-Inspired Examples

This document showcases how to use `synth-pdb` to generate structures that mimic real biological proteins and motifs.

## 1. Human Epidermal Growth Factor (EGF)

EGF is a small protein (53 residues) with a complex disulfide-bonded structure. It contains three critical disulfide bonds (C6-C20, C14-C31, C33-C42) that are essential for its biological activity.

You can generate an EGF-like structure using the following command:

```bash
python -m synth_pdb.main \
  --sequence "NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR" \
  --conformation random \
  --minimize \
  --cap-termini \
  --gen-shifts \
  --gen-relax \
  --output egf_protein.pdb
```

**Key Features of this Example:**
-   **Sequence:** Uses the actual human EGF sequence.
-   **Minimization:** The `--minimize` flag triggers OpenMM to relax the structure and automatically detect/model the disulfide bonds.
-   **NMR Data:** Generates synthetic chemical shifts and relaxation data (`--gen-shifts`, `--gen-relax`).
-   **Biophysical Realism:** Adds N- and C-terminal caps (`--cap-termini`).

## 2. Zinc Finger Motif

Zinc fingers are common structural motifs in proteins that coordinate one or more zinc ions to help stabilize their fold.

`synth-pdb` can automatically detect Zinc-binding motifs (like CCHH or CCCC) and insert a Zinc ion:

```bash
python -m synth_pdb.main \
  --sequence "PYKCPECGKSFSQKSDLVKHQRTHTG" \
  --conformation random \
  --minimize \
  --metal-ions auto \
  --output zinc_finger.pdb
```

The `--metal-ions auto` flag scans the sequence for known binding motifs and places the ZN ion in the geometric center of the coordinating residues.

## 3. Cyclic Peptides

Many natural antibiotics and hormones are cyclic peptides. `synth-pdb` can generate head-to-tail cyclic structures:

```bash
python -m synth_pdb.main \
  --sequence "C-G-G-C" \
  --cyclic \
  --minimize \
  --output cyclic_peptide.pdb
```

The `--cyclic` flag ensures that the N-terminal Nitrogen and C-terminal Carbon are bonded, and the minimizer relaxes the structure into a stable cyclic conformation.

## 4. D-Amino Acid Peptides

D-amino acids are often found in bacterial cell walls and specialized peptides. They are "mirror images" of the standard L-amino acids and can impart resistance to proteases.

```bash
python -m synth_pdb.main \
  --sequence "DAL-GLY-GLY-DPH" \
  --output d_peptide.pdb
```

`synth-pdb` supports the standard PDB 3-letter codes for D-amino acids (e.g., `DAL` for D-Alanine, `DPH` for D-Phenylalanine).
