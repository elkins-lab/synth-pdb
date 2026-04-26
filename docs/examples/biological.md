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

## 5. Green Fluorescent Protein (GFP)

The Green Fluorescent Protein (GFP) contains a unique chromophore formed by the autocatalytic cyclization of a tripeptide motif (Ser65-Tyr66-Gly67).

`synth-pdb` can model this post-translational modification using the `special_chemistry` module:

```bash
python -m synth_pdb.main \
  --sequence "FEGUFSYGVQCFS" \
  --conformation alpha \
  --minimize \
  --output gfp_fragment.pdb
```

**Key Features:**
-   **Chromophore Maturation**: When an `SYG`, `TYG`, or `GYG` motif is detected, `synth-pdb` can be instructed to form the five-membered heterocyclic ring.
-   **Covalent Modeling**: The resulting PDB file contains the matured chromophore with the correct covalent connectivity.
