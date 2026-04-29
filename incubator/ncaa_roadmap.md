# 🧬 NCAA Support: Expanding the Chemical Alphabet

**Status:** 💡 Concept Stage / Research Needed
**Priority:** Medium (High for Drug Discovery use cases)

## 🏗️ The Vision
Extend `synth-pdb` beyond the 20 standard L-amino acids to support **Non-Canonical Amino Acids (NCAAs)**. This would allow the generation of peptidomimetics, stapled peptides, and proteins with expanded genetic codes (e.g., p-azidophenylalanine) at the same "tensor-speed" as standard proteins.

## 🌟 Advantages of NCAA Support
1.  **Peptide Drug Discovery:** Most therapeutic peptides use NCAAs to improve bioavailability, metabolic stability (resistance to proteases), and binding affinity. Supporting these allows `synth-pdb` to be used in real-world medicinal chemistry pipelines.
2.  **Expanded Chemical Space:** Incorporate unique functionalities like fluorophores, "click" chemistry handles, and photo-crosslinkers directly into synthetic structures.
3.  **Enhanced AI Training:** Modern AI models (like AlphaFold-3) are beginning to support NCAAs and ligands. `synth-pdb` could generate massive, labeled datasets of NCAA-containing structures to benchmark these models' understanding of non-standard chemistry.
4.  **Mirror-Image Biology:** While D-amino acids are partially supported, a robust NCAA framework would allow full simulation of "D-proteins" and other chiral variants.

## 🛠️ Technical Implementation Roadmap

### 1. Sequence Parsing & Normalization
*   **Current State:** `_resolve_sequence` handles standard 1-letter/3-letter codes and `D-` isomers.
*   **Target:** A dictionary-based lookup for standard NCAA codes (e.g., `ORN` for Ornithine, `B3A` for Beta-Alanine) and support for custom user-defined residues.

### 2. Custom Residue Templates (The "Biotite Bridge")
*   **Current State:** Relies on `biotite.structure.info.residue` for atom templates.
*   **The Problem:** Biotite's internal library is immutable and limited to standard residues.
*   **The Solution:** Implement a `CustomResidueLibrary` that can load templates from PDB/CIF files or a JSON-based format (re-activating the dormant `AMINO_ACID_ATOMS` in `data.py`).

### 3. Generalized NeRF for Non-Standard Backbones
*   **Current State:** NeRF logic is hardcoded for $\alpha$-amino acid backbone geometry ($N-CA-C$).
*   **Target:** Generalize the `_build_peptide_chains` logic to allow variable backbone atoms (e.g., $\beta$-amino acids with an extra $CB$ in the backbone) or non-standard bond lengths/angles.

### 4. Forcefield Parameterization (The "OpenMM Bottleneck")
*   **Current State:** Uses standard Amber forcefields (`amber14-all.xml`).
*   **The Problem:** OpenMM will crash if it encounters a residue it doesn't recognize in the XML.
*   **The Solution:** 
    *   Tier 1: Manual mapping of NCAAs to "closest standard" residues for basic minimization.
    *   Tier 2: Support for GAFF (General Amber Force Field) via `OpenFF` or `Antechamber` to automatically parameterize new molecules.

## 🧪 Candidate Targets for Prototyping
*   **AIB (Alpha-aminoisobutyric acid):** A strong helix inducer common in peptide drugs.
*   **SEP/TPO/PTR:** Full support for these phosphorylated residues beyond simple renaming.
*   **Stapled Peptides:** Covalent cross-links between side chains (e.g., via hydrocarbon staples).

## 📚 References
*   Parsons et al. (2005). "Practical conversion from torsion space to Cartesian space for in silico protein synthesis." *Journal of Computational Chemistry*.
*   Wang et al. (2004). "Development and testing of a general amber force field." *Journal of Computational Chemistry*.
