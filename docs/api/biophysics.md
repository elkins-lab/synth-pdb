# biophysics Module

The `biophysics` module enhances synthetic structures with realistic biochemical properties, such as pH-dependent protonation and terminal capping.

## Overview

While the core generator builds the 3D atomic coordinates, the `biophysics` module handles the "chemical identity" of the residues, ensuring that protonation states and terminal groups are biologically accurate for a given environment.

## Main Functions

::: synth_pdb.biophysics.apply_ph_titration
    options:
      show_root_heading: true
      show_source: true

::: synth_pdb.biophysics.cap_termini
    options:
      show_root_heading: true
      show_source: true

::: synth_pdb.biophysics.find_salt_bridges
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### pH Titration

Apply pH-dependent protonation to residues like Histidine.

```python
import biotite.structure as struc
from synth_pdb.biophysics import apply_ph_titration

# Load structure
# structure: struc.AtomArray

# Apply acidic pH (renames HIS to HIP)
structure = apply_ph_titration(structure, ph=5.0)

# Apply physiological pH (probabilistically renames HIS to HIE or HID)
structure = apply_ph_titration(structure, ph=7.4)
```

### Terminal Capping

Add Acetyl (ACE) and N-Methylamide (NME) caps to the termini of a peptide fragment.

```python
from synth_pdb.biophysics import cap_termini

# Add caps to structure
structure = cap_termini(structure)
```

### Salt Bridge Detection

Identify potential ionic interactions between acidic and basic residues.

```python
from synth_pdb.biophysics import find_salt_bridges

bridges = find_salt_bridges(structure, cutoff=5.0)
for b in bridges:
    print(f"Salt bridge between {b['res_ia']} and {b['res_ib']}")
```

## Educational Notes

### pH and Protonation

Biological function depends heavily on pH. The most sensitive residue near physiological pH (7.4) is **Histidine** ($pK_a \approx 6.0$).
- **pH < 6.0**: The imidazole ring is protonated and carries a $+1$ charge. Represented as **HIP**.
- **pH > 6.0**: The ring is neutral ($0$ charge). It exists in two tautomeric forms: **HIE** ($\epsilon$ nitrogen protonated) or **HID** ($\delta$ nitrogen protonated).

### Terminal Capping

Uncapped termini ($NH_3^+$ and $COO^-$) introduce strong charges that are often unrealistic for a short peptide fragment intended to represent a region within a larger protein.
- **N-terminus cap (ACE)**: Acetyl group ($CH_3\text{-}CO\text{-}$) replaces the terminal hydrogen.
- **C-terminus cap (NME)**: N-Methylamide group ($\text{-}NH\text{-}CH_3$) replaces the terminal oxygen.

Capping eliminates these artificial terminal charges, providing a more realistic model of internal protein segments.

### Salt Bridges

A salt bridge is a combination of two non-covalent interactions: **Hydrogen Bonding** and **Electrostatic (Ionic) Attraction**. It occurs between a positively charged basic residue (Lys, Arg, His) and a negatively charged acidic residue (Asp, Glu). These bridges are critical for stabilizing tertiary structure and driving specific molecular recognition.

## References

- **Proline Conformation**: MacArthur, M. W., & Thornton, J. M. (1991). "Influence of proline residues on protein conformation." *Journal of Molecular Biology*. [DOI: 10.1016/0022-2836(91)90627-W](https://doi.org/10.1016/0022-2836(91)90627-W)
- **Salt Bridges**: Bosshard, H. R., et al. (2004). "The salt bridge in proteins." *Journal of Molecular Recognition*. [DOI: 10.1002/jmr.657](https://doi.org/10.1002/jmr.657)
- **pH and Proteins**: Tanford, C. (1962). "The interpretation of hydrogen ion titration curves of proteins." *Advances in Protein Chemistry*.

## See Also

- [physics Module](physics.md) - Physics-based refinement
- [validator Module](validator.md) - Geometric validation
- [Scientific Background: Biophysics Fundamentals](../science/biophysics.md)
