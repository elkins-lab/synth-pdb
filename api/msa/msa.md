# msa Module

The `msa` module implements a physical sequence-level evolutionary simulator to generate Multiple Sequence Alignments (MSAs) with co-evolutionary constraints.

## Overview

Based on **Direct Coupling Analysis (DCA)** theory, this module models sequence probability using a **Potts Energy Model**. It uses a Metropolis-Hastings Markov Chain Monte Carlo (MCMC) algorithm to simulate evolutionary drift, ensuring that produced sequences respect the native 3D fold (Contact Map).

## Main Classes

::: synth_pdb.msa.CoevolutionModel
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - calculate_energy
        - calculate_delta_energy

::: synth_pdb.msa.MetropolisHastingsSampler
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - start
        - step

## Main Functions

::: synth_pdb.msa.generate_msa
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Generating a Synthetic MSA

```python
import numpy as np
from synth_pdb.msa import generate_msa

# base_sequence: str
# contact_map: np.ndarray (N x N boolean)

msa = generate_msa(
    base_sequence="ACDEFGHIKL",
    contact_map=my_contact_map,
    num_sequences=50,
    temperature=1.0
)

for seq in msa:
    print(seq)
```

## Educational Notes

### Hydrophobic Core Collapse

Solvent Accessible Surface Area (SASA) is the physical mechanism mapping 3D structure back to 1D sequence constraints. If a residue is "buried" deep inside the protein core (low SASA), evolutionary drift must strictly eliminate charged/polar mutations. Placing a hydrophilic amino acid in the water-free hydrophobic core would rupture the hydrogen-bond network and unfold the protein. The `msa` module enforces this by penalizing hydrophilic mutations at buried positions.

### Electrostatic Compatibility

Proteins use localized regions of electrical charge (**Salt Bridges**) to lock their tertiary folds into stable, lower-energy states. Conversely, placing two like-charges in close proximity causes strong Coulombic repulsion. The Potts model in this module rewards opposite-charge pairs while aggressively penalizing like-charge complexes in contacting residues.

### The "Magic Step" Coupled Mutation

In traditional MCMC, only one site is mutated at a time. However, in evolution, getting from a [Large:Small] pair to a [Small:Large] pair is impossible if the intermediate [Large:Large] state causes a massive steric clash. The "Magic Step" proposes mutations at two contacting residues simultaneously, allowing the simulation to traverse these steric gaps and capture true **Direct Coupling** covariance.

## References

- **Direct Coupling Analysis (DCA)**: Morcos, F., et al. (2011). "Direct-coupling analysis of residue coevolution captures native contacts across many protein families." *Proceedings of the National Academy of Sciences (PNAS)*. [DOI: 10.1073/pnas.1111471108](https://doi.org/10.1073/pnas.1111471108)
- **Potts Models in Evolution**: Weigt, M., et al. (2009). "Identification of direct residue contacts in protein-protein interaction by message passing." *PNAS*. [DOI: 10.1073/pnas.0805923106](https://doi.org/10.1073/pnas.0805923106)

## See Also

- [generator Module](generator.md) - Generating the initial 3D fold
- [Scientific Background: Co-Evolution & MSAs](../science/coevolution.md)
