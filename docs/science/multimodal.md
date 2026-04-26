# Multimodal Scientific Observables

`synth-pdb` is unique in its ability to generate "ground truth" protein structures while simultaneously simulating data from multiple experimental modalities. This allows researchers to test integrative modeling techniques and benchmark how different structural features manifest in experimental observables.

## The Power of Multimodal Data

A single protein structure can be viewed through many different "lenses":

1.  **NMR**: Probes local environments and inter-atomic distances (NOEs, Chemical Shifts).
2.  **Cryo-EM**: Measures the global 3D Coulomb potential (Density Maps).
3.  **SAXS**: Captures the overall shape and size of the protein in solution (Scattering Profiles).

By simulating all three for the same synthetic protein, `synth-pdb` provides a complete "multimodal gold standard" where the underlying coordinates are known perfectly.

## Integrated Workflow

A typical multimodal simulation in `synth-pdb` follows these steps:

### 1. Structure Generation
Generate a structure or an ensemble (for flexible proteins) using the core generator and physics engine.

```python
from synth_pdb.generator import PeptideGenerator
gen = PeptideGenerator("MY_SEQUENCE")
ensemble = gen.generate_ensemble(n_models=20)
```

### 2. Local Observables (NMR)
Calculate chemical shifts and RDCs to understand the local electronic environment and orientation.

```python
from synth_pdb.chemical_shifts import predict_chemical_shifts
shifts = predict_chemical_shifts(ensemble[0])
```

### 3. Global Shape (SAXS)
Simulate the SAXS profile to verify the global dimensions ($R_g$, $D_{max}$) of the ensemble.

```python
from synth_pdb.saxs import calculate_saxs_profile
q, intensity = calculate_saxs_profile(ensemble[0])
```

### 4. 3D Volume (Cryo-EM)
Generate a density map at a specific resolution to simulate what an electron microscope would "see."

```python
from synth_pdb.cryo_em import generate_density_map
density, origin = generate_density_map(ensemble, resolution=4.0)
```

## Cross-Modality Validation

One of the most powerful uses of `synth-pdb` is checking for consistency between modalities. For example:
-   **NMR vs. SAXS**: Does the ensemble that satisfies all NOE distance restraints also match the experimental SAXS curve?
-   **Cryo-EM vs. NMR**: Can you fit the NMR-derived structure into the low-resolution Cryo-EM density?

## Applications in AI/ML

Synthetic multimodal data is essential for training the next generation of "multimodal" AI models that can simultaneously process protein sequences, density maps, and solution data to predict 3D structures. `synth-pdb` enables the creation of large-scale, high-fidelity training sets for these tasks.
