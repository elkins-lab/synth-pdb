# dataset Module

The `dataset` module provides tools for orchestrating the large-scale generation of synthetic protein datasets for AI model training.

## Overview

Generating diverse, balanced datasets is critical for training robust deep learning models like AlphaFold or RosettaFold. The `dataset` module automates the production of thousands of (Structure, Sequence, Constraint) triplets.

## Main Classes

::: synth_pdb.dataset.DatasetGenerator
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - generate
        - prepare_directories

## Main Functions

::: synth_pdb.dataset._generate_single_sample_task
    options:
      show_root_heading: true
      show_source: false

::: synth_pdb.dataset._generate_single_sample_npz_task
    options:
      show_root_heading: true
      show_source: false

## Usage Examples

### Bulk Dataset Generation

Generate a balanced dataset of 1,000 structures with varied secondary structures and lengths.

```python
from synth_pdb.dataset import DatasetGenerator

generator = DatasetGenerator(
    output_dir="./synthetic_dataset",
    num_samples=1000,
    min_length=30,
    max_length=150,
    train_ratio=0.8,
    max_workers=8
)

generator.generate()
```

The resulting directory will contain:
- `train/`: PDB and CASP (contact map) files for training.
- `test/`: PDB and CASP files for testing.
- `dataset_manifest.csv`: A manifest mapping IDs to file paths and metadata.

### AI-Ready NPZ Export

For deep learning frameworks, it is often more efficient to store data in compressed NumPy format.

```python
generator = DatasetGenerator(
    output_dir="./ai_dataset",
    dataset_format="npz"
)
generator.generate()
```

## Educational Notes

### The Balanced Dataset Problem

When training AI models, the **quality and balance** of the data are often more important than the quantity.
1. **The Alpha-Helix Trap**: If a dataset only contains helices, the AI will fail to generalize to beta-sheets or disordered regions.
2. **Mixed Conformations**: This module encourages a mix of 'alpha', 'beta', and 'random' conformations to ensure the model learns the full breadth of protein geometry.
3. **Structural Diversity**: Varying lengths and sequences minimizes "Selection Bias," leading to more robust models.

### Why Distance Matrices instead of Binary Contact Maps?

Binary contact maps (0/1) indicate whether atoms are within a threshold (usually 8.0 Å). While common, they discard detailed geometric information. Modern models (like AlphaFold) use **Distograms** (weighted distance bins) or raw distances to learn a continuous representation of the energy landscape. The `dataset` module can export exact ground-truth distances to support these advanced training objectives.

## See Also

- [batch_generator Module](batch_generator.md) - Vectorized structure generation
- [generator Module](generator.md) - Serial structure generation
- [Scientific Background: Energy Minimization](../science/energy-minimization.md)
