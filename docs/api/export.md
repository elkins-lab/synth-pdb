# export Module

The `export` module provides tools for converting internal data structures (like contact maps and distance matrices) into standard text formats for AI/ML modeling and structural competition benchmarks.

## Overview

Generating 3D structures is only half the battle for ML researchers; the other half is exporting the ground-truth data in a format that training pipelines can ingest. This module handles the conversion of $N \times N$ matrices into CASP-style residue-residue (RR) files or simple CSVs.

### Key Features

-   **CASP RR Support**: Export contact maps in the standard format used by the Critical Assessment of Structure Prediction (CASP).
-   **CSV Export**: Simple, human-readable comma-separated values for rapid prototyping in Python or Excel.
-   **Separation Cutoffs**: Filter out short-range contacts (neighbors) to focus on the long-range interactions that define the protein fold.
-   **Probability Handling**: Supports both binary contacts (0/1) and continuous probability values.

## API Reference

::: synth_pdb.export
    handler: python
    options:
      members:
        - export_constraints

## Scientific Background

### The CASP RR Format
The CASP Residue-Residue (RR) format is the industry standard for inter-residue contact predictions. A typical line looks like:

`i j d_minor d_major prob`

-   `i`, `j`: Residue indices.
-   `d_minor`, `d_major`: The distance bin (e.g., `0.0 8.0` for a standard contact).
-   `prob`: The confidence or probability (1.0 for ground-truth data).

This module automatically maps distances to these bins, allowing synthetic structures from `synth-pdb` to be used as ground-truth targets for benchmarking structure prediction algorithms.

## Usage Example

```python
from synth_pdb.batch_generator import BatchedGenerator
from synth_pdb.export import export_constraints

# 1. Generate a batch of data (including contact maps)
gen = BatchedGenerator(batch_size=1, length=50)
batch = gen.generate_batch()
contact_map = batch["contacts"][0] # 50x50 matrix
sequence = "A" * 50 # Example sequence

# 2. Export to CASP RR format
casp_content = export_constraints(
    contact_map, 
    sequence, 
    fmt="casp", 
    threshold=8.0, 
    separation_cutoff=5
)

with open("contacts.rr", "w") as f:
    f.write(casp_content)

# 3. Export to simple CSV
csv_content = export_constraints(
    contact_map, 
    sequence, 
    fmt="csv", 
    threshold=12.0
)
```
