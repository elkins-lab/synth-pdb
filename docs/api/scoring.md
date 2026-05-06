# `score` — GNN Quality Scoring API

The `synth_pdb.score` module provides a **single-import, zero-configuration** interface
for scoring protein structures using the bundled Graph Attention Network (GNN) quality
classifier. It is the recommended entry point for all quality scoring tasks.

!!! note "Installation"
    ```bash
    pip install synth-pdb[gnn]    # installs torch + torch_geometric
    ```
    The `synth_pdb.score` module can be imported without PyTorch installed —
    the dependency is only checked when a scoring function is actually called.

---

## Quick Start

```python
from synth_pdb.score import score_structure, score_batch

# Score a PDB file by path
result = score_structure("my_helix.pdb")
print(f"Global quality: {result.global_score:.3f}  ({result.label})")
# Global quality: 0.999  (High Quality)

# Inspect per-residue pLDDT confidence
for i, (score, label) in enumerate(zip(result.per_residue, result.residue_labels)):
    print(f"  Residue {i+1:3d}: {score:.3f}  [{label}]")
# Residue   1: 0.958  [Very High]
# Residue   2: 0.960  [Very High]
# ...

# Score a batch efficiently (model loaded once)
results = score_batch(["helix.pdb", "strand.pdb", "decoy.pdb"])
best = max(results, key=lambda r: r.global_score)
print(f"Best structure: global_score={best.global_score:.3f}")
```

---

## `score_structure()`

```python
def score_structure(
    source: str | os.PathLike,
    *,
    model_path: str | None = None,
) -> QualityScore
```

Score a single protein structure and return a rich `QualityScore` object.

### Parameters

| Parameter    | Type                  | Description |
|--------------|-----------------------|-------------|
| `source`     | `str` or path-like    | A file path ending in `.pdb`, **or** a raw PDB-format string. File detection is based on whether the string starts with a PDB record keyword (`ATOM`, `REMARK`, `HEADER`, `MODEL`). |
| `model_path` | `str`, optional       | Path to a custom `.pt` checkpoint. Defaults to the bundled `gnn_quality_v2.pt` (with per-residue head). Falls back to `gnn_quality_v1.pt` if v2 is unavailable. |

### Returns

A [`QualityScore`](#qualityscore) dataclass.

### Raises

| Exception           | Condition |
|---------------------|-----------|
| `FileNotFoundError` | `source` looks like a file path but the file does not exist. |
| `ImportError`       | `torch` or `torch_geometric` are not installed. |
| `ValueError`        | The PDB contains fewer than 2 residues with Cα atoms. |

### Examples

```python
from synth_pdb.score import score_structure

# From a file path
result = score_structure("/data/structures/ubiquitin.pdb")

# From an inline PDB string
pdb_string = open("ubiquitin.pdb").read()
result = score_structure(pdb_string)

# Using a custom checkpoint
result = score_structure("ubiquitin.pdb", model_path="my_retrained_gnn.pt")
```

---

## `score_batch()`

```python
def score_batch(
    sources: list[str | os.PathLike],
    *,
    model_path: str | None = None,
) -> list[QualityScore]
```

Score a list of structures efficiently. The GNN model is loaded **once** and
reused for all structures — significantly faster than calling `score_structure()`
in a loop for large collections.

If any individual structure fails (e.g. unparseable PDB, too few residues),
a sentinel `QualityScore` with `global_score=NaN` and `label="Error"` is
inserted at the corresponding index, so **the output list always has the same
length as the input list**.

### Parameters

| Parameter    | Type                          | Description |
|--------------|-------------------------------|-------------|
| `sources`    | `list[str \| PathLike]`       | Mixed list of file paths and/or PDB strings. |
| `model_path` | `str`, optional               | Custom checkpoint path. |

### Returns

`list[QualityScore]` — one result per input, in the same order.

### Example

```python
import glob
from synth_pdb.score import score_batch

pdb_files = sorted(glob.glob("alphafold_predictions/*.pdb"))
results = score_batch(pdb_files)

# Rank by global quality score
ranked = sorted(zip(pdb_files, results), key=lambda x: x[1].global_score, reverse=True)
for path, r in ranked[:5]:
    print(f"{path:50s}  {r.global_score:.4f}  {r.label}")
```

---

## `QualityScore`

```python
@dataclass
class QualityScore:
    global_score:    float
    label:           str
    per_residue:     list[float]
    residue_labels:  list[str]
    features:        dict[str, float]
    n_residues:      int
```

Returned by `score_structure()`, `score_batch()`, and `GNNQualityClassifier.score()`.

### Fields

| Field            | Type            | Description |
|------------------|-----------------|-------------|
| `global_score`   | `float ∈ [0,1]` | P(Good) — probability the structure is biophysically plausible. Values > 0.5 are classified as High Quality. |
| `label`          | `str`           | `"High Quality"` or `"Low Quality"`. |
| `per_residue`    | `list[float]`   | Per-residue pLDDT-like confidence ∈ [0, 1]. Length equals `n_residues`. Analogous to AlphaFold's per-residue pLDDT. |
| `residue_labels` | `list[str]`     | Human-readable confidence band for each residue. |
| `features`       | `dict[str, float]` | Mean per-feature summary of the GNN input graph (useful for debugging). Keys: `sin_phi`, `cos_phi`, `sin_psi`, `cos_psi`, `b_factor_norm`, `seq_position`, `is_n_terminus`, `is_c_terminus`. |
| `n_residues`     | `int`           | Number of residues with Cα atoms in the PDB. |

### pLDDT Confidence Bands

| Label        | Score range | Interpretation (AlphaFold equivalent) |
|--------------|-------------|---------------------------------------|
| `"Very High"` | ≥ 0.90     | Backbone and side-chain likely accurate |
| `"High"`      | 0.70–0.90  | Backbone likely accurate |
| `"Uncertain"` | 0.50–0.70  | Use with caution |
| `"Low"`       | < 0.50     | Likely disordered or incorrect geometry |

### Example Usage

```python
result = score_structure("my_protein.pdb")

# Find low-confidence regions
low_conf = [
    i + 1  # 1-indexed residue number
    for i, label in enumerate(result.residue_labels)
    if label in ("Uncertain", "Low")
]
print(f"Low-confidence residues: {low_conf}")

# Check mean pLDDT
import numpy as np
mean_plddt = np.mean(result.per_residue)
print(f"Mean pLDDT: {mean_plddt:.3f}")

# Export to pandas for downstream analysis
import pandas as pd
df = pd.DataFrame({
    "residue": range(1, result.n_residues + 1),
    "plddt": result.per_residue,
    "band": result.residue_labels,
})
df.to_csv("plddt_per_residue.csv", index=False)
```

---

## `GNNQualityClassifier`

The lower-level class underlying `score_structure()`. Import it when you need
direct control over checkpoint loading or want to access the `predict()` method
for backward compatibility.

```python
from synth_pdb.quality import GNNQualityClassifier

clf = GNNQualityClassifier()                     # auto-loads bundled weights
clf = GNNQualityClassifier(model_path="v2.pt")   # explicit checkpoint
```

### Methods

#### `score(pdb_content: str) → QualityScore`

The primary method. Equivalent to `score_structure(pdb_content)` but requires
a PDB string (not a file path).

#### `predict(pdb_content: str) → (bool, float, dict)`

Legacy method for backward compatibility with the `ProteinQualityClassifier` (RF)
API. Returns `(is_good, probability, features_dict)`.

#### `save(path: str) → None`

Save model weights and architecture metadata to a `.pt` checkpoint.

#### `load(path: str) → None`

Load a checkpoint. The architecture (node features, hidden dim, etc.) is read
from the checkpoint itself — no configuration file needed.

---

## Retraining the Model

To retrain `gnn_quality_v2.pt` from scratch (e.g. after modifying the architecture
or adding training data):

```bash
python scripts/train_gnn_quality_filter.py \
    --n-samples 200 \
    --epochs 50 \
    --output synth_pdb/quality/models/gnn_quality_v2.pt
```

The training script generates 200 synthetic structures across four classes
(Good / Random / Distorted / Clashing) and trains with a joint objective:

$$\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda \cdot \mathcal{L}_{\text{MSE}}$$

where $\mathcal{L}_{\text{NLL}}$ is the global binary classification loss,
$\mathcal{L}_{\text{MSE}}$ is the per-residue Ramachandran Z-score regression loss,
and $\lambda = 0.3$ by default.

---

## Full API Reference

::: synth_pdb.score
    handler: python
    options:
      members:
        - score_structure
        - score_batch
        - QualityScore

::: synth_pdb.quality.gnn.gnn_classifier
    handler: python
    options:
      members:
        - GNNQualityClassifier
        - QualityScore

---

## See Also

- [Scientific Background: GNN Structure Scoring](../science/gnn-structure-scoring.md)
- [API Reference: benchmark](benchmark.md)
- [Tutorial: GNN pLDDT Explorer](../tutorials/gnn_plddt_explorer.ipynb)
- [Retrain the GNN](../development/ai_strategy.md)
