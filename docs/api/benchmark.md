# `benchmark` — Structure Prediction Benchmarking

The `synth_pdb.benchmark` and `synth_pdb.benchmark_metrics` modules provide
a complete suite for evaluating AI structure prediction models (AlphaFold,
ESMFold, RoseTTAFold, etc.) against ground-truth synthetic structures.

!!! note "The key insight"
    Because synth-pdb **controls the ground truth**, the benchmark is perfectly
    objective: there is no ambiguity about which experimental structure is "correct"
    or whether the reference contains modelling errors. This makes it ideal for
    **blind comparison** of structure prediction models.

---

## Quick Start

```python
from synth_pdb.benchmark import run_benchmark

# Score 20 structures predicted by ESMFold
results = run_benchmark(n_structures=20, predictor="esmfold")

# Print formatted summary
print(results.summary())

# Export to CSV for further analysis
results.to_csv("benchmark_results.csv")
```

Or from the command line:

```bash
# Install dependencies
pip install synth-pdb[gnn] transformers accelerate

# Run benchmark (downloads ESMFold ~700 MB on first use)
python scripts/run_benchmark.py --n-structures 20 --output results.csv
```

---

## `run_benchmark()`

```python
def run_benchmark(
    n_structures: int = 20,
    lengths: list[int] | None = None,
    conformations: list[str] | None = None,
    predictor: str | Callable[[str], str] = "esmfold",
    *,
    compute_shifts: bool = True,
    compute_gnn: bool = True,
    random_state: int = 42,
) -> BenchmarkResults
```

Generate synthetic ground-truth structures, fold them from sequence using a
structure predictor, then evaluate the predictions against the ground truth
using a comprehensive set of structural metrics.

### Parameters

| Parameter         | Type                          | Default         | Description |
|-------------------|-------------------------------|-----------------|-------------|
| `n_structures`    | `int`                         | `20`            | Number of test structures to generate and evaluate. |
| `lengths`         | `list[int]`                   | `[20, 30, 50]`  | Pool of chain lengths to sample from uniformly. |
| `conformations`   | `list[str]`                   | `["alpha", "beta"]` | Pool of secondary structure types. Options: `"alpha"`, `"beta"`, `"random"`. |
| `predictor`       | `str` or `Callable`           | `"esmfold"`     | `"esmfold"` for the built-in ESMFold backend, or any `predictor_fn(sequence: str) → pdb_str` callable. |
| `compute_shifts`  | `bool`                        | `True`          | Compute NMR chemical shift RMSD (requires `synth_pdb.chemical_shifts`). |
| `compute_gnn`     | `bool`                        | `True`          | Score both structures with the GNN pLDDT classifier. |
| `random_state`    | `int`                         | `42`            | RNG seed for reproducibility. |

### Returns

A [`BenchmarkResults`](#benchmarkresults) object.

### Using a Custom Predictor

```python
def my_predictor(sequence: str) -> str:
    """Return a PDB string for the given amino acid sequence."""
    # ... call your model here ...
    return pdb_string

results = run_benchmark(n_structures=10, predictor=my_predictor)
```

This accepts any callable — ColabFold, OmegaFold, a structure database lookup,
even a simple homology modelling pipeline.

---

## `BenchmarkResults`

```python
@dataclass
class BenchmarkResults:
    results:      list[StructureResult]
    predictor:    str
    n_structures: int
    n_success:    int
```

### Methods

#### `summary() → str`

Returns a formatted multi-line summary report:

```
━━ Benchmark: ESMFold (18/20 structures) ━━
  TM-score   mean=0.723  std=0.142  min=0.421  max=0.891
  GDT-TS     mean=0.681  std=0.159
  lDDT       mean=0.714  std=0.128
  Cα-RMSD    mean=2.84 Å  std=1.21 Å
  Shift RMSD mean=0.412 ppm  std=0.193 ppm
  GNN pLDDT  mean=0.834 (predicted structures)

  Structures with TM-score > 0.5 (same fold): 16/18 (89%)
```

#### `to_csv(path: str) → None`

Write full per-structure results (all 12 fields) to a CSV file.

#### `to_dataframe() → pd.DataFrame`

Return results as a pandas DataFrame (requires `pandas`).

---

## `StructureResult`

Per-structure result returned in `BenchmarkResults.results`.

| Field               | Type    | Description |
|---------------------|---------|-------------|
| `sequence`          | `str`   | Amino acid sequence (single-letter code). |
| `length`            | `int`   | Number of residues. |
| `conformation`      | `str`   | Ground-truth secondary structure type. |
| `tm_score`          | `float` | TM-score ∈ [0, 1]. Values > 0.5 indicate the same fold. |
| `gdt_ts`            | `float` | GDT-TS ∈ [0, 1]. CASP standard metric. |
| `lddt_mean`         | `float` | Mean per-residue lDDT ∈ [0, 1]. |
| `rmsd`              | `float` | Cα-RMSD in Å after Kabsch superposition. |
| `shift_rmsd`        | `float` | Weighted chemical shift RMSD in ppm. `NaN` if unavailable. |
| `gnn_score_ref`     | `float` | GNN global quality score for the ground-truth structure. |
| `gnn_score_pred`    | `float` | GNN global quality score for the predicted structure. |
| `predictor_time_s`  | `float` | Wall-clock inference time in seconds. |
| `error`             | `str`   | Non-empty string if prediction failed; other fields are `NaN`. |

---

## Benchmark Metrics Reference

All metric functions live in `synth_pdb.benchmark_metrics` and operate on
`numpy` arrays with no additional dependencies.

### `tm_score(ca_pred, ca_ref)`

```python
def tm_score(
    ca_pred: np.ndarray,   # [N, 3] predicted Cα coordinates
    ca_ref:  np.ndarray,   # [N, 3] reference Cα coordinates
    *,
    normalise_by: int | None = None,
) -> float
```

Compute TM-score. Returns a value in (0, 1]. Two unrelated structures score ≈ 0.17;
two structures with the same fold score > 0.5.

For the mathematical definition and interpretation, see the
[scientific background page](../science/gnn-structure-scoring.md#tm-score).

### `lddt(ca_pred, ca_ref)`

```python
def lddt(
    ca_pred:          np.ndarray,             # [N, 3]
    ca_ref:           np.ndarray,             # [N, 3]
    *,
    inclusion_radius: float = 15.0,           # Å
    thresholds:       tuple = (0.5, 1, 2, 4), # Å
) -> np.ndarray  # [N] per-residue lDDT ∈ [0, 1]
```

Per-residue lDDT. Does not require superposition. The global lDDT is
`float(np.mean(lddt(...)))`.

### `gdt_ts(ca_pred, ca_ref)`

```python
def gdt_ts(
    ca_pred: np.ndarray,             # [N, 3]
    ca_ref:  np.ndarray,             # [N, 3]
    *,
    cutoffs: tuple = (1.0, 2.0, 4.0, 8.0),  # Å
) -> float
```

GDT-TS — average fraction of Cα atoms within {1, 2, 4, 8} Å after superposition.

### `superpose_kabsch(mobile, reference)`

```python
def superpose_kabsch(
    mobile:    np.ndarray,  # [N, 3]
    reference: np.ndarray,  # [N, 3]
) -> tuple[np.ndarray, float]  # (rotated_coords, rmsd)
```

Optimally superpose `mobile` onto `reference` using the Kabsch algorithm (SVD-based).
Returns the rotated coordinate array and the Cα-RMSD in Å.

### `shift_rmsd(pred_shifts, ref_shifts)`

```python
def shift_rmsd(
    pred_shifts:      dict[str, np.ndarray],  # nucleus → per-residue shifts
    ref_shifts:       dict[str, np.ndarray],
    *,
    nucleus_weights:  dict[str, float] | None = None,
) -> float  # weighted shift RMSD in ppm
```

Weighted chemical shift RMSD following SPARTA+ nucleus weights (H=1.0, C=0.25, N=0.1).
NaN residues (missing assignments) are automatically excluded.

```python
from synth_pdb.benchmark_metrics import shift_rmsd
import numpy as np

# Compare predicted and reference ¹H shifts for 10 residues
rmsd = shift_rmsd(
    {"H": np.array([8.1, 8.2, 8.3, 8.0, 7.9, 8.4, 8.1, 8.2, 8.0, 7.8])},
    {"H": np.array([8.0, 8.1, 8.4, 8.0, 7.8, 8.3, 8.2, 8.1, 8.0, 7.9])},
)
print(f"¹H shift RMSD: {rmsd:.4f} ppm")
```

### `extract_ca_coords(pdb_content)`

```python
def extract_ca_coords(pdb_content: str) -> np.ndarray  # [N, 3]
```

Lightweight, pure-Python PDB parser that extracts Cα coordinates in residue order.
Handles duplicate residue numbers (keeps first occurrence per chain/residue pair).

```python
from synth_pdb.benchmark_metrics import extract_ca_coords, tm_score

ca_ref  = extract_ca_coords(open("reference.pdb").read())
ca_pred = extract_ca_coords(open("predicted.pdb").read())

n = min(len(ca_ref), len(ca_pred))
score = tm_score(ca_pred[:n], ca_ref[:n])
print(f"TM-score: {score:.3f}")
```

---

## CLI Reference

```bash
python scripts/run_benchmark.py [OPTIONS]

Options:
  --predictor {esmfold}       Structure prediction backend (default: esmfold)
  --n-structures INT          Number of test structures (default: 20)
  --lengths L [L ...]         Chain lengths to sample (default: 20 30 50)
  --conformations {alpha,beta,random} [...]
                              Secondary structure types (default: alpha beta)
  --output PATH               Save CSV results to this path
  --no-shifts                 Skip chemical shift RMSD
  --no-gnn                    Skip GNN quality scoring
  --random-state INT          RNG seed (default: 42)
  -v, --verbose               Enable DEBUG logging
```

### Example Runs

```bash
# Full benchmark with all metrics
python scripts/run_benchmark.py \
    --n-structures 50 \
    --lengths 20 30 50 \
    --output results/esmfold_benchmark.csv

# Fast geometry-only benchmark
python scripts/run_benchmark.py \
    --n-structures 100 \
    --no-shifts --no-gnn \
    --output results/fast_benchmark.csv

# Alpha-helix only
python scripts/run_benchmark.py \
    --conformations alpha \
    --n-structures 30 \
    --output results/helix_benchmark.csv
```

---

## Full API Reference

::: synth_pdb.benchmark
    handler: python
    options:
      members:
        - run_benchmark
        - BenchmarkResults
        - StructureResult

::: synth_pdb.benchmark_metrics
    handler: python
    options:
      members:
        - tm_score
        - lddt
        - gdt_ts
        - superpose_kabsch
        - shift_rmsd
        - extract_ca_coords

---

## See Also

- [Scientific Background: GNN Structure Scoring](../science/gnn-structure-scoring.md)
- [API Reference: score_structure()](scoring.md)
- [Tutorial: GNN pLDDT Explorer](../tutorials/gnn_plddt_explorer.ipynb)
