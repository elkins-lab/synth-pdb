# `quality` — Publication Visualization & Structure Quality

The `synth_pdb.quality` sub-package provides two complementary capabilities:

1. **Publication-ready plots** — journal-standard figures for chemical shift correlations, Ramachandran analysis, and SAXS profiles.
2. **Structure quality classification** — GNN-based and feature-based classifiers for scoring structural plausibility.

## Optional Dependencies

The plotting functions require `matplotlib`. The correlation plot additionally requires `scipy`. If either is absent the functions return `None` and log an error — they do not raise, so scripts can run in headless environments.

```bash
pip install synth-pdb[viz]   # installs matplotlib + scipy
```

---

## Plotting Functions

Import directly from the sub-package:

```python
from synth_pdb.quality import (
    apply_publication_style,
    save_publication_figure,
    plot_chemical_shift_correlation,
    plot_ramachandran_publication,
    plot_saxs_publication,
)
```

Or from the full module path:

```python
from synth_pdb.quality.plots import plot_ramachandran_publication
```

### `apply_publication_style()`

Sets journal-standard `matplotlib` rcParams globally: sans-serif fonts (Arial/Helvetica), 300 DPI save default, cleaned tick sizes.

!!! note "Side-effect scope"
    This modifies global `rcParams`. It intentionally does **not** set `savefig.format` — the format is always passed explicitly via `save_publication_figure` to avoid silently changing the output format of other figures in the same process.

### `save_publication_figure(fig, path, transparent=False)`

Saves a figure at 300 DPI with `bbox_inches="tight"`. The format is derived from the file extension (defaults to PDF when no extension is given).

```python
save_publication_figure(fig, "output/ca_correlation.pdf")
save_publication_figure(fig, "output/figure1")        # → saves as .pdf
save_publication_figure(fig, "output/thumbnail.png")  # → saves as .png
```

### `plot_chemical_shift_correlation`

```python
fig = plot_chemical_shift_correlation(
    exp_data,           # dict[int, dict[str, float]]  residue → {atom: ppm}
    syn_data,           # dict[int, dict[str, float]]  (same structure)
    atom_type="CA",     # which nucleus to plot
    title=None,         # optional override; auto-generates R value in title
    output_path=None,   # if set, saves the figure
)
```

Generates a scatter plot of experimental vs synthetic shifts with a diagonal reference line. Annotates Pearson R and RMSD automatically.

### `plot_ramachandran_publication`

```python
fig = plot_ramachandran_publication(
    phi,            # np.ndarray of φ angles in degrees
    psi,            # np.ndarray of ψ angles in degrees
    title="Ramachandran Plot",
    output_path=None,
)
```

!!! warning "Approximate region shading"
    The α-helical (blue) and β-strand (red) shaded regions are simplified rectangular approximations. They are **not** the probability-density contours from MolProbity or the Richardson Top8000 dataset. Use them for quick visual reference only — do not cite them as quantitative Ramachandran statistics.

### `plot_saxs_publication`

```python
fig = plot_saxs_publication(
    q,              # np.ndarray — scattering vector (Å⁻¹)
    intensity,      # np.ndarray — I(q)
    rg=None,        # float | None — annotates Rg on the plot when provided
    output_path=None,
)
```

Plots I(q) on a log-linear scale (standard for SAXS publications).

---

## Full API Reference

::: synth_pdb.quality.plots
    handler: python
    options:
      members:
        - apply_publication_style
        - save_publication_figure
        - plot_chemical_shift_correlation
        - plot_ramachandran_publication
        - plot_saxs_publication

---

## Complete Workflow Example

```python
import numpy as np
import biotite.structure.io.pdb as pdb_io

from synth_pdb.bmrb_api import BMRBAPI
from synth_pdb.chemical_shifts import predict_chemical_shifts
from synth_pdb.saxs import calculate_saxs_profile, calculate_radius_of_gyration
from synth_pdb.quality import (
    plot_chemical_shift_correlation,
    plot_ramachandran_publication,
    plot_saxs_publication,
)
import biotite.structure as struc

# Load structure
structure = pdb_io.PDBFile.read("examples/1D3Z.pdb").get_structure(model=1)

# Chemical shift correlation
exp_shifts = BMRBAPI.fetch_chemical_shifts("6457")
syn_shifts_full = predict_chemical_shifts(structure)
syn_shifts = list(syn_shifts_full.values())[0]   # first chain

fig = plot_chemical_shift_correlation(
    exp_shifts, syn_shifts, atom_type="CA",
    output_path="figures/ca_correlation.pdf",
)

# Ramachandran plot
phi, psi, _ = struc.dihedral_backbone(structure)
mask = ~np.isnan(phi) & ~np.isnan(psi)
plot_ramachandran_publication(
    np.degrees(phi[mask]), np.degrees(psi[mask]),
    output_path="figures/ramachandran.pdf",
)

# SAXS profile
q, intensity = calculate_saxs_profile(structure, q_max=0.3)
rg = calculate_radius_of_gyration(structure)
plot_saxs_publication(q, intensity, rg=rg, output_path="figures/saxs.pdf")
```
