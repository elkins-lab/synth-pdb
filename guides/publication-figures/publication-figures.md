# Publication-Ready Visualizations

The `synth_pdb.quality.plots` module provides tools to generate high-fidelity, journal-standard figures from your synthetic data. These plots are designed to meet the rigorous requirements of academic journals like *Nature*, *JACS*, and *Journal of Molecular Biology*.

## Key Features

- **High Resolution:** All plots default to 300 DPI.
- **Vector Graphics:** Support for `.pdf` and `.svg` export for lossless scaling.
- **Standardized Typography:** Uses clean, sans-serif fonts (Arial/Helvetica) with appropriate axis labeling.
- **Scientific Styles:** Includes shaded favored regions for Ramachandran plots and log-linear scaling for SAXS.

## Available Plot Types

### 1. Chemical Shift Correlation
Compare synthetic predictions against experimental BMRB data. The plot automatically calculates the Pearson Correlation ($R$) and RMSD.

```python
from synth_pdb.quality.plots import plot_chemical_shift_correlation

plot_chemical_shift_correlation(
    exp_shifts, syn_shifts, 
    atom_type="CA", 
    output_path="figures/cs_corr.pdf"
)
```

### 2. Ramachandran Plots
Visualize backbone geometry relative to favored alpha and beta regions.

```python
from synth_pdb.quality.plots import plot_ramachandran_publication

plot_ramachandran_publication(
    phi_deg, psi_deg, 
    title="Structural Sanity Check",
    output_path="figures/ramachandran.pdf"
)
```

### 3. SAXS Intensity Profiles
Standard $I(q)$ vs $q$ plots with optional Radius of Gyration ($R_g$) annotation.

```python
from synth_pdb.quality.plots import plot_saxs_publication

plot_saxs_publication(
    q, intensity, rg=12.4,
    output_path="figures/saxs_profile.pdf"
)
```

## Global Styling

You can apply the publication style to your own custom matplotlib plots using the helper function:

```python
from synth_pdb.quality.plots import apply_publication_style
import matplotlib.pyplot as plt

apply_publication_style()
# Your custom plot code here...
```

## Demonstration Script

A full example of generating publication figures for Ubiquitin is available in the repository:

```bash
python scripts/generate_publication_figures.py
```
