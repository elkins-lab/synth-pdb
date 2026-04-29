"""
synth_pdb.quality — Structural Quality Assessment Sub-package.

Public API
----------
Publication-ready visualization functions (requires matplotlib + scipy):
    apply_publication_style        — Set journal-standard rcParams globally
    save_publication_figure        — Save any figure at 300 dpi with explicit format
    plot_chemical_shift_correlation — Exp vs synthetic shift scatter with stats
    plot_ramachandran_publication   — Backbone dihedral plot with region shading
    plot_saxs_publication           — SAXS I(q) vs q log-scale plot

Quality classifiers and features (no matplotlib dependency):
    QualityClassifier  — GNN-based structure-quality classifier
"""

from synth_pdb.quality.plots import (
    apply_publication_style,
    plot_chemical_shift_correlation,
    plot_ramachandran_publication,
    plot_saxs_publication,
    save_publication_figure,
)

__all__ = [
    # Visualization (optional: requires matplotlib + scipy)
    "apply_publication_style",
    "save_publication_figure",
    "plot_chemical_shift_correlation",
    "plot_ramachandran_publication",
    "plot_saxs_publication",
]
