"""
synth_pdb.quality — Structural Quality Assessment Sub-package.

Public API
----------
GNN-based quality scoring (requires ``pip install synth-pdb[gnn]``):
    GNNQualityClassifier  — GNN classifier; load, predict, score()
    QualityScore          — Dataclass returned by score() with per-residue pLDDT

Publication-ready visualization functions (requires matplotlib + scipy):
    apply_publication_style        — Set journal-standard rcParams globally
    save_publication_figure        — Save any figure at 300 dpi with explicit format
    plot_chemical_shift_correlation — Exp vs synthetic shift scatter with stats
    plot_ramachandran_publication   — Backbone dihedral plot with region shading
    plot_saxs_publication           — SAXS I(q) vs q log-scale plot
"""

from synth_pdb.quality.plots import (
    apply_publication_style,
    plot_chemical_shift_correlation,
    plot_ramachandran_publication,
    plot_saxs_publication,
    save_publication_figure,
)

try:
    from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier, QualityScore

    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False


__all__ = [
    # Visualization (optional: requires matplotlib + scipy)
    "apply_publication_style",
    "save_publication_figure",
    "plot_chemical_shift_correlation",
    "plot_ramachandran_publication",
    "plot_saxs_publication",
    # GNN quality scorer (optional: requires torch + torch_geometric)
    "GNNQualityClassifier",
    "QualityScore",
]
