"""
Publication-Quality Visualization Module for synth-pdb.

This module provides standardized plotting functions designed for academic
journals (e.g., Nature, Journal of Molecular Biology, JACS). It focuses on
high-DPI export, consistent typography, and scientifically accurate scales.
"""

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional Matplotlib Dependency
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def apply_publication_style() -> None:
    """Apply global matplotlib settings for publication-ready figures."""
    if not HAS_MATPLOTLIB:
        return

    # Use a clean, professional style as a base
    plt.style.use("seaborn-v0_8-whitegrid")

    # Customize for academic journals
    params = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    }
    mpl.rcParams.update(params)


def save_publication_figure(fig: Any, path: str, transparent: bool = False) -> None:
    """Save a figure with journal-standard defaults."""
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        path += ".pdf"

    fig.savefig(path, dpi=300, transparent=transparent, bbox_inches="tight")
    logger.info(f"Publication figure saved to {path}")


def plot_chemical_shift_correlation(
    exp_data: dict[int, dict[str, float]],
    syn_data: dict[int, dict[str, float]],
    atom_type: str = "CA",
    title: str | None = None,
    output_path: str | None = None,
) -> Any:
    """Generate a high-fidelity correlation plot for chemical shifts."""
    if not HAS_MATPLOTLIB:
        return None

    apply_publication_style()

    # Data alignment
    common_res = sorted(set(exp_data.keys()) & set(syn_data.keys()))
    x_list = []
    y_list = []
    for r in common_res:
        if atom_type in exp_data[r] and atom_type in syn_data[r]:
            x_list.append(exp_data[r][atom_type])
            y_list.append(syn_data[r][atom_type])

    x = np.array(x_list)
    y = np.array(y_list)

    if len(x) < 2:
        logger.warning(f"Insufficient data for {atom_type} correlation plot.")
        return None

    # Calculate statistics
    from scipy.stats import pearsonr

    r_val, _ = pearsonr(x, y)
    rmsd = np.sqrt(np.mean((x - y) ** 2))

    fig, ax = plt.subplots(figsize=(4.5, 4))

    # Use a professional color (Teal for synth-pdb)
    ax.scatter(x, y, s=25, alpha=0.6, edgecolors="none", color="#008080", label=f"n={len(x)}")

    # Diagonal line
    all_data = np.concatenate([x, y])
    padding = (float(all_data.max()) - float(all_data.min())) * 0.05
    lims = [float(all_data.min()) - padding, float(all_data.max()) + padding]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, zorder=0)

    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f"Experimental {atom_type} Shift (ppm)")
    ax.set_ylabel(f"Synthetic {atom_type} Shift (ppm)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{atom_type} Correlation ($R = {r_val:.3f}$)")

    # Stats annotation
    stats_text = f"RMSD: {rmsd:.2f} ppm\nPearson R: {r_val:.3f}"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    plt.tight_layout()
    if output_path:
        save_publication_figure(fig, output_path)

    return fig


def plot_ramachandran_publication(
    phi: np.ndarray,
    psi: np.ndarray,
    title: str = "Ramachandran Plot",
    output_path: str | None = None,
) -> Any:
    """Generate a publication-quality Ramachandran plot with favored regions."""
    if not HAS_MATPLOTLIB:
        return None

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Professional shaded background for favored regions
    # (Approximate regions for general protein backbone)
    # Alpha region
    ax.add_patch(plt.Rectangle((-180, -120), 150, 180, color="blue", alpha=0.08, zorder=0))
    # Beta region
    ax.add_patch(plt.Rectangle((-180, 60), 135, 120, color="red", alpha=0.08, zorder=0))
    # Wraparound for beta
    ax.add_patch(plt.Rectangle((-180, -180), 135, 40, color="red", alpha=0.08, zorder=0))

    # Scatter points
    ax.scatter(phi, psi, s=20, alpha=0.7, edgecolors="black", linewidth=0.5, color="#e74c3c")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)

    ax.set_xlabel(r"$\phi$ (degrees)")
    ax.set_ylabel(r"$\psi$ (degrees)")
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    if output_path:
        save_publication_figure(fig, output_path)

    return fig


def plot_saxs_publication(
    q: np.ndarray,
    intensity: np.ndarray,
    rg: float | None = None,
    output_path: str | None = None,
) -> Any:
    """Standardized SAXS I(q) vs q plot for papers."""
    if not HAS_MATPLOTLIB:
        return None

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    # Use log scale for Y as is standard
    ax.semilogy(q, intensity, color="#2c3e50", linewidth=1.5, label="Synthetic $I(q)$")

    ax.set_xlabel(r"$q$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$\log I(q)$")
    ax.set_title("Scattering Profile")

    if rg:
        ax.text(
            0.95,
            0.95,
            rf"$R_g = {rg:.2f} \AA$",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if output_path:
        save_publication_figure(fig, output_path)

    return fig
