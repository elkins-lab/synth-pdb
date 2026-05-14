"""
Visualization Module for SAXS Profiles.
Provides plotting capabilities for I(q), Kratky, and Guinier plots.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional Matplotlib Dependency
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_saxs_results(
    q: np.ndarray,
    intensity: np.ndarray,
    title: str = "Synthetic SAXS Profile",
    output_path: str | None = None,
    plot_type: str = "standard",
    rg: float | None = None,
) -> Any:
    """Generate SAXS plots (Standard, Kratky, or Guinier).

    EDUCATIONAL RATIONALE:
    ----------------------
    SAXS data is a 1D representation of 3D structure. While the raw I(q) curve
    is the fundamental measurement, biological insights are often hidden in
    transformed plots.
    1. Standard (log I vs q): Shows the overall scattering decay.
    2. Kratky (q^2 * I vs q): Highly sensitive to the protein's folding state.
    3. Guinier (ln I vs q^2): Used to measure the overall size (Rg).

    Args:
        q: Scattering vector magnitudes.
        intensity: Scattering intensities I(q).
        title: Plot title.
        output_path: If provided, saves plot to file.
        plot_type: 'standard', 'kratky', 'guinier', or 'all'.
        rg: Optional Radius of Gyration (A) to overlay on Guinier plot.

    Returns:
        The matplotlib figure object, or None if matplotlib is missing.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not installed. Skipping SAXS visualization.")
        print("\n[INFO]  To enable SAXS visualization, install matplotlib: pip install matplotlib")
        return None

    if plot_type == "all":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        _draw_standard_plot(axes[0], q, intensity, title)
        _draw_kratky_plot(axes[1], q, intensity)
        _draw_guinier_plot(axes[2], q, intensity, rg)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        if plot_type == "standard":
            _draw_standard_plot(ax, q, intensity, title)
        elif plot_type == "kratky":
            _draw_kratky_plot(ax, q, intensity, title)
        elif plot_type == "guinier":
            _draw_guinier_plot(ax, q, intensity, rg, title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"SAXS plot saved to {output_path}")

    return fig


def _draw_standard_plot(ax: Any, q: np.ndarray, intensity: np.ndarray, title: str = "") -> None:
    """Log-linear I(q) vs q plot."""
    ax.semilogy(q, intensity, "b-", linewidth=2, label="I(q)")
    ax.set_xlabel(r"q ($\AA^{-1}$)", fontsize=12)
    ax.set_ylabel("log I(q)", fontsize=12)
    ax.set_title(title or "SAXS Intensity Profile", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()


def _draw_kratky_plot(ax: Any, q: np.ndarray, intensity: np.ndarray, title: str = "") -> None:
    """Dimensionless-style Kratky plot (q^2 * I(q) vs q).

    EDUCATIONAL NOTE - The Kratky Plot:
    ----------------------------------
    The Kratky plot is used to assess the "compactness" or folding state of
    a protein in solution.
    - Folded Globular Proteins: Show a clear bell-shaped curve (peak) that
      returns toward the baseline at high q. This is because I(q) for a sphere
      decays faster than 1/q^2.
    - Unfolded/Random Coil Proteins: Show a curve that continues to rise or
      plateaus at high q. This indicates a lack of a well-defined compact core.
    """
    kratky = (q**2) * intensity
    ax.plot(q, kratky, "r-", linewidth=2, label=r"$q^2 \cdot I(q)$")
    ax.set_xlabel(r"q ($\AA^{-1}$)", fontsize=12)
    ax.set_ylabel(r"$q^2 \cdot I(q)$", fontsize=12)
    ax.set_title(title or "Kratky Plot (Folding/Flexibility)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Note on interpretation
    # A bell shape indicates a folded globular protein.
    # A rising curve at high q indicates an unfolded/flexible ensemble.
    ax.legend()


def _draw_guinier_plot(
    ax: Any, q: np.ndarray, intensity: np.ndarray, rg: float | None = None, title: str = ""
) -> None:
    """Guinier plot (ln(I) vs q^2) for Rg estimation.

    EDUCATIONAL NOTE - The Guinier Approximation:
    --------------------------------------------
    At very low scattering angles (low q), the scattering intensity can be
    approximated as:
    I(q) ~ I(0) * exp(-q^2 * Rg^2 / 3)

    By plotting ln(I) vs q^2, we get a straight line in the low-q region.
    The slope of this line is -Rg^2 / 3. This is the most common method
    for determining the Radius of Gyration (Rg) of a protein in solution.
    """
    # Only use the low-q region (q*Rg < 1.3)
    # Since we don't always know Rg, we take the first 10% of points as a heuristic
    cut = max(5, len(q) // 10)
    q_low = q[:cut]
    i_low = intensity[:cut]

    q2 = q_low**2
    ln_i = np.log(i_low)

    ax.plot(q2, ln_i, "go", markersize=4, label="Low-q Data")

    # Linear fit
    if len(q2) > 2:
        slope, intercept = np.polyfit(q2, ln_i, 1)
        rg_est = np.sqrt(-3 * slope)
        fit_line = slope * q2 + intercept
        ax.plot(q2, fit_line, "k--", alpha=0.7, label=rf"Fit ($R_g \approx {rg_est:.2f} \AA$)")

    if rg is not None:
        ax.annotate(
            rf"True $R_g = {rg:.2f} \AA$",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            bbox={"boxstyle": "round", "fc": "w", "alpha": 0.5},
        )

    ax.set_xlabel(r"$q^2$ ($\AA^{-2}$)", fontsize=12)
    ax.set_ylabel("ln I(q)", fontsize=12)
    ax.set_title(title or "Guinier Plot", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
