"""Circular Dichroism (CD) Spectroscopy Simulation.

This module provides tools for generating synthetic Circular Dichroism spectra
from 3D protein structures based on secondary structure content.

SCIENTIFIC BACKGROUND:
----------------------
Circular Dichroism (CD) measures the differential absorption of left-handed
and right-handed circularly polarized light by chiral molecules. In proteins,
the peptide bond (amide chromophore) absorbs in the far-UV region (190-250 nm).

The transition dipoles of the amide groups are coupled in a regular secondary
structure, leading to characteristic CD signatures:
- Alpha Helix: Strong negative peaks at 222 nm and 208 nm, positive peak at 192 nm.
- Beta Sheet: Negative peak at 217 nm, positive peak at 195 nm.
- Random Coil: Negative peak at 198 nm, weak positive signal near 220 nm.

This simulator uses "Basis Spectra" derived from experimental standards
(Greenfield & Fasman, 1969) to synthesize a spectrum weighted by the
fraction of each secondary structure element in the PDB model.
"""

import logging
from typing import Any

import numpy as np
import biotite.structure as struc

logger = logging.getLogger(__name__)

# Wavelength range for Far-UV CD (nm)
WAVELENGTHS = np.arange(190, 251, 1)

# Basis Spectra (Molar Ellipticity [theta] in deg*cm^2/dmol)
# Values approximated from Greenfield & Fasman (1969) / Provencher (1981)
BASIS_SPECTRA = {
    "H": np.interp(
        WAVELENGTHS,
        [190, 192, 200, 208, 215, 222, 240, 250],
        [0, 70000, 0, -36000, -25000, -38000, 0, 0],
    ),  # Helix
    "E": np.interp(
        WAVELENGTHS, [190, 195, 205, 217, 230, 240, 250], [-10000, 30000, 0, -18000, 0, 0, 0]
    ),  # Sheet
    "C": np.interp(
        WAVELENGTHS, [190, 198, 210, 220, 235, 250], [15000, -35000, -10000, 5000, 0, 0]
    ),  # Coil
}


class CDSimulator:
    """Simulates Circular Dichroism spectra from structural data.

    ### EDUCATIONAL NOTE — CD Background:
    # Circular Dichroism (CD) measures the differential absorption
    # of left and right circularly polarized light. In the far-UV
    # (190-250 nm), it is the premier tool for measuring the
    # overall secondary structure content of a protein sample.
    #
    # The physics is based on the interaction between amide
    # chromophores. For a given conformation, we can synthesize
    # the expected spectrum as a weighted average of basis
    # spectra (Greenfield & Fasman, 1969, Biochemistry 8:4108):
    #   [θ]total = f_helix · [θ]helix + f_sheet · [θ]sheet + f_coil · [θ]coil
    """

    def __init__(self, structure: struc.AtomArray):
        self.structure = structure
        self.sse = struc.annotate_sse(structure)
        self.fractions = self._calculate_fractions()

    def _calculate_fractions(self) -> dict[str, float]:
        """Calculates the fraction of H, E, and C in the structure."""
        n = len(self.sse)
        if n == 0:
            return {"H": 0.0, "E": 0.0, "C": 1.0}

        counts = {"H": 0, "E": 0, "C": 0}
        for s in self.sse:
            s_upper = s.upper()
            if s_upper in ["H", "G", "I", "A"]:  # Helices (H, G, I or A for alpha)
                counts["H"] += 1
            elif s_upper in ["E", "B"]:  # Sheets (E for strand, B for bridge)
                counts["E"] += 1
            else:  # Coil/Turn
                counts["C"] += 1

        return {k: v / n for k, v in counts.items()}

    def get_spectrum(self, noise_level: float = 500.0) -> np.ndarray:
        """Synthesizes the CD spectrum based on fractions."""
        spectrum = np.zeros_like(WAVELENGTHS, dtype=float)
        for sse_type, fraction in self.fractions.items():
            spectrum += fraction * BASIS_SPECTRA[sse_type]

        # Add random experimental noise
        if noise_level > 0:
            spectrum += np.random.normal(0, noise_level, size=spectrum.shape)

        return spectrum

    def plot(self, save_path: str | None = None) -> None:
        """Plots the synthetic CD spectrum."""
        import matplotlib.pyplot as plt

        spectrum = self.get_spectrum()

        plt.figure(figsize=(8, 5))
        plt.plot(WAVELENGTHS, spectrum, "k-", linewidth=2, label="Synthetic CD")
        plt.axhline(0, color="gray", linestyle="--", alpha=0.5)

        # Mark characteristic peaks
        if self.fractions["H"] > 0.3:
            plt.annotate(
                "222 nm",
                xy=(222, spectrum[WAVELENGTHS == 222]),
                xytext=(230, -30000),
                arrowprops={"arrowstyle": "->"},
            )
            plt.annotate(
                "208 nm",
                xy=(208, spectrum[WAVELENGTHS == 208]),
                xytext=(200, -45000),
                arrowprops={"arrowstyle": "->"},
            )

        plt.title(
            f"Synthetic Far-UV CD (H={self.fractions['H']:.1%}, E={self.fractions['E']:.1%}, C={self.fractions['C']:.1%})"
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(r"Molar Ellipticity $[\theta]$ ($deg \cdot cm^2 \cdot dmol^{-1}$)")
        plt.grid(alpha=0.3)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"CD plot saved to {save_path}")
        else:
            plt.show()


def validate_cd_against_literature(fractions: dict[str, float], spectrum: np.ndarray) -> list[str]:
    """Validates the synthetic CD spectrum against known literature values.

    Checks:
    1. Helix [theta]_222 should be ~ -36,000 for pure helix.
    2. Sheet [theta]_217 should be ~ -18,000 for pure sheet.
    """
    findings = []

    # Helix check
    if fractions["H"] > 0.8:
        val_222 = spectrum[WAVELENGTHS == 222][0]
        if -42000 < val_222 < -30000:
            findings.append(f"✓ Helix 222nm peak ({val_222:.0f}) matches literature (~ -36,000)")
        else:
            findings.append(f"⚠ Helix 222nm peak ({val_222:.0f}) deviates from standard helix.")

    # Sheet check
    if fractions["E"] > 0.8:
        val_217 = spectrum[WAVELENGTHS == 217][0]
        if -22000 < val_217 < -14000:
            findings.append(
                f"✓ Beta Sheet 217nm peak ({val_217:.0f}) matches literature (~ -18,000)"
            )
        else:
            findings.append(
                f"⚠ Beta Sheet 217nm peak ({val_217:.0f}) deviates from standard sheet."
            )

    return findings
