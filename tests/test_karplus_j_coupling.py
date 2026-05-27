"""
Scientific Validation: Karplus Equation Fidelity.

Validates that calculate_hn_ha_coupling correctly implements the Karplus
equation and that couplings from generated structures are physically sensible.

SCIENTIFIC BASIS:
  The synth_nmr engine uses Vuister & Bax (1993) Karplus coefficients:
      A = 6.51 Hz,  B = -1.76 Hz,  C = 1.60 Hz
  Formula:  J(theta) = A * cos^2(theta) + B * cos(theta) + C
  where theta = phi - 60 degrees for L-amino acids.

  Expected values by secondary structure:
    - alpha-helix (phi ~ -60):  J ~ 4.1 Hz   (small)
    - beta-sheet  (phi ~ -135): J ~ 9.4 Hz   (large)
    - PPII        (phi ~ -75):  J ~ 6.1 Hz   (intermediate)

REFERENCES:
  Vuister, G.W. & Bax, A. (1993). Quantitative J correlation: a new
  approach for measuring homonuclear three-bond J(HN-HA) coupling constants
  in 15N-enriched proteins. J Am Chem Soc, 115, 7772-7777.
  DOI: 10.1021/ja00070a024

  Karplus, M. (1963). Vicinal proton coupling in nuclear magnetic resonance.
  J Am Chem Soc, 85, 2870-2871. DOI: 10.1021/ja00901a059
"""

import math

import pytest

from synth_pdb.coupling import calculate_hn_ha_coupling

# Vuister & Bax 1993 Karplus coefficients (as used in synth_nmr engine)
A_VB = 6.51
B_VB = -1.76
C_VB = 1.60


def karplus_vb(phi_deg: float) -> float:
    """Reference Karplus: theta = phi - 60 deg; J = A*cos^2 + B*cos + C."""
    theta = math.radians(phi_deg - 60.0)
    ct = math.cos(theta)
    return A_VB * ct**2 + B_VB * ct + C_VB


# Analytical test cases derived from the Vuister & Bax parameterisation
_KARPLUS_CASES = [
    (-60.0, karplus_vb(-60.0), "ideal_alpha_helix"),
    (-135.0, karplus_vb(-135.0), "ideal_beta_sheet"),
    (-75.0, karplus_vb(-75.0), "PPII_like"),
    (180.0, karplus_vb(180.0), "extended"),
    (0.0, karplus_vb(0.0), "phi_zero"),
]


@pytest.mark.parametrize("phi_deg,expected,label", _KARPLUS_CASES)
def test_karplus_analytical_values(phi_deg, expected, label):
    """calculate_hn_ha_coupling must match the Vuister & Bax Karplus reference.

    EDUCATIONAL NOTE:
    We call the low-level scalar function (not the structure-level wrapper)
    to test the Karplus implementation in isolation. The tolerance of 0.3 Hz
    reflects the typical measurement precision of modern 3J experiments and
    acceptable parameterisation differences across implementations.
    """
    result = calculate_hn_ha_coupling(phi_deg)
    print(f"\n  {label}: phi={phi_deg}deg  J_ref={expected:.3f} Hz  J_calc={result:.3f} Hz")
    assert abs(result - expected) < 0.3, (
        f"{label}: got {result:.3f} Hz, expected {expected:.3f} Hz (tol 0.3 Hz)"
    )


def test_alpha_helix_coupling_is_small():
    """Helical residues (phi ~ -60deg) must have 3J < 5 Hz.

    SCIENTIFIC BASIS:
    In ideal alpha-helices, phi ~ -60deg. This places the H-N-Calpha-H dihedral at
    theta ~ -120deg, where cos^2(-120deg) ~ 0.25 - yielding small J values.
    Measured values for helical proteins routinely fall in the 3-5 Hz range.
    """
    j = calculate_hn_ha_coupling(-60.0)
    assert j < 5.0, f"alpha-helix coupling should be < 5 Hz, got {j:.3f} Hz"
    assert j > 2.0, f"alpha-helix coupling should be > 2 Hz (non-trivial), got {j:.3f} Hz"


def test_beta_sheet_coupling_is_large():
    """Sheet residues (phi ~ -135deg) must have 3J > 7 Hz.

    SCIENTIFIC BASIS:
    Extended beta-sheet conformations have phi ~ -135deg, placing the dihedral
    near 180deg. cos^2(180deg) = 1, giving the maximum Karplus value.
    Measured values for beta-sheet proteins are consistently 8-10 Hz.
    """
    j = calculate_hn_ha_coupling(-135.0)
    assert j > 7.0, f"beta-sheet coupling should be > 7 Hz, got {j:.3f} Hz"
    assert j < 12.0, f"beta-sheet coupling should be < 12 Hz (physical max), got {j:.3f} Hz"


def test_coupling_physically_bounded():
    """3J(HN-HA) must stay within physical bounds [0, 12] Hz for all phi."""
    for phi in range(-180, 181, 5):
        j = calculate_hn_ha_coupling(float(phi))
        assert 0.0 <= j <= 12.0, (
            f"Coupling {j:.3f} Hz at phi={phi}deg is outside physical bounds [0, 12] Hz"
        )


def test_coupling_periodicity():
    """Karplus equation must be periodic with period 360deg in phi."""
    for phi in [-170.0, -90.0, 0.0, 45.0, 120.0]:
        j1 = calculate_hn_ha_coupling(phi)
        j2 = calculate_hn_ha_coupling(phi + 360.0)
        assert abs(j1 - j2) < 1e-6, (
            f"Karplus not periodic at phi={phi}deg: J({phi})={j1:.4f}, J({phi + 360})={j2:.4f}"
        )


def test_helix_sheet_coupling_ordering():
    """beta-sheet J must be substantially larger than alpha-helix J.

    This validates the directionality of the implementation - not just that
    both lie in range, but that the qualitative ordering matches experiment.
    """
    j_helix = calculate_hn_ha_coupling(-60.0)
    j_sheet = calculate_hn_ha_coupling(-135.0)
    assert j_sheet > j_helix + 3.0, (
        f"Sheet J ({j_sheet:.2f} Hz) should exceed helix J ({j_helix:.2f} Hz) by > 3 Hz"
    )
