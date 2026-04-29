"""
Scientific Validation: SAXS Rg Calibration and Profile Shape.

Validates Rg physical bounds and Guinier-regime monotonicity.

REFERENCES:
  Guinier, A. (1939). Ann Phys (Paris), 12, 161-237.
  Millett et al. (2002). Adv Protein Chem, 62, 241-262.
"""

import io
import numpy as np
import pytest

import biotite.structure.io.pdb as pdb_io
from synth_pdb.generator import generate_pdb_content
from synth_pdb.saxs import calculate_radius_of_gyration, calculate_saxs_profile

UBQ_SEQUENCE = "MQIFVKTLTGKTITLEVEPS"


@pytest.fixture(scope="module")
def ubq_struct():
    pdb_content = generate_pdb_content(sequence_str=UBQ_SEQUENCE)
    return pdb_io.PDBFile.read(io.StringIO(pdb_content)).get_structure(model=1)


def test_rg_positive_and_finite(ubq_struct):
    """Rg must be a positive, finite real number."""
    rg = calculate_radius_of_gyration(ubq_struct)
    assert np.isfinite(rg), f"Rg is not finite: {rg}"
    assert rg > 0.0


def test_rg_physically_sensible_for_20_residue_peptide(ubq_struct):
    """Rg for a 20-residue peptide must lie in [5, 30] A.

    SCIENTIFIC BASIS:
    Flory scaling: Rg ~ 3.0 * N^0.38 A. For N=20: ~7.8 A (compact globule).
    Extended conformation upper bound ~20 A. Bounds [5, 30] A cover all cases.
    """
    rg = calculate_radius_of_gyration(ubq_struct)
    print(f"\n  Rg (20-residue fragment) = {rg:.2f} A")
    assert 5.0 <= rg <= 30.0, f"Rg {rg:.2f} A outside physical bounds [5, 30] A"


def test_saxs_profile_shape(ubq_struct):
    """q and I(q) arrays must be equal-length with correct endpoints."""
    q, intensity = calculate_saxs_profile(ubq_struct, q_max=0.3, n_points=31)
    assert q.shape == intensity.shape
    assert len(q) == 31
    assert q[0] == pytest.approx(0.0, abs=1e-6)
    assert q[-1] == pytest.approx(0.3, abs=1e-3)


def test_saxs_intensity_positive(ubq_struct):
    """All I(q) must be positive (physical requirement)."""
    _, intensity = calculate_saxs_profile(ubq_struct, q_max=0.3, n_points=31)
    assert np.all(intensity > 0), f"Negative I(q) found; min = {intensity.min():.4g}"


def test_saxs_low_q_monotonic_decrease(ubq_struct):
    """I(q) must decrease monotonically in Guinier regime (q < 0.08 A^-1).

    SCIENTIFIC BASIS:
    Guinier: I(q) = I(0) * exp(-q^2 * Rg^2 / 3), strictly decreasing.
    """
    q, intensity = calculate_saxs_profile(ubq_struct, q_max=0.3, n_points=61)
    mask = q <= 0.08
    i_low = intensity[mask]
    if len(i_low) < 3:
        pytest.skip("Insufficient q-points in Guinier regime")
    diffs = np.diff(i_low)
    assert np.all(diffs <= 0), "I(q) not monotonically decreasing at q < 0.08 A^-1"


def test_saxs_guinier_rg_consistent_with_direct(ubq_struct):
    """Guinier-fitted Rg must agree with direct Rg within 30%."""
    rg_direct = calculate_radius_of_gyration(ubq_struct)
    q, intensity = calculate_saxs_profile(ubq_struct, q_max=0.3, n_points=61)
    q_max_g = min(1.3 / rg_direct, 0.1)
    mask = (q > 1e-3) & (q <= q_max_g)
    q_g, i_g = q[mask], intensity[mask]
    if len(q_g) < 3:
        pytest.skip("Too few points in Guinier region for fit")
    coeffs = np.polyfit(q_g**2, np.log(i_g), 1)
    rg_guinier = np.sqrt(-3.0 * coeffs[0])
    print(f"\n  Rg direct={rg_direct:.2f} A, Guinier={rg_guinier:.2f} A")
    assert np.isfinite(rg_guinier) and rg_guinier > 0
    assert (
        abs(rg_guinier - rg_direct) / rg_direct < 0.30
    ), f"Guinier Rg ({rg_guinier:.2f}) vs direct ({rg_direct:.2f}) differ > 30%"
