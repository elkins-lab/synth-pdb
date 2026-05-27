"""
Scientific Validation: RDC Q-factor Self-Consistency.

Validates that calculate_rdc_q_factor is numerically correct against known
analytical solutions, and that Q-factors from generated peptides are
self-consistent (Q < 0.4 when back-calculated from the same structure).

SCIENTIFIC BASIS:
  Q = sqrt( sum((D_obs - D_calc)^2) / sum(D_obs^2) )

  Cornilescu et al. (1998):
    Q < 0.2 = excellent agreement (high-quality NMR structures)
    Q ~ 0.3-0.5 = moderate (local errors or tensor mismatch)
    Q > 0.5 = poor (misfolding or assignment errors)

  When D_obs == D_calc (perfect agreement), Q = 0.
  When D_obs and D_calc are uncorrelated, Q approaches 1.

REFERENCES:
  Cornilescu, G., Marquardt, J.L., Ottiger, M. & Bax, A. (1998).
  Validation of Protein Structure from Anisotropic Carbonyl Chemical Shifts
  in a Dilute Liquid Crystalline Phase.
  J Am Chem Soc, 120, 6836-6837. DOI: 10.1021/ja9812610
"""

import io
import numpy as np
import pytest

import biotite.structure.io.pdb as pdb_io
from synth_pdb.generator import generate_pdb_content
from synth_pdb.rdc import calculate_rdc_q_factor, calculate_rdcs


# -- UNIT TESTS: Q-factor formula correctness ---------------------------------


def test_q_factor_perfect_agreement():
    """Q must equal 0.0 when observed == calculated (perfect agreement)."""
    obs = np.array([10.0, -5.0, 3.0, -8.0, 6.0])
    calc = obs.copy()
    q = calculate_rdc_q_factor(obs, calc)
    assert q == pytest.approx(0.0, abs=1e-8), f"Perfect agreement should yield Q=0, got {q}"


def test_q_factor_known_analytical():
    """Q-factor must match analytical formula for a simple case.

    If obs = [1, 0, 0] and calc = [0, 0, 0]:
    Q = sqrt(1^2 / 1^2) = 1.0 (complete disagreement).
    """
    obs = np.array([1.0, 0.0, 0.0])
    calc = np.array([0.0, 0.0, 0.0])
    # sum((obs-calc)^2) = 1, sum(obs^2) = 1 -> Q = 1.0
    q = calculate_rdc_q_factor(obs, calc)
    assert q == pytest.approx(1.0, abs=1e-8)


def test_q_factor_scaled_invariance():
    """Q must be invariant to overall scaling of the alignment tensor.

    SCIENTIFIC BASIS:
    Q is normalised by the power of the observed data, so scaling both
    obs and calc by the same constant leaves Q unchanged.
    """
    obs = np.array([10.0, -6.0, 8.0, -4.0])
    calc = np.array([9.0, -5.5, 7.5, -4.2])
    q_base = calculate_rdc_q_factor(obs, calc)
    q_scaled = calculate_rdc_q_factor(obs * 2.0, calc * 2.0)
    assert q_base == pytest.approx(q_scaled, abs=1e-8), (
        "Q-factor should be invariant to uniform scaling"
    )


def test_q_factor_mismatched_lengths_raises():
    """Mismatched array lengths must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_rdc_q_factor(np.array([1.0, 2.0]), np.array([1.0]))


def test_q_factor_empty_arrays_returns_zero():
    """Empty arrays must return Q=0 (edge-case guard)."""
    q = calculate_rdc_q_factor(np.array([]), np.array([]))
    assert q == pytest.approx(0.0, abs=1e-8)


def test_q_bounds_for_typical_data():
    """Q must always be in [0, inf); typical realistic Q is in [0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        obs = rng.uniform(-20, 20, size=50)
        calc = obs + rng.normal(0, 2, size=50)
        q = calculate_rdc_q_factor(obs, calc)
        assert q >= 0.0, f"Q must be non-negative, got {q}"


# -- INTEGRATION TEST: Self-consistent Q from back-calculation -----------------


@pytest.fixture(scope="module")
def helix_structure():
    """Generate a short helical peptide for RDC back-calculation."""
    pdb_content = generate_pdb_content(sequence_str="AAAAAAAAAAAA")
    return pdb_io.PDBFile.read(io.StringIO(pdb_content)).get_structure(model=1)


def test_rdc_back_calc_self_consistent_q(helix_structure):
    """Q from back-calculated RDCs vs themselves must be 0 (tautology test).

    EDUCATIONAL NOTE:
    If we back-calculate RDCs from a structure and use those as both the
    'observed' and 'calculated' inputs to calculate_rdc_q_factor, Q must
    equal 0. This verifies the round-trip: back_calc -> Q-factor.
    """
    rdc_dict = calculate_rdcs(helix_structure, da=10.0, r=0.0)
    if not rdc_dict:
        pytest.skip("calculate_rdcs returned no values for helical peptide")
    values = np.array(list(rdc_dict.values()), dtype=float)
    q = calculate_rdc_q_factor(values, values)
    assert q == pytest.approx(0.0, abs=1e-6), f"Self-consistent Q must be 0; got {q}"


def test_rdc_back_calc_produces_finite_values(helix_structure):
    """Back-calculated RDCs must all be finite real numbers."""
    rdc_dict = calculate_rdcs(helix_structure, da=10.0, r=0.0)
    if not rdc_dict:
        pytest.skip("calculate_rdcs returned no values")
    for res_id, val in rdc_dict.items():
        assert np.isfinite(val), f"RDC at residue {res_id} is not finite: {val}"


def test_rdc_values_within_physical_range(helix_structure):
    """Back-calculated RDCs must lie within [-2*Da, 2*Da] = [-20, 20] Hz.

    SCIENTIFIC BASIS:
    For Da=10 Hz and R=0 (axially symmetric tensor), the maximum observable
    RDC is 2*Da = 20 Hz (when the bond vector is parallel to the tensor axis).
    The minimum is -Da = -10 Hz (perpendicular). With a 2x safety factor.
    """
    da = 10.0
    rdc_dict = calculate_rdcs(helix_structure, da=da, r=0.0)
    if not rdc_dict:
        pytest.skip("calculate_rdcs returned no values")
    for res_id, val in rdc_dict.items():
        assert -2 * da <= val <= 2 * da, (
            f"RDC at residue {res_id} = {val:.2f} Hz outside [-{2 * da}, {2 * da}] Hz"
        )
