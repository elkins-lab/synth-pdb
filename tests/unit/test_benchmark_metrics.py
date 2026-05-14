"""tests/unit/test_benchmark_metrics.py.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for synth_pdb.benchmark_metrics.

Tests verify mathematical correctness of all metric functions:
  - superpose_kabsch: identical structures give RMSD=0 and exact rotation
  - tm_score: identical structures -> 1.0; random structures -> < 0.5
  - lddt: perfect prediction -> 1.0; degraded -> decreasing
  - gdt_ts: perfect prediction -> 1.0
  - shift_rmsd: identical shifts -> 0; offset shifts -> correct value
  - extract_ca_coords: correct parsing of PDB ATOM records
"""

import math

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def helix_coords():
    """Idealized alpha-helix Calpha coordinates for 10 residues.

    Generated analytically using:
        x(i) = r * cos(i * 100deg)
        y(i) = r * sin(i * 100deg)
        z(i) = i * rise_per_residue
    where r=2.3 A and rise=1.5 A/residue (standard helix parameters).
    """
    n = 10
    r = 2.3  # A
    rise = 1.5  # A per residue
    angle_per_res = math.radians(100.0)  # 3.6 residues/turn

    coords = np.array(
        [
            [r * math.cos(i * angle_per_res), r * math.sin(i * angle_per_res), i * rise]
            for i in range(n)
        ],
        dtype=np.float32,
    )
    return coords


@pytest.fixture
def random_coords():
    """Random Calpha coordinates - not a physical protein, just for metric testing."""
    rng = np.random.default_rng(42)
    return (rng.random((10, 3)) * 30.0).astype(np.float32)


@pytest.fixture
def simple_pdb():
    """Minimal PDB string with 5 Calpha atoms for testing extract_ca_coords()."""
    lines = []
    for i in range(1, 6):
        x, y, z = float(i), float(i * 2), float(i * 3)
        line = (
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    " f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        lines.append(line)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# superpose_kabsch
# -----------------------------------------------------------------------------


class TestSuperposekabsch:
    def test_identical_structures_rmsd_zero(self, helix_coords):
        from synth_pdb.benchmark_metrics import superpose_kabsch

        rotated, rmsd = superpose_kabsch(helix_coords, helix_coords)
        assert rmsd < 1e-5, f"RMSD of identical structures should be ~0, got {rmsd}"

    def test_translated_structure_rmsd_zero(self, helix_coords):
        from synth_pdb.benchmark_metrics import superpose_kabsch

        translated = helix_coords + np.array([10.0, -5.0, 3.0])
        _, rmsd = superpose_kabsch(translated, helix_coords)
        assert rmsd < 1e-4, f"Pure translation should give RMSD ~0 after superposition, got {rmsd}"

    def test_rotated_structure_rmsd_zero(self, helix_coords):
        from synth_pdb.benchmark_metrics import superpose_kabsch

        # Rotate 90deg around z-axis
        theta = math.pi / 2
        rot = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        rotated = helix_coords @ rot.T
        _, rmsd = superpose_kabsch(rotated, helix_coords)
        assert rmsd < 1e-4, f"Pure rotation should give RMSD ~0 after superposition, got {rmsd}"

    def test_noisy_structure_positive_rmsd(self, helix_coords):
        from synth_pdb.benchmark_metrics import superpose_kabsch

        rng = np.random.default_rng(0)
        noisy = helix_coords + rng.normal(0, 0.5, helix_coords.shape).astype(np.float32)
        _, rmsd = superpose_kabsch(noisy, helix_coords)
        assert rmsd > 0.0, "Noisy structure should have positive RMSD"
        assert rmsd < 5.0, f"RMSD of mildly noisy structure should be <5 A, got {rmsd}"


# -----------------------------------------------------------------------------
# tm_score
# -----------------------------------------------------------------------------


class TestTMScore:
    def test_identical_structures_score_one(self, helix_coords):
        from synth_pdb.benchmark_metrics import tm_score

        score = tm_score(helix_coords, helix_coords)
        assert (
            abs(score - 1.0) < 1e-4
        ), f"Identical structures should have TM-score ~ 1.0, got {score}"

    def test_score_in_unit_interval(self, helix_coords, random_coords):
        from synth_pdb.benchmark_metrics import tm_score

        for ca1, ca2 in [(helix_coords, helix_coords), (helix_coords, random_coords)]:
            score = tm_score(ca1, ca2)
            assert 0.0 <= score <= 1.0 + 1e-6, f"TM-score {score} outside [0, 1]"

    def test_random_structures_score_below_threshold(self, helix_coords, random_coords):
        from synth_pdb.benchmark_metrics import tm_score

        score = tm_score(random_coords, helix_coords)
        # Two unrelated structures should score well below 0.5
        assert score < 0.5, f"Random vs helix TM-score {score} should be < 0.5"

    def test_translated_structures_score_one(self, helix_coords):
        from synth_pdb.benchmark_metrics import tm_score

        translated = helix_coords + 20.0
        score = tm_score(translated, helix_coords)
        assert abs(score - 1.0) < 1e-4


# -----------------------------------------------------------------------------
# lddt
# -----------------------------------------------------------------------------


class TestLddt:
    def test_identical_structures_lddt_one(self, helix_coords):
        from synth_pdb.benchmark_metrics import lddt

        scores = lddt(helix_coords, helix_coords)
        assert np.allclose(
            scores, 1.0, atol=1e-4
        ), f"Identical structures should have per-residue lDDT = 1.0, got {scores}"

    def test_lddt_output_shape(self, helix_coords):
        from synth_pdb.benchmark_metrics import lddt

        scores = lddt(helix_coords, helix_coords)
        assert scores.shape == (len(helix_coords),)

    def test_lddt_in_unit_interval(self, helix_coords, random_coords):
        from synth_pdb.benchmark_metrics import lddt

        scores = lddt(random_coords, helix_coords)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0 + 1e-6)

    def test_noisy_structure_lddt_below_one(self, helix_coords):
        from synth_pdb.benchmark_metrics import lddt

        rng = np.random.default_rng(1)
        noisy = helix_coords + rng.normal(0, 1.0, helix_coords.shape).astype(np.float32)
        scores = lddt(noisy, helix_coords)
        assert float(np.mean(scores)) < 1.0


# -----------------------------------------------------------------------------
# gdt_ts
# -----------------------------------------------------------------------------


class TestGdtTs:
    def test_identical_structures_gdt_one(self, helix_coords):
        from synth_pdb.benchmark_metrics import gdt_ts

        score = gdt_ts(helix_coords, helix_coords)
        assert abs(score - 1.0) < 1e-4

    def test_gdt_in_unit_interval(self, helix_coords, random_coords):
        from synth_pdb.benchmark_metrics import gdt_ts

        score = gdt_ts(random_coords, helix_coords)
        assert 0.0 <= score <= 1.0 + 1e-6

    def test_perfect_after_translation(self, helix_coords):
        from synth_pdb.benchmark_metrics import gdt_ts

        translated = helix_coords + 5.0
        score = gdt_ts(translated, helix_coords)
        assert abs(score - 1.0) < 1e-4


# -----------------------------------------------------------------------------
# shift_rmsd
# -----------------------------------------------------------------------------


class TestShiftRmsd:
    def test_identical_shifts_rmsd_zero(self):
        from synth_pdb.benchmark_metrics import shift_rmsd

        shifts = {"H": np.array([8.0, 8.1, 8.2]), "C": np.array([120.0, 121.0, 122.0])}
        rmsd = shift_rmsd(shifts, shifts)
        assert rmsd < 1e-6, f"Identical shifts should give RMSD=0, got {rmsd}"

    def test_constant_offset_rmsd(self):
        from synth_pdb.benchmark_metrics import shift_rmsd

        ref = {"H": np.array([8.0, 8.0, 8.0])}
        pred = {"H": np.array([8.1, 8.1, 8.1])}
        rmsd = shift_rmsd(pred, ref)
        assert abs(rmsd - 0.1) < 1e-5, f"Expected RMSD=0.1 ppm, got {rmsd}"

    def test_multi_nucleus_weighted_rmsd(self):
        from synth_pdb.benchmark_metrics import shift_rmsd

        # H shift offset 0.1 ppm, C shift offset 1.0 ppm
        # With weights H=1.0, C=0.25:
        # sq_sum = 1.0*(0.1^2)*3 + 0.25*(1.0^2)*3 = 0.03 + 0.75 = 0.78
        # weight_sum = 1.0*3 + 0.25*3 = 3 + 0.75 = 3.75
        # rmsd = sqrt(0.78/3.75) = sqrt(0.208) ~ 0.4561
        ref = {"H": np.array([8.0, 8.0, 8.0]), "C": np.array([120.0, 120.0, 120.0])}
        pred = {"H": np.array([8.1, 8.1, 8.1]), "C": np.array([121.0, 121.0, 121.0])}
        rmsd = shift_rmsd(pred, ref)
        expected = math.sqrt(0.78 / 3.75)
        assert abs(rmsd - expected) < 1e-4, f"Expected {expected:.4f}, got {rmsd:.4f}"

    def test_missing_nucleus_in_pred_warns_not_raises(self):
        from synth_pdb.benchmark_metrics import shift_rmsd

        ref = {"H": np.array([8.0, 8.1]), "N": np.array([120.0, 121.0])}
        pred = {"H": np.array([8.0, 8.1])}  # N missing
        # Should not raise; N is skipped with a warning
        rmsd = shift_rmsd(pred, ref)
        assert rmsd < 1e-6  # only H contributes; identical -> 0


# -----------------------------------------------------------------------------
# extract_ca_coords
# -----------------------------------------------------------------------------


class TestExtractCaCoords:
    def test_correct_number_of_residues(self, simple_pdb):
        from synth_pdb.benchmark_metrics import extract_ca_coords

        coords = extract_ca_coords(simple_pdb)
        assert coords.shape == (5, 3)

    def test_first_residue_coordinates(self, simple_pdb):
        from synth_pdb.benchmark_metrics import extract_ca_coords

        coords = extract_ca_coords(simple_pdb)
        np.testing.assert_allclose(coords[0], [1.0, 2.0, 3.0], atol=1e-3)

    def test_too_few_residues_raises(self):
        from synth_pdb.benchmark_metrics import extract_ca_coords

        minimal = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C"
        with pytest.raises(ValueError, match="at least 2"):
            extract_ca_coords(minimal)

    def test_real_synth_pdb_structure(self):
        from synth_pdb.benchmark_metrics import extract_ca_coords
        from synth_pdb.generator import generate_pdb_content

        pdb = generate_pdb_content(length=15, conformation="alpha", minimize_energy=False)
        coords = extract_ca_coords(pdb)
        assert coords.shape[0] == 15
        assert coords.shape[1] == 3
