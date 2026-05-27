"""TDD Tests for synth_pdb.rdc shim module.

These tests are written BEFORE the implementation (test-first / TDD).
They validate that:
  1. The shim is importable from synth_pdb.rdc.
  2. The exported calculate_rdcs function produces physically correct values.
  3. Edge cases (Proline, missing H, zero Da) are handled gracefully.

Evidence basis for expected values:
--------------------------------------
The RDC formula for a backbone N-H vector is (Tjandra & Bax, 1997, Science 278:1111):

    D(theta, phi) = Da * [(3*cos^2theta - 1) + (3/2)*R*sin^2theta*cos(2phi)]

where:
  - Da  = axial component of the alignment tensor (Hz)
  - R   = rhombicity of the alignment tensor (dimensionless, 0 <= R <= 2/3)
  - theta   = polar angle of the N-H bond vector with respect to the principal (Z) axis
  - phi   = azimuthal angle of the N-H bond vector in the XY plane

Reference:
  Tjandra, N. & Bax, A. (1997). Direct measurement of distances and angles in
  biomolecules by NMR in a dilute liquid crystalline medium. Science, 278, 1111-1114.
  DOI: 10.1126/science.278.5340.1111

Additional validation context:
  Prestegard, J., Al-Hashimi, H. & Tolman, J. (2000). NMR structures of biomolecules
  using field oriented media and residual dipolar couplings.
  Q Rev Biophys, 33, 371-424. DOI: 10.1017/S0033583500003656
"""

import biotite.structure as struc
import pytest


class TestRDCShimImport:
    """Verify the shim module is importable and exports the correct interface."""

    def test_rdc_shim_importable(self) -> None:
        """The synth_pdb.rdc module must exist and expose calculate_rdcs.

        EDUCATIONAL NOTE - Why shims?
        ==============================
        synth-pdb acts as the structure-generation layer; synth-nmr is the
        NMR-observable computation layer. Rather than duplicating code, synth-pdb
        exposes synth-nmr's functions via thin re-export 'shim' modules, preserving
        a clean API for downstream users who only install synth-pdb.
        """
        # This import will FAIL until synth_pdb/rdc.py is created - that is expected
        # for the Red phase of Red-Green-Refactor (TDD).
        from synth_pdb.rdc import calculate_rdcs  # noqa: F401

        assert callable(calculate_rdcs)

    def test_rdc_all_exports(self) -> None:
        """__all__ must include calculate_rdcs so 'from synth_pdb.rdc import *' works."""
        import synth_pdb.rdc as rdc_module

        assert hasattr(rdc_module, "__all__"), "rdc.py must define __all__"
        assert "calculate_rdcs" in rdc_module.__all__


class TestRDCPhysics:
    """Physics-grounded unit tests for the RDC shim.

    Each test constructs a minimal biotite AtomArray with a precisely placed N-H
    pair, computes the expected RDC analytically from the formula, and checks that
    the shim returns the same value.

    Alignment tensor parameters used throughout:
      Da = 10.0 Hz  (typical mid-range; real proteins: 5-25 Hz)
      R  = 0.5      (moderate rhombicity; R=0 -> axially symmetric tensor)
    """

    @pytest.fixture
    def da(self) -> float:
        return 10.0

    @pytest.fixture
    def R(self) -> float:  # noqa: N802
        return 0.5

    def _make_nh_structure(
        self, n_coord: list, h_coord: list, res_name: str = "GLY"
    ) -> "struc.AtomArray":
        """Helper: creates a minimal structure with one N and one H atom."""
        n_atom = struc.Atom(
            coord=list(n_coord),
            atom_name="N",
            element="N",
            res_id=1,
            res_name=res_name,
            chain_id="A",
        )
        h_atom = struc.Atom(
            coord=list(h_coord),
            atom_name="H",
            element="H",
            res_id=1,
            res_name=res_name,
            chain_id="A",
        )
        return struc.array([n_atom, h_atom])

    def test_z_axis_aligned_vector(self, da: float, R: float) -> None:  # noqa: N803
        """A vector perfectly aligned with the Z-axis (principal axis) gives RDC = 2*Da.

        Derivation (Tjandra & Bax, 1997):
          theta = 0deg -> cos theta = 1, sin theta = 0
          Rhombic term = (3/2)*R*0*cos(2phi) = 0
          RDC = Da * (3*1^2 - 1) = 2*Da

        With Da = 10 Hz -> expected RDC = 20.0 Hz.
        This is also the maximum possible RDC for an axially symmetric tensor.
        """
        from synth_pdb.rdc import calculate_rdcs

        structure = self._make_nh_structure([0, 0, 0], [0, 0, 1.02])
        rdcs = calculate_rdcs(structure, da=da, r=R)

        expected = 2 * da  # = 20.0 Hz
        assert 1 in rdcs
        assert rdcs[1] == pytest.approx(expected, abs=1e-2)

    def test_x_axis_aligned_vector(self, da: float, R: float) -> None:  # noqa: N803
        """A vector along the X-axis gives RDC = Da*(-1 + 1.5*R).

        Derivation:
          theta = 90deg, phi = 0deg
          cos theta = 0, sin^2theta = 1, cos(2phi) = cos(0deg) = 1
          RDC = Da * [(3*0 - 1) + 1.5*R*1*1]
              = Da * (-1 + 1.5R)

        With Da=10 Hz, R=0.5 -> -1 + 0.75 = -0.25 -> RDC = -2.5 Hz.
        Negative RDCs are perfectly physical: they simply mean the N-H bond
        is nearly perpendicular to the alignment axis.
        """
        from synth_pdb.rdc import calculate_rdcs

        structure = self._make_nh_structure([0, 0, 0], [1.02, 0, 0])
        rdcs = calculate_rdcs(structure, da=da, r=R)

        expected = da * (-1 + 1.5 * R)  # = -2.5 Hz
        assert rdcs[1] == pytest.approx(expected, abs=1e-2)

    def test_y_axis_aligned_vector(self, da: float, R: float) -> None:  # noqa: N803
        """A vector along the Y-axis gives RDC = Da*(-1 - 1.5*R).

        Derivation:
          theta = 90deg, phi = 90deg
          cos(2phi) = cos(180deg) = -1
          RDC = Da * [(-1) + 1.5*R*1*(-1)]
              = Da * (-1 - 1.5R)

        With Da=10, R=0.5 -> RDC = -17.5 Hz.
        This is the minimum possible RDC for this tensor.
        """
        from synth_pdb.rdc import calculate_rdcs

        structure = self._make_nh_structure([0, 0, 0], [0, 1.02, 0])
        rdcs = calculate_rdcs(structure, da=da, r=R)

        expected = da * (-1 - 1.5 * R)  # = -17.5 Hz
        assert rdcs[1] == pytest.approx(expected, abs=1e-2)

    def test_rdc_value_in_physical_range(self, da: float, R: float) -> None:  # noqa: N803
        """RDC values must fall within [Da*(-1-1.5R), 2*Da].

        Physical constraint (Prestegard et al., 2000):
          The maximum RDC occurs when theta=0 -> 2*Da.
          The minimum occurs when theta=90deg, phi=90deg -> Da*(-1 - 1.5R).
        """
        from synth_pdb.rdc import calculate_rdcs

        # Use a diagonal vector (not on any axis)
        structure = self._make_nh_structure([0, 0, 0], [0.59, 0.59, 0.59])
        rdcs = calculate_rdcs(structure, da=da, r=R)

        if rdcs:
            val = rdcs[1]
            rdc_max = 2 * da
            rdc_min = da * (-1 - 1.5 * R)
            assert rdc_min <= val <= rdc_max, (
                f"RDC={val:.2f} Hz outside theoretical range [{rdc_min:.2f}, {rdc_max:.2f}] Hz"
            )


class TestRDCEdgeCases:
    """Validate graceful handling of biologically special cases."""

    def test_proline_skipped(self) -> None:
        """Proline (PRO) has no backbone amide proton due to its cyclic side-chain
        linking the nitrogen: the N atom forms a tertiary amine with no H attached.

        EDUCATIONAL NOTE - Proline as a secondary amine:
        =================================================
        In all other amino acids, the backbone nitrogen is a primary amine (NH).
        In Proline, the side-chain delta carbon bonds back to the backbone nitrogen,
        forming a five-membered pyrrolidine ring and making the N a secondary amine
        with no exchangeable H. This means:
          - No backbone amide proton  -> no 1DNH RDC
          - No backbone NH peak in HSQC spectra
          - Important structural constraint: cis/trans isomerism of the Xaa-Pro bond

        Reference: Richardson, J.S. (1981). The anatomy and taxonomy of protein structure.
        Adv Protein Chem, 34, 167-339. DOI: 10.1016/S0065-3233(08)60520-3
        """
        from synth_pdb.rdc import calculate_rdcs

        pro_n = struc.Atom(
            [0, 0, 0], atom_name="N", element="N", res_id=1, res_name="PRO", chain_id="A"
        )
        structure = struc.array([pro_n])
        rdcs = calculate_rdcs(structure, da=10.0, r=0.5)

        assert 1 not in rdcs, "Proline should never appear in the RDC output"

    def test_missing_amide_h_skipped(self) -> None:
        """A residue with a backbone N but no corresponding H atom is silently skipped.
        This handles truncated structures or non-standard residues gracefully.
        """
        from synth_pdb.rdc import calculate_rdcs

        # N at res 1, H at res 2 (mismatched - no H for res 1)
        n_atom = struc.Atom(
            [0, 0, 0], atom_name="N", element="N", res_id=1, res_name="ALA", chain_id="A"
        )
        h_atom = struc.Atom(
            [0, 0, 1], atom_name="H", element="H", res_id=2, res_name="ALA", chain_id="A"
        )
        structure = struc.array([n_atom, h_atom])
        rdcs = calculate_rdcs(structure, da=10.0, r=0.5)

        assert 1 not in rdcs

    def test_multi_residue_structure(self) -> None:
        """A 3-residue structure (GLY, PRO, ALA) should only produce RDCs for
        non-Proline residues that have both N and H atoms.

        Expected result: res_id {1: float, 3: float}, res 2 (PRO) absent.
        """
        from synth_pdb.rdc import calculate_rdcs

        atoms = [
            struc.Atom(
                [0, 0, 0], atom_name="N", element="N", res_id=1, res_name="GLY", chain_id="A"
            ),
            struc.Atom(
                [0, 0, 1], atom_name="H", element="H", res_id=1, res_name="GLY", chain_id="A"
            ),
            struc.Atom(
                [3, 0, 0], atom_name="N", element="N", res_id=2, res_name="PRO", chain_id="A"
            ),
            struc.Atom(
                [6, 0, 0], atom_name="N", element="N", res_id=3, res_name="ALA", chain_id="A"
            ),
            struc.Atom(
                [6, 0, 1], atom_name="H", element="H", res_id=3, res_name="ALA", chain_id="A"
            ),
        ]
        structure = struc.array(atoms)
        rdcs = calculate_rdcs(structure, da=10.0, r=0.0)  # R=0 -> pure axial

        assert 1 in rdcs, "GLY residue should have an RDC"
        assert 2 not in rdcs, "PRO should be absent (no amide H)"
        assert 3 in rdcs, "ALA residue should have an RDC"
        assert len(rdcs) == 2

    def test_returns_dict(self) -> None:
        """Return type must always be a dictionary, even for edge-case inputs."""
        from synth_pdb.rdc import calculate_rdcs

        n_atom = struc.Atom(
            [0, 0, 0], atom_name="N", element="N", res_id=1, res_name="GLY", chain_id="A"
        )
        h_atom = struc.Atom(
            [0, 0, 1], atom_name="H", element="H", res_id=1, res_name="GLY", chain_id="A"
        )
        structure = struc.array([n_atom, h_atom])
        result = calculate_rdcs(structure, da=10.0, r=0.0)

        assert isinstance(result, dict)
