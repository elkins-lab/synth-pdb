import io

import biotite.structure.io.pdb as pdb

from synth_pdb.chemical_shifts import predict_chemical_shifts
from synth_pdb.generator import generate_pdb_content
from synth_pdb.rdc import calculate_rdcs


class TestNMRNonStandardMotifs:
    """
    Test suite to ensure NMR feature prediction (Shifts, RDCs)
    is robust for non-standard motifs (D-amino acids, PTMs).
    """

    def test_shift_prediction_completeness(self) -> None:
        """
        Verify that chemical shifts are predicted for all residues,
        including DAL and SEP.
        """
        sequence = "ALA-D-ALA-SEP-ALA"
        pdb_content = generate_pdb_content(sequence_str=sequence, minimize_energy=True)

        f = io.StringIO(pdb_content)
        struc = pdb.PDBFile.read(f).get_structure(model=1)

        # Predict shifts (empirical mode for CI speed/stability)
        shifts = predict_chemical_shifts(struc, use_shiftx2=False)

        # Check that chain 'A' is present
        assert "A" in shifts
        chain_shifts = shifts["A"]

        # Verify all 4 residues have shift entries
        assert set(chain_shifts.keys()) == {1, 2, 3, 4}

        # Verify specific non-standard residue results
        # Res 2 is DAL (D-Ala)
        assert "CA" in chain_shifts[2]
        assert 50.0 < chain_shifts[2]["CA"] < 60.0  # Standard ALA range

        # Res 3 is SEP (Phosphoserine)
        assert "CA" in chain_shifts[3]
        assert 50.0 < chain_shifts[3]["CA"] < 65.0

    def test_rdc_prediction_completeness(self) -> None:
        """
        Verify that RDCs are predicted for all residues,
        including DAL and SEP.
        """
        sequence = "ALA-D-ALA-SEP-ALA"
        pdb_content = generate_pdb_content(sequence_str=sequence, minimize_energy=True)

        f = io.StringIO(pdb_content)
        struc = pdb.PDBFile.read(f).get_structure(model=1)

        rdcs = calculate_rdcs(struc, da=15.0, r=0.1)

        # Res 1 often missing H in some builders, but generate_pdb_content adds caps or H
        # ALA-DAL-SEP-ALA without caps should have N-term NH3+, might only have N-H for 2,3,4
        # Let's check how many were returned.
        assert len(rdcs) >= 3
        assert 2 in rdcs  # DAL
        assert 3 in rdcs  # SEP

    def test_l_d_parity_shifts(self) -> None:
        """
        Compare chemical shifts of L-ALA vs D-ALA in identical geometric context.
        Note: The context won't be EXACTLY identical because mirror images
        invert backbone planes, but they should be close.
        """
        l_pdb = generate_pdb_content(sequence_str="ALA-ALA-ALA", minimize_energy=False)
        d_pdb = generate_pdb_content(sequence_str="ALA-D-ALA-ALA", minimize_energy=False)

        def get_shifts(pdb_str: str) -> dict[int, dict[str, float]]:
            f = io.StringIO(pdb_str)
            s = pdb.PDBFile.read(f).get_structure(model=1)
            return predict_chemical_shifts(s, use_shiftx2=False)["A"]

        l_shifts = get_shifts(l_pdb)
        d_shifts = get_shifts(d_pdb)

        # Res 2 in both should have similar CA/CB shifts
        # (ALA vs DAL)
        for atom in ["CA", "CB", "N"]:
            if atom in l_shifts[2] and atom in d_shifts[2]:
                diff = abs(l_shifts[2][atom] - d_shifts[2][atom])
                # They should be very close (< 0.7 ppm) since we mapped DAL -> ALA
                # and local geometry is just mirrored.
                assert diff < 0.7, f"Large shift difference for {atom}: {diff:.3f}"

    def test_ptm_parity_shifts(self) -> None:
        """Verify SEP shift is similar to SER."""
        ser_pdb = generate_pdb_content(sequence_str="ALA-SER-ALA", minimize_energy=False)
        sep_pdb = generate_pdb_content(sequence_str="ALA-SEP-ALA", minimize_energy=False)

        def get_shifts(pdb_str: str) -> dict[int, dict[str, float]]:
            f = io.StringIO(pdb_str)
            s = pdb.PDBFile.read(f).get_structure(model=1)
            return predict_chemical_shifts(s, use_shiftx2=False)["A"]

        ser_s = get_shifts(ser_pdb)
        sep_s = get_shifts(sep_pdb)

        # Should be similar because SEP is mapped to SER, but allow for minor
        # geometric drift due to phosphate sterics.
        diff_ca = abs(ser_s[2]["CA"] - sep_s[2]["CA"])
        assert diff_ca < 0.7
