import pytest
from synth_pdb import data


def test_standard_amino_acids():
    """Ensure all standard 20 amino acids are present."""
    assert len(data.STANDARD_AMINO_ACIDS) == 20
    assert "ALA" in data.STANDARD_AMINO_ACIDS
    assert "GLY" in data.STANDARD_AMINO_ACIDS


def test_l_to_d_mapping():
    """Verify L to D mapping consistency."""
    for l_code, d_code in data.L_TO_D_MAPPING.items():
        assert l_code in data.STANDARD_AMINO_ACIDS or l_code == "CYS"
        assert d_code.startswith("D")

    # Glycine should not have a D-form
    assert "GLY" not in data.L_TO_D_MAPPING


def test_rotamer_library_completeness():
    """Check that all standard amino acids have an entry in the rotamer library."""
    for aa in data.STANDARD_AMINO_ACIDS:
        assert aa in data.ROTAMER_LIBRARY


def test_chi_definitions():
    """Verify that chi definitions use valid atom names."""
    valid_atoms = {
        "N",
        "CA",
        "CB",
        "CG",
        "CG1",
        "CG2",
        "CD",
        "CD1",
        "CD2",
        "CE",
        "CE1",
        "CE2",
        "CZ",
        "CZ2",
        "CZ3",
        "CH2",
        "NE",
        "NE1",
        "NE2",
        "NH1",
        "NH2",
        "ND1",
        "ND2",
        "OG",
        "OG1",
        "OD1",
        "OD2",
        "OE1",
        "OE2",
        "OH",
        "SD",
        "SG",
        "NZ",
    }

    for aa, chis in data.AMINO_ACID_CHI_DEFINITIONS.items():
        for chi in chis:
            for atom in chi["atoms"]:
                assert atom in valid_atoms, f"Invalid atom {atom} in {aa} {chi['name']}"


def test_ramachandran_presets():
    """Ensure core secondary structure presets are defined."""
    assert "alpha" in data.RAMACHANDRAN_PRESETS
    assert "beta" in data.RAMACHANDRAN_PRESETS
    assert data.RAMACHANDRAN_PRESETS["alpha"]["phi"] == -57.0


def test_vdw_radii():
    """Check VDW radii for common atoms."""
    assert data.VAN_DER_WAALS_RADII["C"] == 1.70
    assert data.VAN_DER_WAALS_RADII["H"] == 1.20


def test_one_to_three_letter_code():
    """Verify 1-to-3 letter mapping."""
    assert data.ONE_TO_THREE_LETTER_CODE["A"] == "ALA"
    assert data.ONE_TO_THREE_LETTER_CODE["G"] == "GLY"
    assert len(data.ONE_TO_THREE_LETTER_CODE) >= 20


def test_amino_acid_atoms_validity():
    """Check that atomic definitions (if present) are consistent."""
    for aa, atoms in data.AMINO_ACID_ATOMS.items():
        assert aa in data.STANDARD_AMINO_ACIDS
        for atom in atoms:
            assert "name" in atom
            assert "element" in atom
            assert "coords" in atom
            assert len(atom["coords"]) == 3
