import pytest
from synth_pdb.generator import generate_pdb_content
import biotite.structure.io.pdb as pdb
import io


def test_sec_physics_minimization():
    """Test that SEC residues are correctly handled during energy minimization.

    This verifies:
    1. No crash during OpenMM minimization (SEC -> CYS mapping).
    2. SEC identity is restored in the final PDB.
    3. SE atom name is restored.
    4. SEC is treated as ATOM (not HETATM).
    """
    sequence = "ALA-SEC-ALA"

    # This triggers the physics engine path
    pdb_content = generate_pdb_content(
        sequence_str=sequence, minimize_energy=True, forcefield="amber14-all.xml"
    )

    # 1. Check if SEC string is in the PDB
    assert "SEC" in pdb_content
    assert "SE" in pdb_content

    # 2. Parse with biotite to verify structure
    pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
    struct = pdb_file.get_structure(model=1)

    # Check residue names
    res_names = struct.res_name[struct.atom_name == "CA"]
    assert list(res_names) == ["ALA", "SEC", "ALA"]

    # 3. Check for SE atom in the SEC residue
    sec_atoms = struct[struct.res_name == "SEC"]
    assert "SE" in sec_atoms.atom_name
    assert "SG" not in sec_atoms.atom_name

    # 4. Check ATOM vs HETATM
    # Biotite hetero flag should be False for SEC now
    sec_hetero = sec_atoms.hetero
    assert not any(sec_hetero), "SEC should be ATOM, not HETATM"


def test_sec_random_exclusion():
    """Verify that SEC is NOT picked during random generation (as it is non-standard)."""
    from synth_pdb.data import STANDARD_AMINO_ACIDS

    assert "SEC" not in STANDARD_AMINO_ACIDS

    # Generate many random peptides and ensure SEC never appears
    for _ in range(10):
        pdb_content = generate_pdb_content(length=20, conformation="random")
        assert "SEC" not in pdb_content


def test_sec_sasa_s2_calculation():
    """Verify that SEC doesn't break S2/SASA prediction."""
    sequence = "GLY-SEC-GLY"

    # This triggers predict_order_parameters which calls SASA
    pdb_content = generate_pdb_content(
        sequence_str=sequence, minimize_energy=False  # Test the path in generator.py directly
    )

    # If we reached here without "SASA calculation failed" error in logs (which we can't easily check here),
    # but we can verify that B-factors are generated (not 0.0)
    bfactors = []
    for line in pdb_content.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            bfactor_str = line[60:66].strip()
            if bfactor_str:
                bfactors.append(float(bfactor_str))

    # B-factors should be non-zero and realistic
    assert len(bfactors) > 0
    assert all(b > 0 for b in bfactors)
    assert "SEC" in pdb_content
