from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


def test_physical_energy_validation_tdd(tmp_path):
    """TDD: Verify that PDBValidator can distinguish between low-energy and high-energy structures."""
    # 1. Generate a "Good" structure
    from synth_pdb.physics import EnergyMinimizer
    pdb_raw = generate_pdb_content(sequence_str="AAAAA")

    in_pdb = tmp_path / "in.pdb"
    out_pdb = tmp_path / "out.pdb"
    in_pdb.write_text(pdb_raw)

    engine = EnergyMinimizer()
    success = engine.add_hydrogens_and_minimize(str(in_pdb), str(out_pdb))
    assert success

    pdb_good = out_pdb.read_text()
    val_good = PDBValidator(pdb_content=pdb_good)
    energy_good = val_good.calculate_potential_energy()

    # 2. Generate a "Bad" structure (manually create a clash in PDB string)
    # We take the first two ATOM lines and make their coordinates identical
    lines = pdb_good.splitlines()
    atom_lines = [l for l in lines if l.startswith("ATOM")]

    # "ATOM      1  N   ALA A   1      -0.001   0.005   0.000  1.00  0.00           N"
    # Overlap Atom 2 onto Atom 1
    a1 = atom_lines[0]
    coords_part = a1[30:54]
    a2_clashed = atom_lines[1][:30] + coords_part + atom_lines[1][54:]

    bad_lines = [l for l in lines if not l.startswith("ATOM")]
    bad_lines.insert(1, a1)
    bad_lines.insert(2, a2_clashed)
    # Add rest of atoms
    bad_lines.extend(atom_lines[2:])

    pdb_bad = "\n".join(bad_lines)
    val_bad = PDBValidator(pdb_content=pdb_bad)
    energy_bad = val_bad.calculate_potential_energy()

    print(f"Good Energy: {energy_good:.2f} kJ/mol")
    print(f"Bad Energy: {energy_bad:.2f} kJ/mol")

    assert energy_good < 0
    assert energy_bad > 1e6 or energy_bad == float('inf')

def test_quality_score_integration():
    """TDD: Verify that energy is integrated into a multi-metric quality score."""
    pdb_content = generate_pdb_content(length=10)
    validator = PDBValidator(pdb_content=pdb_content)

    # New method to get a summarized "Evidence-Based Quality Report"
    report = validator.get_quality_report()

    assert "potential_energy_kj_mol" in report
    assert "ramachandran_stats" in report
    assert "is_physically_plausible" in report
