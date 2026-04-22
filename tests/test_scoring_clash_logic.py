import biotite.structure as struc

from synth_pdb.scoring import calculate_clash_score


def test_calculate_clash_score_no_atoms() -> None:
    """Test score for empty structure."""
    atoms = struc.AtomArray(0)
    assert calculate_clash_score(atoms) == 0.0


def test_calculate_clash_score_single_atom() -> None:
    """Test score for single atom (no neighbors)."""
    atom = struc.Atom([0, 0, 0], res_name="ALA", res_id=1, atom_name="CA", element="C")
    atoms = struc.array([atom])
    assert calculate_clash_score(atoms) == 0.0


def test_calculate_clash_score_intra_residue_exclusion() -> None:
    """Verify that atoms in the same residue do NOT contribute to clash score."""
    # Place two atoms in same residue very close (0.5A)
    # They would normally clash severely if in different residues
    atoms = [
        struc.Atom([0, 0, 0], res_name="ALA", res_id=1, atom_name="N", element="N"),
        struc.Atom([0.5, 0, 0], res_name="ALA", res_id=1, atom_name="CA", element="C"),
    ]
    peptide = struc.array(atoms)
    assert calculate_clash_score(peptide) == 0.0


def test_calculate_clash_score_adjacent_backbone_exclusion() -> None:
    """Verify that adjacent backbone atoms do NOT contribute to clash score."""
    # N(i) and C(i-1) are normally bonded at ~1.33A.
    # If we place them at 1.0A, they are "clashing" by distance but should be excluded.
    atoms = [
        struc.Atom([0, 0, 0], res_name="ALA", res_id=1, atom_name="C", element="C"),
        struc.Atom([1.0, 0, 0], res_name="ALA", res_id=2, atom_name="N", element="N"),
    ]
    peptide = struc.array(atoms)
    assert calculate_clash_score(peptide) == 0.0


def test_calculate_clash_score_actual_clash() -> None:
    """Verify that atoms in different residues DO contribute to clash score when overlapping."""
    # Res 1 CA and Res 10 CA (far apart in sequence, no exclusion)
    # Distance = 1.0A. VdW radii sum for C+C is ~3.4A. 0.8 * 3.4 = 2.72A. 1.0 < 2.72.
    atoms = [
        struc.Atom([0, 0, 0], res_name="ALA", res_id=1, atom_name="CA", element="C"),
        struc.Atom([1.0, 0, 0], res_name="ALA", res_id=10, atom_name="CA", element="C"),
    ]
    peptide = struc.array(atoms)
    score = calculate_clash_score(peptide)
    assert score > 0.0


def test_calculate_clash_score_sidechain_clash_adjacent() -> None:
    """Verify that sidechain atoms in adjacent residues DO clash."""
    # Res 1 CB and Res 2 CB
    # Distance = 1.0A.
    atoms = [
        struc.Atom([0, 0, 0], res_name="ALA", res_id=1, atom_name="CB", element="C"),
        struc.Atom([1.0, 0, 0], res_name="ALA", res_id=2, atom_name="CB", element="C"),
    ]
    peptide = struc.array(atoms)
    score = calculate_clash_score(peptide)
    assert score > 0.0


def test_calculate_clash_score_radii_lookup() -> None:
    """Verify that different elements use appropriate radii."""
    # Sulfur has larger radius (~1.8A) than Carbon (~1.7A)
    # C-C overlap vs S-S overlap at same distance

    dist = 2.0

    # C-C
    c_atoms = [
        struc.Atom([0, 0, 0], res_name="ALA", res_id=1, atom_name="CA", element="C"),
        struc.Atom([dist, 0, 0], res_name="ALA", res_id=10, atom_name="CA", element="C"),
    ]
    score_c = calculate_clash_score(struc.array(c_atoms))

    # S-S
    s_atoms = [
        struc.Atom([0, 0, 0], res_name="CYS", res_id=1, atom_name="SG", element="S"),
        struc.Atom([dist, 0, 0], res_name="CYS", res_id=10, atom_name="SG", element="S"),
    ]
    score_s = calculate_clash_score(struc.array(s_atoms))

    # S has larger radius, so overlap should be larger, score should be higher
    assert score_s > score_c
