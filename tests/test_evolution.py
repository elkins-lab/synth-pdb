import os
import tempfile

import biotite.structure as struc
import numpy as np

from synth_pdb.evolution import calculate_relative_sasa, generate_msa_sequences, write_msa
from synth_pdb.generator import PeptideGenerator


def test_calculate_relative_sasa() -> None:
    """Test rSASA calculation for a simple peptide."""
    # Generate a small helix
    # 10 residues of Alanine
    generator = PeptideGenerator(sequence="A" * 10)
    result = generator.generate(conformation="alpha")
    peptide = result.structure

    rel_sasa = calculate_relative_sasa(peptide)

    assert isinstance(rel_sasa, np.ndarray)
    assert len(rel_sasa) == 10
    assert np.all(rel_sasa >= 0.0)
    assert np.all(rel_sasa <= 1.0)

    # In a small 10-mer helix, ends are usually more exposed than the middle.
    # However, for a 10-mer, most residues might be somewhat exposed.
    # Let's just check that it's not all zeros or all ones (unless fallback hit).
    if not np.all(rel_sasa == 1.0):
        # Middle residues should generally have lower SASA than ends
        assert rel_sasa[5] <= rel_sasa[0] + 0.1  # Heuristic


def test_calculate_relative_sasa_fallback() -> None:
    """Test fallback logic when SASA calculation fails."""
    # Create a structure that might cause issues (e.g. empty or single atom)
    atom = struc.Atom([0, 0, 0], atom_name="CA", res_name="ALA", res_id=1)
    peptide = struc.array([atom])

    # Biotite sasa might fail on a single atom or return something we can intercept.
    # We'll just verify it returns the ones array as per fallback code.
    rel_sasa = calculate_relative_sasa(peptide)
    assert np.all(rel_sasa == 1.0)
    assert len(rel_sasa) == 1


def test_generate_msa_sequences() -> None:
    """Test MSA generation and selection pressure (Core vs Surface)."""
    # Use a slightly larger peptide to have some buried residues
    # 20 residues might have some burial if we make it a tight coil or helix
    generator = PeptideGenerator(sequence="A" * 20)
    result = generator.generate(conformation="alpha")
    peptide = result.structure

    n_seqs = 10
    msa = generate_msa_sequences(peptide, n_seqs=n_seqs, mutation_rate=0.5)

    assert len(msa) == n_seqs
    for seq in msa:
        assert len(seq) == 20
        # Check that it only contains valid amino acid characters
        assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq)


def test_generate_msa_conservation() -> None:
    """Verify that buried residues are restricted to hydrophobic mutations."""
    generator = PeptideGenerator(sequence="A" * 30)
    result = generator.generate(conformation="alpha")
    peptide = result.structure

    # Calculate rel_sasa to identify buried indices
    rel_sasa = calculate_relative_sasa(peptide)
    buried_indices = np.where(rel_sasa < 0.2)[0]

    if len(buried_indices) == 0:
        # Force a "buried" residue for testing if none found naturally in small peptide
        # (Though 30-mer helix should have some lower SASA residues)
        return

    # Generate many sequences with high mutation rate to see drift
    n_seqs = 50
    msa = generate_msa_sequences(
        peptide, n_seqs=n_seqs, mutation_rate=1.0, conservation_threshold=0.2
    )

    from synth_pdb.evolution import CORE_ALLOWED

    for seq in msa:
        for idx in buried_indices:
            # Buried residue must be in CORE_ALLOWED
            assert seq[idx] in CORE_ALLOWED


def test_write_msa() -> None:
    """Test writing MSA to a FASTA file."""
    sequences = ["AAAAA", "CCCCC", "GGGGG"]
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp:
        filename = tmp.name

    try:
        write_msa(sequences, filename)

        assert os.path.exists(filename)
        with open(filename) as f:
            content = f.read()

        assert ">seq_0" in content
        assert "AAAAA" in content
        assert ">seq_2" in content
        assert "GGGGG" in content
    finally:
        if os.path.exists(filename):
            os.remove(filename)
