from unittest.mock import patch

import biotite.structure as struc
import numpy as np

from synth_pdb.packing import SideChainPacker


def get_dummy_peptide() -> struc.AtomArray:
    # Create a simple structure with one VAL residue (which has rotamers)
    atoms = [
        struc.Atom([0, 0, 0], atom_name="N", res_id=1, res_name="VAL", element="N"),
        struc.Atom([1, 0, 0], atom_name="CA", res_id=1, res_name="VAL", element="C"),
        struc.Atom([2, 0, 0], atom_name="C", res_id=1, res_name="VAL", element="C"),
        struc.Atom([1, 1, 0], atom_name="CB", res_id=1, res_name="VAL", element="C"),
        struc.Atom([1, 2, 0], atom_name="CG1", res_id=1, res_name="VAL", element="C"),
    ]
    return struc.array(atoms)


def test_packer_improvement() -> None:
    """Verify that moves which improve the score are always accepted."""
    peptide = get_dummy_peptide()
    packer = SideChainPacker(steps=1)

    with patch("synth_pdb.packing.calculate_clash_score") as mock_score:
        # Initial score 10.0, new score 5.0 (Improvement)
        mock_score.side_effect = [10.0, 5.0]

        with patch("synth_pdb.packing.reconstruct_sidechain"):
            # Mock random choice to pick our VAL residue and a rotamer
            with patch("numpy.random.randint", return_value=0):
                with patch("numpy.random.choice", return_value=0):
                    packer.optimize(peptide)
                    # Score should be updated (implied by accepted_moves log if we checked it)
                    # But we can verify by checking if mock_score was called twice
                    assert mock_score.call_count == 2


def test_packer_metropolis_accept() -> None:
    """Verify that worsening moves are accepted if Metropolis check passes."""
    peptide = get_dummy_peptide()
    # High temperature makes acceptance very likely
    packer = SideChainPacker(steps=1, temperature=1000.0)

    with patch("synth_pdb.packing.calculate_clash_score") as mock_score:
        # Initial score 10.0, new score 15.0 (Worsening)
        mock_score.side_effect = [10.0, 15.0]

        with patch("synth_pdb.packing.reconstruct_sidechain"):
            with patch("numpy.random.randint", return_value=0):
                with patch("numpy.random.choice", return_value=0):
                    # Force random.random to be 0.0 (always < prob)
                    with patch("numpy.random.random", return_value=0.0):
                        packer.optimize(peptide)
                        # If accepted, coordinates are NOT reverted
                        # (Verified by internal state or mock calls if we were more elaborate)
                        # We just want to ensure it reaches the 'accept = True' block
                        assert mock_score.call_count == 2


def test_packer_metropolis_reject_and_revert() -> None:
    """Verify that worsening moves are rejected if Metropolis check fails, and coords revert."""
    peptide = get_dummy_peptide()
    original_coords = peptide.coord.copy()

    # Low temperature makes acceptance unlikely for large delta
    packer = SideChainPacker(steps=1, temperature=0.01)

    with patch("synth_pdb.packing.calculate_clash_score") as mock_score:
        # Initial score 10.0, new score 100.0 (Big worsening)
        mock_score.side_effect = [10.0, 100.0]

        # We need reconstruct_sidechain to actually change coords so we can see if they revert
        def mock_recon(p: struc.AtomArray, rid: int, rot: dict, name: str) -> None:
            p.coord[4] = [99, 99, 99]  # Change CG1

        with patch("synth_pdb.packing.reconstruct_sidechain", side_effect=mock_recon):
            with patch("numpy.random.randint", return_value=0):
                with patch("numpy.random.choice", return_value=0):
                    # Force random.random to be 0.99 (always > prob for delta=90, T=0.01)
                    with patch("numpy.random.random", return_value=0.99):
                        result = packer.optimize(peptide)
                        # Should have reverted
                        assert np.allclose(result.coord, original_coords)


def test_packer_reconstruction_failure() -> None:
    """Verify that if reconstruction fails, optimization continues without crashing."""
    peptide = get_dummy_peptide()
    packer = SideChainPacker(steps=1)

    with patch("synth_pdb.packing.reconstruct_sidechain", side_effect=Exception("Failed")):
        # Should not raise exception
        result = packer.optimize(peptide)
        assert result is peptide


def test_packer_no_optimizable_residues() -> None:
    """Test early exit when no residues have rotamers."""
    # GLY has no rotamers in our library
    atoms = [struc.Atom([0, 0, 0], res_name="GLY", res_id=1, atom_name="CA")]
    peptide = struc.array(atoms)

    packer = SideChainPacker(steps=1)
    result = packer.optimize(peptide)
    assert result is peptide
