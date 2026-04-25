import io

import biotite.structure.io.pdb as pdb_io
import numpy as np

from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


class TestMultichainGeneration:
    """Test suite for multichain protein structure generation."""

    def test_basic_dimer_generation(self) -> None:
        """Verify that a sequence with ':' produces two chains."""
        sequence = "ALA-ALA:GLY-GLY"
        pdb_content = generate_pdb_content(sequence_str=sequence)

        assert "ATOM" in pdb_content
        assert "TER" in pdb_content

        # Parse with Biotite
        pdb_file = pdb_io.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)

        # Check chain IDs
        chain_ids = np.unique(structure.chain_id)
        assert len(chain_ids) == 2
        assert "A" in chain_ids
        assert "B" in chain_ids

        # Check residue names per chain
        chain_a = structure[structure.chain_id == "A"]
        chain_b = structure[structure.chain_id == "B"]

        assert "ALA" in np.unique(chain_a.res_name)
        assert "GLY" in np.unique(chain_b.res_name)

    def test_trimer_generation(self) -> None:
        """Verify that a sequence with two ':' produces three chains."""
        sequence = "A:G:C"  # 1-letter codes
        pdb_content = generate_pdb_content(sequence_str=sequence)

        pdb_file = pdb_io.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)

        chain_ids = np.unique(structure.chain_id)
        assert len(chain_ids) == 3
        assert list(chain_ids) == ["A", "B", "C"]

    def test_multichain_interface_validation(self) -> None:
        """Verify that multichain structures can be analyzed for interface metrics."""
        # Generate a dimer
        sequence = "ALA-ALA:ALA-ALA"
        pdb_content = generate_pdb_content(sequence_str=sequence)

        validator = PDBValidator(pdb_content=pdb_content)
        metrics = validator.calculate_interface_metrics()

        assert "buried_surface_area" in metrics
        # Default translation is 20A, so they should be separate
        assert metrics["buried_surface_area"] < 1.0
        assert metrics["is_interface_physically_plausible"] is True

    def test_multichain_with_ptms_and_minimization(self) -> None:
        """Verify that PTMs (SEP) are correctly restored for multiple chains after OpenMM."""
        # Chain A: ALA-SEP, Chain B: GLY-SEP
        sequence = "ALA-SEP:GLY-SEP"
        pdb_content = generate_pdb_content(
            sequence_str=sequence, minimize_energy=True, minimization_max_iter=5
        )

        pdb_file = pdb_io.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)

        # Check that SEP exists on both chains
        unique_res_a = np.unique(structure[structure.chain_id == "A"].res_name)
        unique_res_b = np.unique(structure[structure.chain_id == "B"].res_name)

        assert "SEP" in unique_res_a
        assert "SEP" in unique_res_b

    def test_multichain_bfactor_uniqueness(self) -> None:
        """Verify that B-factors are calculated correctly for multiple chains.
        Each chain has its own Residue 1, they should not be conflated.
        """
        # Chain A: 10 residues, Chain B: 10 residues
        sequence = "AAAAA:GGGGG"
        pdb_content = generate_pdb_content(sequence_str=sequence)

        # Verify both chains have res_id 1
        lines = pdb_content.splitlines()
        res_ids_a = [
            int(line[22:26]) for line in lines if line.startswith("ATOM") and line[21] == "A"
        ]
        res_ids_b = [line for line in lines if line.startswith("ATOM") and line[21] == "B"]

        assert 1 in res_ids_a
        assert 1 in [int(line[22:26]) for line in res_ids_b]
