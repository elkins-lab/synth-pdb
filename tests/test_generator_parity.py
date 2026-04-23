import io

import numpy as np
import pytest
from biotite.structure.io.pdb import PDBFile

from synth_pdb.batch_generator import BatchedGenerator
from synth_pdb.generator import generate_pdb_content


class TestGeneratorParity:
    """
    Test suite to ensure parity between standard PeptideGenerator
    and the optimized BatchedGenerator.

    NOTE: Parity is only expected when:
    1. full_atom=True is used (to trigger template superimposition in both).
    2. omega is fixed (e.g. 180.0) or passed as a list.
    3. drift=0.0.
    """

    def test_backbone_parity_alpha(self) -> None:
        """
        Check that both generators produce identical backbone coordinates
        for an alpha-helical sequence.
        """
        sequence = "ALA-GLY-SER-LEU-GLU"
        length = 5
        seed = 42

        # 1. Batched Generator (must use full_atom=True for parity with serial generator)
        bg = BatchedGenerator(sequence, n_batch=1, full_atom=True)
        batch_res = bg.generate_batch(seed=seed, conformation="alpha", drift=0.0)

        # 2. Standard Generator
        omega_list = [180.0] * (length - 1)
        pdb_content = generate_pdb_content(
            sequence_str=sequence,
            conformation="alpha",
            omega_list=omega_list,
            drift=0.0,
            seed=seed,
            minimize_energy=False,
        )

        f = PDBFile.read(io.StringIO(pdb_content))
        struc = f.get_structure(model=1)

        # Verify backbone atoms (N, CA, C, O)
        b_res_indices = np.array(batch_res.residue_indices)
        b_atom_names = np.array(batch_res.atom_names)

        for res_id in range(1, length + 1):
            for atom_name in ["N", "CA", "C", "O"]:
                b_mask = (b_res_indices == res_id) & (b_atom_names == atom_name)
                s_mask = (struc.res_id == res_id) & (struc.atom_name == atom_name)

                b_coord = batch_res.coords[0][b_mask]
                s_coord = struc.coord[s_mask]

                # Check for presence
                assert b_coord.size > 0, f"Atom {atom_name} missing in batch Res{res_id}"
                assert s_coord.size > 0, f"Atom {atom_name} missing in serial Res{res_id}"

                # Compare values. We allow 0.015A tolerance because:
                # - PDB format has only 3 decimal places.
                # - SVD/Kabsch implementations (Biotite vs Custom) might differ slightly.
                np.testing.assert_allclose(b_coord, s_coord, atol=0.015)

    def test_backbone_parity_beta(self) -> None:
        """Check parity for beta-sheet conformation."""
        sequence = "VAL-ILE-TYR-PHE"
        length = 4
        seed = 123

        bg = BatchedGenerator(sequence, n_batch=1, full_atom=True)
        batch_res = bg.generate_batch(seed=seed, conformation="beta", drift=0.0)

        omega_list = [180.0] * (length - 1)
        pdb_content = generate_pdb_content(
            sequence_str=sequence,
            conformation="beta",
            omega_list=omega_list,
            drift=0.0,
            seed=seed,
            minimize_energy=False,
        )

        f = PDBFile.read(io.StringIO(pdb_content))
        struc = f.get_structure(model=1)

        b_res_indices = np.array(batch_res.residue_indices)
        b_atom_names = np.array(batch_res.atom_names)

        for res_id in range(1, length + 1):
            for atom_name in ["N", "CA", "C", "O"]:
                b_mask = (b_res_indices == res_id) & (b_atom_names == atom_name)
                s_mask = (struc.res_id == res_id) & (struc.atom_name == atom_name)

                b_coord = batch_res.coords[0][b_mask]
                s_coord = struc.coord[s_mask]

                np.testing.assert_allclose(b_coord, s_coord, atol=0.015)

    def test_drift_consistency(self) -> None:
        """
        BatchedGenerator should be consistent with itself when using seeds.
        """
        sequence = "ALA-ALA-ALA"
        seed = 42
        drift = 5.0

        bg = BatchedGenerator(sequence, n_batch=1, full_atom=True)
        res1 = bg.generate_batch(seed=seed, drift=drift)
        res2 = bg.generate_batch(seed=seed, drift=drift)

        np.testing.assert_array_equal(res1.coords, res2.coords)

        # Verify that drift actually changed coordinates from zero-drift
        res_no_drift = bg.generate_batch(seed=seed, drift=0.0)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(res1.coords, res_no_drift.coords, atol=1e-5)

    def test_d_amino_parity(self) -> None:
        """Check parity for sequences containing D-amino acids."""
        sequence = "ALA-D-ALA-ALA"
        length = 3
        seed = 42

        bg = BatchedGenerator(sequence, n_batch=1, full_atom=True)
        batch_res = bg.generate_batch(seed=seed, conformation="alpha", drift=0.0)

        omega_list = [180.0] * (length - 1)
        pdb_content = generate_pdb_content(
            sequence_str=sequence,
            conformation="alpha",
            omega_list=omega_list,
            drift=0.0,
            seed=seed,
            minimize_energy=False,
        )

        f = PDBFile.read(io.StringIO(pdb_content))
        struc = f.get_structure(model=1)

        b_res_indices = np.array(batch_res.residue_indices)
        b_atom_names = np.array(batch_res.atom_names)

        for res_id in range(1, length + 1):
            for atom_name in ["N", "CA", "C", "O"]:
                b_mask = (b_res_indices == res_id) & (b_atom_names == atom_name)
                s_mask = (struc.res_id == res_id) & (struc.atom_name == atom_name)

                b_coord = batch_res.coords[0][b_mask]
                s_coord = struc.coord[s_mask]

                np.testing.assert_allclose(b_coord, s_coord, atol=0.015)

    def test_full_atom_parity_ala(self) -> None:
        """Check full-atom parity for ALA (fixed sidechain)."""
        sequence = "ALA-ALA"
        length = 2
        bg = BatchedGenerator(sequence, n_batch=1, full_atom=True)
        batch_res = bg.generate_batch(seed=42, conformation="alpha", drift=0.0)

        omega_list = [180.0] * (length - 1)
        pdb_content = generate_pdb_content(
            sequence_str=sequence,
            conformation="alpha",
            omega_list=omega_list,
            drift=0.0,
            seed=42,
            minimize_energy=False,
        )

        f = PDBFile.read(io.StringIO(pdb_content))
        struc = f.get_structure(model=1)

        b_res_indices = np.array(batch_res.residue_indices)
        b_atom_names = np.array(batch_res.atom_names)

        # Compare ALL heavy atoms
        for res_id in range(1, length + 1):
            res_atoms = struc[struc.res_id == res_id]
            for atom in res_atoms:
                if atom.element == "H":
                    continue

                b_mask = (b_res_indices == res_id) & (b_atom_names == atom.atom_name)
                b_coord = batch_res.coords[0][b_mask]

                if b_coord.size > 0:
                    np.testing.assert_allclose(b_coord.squeeze(), atom.coord, atol=0.015)
