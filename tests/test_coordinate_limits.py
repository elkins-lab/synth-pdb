import io
import numpy as np
import pytest
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from synth_pdb.generator import generate_pdb_content
from synth_pdb.batch_generator import BatchedGenerator


class TestCoordinateLimits:
    """Tests for the PDB coordinate limit (F8.3) and centering logic."""

    def test_serial_centering_threshold(self):
        """Verify serial generator centers only when coordinates > 500A."""
        # Small peptide should NOT be centered (origin aligned)
        pdb_small = generate_pdb_content(sequence_str="AAA", minimize_energy=False)
        struc_small = PDBFile.read(io.StringIO(pdb_small)).get_structure(model=1)
        # N-term of first residue should be at [0,0,0] (with tiny floating point slack)
        np.testing.assert_allclose(struc_small.coord[0], [0.0, 0.0, 0.0], atol=1e-2)

        # Very long peptide should be centered
        # A helix is ~1.5A per residue. 1000 residues ~ 1500A.
        # This will definitely exceed 500A from origin.
        pdb_large = generate_pdb_content(sequence_str="A" * 1000, minimize_energy=False)
        struc_large = PDBFile.read(io.StringIO(pdb_large)).get_structure(model=1)

        # Check that it is centered (centroid near origin)
        centroid = np.mean(struc_large.coord, axis=0)
        np.testing.assert_allclose(
            centroid, [0.0, 0.0, 0.0], atol=1.0
        )  # More slack for 1000 residues

        # Check that NO coordinate exceeds PDB limits (-1000, 9999)
        assert np.all(struc_large.coord > -999.0)
        assert np.all(struc_large.coord < 9999.0)

    def test_batch_centering_mixed(self):
        """Verify batched generator centers only the structures that overflow."""
        # BatchedGenerator currently takes a single sequence.
        # Let's test that a large batch is centered.
        bg = BatchedGenerator("A" * 1000, n_batch=1, full_atom=True)
        batch = bg.generate_batch(drift=0.0)

        coords = batch.coords[0]
        centroid = np.mean(coords, axis=0)
        print(f"\nBatch Centroid: {centroid}")
        np.testing.assert_allclose(centroid, [0.0, 0.0, 0.0], atol=1.0)
        assert np.all(coords > -999.0)
        assert np.all(coords < 9999.0)

    def test_multi_chain_offset_preservation(self):
        """Verify that inter-chain offsets are preserved after centering."""
        # Use two 1000-residue chains to trigger per-chain centering.
        sequence = "A" * 1000 + ":" + "A" * 1000
        pdb_content = generate_pdb_content(
            sequence_str=sequence, conformation="alpha", seed=42, minimize_energy=False
        )
        struc_complex = PDBFile.read(io.StringIO(pdb_content)).get_structure(model=1)

        chains = np.unique(struc_complex.chain_id)
        chain_a = struc_complex[struc_complex.chain_id == chains[0]]
        chain_b = struc_complex[struc_complex.chain_id == chains[1]]

        centroid_a = np.mean(chain_a.coord, axis=0)
        centroid_b = np.mean(chain_b.coord, axis=0)

        rel_offset = centroid_b - centroid_a
        print(f"\nRelative Offset: {rel_offset}")

        # Because both chains were centered at [0,0,0] before offset was added:
        # Chain A centroid should be [0,0,0] + 0*offset = [0,0,0]
        # Chain B centroid should be [0,0,0] + 1*offset = [20,20,20]
        np.testing.assert_allclose(rel_offset, [20.0, 20.0, 20.0], atol=1.0)

        # The whole complex has centroids at [0,0,0] and [20,20,20] initially.
        # But because the extent of a 2000-res complex is > 500A, the whole
        # complex itself will also be centered at [0,0,0].
        complex_centroid = np.mean(struc_complex.coord, axis=0)
        np.testing.assert_allclose(complex_centroid, [0.0, 0.0, 0.0], atol=1.0)
