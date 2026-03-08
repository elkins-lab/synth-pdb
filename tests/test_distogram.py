import json

import biotite.structure as struc
import numpy as np
import pytest

# Try to import the module (will fail initially)
try:
    from synth_pdb import distogram
except ImportError:
    distogram = None


def create_triangle_structure():
    """
    Creates a simple 3-residue structure forming a right triangle.
    Res 1: (0,0,0)
    Res 2: (3,0,0) -> Distance 1-2 = 3.0
    Res 3: (0,4,0) -> Distance 1-3 = 4.0, Distance 2-3 = 5.0 (3-4-5 Triangle)
    """
    atoms = struc.AtomArray(3)
    atoms.res_name = np.array(["ALA", "ALA", "ALA"])
    atoms.res_id = np.array([1, 2, 3])
    atoms.atom_name = np.array(["CA", "CA", "CA"])  # Use CA for distance
    atoms.coord = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    return atoms


class TestDistogramExport:

    def test_module_exists(self):
        if distogram is None:
            pytest.fail("synth_pdb.distogram module not found")

    def test_distogram_calculation(self):
        """Test calculation of NxN distance matrix."""
        if distogram is None:
            pytest.skip("Module not implemented")

        atoms = create_triangle_structure()

        # Calculate matrix
        matrix = distogram.calculate_distogram(atoms, method="ca")

        assert matrix.shape == (3, 3)
        # Check diagonal
        assert np.allclose(np.diag(matrix), 0.0)

        # Check specific distances
        # 1-2 = 3.0
        assert np.isclose(matrix[0, 1], 3.0)
        assert np.isclose(matrix[1, 0], 3.0)

        # 1-3 = 4.0
        assert np.isclose(matrix[0, 2], 4.0)

        # 2-3 = 5.0 (hypotenuse)
        assert np.isclose(matrix[1, 2], 5.0)

    def test_export_json_format(self, tmp_path):
        """Test export to JSON."""
        if distogram is None:
            pytest.skip("Module not implemented")

        atoms = create_triangle_structure()
        matrix = distogram.calculate_distogram(atoms)
        outfile = tmp_path / "dist.json"

        distogram.export_distogram(matrix, str(outfile), fmt="json")

        assert outfile.exists()
        data = json.loads(outfile.read_text())
        assert len(data) == 3  # 3 rows
        assert len(data[0]) == 3  # 3 cols
        assert np.isclose(data[0][1], 3.0)

    def test_export_csv_format(self, tmp_path):
        """Test export to CSV."""
        if distogram is None:
            pytest.skip("Module not implemented")

        atoms = create_triangle_structure()
        matrix = distogram.calculate_distogram(atoms)
        outfile = tmp_path / "dist.csv"

        distogram.export_distogram(matrix, str(outfile), fmt="csv")

        assert outfile.exists()
        content = outfile.read_text()
        # Should be a matrix CSV or flattened? Usually matrix.
        # Check headers if any, or just numbers.
        # Implementation decision: No headers for matrix CSV usually, or simple residue index.
        # Let's verify it contains the numbers.
        assert "3.0" in content
        assert "5.0" in content

    def test_export_numpy_format(self, tmp_path):
        """Test export to .npz or .npy."""
        if distogram is None:
            pytest.skip("Module not implemented")

        atoms = create_triangle_structure()
        matrix = distogram.calculate_distogram(atoms)
        outfile = tmp_path / "dist.npz"

        distogram.export_distogram(matrix, str(outfile), fmt="npz")

        assert outfile.exists()
        # Verify load
        loaded = np.load(str(outfile))
        # Typically npz stores files inside. We might save as 'distogram'.
        assert "distogram" in loaded
        assert np.allclose(loaded["distogram"], matrix)

    def test_distogram_calculation_cb(self):
        """Test calculation using CB atoms (and fallback)."""
        if distogram is None:
            pytest.skip("Module not implemented")

        atoms = create_triangle_structure()
        # Mock CB atoms by renaming CA to CB for test simplicity
        # Or add CB atoms.
        # Let's add CB atoms to the atom array.
        # Actually simpler: atoms.atom_name[:] = "CB"
        atoms.atom_name[:] = "CB"

        matrix = distogram.calculate_distogram(atoms, method="cb")
        assert matrix.shape == (3, 3)
        assert np.isclose(matrix[0, 1], 3.0)

    def test_export_invalid_format(self, tmp_path):
        """Test error on unknown format."""
        if distogram is None:
            pytest.skip("Module not implemented")

        atoms = create_triangle_structure()
        matrix = distogram.calculate_distogram(atoms)
        outfile = tmp_path / "dist.txt"

        with pytest.raises(ValueError, match="Unknown format"):
            distogram.export_distogram(matrix, str(outfile), fmt="txt")
