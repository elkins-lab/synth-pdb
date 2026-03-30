import io

import biotite.structure.io.pdb as pdb
import numpy as np
import pytest

from synth_pdb import torsion


class TestTorsionExport:

    @pytest.fixture
    def alpha_helix_pdb(self):
        """Returns a string wrapper for a simple 3-residue alpha helix PDB content."""
        return """HEADER    EXTENDED
ATOM      1  N   ALA A   1      -1.458   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.525   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.100   1.200   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       2.300  -1.200   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       3.750  -1.200   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       4.500   0.100   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       4.000   1.200   0.000  1.00  0.00           O
ATOM      9  N   ALA A   3       5.800   0.100   0.000  1.00  0.00           N
ATOM     10  CA  ALA A   3       6.500   1.400   0.000  1.00  0.00           C
ATOM     11  C   ALA A   3       8.000   1.400   0.000  1.00  0.00           C
ATOM     12  O   ALA A   3       8.600   2.500   0.000  1.00  0.00           O
END
"""

    def test_calculate_torsion_angles_structure(self, alpha_helix_pdb):
        """Test calculation of angles from structure."""
        pdb_file = pdb.PDBFile.read(io.StringIO(alpha_helix_pdb))
        atom_array = pdb_file.get_structure(model=1)

        # Calculate
        df = torsion.calculate_torsion_angles(atom_array)

        # Check structure of return
        assert isinstance(df, list)
        assert len(df) == 3  # 3 residues

        # First residue has no Phi
        assert df[0]["phi"] is None or np.isnan(df[0]["phi"])
        assert df[0]["psi"] is not None

        # Second residue has both
        assert df[1]["phi"] is not None
        assert df[1]["psi"] is not None

        # Last residue has no Psi
        assert df[2]["psi"] is None or np.isnan(df[2]["psi"])

    def test_export_csv(self, tmp_path):
        """Test export to CSV format."""
        data = [
            {"residue": "ALA", "res_id": 1, "phi": np.nan, "psi": -57.0, "omega": 180.0},
            {"residue": "GLY", "res_id": 2, "phi": -60.0, "psi": -45.0, "omega": 180.0},
        ]

        outfile = tmp_path / "angles.csv"
        torsion.export_torsion_angles(data, str(outfile), fmt="csv")

        assert outfile.exists()
        content = outfile.read_text()
        assert "residue,res_id,phi,psi,omega" in content.lower()
        assert "ALA,1,, -57.0,180.0" in content or "ALA,1,nan,-57.0,180.0" in content

    def test_export_json(self, tmp_path):
        """Test export to JSON format."""
        data = [
            {"residue": "ALA", "res_id": 1, "phi": None, "psi": -57.0},
        ]

        outfile = tmp_path / "angles.json"
        torsion.export_torsion_angles(data, str(outfile), fmt="json")

        assert outfile.exists()
        import json

        loaded = json.loads(outfile.read_text())
        assert loaded[0]["residue"] == "ALA"
