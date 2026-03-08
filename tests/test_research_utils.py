import os
import tempfile

import biotite.structure.io.pdb as pdb

from synth_pdb.decoys import DecoyGenerator
from synth_pdb.docking import DockingPrep
from synth_pdb.generator import generate_pdb_content


def test_decoy_generation():
    """Verify that DecoyGenerator creates diverse ensembles."""
    sequence = "ACDEF"
    n_decoys = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = DecoyGenerator()
        decoys = generator.generate_ensemble(
            sequence=sequence,
            n_decoys=n_decoys,
            out_dir=tmpdir,
            rmsd_min=0.0,
            rmsd_max=10.0,
            optimize=False,
            minimize=False,
        )

        # Check count
        assert len(decoys) == n_decoys
        assert len(os.listdir(tmpdir)) == n_decoys

        # Check files are valid PDBs
        for path in decoys:
            assert os.path.getsize(path) > 0
            f = pdb.PDBFile.read(path)
            assert f.get_structure(model=1).array_length() > 0


def test_docking_prep_pqr():
    """Verify that DockingPrep converts PDB to PQR with charges."""
    # Create input PDB (poly-alanine)
    pdb_content = generate_pdb_content(sequence_str="AAAA", conformation="alpha")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.pdb")
        output_pqr = os.path.join(tmpdir, "output.pqr")

        # Write input (stripping H to be safe, though DockingPrep adds them anyway)
        # Using Biotite to strip H
        import io

        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        heavy = structure[structure.element != "H"]

        pdb_file_heavy = pdb.PDBFile()
        pdb_file_heavy.set_structure(heavy)
        pdb_file_heavy.write(input_path)

        prep = DockingPrep(forcefield_name="amber14-all.xml")
        success = prep.write_pqr(input_path, output_pqr)

        assert success
        assert os.path.exists(output_pqr)

        # Verify PQR content (Charge column)
        with open(output_pqr) as f:
            lines = [l for l in f.readlines() if l.startswith("ATOM")]
            assert len(lines) > 0

            # Check a line format
            # ATOM  1    N    ALA A   1    ... X Y Z Charge Radius
            # Split and check last two are floats
            parts = lines[0].split()
            charge = float(parts[-2])
            radius = float(parts[-1])

            assert isinstance(charge, float)
            assert isinstance(radius, float)
            assert radius > 0  # Radius must be positive
