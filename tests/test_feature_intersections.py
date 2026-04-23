import io

import biotite.structure.io.pdb as pdb
import numpy as np
import pytest

from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import HAS_OPENMM
from synth_pdb.validator import PDBValidator


@pytest.mark.skipif(
    not HAS_OPENMM, reason="Complex intersection tests require OpenMM physics engine"
)
class TestFeatureIntersections:
    """
    Integration tests for features that intersect:
    1. Head-to-Tail Cyclization
    2. D-Amino Acids
    3. Metal Ion Coordination (Zinc Finger)
    """

    def test_complex_minimized_closure_and_zinc(self) -> None:
        """
        Verify that a minimized cyclic peptide with Zinc has correct
        CONECT records and physical closure.
        """
        sequence = "CYS-ALA-CYS-GLY-HIS-PRO-HIS-GLY"  # 8 residues

        pdb_content = generate_pdb_content(
            sequence_str=sequence,
            cyclic=True,
            minimize_energy=True,
            metal_ions="auto",
            conformation="alpha",
            seed=42,
        )

        # 1. Verify CONECT records exist for ring closure (Fixed bug)
        assert "CONECT" in pdb_content

        # 2. Verify Physical Closure
        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)
        n_atom = structure[(structure.res_id == 1) & (structure.atom_name == "N")][0]
        c_atom = structure[(structure.res_id == 8) & (structure.atom_name == "C")][0]
        dist_closure = np.linalg.norm(n_atom.coord - c_atom.coord)
        assert dist_closure < 1.6, f"Cyclic bond not closed physically: {dist_closure:.3f} A"

        # 3. Verify Zinc is present
        assert "ZN" in pdb_content
        zn_atoms = structure[structure.element == "ZN"]
        assert len(zn_atoms) == 1

    def test_complex_intersection_no_explosions(self) -> None:
        """Verify that combining all features doesn't cause severe steric clashes."""
        sequence = "CYS-ALA-CYS-D-ALA-HIS-PRO-HIS-GLY"
        pdb_content = generate_pdb_content(
            sequence_str=sequence,
            cyclic=True,
            minimize_energy=True,
            metal_ions="auto",
            conformation="alpha",
        )

        validator = PDBValidator(pdb_content)
        validator.validate_steric_clashes()
        clash_violations = [v for v in validator.violations if "Steric clash" in v]
        # Should be reasonable
        assert len(clash_violations) < 30

    def test_chirality_inversion_direct(self) -> None:
        """Simple check that D-ALA has inverted sign of improper vs L-ALA."""
        l_pdb = generate_pdb_content(sequence_str="ALA", minimize_energy=False)
        d_pdb = generate_pdb_content(sequence_str="D-ALA", minimize_energy=False)

        def get_imp(pdb_str: str) -> float:
            val = PDBValidator(pdb_str)
            ats = val.atoms

            def get_c(name: str) -> np.ndarray:
                return [a["coords"] for a in ats if a["atom_name"] == name][0]

            return float(
                val._calculate_dihedral_angle(get_c("N"), get_c("CA"), get_c("C"), get_c("CB"))
            )

        l_imp = get_imp(l_pdb)
        d_imp = get_imp(d_pdb)
        assert np.sign(l_imp) != np.sign(d_imp)
