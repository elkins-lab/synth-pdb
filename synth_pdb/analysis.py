import logging
from typing import Any, Dict, List

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

from .geometry import (
    calculate_dihedral,
    calculate_rmsd,
    calculate_rmsd_to_average,
    find_medoid,
    kabsch_superposition,
)

logger = logging.getLogger(__name__)

class GeometryAnalyzer:
    """High-level analysis suite for protein geometry and ensembles."""

    @staticmethod
    def compare_pdbs(pdb_path1: str, pdb_path2: str, ca_only: bool = True) -> Dict[str, Any]:
        """Compare two PDB files and calculate RMSD and optimal transformation.

        Args:
            pdb_path1: Path to the first PDB file (mobile).
            pdb_path2: Path to the second PDB file (reference).
            ca_only: If True, only uses C-alpha atoms for alignment and RMSD.

        Returns:
            Dictionary containing 'rmsd', 'rotation', and 'translation'.
        """
        s1 = pdb.PDBFile.read(pdb_path1).get_structure(model=1)
        s2 = pdb.PDBFile.read(pdb_path2).get_structure(model=1)

        if ca_only:
            m1 = s1[s1.atom_name == "CA"]
            m2 = s2[s2.atom_name == "CA"]
        else:
            # Common heavy atoms
            m1 = s1[struc.filter_heavy(s1)]
            m2 = s2[struc.filter_heavy(s2)]

        if len(m1) != len(m2):
            raise ValueError(f"Atom count mismatch: {len(m1)} vs {len(m2)}")

        r1 = m1.coord
        r2 = m2.coord

        rot, trans = kabsch_superposition(r1, r2)
        fitted = (rot @ r1.T).T + trans
        rmsd = calculate_rmsd(fitted, r2)

        return {
            "rmsd": rmsd,
            "rotation": rot,
            "translation": trans,
        }

    @staticmethod
    def analyze_ensemble_pdbs(pdb_paths: List[str]) -> Dict[str, Any]:
        """Analyze a list of PDB files as an NMR-style ensemble.

        Args:
            pdb_paths: List of file paths to PDB files.

        Returns:
            Dictionary with 'avg_rmsd', 'medoid_path', and 'medoid_index'.
        """
        coords_list = []
        for path in pdb_paths:
            s = pdb.PDBFile.read(path).get_structure(model=1)
            ca = s[s.atom_name == "CA"]
            coords_list.append(ca.coord)

        avg_rmsd, _ = calculate_rmsd_to_average(coords_list)
        medoid_idx = find_medoid(coords_list, superimpose=True)

        return {
            "avg_rmsd": avg_rmsd,
            "medoid_index": medoid_idx,
            "medoid_path": pdb_paths[medoid_idx],
        }

    @staticmethod
    def calculate_residue_strain(pdb_path: str) -> Dict[int, float]:
        """Calculates 'geometric strain' per residue.

        Currently defined as the deviation of the peptide bond omega angle
        from trans (180 deg).
        """
        s = pdb.PDBFile.read(pdb_path).get_structure(model=1)
        res_ids = np.unique(s.res_id)
        strain = {}

        for i in range(len(res_ids) - 1):
            rid1 = res_ids[i]
            rid2 = res_ids[i+1]

            # Omega: CA(i) - C(i) - N(i+1) - CA(i+1)
            ca1 = s[(s.res_id == rid1) & (s.atom_name == "CA")]
            c1 = s[(s.res_id == rid1) & (s.atom_name == "C")]
            n2 = s[(s.res_id == rid2) & (s.atom_name == "N")]
            ca2 = s[(s.res_id == rid2) & (s.atom_name == "CA")]

            if all(len(x) > 0 for x in [ca1, c1, n2, ca2]):
                omega = calculate_dihedral(ca1[0].coord, c1[0].coord, n2[0].coord, ca2[0].coord)
                # Map to [0, 180] deviation from trans (180 deg)
                # (omega - 180 + 180) % 360 - 180 gives signed distance to 180
                diff = (omega - 180 + 180) % 360 - 180
                deviation = abs(diff)
                strain[int(rid1)] = deviation

        return strain
