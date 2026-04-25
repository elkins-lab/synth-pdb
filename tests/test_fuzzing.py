import io

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from synth_pdb.data import NON_STANDARD_RESIDUES, STANDARD_AMINO_ACIDS
from synth_pdb.generator import generate_pdb_content

# All residues including standard and PTMs/Non-standard ones
ALL_RESIDUES = sorted(list(STANDARD_AMINO_ACIDS) + list(NON_STANDARD_RESIDUES))


class TestPropertyBasedFuzzing:
    """
    Mathematical and numerical robustness suite using Hypothesis.
    Aims to ensure the generator never crashes or produces invalid math (NaNs)
    regardless of input sequence or length.
    """

    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(length=st.integers(min_value=1, max_value=100))
    def test_random_length_stability(self, length: int) -> None:
        """Ensure generator is stable for any length from 1 to 100."""
        pdb_content = generate_pdb_content(length=length)

        # 1. Coordinate Invariant: No NaNs or Infs
        assert "NaN" not in pdb_content
        assert "Inf" not in pdb_content

        # 2. Parsability Invariant: Biotite must be able to read it back
        f = io.StringIO(pdb_content)
        pdb_file = pdb.PDBFile.read(f)
        structure = pdb_file.get_structure(model=1)

        assert len(structure) > 0
        assert not np.any(np.isnan(structure.coord))

    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        sequence=st.text(alphabet=st.sampled_from("ACDEFGHIKLMNPQRSTVWY"), min_size=1, max_size=50)
    )
    def test_random_sequence_stability(self, sequence: str) -> None:
        """Ensure generator is stable for any standard amino acid sequence."""
        pdb_content = generate_pdb_content(sequence_str=sequence)

        f = io.StringIO(pdb_content)
        pdb_file = pdb.PDBFile.read(f)
        structure = pdb_file.get_structure(model=1)

        # Geometry Invariant: Bond lengths should be within reasonable bounds
        structure.bonds = struc.connect_via_residue_names(structure)
        bonds = structure.bonds.as_array()

        for i, j, _ in bonds:
            dist = np.linalg.norm(structure.coord[i] - structure.coord[j])

            atom_i = structure.atom_name[i]
            atom_j = structure.atom_name[j]

            # Use very relaxed bounds for fuzzing to account for all possible geometries
            # The goal is catching NaNs or massive blowups (>10A)
            assert 0.5 < dist < 5.0, f"Extreme blowup: {dist:.2f}A between {atom_i} and {atom_j}"

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(res_list=st.lists(st.sampled_from(ALL_RESIDUES), min_size=1, max_size=30))
    def test_non_standard_residue_stability(self, res_list: list) -> None:
        """Ensure generator handles mixed standard and non-standard/PTM residues."""
        # Join with '-' to ensure 3-letter code parsing
        sequence = "-".join(res_list)
        pdb_content = generate_pdb_content(sequence_str=sequence)

        assert "NaN" not in pdb_content

        f = io.StringIO(pdb_content)
        pdb_file = pdb.PDBFile.read(f)
        structure = pdb_file.get_structure(model=1)
        assert len(structure) > 0

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        length=st.integers(min_value=3, max_value=20), cyclic=st.booleans(), minimize=st.booleans()
    )
    def test_generator_flag_permutations(self, length: int, cyclic: bool, minimize: bool) -> None:
        """Ensure stability across combinations of generation flags."""
        pdb_content = generate_pdb_content(length=length, cyclic=cyclic, minimize_energy=minimize)

        f = io.StringIO(pdb_content)
        pdb_file = pdb.PDBFile.read(f)
        structure = pdb_file.get_structure(model=1)
        assert len(structure) > 0
