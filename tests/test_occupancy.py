"""Tests for occupancy value assignment.

EDUCATIONAL NOTE - What is Occupancy?
=====================================
Occupancy (columns 55-60 in PDB ATOM records) represents the fraction of
molecules in a crystal where an atom occupies a particular position.

Physical Meaning:
- 1.00 = Atom present in 100% of molecules (fully ordered)
- 0.85 = Atom present in 85% of molecules (some disorder)
- 0.50 = Two equally populated alternate conformations
- <0.50 = Highly disordered or multiple conformations

Why Occupancy Varies:
1. **Conformational Flexibility**: Flexible regions (loops, termini) show disorder
2. **Alternate Conformations**: Side chains can adopt multiple rotamers
3. **Crystal Packing**: Surface residues may be less ordered
4. **Temperature**: Higher temperature increases disorder

Relationship to B-factors:
- Both indicate atomic disorder
- Low occupancy often correlates with high B-factors
- Occupancy = static disorder (multiple positions)
- B-factor = dynamic disorder (thermal motion)

In Real Structures:
- Core residues: occupancy ≈ 1.00 (well-ordered)
- Surface residues: occupancy 0.90-1.00 (some disorder)
- Flexible loops: occupancy 0.85-0.95 (disordered)
- Alternate conformations: occupancy 0.40-0.60 each (split)
"""

import numpy as np

from synth_pdb.generator import _calculate_occupancy, generate_pdb_content


class TestOccupancyCalculation:
    """Test suite for occupancy value assignment."""

    def test_occupancy_function_exists(self):
        """Test that occupancy calculation function exists."""
        # This will fail until we implement the function (TDD Red phase)
        assert callable(_calculate_occupancy)

    def test_backbone_vs_sidechain_occupancy(self):
        """Test that backbone atoms have higher occupancy than side chain atoms.

        EDUCATIONAL NOTE:
        Backbone atoms (N, CA, C, O) are constrained by peptide bonds and
        typically more ordered than side chain atoms, which can adopt multiple
        rotameric conformations.

        In crystal structures:
        - Backbone occupancy: typically 0.95-1.00
        - Side chain occupancy: typically 0.85-0.95 (more variable)
        """
        # Middle residue to avoid terminal effects
        residue_num = 10
        total_residues = 20
        bfactor = 20.0  # Average B-factor

        # Average over multiple calls to account for random variation
        backbone_avg = np.mean(
            [
                _calculate_occupancy("CA", residue_num, total_residues, "ALA", bfactor)
                for _ in range(10)
            ]
        )
        sidechain_avg = np.mean(
            [
                _calculate_occupancy("CB", residue_num, total_residues, "ALA", bfactor)
                for _ in range(10)
            ]
        )

        # Backbone should have higher occupancy (more ordered)
        assert (
            backbone_avg > sidechain_avg
        ), f"Backbone occupancy ({backbone_avg:.2f}) should be > side chain ({sidechain_avg:.2f})"

    def test_terminal_residues_lower_occupancy(self):
        """Test that terminal residues have lower occupancy than middle residues.

        EDUCATIONAL NOTE:
        Terminal residues are more flexible because they have fewer constraints:
        - N-terminus: No preceding residue to constrain backbone
        - C-terminus: No following residue to constrain backbone
        - This "terminal fraying" causes both higher B-factors AND lower occupancy

        The effect is especially pronounced in:
        - Short peptides (our case)
        - Unstructured regions
        - Surface-exposed termini
        """
        total_residues = 20
        bfactor = 20.0

        # Average over multiple calls
        n_term_avg = np.mean(
            [_calculate_occupancy("CA", 1, total_residues, "ALA", bfactor) for _ in range(10)]
        )
        middle_avg = np.mean(
            [_calculate_occupancy("CA", 10, total_residues, "ALA", bfactor) for _ in range(10)]
        )
        c_term_avg = np.mean(
            [_calculate_occupancy("CA", 20, total_residues, "ALA", bfactor) for _ in range(10)]
        )

        # Termini should have lower occupancy (more disordered)
        assert (
            n_term_avg < middle_avg
        ), f"N-terminus ({n_term_avg:.2f}) should have lower occupancy than middle ({middle_avg:.2f})"
        assert (
            c_term_avg < middle_avg
        ), f"C-terminus ({c_term_avg:.2f}) should have lower occupancy than middle ({middle_avg:.2f})"

    def test_flexible_residues_lower_occupancy(self):
        """Test that flexible residues have lower occupancy.

        EDUCATIONAL NOTE - Residue Flexibility:
        Some amino acids are inherently more flexible:

        Flexible (lower occupancy):
        - GLY: No side chain, high backbone flexibility
        - SER: Small polar side chain, multiple rotamers
        - ASN: Polar side chain with amide group, flexible
        - GLN: Long polar side chain, very flexible

        These residues often show:
        - Multiple side chain conformations
        - Lower occupancy values
        - Higher B-factors
        """
        residue_num = 10
        total_residues = 20
        bfactor = 20.0

        # Compare flexible GLY to average ALA
        gly_avg = np.mean(
            [
                _calculate_occupancy("CA", residue_num, total_residues, "GLY", bfactor)
                for _ in range(10)
            ]
        )
        ala_avg = np.mean(
            [
                _calculate_occupancy("CA", residue_num, total_residues, "ALA", bfactor)
                for _ in range(10)
            ]
        )

        assert (
            gly_avg < ala_avg
        ), f"GLY ({gly_avg:.2f}) should have lower occupancy than ALA ({ala_avg:.2f})"

    def test_rigid_residues_higher_occupancy(self):
        """Test that rigid residues have higher occupancy.

        EDUCATIONAL NOTE - Rigid Residues:
        Some amino acids are more constrained:

        Rigid (higher occupancy):
        - PRO: Cyclic structure restricts backbone
        - TRP: Large aromatic side chain, fewer rotamers
        - PHE: Large aromatic, relatively constrained

        These residues often show:
        - Single dominant conformation
        - Higher occupancy values
        - Lower B-factors
        """
        residue_num = 10
        total_residues = 20
        bfactor = 20.0

        # Compare rigid PRO to average ALA
        pro_avg = np.mean(
            [
                _calculate_occupancy("CA", residue_num, total_residues, "PRO", bfactor)
                for _ in range(10)
            ]
        )
        ala_avg = np.mean(
            [
                _calculate_occupancy("CA", residue_num, total_residues, "ALA", bfactor)
                for _ in range(10)
            ]
        )

        assert (
            pro_avg > ala_avg
        ), f"PRO ({pro_avg:.2f}) should have higher occupancy than ALA ({ala_avg:.2f})"

    def test_occupancy_realistic_range(self):
        """Test that occupancy values are in realistic range (0.85-1.00).

        EDUCATIONAL NOTE - Typical Occupancy Ranges:

        In real crystal structures:
        - Well-ordered core: 0.95-1.00
        - Average residues: 0.90-0.95
        - Flexible regions: 0.85-0.90
        - Very disordered: 0.70-0.85 (rare)
        - Alternate conformations: 0.40-0.60 each

        For our linear peptides:
        - We use 0.85-1.00 range
        - No alternate conformations (would need special handling)
        - Reflects typical disorder in unfolded/extended structures
        """
        # Test various scenarios
        test_cases = [
            ("CA", 1, 20, "ALA", 35.0),  # N-terminus, high B-factor
            ("CB", 1, 20, "ALA", 35.0),  # N-terminus sidechain
            ("CA", 10, 20, "ALA", 20.0),  # Middle, average B-factor
            ("CB", 10, 20, "ALA", 25.0),  # Middle sidechain
            ("CA", 20, 20, "ALA", 35.0),  # C-terminus
            ("CA", 10, 20, "GLY", 25.0),  # Flexible residue
            ("CA", 10, 20, "PRO", 15.0),  # Rigid residue
        ]

        for atom_name, res_num, total_res, res_name, bf in test_cases:
            occupancy = _calculate_occupancy(atom_name, res_num, total_res, res_name, bf)
            assert (
                0.85 <= occupancy <= 1.00
            ), f"Occupancy {occupancy:.2f} out of range for {res_name} {atom_name}"

    def test_occupancy_correlates_with_bfactor(self):
        """Test that lower occupancy correlates with higher B-factors.

        EDUCATIONAL NOTE - Occupancy vs B-factor Relationship:

        Both measure disorder, but differently:

        B-factor (Temperature Factor):
        - Measures thermal motion and positional uncertainty
        - Dynamic disorder (atom vibrating)
        - Continuous distribution around mean position

        Occupancy:
        - Measures static disorder
        - Multiple discrete positions
        - Fraction of molecules with atom at each position

        Correlation:
        - High B-factor often accompanies low occupancy
        - Both indicate flexible/disordered regions
        - Termini typically have both high B-factors AND low occupancy
        - Core residues have both low B-factors AND high occupancy
        """
        residue_num = 10
        total_residues = 20

        # Test with low B-factor (rigid)
        occ_low_bf = np.mean(
            [
                _calculate_occupancy("CA", residue_num, total_residues, "ALA", 15.0)
                for _ in range(10)
            ]
        )

        # Test with high B-factor (flexible)
        occ_high_bf = np.mean(
            [
                _calculate_occupancy("CA", residue_num, total_residues, "ALA", 45.0)
                for _ in range(10)
            ]
        )

        # Lower B-factor should have higher occupancy
        assert (
            occ_low_bf > occ_high_bf
        ), f"Low B-factor occupancy ({occ_low_bf:.2f}) should be > high B-factor ({occ_high_bf:.2f})"

    def test_occupancy_in_generated_pdb(self):
        """Test that generated PDB files contain realistic occupancy values.

        EDUCATIONAL NOTE - Occupancy in PDB Files:
        Occupancy appears in columns 55-60 of ATOM records.
        Format: 6.2f (6 characters, 2 decimal places)

        Example ATOM line:
        ATOM      1  N   ALA A   1       0.000   0.000   0.000  0.95 20.00           N
                                                                  ^^^^
                                                                  Occupancy

        Should not all be 1.00 (which indicates no disorder information).
        """
        pdb_content = generate_pdb_content(length=10, conformation="alpha")

        # Extract occupancy values from ATOM lines
        occupancies = []
        for line in pdb_content.split("\n"):
            if line.startswith("ATOM"):
                # Occupancy is in columns 55-60 (0-indexed: 54-60)
                occ_str = line[54:60].strip()
                if occ_str:
                    occupancies.append(float(occ_str))

        # Should have occupancy for all atoms
        assert len(occupancies) > 0, "No occupancy values found in PDB"

        # Should not all be 1.00
        assert not all(o == 1.00 for o in occupancies), "All occupancies are 1.00 (not realistic)"

        # Should be in realistic range
        assert all(
            0.85 <= o <= 1.00 for o in occupancies
        ), f"Some occupancies out of range: {occupancies}"

        # Should show variation (not all the same)
        assert len(set(occupancies)) > 1, "All occupancies are identical (should vary)"

    def test_occupancy_variation_along_chain(self):
        """Test that occupancy varies along the peptide chain.

        EDUCATIONAL NOTE - Occupancy Gradients:
        In real structures, occupancy typically shows patterns:

        - Gradual decrease toward termini
        - Lower values in loops vs helices/sheets
        - Side chains more variable than backbone
        - Correlation with secondary structure

        For linear peptides:
        - Terminal fraying causes lower occupancy at ends
        - Middle residues more ordered
        - Creates observable gradient
        """
        pdb_content = generate_pdb_content(length=20, conformation="alpha")

        # Extract CA occupancy by residue number
        ca_occupancies = {}
        for line in pdb_content.split("\n"):
            if line.startswith("ATOM") and " CA " in line:
                res_num = int(line[22:26].strip())
                occupancy = float(line[54:60].strip())
                ca_occupancies[res_num] = occupancy

        # Should have occupancy for all residues
        assert len(ca_occupancies) == 20

        # Terminal residues should generally have lower occupancy than middle
        n_term_avg = np.mean([ca_occupancies[i] for i in range(1, 4)])
        middle_avg = np.mean([ca_occupancies[i] for i in range(9, 13)])
        c_term_avg = np.mean([ca_occupancies[i] for i in range(18, 21)])

        # At least one terminus should be lower than middle
        assert (n_term_avg < middle_avg) or (
            c_term_avg < middle_avg
        ), "Termini should have lower occupancy than middle"
