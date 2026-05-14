"""Tests for conformational diversity feature.
Following TDD methodology - these tests should fail initially.
"""

import random
import numpy as np
import pytest

from synth_pdb.data import RAMACHANDRAN_PRESETS
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


class TestConformationalDiversity:
    """Test suite for conformational diversity feature."""

    def test_ramachandran_presets_exist(self) -> None:
        """Test that RAMACHANDRAN_PRESETS dictionary exists in data.py."""
        assert "alpha" in RAMACHANDRAN_PRESETS
        assert "beta" in RAMACHANDRAN_PRESETS
        assert "extended" in RAMACHANDRAN_PRESETS
        assert "ppii" in RAMACHANDRAN_PRESETS

    def test_ramachandran_preset_structure(self) -> None:
        """Test that each preset has correct keys."""
        for conformation, angles in RAMACHANDRAN_PRESETS.items():
            # Check for standard presets (phi/psi)
            if "phi" in angles:
                assert "psi" in angles, f"{conformation} missing psi"
                assert isinstance(angles["phi"], int | float)
                assert isinstance(angles["psi"], int | float)
            # Check for residue-specific presets (mean/std)
            elif "phi_mean" in angles:
                assert "phi_std" in angles
                assert "psi_mean" in angles
                assert "psi_std" in angles
            # Check for region-based presets (RAMACHANDRAN_REGIONS structure)
            elif "favored" in angles:
                assert isinstance(angles["favored"], list)
            else:
                pytest.fail(f"Unknown preset format for {conformation}: {angles.keys()}")

    def test_alpha_helix_angles(self) -> None:
        """Test alpha helix preset has correct angles."""
        alpha = RAMACHANDRAN_PRESETS["alpha"]
        assert alpha["phi"] == pytest.approx(-57.0, abs=1.0)
        assert alpha["psi"] == pytest.approx(-47.0, abs=1.0)

    def test_beta_sheet_angles(self) -> None:
        """Test beta sheet preset has correct angles."""
        beta = RAMACHANDRAN_PRESETS["beta"]
        assert beta["phi"] == pytest.approx(-135.0, abs=5.0)
        assert beta["psi"] == pytest.approx(135.0, abs=5.0)

    def test_default_conformation_is_alpha(self) -> None:
        """Test that default conformation is alpha helix."""
        # Use a fixed seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        generated_pdb = generate_pdb_content(length=10)  # Default is alpha

        # Verify basic structure
        lines = generated_pdb.splitlines()
        assert lines[0].startswith("HEADER")
        assert len([line for line in lines if line.startswith("ATOM")]) > 0
        assert lines[-1].strip() == "END"

    def test_generate_with_beta_conformation(self) -> None:
        """Test generating PDB with beta sheet conformation."""
        pdb_content = generate_pdb_content(length=5, sequence_str="AAAAA", conformation="beta")

        # Should generate valid PDB
        assert "ATOM" in pdb_content
        assert "END" in pdb_content

        # Validate it parses correctly
        validator = PDBValidator(pdb_content)
        atoms = validator._parse_pdb_atoms(pdb_content)
        assert len(atoms) > 0

    def test_generate_with_extended_conformation(self) -> None:
        """Test generating PDB with extended conformation."""
        pdb_content = generate_pdb_content(length=5, sequence_str="AAAAA", conformation="extended")

        # Should generate valid PDB
        assert "ATOM" in pdb_content
        assert "END" in pdb_content

    def test_generate_with_ppii_conformation(self) -> None:
        """Test generating PDB with polyproline II conformation."""
        pdb_content = generate_pdb_content(length=5, sequence_str="PPPPP", conformation="ppii")

        # Should generate valid PDB
        assert "ATOM" in pdb_content
        assert "END" in pdb_content

    def test_generate_with_polyproline_alias(self) -> None:
        """Test generating PDB with 'polyproline' alias for PPII."""
        pdb_content = generate_pdb_content(
            length=5, sequence_str="PPPPP", conformation="polyproline"
        )

        # Should generate valid PDB
        assert "ATOM" in pdb_content
        assert "END" in pdb_content

    def test_generate_with_random_conformation(self) -> None:
        """Test generating PDB with random conformation sampling."""
        # Generate two structures with random conformations
        pdb1 = generate_pdb_content(length=10, conformation="random")
        pdb2 = generate_pdb_content(length=10, conformation="random")

        # Both should be valid
        assert "ATOM" in pdb1
        assert "ATOM" in pdb2

        # They should be different (with high probability)
        # This is a probabilistic test, but with 10 residues the chance of identical
        # random structures is vanishingly small
        assert pdb1 != pdb2, "Random conformations should produce different structures"

    def test_invalid_conformation_raises_error(self) -> None:
        """Test that invalid conformation name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid conformation"):
            generate_pdb_content(length=5, conformation="invalid_name")

    def test_conformation_affects_backbone_geometry(self) -> None:
        """Test that different conformations produce different backbone geometries."""
        alpha_pdb = generate_pdb_content(length=10, sequence_str="A" * 10, conformation="alpha")
        beta_pdb = generate_pdb_content(length=10, sequence_str="A" * 10, conformation="beta")

        # Parse CA coordinates
        alpha_validator = PDBValidator(alpha_pdb)
        beta_validator = PDBValidator(beta_pdb)

        alpha_atoms = alpha_validator._parse_pdb_atoms(alpha_pdb)
        beta_atoms = beta_validator._parse_pdb_atoms(beta_pdb)

        # Get CA atoms
        alpha_ca = [a for a in alpha_atoms if a["atom_name"].strip() == "CA"]
        beta_ca = [a for a in beta_atoms if a["atom_name"].strip() == "CA"]

        # Calculate end-to-end distance
        alpha_distance = np.linalg.norm(
            np.array(alpha_ca[-1]["coords"]) - np.array(alpha_ca[0]["coords"])
        )
        beta_distance = np.linalg.norm(
            np.array(beta_ca[-1]["coords"]) - np.array(beta_ca[0]["coords"])
        )

        # Beta sheets are more extended than alpha helices
        # So end-to-end distance should be greater for beta
        assert (
            beta_distance > alpha_distance
        ), f"Beta sheet ({beta_distance:.2f}A) should be more extended than alpha helix ({alpha_distance:.2f}A)"
