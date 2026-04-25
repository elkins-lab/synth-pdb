from synth_pdb.validator import PDBValidator

# Mock a simple dimer PDB: Two ALA residues, one on Chain A, one on Chain B.
DIMER_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   1.200   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       1.400   2.200   0.000  1.00  0.00           O\n"
    "TER       5      ALA A   1\n"
    "ATOM      6  N   ALA B   2      10.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      7  CA  ALA B   2      11.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      8  C   ALA B   2      12.000   1.200   0.000  1.00  0.00           C\n"
    "ATOM      9  O   ALA B   2      11.400   2.200   0.000  1.00  0.00           O\n"
    "TER      10      ALA B   2\n"
)

# Mock a clashing dimer
CLASHING_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "TER       2      ALA A   1\n"
    "ATOM      3  CA  ALA B   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "TER       4      ALA B   1\n"
)

# Mock a salt bridge across chains
SALT_BRIDGE_PDB = (
    "ATOM      1  OD1 ASP A   1       0.000   0.000   0.000  1.00  0.00           O\n"
    "TER       2      ASP A   1\n"
    "ATOM      3  NZ  LYS B  10       3.000   0.000   0.000  1.00  0.00           N\n"
    "TER       4      LYS B  10\n"
)


class TestInterfaceAnalytics:
    """Test suite for Protein-Protein Interface Analytics."""

    def test_single_chain_graceful_exit(self) -> None:
        """Verify that interface metrics return empty for a single chain."""
        pdb_content = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        )
        validator = PDBValidator(pdb_content=pdb_content)
        metrics = validator.calculate_interface_metrics()

        assert metrics == {}

    def test_bsa_calculation(self) -> None:
        """Verify BSA calculation: BSA = SASA_A + SASA_B - SASA_AB."""
        # For a distant dimer (DIMER_PDB), SASA_AB should roughly equal SASA_A + SASA_B.
        # BSA should be near 0.
        validator = PDBValidator(pdb_content=DIMER_PDB)
        metrics = validator.calculate_interface_metrics()

        assert "buried_surface_area" in metrics
        assert metrics["buried_surface_area"] >= 0
        # Given they are 10A apart, BSA should be zero
        assert metrics["buried_surface_area"] < 1.0

    def test_inter_chain_clashes(self) -> None:
        """Verify detection of steric clashes between different chains."""
        validator = PDBValidator(pdb_content=CLASHING_PDB)
        metrics = validator.calculate_interface_metrics()

        assert "inter_chain_clashes" in metrics
        assert len(metrics["inter_chain_clashes"]) == 1
        clash_str = metrics["inter_chain_clashes"][0]
        assert "Chain A" in clash_str
        assert "Chain B" in clash_str
        assert "too close" in clash_str

    def test_inter_chain_salt_bridge(self) -> None:
        """Verify detection of salt bridges across the interface."""
        validator = PDBValidator(pdb_content=SALT_BRIDGE_PDB)
        metrics = validator.calculate_interface_metrics()

        assert "inter_chain_salt_bridges" in metrics
        assert len(metrics["inter_chain_salt_bridges"]) == 1
        bridge = metrics["inter_chain_salt_bridges"][0]
        assert bridge["chain_a"] == "A"
        assert bridge["chain_b"] == "B"
        assert bridge["distance"] < 4.0
