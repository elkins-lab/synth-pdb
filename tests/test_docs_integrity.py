import os
import unittest

"""
Enforce the philosophy that the code is the textbook by checking for educational note comments.
Also see test_comment_density.py for tests of the comment density.
"""


class TestDocumentationIntegrity(unittest.TestCase):
    """Safeguard to ensure educational notes are not accidentally removed.

    These tests scan the source code for specific educational content that
    must be preserved to maintain the pedagogical value of the project.
    """

    def setUp(self):
        # Define paths relative to this test file
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.generator_path = os.path.join(self.base_dir, "synth_pdb", "generator.py")
        self.bfactor_test_path = os.path.join(self.base_dir, "tests", "test_bfactor.py")
        self.ramachandran_test_path = os.path.join(self.base_dir, "tests", "test_ramachandran.py")
        self.decoys_path = os.path.join(self.base_dir, "synth_pdb", "decoys.py")
        self.biophysics_path = os.path.join(self.base_dir, "synth_pdb", "biophysics.py")
        self.physics_path = os.path.join(self.base_dir, "synth_pdb", "physics.py")
        self.validator_path = os.path.join(self.base_dir, "synth_pdb", "validator.py")
        self.cofactors_path = os.path.join(self.base_dir, "synth_pdb", "cofactors.py")
        self.geometry_path = os.path.join(self.base_dir, "synth_pdb", "geometry.py")
        self.packing_path = os.path.join(self.base_dir, "synth_pdb", "packing.py")
        self.scoring_path = os.path.join(self.base_dir, "synth_pdb", "scoring.py")
        self.dataset_path = os.path.join(self.base_dir, "synth_pdb", "dataset.py")
        self.batch_generator_path = os.path.join(self.base_dir, "synth_pdb", "batch_generator.py")
        self.data_path = os.path.join(self.base_dir, "synth_pdb", "data.py")
        self.orientogram_path = os.path.join(self.base_dir, "synth_pdb", "orientogram.py")
        self.special_chemistry_path = os.path.join(
            self.base_dir, "synth_pdb", "special_chemistry.py"
        )
        self.rdc_path = os.path.join(self.base_dir, "synth_pdb", "rdc.py")
        self.docking_path = os.path.join(self.base_dir, "synth_pdb", "docking.py")
        self.main_path = os.path.join(self.base_dir, "synth_pdb", "main.py")

    def _check_file_contains(self, filepath, substrings):
        """Helper to assert file contains list of substrings."""
        with open(filepath, encoding="utf-8") as f:
            content = " ".join(f.read().split())

        for substring in substrings:
            normalized_substring = " ".join(substring.split())
            self.assertIn(
                normalized_substring,
                content,
                f"Missing educational note in {os.path.basename(filepath)}: '{substring[:50]}...'",
            )

    def test_decoys_educational_notes(self):
        """Ensure decoys.py retains key educational blocks."""
        required_notes = [
            'EDUCATIONAL NOTE - "Decoys" vs "NMR Ensembles"',
            "* **NMR Ensemble**: A set of structures that all satisfy experimental restraints",
            "* **Decoys**: Independent random conformations",
            "represent the SEARCH SPACE",
            'EDUCATIONAL NOTE - "Hard Decoys" for AI Models:',
            "Threading",
            "Shuffling",
            "torsion errors (Drift)",
            "EDUCATIONAL NOTE - RMSD (Root Mean Square Deviation):",
        ]
        self._check_file_contains(self.decoys_path, required_notes)

    def test_biophysics_educational_notes(self):
        """Ensure biophysics.py retains key educational blocks."""
        required_notes = [
            "Educational Note - pH and Protonation:",
            "Biological function depends on pH",
            "Imidazole ring is neutral",
            "EDUCATIONAL NOTE - Terminal Capping:",
            "Uncapped termini (NH3+ and COO-) introduce strong charges",
            "EDUCATIONAL NOTE: Salt Bridges",
            "combination of two non-covalent interactions",
        ]
        self._check_file_contains(self.biophysics_path, required_notes)

    def test_physics_educational_notes(self):
        """Ensure physics.py retains key educational blocks."""
        required_notes = [
            "Educational Note: What is Energy Minimization?",
            "potential energy of the protein",
            "local minimum",
            "Educational Note - Metal Coordination in Physics:",
            "Harmonic Constraints",
            "Educational Note - Computational Efficiency:",
            "O(N^2)",
            "EDUCATIONAL NOTE - Topology Bridging",
            "EDUCATIONAL NOTE - Serialization:",
            "EDUCATIONAL NOTE - Anatomy of a Forcefield:",
            "Bonded Terms (Springs)",
            "EDUCATIONAL NOTE - Explicit vs. Implicit Solvent:",
            "EDUCATIONAL NOTE - PDB PRE-PROCESSING (OpenMM Template Fix):",
            "EDUCATIONAL NOTE - Topological Validation:",
            "EDUCATIONAL NOTE - Robust Backbone Stitching (Heuristic Bonding):",
            "EDUCATIONAL NOTE - The SSBOND Capture Radius:",
            "EDUCATIONAL NOTE - Salt Bridges & Electrostatics:",
            "EDUCATIONAL NOTE - CYX Renaming & Thiol Stripping:",
            "EDUCATIONAL NOTE - Constraints and Macrocycles:",
            'EDUCATIONAL NOTE - The "Nuclear Option" & "Shadow Caps":',
            'EDUCATIONAL NOTE - Harmonic "Pull" Restraints & Hard Constraints:',
            "EDUCATIONAL NOTE - Thermal Jiggling (Simulated Annealing):",
            "EDUCATIONAL NOTE - Thermal Equilibration (MD):",
            "Educational Note - Thermal Equilibration:",
            "find a stable dynamic average",
            "EDUCATIONAL NOTE - System Creation & Solvent Handling:",
            "Born Radii",
            "EDUCATIONAL NOTE - Thermodynamic Ensembles & Integrators:",
            "NVE",
            "NVT",
            "NPT",
            "EDUCATIONAL NOTE - The Importance of Metadata Restoration:",
            "violently mutated the input structure",
            "EDUCATIONAL NOTE - PDB Atom Sorting:",
        ]
        self._check_file_contains(self.physics_path, required_notes)

    def test_validator_educational_notes(self):
        """Ensure validator.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE:",
            "IUPAC convention for protein dihedrals",
            "Educational Note - Computational Efficiency & Convergence:",
            "performance optimization for Energy Minimization",
            "Outlier",
            "Educational Note - Side Chain Packing:",
            "Backbone-Dependent Library",
            "gauche+",
            "gauche-",
            "trans",
            "EDUCATIONAL NOTE - Physics of the Ramachandran Plot:",
            "HARD-SPHERE STERICS",
        ]
        self._check_file_contains(self.validator_path, required_notes)

    def test_cofactors_educational_notes(self):
        """Ensure cofactors.py retains coordination chemistry note."""
        required_notes = [
            "Educational Note - Coordination Chemistry:",
            "Cys Sulfur, His Nitrogen",
            "Tetrahedral",
        ]
        self._check_file_contains(self.cofactors_path, required_notes)

    def test_geometry_educational_notes(self):
        """Ensure geometry.py retains Z-Matrix and NeRF notes."""
        required_notes = [
            "EDUCATIONAL NOTE - Z-Matrix Construction",
            "Bond Length",
            "Bond Angle",
            "Torsion/Dihedral Angle",
            "EDUCATIONAL NOTE - NeRF Geometry",
            "Natural Extension Reference Frame",
            "EDUCATIONAL NOTE - SIMD & Parallel Geometry:",
            "EDUCATIONAL NOTE - Vectorized Kabsch Algorithm:",
            "Root Mean Square Deviation (RMSD)",
            "EDUCATIONAL NOTE - GPU-First Operations:",
            "Memory bandwidth",
            "EDUCATIONAL NOTE - Circular Statistics (The 180/-180 Problem):",
            "Boundary Artifact",
            "mathematical precision",
        ]
        self._check_file_contains(self.geometry_path, required_notes)

    def test_batch_generator_educational_notes(self):
        """Ensure batch_generator.py retains performance notes."""
        required_notes = [
            "EDUCATIONAL OVERVIEW - Batched Generation (GPU-First):",
            "Broadcasting",
            "Hardware Acceleration",
            "EDUCATIONAL NOTE - Peptidyl Chain Walk:",
            'EDUCATIONAL NOTE - The "Memory Wall" in AI Training:',
            "PCIE Latency",
        ]
        self._check_file_contains(self.batch_generator_path, required_notes)

    def test_data_educational_notes(self):
        """Ensure data.py retains geometric and amino acid notes."""
        required_notes = [
            "EDUCATIONAL NOTE - Engh & Huber Parameters (The Gold Standard):",
            "Gold Standard",
            'EDUCATIONAL NOTE - Proline Sterics (The "Proline Effect"):',
            "structure breaker",
            'EDUCATIONAL NOTE - The "Mirror Image" World:',
            "EDUCATIONAL NOTE - Backbone Dependency:",
            "EDUCATIONAL NOTE - Rotamers for Non-Branched Residues:",
            "EDUCATIONAL NOTE - Aromatic Residues (PHE, TYR, TRP):",
            "EDUCATIONAL NOTE - Electrostatics vs Sterics:",
        ]
        self._check_file_contains(self.data_path, required_notes)

    def test_packing_and_scoring_educational_notes(self):
        """Ensure packing.py and scoring.py retain optimization notes."""
        self._check_file_contains(
            self.packing_path, ["EDUCATIONAL NOTE - Monte Carlo Optimization"]
        )
        self._check_file_contains(
            self.scoring_path,
            ["EDUCATIONAL NOTE - Steric Repulsion and Forces", "Lennard-Jones Potential"],
        )

    def test_generator_educational_notes(self):
        """Ensure generator.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - B-factors (Temperature Factors)",
            "B = 8π²<u²>",  # Physics formula
            "Lipari-Szabo",
            "EDUCATIONAL NOTE - Hard Decoy Support (AI Training):",
            "Torsion Drift",
            "Threading",
        ]
        self._check_file_contains(self.generator_path, required_notes)

    def test_bfactor_test_educational_notes(self):
        """Ensure test_bfactor.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - What are B-factors?",
            "Backbone atoms (N, CA, C, O) are constrained by peptide bonds",
        ]
        self._check_file_contains(self.bfactor_test_path, required_notes)

    def test_ramachandran_test_educational_notes(self):
        """Ensure test_ramachandran.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - The Ramachandran Plot",
            "Why Glycine is Special",
            "Why Proline is Special",
        ]
        self._check_file_contains(self.ramachandran_test_path, required_notes)

    def test_viewer_educational_notes(self):
        """Ensure viewer.py retains key educational blocks and examples."""
        viewer_path = os.path.join(self.base_dir, "synth_pdb", "viewer.py")
        required_notes = [
            "EDUCATIONAL NOTE - Why Browser-Based Visualization:",
            "EDUCATIONAL NOTE - 3Dmol.js:",
            "NMR Short-Range Restraints (NOEs) roughly depend on 1/r^6",
        ]
        self._check_file_contains(viewer_path, required_notes)

    def test_readme_educational_notes(self):
        """Ensure README.md retains key academic notes."""
        readme_path = os.path.join(self.base_dir, "README.md")
        required_notes = [
            'Academic Note - "Amphipathic"',
            "Hydrophobic Face",
            "Hydrophilic Face",
            "Atomic Records & B-Factors",
        ]
        self._check_file_contains(readme_path, required_notes)

    def test_dataset_educational_notes(self):
        """Ensure dataset.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - The Balanced Dataset Problem:",
            "Alpha-Helix Trap",
            "Halls of Mirrors",
            "Data Factory Overview:",
        ]
        self._check_file_contains(self.dataset_path, required_notes)

    def test_orientogram_educational_notes(self):
        """Ensure orientogram.py retains orientations note."""
        required_notes = [
            "EDUCATIONAL NOTE - trRosetta 6D Orientations:",
            "dist, omega, theta, phi",
            "Internal Coordinates",
            "Virtual C-beta",
        ]
        self._check_file_contains(self.orientogram_path, required_notes)

    def test_special_chemistry_educational_notes(self):
        """Ensure special_chemistry.py retains GFP note."""
        required_notes = [
            "EDUCATIONAL OVERVIEW - GFP Chromophore Maturation:",
            "residues Ser65, Tyr66, and Gly67",
            "nucleophilic attack",
            "Cyclization",
            "Dehydration",
            "Oxidation",
            "imidazolinone ring",
        ]
        self._check_file_contains(self.special_chemistry_path, required_notes)

    def test_rdc_educational_notes(self):
        """Ensure rdc.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE — What are Residual Dipolar Couplings (RDCs)?",
        ]
        self._check_file_contains(self.rdc_path, required_notes)

    def test_docking_educational_notes(self):
        """Ensure docking.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE: Ensure connectivity before hydrogen addition",
        ]
        self._check_file_contains(self.docking_path, required_notes)

    def test_main_educational_notes(self):
        """Ensure main.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE — Scientific Reproducibility:",
            "EDUCATIONAL NOTE — RDC Background:",
            "EDUCATIONAL NOTE — Predictor Selection:",
            "EDUCATIONAL NOTE — RDC Calculation:",
        ]
        self._check_file_contains(self.main_path, required_notes)
