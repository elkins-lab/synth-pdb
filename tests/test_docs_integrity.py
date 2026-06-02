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

    def setUp(self) -> None:
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
        self.cryo_em_path = os.path.join(self.base_dir, "synth_pdb", "cryo_em.py")
        self.saxs_path = os.path.join(self.base_dir, "synth_pdb", "saxs.py")
        self.visualization_saxs_path = os.path.join(
            self.base_dir, "synth_pdb", "visualization_saxs.py"
        )
        self.msa_path = os.path.join(self.base_dir, "synth_pdb", "msa.py")
        self.chemical_shifts_path = os.path.join(self.base_dir, "synth_pdb", "chemical_shifts.py")
        self.cd_simulator_path = os.path.join(self.base_dir, "synth_pdb", "cd_simulator.py")
        self.relaxation_path = os.path.join(self.base_dir, "synth_pdb", "relaxation.py")
        self.nmr_path = os.path.join(self.base_dir, "synth_pdb", "nmr.py")
        self.coupling_path = os.path.join(self.base_dir, "synth_pdb", "coupling.py")
        self.structure_utils_path = os.path.join(self.base_dir, "synth_pdb", "structure_utils.py")
        self.j_coupling_path = os.path.join(self.base_dir, "synth_pdb", "j_coupling.py")
        self.benchmark_path = os.path.join(self.base_dir, "synth_pdb", "benchmark.py")
        self.gnn_classifier_path = os.path.join(
            self.base_dir, "synth_pdb", "quality", "gnn", "gnn_classifier.py"
        )
        self.gnn_graph_path = os.path.join(self.base_dir, "synth_pdb", "quality", "gnn", "graph.py")

        # Script paths
        self.train_gnn_path = os.path.join(self.base_dir, "scripts", "train_gnn_quality_filter.py")
        self.compare_gnn_path = os.path.join(self.base_dir, "scripts", "compare_gnn_diversity.py")
        self.stress_test_gnn_path = os.path.join(self.base_dir, "scripts", "stress_test_gnn.py")

        # Test file paths
        self.test_rdc_q_path = os.path.join(
            self.base_dir, "tests", "test_rdc_q_factor_validation.py"
        )
        self.test_karplus_path = os.path.join(self.base_dir, "tests", "test_karplus_j_coupling.py")
        self.test_main_rdc_path = os.path.join(
            self.base_dir, "tests", "test_main_rdc_and_predictor.py"
        )
        self.test_scientific_path = os.path.join(
            self.base_dir, "tests", "test_scientific_validation.py"
        )
        self.test_ssbond_path = os.path.join(self.base_dir, "tests", "test_ssbond.py")
        self.test_chirality_path = os.path.join(self.base_dir, "tests", "test_chirality.py")
        self.test_rdc_shim_path = os.path.join(self.base_dir, "tests", "test_rdc.py")
        self.test_cis_proline_path = os.path.join(self.base_dir, "tests", "test_cis_proline.py")
        self.test_rigor_path = os.path.join(
            self.base_dir, "tests", "functional", "test_scientific_rigor.py"
        )
        self.test_occupancy_path = os.path.join(self.base_dir, "tests", "test_occupancy.py")

        # Modular Geometry Paths
        self.geometry_nerf_path = os.path.join(self.base_dir, "synth_pdb", "geometry", "nerf.py")
        self.geometry_dihedral_path = os.path.join(
            self.base_dir, "synth_pdb", "geometry", "dihedral.py"
        )
        self.geometry_vectorized_path = os.path.join(
            self.base_dir, "synth_pdb", "geometry", "vectorized.py"
        )

    def _check_file_contains(self, filepath: str, substrings: list[str]) -> None:
        """Helper to assert file contains list of substrings."""

        with open(filepath, encoding="utf-8") as f:
            # Replace '#' with ' ' to handle line-wrapped comments
            raw_content = f.read().replace("#", " ")
            content = " ".join(raw_content.split())

        for substring in substrings:
            normalized_substring = " ".join(substring.split())
            self.assertIn(
                normalized_substring,
                content,
                f"Missing educational note in {os.path.basename(filepath)}: '{substring[:50]}...'",
            )

    def test_decoys_educational_notes(self) -> None:
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

    def test_biophysics_educational_notes(self) -> None:
        """Ensure biophysics.py retains key educational blocks."""
        required_notes = [
            "Educational Note - pH and Protonation:",
            "Biological function depends on pH",
            "Imidazole ring is neutral",
            "EDUCATIONAL NOTE - Terminal Capping:",
            "Uncapped termini (NH3+ and COO-) introduce strong charges",
            "EDUCATIONAL NOTE: Salt Bridges",
            "combination of two non-covalent interactions",
            "EDUCATIONAL NOTE - Conformational Assumptions:",
            "EDUCATIONAL NOTE - Vectorized Proximity Search:",
        ]
        self._check_file_contains(self.biophysics_path, required_notes)

    def test_physics_educational_notes(self) -> None:
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
            "Educational Note: Minimization Reporters",
            "black box",
            "L-BFGS",
            "callback",
            'peek" into the C++ optimization loop',
            "Monitoring convergence",
            "Identifying exactly which iteration caused a system to",
            "Stopping the process if the energy goes above a threshold",
            "EDUCATIONAL NOTE - Cyclic CONECT Stripping:",
            "EDUCATIONAL NOTE - Ion Stripping:",
            "EDUCATIONAL NOTE - Dummy OXT Insertion:",
            "EDUCATIONAL NOTE: We do NOT add the bond to the Topology here.",
            "EDUCATIONAL NOTE - Why Add Hydrogens?",
            "EDUCATIONAL NOTE - Atom Index Refresh:",
            "EDUCATIONAL NOTE - Why we avoid adding a hard constraint initially:",
            "EDUCATIONAL NOTE - Simulation Setup:",
            "EDUCATIONAL NOTE - Disulfide Mapping:",
            "EDUCATIONAL NOTE - CONECT Records & Visualization:",
            "EDUCATIONAL NOTE - HETATM Restoration:",
            "EDUCATIONAL NOTE: Gradient Descent (L-BFGS)",
            "Limited-memory Broyden-Fletcher-Goldfarb-Shanno",
            "quasi-Newton method",
            "approximates the second derivative (Hessian)",
            "store the full Hessian matrix",
        ]
        self._check_file_contains(self.physics_path, required_notes)

    def test_validator_educational_notes(self) -> None:
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
            "EDUCATIONAL NOTE - The Shrake-Rupley Algorithm:",
            "EDUCATIONAL NOTE - VdW Radii:",
            "EDUCATIONAL NOTE - Energy as a Clash Detector:",
            "EDUCATIONAL NOTE - Performance Optimization via Vectorization:",
            "nested loops for clash detection",
            "O(N^2) operation",
            "scipy.spatial.distance.cdist",
            "NumPy broadcasting",
            "Pre-calculated for all pairs as a matrix",
            "EDUCATIONAL NOTE - NOE Effective Distance:",
            "EDUCATIONAL NOTE - The Structural Interactome:",
        ]
        self._check_file_contains(self.validator_path, required_notes)

    def test_cofactors_educational_notes(self) -> None:
        """Ensure cofactors.py retains coordination chemistry note."""
        required_notes = [
            "Educational Note - Coordination Chemistry:",
            "Cys Sulfur, His Nitrogen",
            "Tetrahedral",
            "EDUCATIONAL NOTE: Local Optimization",
            "EDUCATIONAL NOTE: Geometric Realism",
        ]
        self._check_file_contains(self.cofactors_path, required_notes)

    def test_geometry_educational_notes(self) -> None:
        """Ensure geometry modules retain Z-Matrix, NeRF, and performance notes."""
        nerf_notes = [
            "EDUCATIONAL NOTE - Z-Matrix Construction",
            "Bond Length",
            "Bond Angle",
            "Torsion/Dihedral Angle",
            "EDUCATIONAL NOTE - NeRF Geometry",
            "Natural Extension Reference Frame",
            "mathematical precision",
        ]
        self._check_file_contains(self.geometry_nerf_path, nerf_notes)

        dihedral_notes = [
            "EDUCATIONAL NOTE - Circular Statistics (The 180/-180 Problem):",
            "Boundary Artifact",
        ]
        self._check_file_contains(self.geometry_dihedral_path, dihedral_notes)

        vectorized_notes = [
            "EDUCATIONAL NOTE - SIMD & Parallel Geometry:",
            "EDUCATIONAL NOTE - Vectorized Kabsch Algorithm:",
            "EDUCATIONAL NOTE - GPU-First Operations:",
            "Memory bandwidth",
        ]
        self._check_file_contains(self.geometry_vectorized_path, vectorized_notes)

    def test_batch_generator_educational_notes(self) -> None:
        """Ensure batch_generator.py retains performance notes."""
        required_notes = [
            "EDUCATIONAL OVERVIEW - Batched Generation (GPU-First):",
            "Broadcasting",
            "Hardware Acceleration",
            "EDUCATIONAL NOTE - Peptidyl Chain Walk:",
            'EDUCATIONAL NOTE - The "Memory Wall" in AI Training:',
            "PCIE Latency",
            "EDUCATIONAL NOTE - PDB Specification:",
        ]
        self._check_file_contains(self.batch_generator_path, required_notes)

    def test_data_educational_notes(self) -> None:
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
            "EDUCATIONAL NOTE - Why Stricter Standards?",
        ]
        self._check_file_contains(self.data_path, required_notes)

    def test_packing_and_scoring_educational_notes(self) -> None:
        """Ensure packing.py and scoring.py retain optimization notes."""
        self._check_file_contains(
            self.packing_path, ["EDUCATIONAL NOTE - Monte Carlo Optimization"]
        )
        self._check_file_contains(
            self.scoring_path,
            ["EDUCATIONAL NOTE - Steric Repulsion and Forces", "Lennard-Jones Potential"],
        )

    def test_generator_educational_notes(self) -> None:
        """Ensure generator.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - B-factors (Temperature Factors)",
            "B = 8pi^2<u^2>",  # Physics formula
            "Lipari-Szabo",
            "EDUCATIONAL NOTE - Hard Decoy Support (AI Training):",
            "Torsion Drift",
            "Threading",
            "EDUCATIONAL OVERVIEW - How Synthetic Protein Generation Works:",
            "EDUCATIONAL NOTE - PDB ATOM Record Format:",
            "EDUCATIONAL NOTE - Disulfide Bond Detection:",
            "EDUCATIONAL NOTE - PDB SSBOND Format:",
            "EDUCATIONAL NOTE - Why This Matters:",
            "EDUCATIONAL NOTE - Return Format:",
            "EDUCATIONAL NOTE - Design Decisions:",
            "EDUCATIONAL NOTE - Data Structure Choice:",
            "EDUCATIONAL NOTE - Why These Checks Matter:",
            "EDUCATIONAL NOTE - Why We Forbid Overlaps:",
            "EDUCATIONAL NOTE - Error Message Design:",
            "EDUCATIONAL NOTE - What Happens to Gaps:",
            "EDUCATIONAL NOTE - Input Validation:",
            "EDUCATIONAL NOTE - Per-Residue Conformation Assignment:",
            "EDUCATIONAL NOTE - Gap Handling:",
            "EDUCATIONAL NOTE - Why We Don't Validate Conformations Here:",
            "EDUCATIONAL NOTE - D-Amino Acid Handling:",
            "EDUCATIONAL NOTE - PDB Coordinate Range Limits:",
            "EDUCATIONAL NOTE - Biophysical Realism (Phase 2):",
            "EDUCATIONAL NOTE - Side-Chain Optimization:",
            "EDUCATIONAL NOTE - Metal Ion Coordination (Phase 15):",
            "EDUCATIONAL NOTE - Energy Minimization (Phase 2):",
            "EDUCATIONAL NOTE - Explicit Solvent Pruning:",
            "EDUCATIONAL NOTE - Adding Realistic B-factors:",
            "EDUCATIONAL NOTE - Adding Realistic Occupancy:",
            "EDUCATIONAL NOTE - HETATM vs ATOM:",
            "EDUCATIONAL NOTE - New Feature: Cyclic Peptides",
            "EDUCATIONAL NOTE - Why Per-Region Conformations Matter:",
            "EDUCATIONAL NOTE - Macrocyclization (Cyclic Peptides):",
            "EDUCATIONAL NOTE - Multi-Chain Complex Generation (Phase 16):",
            "EDUCATIONAL NOTE - Post-Translational Modifications (PTMs):",
        ]
        self._check_file_contains(self.generator_path, required_notes)

    def test_bfactor_test_educational_notes(self) -> None:
        """Ensure test_bfactor.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - What are B-factors?",
            "Backbone atoms (N, CA, C, O) are constrained by peptide bonds",
        ]
        self._check_file_contains(self.bfactor_test_path, required_notes)

    def test_ramachandran_test_educational_notes(self) -> None:
        """Ensure test_ramachandran.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - The Ramachandran Plot",
            "Why Glycine is Special",
            "Why Proline is Special",
        ]
        self._check_file_contains(self.ramachandran_test_path, required_notes)

    def test_viewer_educational_notes(self) -> None:
        """Ensure viewer.py retains key educational blocks and examples."""
        viewer_path = os.path.join(self.base_dir, "synth_pdb", "viewer.py")
        required_notes = [
            "EDUCATIONAL NOTE - Why Browser-Based Visualization:",
            "EDUCATIONAL NOTE - 3Dmol.js:",
            "NMR Short-Range Restraints (NOEs) roughly depend on 1/r^6",
        ]
        self._check_file_contains(viewer_path, required_notes)

    def test_readme_educational_notes(self) -> None:
        """Ensure README.md retains key academic notes."""
        readme_path = os.path.join(self.base_dir, "README.md")
        required_notes = [
            'Academic Note - "Amphipathic"',
            "Hydrophobic Face",
            "Hydrophilic Face",
            "Atomic Records & B-Factors",
        ]
        self._check_file_contains(readme_path, required_notes)

    def test_dataset_educational_notes(self) -> None:
        """Ensure dataset.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - The Balanced Dataset Problem:",
            "Alpha-Helix Trap",
            "Halls of Mirrors",
            "Data Factory Overview:",
        ]
        self._check_file_contains(self.dataset_path, required_notes)

    def test_orientogram_educational_notes(self) -> None:
        """Ensure orientogram.py retains orientations note."""
        required_notes = [
            "EDUCATIONAL NOTE - trRosetta 6D Orientations:",
            "dist, omega, theta, phi",
            "Internal Coordinates",
            "Virtual C-beta",
        ]
        self._check_file_contains(self.orientogram_path, required_notes)

    def test_special_chemistry_educational_notes(self) -> None:
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

    def test_rdc_educational_notes(self) -> None:
        """Ensure rdc.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - What are Residual Dipolar Couplings (RDCs)?",
            "EDUCATIONAL NOTE — The Zero-Denominator Hazard in Q-Factor Calculation:",
        ]
        self._check_file_contains(self.rdc_path, required_notes)

    def test_docking_educational_notes(self) -> None:
        """Ensure docking.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE: Ensure connectivity before hydrogen addition",
        ]
        self._check_file_contains(self.docking_path, required_notes)

    def test_main_educational_notes(self) -> None:
        """Ensure main.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - Scientific Reproducibility:",
            "EDUCATIONAL NOTE - RDC Background:",
            "EDUCATIONAL NOTE - Predictor Selection:",
            "EDUCATIONAL NOTE - RDC Calculation:",
            "EDUCATIONAL NOTE - CD Background:",
            "Circular Dichroism (CD) measures the differential absorption",
            "weighted average of basis",
            "spectra (Greenfield & Fasman, 1969, Biochemistry 8:4108):",
            "f_helix",
            "f_sheet",
            "f_coil",
        ]
        self._check_file_contains(self.main_path, required_notes)

    def test_cryo_em_educational_notes(self) -> None:
        """Ensure cryo_em.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL OVERVIEW - Cryo-EM Density Simulation:",
            "Coulomb potential",
            "Gaussian Approximation",
            "Ensemble Averaging",
            "EDUCATIONAL NOTE - The MRC Format:",
            "standard for 3D electron microscopy",
            "1024-byte header",
        ]
        self._check_file_contains(self.cryo_em_path, required_notes)

    def test_saxs_educational_notes(self) -> None:
        """Ensure saxs.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL OVERVIEW - SAXS Curve Simulation:",
            "The Debye Formula:",
            "Atomic Form Factors:",
            "Solvent Contrast:",
        ]
        self._check_file_contains(self.saxs_path, required_notes)

    def test_visualization_saxs_educational_notes(self) -> None:
        """Ensure visualization_saxs.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL RATIONALE:",
            "EDUCATIONAL NOTE - The Kratky Plot:",
            "EDUCATIONAL NOTE - The Guinier Approximation:",
            "slope of this line is -Rg^2 / 3",
        ]
        self._check_file_contains(self.visualization_saxs_path, required_notes)

    def test_msa_educational_notes(self) -> None:
        """Ensure msa.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - Steric Volume Compatibility:",
            "Educational Note: Hydrophobic Core Collapse",
            "Educational Note: Electrostatic Compatibility",
            "Educational Note: Big-O Performance Breakthrough",
            'Educational Note: The "Magic Step" coupled mutation',
        ]
        self._check_file_contains(self.msa_path, required_notes)

    def test_chemical_shifts_educational_notes(self) -> None:
        """Ensure chemical_shifts.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - Chemical Shifts in Structural Biology",
            "EDUCATIONAL NOTE - Approximation via Parent Mapping:",
        ]
        self._check_file_contains(self.chemical_shifts_path, required_notes)

    def test_cd_simulator_educational_notes(self) -> None:
        """Ensure cd_simulator.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - CD Background:",
            "Circular Dichroism (CD) measures the differential absorption",
            "premier tool for measuring the",
            "overall secondary structure",
            "weighted average of basis",
            "spectra (Greenfield & Fasman, 1969, Biochemistry 8:4108):",
            "f_helix",
            "f_sheet",
            "f_coil",
        ]
        self._check_file_contains(self.cd_simulator_path, required_notes)

    def test_relaxation_educational_notes(self) -> None:
        """Ensure relaxation.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - NMR Relaxation and the Lipari-Szabo Model",
        ]
        self._check_file_contains(self.relaxation_path, required_notes)

    def test_nmr_educational_notes(self) -> None:
        """Ensure nmr.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - The Nuclear Overhauser Effect (NOE)",
            "EDUCATIONAL NOTE — Key-Format Normalization and the Falsy-Zero Hazard:",
            "EDUCATIONAL NOTE — Precision in RPF and the Set-Lookup Pattern:",
            "EDUCATIONAL NOTE — The F-Measure (S\u00f8rensen-Dice Coefficient):",
        ]
        self._check_file_contains(self.nmr_path, required_notes)

    def test_coupling_educational_notes(self) -> None:
        """Ensure coupling.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - The Karplus Equation and 3J(HN-HA) Couplings",
            "EDUCATIONAL NOTE - Proline and Secondary Amines",
            "EDUCATIONAL NOTE - D-Amino Acids and Stereochemistry",
        ]
        self._check_file_contains(self.coupling_path, required_notes)

    def test_structure_utils_educational_notes(self) -> None:
        """Ensure structure_utils.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - NMR Structure Utilities and RMSD",
        ]
        self._check_file_contains(self.structure_utils_path, required_notes)

    def test_j_coupling_educational_notes(self) -> None:
        """Ensure j_coupling.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - Scalar J-Couplings and Karplus Equations",
        ]
        self._check_file_contains(self.j_coupling_path, required_notes)

    def test_benchmark_educational_notes(self) -> None:
        """Ensure benchmark.py retains key structural biology and AI notes."""
        required_notes = [
            "The benchmark asks a simple but powerful question:",
            "SCIENTIFIC RATIONALE - Why benchmark against synthetic structures?",
            "METRICS EXPLAINED - The Structural Biology Toolkit",
            "Metrics computed",
            "BENCHMARK LIFECYCLE",
            "Generates ``n_structures`` synthetic protein structures",
        ]
        self._check_file_contains(self.benchmark_path, required_notes)

    def test_gnn_library_educational_notes(self) -> None:
        """Ensure GNN library files retain key educational blocks."""
        graph_notes = [
            "EDUCATIONAL BACKGROUND - Why represent a protein as a graph?",
            "GEOMETRIC VECTORIZATION - Why skip PDB parsing?",
            "TOPOLOGY VS CARTESIAN - Why Graphs Win",
            "GRAPH STRUCTURE",
        ]
        self._check_file_contains(self.gnn_graph_path, graph_notes)

        classifier_notes = [
            "DESIGN CONTRACT - Same API as ProteinQualityClassifier (RF)",
            "CHECKPOINT FORMAT (.pt)",
            "HIGH-THROUGHPUT AUDITING - Vectorized Ensemble Scoring",
            "VECTORIZED ENSEMBLE AUDITING",
        ]
        self._check_file_contains(self.gnn_classifier_path, classifier_notes)

    def test_gnn_scripts_educational_notes(self) -> None:
        """Ensure GNN training and benchmark scripts retain key educational blocks."""
        train_notes = [
            "STRUCTURAL BIOLOGY CONTEXT - Why this multi-task approach?",
            "Educational note - Ramachandran Z-score and Steric Constraints",
            "Why these four classes?",
            "Good - idealized backbone geometry with physical 'wobble'",
            'High Torsion Drift (40-60deg) - Marginal "Bad"',
            "Surgical Torsion Corruption (1-2 bad residues in good backbone)",
        ]
        self._check_file_contains(self.train_gnn_path, train_notes)

        compare_notes = [
            "SCIENTIFIC OBJECTIVE:",
            "Educational Insight:",
        ]
        self._check_file_contains(self.compare_gnn_path, compare_notes)

        stress_notes = [
            "SCIENTIFIC OBJECTIVE:",
            "Torsion Drift (Backbone Wobble)",
            "Local Corruption (One Bad Apple)",
        ]
        self._check_file_contains(self.stress_test_gnn_path, stress_notes)

    def test_test_files_educational_notes(self) -> None:
        """Ensure test files retain key educational blocks."""
        self._check_file_contains(self.test_rdc_q_path, ["EDUCATIONAL NOTE:"])
        self._check_file_contains(self.test_karplus_path, ["EDUCATIONAL NOTE:"])
        self._check_file_contains(
            self.test_main_rdc_path,
            [
                "EDUCATIONAL NOTE - Synthetic RDCs in AI Training:",
                "EDUCATIONAL NOTE - Chemical Shift Prediction Methods:",
                "EDUCATIONAL NOTE - Provenance in PDB Headers:",
            ],
        )
        self._check_file_contains(
            self.test_scientific_path,
            ["EDUCATIONAL NOTE: This sliding-window approach (Local RMSD) is used in"],
        )
        self._check_file_contains(
            self.test_ssbond_path,
            [
                "EDUCATIONAL NOTE - Disulfide Bonds in Proteins",
                "EDUCATIONAL NOTE:",
                "EDUCATIONAL NOTE - Distance Criteria:",
                "EDUCATIONAL NOTE - False Positives:",
                "EDUCATIONAL NOTE - PDB SSBOND Format:",
                "EDUCATIONAL NOTE - Edge Cases:",
                "EDUCATIONAL NOTE - Multiple Disulfides:",
            ],
        )
        self._check_file_contains(
            self.test_chirality_path, ["EDUCATIONAL NOTE - Molecular Chirality (Handedness)"]
        )
        self._check_file_contains(
            self.test_rdc_shim_path,
            [
                "EDUCATIONAL NOTE - Why shims?",
                "EDUCATIONAL NOTE - Proline as a secondary amine:",
            ],
        )
        self._check_file_contains(
            self.test_cis_proline_path, ["EDUCATIONAL NOTE - Peptide Bond Isomerism (Cis vs Trans)"]
        )
        self._check_file_contains(
            self.test_rigor_path,
            [
                "EDUCATIONAL NOTE - The Philosophy of Scientific Verification:",
                "EDUCATIONAL NOTE - The Chemical Shift Index (CSI):",
            ],
        )
        self._check_file_contains(
            self.test_occupancy_path,
            [
                "EDUCATIONAL NOTE - What is Occupancy?",
                "EDUCATIONAL NOTE:",
                "EDUCATIONAL NOTE - Residue Flexibility:",
                "EDUCATIONAL NOTE - Rigid Residues:",
                "EDUCATIONAL NOTE - Typical Occupancy Ranges:",
                "EDUCATIONAL NOTE - Occupancy vs B-factor Relationship:",
                "EDUCATIONAL NOTE - Occupancy in PDB Files:",
                "EDUCATIONAL NOTE - Occupancy Gradients:",
            ],
        )
