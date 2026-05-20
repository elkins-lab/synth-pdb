import os
import shutil
import io
import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import biotite.structure.io.pdb as pdb_io
from synth_pdb.main import main


class TestGalleryExamples:
    """Test suite executing examples from the documentation gallery and advanced guides."""

    @pytest.fixture(autouse=True)
    def mock_visualize(self):
        """Mock browser visualization and SAXS plotting to avoid opening windows."""
        with (
            patch("webbrowser.open") as mock_web,
            patch("synth_pdb.viewer.view_structure_in_browser") as mock_view,
            patch("synth_pdb.visualization_saxs.plot_saxs_results") as mock_saxs_plot,
        ):
            yield mock_web, mock_view, mock_saxs_plot

    # --- Basic Gallery Examples ---

    def test_alpha_helix_example(self, tmp_path: Path) -> None:
        """Example: Alpha Helix structure generation."""
        output_file = tmp_path / "helix.pdb"
        test_args = [
            "synth_pdb",
            "--length",
            "20",
            "--conformation",
            "alpha",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ATOM" in output_file.read_text()

    def test_beta_sheet_example(self, tmp_path: Path) -> None:
        """Example: Beta Sheet structure generation."""
        output_file = tmp_path / "sheet.pdb"
        test_args = [
            "synth_pdb",
            "--length",
            "20",
            "--conformation",
            "beta",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ATOM" in output_file.read_text()

    def test_leucine_zipper_example(self, tmp_path: Path) -> None:
        """Example: Leucine Zipper motif with minimization."""
        output_file = tmp_path / "zipper.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "LKELEKELEKELEKELEKELEKEL",
            "--conformation",
            "alpha",
            "--minimize",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        content = output_file.read_text()
        assert "ATOM" in content
        assert "LEU" in content
        assert "GLU" in content

    def test_zinc_finger_example(self, tmp_path: Path) -> None:
        """Example: Zinc Finger coordinating metal ion."""
        output_file = tmp_path / "zinc_finger.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "KCPVCHKKFSRSDELTRHIRIHTG",
            "--conformation",
            "random",
            "--metal-ions",
            "auto",
            "--seed",
            "42",
            "--minimize",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        content = output_file.read_text()
        assert "HETATM" in content
        assert "ZN" in content

    def test_collagen_triple_helix_example(self, tmp_path: Path) -> None:
        """Example: Collagen-like polyproline II helix."""
        output_file = tmp_path / "collagen.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "GPPGPPGPPGPPGPPGPPGPP",
            "--conformation",
            "polyproline",
            "--minimize",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ATOM" in output_file.read_text()

    def test_silk_fibroin_example(self, tmp_path: Path) -> None:
        """Example: Silk Fibroin beta-sheet repeats."""
        output_file = tmp_path / "silk.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "AGAGAGAGAGAGAGAGAGAG",
            "--conformation",
            "beta",
            "--minimize",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ATOM" in output_file.read_text()

    def test_cyclic_peptide_example(self, tmp_path: Path) -> None:
        """Example: Head-to-tail cyclic peptide."""
        output_file = tmp_path / "cyclic.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "GGGGGGGGGGGG",
            "--cyclic",
            "--minimize",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        content = output_file.read_text()
        assert "CONECT" in content

    def test_disulfide_bonds_example(self, tmp_path: Path) -> None:
        """Example: Automatic disulfide bond detection."""
        output_file = tmp_path / "disulfide.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "CGGGGGGGGGGC",
            "--conformation",
            "alpha",
            "--minimize",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ATOM" in output_file.read_text()

    def test_mixed_secondary_structure_example(self, tmp_path: Path) -> None:
        """Example: Helix-turn-helix mixed topology."""
        output_file = tmp_path / "mixed.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "ACDEFGHIKLMNPQRSTVWY",
            "--structure",
            "1-7:alpha,8-13:random,14-20:alpha",
            "--minimize",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ATOM" in output_file.read_text()

    def test_d_amino_acids_example(self, tmp_path: Path) -> None:
        """Example: D-Amino Acids mirroring."""
        output_file = tmp_path / "d_amino.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "ALA-dALA-GLY-dGLY-SER-dSER",
            "--conformation",
            "alpha",
            "--minimize",
            "--output",
            str(output_file),
            "--visualize",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        content = output_file.read_text()
        # DAL is the code for D-ALA, DSE for D-SER
        assert "DAL" in content
        assert "DSE" in content
        assert "GLY" in content

    # --- Biological and Advanced Examples ---

    def test_human_egf_example(self, tmp_path: Path) -> None:
        """Example: Human Epidermal Growth Factor (EGF) with PTMs and NMR data."""
        output_file = tmp_path / "egf_protein.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR",
            "--conformation",
            "random",
            "--minimize",
            "--cap-termini",
            "--gen-shifts",
            "--gen-relax",
            "--output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ACE" in output_file.read_text()  # Check for capping
        assert (tmp_path / "egf_protein_shifts.nef").exists()
        assert (tmp_path / "egf_protein_relax.nef").exists()

    def test_gfp_fragment_example(self, tmp_path: Path) -> None:
        """Example: Green Fluorescent Protein (GFP) chromophore fragment."""
        output_file = tmp_path / "gfp_fragment.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "FEGUFSYGVQCFS",  # Now supports 'U' (SEC)
            "--conformation",
            "alpha",
            "--minimize",
            "--output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "SEC" in output_file.read_text()

    def test_pymol_scripting_example(self, tmp_path: Path) -> None:
        """Example: Generating PyMOL scripts for visualization."""
        pdb_file = tmp_path / "structure.pdb"
        generated_nef = tmp_path / "structure.nef"
        pml_file = tmp_path / "view_restraints.pml"

        # 1. Create PDB and NEF first
        test_args_gen = ["synth_pdb", "--sequence", "AAA", "--gen-nef", "--output", str(pdb_file)]
        with patch("sys.argv", test_args_gen):
            main()
        assert generated_nef.exists()

        # 2. Run PyMOL mode
        test_args_pml = [
            "synth_pdb",
            "--mode",
            "pymol",
            "--input-pdb",
            str(pdb_file),
            "--input-nef",
            str(generated_nef),
            "--output-pml",
            str(pml_file),
        ]
        with patch("sys.argv", test_args_pml):
            main()
        assert pml_file.exists()
        content = pml_file.read_text()
        assert "load" in content
        assert "structure.pdb" in content

    def test_explicit_solvent_example(self, tmp_path: Path) -> None:
        """Example: High-fidelity explicit solvent visualization."""
        output_file = tmp_path / "structure_with_water.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "MEELQK",
            "--minimize",
            "--solvent",
            "explicit",
            "--keep-solvent",
            "--output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        content = output_file.read_text()
        assert "HOH" in content  # Check for water molecules

    def test_contact_map_example(self, tmp_path: Path) -> None:
        """Example: Exporting structural contact maps as CSV."""
        output_csv = tmp_path / "contacts.csv"
        test_args = [
            "synth_pdb",
            "--sequence",
            "ALA-GLY-SER-THR-VAL",
            "--export-constraints",
            str(output_csv),
            "--constraint-format",
            "csv",
            "--constraint-cutoff",
            "8.0",
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_csv.exists()
        # CSV header is Res1,Res2,Value
        assert "Res1,Res2,Value" in output_csv.read_text()

    def test_msa_generation_example(self, tmp_path: Path) -> None:
        """Example: Synthetic Multiple Sequence Alignment (MSA) generation."""
        output_file = tmp_path / "structure.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "MEELQK",
            "--gen-msa",
            "--msa-depth",
            "10",  # Reduced depth for speed
            "--evolution-temp",
            "1.8",
            "--output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        # Check for MSA output file (Implementation uses .fasta replacing .pdb)
        msa_file = tmp_path / "structure.fasta"
        assert msa_file.exists()
        assert ">" in msa_file.read_text()

    def test_ai_interpolation_example(self, tmp_path: Path) -> None:
        """Example: AI-driven structural interpolation."""
        start_pdb = tmp_path / "helix.pdb"
        end_pdb = tmp_path / "sheet.pdb"

        # 1. Create endpoint structures (Ensure same length, disable metal ions)
        with patch(
            "sys.argv",
            [
                "synth_pdb",
                "--length",
                "10",
                "--conformation",
                "alpha",
                "--metal-ions",
                "none",
                "--output",
                str(start_pdb),
            ],
        ):
            main()
        with patch(
            "sys.argv",
            [
                "synth_pdb",
                "--length",
                "10",
                "--conformation",
                "beta",
                "--metal-ions",
                "none",
                "--output",
                str(end_pdb),
            ],
        ):
            main()

        # 2. Run interpolation
        output_prefix = tmp_path / "interpolation"
        test_args = [
            "synth_pdb",
            "--mode",
            "ai",
            "--ai-op",
            "interpolate",
            "--start-pdb",
            str(start_pdb),
            "--end-pdb",
            str(end_pdb),
            "--steps",
            "5",
            "--output",
            str(output_prefix),
        ]
        with patch("sys.argv", test_args):
            main()
        # Should generate interpolation_0.pdb to interpolation_5.pdb
        assert (tmp_path / "interpolation_0.pdb").exists()
        assert (tmp_path / "interpolation_5.pdb").exists()

    def test_sidechain_optimization_example(self, tmp_path: Path) -> None:
        """Example: Monte Carlo side-chain packing optimization."""
        output_file = tmp_path / "packed.pdb"
        test_args = [
            "synth_pdb",
            "--sequence",
            "ALA-GLY-TRP-PHE-SER",
            "--optimize",
            "--output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ATOM" in output_file.read_text()

    # --- Mode-based Examples from Gallery ---

    def test_nmr_chemical_shifts_gallery(self, tmp_path: Path) -> None:
        """Gallery Example: Predicted NMR chemical shifts."""
        output_file = tmp_path / "nmr_structure.pdb"
        test_args = [
            "synth_pdb",
            "--length",
            "30",
            "--conformation",
            "alpha",
            "--gen-shifts",
            "--output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert (tmp_path / "nmr_structure_shifts.nef").exists()

    def test_bulk_dataset_example(self, tmp_path: Path) -> None:
        """Example: Bulk dataset generation (NPZ format)."""
        output_dir = tmp_path / "training_data"
        test_args = [
            "synth_pdb",
            "--mode",
            "dataset",
            "--num-samples",
            "5",
            "--min-length",
            "10",
            "--max-length",
            "20",
            "--dataset-format",
            "npz",
            "--output",
            str(output_dir),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_dir.exists()
        assert (output_dir / "dataset_manifest.csv").exists()
        assert len(list((output_dir / "train").glob("*.npz"))) >= 1

    def test_cryo_em_density_example(self, tmp_path: Path) -> None:
        """Example: Cryo-EM 3D density map simulation."""
        output_file = tmp_path / "synthetic_map.mrc"
        test_args = [
            "synth_pdb",
            "--mode",
            "cryo-em",
            "--sequence",
            "MEELQK",
            "--resolution",
            "4.0",
            "--mrc-output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()

    def test_saxs_profile_example(self, tmp_path: Path) -> None:
        """Example: SAXS scattering curve simulation."""
        output_file = tmp_path / "protein_saxs.dat"
        test_args = [
            "synth_pdb",
            "--mode",
            "saxs",
            "--sequence",
            "NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR",
            "--q-max",
            "0.4",
            "--saxs-output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "q (A^-1)" in output_file.read_text()

    def test_docking_preparation_example(self, tmp_path: Path) -> None:
        """Example: PDB to PQR conversion for docking."""
        output_file = tmp_path / "ready_for_docking.pqr"
        test_args = [
            "synth_pdb",
            "--mode",
            "docking",
            "--sequence",
            "ACDEFGHIK",
            "--output",
            str(output_file),
        ]
        with patch("sys.argv", test_args):
            main()
        assert output_file.exists()
        assert "ATOM" in output_file.read_text()

    def test_multimodal_factory_example(self, tmp_path: Path) -> None:
        """Example: High-throughput Multi-Modal Dataset Builder script."""
        import subprocess

        output_dir = tmp_path / "my_ai_dataset"

        cmd = [
            sys.executable,
            "scripts/build_multimodal_dataset.py",
            "--n",
            "2",
            "--output-dir",
            str(output_dir),
            "--sequence",
            "AGS",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0
        assert (output_dir / "metadata.csv").exists()
        assert len(list((output_dir / "pdbs").glob("*.pdb"))) == 2
