import os
import shutil
import subprocess


def test_multimodal_dataset_builder_cli() -> None:
    """Verify that the multimodal dataset builder script runs without errors."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(project_root, "scripts", "build_multimodal_dataset.py")
    output_dir = os.path.join(project_root, "tests", "test_multimodal_out")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    try:
        # Run script for 2 samples
        cmd = [
            "python3",
            script_path,
            "--n",
            "2",
            "--output-dir",
            output_dir,
            "--sequence",
            "AGS",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, "metadata.csv"))
        assert len(os.listdir(os.path.join(output_dir, "pdbs"))) == 2
        assert len(os.listdir(os.path.join(output_dir, "mrcs"))) == 2
        assert len(os.listdir(os.path.join(output_dir, "saxs"))) == 2
        assert len(os.listdir(os.path.join(output_dir, "nmr"))) == 2

        # Verify NMR RDC file content
        nmr_file = os.path.join(output_dir, "nmr", "sample_0000_rdc.csv")
        with open(nmr_file) as f:
            lines = f.readlines()
            assert "residue,rdc_hz" in lines[0]
            assert len(lines) > 1

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
