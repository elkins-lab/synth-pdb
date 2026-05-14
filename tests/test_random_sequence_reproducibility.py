import subprocess
import pytest
import re
from pathlib import Path


def test_decoy_sequence_reproducibility(tmp_path: Path):
    """Verify that --mode decoys --length X --seed Y produces the same sequence every time.
    Uses tmp_path to avoid leaving artifacts in the repo root.
    """
    cmd = [
        "python3",
        "-m",
        "synth_pdb.main",
        "--mode",
        "decoys",
        "--length",
        "5",
        "--seed",
        "42",
        "--n-decoys",
        "1",
        "--output",
        str(tmp_path / "decoys"),
    ]

    # Run once
    result1 = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, encoding="utf-8")
    assert result1.returncode == 0, f"Command failed: {result1.stderr}"
    # The sequence is logged at the end of a line containing "sequence"
    match1 = re.search(r"sequence:?\s+([\w-]+)", result1.stderr)
    if not match1:
        pytest.fail(f"Sequence not found in logs. Stderr: {result1.stderr}")
    seq1 = match1.group(1)

    # Run again
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, encoding="utf-8")
    assert result2.returncode == 0, f"Command failed: {result2.stderr}"
    match2 = re.search(r"sequence:?\s+([\w-]+)", result2.stderr)
    if not match2:
        pytest.fail(f"Sequence not found in logs. Stderr: {result2.stderr}")
    seq2 = match2.group(1)

    assert seq1 == seq2, f"Sequences differ: {seq1} != {seq2}"


def test_cryo_em_sequence_reproducibility(tmp_path: Path):
    """Verify that --mode cryo-em --length X --seed Y produces the same sequence every time.
    Uses tmp_path to avoid leaving artifacts in the repo root.
    """
    cmd = [
        "python3",
        "-m",
        "synth_pdb.main",
        "--mode",
        "cryo-em",
        "--length",
        "5",
        "--seed",
        "42",
        "--n-decoys",
        "1",
        "--mrc-output",
        str(tmp_path / "test.mrc"),
    ]

    # Run once
    result1 = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, encoding="utf-8")
    assert result1.returncode == 0, f"Command failed: {result1.stderr}"
    match1 = re.search(r"sequence:\s+([\w-]+)", result1.stderr)
    if not match1:
        pytest.fail(f"Sequence not found in logs. Stderr: {result1.stderr}")
    seq1 = match1.group(1)

    # Run again
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, encoding="utf-8")
    assert result2.returncode == 0, f"Command failed: {result2.stderr}"
    match2 = re.search(r"sequence:\s+([\w-]+)", result2.stderr)
    if not match2:
        pytest.fail(f"Sequence not found in logs. Stderr: {result2.stderr}")
    seq2 = match2.group(1)

    assert seq1 == seq2, f"Sequences differ: {seq1} != {seq2}"


def test_saxs_sequence_reproducibility(tmp_path: Path):
    """Verify that --mode saxs --length X --seed Y produces the same sequence every time.
    Uses tmp_path to avoid leaving artifacts in the repo root.
    """
    cmd = [
        "python3",
        "-m",
        "synth_pdb.main",
        "--mode",
        "saxs",
        "--length",
        "5",
        "--seed",
        "42",
        "--n-decoys",
        "1",
        "--saxs-output",
        str(tmp_path / "test.dat"),
    ]

    # Run once
    result1 = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, encoding="utf-8")
    assert result1.returncode == 0, f"Command failed: {result1.stderr}"
    match1 = re.search(r"sequence:\s+([\w-]+)", result1.stderr)
    if not match1:
        pytest.fail(f"Sequence not found in logs. Stderr: {result1.stderr}")
    seq1 = match1.group(1)

    # Run again
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, encoding="utf-8")
    assert result2.returncode == 0, f"Command failed: {result2.stderr}"
    match2 = re.search(r"sequence:\s+([\w-]+)", result2.stderr)
    if not match2:
        pytest.fail(f"Sequence not found in logs. Stderr: {result2.stderr}")
    seq2 = match2.group(1)

    assert seq1 == seq2, f"Sequences differ: {seq1} != {seq2}"
