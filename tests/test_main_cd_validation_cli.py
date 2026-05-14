"""Regression test for the --gen-cd validation report.

The CD spectrum simulator ships a ``validate_cd_against_literature`` helper that
compares the synthetic 222 nm / 217 nm peaks against published values. An earlier
mmcif-support revision dropped the call site so the report quietly disappeared
from the CLI even though the function still existed. This test guards the wiring.
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from synth_pdb.main import main


def test_gen_cd_logs_validation_report(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """--gen-cd on an alpha-helix sequence must log the Synthetic CD Validation Report
    plus at least one helix-related finding line.

    We force ``--conformation alpha`` so the helix fraction is high enough that
    ``validate_cd_against_literature`` actually emits a finding rather than an
    empty list - that way we test both the wrapper logs *and* the per-finding loop.
    """
    caplog.set_level(logging.INFO)

    test_args = [
        "synth_pdb",
        "--sequence",
        "ALAALAALAALAALAALA",  # 18 alanines, clearly alpha-helical
        "--gen-cd",
        "--conformation",
        "alpha",
        "--output",
        str(tmp_path / "test.pdb"),
    ]

    with patch("sys.argv", test_args):
        main()

    assert (
        "Synthetic CD Validation Report" in caplog.text
    ), "CD validation report header missing from logs - was validate_cd_against_literature called?"
    # At least one finding line for a helical signature must be logged.
    assert "Helix" in caplog.text or "helix" in caplog.text, (
        "No helix-related CD finding logged. Either the validator returned no findings "
        "(unexpected for an alpha-helical poly-Ala) or the per-finding loop is broken."
    )
