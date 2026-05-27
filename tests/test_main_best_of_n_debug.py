"""Regression test for the DEBUG-level enumeration of validation violations
inside the --guarantee-valid retry loop.

When a generation attempt is rejected for violations, master logged each
individual violation message at DEBUG level inside a "PDB Validation Report
for failed attempt" / "End Validation Report" block, so an operator running
with --log-level DEBUG could see *why* the attempt failed without re-running.
An earlier mmcif-support revision collapsed that to a single warning line
(only the count survived), discarding the diagnostic detail. This test
guards the per-violation enumeration.

Strategy: stub the validator to return canned violations on the first call
and an empty list on the second so the retry loop exits cleanly. This keeps
the test fast and independent of any specific clashing-geometry recipe.
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from synth_pdb import main


def test_best_of_n_debug_enumerates_each_violation(
    mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """At DEBUG level, --guarantee-valid must enumerate every rejected
    violation inside a clearly-delimited report block."""
    caplog.set_level(logging.DEBUG)

    canned_violations = [
        "mock violation alpha: CA-CA distance too short",
        "mock violation beta: peptide bond out of plane",
    ]

    # Stub generate_pdb_content with two distinguishable-looking PDB strings -
    # the actual content doesn't matter since we also stub the validator.
    minimal_pdb = (
        "HEADER    stub\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    mocker.patch(
        "synth_pdb.main.generate_pdb_content",
        side_effect=[minimal_pdb, minimal_pdb],
    )

    # Patch PDBValidator at the call site (synth_pdb.main) so the
    # ``validator = PDBValidator(...)`` in the retry loop hits our stub.
    fake_validator_first = MagicMock()
    fake_validator_first.get_violations.return_value = canned_violations
    fake_validator_first.validate_all.return_value = None
    fake_validator_second = MagicMock()
    fake_validator_second.get_violations.return_value = []
    fake_validator_second.validate_all.return_value = None
    mocker.patch(
        "synth_pdb.main.PDBValidator",
        side_effect=[fake_validator_first, fake_validator_second],
    )
    # Prevent sys.exit short-circuiting the test if anything downstream raises.
    mocker.patch("sys.exit")

    test_args = [
        "synth_pdb",
        "--length",
        "1",
        "--guarantee-valid",
        "--max-attempts",
        "2",
        "--output",
        str(tmp_path / "test.pdb"),
        "--log-level",
        "DEBUG",
    ]
    mocker.patch("sys.argv", test_args)

    main.main()

    # The block must be opened, every individual violation must be present,
    # and the block must be closed - all at DEBUG.
    assert "--- PDB Validation Report for failed attempt ---" in caplog.text, (
        "Missing DEBUG report header - per-violation enumeration was likely "
        "dropped from the --guarantee-valid retry path."
    )
    assert "--- End Validation Report ---" in caplog.text, (
        "Missing DEBUG report footer - enumeration block is unbalanced."
    )
    for v in canned_violations:
        assert v in caplog.text, f"Canned violation {v!r} not enumerated at DEBUG"

    # And those specific lines must have come through at DEBUG level - not at
    # warning - so they don't spam normal runs. Check via the records, not text.
    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    for v in canned_violations:
        assert v in debug_messages, (
            f"Violation {v!r} appeared in logs but not at DEBUG level - "
            "regression: enumeration is leaking at a higher level."
        )


def test_best_of_n_debug_silent_at_info_level(
    mocker: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """At INFO level, the per-violation enumeration must NOT appear - only the
    summary warning. Guards against the enumeration accidentally being logged
    above DEBUG and flooding normal output."""
    caplog.set_level(logging.INFO)

    canned_violations = ["mock violation gamma: bond length anomaly"]

    minimal_pdb = (
        "HEADER    stub\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    mocker.patch(
        "synth_pdb.main.generate_pdb_content",
        side_effect=[minimal_pdb, minimal_pdb],
    )
    fake_first = MagicMock()
    fake_first.get_violations.return_value = canned_violations
    fake_first.validate_all.return_value = None
    fake_second = MagicMock()
    fake_second.get_violations.return_value = []
    fake_second.validate_all.return_value = None
    mocker.patch(
        "synth_pdb.main.PDBValidator",
        side_effect=[fake_first, fake_second],
    )
    mocker.patch("sys.exit")

    test_args = [
        "synth_pdb",
        "--length",
        "1",
        "--guarantee-valid",
        "--max-attempts",
        "2",
        "--output",
        str(tmp_path / "test.pdb"),
        "--log-level",
        "INFO",
    ]
    mocker.patch("sys.argv", test_args)

    main.main()

    # Summary warning should appear; the DEBUG enumeration should not.
    assert "1 violations. Retrying..." in caplog.text
    assert "--- PDB Validation Report for failed attempt ---" not in caplog.text, (
        "DEBUG-only enumeration block leaked at INFO level."
    )
    for v in canned_violations:
        assert v not in caplog.text, f"Per-violation enumeration leaked at INFO: {v!r}"
