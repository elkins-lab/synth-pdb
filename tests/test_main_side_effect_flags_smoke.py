"""Parametrized smoke tests for CLI flags that produce side-effect output files.

This file deliberately does *not* assert anything about the *contents* of the
output files - narrower regression tests live alongside it. Its job is to catch
the entire "schema rewrite forgot the call site" class of regression at minimal
cost: if `--gen-X` produces no file (because the call site was deleted, or now
raises, or no-ops silently), this fails.

The bug that motivated these tests: a refactor swapped the J-coupling export to
a new API but never validated the call site end-to-end. The CLI crashed inside
a broad ``except Exception`` and the only signal was a generic error log line.
A 2-line smoke test for ``--gen-couplings`` would have caught it instantly.

Each case shares a tiny base CLI invocation (3-residue sequence, no
minimization) so the full sweep stays under a few seconds.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from synth_pdb.main import main


# Each case = (id, extra CLI args list, list of expected output filenames)
# Expected filenames are resolved relative to tmp_path. Filenames that are
# referenced by the --output base name (e.g. "_cd.png" suffix) get the literal
# tmp-path-relative form here.
CASES = [
    (
        "gen-couplings",
        ["--gen-couplings", "--coupling-output", "{tmp}/out_j.csv"],
        ["out_j.csv"],
    ),
    (
        "gen-cd",
        ["--gen-cd"],
        # CDSimulator writes <output-base>_cd.png - base comes from --output.
        ["test_cd.png"],
    ),
    (
        "gen-shifts",
        ["--gen-shifts", "--shift-output", "{tmp}/out_shifts.nef"],
        ["out_shifts.nef"],
    ),
    (
        "export-constraints-csv",
        [
            "--export-constraints",
            "{tmp}/out_c.csv",
            "--constraint-format",
            "csv",
        ],
        ["out_c.csv"],
    ),
    (
        "export-torsion",
        ["--export-torsion", "{tmp}/out_t.csv"],
        ["out_t.csv"],
    ),
    (
        "output-rdcs",
        ["--output-rdcs", "{tmp}/out_rdcs.tbl"],
        ["out_rdcs.tbl"],
    ),
    (
        "gen-msa",
        ["--gen-msa", "--msa-depth", "3"],
        # --gen-msa repurposes --output as the MSA fasta path, so the only file
        # we need to assert is the one we passed via --output below.
        ["test.pdb"],
    ),
]


@pytest.mark.parametrize(
    "case_id,extra_args,expected_outputs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_side_effect_flag_produces_file(
    case_id: str,
    extra_args: list[str],
    expected_outputs: list[str],
    tmp_path: Path,
    capsys: Any,
) -> None:
    """Each --gen-X / --export-X / --output-X flag must produce its named
    output file end-to-end. Sole assertion is existence - narrower tests
    verify schema/content separately."""
    pdb_output = tmp_path / "test.pdb"
    formatted_extra = [a.format(tmp=str(tmp_path)) for a in extra_args]

    test_args = [
        "synth_pdb",
        "--sequence",
        "ALA-GLY-PHE",
        "--output",
        str(pdb_output),
        *formatted_extra,
    ]

    with patch("sys.argv", test_args):
        try:
            main()
        except SystemExit as exc:
            # sys.exit(0) is fine, any non-zero is a smoke failure.
            code = exc.code if isinstance(exc.code, int) else 1
            assert code == 0, f"CLI for {case_id} exited with code {code}"

    for fname in expected_outputs:
        if case_id == "gen-cd" and fname == "test_cd.png":
            try:
                import matplotlib
            except ImportError:
                pytest.skip("matplotlib not installed, skipping CD plot existence check")

        out_path = tmp_path / fname
        assert out_path.exists(), (
            f"CLI flag set {case_id!r} did not produce expected file {fname!r}. "
            f"Existing files in tmp_path: {sorted(p.name for p in tmp_path.iterdir())}"
        )
        # A 0-byte output is almost certainly a regression (the call site
        # opened the file but never wrote - happens when an exception is
        # swallowed mid-write).
        assert (
            out_path.stat().st_size > 0
        ), f"{case_id!r} produced an empty {fname!r} - write path likely failed silently."
