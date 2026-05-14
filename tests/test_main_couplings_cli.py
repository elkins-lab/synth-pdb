"""Regression tests for the --gen-couplings CLI export.

Guards three things that have all been broken at some point:
- the call site actually runs without raising (master and an earlier mmcif-support
  revision both crashed inside the J-coupling block because the wrong API was used);
- the CSV schema stays ``res_id,residue,J_HN_HA`` so downstream tooling keyed on that
  column layout doesn't silently break;
- residues that legitimately have no 3J(HN-HA) coupling (first residue lacks phi,
  prolines lack an amide proton) are written as ``nan`` rows rather than silently
  omitted, so the file has one row per residue.
"""

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

from synth_pdb.main import main


SEQ = "ALA-GLY-PRO-PHE-ALA"
SEQ_RESIDUES = ["ALA", "GLY", "PRO", "PHE", "ALA"]


def _run_gen_couplings(tmp_path: Path) -> Path:
    """Run the CLI with --gen-couplings on SEQ and return the CSV path."""
    out_pdb = tmp_path / "test.pdb"
    csv_path = tmp_path / "test_j.csv"
    test_args = [
        "synth_pdb",
        "--sequence",
        SEQ,
        "--gen-couplings",
        "--coupling-output",
        str(csv_path),
        "--output",
        str(out_pdb),
    ]
    with patch("sys.argv", test_args):
        main()
    return csv_path


def test_gen_couplings_csv_is_written(tmp_path: Path) -> None:
    """The CLI must actually produce the CSV (regresses the broken-call bug)."""
    csv_path = _run_gen_couplings(tmp_path)
    assert csv_path.exists(), "CSV not produced by --gen-couplings"


def test_gen_couplings_csv_header_schema(tmp_path: Path) -> None:
    """Schema must be res_id,residue,J_HN_HA — third column is the floating point value."""
    csv_path = _run_gen_couplings(tmp_path)
    header = csv_path.read_text().splitlines()[0]
    assert header == "res_id,residue,J_HN_HA", f"Unexpected header: {header!r}"


def test_gen_couplings_one_row_per_residue(tmp_path: Path) -> None:
    """One data row per input residue — prolines and N-term must not be silently skipped."""
    csv_path = _run_gen_couplings(tmp_path)
    data_rows = csv_path.read_text().splitlines()[1:]
    assert len(data_rows) == len(
        SEQ_RESIDUES
    ), f"Expected {len(SEQ_RESIDUES)} rows for {SEQ}, got {len(data_rows)}: {data_rows}"


def test_gen_couplings_first_residue_is_nan(tmp_path: Path) -> None:
    """First residue has no phi angle so J(HN-HA) must be NaN, not absent or 0."""
    csv_path = _run_gen_couplings(tmp_path)
    rows = [r.split(",") for r in csv_path.read_text().splitlines()[1:]]
    rid, res, jval = rows[0]
    assert rid == "1"
    assert res == "ALA"
    assert math.isnan(float(jval)), f"Row 1 J-coupling should be NaN, got {jval!r}"


def test_gen_couplings_proline_is_nan(tmp_path: Path) -> None:
    """Proline has no backbone amide proton — must appear in CSV as NaN, not be omitted."""
    csv_path = _run_gen_couplings(tmp_path)
    rows = [r.split(",") for r in csv_path.read_text().splitlines()[1:]]
    pro_rows = [r for r in rows if r[1] == "PRO"]
    assert pro_rows, "PRO row missing from CSV — proline was silently dropped"
    for rid, _res, jval in pro_rows:
        assert math.isnan(float(jval)), (
            f"PRO at residue {rid} should be NaN, got {jval!r} — "
            "proline lacks backbone amide proton so 3J(HN-HA) is physically undefined"
        )


def test_gen_couplings_interior_residue_is_finite(tmp_path: Path) -> None:
    """A non-proline interior residue must produce a finite, plausible J value."""
    csv_path = _run_gen_couplings(tmp_path)
    rows = [r.split(",") for r in csv_path.read_text().splitlines()[1:]]
    # PHE at residue 4 — has phi (not first), not proline
    phe_rows = [r for r in rows if r[1] == "PHE"]
    assert phe_rows, "PHE row missing"
    jval = float(phe_rows[0][2])
    assert math.isfinite(jval), f"PHE J-coupling should be finite, got {jval!r}"
    # Karplus output should sit in a sane biophysical range (~0-12 Hz)
    assert 0.0 <= jval <= 12.0, f"PHE J-coupling {jval} outside plausible range 0-12 Hz"
