import os
import tempfile

import pytest

from synth_pdb.chemical_shifts import read_shift_file


def test_read_shift_file_success() -> None:
    """Verify parsing of a valid chemical shift file."""
    content = """# ResID AtomName Value
1 N 120.5
1 H 8.2

2 N 118.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".str", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        shifts = read_shift_file(temp_path)
        assert len(shifts) == 3
        assert shifts[0] == {"res_id": 1, "atom_name": "N", "value": 120.5}
        assert shifts[2]["res_id"] == 2
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_read_shift_file_whitespace_and_comments() -> None:
    """Verify robust parsing of varied whitespace and comments."""
    content = """
    # Multiple comment lines
    # To test robustness

    10    CA    55.20

    11    CB    32.1    # Inline comment should be ignored (if handled by line.split)

    # Another comment
    12    N     115.5
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".str", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        shifts = read_shift_file(temp_path)
        assert len(shifts) == 3
        assert shifts[0]["res_id"] == 10
        assert shifts[1]["atom_name"] == "CB"
        assert shifts[2]["value"] == 115.5
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_read_shift_file_not_found() -> None:
    """Verify FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_shift_file("missing_shifts.str")


def test_read_shift_file_invalid_format() -> None:
    """Verify ValueError for bad data."""
    content = "1 N not_a_float"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".str", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Failed to parse shift file"):
            read_shift_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_read_shift_file_too_few_columns() -> None:
    """Verify that lines with < 3 columns are ignored."""
    content = "1 N"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".str", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        shifts = read_shift_file(temp_path)
        assert len(shifts) == 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
