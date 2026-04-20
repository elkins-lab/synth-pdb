import os
import tempfile

import pytest

from synth_pdb.rdc import read_rdc_file


def test_read_rdc_file_success() -> None:
    """Verify parsing of a valid RDC file."""
    content = """# Residue1 Atom1 Residue2 Atom2 Value
1 N 1 H 10.5
2 N 2 H -5.2

3 N 3 H 12.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rdc", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        rdcs = read_rdc_file(temp_path)
        assert len(rdcs) == 3
        assert rdcs[0] == {"res_1": 1, "atom_1": "N", "res_2": 1, "atom_2": "H", "value": 10.5}
        assert rdcs[1]["value"] == -5.2
        assert rdcs[2]["res_1"] == 3
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_read_rdc_file_not_found() -> None:
    """Verify FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        read_rdc_file("non_existent_file.rdc")


def test_read_rdc_file_invalid_format() -> None:
    """Verify ValueError for malformed data."""
    content = "1 N 1 H not_a_float"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rdc", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Failed to parse RDC file"):
            read_rdc_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_read_rdc_file_too_few_columns() -> None:
    """Verify that lines with too few columns are ignored."""
    content = "1 N 1 H"  # Missing value
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rdc", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        rdcs = read_rdc_file(temp_path)
        assert len(rdcs) == 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
