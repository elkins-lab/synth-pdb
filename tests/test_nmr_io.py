from unittest.mock import patch

import pytest

from synth_pdb.nmr import read_restraint_file


def test_read_restraint_file_not_found():
    """Test that FileNotFoundError is raised if the file does not exist."""
    with pytest.raises(FileNotFoundError, match="Restraint file not found"):
        read_restraint_file("non_existent_file.txt")


def test_read_restraint_file_whitespace_format(tmp_path):
    """Test parsing of a simple whitespace-separated restraint file.

    SCIENTIFIC BASIS:
    Distance restraints are often formatted as: res_i atom_i res_j atom_j upper_bound.
    A value of 3.5 A is a typical upper bound for a medium-range NOE contact.
    """
    d = tmp_path / "subdir"
    d.mkdir()
    restraint_file = d / "test.restraints"
    content = (
        "# This is a comment\n"
        "1 HN 5 HA 3.5\n"
        "2 HB 3 HB 5.0\n"
        "\n"  # Empty line
        "10 CA 12 CA 6.0 # Inline comment\n"
    )
    restraint_file.write_text(content)

    restraints = read_restraint_file(str(restraint_file))

    assert len(restraints) == 3
    assert restraints[0] == {
        "index_1": 1,
        "atom_name_1": "HN",
        "index_2": 5,
        "atom_name_2": "HA",
        "upper_limit": 3.5,
    }
    assert restraints[1]["upper_limit"] == 5.0
    assert restraints[2]["index_1"] == 10


def test_read_restraint_file_malformed(tmp_path):
    """Test that a ValueError is raised for malformed restraint files."""
    restraint_file = tmp_path / "bad.txt"
    restraint_file.write_text("1 HN 5 HA not_a_float")

    with pytest.raises(ValueError, match="Failed to parse restraint file"):
        read_restraint_file(str(restraint_file))


@patch("synth_pdb.nmr.os.path.exists", return_value=True)
@patch("synth_pdb.nef_io.read_nef_restraints")
def test_read_restraint_file_nef_delegation(mock_read_nef, mock_exists):
    """Test that NEF files are correctly delegated to the NEF parser.

    SCIENTIFIC BASIS:
    NEF (NMR Exchange Format) is the modern standard (IUPAC) for NMR data.
    """
    mock_read_nef.return_value = [
        {"res_i": 1, "atom_i": "H", "res_j": 2, "atom_j": "H", "upper_bound": 5.0}
    ]

    restraints = read_restraint_file("test.nef")

    mock_read_nef.assert_called_once_with("test.nef")
    assert len(restraints) == 1
    assert restraints[0]["res_i"] == 1
