from unittest.mock import MagicMock, patch
import pytest
from synth_pdb.bmrb_api import BMRBAPI, PDBValidationAPI


def test_bmrb_search_failure() -> None:
    """Test search failure handling."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")
        results = BMRBAPI.search_entries_with_restraints("test")
        assert results == []


def test_bmrb_fetch_restraints_404() -> None:
    """Test fetch restraints with 404 response."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 404
        results = BMRBAPI.fetch_restraints("6457")
        assert results == []


def test_bmrb_download_pdb_failure() -> None:
    """Test PDB download failure."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Network error")
        success = BMRBAPI.download_pdb("1D3Z", "test.pdb")
        assert success is False


def test_pdbe_summary_failure() -> None:
    """Test PDBe validation summary failure."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("API down")
        summary = PDBValidationAPI.get_validation_summary("1D3Z")
        assert summary == {}


def test_pdbe_outliers_failure() -> None:
    """Test PDBe validation outliers failure."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("API down")
        outliers = PDBValidationAPI.get_validation_outliers("1D3Z")
        assert outliers == {}
