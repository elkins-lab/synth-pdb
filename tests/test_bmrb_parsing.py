from unittest.mock import MagicMock, patch

from synth_pdb.bmrb_api import BMRBAPI, PDBValidationAPI


def test_fetch_restraints_parsing_logic():
    """Test that BMRBAPI correctly parses a raw BMRB-style JSON response.
    This closes the gap where we previously mocked the whole method.
    """
    # Realistic BMRB API v2 response structure for _Gen_dist_constraint
    mock_response_data = {
        "6457": {
            "tags": [
                "ID",
                "Auth_seq_id_1", "Atom_id_1",
                "Auth_seq_id_2", "Atom_id_2",
                "Distance_upper_bound_val"
            ],
            "data": [
                ["1", "1", "H", "2", "H", "5.0"],
                ["2", "43", "H", "44", "H", "3.5"]
            ]
        }
    }

    with patch("requests.get") as mock_get:
        # Mock successful response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_data
        mock_get.return_value = mock_resp

        # Call the method
        restraints = BMRBAPI.fetch_restraints("6457")

        # Verify results
        assert len(restraints) == 2
        assert restraints[0]["index_1"] == 1
        assert restraints[0]["atom_name_1"] == "H"
        assert restraints[0]["upper_limit"] == 5.0
        assert restraints[1]["index_2"] == 44
        assert restraints[1]["upper_limit"] == 3.5

        # Verify that requests.get was called with the expected URL
        # BMRBAPI tries _Gen_dist_constraint first
        mock_get.assert_called_with("https://api.bmrb.io/v2/entry/6457/loop/_Gen_dist_constraint")

def test_fetch_restraints_fallback_logic():
    """Test that BMRBAPI tries legacy category names if the first one fails."""
    with patch("requests.get") as mock_get:
        # First call fails (404), second call succeeds
        mock_resp_404 = MagicMock()
        mock_resp_404.status_code = 404

        mock_resp_success = MagicMock()
        mock_resp_success.status_code = 200
        mock_resp_success.json.return_value = {
            "Gen_dist_constraint": {
                "tags": ["ID", "res_id_1", "atom_name_1", "res_id_2", "atom_name_2", "distance_upper_bound_val"],
                "data": [["1", "10", "HA", "11", "HA", "4.0"]]
            }
        }

        mock_get.side_effect = [mock_resp_404, mock_resp_success]

        restraints = BMRBAPI.fetch_restraints("6457")

        assert len(restraints) == 1
        assert restraints[0]["index_1"] == 10
        assert mock_get.call_count == 2

def test_get_entry_metadata():
    """Test that BMRBAPI correctly fetches and extracts entry metadata."""
    mock_data = {"6457": {"entry_title": "Ubiquitin", "author": "Bax"}}
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        meta = BMRBAPI.get_entry_metadata("6457")
        assert meta["entry_title"] == "Ubiquitin"
        mock_get.assert_called_with("https://api.bmrb.io/v2/entry/6457")

def test_search_entries_with_restraints():
    """Test BMRB entry search logic."""
    mock_data = {"results": [{"entry_id": "6457"}, {"entry_id": "1234"}]}
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        results = BMRBAPI.search_entries_with_restraints("ubiquitin")
        assert len(results) == 2
        assert "6457" in results

def test_pdbe_validation_summary():
    """Test PDBValidationAPI summary fetching."""
    mock_data = {"1ubq": {"summary_metric": 0.95}}
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        summary = PDBValidationAPI.get_validation_summary("1UBQ")
        assert summary["summary_metric"] == 0.95
        mock_get.assert_called_with("https://www.ebi.ac.uk/pdbe/api/validation/summary/entry/1ubq")

def test_pdbe_validation_outliers():
    """Test PDBValidationAPI outliers fetching."""
    mock_data = {"1ubq": {"outliers": []}}
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        outliers = PDBValidationAPI.get_validation_outliers("1UBQ")
        assert "outliers" in outliers
        mock_get.assert_called_with("https://www.ebi.ac.uk/pdbe/api/validation/outliers/entry/1ubq")

def test_api_error_handling(caplog):
    """Verify that APIs handle network errors gracefully by logging and returning defaults."""
    with patch("requests.get", side_effect=Exception("Network down")):
        # BMRB search returns empty list on error
        assert BMRBAPI.search_entries_with_restraints("any") == []

        # PDBValidation returns empty dict on error
        assert PDBValidationAPI.get_validation_summary("1UBQ") == {}
        assert PDBValidationAPI.get_validation_outliers("1UBQ") == {}

        # fetch_restraints returns empty list
        assert BMRBAPI.fetch_restraints("6457") == []
