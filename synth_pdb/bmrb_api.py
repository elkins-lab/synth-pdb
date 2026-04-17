import logging
from typing import Any, Dict, List, cast

import requests

logger = logging.getLogger(__name__)

class BMRBAPI:
    """Interface to Biological Magnetic Resonance Data Bank (BMRB) API.

    This provides empirical validation data for NMR structures by fetching
    peer-reviewed experimental restraints and chemical shifts.
    """

    BASE_URL = "https://api.bmrb.io/v2"

    @staticmethod
    def get_entry_metadata(bmrb_id: str) -> Dict[str, Any]:
        """Fetch metadata for a BMRB entry.

        Args:
            bmrb_id: BMRB ID (e.g., '6457' for Ubiquitin).
        """
        url = f"{BMRBAPI.BASE_URL}/entry/{bmrb_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return cast(Dict[str, Any], data.get(bmrb_id, {}))

    @staticmethod
    def search_entries_with_restraints(search_term: str = "ubiquitin") -> List[str]:
        """Search for BMRB entries that likely have restraint data."""
        url = f"{BMRBAPI.BASE_URL}/search/entry?q={search_term}&field=nmr_star_loop_category&value=_Gen_dist_constraint"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return [item["entry_id"] for item in response.json().get("results", [])]
        except Exception:
            pass
        return []


    @staticmethod
    def fetch_restraints(bmrb_id: str) -> List[Dict[str, Any]]:
        """Fetch distance restraints from BMRB.

        SCIENTIFIC BASIS:
        Restraints in the BMRB are the ground truth for structural modeling.
        Comparing synthetic models against these ensures biological realism.
        """
        # We try both modern _Gen_dist_constraint and legacy names
        categories = ["_Gen_dist_constraint", "Gen_dist_constraint"]

        for cat in categories:
            url = f"{BMRBAPI.BASE_URL}/entry/{bmrb_id}/loop/{cat}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    # Parse the BMRB API v2 response format
                    # Expected format: { "cat_name": { "tags": [...], "data": [[...]] } }
                    loop_key = list(data.keys())[0]
                    tags = data[loop_key]["tags"]
                    rows = data[loop_key]["data"]

                    # Map tags to indices
                    tag_to_idx = {tag.lower(): i for i, tag in enumerate(tags)}

                    # Required tags (handle variations in naming)
                    idx_id = tag_to_idx.get("id") or 0
                    idx_res1 = tag_to_idx.get("auth_seq_id_1") or tag_to_idx.get("res_id_1")
                    idx_atom1 = tag_to_idx.get("atom_id_1") or tag_to_idx.get("atom_name_1")
                    idx_res2 = tag_to_idx.get("auth_seq_id_2") or tag_to_idx.get("res_id_2")
                    idx_atom2 = tag_to_idx.get("atom_id_2") or tag_to_idx.get("atom_name_2")
                    idx_upper = tag_to_idx.get("distance_upper_bound_val")

                    restraints = []
                    for row in rows:
                        if all(idx is not None for idx in [idx_res1, idx_atom1, idx_res2, idx_atom2]):
                            restraints.append({
                                "id": row[idx_id] if idx_id < len(row) else None,
                                "index_1": int(row[idx_res1]),
                                "atom_name_1": row[idx_atom1],
                                "index_2": int(row[idx_res2]),
                                "atom_name_2": row[idx_atom2],
                                "upper_limit": float(row[idx_upper]) if idx_upper and row[idx_upper] else 5.0
                            })
                    if restraints:
                        return restraints
            except Exception as e:
                logger.debug(f"Failed to fetch {cat} for {bmrb_id}: {e}")
                continue

        return []

class PDBValidationAPI:
    """Interface to PDBe Validation API for geometric assessment.

    Provides peer-reviewed geometric quality metrics compared to the entire PDB.
    """

    BASE_URL = "https://www.ebi.ac.uk/pdbe/api/validation"

    @staticmethod
    def get_validation_summary(pdb_id: str) -> Dict[str, Any]:
        """Fetch validation summary for an existing PDB entry."""
        url = f"{PDBValidationAPI.BASE_URL}/summary/entry/{pdb_id.lower()}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return cast(Dict[str, Any], response.json().get(pdb_id.lower(), {}))
        except Exception as e:
            logger.error(f"Failed to fetch PDBe summary for {pdb_id}: {e}")
            return {}

    @staticmethod
    def get_validation_outliers(pdb_id: str) -> Dict[str, Any]:
        """Fetch detailed geometric outliers (Ramachandran, etc.) for a PDB entry."""
        url = f"{PDBValidationAPI.BASE_URL}/outliers/entry/{pdb_id.lower()}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return cast(Dict[str, Any], response.json().get(pdb_id.lower(), {}))
        except Exception as e:
            logger.error(f"Failed to fetch PDBe outliers for {pdb_id}: {e}")
            return {}
