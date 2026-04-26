import logging
from typing import Any, cast

import requests

try:
    import pynmrstar
except ImportError:
    pynmrstar = None

logger = logging.getLogger(__name__)


class BMRBAPI:
    """Interface to Biological Magnetic Resonance Data Bank (BMRB) API.

    This provides empirical validation data for NMR structures by fetching
    peer-reviewed experimental restraints and chemical shifts.
    """

    BASE_URL = "https://api.bmrb.io/v2"

    @staticmethod
    def get_entry_metadata(bmrb_id: str) -> dict[str, Any]:
        """Fetch metadata for a BMRB entry.

        Args:
            bmrb_id: BMRB ID (e.g., '6457' for Ubiquitin).
        """
        url = f"{BMRBAPI.BASE_URL}/entry/{bmrb_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return cast(dict[str, Any], data.get(bmrb_id, {}))

    @staticmethod
    def search_entries_with_restraints(search_term: str = "ubiquitin") -> list[str]:
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
    def fetch_restraints(bmrb_id: str) -> list[dict[str, Any]]:
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
                        if all(
                            idx is not None for idx in [idx_res1, idx_atom1, idx_res2, idx_atom2]
                        ):
                            restraints.append(
                                {
                                    "id": row[idx_id] if idx_id < len(row) else None,
                                    "index_1": int(row[idx_res1]),
                                    "atom_name_1": row[idx_atom1],
                                    "index_2": int(row[idx_res2]),
                                    "atom_name_2": row[idx_atom2],
                                    "upper_limit": (
                                        float(row[idx_upper])
                                        if idx_upper and row[idx_upper]
                                        else 5.0
                                    ),
                                }
                            )
                    if restraints:
                        return restraints
            except Exception as e:
                logger.debug(f"Failed to fetch {cat} for {bmrb_id}: {e}")
                continue

        return []

    @staticmethod
    def fetch_chemical_shifts(bmrb_id: str) -> dict[int, dict[str, float]]:
        """Fetch chemical shifts from BMRB using pynmrstar.

        Returns:
            Dict[int, Dict[str, float]]: Mapping of res_id -> {atom: value}.
        """
        if pynmrstar is None:
            logger.error("pynmrstar not installed. Cannot fetch shifts.")
            return {}
        try:
            entry = pynmrstar.Entry.from_database(bmrb_id)
            loops = entry.get_loops_by_category("_Atom_chem_shift")
            if not loops:
                return {}

            loop = loops[0]
            tag_to_idx = {tag.lower(): i for i, tag in enumerate(loop.tags)}

            idx_res = tag_to_idx.get("comp_index_id")
            if idx_res is None:
                idx_res = tag_to_idx.get("seq_id")

            idx_atom = tag_to_idx.get("atom_id")
            idx_val = tag_to_idx.get("val")

            shifts: dict[int, dict[str, float]] = {}
            for row in loop.data:
                try:
                    if idx_res is None or idx_atom is None or idx_val is None:
                        continue
                    res_id = int(row[idx_res])
                    atom_name = row[idx_atom]
                    val = float(row[idx_val])

                    if atom_name == "H":
                        atom_name = "HN"

                    if res_id not in shifts:
                        shifts[res_id] = {}
                    shifts[res_id][atom_name] = val
                except (ValueError, TypeError):
                    continue
            return shifts
        except Exception as e:
            logger.error(f"Failed to fetch shifts for BMRB {bmrb_id}: {e}")
            return {}

    @staticmethod
    def download_pdb(pdb_id: str, output_path: str) -> bool:
        """Download a PDB file from RCSB.

        Args:
            pdb_id: 4-character PDB ID.
            output_path: Destination file path.
        """
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_path, "w") as f:
                    f.write(response.text)
                return True
        except Exception as e:
            logger.error(f"Failed to download PDB {pdb_id}: {e}")
        return False


class PDBValidationAPI:
    """Interface to PDBe Validation API for geometric assessment.

    Provides peer-reviewed geometric quality metrics compared to the entire PDB.
    """

    BASE_URL = "https://www.ebi.ac.uk/pdbe/api/validation"

    @staticmethod
    def get_validation_summary(pdb_id: str) -> dict[str, Any]:
        """Fetch validation summary for an existing PDB entry."""
        url = f"{PDBValidationAPI.BASE_URL}/summary/entry/{pdb_id.lower()}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return cast(dict[str, Any], response.json().get(pdb_id.lower(), {}))
        except Exception as e:
            logger.error(f"Failed to fetch PDBe summary for {pdb_id}: {e}")
            return {}

    @staticmethod
    def get_validation_outliers(pdb_id: str) -> dict[str, Any]:
        """Fetch detailed geometric outliers (Ramachandran, etc.) for a PDB entry."""
        url = f"{PDBValidationAPI.BASE_URL}/outliers/entry/{pdb_id.lower()}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return cast(dict[str, Any], response.json().get(pdb_id.lower(), {}))
        except Exception as e:
            logger.error(f"Failed to fetch PDBe outliers for {pdb_id}: {e}")
            return {}
