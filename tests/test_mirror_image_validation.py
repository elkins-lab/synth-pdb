import pytest

from synth_pdb.bmrb_api import BMRBAPI
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


@pytest.mark.network
def test_mirror_image_protein_g_validation() -> None:
    """Aggressive validation: Validate mirror-image Protein G (D-GB1).

    SCIENTIFIC BASIS:
    PDB 7W99 / BMRB 36464 is the mirror image of the B1 domain of Protein G.
    It is composed entirely of D-amino acids.

    A scientifically accurate validator must:
    1. Detect the D-chirality correctly.
    2. Handle D-amino acids placed on a standard backbone (L-conformation).
    """
    bmrb_id = "36464"

    # 1. Fetch metadata to confirm it's the right entry
    metadata = BMRBAPI.get_entry_metadata(bmrb_id)
    assert metadata["entry_id"] == bmrb_id

    # 2. Sequence for GB1 (D-form)
    gb1_seq = "MTFKLIINGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

    # 3. Generate a D-protein model
    from synth_pdb.data import ONE_TO_THREE_LETTER_CODE

    d_sequence_parts = []
    for aa in gb1_seq:
        three_letter = ONE_TO_THREE_LETTER_CODE.get(aa, "ALA")
        if aa != "G":
            d_sequence_parts.append(f"D-{three_letter}")
        else:
            d_sequence_parts.append(three_letter)

    d_seq_str = "-".join(d_sequence_parts)
    print(f"Generating D-Protein model for sequence string: {d_seq_str[:30]}...")
    pdb_content = generate_pdb_content(sequence_str=d_seq_str)

    # 4. Validate Chirality
    validator = PDBValidator(pdb_content=pdb_content)
    validator.validate_chirality()

    # Correct implementation: 0 chirality violations for D-residues
    chirality_violations = [v for v in validator.violations if "Chirality" in v]
    assert (
        len(chirality_violations) == 0
    ), f"Should have 0 chirality violations, got: {chirality_violations}"

    # 5. Backbone Check
    # By default, the generator keeps the standard backbone (negative Phi/Psi)
    # but mirrors the sidechains to create D-amino acids.
    phi_psi = validator.calculate_all_phi_psi()
    for res_num in sorted(phi_psi.keys())[:5]:
        phi, psi = phi_psi[res_num]
        assert phi < 0, f"Residue {res_num} should have negative Phi (L-conformation backbone)"


@pytest.mark.network
def test_bmrb_36464_chemical_shift_stats() -> None:
    """Verify that BMRB 36464 has the expected chemical shift density for validation."""
    bmrb_id = "36464"
    metadata = BMRBAPI.get_entry_metadata(bmrb_id)

    found_shifts = False
    for sf in metadata.get("saveframes", []):
        if sf.get("category") == "assigned_chemical_shifts":
            found_shifts = True
            break

    assert found_shifts, "BMRB 36464 should contain assigned chemical shifts"
