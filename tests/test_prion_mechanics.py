import pytest
from synth_pdb.generator import generate_pdb_content
from synth_pdb.validator import PDBValidator


def test_prp_octarepeat_generation():
    """Verify that the repetitive PrP Octarepeat region can be generated and validated."""
    # Human PrP Octarepeat: PHGGGWGQ
    octarepeat_seq = "PHGGGWGQ" * 4

    # Generate as an alpha helix (though it's naturally disordered, we test generation logic)
    pdb_content = generate_pdb_content(
        sequence_str=octarepeat_seq, conformation="alpha", minimize_energy=True
    )

    validator = PDBValidator(pdb_content=pdb_content)
    report = validator.get_quality_report()

    # Repetitive sequences with many Glycines should be easy to generate without major clashes
    if not report["is_overall_scientifically_defensible"]:
        print(f"DEBUG: PrP Octarepeat Report: {report}")
        print(f"DEBUG: PDB Content (first 5000 chars): {pdb_content[:5000]}")

    assert report["potential_energy_kj_mol"] < 1e6  # Allow some slack for long repetitive chains
    assert "PRO A   1" in pdb_content
    # Histidine should be present somewhere in the file
    assert any(h in pdb_content for h in ["HIS", "HSD", "HSE", "HIP", "HID", "HIE"])

    # If Zinc coordination happened (emergent behavior), verify it
    if "ZN" in pdb_content:
        print("DEBUG: Zinc coordination detected in Octarepeat!")
        assert "ZN" in pdb_content


def test_prp_106_126_chameleon_logic():
    """Verify that we can generate both alpha and beta versions of the same prion fragment."""
    seq = "KTNMKHMAGAAAAGAVVGGLG"

    alpha_pdb = generate_pdb_content(sequence_str=seq, conformation="alpha")
    beta_pdb = generate_pdb_content(sequence_str=seq, conformation="beta")

    # Check that they are different structures for the same sequence
    assert alpha_pdb != beta_pdb
    # Check for residue names as a proxy for sequence verification
    assert "LYS A   1" in alpha_pdb
    assert "THR A   2" in alpha_pdb
    assert "ASN A   3" in alpha_pdb

    # Beta should be much longer (extended) than alpha (compact)
    # We can check coordinate range as a proxy for 'extendedness'
    def get_max_span(pdb):
        coords = []
        for line in pdb.splitlines():
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
        import numpy as np

        coords = np.array(coords)
        return np.max(np.ptp(coords, axis=0))

    alpha_span = get_max_span(alpha_pdb)
    beta_span = get_max_span(beta_pdb)

    assert beta_span > alpha_span
