import numpy as np
import pytest
import io
import biotite.structure.io.pdb as pdb
from synth_pdb.j_coupling import calculate_hn_ha_coupling
from synth_pdb.generator import generate_pdb_content


def test_calculate_hn_ha_coupling_shim() -> None:
    """Test that the j_coupling shim correctly calls the underlying nmr logic."""
    # Generate a simple alpha helix
    content = generate_pdb_content(length=5, conformation="alpha")
    f = pdb.PDBFile.read(io.StringIO(content))
    structure = f.get_structure(model=1)

    # Returns Dict[Chain][ResID]
    results = calculate_hn_ha_coupling(structure)

    assert "A" in results
    # Residue 1 usually doesn't have an HN so might be missing, check res 2
    if 2 in results["A"]:
        val = results["A"][2]
        # Helix J values are typically 3-6 Hz
        assert 3.0 < val < 6.0
