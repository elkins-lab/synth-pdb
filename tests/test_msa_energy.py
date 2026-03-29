"""Tests for the Potts Model Pseudo-Energy function used in Synthetic MSA generation.
Validated against statistical physics literature on Direct Coupling Analysis (DCA).
"""

import numpy as np

from synth_pdb.msa import CoevolutionModel


def test_potts_model_initialization():
    """Verify that the model properly initializes Fields (h_i) and Couplings (J_ij)."""
    # 5-residue sequence
    base_sequence = "AGAAG"

    # Contact map: only Residue 0 and Residue 4 are in physical contact
    contact_map = np.zeros((5, 5), dtype=bool)
    contact_map[0, 4] = True
    contact_map[4, 0] = True

    model = CoevolutionModel(base_sequence, contact_map)

    # Fields should be (L, 20)
    assert model.fields.shape == (5, 20)

    # Couplings should be (L, L, 20, 20)
    assert model.couplings.shape == (5, 5, 20, 20)

    # Verify that ONLY the defined contacts have non-zero coupling energy potential
    assert np.any(model.couplings[0, 4] != 0.0)
    assert np.all(model.couplings[0, 1] == 0.0)


def test_potts_energy_compensatory_mutation():
    """Test the core thesis of Co-evolution:
    1. Base Ground State = Low Energy
    2. Single Disruption = High Energy
    3. Compensatory Disruption = Low Energy (Rescue).
    """
    base_sequence = "AGTTG"

    # Residue 0 and 4 are in tight physical contact in the 3D structure
    contact_map = np.zeros((5, 5), dtype=bool)
    contact_map[0, 4] = True
    contact_map[4, 0] = True

    model = CoevolutionModel(base_sequence, contact_map)

    # E0: Native Sequence (Ala-Gly)
    e_native = model.calculate_energy(base_sequence)

    # E1: Mutate Residue 0 from small Ala to massive Trp
    # AND mutuate Residue 4 from small Gly to massive Trp
    # This should create a severe steric clash (volume 4 + 4 = 8 > 5)
    clash_sequence = "WGTTW"
    e_clash = model.calculate_energy(clash_sequence)
    assert (
        e_clash > e_native
    ), "Double steric mutation should drastically increase potential energy."

    # E2: Compensatory Mutation. Residue 4 mutates to the smallest amino acid (Gly)
    # to make room for the massive Trp at Residue 0. (volume 4 + 0 = 4 < 5)
    rescue_sequence = "WGTTG"
    e_rescue = model.calculate_energy(rescue_sequence)

    assert e_rescue < e_clash, "A compensatory volume mutation must lower the coupling energy."
