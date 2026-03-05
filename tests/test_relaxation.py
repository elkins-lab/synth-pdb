import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.relaxation import calculate_relaxation_rates, spectral_density
import logging

logger = logging.getLogger(__name__)

def test_spectral_density_function():
    """Test standard J(w) behavior."""
    # Tests that J(w) decreases with frequency
    tau_m = 10e-9 # 10ns
    s2 = 0.85 # Define order parameter (rigid)
    
    j_0 = spectral_density(0, tau_m, s2)
    j_high = spectral_density(1e9, tau_m, s2)
    
    assert j_0 > 0
    assert j_high > 0
    assert j_0 > j_high # Spectral density decays at high frequency

def test_relaxation_trends():
    """Test that rigid regions have different rates than flexible ones."""
    # Create dummy structure: 3 residues
    # 1 and 3 are termini (flexible S2=0.5), 2 is center (rigid S2=0.85)
    
    structure = struc.AtomArray(15)
    # 5 Residues: 1, 2, 3, 4, 5
    # 1,5 = Termini (0.5)
    # 2,4 = Penultimate (0.7)
    # 3 = Core (0.85)
    
    ids = []
    for i in range(1, 6):
        ids.extend([i, i, i])
    
    structure.res_id = np.array(ids)
    structure.res_name = np.array(["ALA"]*15)
    structure.atom_name = np.array(["N", "CA", "H"]*5)
    
    # Set seed for reproducibility of random noise
    np.random.seed(42)
    
    rates = calculate_relaxation_rates(structure, field_mhz=600, tau_m_ns=10.0)
    
    s2_term = rates[1]['S2']
    s2_core = rates[3]['S2']
    
    noe_term = rates[1]['NOE']
    noe_core = rates[3]['NOE']
    
    logger.info(f"Term S2: {s2_term}, Core S2: {s2_core}")
    logger.info(f"Term NOE: {noe_term}, Core NOE: {noe_core}")
    
    # Core should be more rigid (Higher S2)
    # With new predict_order_parameters, termini get ~0.45, core ~0.85 (if alpha) or 0.65 (if coil)
    # The dummy structure has no secondary structure (Phi/Psi=0/NaN?), so it might default to Coil or Termini
    # Termini logic overrides.
    assert s2_core > s2_term
    
    # PHYSICS NOTE:
    # Rigid (High S2) -> Larger R2 (faster transverse decay)
    # R2 ~ S2 * tau_m
    assert rates[3]['R2'] > rates[1]['R2']
    
    # PHYSICS (fixed):
    # With per-residue tau_f, the fast-motion (1-S2) term breaks the S2 cancellation in
    # the NOE ratio. Flexible termini (low S2, longer tau_f) now correctly produce a
    # lower HetNOE than rigid core residues.
    # Reference: Lipari & Szabo (1982) J Am Chem Soc 104:4546.
    assert noe_core > noe_term

def test_proline_exclusion():
    """Ensure Prolines are skipped (no amide proton)."""
    structure = struc.AtomArray(3)
    structure.res_id = [1, 1, 1]
    structure.res_name = ["PRO", "PRO", "PRO"]
    structure.atom_name = ["N", "CA", "CD"] # No H
    
    rates = calculate_relaxation_rates(structure)
    assert len(rates) == 0

