import logging

import numpy as np

from synth_pdb.generator import generate_pdb_content

logger = logging.getLogger(__name__)


def test_bfactor_reflects_dynamics():
    """Verify that B-factors in the generated PDB reflect structural dynamics.

    Helical residues (rigid, high S2) should have LOW B-factors.
    Terminal residues (flexible, low S2) should have HIGH B-factors.
    """
    # Generate a helix
    pdb_content = generate_pdb_content(sequence_str="A" * 15, conformation="alpha", seed=42)

    # Manually parse B-factors from PDB string to avoid Biotite annotation issues
    bfactors = {}
    for line in pdb_content.splitlines():
        if line.startswith("ATOM") and " CA " in line:
            # Extract fields
            # B-factor is columns 61-66 (0-indexed 60-66)
            res_id = int(line[22:26].strip())
            bf_str = line[60:66].strip()
            bf = float(bf_str)
            bfactors[res_id] = bf

    logger.info(f"B-factors: {bfactors}")

    # Terminal residues (1, 2, 14, 15) should have higher B-factors than core (7, 8)
    term_bfactor = np.mean([bfactors[1], bfactors[2], bfactors[14], bfactors[15]])
    core_bfactor = np.mean([bfactors[7], bfactors[8]])

    logger.info(f"Terminal avg B-factor: {term_bfactor:.2f}")
    logger.info(f"Core avg B-factor: {core_bfactor:.2f}")

    # Core should be more rigid (lower B-factor)
    assert core_bfactor < term_bfactor, "Core residues should have lower B-factors than termini"

    # Verify reasonable range (5-100 Angstrom^2)
    for rid, bf in bfactors.items():
        assert 5.0 <= bf <= 100.0, f"B-factor {bf} for residue {rid} out of realistic range"


def test_bfactor_loop_vs_helix():
    """Verify that loop regions have higher B-factors than helical regions."""
    # Generate a structure with explicit regions: helix-loop-helix
    # Use a longer sequence to get more stable averages
    pdb_content = generate_pdb_content(
        sequence_str="A" * 40, structure="1-15:alpha,16-25:random,26-40:alpha", seed=42
    )

    # Manually parse B-factors
    bfactors = {}
    for line in pdb_content.splitlines():
        if line.startswith("ATOM") and " CA " in line:
            res_id = int(line[22:26].strip())
            bf = float(line[60:66].strip())
            bfactors[res_id] = bf

    # Helix regions: 5-10, 30-35 (avoiding termini effects and transition regions)
    helix_res = list(range(5, 11)) + list(range(30, 36))
    loop_res = list(range(18, 23))

    helix_bfactor = np.mean([bfactors[i] for i in helix_res if i in bfactors])
    loop_bfactor = np.mean([bfactors[i] for i in loop_res if i in bfactors])

    logger.info(f"Helix avg B-factor: {helix_bfactor:.2f}")
    logger.info(f"Loop avg B-factor: {loop_bfactor:.2f}")

    # Loop should be more flexible (higher B-factor)
    # We use a small delta to ensure it's meaningfully higher, or just > if we use enough residues
    assert loop_bfactor > helix_bfactor, f"Loop regions ({loop_bfactor:.2f}) should have higher B-factors than helices ({helix_bfactor:.2f})"
