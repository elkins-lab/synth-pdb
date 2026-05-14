"""Evolutionary Simulation Module for Synthetic PDB.

This module implements "In Silico Evolution" to generate synthetic Multiple Sequence Alignments (MSAs).
It uses a Structural Bioinformatics approach:
1.  **SASA Calculation**: Identify which residues are buried in the hydrophobic core vs exposed to solvent.
2.  **Neutral Drift Simulation**: Mutate the sequence while applying selection pressure based on the structural environment.
    - **Core**: High selection pressure. Mutations restricted to hydrophobic residues to preserve stability.
    - **Surface**: Low selection pressure. Mutations allowed to drift more freely.

Theoretical Basis:
- **Neutral Theory of Molecular Evolution** (Kimura, 1968): Most mutations are effectively neutral and fixate via genetic drift.
- **Structural Constraints**: The fold of the protein is more conserved than the sequence. The core packing is the primary constraint.
"""

import logging
import random  # We use random for stochastic mutation

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)

# --- Educational Data: Amino Acid Properties ---
# For simulation, we classify amino acids to define "safe" mutations.

# Hydrophobic Core residues (Non-polar)
CORE_ALLOWED = {"A", "V", "I", "L", "M", "F", "W", "Y"}
# Note: Y/W are aromatic but can be buried. A is small and ambivalent.

# Surface residues (Polar/Charged/All)
# In reality any AA can be on surface, but polar/charged are favored.
# For simplicity in Neutral Drift, we allow ANY mutation on surface, assuming water accommodates.
ALL_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def calculate_relative_sasa(atom_array: struc.AtomArray) -> np.ndarray:
    """Calculate the Relative Solvent Accessible Surface Area (rSASA) for each residue.

    Why Relative?
    - Absolute SASA (in A^2) depends on the size of the amino acid.
    - rSASA gives the fraction of the residue exposed, normalized by its theoretical max exposure in a tripeptide (Gly-X-Gly).
    - Rule of thumb: rSASA < 0.2 is "Buried".

    Args:
        atom_array: The full atomic structure.

    Returns:
        np.array: Array of rSASA values (0.0 to 1.0) for each residue.

    """
    logger.info("Calculating Solvent Accessible Surface Area (SASA)...")

    # 1. Calculate Absolute SASA using Shrake-Rupley algorithm
    # Biotite provides this. It requires vdW radii.
    try:
        # Get standard vdW radii for atoms
        # Biotite 0.38+ supports this
        sasa = struc.sasa(atom_array, probe_radius=1.4)  # 1.4A is radius of water molecule
    except Exception as e:
        logger.warning(f"SASA Check failed ({e}). Falling back to dummy SASA (all exposed).")
        # Fallback if libraries missing or structure invalid
        residue_count = struc.get_residue_count(atom_array)
        return np.ones(residue_count)

    # 2. Sum SASA per residue
    res_sasa = struc.apply_residue_wise(atom_array, sasa, np.sum)

    # 3. Normalize (Approximation for educational purposes)
    # We use roughly approximated max SASA values from literature (Tien et al, 2013)
    # For a rigorous implementation, we'd lookup exact values.
    # Here we define an approximate average max to assume "normalization".
    # Simplified approach: If SASA > 20 A^2, it's exposed.
    # We return normalized-ish values or just a raw fraction if we had maxes.

    # Let's map strict "Buried" threshold to return binary-like logic for the mutator,
    # or implement real max values.
    # Implementing approximate Max SASA (in A^2) for X in G-X-G
    max_sasa = {
        "A": 121,
        "R": 265,
        "N": 187,
        "D": 187,
        "C": 148,
        "Q": 214,
        "E": 214,
        "G": 97,
        "H": 216,
        "I": 195,
        "L": 191,
        "K": 230,
        "M": 203,
        "F": 228,
        "P": 154,
        "S": 143,
        "T": 163,
        "W": 264,
        "Y": 255,
        "V": 165,
    }

    # Get residue names to look up max
    # We iterate residue starts
    res_names = [atom_array.res_name[i] for i in struc.get_residue_starts(atom_array)]
    from .data import ONE_TO_THREE_LETTER_CODE

    three_to_one = {v: k for k, v in ONE_TO_THREE_LETTER_CODE.items()}

    rel_sasa = []
    for i, area in enumerate(res_sasa):
        res_3 = res_names[i]
        res_1 = three_to_one.get(res_3, "A")
        max_area = max_sasa.get(res_1, 200.0)
        rel_sasa.append(min(1.0, area / max_area))

    return np.array(rel_sasa)


def generate_msa_sequences(
    structure: struc.AtomArray,
    n_seqs: int = 100,
    mutation_rate: float = 0.1,
    conservation_threshold: float = 0.2,
) -> list[str]:
    """Generate synthetic homologs via simulated neutral drift.

    Args:
        structure: Source structure.
        n_seqs: Number of sequences to generate.
        mutation_rate: Probability of mutation per position per sequence.
        conservation_threshold: rSASA below which a residue is considered BURIED.

    Returns:
        List of sequence strings.

    """
    # 1. Analyze Structure
    rel_sasa = calculate_relative_sasa(structure)

    # Get initial sequence
    res_names = [structure.res_name[i] for i in struc.get_residue_starts(structure)]
    from .data import ONE_TO_THREE_LETTER_CODE

    three_to_one = {v: k for k, v in ONE_TO_THREE_LETTER_CODE.items()}
    initial_seq = [three_to_one.get(r, "A") for r in res_names]

    msa = []
    logger.info(f"Simulating evolution for {n_seqs} generations...")

    for _ in range(n_seqs):
        # Start from original (Star phylogeny) or evolve cumulatively?
        # Star phylogeny (all independent variants of WT) is simpler for training data usually.
        # But real evolution is a tree.
        # Let's do Star Phylogeny for AlphaFold inputs (simulating independent homologs found in DB).

        new_seq = list(initial_seq)

        for i, _ref_aa in enumerate(initial_seq):
            # Roll for mutation
            if random.random() < mutation_rate:
                # Decide allowed mutations based on structure
                is_buried = rel_sasa[i] < conservation_threshold

                if is_buried:
                    # Burial Constraint: Must result in hydrophobic residue
                    choice = random.choice(list(CORE_ALLOWED))
                else:
                    # Surface: Free mutation (Neutral drift)
                    choice = random.choice(ALL_AMINO_ACIDS)

                new_seq[i] = choice

        msa.append("".join(new_seq))

    return msa


def write_msa(sequences: list[str], filename: str) -> None:
    """Write MSA in FASTA format."""
    with open(filename, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n")
            f.write(f"{seq}\n")
    logger.info(f"MSA written to {filename}")
