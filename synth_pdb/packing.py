import logging

import biotite.structure as struc
import numpy as np

from .data import ROTAMER_LIBRARY
from .geometry import reconstruct_sidechain
from .scoring import calculate_clash_score

logger = logging.getLogger(__name__)

class SideChainPacker:
    """
    Optimizes amino acid side-chain conformations to minimize steric clashes.
    Uses a Monte Carlo approach with the specialized rotamer library.

    # EDUCATIONAL NOTE - Monte Carlo Optimization
    # How do we find the best shape for a protein?
    #
    # We could try every combination, but that's impossible (Combinatorial Explosion).
    # Instead, we use "Monte Carlo" simulation:
    # 1. Make a random change (like rotating a side chain).
    # 2. Measure the energy (Clash Score).
    # 3. If energy is lower (better), ACCEPT the move.
    # 4. If energy is higher (worse), we might still accept it based on probability.
    #
    # Why accept a worse state? To escape "Local Minima".
    # Imagine being stuck in a small pothole while trying to reach the bottom of the Grand Canyon.
    # You need to climb UP out of the pothole to go down further.
    #
    # This probability is given by the Boltzmann factor: P = exp(-DeltaE / Temperature)
    """

    def __init__(self, steps: int = 500, temperature: float = 0.5) -> None:
        """
        Args:
            steps: Number of Monte Carlo steps (attempts).
            temperature: Simulation temperature (proportional to acceptance probability of worse steps).
                         Higher temp = more likely to accept clashes (good for escaping local minima).
        """
        self.steps = steps
        self.temperature = temperature

    def optimize(self, peptide: struc.AtomArray) -> struc.AtomArray:
        """
        Run the optimization protocol on the given peptide structure.
        Modifies the structure in-place.
        
        Args:
            peptide: The input structure to optimize.
            
        Returns:
            The optimized structure (reference to input).
        """
        logger.info(f"Starting side-chain packing optimization ({self.steps} steps)...")

        # 1. Identify valid residues for optimization
        # (Those that have rotamer options in our library)
        residue_ids = sorted(list(set(peptide.res_id)))
        optimizable_residues = []

        # Build map of resid -> res_name
        # Using CA atom to identify residue name
        ca_atoms = peptide[peptide.atom_name == "CA"]
        res_map = {atom.res_id: atom.res_name for atom in ca_atoms}

        for res_id in residue_ids:
            res_name = res_map.get(res_id)
            if res_name in ROTAMER_LIBRARY and ROTAMER_LIBRARY[res_name]:
                optimizable_residues.append((res_id, res_name))

        if not optimizable_residues:
            logger.info("No optimizable residues found.")
            return peptide

        # Initial score
        current_score = calculate_clash_score(peptide)
        best_score = current_score

        # We need a way to revert changes if we reject a step.
        # Since 'reconstruct_sidechain' modifies in place, we can:
        # A) Make a full copy of the structure every step (Slow for large proteins, OK for peptides)
        # B) Store the old coordinates of the changing residue only.

        # For 'synth-pdb' (educational peptides), structure is small. Copying is acceptable for code simplicity.
        # But efficiently: store old coords of relevant atoms.

        logger.info(f"Initial Clash Score: {current_score:.4f}")

        accepted_moves = 0

        for step in range(self.steps):
            # Pick random residue
            target_res_id, target_res_name = optimizable_residues[np.random.randint(len(optimizable_residues))]

            # Pick random new rotamer
            rotamer_options = ROTAMER_LIBRARY[target_res_name]
            weights = [r.get('prob', 1.0) for r in rotamer_options]
            # Normalize weights
            weights = np.array(weights) / np.sum(weights)

            # Select index
            rotamer_idx = np.random.choice(len(rotamer_options), p=weights)
            new_rotamer = rotamer_options[rotamer_idx]

            # Save state (coordinates of this residue)
            # Efficient backup: find indices
            mask = peptide.res_id == target_res_id
            indices = np.where(mask)[0]
            old_coords = peptide.coord[indices].copy()

            # Apply move
            try:
                # The reconstruct_sidechain will need to get the template residue.
                # Since we can't change it there easily without affecting other code,
                # let's modify it here if needed, or check if reconstruct_sidechain
                # calls it correctly. Wait, reconstruct_sidechain is imported.
                # It's likely failing inside reconstruct_sidechain? Or is it failing here?
                # The traceback says: packing.py:117 Failed to apply rotamer: residue() takes 1 positional argument but 2 were given
                # But line 117 is the log message! The error happens in reconstruct_sidechain call at line 115.
                # Let's check geometry.py:reconstruct_sidechain
                reconstruct_sidechain(peptide, target_res_id, new_rotamer, target_res_name)
            except Exception as e:
                logger.warning(f"Failed to apply rotamer: {e}")
                continue

            # Calculate new score
            # Optimization: Only calculate score contribution of this residue?
            # For correctness/simplicity now: Full calculation
            new_score = calculate_clash_score(peptide)

            # Metropolis Criterion
            delta = new_score - current_score

            accept = False
            if delta < 0:
                # Improvement
                accept = True
            else:
                # Worsening - check probability
                # P = exp(-delta / T)
                # If delta is huge (clash), P -> 0.
                if self.temperature > 0:
                    prob = np.exp(-delta / self.temperature)
                    if np.random.random() < prob:
                        accept = True

            if accept:
                current_score = new_score
                accepted_moves += 1
                if new_score < best_score:
                    best_score = new_score
            else:
                # Revert
                peptide.coord[indices] = old_coords

        logger.info(f"Optimization complete. Final Clash Score: {current_score:.4f} (Moves accepted: {accepted_moves})")
        return peptide

def optimize_sidechains(peptide: struc.AtomArray, steps: int = 500) -> struc.AtomArray:
    """Convenience wrapper."""
    packer = SideChainPacker(steps=steps)
    return packer.optimize(peptide)
