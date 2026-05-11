import logging

import biotite.structure as struc
import numpy as np

from .data import ROTAMER_LIBRARY
from .geometry import reconstruct_sidechain
from .scoring import calculate_clash_score

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
# We initialize a logger for this module to provide transparent feedback during
# long-running optimization tasks.
logger = logging.getLogger(__name__)


class SideChainPacker:
    """Optimizes amino acid side-chain conformations to minimize steric clashes.
    Uses a Monte Carlo approach with the specialized rotamer library.

    # EDUCATIONAL NOTE - Monte Carlo Optimization
    # ===========================================
    # How do we find the best shape for a protein?
    #
    # We could try every combination of rotamers, but with 10 options per residue
    # and 100 residues, that's 10^100 combinations—impossible (Combinatorial Explosion).
    #
    # Instead, we use "Monte Carlo" simulation, specifically the Metropolis-Hastings
    # algorithm. This mirrors how physical systems find minima:
    #
    # 1. PERTURBATION: Make a random change (e.g., rotating a side chain to a new rotamer).
    # 2. EVALUATION: Measure the potential energy or 'cost' (here, the Clash Score).
    # 3. SELECTION:
    #    - If energy is lower (better), ALWAYS ACCEPT the move.
    #    - If energy is higher (worse), we might still accept it based on probability.
    #
    # WHY ACCEPT A WORSE STATE?
    # -------------------------
    # To escape "Local Minima". Imagine being stuck in a small pothole while trying to
    # reach the bottom of the Grand Canyon. You must climb UP out of the pothole
    # (temporary worsening) to continue going down further.
    #
    # This acceptance probability is derived from statistical mechanics:
    # P = exp(-DeltaE / Temperature)
    #
    # High temperature allows for more exploration (jumping over mountains),
    # while low temperature leads to exploitation (sliding into the nearest valley).
    """

    def __init__(self, steps: int = 500, temperature: float = 0.1) -> None:
        """Initialize the optimizer with simulation hyperparameters.

        Args:
            steps: Number of Monte Carlo steps (attempts to move side chains).
                   More steps increase the likelihood of reaching a global minimum.
            temperature: Proportional to the acceptance probability of worse steps.
                         High T = exploration (escapes peaks). Low T = exploitation (greedy).

        """
        # Store configuration parameters for use in the optimize() loop.
        self.steps = steps
        self.temperature = temperature

    def optimize(self, peptide: struc.AtomArray) -> struc.AtomArray:
        """Run the optimization protocol on the given peptide structure.
        Modifies the structure in-place to the best state found during the search.

        ALGORITHM LOGIC:
        ----------------
        1. Setup: Filter for residues that have known rotamer libraries.
        2. Baseline: Calculate initial clash score and cache coordinates.
        3. Iterate:
           a. Randomly pick an optimizable residue.
           b. Sample a new rotamer from the Dunbrack distribution.
           c. Apply the change using NeRF-based reconstruction.
           d. Apply Metropolis acceptance criterion.
           e. Track the 'Global Best' structure seen so far.
        4. Finalize: Revert to the global best structure before returning.

        Args:
            peptide: The input structure to optimize (AtomArray).

        Returns:
            The optimized structure (reference to input).

        """
        # Log the start of the optimization process.
        logger.info(f"Starting side-chain packing optimization ({self.steps} steps)...")

        # ── LOCAL DETERMINISM ──────────────────────────────────────────────────
        # We use a localized random generator to ensure that results are
        # reproducible across platforms while maintaining thread-safety.
        # This prevents 'stochastic drift' in scientific datasets.
        # We use a fixed seed (42) for pedagogical consistency.
        rng = np.random.default_rng(42)

        # ── INITIALIZATION ─────────────────────────────────────────────────────
        # Identify residues that we are capable of optimizing.
        # Standard residues (ALA, GLY) are skipped as they lack side-chain torsions.
        residue_ids = sorted(set(peptide.res_id))
        optimizable_residues = []

        # We build a residue map to quickly identify residue types by ID.
        # CA (Alpha Carbon) is used as the representative atom for each residue.
        # This approach is highly efficient for large protein structures.
        ca_atoms = peptide[peptide.atom_name == "CA"]
        res_map = {atom.res_id: atom.res_name for atom in ca_atoms}

        # Filter the sequence for 'moving' parts based on our library availability.
        for res_id in residue_ids:
            res_name = res_map.get(res_id)
            # Only optimize residues with valid rotamer libraries.
            if res_name in ROTAMER_LIBRARY and ROTAMER_LIBRARY[res_name]:
                optimizable_residues.append((res_id, res_name))

        # Heuristic check: if there is nothing to pack, we are done.
        # This prevents division-by-zero or empty-loop errors.
        if not optimizable_residues:
            logger.info("No optimizable residues found.")
            return peptide

        # ── BASELINE SCORING ───────────────────────────────────────────────────
        # We measure the starting state to set the initial bar for 'Best'.
        # The clash score is our objective function (potential energy).
        current_score = calculate_clash_score(peptide)
        best_score = current_score

        # ELITISM STRATEGY:
        # We store the absolute best coordinates found. Even if simulated
        # annealing accepts a worse move later, we can always revert to this
        # optimal peak at the end of the simulation.
        # This prevents the "ending on a peak" problem common in short MC runs.
        best_coords = peptide.coord.copy()

        # Log the initial score for debugging and scientific audit trails.
        logger.info(f"Initial Clash Score: {current_score:.4f}")

        # Keep track of how many moves were actually accepted.
        accepted_moves = 0

        # ── MAIN MONTE CARLO LOOP ──────────────────────────────────────────────
        for _step in range(self.steps):
            # STEP A: Selection
            # Randomly select a residue to mutate. Uniform sampling ensures
            # all residues get roughly equal attention over many steps.
            target_res_id, target_res_name = optimizable_residues[
                rng.integers(len(optimizable_residues))
            ]

            # STEP B: Sampling
            # Retrieve available rotamers for this residue type.
            # We use probability-weighted selection based on Dunbrack frequencies.
            # This ensures that we sample chemically plausible conformations first.
            rotamer_options = ROTAMER_LIBRARY[target_res_name]
            weights = [r.get("prob", 1.0) for r in rotamer_options]
            # Normalize the weights into a valid probability distribution.
            weights = np.array(weights) / np.sum(weights)

            # Pick a target conformation using the weighted distribution.
            rotamer_idx = rng.choice(len(rotamer_options), p=weights)
            new_rotamer = rotamer_options[rotamer_idx]

            # STEP C: Backup
            # Identify which atom indices belong to this residue so we can
            # surgically undo the change if the move is rejected.
            # This avoids full-protein copying, making the loop O(N_residue)
            # rather than O(N_protein) in terms of coordinate stashing.
            mask = peptide.res_id == target_res_id
            indices = np.where(mask)[0]
            old_coords = peptide.coord[indices].copy()

            # STEP D: Application
            # Apply the new Chi angles using the NeRF (Internal Coordinate) system.
            # This handles side-chain heavy atoms and hydrogens.
            # We use the specialized reconstruct_sidechain helper from geometry.py.
            try:
                reconstruct_sidechain(peptide, target_res_id, new_rotamer, target_res_name)
            except Exception as e:
                # If geometry construction fails (rare), we skip this step.
                # This could happen if backbone geometry is severely distorted.
                logger.warning(f"Failed to apply rotamer: {e}")
                continue

            # STEP E: Assessment
            # Calculate the new pseudo-energy (Clash Score).
            new_score = calculate_clash_score(peptide)
            delta = new_score - current_score

            # STEP F: Metropolis Acceptance Criterion
            # This logic determines if the system 'moves' to the new state.
            accept = False
            if delta < 0:
                # Always accept improvements (Gradient Descent phase).
                accept = True
            elif accepted_moves == 0 and _step < (self.steps // 10) and new_score > 0:
                # AGGRESSIVE INITIAL EXPLORATION:
                # If we haven't found any valid moves yet, accept anything to
                # jump out of a potentially singular starting state.
                # This ensures the optimizer doesn't get stuck at the very beginning.
                accept = True
            else:
                # Boltzmann-weighted acceptance for worsening moves.
                # Higher temperatures make the system more 'liquid' and explorative.
                if self.temperature > 0:
                    prob = np.exp(-delta / self.temperature)
                    if rng.random() < prob:
                        accept = True

            # STEP G: State Management
            if accept:
                # MOVE ACCEPTED: Update current state and check for global best.
                current_score = new_score
                accepted_moves += 1
                # Check for Global Best (Elitism).
                if new_score < best_score:
                    best_score = new_score
                    best_coords = peptide.coord.copy()
            else:
                # MOVE REJECTED: Restore the previous coordinates for this residue.
                peptide.coord[indices] = old_coords

        # ── FINALIZATION ───────────────────────────────────────────────────────
        # Ensure we return the absolute best configuration found during the
        # entire search, not just the state of the final iteration.
        # This is a critical architectural guarantee for structural quality.
        peptide.coord = best_coords

        # Log final results for scientific audit.
        logger.info(
            f"Optimization complete. Final Best Score: {best_score:.4f} "
            f"(Moves accepted: {accepted_moves})"
        )
        return peptide


def optimize_sidechains(peptide: struc.AtomArray, steps: int = 500) -> struc.AtomArray:
    """Convenience wrapper for SideChainPacker.

    This function provides a stateless entry point for rapid structural refinement.
    It encapsulates the configuration and execution of the SideChainPacker class.

    Args:
        peptide: The protein structure to optimize (Biotite AtomArray).
        steps: Total number of Monte Carlo iterations to perform.

    Returns:
        The refined protein structure with minimized steric clashes.
    """
    # Instantiate the packer with default temperature (0.1).
    packer = SideChainPacker(steps=steps)
    # Execute the optimization and return the results.
    return packer.optimize(peptide)
