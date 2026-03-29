"""Synthetic Multiple Sequence Alignment (MSA) Generation with Co-Evolutionary Constraints.

This module implements a physical sequence-level evolutionary simulator.
Based on Direct Coupling Analysis (DCA) theory, it models sequence probability using a
Potts Energy Model. A Metropolis-Hastings Markov Chain Monte Carlo (MCMC) algorithm is
then used to simulate evolutionary drift over thousands of generations, ensuring that
produced sequences respect the native 3D fold (Contact Map) via co-evolution constraints.

Literature Reference:
- Morcos, F., et al. (2011). "Direct-coupling analysis of residue coevolution
  captures native contacts across many protein families." PNAS, 108(49), E1293-E1301.
"""

import random

import numpy as np

# Standard 20 biological amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Rough volume/steric penalty groupings for simplified physics
# 0: Tiny (G, A, S)
# 1: Small (C, D, P, N, T)
# 2: Medium (V, E, Q, I, L)
# 3: Large (K, M, H, F)
# 4: Massive (Y, R, W)
RESIDUE_VOLUMES = {
    "G": 0,
    "A": 0,
    "S": 0,
    "C": 1,
    "D": 1,
    "P": 1,
    "N": 1,
    "T": 1,
    "V": 2,
    "E": 2,
    "Q": 2,
    "I": 2,
    "L": 2,
    "K": 3,
    "M": 3,
    "H": 3,
    "F": 3,
    "Y": 4,
    "R": 4,
    "W": 4,
}


class CoevolutionModel:
    """Defines the statistical Potts Model energy landscape for a given Protein Fold.

    The energy function is defined as:
        E(S) = sum_i(h_i(S_i)) + sum_{i<j}(J_ij(S_i, S_j))
    Where:
        - h_i is the local site preference (Fields)
        - J_ij is the pairwise coupling constraint between contacting residues.
    """

    def __init__(self, base_sequence: str, contact_map: np.ndarray):
        """Initialize the Coevolution Model.

        Args:
            base_sequence (str): The native, starting sequence.
            contact_map (np.ndarray): N x N boolean matrix where True indicates physical contact.

        """
        self.length = len(base_sequence)
        self.contact_map = contact_map
        self.vocab_size = len(AMINO_ACIDS)
        self.aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

        # Initialize empty Fields and Couplings
        self.fields = np.zeros((self.length, self.vocab_size))
        self.couplings = np.zeros((self.length, self.length, self.vocab_size, self.vocab_size))

        self._build_physical_couplings(base_sequence)

    def _build_physical_couplings(self, base_sequence: str) -> None:
        """Constructs the J_ij coupling matrix based on simplified sterics.
        If residues closely pack in 3D space, massive-massive combinations are penalized (steric clash).
        To lower the energy back down, one residue must mutate to a Tiny volume (compensatory mutation).
        """
        for i in range(self.length):
            for j in range(i + 1, self.length):
                if self.contact_map[i, j]:
                    # They are in 3D contact! They must co-evolve.
                    for aa1_idx, aa1 in enumerate(AMINO_ACIDS):
                        vol1 = RESIDUE_VOLUMES[aa1]
                        for aa2_idx, aa2 in enumerate(AMINO_ACIDS):
                            vol2 = RESIDUE_VOLUMES[aa2]

                            # Simple Steric Potential:
                            # A "perfect" packing volume is ~4.
                            # If vol1 + vol2 > 5, it's a massive clash (High Energy).
                            # If vol1 + vol2 < 2, it's a structural void (High Energy).
                            total_vol = vol1 + vol2

                            if total_vol > 5:
                                penalty = (total_vol - 5) * 10.0  # Extreme steric clash penalty
                            elif total_vol < 2:
                                penalty = (2 - total_vol) * 5.0  # Moderate void penalty
                            else:
                                penalty = 0.0  # Perfect packing

                            # Symmetric coupling
                            self.couplings[i, j, aa1_idx, aa2_idx] = penalty
                            self.couplings[j, i, aa2_idx, aa1_idx] = penalty

    def calculate_energy(self, sequence: str) -> float:
        """Calculate the pseudo-energy of a given sequence on this fold's energetic landscape.
        Lower energy indicates a more stable, evolutionarily favored protein.
        """
        energy = 0.0

        # Add local fields (h_i) - currently zeroed out for pure co-evolution focus
        for i, aa in enumerate(sequence):
            aa_idx = self.aa_to_idx[aa]
            energy += self.fields[i, aa_idx]

        # Add pairwise couplings (J_ij)
        for i in range(self.length):
            aa1_idx = self.aa_to_idx[sequence[i]]
            for j in range(i + 1, self.length):
                if self.contact_map[i, j]:
                    aa2_idx = self.aa_to_idx[sequence[j]]
                    energy += self.couplings[i, j, aa1_idx, aa2_idx]

        return energy


class MetropolisHastingsSampler:
    """Simulates the evolutionary drift of a sequence using Markov Chain Monte Carlo.
    A mutation is proposed:
        - If Energy decreases: It is always accepted (favorable).
        - If Energy increases: It is accepted with probability P = exp(-DeltaE / T).
    """

    def __init__(self, model: CoevolutionModel, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature
        # State
        self.current_sequence = ""
        self.current_energy = 0.0

    def start(self, base_sequence: str) -> None:
        """Initialize the MCMC chain."""
        self.current_sequence = base_sequence
        self.current_energy = self.model.calculate_energy(base_sequence)

    def step(self) -> bool:
        """Propose exactly ONE amino acid point mutation and probabilistically accept or reject it.
        Returns True if the mutation was accepted.
        """
        if not self.current_sequence:
            return False

        # 1. Propose mutation (choose random site, choose random new amino acid)
        site = random.randint(0, self.model.length - 1)
        current_aa = self.current_sequence[site]
        new_aa = random.choice(AMINO_ACIDS)

        if current_aa == new_aa:
            return False

        proposed_seq_list = list(self.current_sequence)
        proposed_seq_list[site] = new_aa
        proposed_seq = "".join(proposed_seq_list)

        # 2. Evaluate energetic change
        proposed_energy = self.model.calculate_energy(proposed_seq)
        delta_e = proposed_energy - self.current_energy

        # 3. Metropolis Criterion
        accept = False
        if delta_e <= 0:
            accept = True  # Favorable drift
        else:
            # Unfavorable drift (thermal noise allows escaping local minima)
            # Guard against ZeroDivisionError for T=0
            if self.temperature <= 0.0001:
                prob = 0.0
            else:
                prob = np.exp(-delta_e / self.temperature)

            if random.random() < prob:
                accept = True

        if accept:
            self.current_sequence = proposed_seq
            self.current_energy = proposed_energy
            return True

        return False


def generate_msa(
    base_sequence: str,
    contact_map: np.ndarray,
    num_sequences: int = 100,
    temperature: float = 1.0,
    steps_between_samples: int = 20,
) -> list[str]:
    """Generates a Synthetic Multiple Sequence Alignment (MSA) imbued with Co-Evolutionary constraints.

    Args:
        base_sequence: Starting 'wild type' amino acid sequence string.
        contact_map: N x N valid binary contact map bounding the 3D structure.
        num_sequences: Number of homologous sequences to scrape from the simulation.
        temperature: "Thermal Noise" of evolution. Higher = more divergence, lower contact fidelity.
        steps_between_samples: Number of MCMC mutation proposals to try between saving a sequence.

    Returns:
        List of strings representing the aligned MSA.

    """
    # 1. Initialize Physics Engine
    model = CoevolutionModel(base_sequence, contact_map)
    sampler = MetropolisHastingsSampler(model, temperature=temperature)
    sampler.start(base_sequence)

    msa = []

    # 2. Burn-in Phase
    # Run the chain for a while before sampling to lose the "memory" of the starting sequence
    burn_in_steps = len(base_sequence) * 10
    for _ in range(burn_in_steps):
        sampler.step()

    # 3. Generation Phase
    for _ in range(num_sequences):
        for _ in range(steps_between_samples):
            sampler.step()
        msa.append(sampler.current_sequence)

    return msa
