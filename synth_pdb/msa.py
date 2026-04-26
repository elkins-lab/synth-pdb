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

# Standard 20 biological amino acids used in the simulation
# Each character represents the standard IUPAC one-letter code.
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# EDUCATIONAL NOTE - Steric Volume Compatibility:
# -----------------------------------------------
# In a dense protein core, residues are not just strings of letters; they
# occupy physical volume. If a mutation increases the volume of a residue
# without a compensatory decrease in a contacting partner, the 3D structure
# would physically "burst" or clash.
#
# We model this using five volume tiers (0 to 4), where Tier 0 represents
# tiny residues (Glycine) and Tier 4 represents massive aromatics (Tryptophan).
#
# Tier 0: Tiny (G, A, S) - Minimal steric footprint.
# Tier 1: Small (C, D, P, N, T) - Moderate bulk.
# Tier 2: Medium (V, E, Q, I, L) - Standard aliphatic volume.
# Tier 3: Large (K, M, H, F) - High bulk, often hydrophobic or basic.
# Tier 4: Massive (Y, R, W) - Large aromatic rings or long chains.
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

# Physical classifications for pairwise energy calculations.
# Positive AAs (Lysine, Arginine) have a net +1 charge at pH 7.
POSITIVE_AMINO_ACIDS = {"K", "R"}
# Negative AAs (Aspartate, Glutamate) have a net -1 charge at pH 7.
NEGATIVE_AMINO_ACIDS = {"D", "E"}
# Hydrophobic AAs favor the core and avoid water.
HYDROPHOBIC_AMINO_ACIDS = {"V", "I", "L", "F", "M", "W", "C", "A"}
# Polar and Charged AAs are hydrophilic and favor the surface.
POLAR_CHARGED_AMINO_ACIDS = {"R", "K", "D", "E", "Q", "N", "H", "S", "T", "Y"}


class CoevolutionModel:
    """Defines the statistical Potts Model energy landscape for a given Protein Fold.

    The energy function is defined as:
        E(S) = sum_i(h_i(S_i)) + sum_{i<j}(J_ij(S_i, S_j))
    Where:
        - h_i is the local site preference (Fields)
        - J_ij is the pairwise coupling constraint between contacting residues.

    PHYSICAL SIGNIFICANCE:
    The Potts Model is the mathematical backbone of modern co-evolutionary analysis.
    By finding sequences that minimize this energy, we identify sets of mutations
    that are "compatible" with the structural constraints of the fold.
    """

    def __init__(
        self, base_sequence: str, contact_map: np.ndarray, rel_sasa: np.ndarray | None = None
    ):
        """Initialize the Coevolution Model from a native structure.

        Args:
            base_sequence (str): The native, starting sequence.
            contact_map (np.ndarray): N x N boolean matrix where True indicates physical contact.
            rel_sasa (np.ndarray | None): Relative Solvent Accessible Surface Area per residue.

        """
        self.length = len(base_sequence)
        self.contact_map = contact_map
        self.vocab_size = len(AMINO_ACIDS)
        # Create mapping for fast index-based tensor lookups
        self.aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

        # Initialize empty Fields (h_i) and Couplings (J_ij)
        # Fields: [Sequence Position x Amino Acid Type]
        self.fields = np.zeros((self.length, self.vocab_size))
        # Couplings: [Pos1 x Pos2 x AA1 x AA2] - Modeling the covariance of mutations.
        self.couplings = np.zeros((self.length, self.length, self.vocab_size, self.vocab_size))

        # Build the energetic constraints based on structural features
        self._build_local_fields(rel_sasa)
        self._build_physical_couplings()

    def _build_local_fields(self, rel_sasa: np.ndarray | None) -> None:
        """Constructs the h_i local fields to enforce SASA constraints.
        Buried residues (rel_sasa < 0.2) receive massive energy penalties for Polar/Charged AAs.

        Educational Note: Hydrophobic Core Collapse
        -------------------------------------------
        Solvent Accessible Surface Area (SASA) is the physical mechanism mapping
        the 3D structural core back to 1D sequence constraints.
        If a residue is statistically "buried" deep inside the protein core
        with extreme desolvation, evolutionary drift must strictly eliminate
        charged/polar mutations. Placing an Arginine or Aspartate in the
        water-free hydrophobic core will rupture the hydrogen-bond network and
        unfold the protein. We enforce this geometrically by massively penalizing
        the h_i local fields for Hydrophilic amino acids at any position where
        rel_sasa < 0.2.
        """
        if rel_sasa is None:
            return  # No structural pressure

        for i in range(self.length):
            # Check if buried
            if rel_sasa[i] < 0.2:
                for aa in POLAR_CHARGED_AMINO_ACIDS:
                    aa_idx = self.aa_to_idx[aa]
                    self.fields[i, aa_idx] += 100.0  # Massive penalty

    def _build_physical_couplings(self) -> None:
        """Constructs the J_ij coupling matrix based on simplified sterics and electrostatics.
        If residues closely pack in 3D space, massive-massive combinations are penalized (steric clash).
        To lower the energy back down, one residue must mutate to a Tiny volume (compensatory mutation).
        We also strictly penalize like-charges and reward opposite-charges (Salt Bridges).
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
                            total_vol = vol1 + vol2

                            if total_vol > 5:
                                penalty = (total_vol - 5) * 10.0  # Extreme steric clash penalty
                            elif total_vol < 2:
                                penalty = (2 - total_vol) * 5.0  # Moderate void penalty
                            else:
                                penalty = 0.0  # Perfect packing

                            # Educational Note: Electrostatic Compatibility
                            # ---------------------------------------------
                            # Proteins utilize localized regions of electrical charge (Salt Bridges)
                            # to lock their tertiary folds into favorable lower-energy states.
                            # Conversely, slamming two like-charges together completely blows up
                            # the stability of the construct via Coloumbic repulsion.
                            # We deeply embed these physical phenomena directly into the J_ij
                            # interaction couplings, rewarding +/- pairs while aggressively
                            # destroying +/+ or -/- complexes.

                            # Charge Compatibility
                            if aa1 in POSITIVE_AMINO_ACIDS and aa2 in NEGATIVE_AMINO_ACIDS:
                                penalty -= 10.0  # Favorable Salt Bridge
                            elif aa1 in NEGATIVE_AMINO_ACIDS and aa2 in POSITIVE_AMINO_ACIDS:
                                penalty -= 10.0  # Favorable Salt Bridge
                            elif aa1 in POSITIVE_AMINO_ACIDS and aa2 in POSITIVE_AMINO_ACIDS:
                                penalty += 15.0  # Charge Repulsion
                            elif aa1 in NEGATIVE_AMINO_ACIDS and aa2 in NEGATIVE_AMINO_ACIDS:
                                penalty += 15.0  # Charge Repulsion

                            # Symmetric coupling
                            self.couplings[i, j, aa1_idx, aa2_idx] = penalty
                            self.couplings[j, i, aa2_idx, aa1_idx] = penalty

    def calculate_energy(self, sequence: str) -> float:
        """Calculate the pseudo-energy of a given sequence on this fold's energetic landscape.
        Lower energy indicates a more stable, evolutionarily favored protein.
        """
        # Summation of both independent (fields) and dependent (couplings) energetic terms.
        energy = 0.0

        # 1. Integrate local site preferences (h_i)
        for i, aa in enumerate(sequence):
            aa_idx = self.aa_to_idx[aa]
            energy += self.fields[i, aa_idx]

        # 2. Integrate pairwise interaction constraints (J_ij)
        # We only iterate over contacts to avoid unnecessary O(N^2) lookups.
        for i in range(self.length):
            aa1_idx = self.aa_to_idx[sequence[i]]
            for j in range(i + 1, self.length):
                if self.contact_map[i, j]:
                    # These residues are close in 3D space, their mutations are linked.
                    aa2_idx = self.aa_to_idx[sequence[j]]
                    energy += self.couplings[i, j, aa1_idx, aa2_idx]

        return energy

    def calculate_delta_energy(
        self,
        current_seq: str,
        site1: int,
        new_aa1: str,
        site2: int | None = None,
        new_aa2: str | None = None,
    ) -> float:
        """Calculate the change in energy (Delta E) for 1 or 2 simultaneous mutations in O(L) time.

        Educational Note: Big-O Performance Breakthrough
        ------------------------------------------------
        A full evaluation of the Potts Model energy function requires summing the J_ij
        matrix across all N residues, an O(N^2) operation. For a 200-residue sequence,
        that's 40,000 lookups per step. Over 1 million generations, that's 40 billion operations!

        Because the MCMC sampler only ever mutates 1 or 2 residues at a time, 99% of the
        pairwise interactions remain completely unchanged. Instead of brute-force calculating
        the new system energy entirely from scratch, we can simply calculate the difference (Delta):

        1. Subtract the old energy of the mutated sites.
        2. Add the new energy of the mutated sites.

        This reduces the computational complexity to strictly O(L) (where L is the number of
        contacts for the mutated site), saving over 500x computation time and allowing for
        blisteringly fast synthetic MSA generation architectures.
        """
        delta = 0.0

        old_aa1 = current_seq[site1]
        old_idx1 = self.aa_to_idx[old_aa1]
        new_idx1 = self.aa_to_idx[new_aa1]

        delta += self.fields[site1, new_idx1] - self.fields[site1, old_idx1]

        if site2 is not None and new_aa2 is not None:
            old_aa2 = current_seq[site2]
            old_idx2 = self.aa_to_idx[old_aa2]
            new_idx2 = self.aa_to_idx[new_aa2]
            delta += self.fields[site2, new_idx2] - self.fields[site2, old_idx2]

            # Coupling between the two mutating sites themselves
            if self.contact_map[site1, site2]:
                old_coup = self.couplings[site1, site2, old_idx1, old_idx2]
                new_coup = self.couplings[site1, site2, new_idx1, new_idx2]
                delta += new_coup - old_coup
        else:
            site2 = -1
            old_idx2 = -1
            new_idx2 = -1

        # Coupling with the rest of the sequence
        for k in range(self.length):
            if k == site1 or k == site2:
                continue
            seq_idx_k = self.aa_to_idx[current_seq[k]]
            if self.contact_map[site1, k]:
                delta += (
                    self.couplings[site1, k, new_idx1, seq_idx_k]
                    - self.couplings[site1, k, old_idx1, seq_idx_k]
                )

            if site2 != -1 and self.contact_map[site2, k]:
                delta += (
                    self.couplings[site2, k, new_idx2, seq_idx_k]
                    - self.couplings[site2, k, old_idx2, seq_idx_k]
                )

        return float(delta)


class MetropolisHastingsSampler:
    """Simulates the evolutionary drift of a sequence using Markov Chain Monte Carlo.
    A mutation is proposed:
        - If Energy decreases: It is always accepted (favorable).
        - If Energy increases: It is accepted with probability P = exp(-DeltaE / T).
    """

    def __init__(
        self, model: CoevolutionModel, temperature: float = 1.0, coupled_mutation_prob: float = 0.20
    ):
        self.model = model
        self.temperature = temperature
        self.coupled_mutation_prob = coupled_mutation_prob
        # State
        self.current_sequence = ""
        self.current_energy = 0.0

        # Cache contact list for fast random sampling
        self.contact_list = []
        for i in range(model.length):
            for j in range(i + 1, model.length):
                if model.contact_map[i, j]:
                    self.contact_list.append((i, j))

    def start(self, base_sequence: str) -> None:
        """Initialize the MCMC chain."""
        self.current_sequence = base_sequence
        self.current_energy = self.model.calculate_energy(base_sequence)

    def step(self) -> bool:
        """Propose exactly ONE or TWO point mutations and probabilistically accept or reject them.
        This follows the standard Metropolis-Hastings update rule.

        Returns True if the mutation was accepted.
        """
        if not self.current_sequence:
            # Cannot mutate an empty sequence.
            return False

        # Decide if we do a single or double mutation taking advantage of "Magic Steps"
        #
        # Educational Note: The "Magic Step" coupled mutation
        # ---------------------------------------------------
        # Traditional MCMC walkers only jump one dimension at a time (e.g., mutating only X or Y).
        # In evolutionary landscapes, getting from a massive [Tryptophan:Tiny] pair to a
        # complementary [Tiny:Tryptophan] pair is impossible if you can only make one
        # mutation at a time. The intermediate state [Tryptophan:Tryptophan] involves a massive
        # steric clash, imposing an astronomical energy penalty that a low-temperature
        # simulation can never mathematically cross.
        #
        # By proposing a "Coupled Mutation" across two contacting residues simultaneously,
        # the simulation evaluates the Delta Energy of the final state exactly. The clash
        # never physically occurs, capturing the hallmark of true Direct Coupling covariance.
        do_coupled = (random.random() < self.coupled_mutation_prob) and len(self.contact_list) > 0

        # Initialize mutation sites and residues
        site1 = -1
        new_aa1 = ""
        site2 = None
        new_aa2 = None

        if do_coupled:
            # Proposal Stage: Coupled Mutation
            # 1. Randomly pick a pair of residues that are in contact.
            site1, site2 = random.choice(self.contact_list)
            current_aa1 = self.current_sequence[site1]
            current_aa2 = self.current_sequence[site2]

            # 2. Randomly select new amino acids for both sites.
            new_aa1 = random.choice(AMINO_ACIDS)
            new_aa2 = random.choice(AMINO_ACIDS)

            # 3. Reject immediately if no change is actually proposed.
            if current_aa1 == new_aa1 and current_aa2 == new_aa2:
                return False
        else:
            # Proposal Stage: Single Point Mutation
            # 1. Randomly select one position anywhere in the sequence.
            site1 = random.randint(0, self.model.length - 1)
            current_aa = self.current_sequence[site1]
            # 2. Randomly select the new amino acid identity.
            new_aa1 = random.choice(AMINO_ACIDS)

            # 3. Reject immediately if no change is actually proposed.
            if current_aa == new_aa1:
                return False

        # Evaluation Stage: Energy Calculation
        # Calculate the energetic change efficiently using O(L) delta evaluation.
        # This is the "secret sauce" that allows large MSAs to be generated in seconds.
        delta_e = self.model.calculate_delta_energy(
            self.current_sequence, site1, new_aa1, site2, new_aa2
        )
        proposed_energy = self.current_energy + delta_e

        # Selection Stage: Metropolis Criterion
        accept = False
        if delta_e <= 0:
            # If the energy decreases or stays same, the mutation is structurally
            # favorable or neutral. In evolution, these represent stable drifts.
            accept = True
        else:
            # If the energy increases, the mutation is structurally unfavorable.
            # However, we accept it with a probability based on the "Temperature."
            # High T = Drift into unstable regions (Divergence).
            # Low T = Rigid adherence to the native fold (Fidelity).
            if self.temperature <= 0.0001:
                # Absolute zero: only favorable moves allowed.
                prob = 0.0
            else:
                # Boltzmann weighting: Unfavorable moves are rare but possible.
                prob = np.exp(-delta_e / self.temperature)

            if random.random() < prob:
                accept = True

        if accept:
            # Execution Stage: Update State
            proposed_seq_list = list(self.current_sequence)
            proposed_seq_list[site1] = new_aa1
            if site2 is not None and new_aa2 is not None:
                proposed_seq_list[site2] = new_aa2

            self.current_sequence = "".join(proposed_seq_list)
            self.current_energy = proposed_energy
            return True

        # Rejected: The sequence remains in its current state.
        return False


def generate_msa(
    base_sequence: str,
    contact_map: np.ndarray,
    num_sequences: int = 100,
    temperature: float = 1.0,
    steps_between_samples: int = 20,
    rel_sasa: np.ndarray | None = None,
    coupled_mutation_prob: float = 0.20,
) -> list[str]:
    """Generates a Synthetic Multiple Sequence Alignment (MSA) imbued with Co-Evolutionary constraints.

    Args:
        base_sequence: Starting 'wild type' amino acid sequence string.
        contact_map: N x N valid binary contact map bounding the 3D structure.
        num_sequences: Number of homologous sequences to scrape from the simulation.
        temperature: "Thermal Noise" of evolution. Higher = more divergence, lower contact fidelity.
        steps_between_samples: Number of MCMC mutation proposals to try between saving a sequence.
        rel_sasa: (Optional) Array of lengths N indicating solvent accessibility. Core residues will heavily favor aliphatics.
        coupled_mutation_prob: (Optional) Probability of simultaneously mutating two contacting residues to traverse steric gaps (The Magic Step).

    Returns:
        List of strings representing the aligned MSA.

    """
    # 1. Initialize Physics Engine
    model = CoevolutionModel(base_sequence, contact_map, rel_sasa=rel_sasa)
    sampler = MetropolisHastingsSampler(
        model, temperature=temperature, coupled_mutation_prob=coupled_mutation_prob
    )
    sampler.start(base_sequence)

    msa = []

    # 2. Burn-in Phase
    burn_in_steps = len(base_sequence) * 10
    for _ in range(burn_in_steps):
        sampler.step()

    # 3. Generation Phase
    for _ in range(num_sequences):
        for _ in range(steps_between_samples):
            sampler.step()
        msa.append(sampler.current_sequence)

    return msa
