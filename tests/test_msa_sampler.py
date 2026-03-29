import numpy as np

from synth_pdb.msa import CoevolutionModel, MetropolisHastingsSampler


def test_sampler_initialization():
    seq = "AAAAA"
    cmap = np.zeros((5, 5), dtype=bool)
    model = CoevolutionModel(seq, cmap)

    # Initialize Sampler with high temperature (pure random drift)
    sampler = MetropolisHastingsSampler(model, temperature=5.0)
    sampler.start(seq)
    assert sampler.current_sequence == "AAAAA"
    assert sampler.current_energy == model.calculate_energy("AAAAA")


def test_sampler_acceptance_high_temp():
    """At extremely high temperatures, almost all mutations should be accepted,
    even energetically unfavorable ones (modeling random genetic drift).
    """
    seq = "AAAAA"
    cmap = np.zeros((5, 5), dtype=bool)
    model = CoevolutionModel(seq, cmap)

    sampler = MetropolisHastingsSampler(model, temperature=100.0)
    sampler.start(seq)

    accepted = 0
    trials = 100
    for _ in range(trials):
        # Force the sampler to evaluate a random mutation
        if sampler.step():
            accepted += 1

    # Acceptance rate should be very high due to T=100.0
    assert accepted > 80, f"Expected >80% acceptance at high T, got {accepted}%"

    # Sequence should have drifted far from starting state
    assert sampler.current_sequence != "AAAAA"


def test_sampler_rejection_low_temp():
    """At extremely low temperatures (near absolute zero), the sampler should
    rapidly minimize energy and reject ANY mutation that increases it (greedy algorithm).
    """
    seq = "WGTTW"  # Two massive Tryptophans

    # Force them to be in severe steric contact
    cmap = np.zeros((5, 5), dtype=bool)
    cmap[0, 4] = True
    cmap[4, 0] = True

    model = CoevolutionModel(seq, cmap)
    sampler = MetropolisHastingsSampler(model, temperature=0.01)
    sampler.start(seq)

    initial_energy = sampler.current_energy

    # Run 100 steps
    for _ in range(100):
        sampler.step()

    final_energy = sampler.current_energy

    # Energy must monotonicially decrease or stay identical at freezing temperatures
    assert (
        final_energy <= initial_energy
    ), "Low Temperature MCMC should never accept higher energy states."

    # The two bulky Tryptophans in contact should have been mutated to smaller residues
    # to relieve the heavy steric penalty defined in the coupling matrix.
    assert sampler.current_sequence[0] != "W" or sampler.current_sequence[4] != "W"
