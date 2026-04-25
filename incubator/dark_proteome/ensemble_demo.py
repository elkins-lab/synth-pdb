import numpy as np

from synth_pdb.batch_generator import BatchedGenerator
from synth_pdb.geometry.rmsd import calculate_rmsd

sequence = "MDVFMKGLSKAKEGVVAA"
n_states = 5

print(f"Generating an ensemble of {n_states} states for Alpha-synuclein...")
generator = BatchedGenerator(sequence, n_batch=n_states)
# Higher drift (50.0) to simulate disordered diversity
batch = generator.generate_batch(drift=50.0)

# Calculate pairwise RMSD to show diversity
rmsds = []
for i in range(n_states):
    for j in range(i + 1, n_states):
        rmsd = calculate_rmsd(batch.coords[i], batch.coords[j])
        rmsds.append(rmsd)

print("\nDiversity Metrics:")
print(f"Mean Pairwise RMSD: {np.mean(rmsds):.2f} Angstroms")
print(f"Max Pairwise RMSD:  {np.max(rmsds):.2f} Angstroms")
print("\nConclusion: The 'conformation' is not a single point, but a high-variance cloud.")
