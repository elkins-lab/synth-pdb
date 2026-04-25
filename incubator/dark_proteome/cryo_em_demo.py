import os

from synth_pdb.batch_generator import BatchedGenerator
from synth_pdb.cryo_em import generate_density_map, save_mrc_file

# Ubiquitin Sequence (standard biophysical candle)
sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
n_models = 50

print(f"Generating Advanced Cryo-EM 'Standard Candle' Ensemble ({n_models} states)...")

# 1. Generate Ensemble with conformational noise
# We use a drift of 8.0 degrees to simulate significant thermal fluctuations
generator = BatchedGenerator(sequence, n_batch=n_models, full_atom=True)
batch = generator.generate_batch(drift=8.0, seed=42)

# 2. Convert to density map (MRC format)
# Resolution set to 4.0A to simulate a typical medium-resolution map
resolution = 4.0
print(f"Simulating 3D Density Map at {resolution}Å resolution...")

stack = batch.to_stack()
density, origin = generate_density_map(stack, resolution=resolution, grid_spacing=1.0)

output_path = "incubator/dark_proteome/ubiquitin_thermal_ensemble.mrc"
save_mrc_file(output_path, density, origin, spacing=1.0)

print(f"Successfully generated Cryo-EM density map at: {os.path.abspath(output_path)}")
print("This map represents the averaged electron density of a dynamic protein ensemble.")
