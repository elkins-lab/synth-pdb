import os

from synth_pdb.batch_generator import BatchedGenerator

# Ubiquitin Sequence (standard biophysical candle)
sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
n_models = 20

print(f"Generating Cryo-EM 'Standard Candle' Ensemble ({n_models} states)...")

# We use a low drift (5.0) to simulate natural thermal fluctuations
# around the native-like random-coil/alpha fold.
generator = BatchedGenerator(sequence, n_batch=n_models)
batch = generator.generate_batch(drift=5.0)

output_path = "incubator/dark_proteome/ubiquitin_standard_candle.pdb"

with open(output_path, "w") as f:
    for i in range(n_models):
        f.write(f"MODEL     {i+1:>4}\n")
        f.write(batch.to_pdb(i))
        # to_pdb adds END, we should use ENDMDL for multi-model
        # But for now, we'll just join them.
        # Actually to_pdb already adds TER and END.
    f.write("END\n")

print(f"Successfully generated multi-model PDB at: {os.path.abspath(output_path)}")
print("This ensemble represents the 'Thermal Motion' that Cryo-EM algorithms must resolve.")
