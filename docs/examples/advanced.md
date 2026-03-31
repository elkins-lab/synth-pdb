# Advanced Features

This document explores the advanced capabilities of `synth-pdb` for machine learning and complex structural generation.

## 1. Multiple Sequence Alignments (MSA)

For training co-evolutionary models, you can generate synthetic MSAs that maintain a consistent structure:

```bash
python -m synth_pdb.main \
  --sequence "MEELQK" \
  --gen-msa \
  --msa-depth 200 \
  --evolution-temp 1.8
```

The `--gen-msa` flag uses a Markov Chain Monte Carlo (MCMC) algorithm to sample sequence variations that are biophysically compatible with the starting structure, simulating the constraints of natural selection.

## 2. Hard Decoy Generation

Machine learning models for structure prediction benefit from training on "hard decoys"—structures that are close to the correct fold but contain subtle errors.

```bash
python -m synth_pdb.main \
  --mode decoys \
  --n-decoys 10 \
  --rmsd-range 2.0-5.0 \
  --hard
```

The `--hard` flag enables threading, sequence shuffling, and torsion drift to create realistic decoy ensembles that challenge structure-quality classifiers.

## 3. Dataset Factory

You can generate bulk datasets for training neural networks in one command:

```bash
python -m synth_pdb.main \
  --mode dataset \
  --num-samples 1000 \
  --min-length 10 \
  --max-length 100 \
  --dataset-format npz \
  --train-ratio 0.8
```

This will generate 1000 structures, split them into training (80%) and validation (20%) sets, and save them as compressed NumPy arrays (`.npz`) for efficient loading in PyTorch or TensorFlow.

## 4. AI-Driven Interpolation

`synth-pdb` supports interpolating between two structures, which is useful for visualizing conformational transitions or latent space traversals:

```bash
python -m synth_pdb.main \
  --mode ai \
  --ai-op interpolate \
  --start-pdb helix.pdb \
  --end-pdb sheet.pdb \
  --steps 10
```

This generates a series of intermediate PDB files that represent a linear (or smoothed) transition between the two endpoint structures.

## 5. Side-Chain Optimization

For high-resolution structures, you can perform Monte Carlo side-chain optimization:

```bash
python -m synth_pdb.main \
  --sequence "ALA-GLY-TRP-PHE-SER" \
  --optimize
```

This algorithm searches through rotamer libraries for each residue to find the combination that minimizes steric clashes, resulting in a more energetically favorable side-chain packing.
