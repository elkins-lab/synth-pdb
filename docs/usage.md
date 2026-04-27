# Usage

## Command-Line Arguments

### Structure Definition

- `--length <LENGTH>`: Number of residues (Default: 10)
- `--sequence <SEQUENCE>`: Specify sequence (e.g. "ACDEF")
- `--conformation <TYPE>`: `alpha`, `beta`, `ppii`, `extended`, `random`
- `--structure <REGIONS>`: Define mixed structure (e.g. `1-10:alpha,11-20:beta`)

### Physics & Refinement

- `--minimize`: Run OpenMM energy minimization (Implicit Solvent)
- `--optimize`: Run Monte Carlo side-chain packing
- `--refine-clashes <N>`: Iteratively adjust atoms to reduce clashes

### NMR Data Generation

- `--gen-nef`: Generate NOE restraints (NEF format)
- `--gen-shifts`: Predict chemical shifts (SPARTA-lite)
- `--gen-relax`: Generate relaxation data ($R_1, R_2, NOE$)

### Multi-Modal Simulation

- `--mode <MODE>`: `generate` (default), `cryo-em`, `saxs`, `decoys`, `dataset`, `ai`
- `--n-decoys <N>`: Number of models for ensembles/simulations (Default: 10)
- `--drift <DEG>`: Conformational drift for ensembles (Default: 3.0°)
- `--resolution <Å>`: Target resolution for Cryo-EM maps (Default: 3.0Å)
- `--mrc-output <FILE>`: Filename for simulated Cryo-EM map
- `--q-max <q>`: Max scattering vector for SAXS profiles (Default: 0.5 Å⁻¹)
- `--saxs-output <FILE>`: Filename for simulated SAXS profile (.dat)

## Examples

### Basic Peptides

```bash
# Alpha helix
synth-pdb --length 20 --conformation alpha --output helix.pdb

# Beta sheet
synth-pdb --length 20 --conformation beta --output sheet.pdb
```

### Multi-Modal Ensembles

```bash
# Generate Cryo-EM density map (3.5A res) for a 50-model ensemble
synth-pdb --sequence "MQIFVKTLTGK" --mode cryo-em --n-decoys 50 --resolution 3.5 --mrc-output ubiquitin.mrc

# Generate SAXS profile for a flexible ensemble (high drift)
synth-pdb --length 30 --mode saxs --n-decoys 100 --drift 8.0 --saxs-output ensemble.dat

# Generate and visualize SAXS plots (Kratky/Guinier)
synth-pdb --sequence "LKELEKELE" --mode saxs --visualize --plot-type all
```

### Complex Assemblies (Multichain)

```bash
# Generate a heterodimer with Chain A (Alpha) and Chain B (Beta)
synth-pdb --sequence "ALA-GLY-SER:VAL-THR-LEU" --structure "1-3:alpha,4-6:beta" --minimize
```

## High-Throughput Dataset Factory

For large-scale AI training, use the dedicated builder script:

```bash
# Build a synchronized dataset of 1,000 multi-modal samples
python3 scripts/build_multimodal_dataset.py --n 1000 --output-dir my_dataset
```
