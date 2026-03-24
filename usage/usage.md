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

## Examples

### Basic Peptides

```bash
# Alpha helix
synth-pdb --length 20 --conformation alpha --output helix.pdb

# Beta sheet
synth-pdb --length 20 --conformation beta --output sheet.pdb
```

### Complex Topologies

```bash
# Helix-Turn-Helix
synth-pdb --structure "1-10:alpha,11-15:random,16-25:alpha" --minimize
```
