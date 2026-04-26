# Command-Line Reference

This guide provides a comprehensive reference for the `synth-pdb` command-line interface.

## Basic Usage

The simplest way to use `synth-pdb` is to specify a sequence length:

```bash
python -m synth_pdb.main --length 20 --output my_protein.pdb
```

Or provide a specific amino acid sequence:

```bash
python -m synth_pdb.main --sequence "ALA-GLY-SER-THR-VAL" --output test.pdb
```

## Core Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `--length` | Length of the amino acid sequence (number of residues). | 10 |
| `--sequence` | Specify an amino acid sequence (e.g., 'AGV' or 'ALA-GLY-VAL'). | (Random) |
| `--output` | Output PDB filename. | (Generated) |
| `--conformation` | Secondary structure conformation: `alpha`, `beta`, `ppii`, `extended`, `random`. | `alpha` |
| `--structure` | Per-region conformation specification (e.g., '1-10:alpha,11-14:typeII,15-20:beta'). | - |
| `--seed` | Random seed for reproducible generation. | - |

## Validation & Refinement

| Option | Description |
| :--- | :--- |
| `--validate` | Run validation checks (bond lengths, angles, Ramachandran). |
| `--guarantee-valid` | Repeatedly generate until a valid structure is produced. |
| `--max-attempts` | Maximum number of attempts for `--guarantee-valid`. |
| `--best-of-N` | Generate N structures and select the one with the fewest violations. |
| `--refine-clashes` | Number of iterations to minimally adjust clashing atoms. |
| `--optimize` | Run Monte Carlo side-chain optimization. |
| `--minimize` | Run physics-based energy minimization using OpenMM. |

## Scientific Features

### NMR Observables
| Option | Description |
| :--- | :--- |
| `--gen-shifts` | Generate synthetic Chemical Shift data (H, N, CA, CB, C). |
| `--shift-predictor` | Predictor to use: `shiftx2` (default) or `empirical`. |
| `--gen-relax` | Generate synthetic NMR relaxation data (R1, R2, NOE). |
| `--output-rdcs` | Generate backbone N-H Residual Dipolar Coupling (RDC) data. |
| `--gen-couplings` | Generate synthetic 3J(HN-HA) scalar couplings. |
| `--gen-nef` | Generate synthetic NMR data (NOE restraints) in NEF format. |

### MSA & Evolution
| Option | Description |
| :--- | :--- |
| `--gen-msa` | Generate synthetic Multiple Sequence Alignment (MSA) via simulated evolution. |
| `--msa-depth` | Number of sequences to generate for MSA (default: 100). |
| `--evolution-temp` | Thermal Noise of MSA MCMC evolution (default: 1.5). |

### Physics & Chemistry
| Option | Description |
| :--- | :--- |
| `--forcefield` | Forcefield for minimization (default: `amber14-all.xml`). |
| `--solvent` | Solvent model: `obc2`, `obc1`, `gbn`, `gbn2`, `hct`, `explicit`. |
| `--cap-termini` | Add N-terminal Acetyl (ACE) and C-terminal N-methylamide (NME) caps. |
| `--ph` | pH for determining protonation states (default: 7.4). |

## Advanced Modes (`--mode`)

`synth-pdb` supports several specialized operation modes:

-   `generate`: (Default) Generate a single structure.
-   `decoys`: Generate an ensemble of structures (decoys).
-   `cryo-em`: Generate 3D density maps (MRC format) from structures or ensembles.
-   `saxs`: Simulate Small-Angle X-ray Scattering (SAXS) profiles.
-   `docking`: Prepare structures for docking (PQR format, charge assignment).
-   `pymol`: Generate PyMOL scripts for visualization.
-   `dataset`: Bulk generation for machine learning datasets.
-   `ai`: Structure interpolation and clustering.

### Cryo-EM Mode Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `--resolution` | Target resolution in Angstroms (Å). | 3.0 |
| `--grid-spacing` | Voxel size in Angstroms (Å). | 1.0 |
| `--mrc-output` | Filename for the output density map. | `synthetic_map.mrc` |

### SAXS Mode Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `--q-max` | Maximum scattering vector $q$ (Å⁻¹). | 0.5 |
| `--saxs-points` | Number of points in the $I(q)$ curve. | 51 |
| `--saxs-output` | Filename for the output `.dat` file. | `synthetic_saxs.dat` |

### AI Mode Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `--ai-op` | AI operation: `interpolate` or `cluster`. | - |
| `--start-pdb` | Start PDB file for `interpolate`. | - |
| `--end-pdb` | End PDB file for `interpolate`. | - |
| `--steps` | Number of steps for `interpolate`. | 10 |
| `--input-pattern` | Glob pattern for input PDB files for `cluster` (e.g., 'decoys/*.pdb'). | - |
| `--n-clusters` | Number of clusters to form for `cluster`. | 5 |

## Visualization

Use the `--visualize` flag to open the generated structure in a browser-based 3D viewer (powered by 3Dmol.js).

```bash
python -m synth_pdb.main --sequence "MEELQK" --visualize
```
