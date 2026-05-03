# synth-pdb

<p align="left">
  <a href="https://pypi.org/project/synth-pdb/"><img src="https://img.shields.io/pypi/v/synth-pdb.svg?label=pypi%20package&color=brightgreen" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://doi.org/10.5281/zenodo.18357242"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18357242.svg" alt="DOI"></a>
  <a href="https://github.com/elkins/synth-pdb/actions/workflows/test.yml"><img src="https://github.com/elkins/synth-pdb/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/elkins/synth-pdb"><img src="https://codecov.io/gh/elkins/synth-pdb/branch/master/graph/badge.svg" alt="codecov"></a>
</p>

**Generate realistic PDB files with mixed secondary structures for bioinformatics testing, education, and tool development.**

> ⚠️ **Important**: The generated structures use idealized geometries and may contain violations of standard structural constraints. These files are intended for **testing computational tools** and **educational demonstrations**, not for simulation or experimental validation.

---

## Why synth-pdb?

In structural biology and bioinformatics, researchers frequently require datasets of protein structures to test algorithms, train machine learning models, or validate analytical pipelines. While the Protein Data Bank (PDB) contains over 200,000 experimental structures, relying solely on experimental data has limitations:

1.  **Bias**: PDB data is biased toward crystallizable or stable proteins.
2.  **Complexity**: Experimental files often contain artifacts, missing atoms, or non-standard residues.
3.  **Lack of Ground Truth**: For NMR assignment or structure calculation, "perfect" synthetic data is essential for unit testing.

`synth-pdb` fills this gap by providing a lightweight, deterministic generator that produces chemically valid, full-atom PDB files with user-defined secondary structures (helices, sheets) in seconds.

## Educational Philosophy: Code as Textbook 🎓

`synth-pdb` is built on the principle that scientific software should be readable and educational.

*   **Code as Textbook**: We reject "black box" algorithms. Our source code (e.g., `generator.py`, `physics.py`) is heavily annotated with biophysical reasons—explaining concepts like Boltzmann weighting, order parameters ($S^2$), and NOE distance dependence ($r^{-6}$).
*   **Visual Learning**: With the `--visualize` flag, students can instantly see how abstract concepts manifest in 3D, bridging the gap between equations and biology.
*   **Integrity**: Specialized tests ensure educational notes remain in the codebase, preventing refactoring from stripping away the scientific context.

---

## Key Features

### ✨ Structure Generation
- **Full atomic representation** with backbone and side-chain heavy atoms + hydrogens.
- **Customizable sequence** (1-letter or 3-letter amino acid codes).
- **Conformational diversity**: Generate alpha helices, beta sheets, extended chains, or random conformations.
- **Rotamer-based side-chain placement** for all 20 standard amino acids using the Dunbrack library.
- **Advanced Chemistry**: Metal coordination (Zn2+), Disulfide bonds (SSBOND), and PTM support (SEP, TPO, PTR).

### 🔬 Validation Suite
- **Geometric Checks**: Bond length, bond angle (Engh & Huber Z-scores), and peptide plane planarity.
- **Ramachandran Checking**: Upgraded to Top2018 high-resolution datasets.
- **Physical Validation**: Steric clash detection and SASA-based burial ratios.

### ⚙️ Quality Control & Physics
- **--best-of-N**: Generate multiple structures and select the one with the fewest violations.
- **Energy Minimization**: Relax structures using OpenMM (Implicit Solvent / AMBER forcefield).
- **Quality Filtering**: Integrated Random Forest and GNN classifiers for structural plausibility.

---

## 📚 Interactive Tutorial Catalog

Explore `synth-pdb` through our curated interactive tutorials. Each notebook can be opened directly in Google Colab.

### 🔬 Core Biophysics & NMR
| Tutorial | Difficulty | Time | Action |
| :--- | :--- | :---: | :--- |
| [**The Virtual NMR Spectrometer**](tutorials/virtual_nmr_spectrometer.ipynb) | ⭐⭐ | 25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/virtual_nmr_spectrometer.ipynb) |
| [**Cryo-EM & SAXS Lab**](tutorials/cryo_em_saxs_lab.ipynb) | ⭐ | 20 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/cryo_em_saxs_lab.ipynb) |
| [**BMRB Validation Pipeline**](tutorials/bmrb_validation.ipynb) | ⭐⭐ | 25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/bmrb_validation.ipynb) |
| [**Ubiquitin Validation Suite**](tutorials/ubiquitin_chemical_shift_validation.ipynb) | ⭐⭐⭐ | 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/ubiquitin_chemical_shift_validation.ipynb) |
| [**RDC Alignment Tensor Explorer**](tutorials/rdc_alignment_explorer.ipynb) | ⭐⭐ | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/rdc_alignment_explorer.ipynb) |
| [**RPF Score Validation**](tutorials/nmr_validation_rpf.ipynb) | ⭐⭐ | 25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/nmr_validation_rpf.ipynb) |
| [**NeRF Geometry Lab**](tutorials/nerf_geometry_lab.ipynb) | ⭐⭐ | 25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/nerf_geometry_lab.ipynb) |
| [**The GFP Molecular Forge**](tutorials/gfp_molecular_forge.ipynb) | ⭐⭐ | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/gfp_molecular_forge.ipynb) |
| [**IDP Conformational Ensembles**](tutorials/idp_ensemble_validation.ipynb) | ⭐⭐⭐ | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/idp_ensemble_validation.ipynb) |
| [**AlphaFold pLDDT vs NMR S²**](tutorials/alphafold_vs_nmr_dynamics.ipynb) | ⭐⭐⭐ | 35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/alphafold_vs_nmr_dynamics.ipynb) |

### 🤖 ML & AI Integration
| Tutorial | Difficulty | Time | Action |
| :--- | :--- | :---: | :--- |
| [**Bulk Dataset Factory**](tutorials/dataset_factory.ipynb) | ⭐ | 15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/ml_integration/dataset_factory.ipynb) |
| [**Hard Decoy Challenge**](tutorials/hard_decoy_challenge.ipynb) | ⭐⭐⭐ | 35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/ml_integration/hard_decoy_challenge.ipynb) |
| [**PLM Embeddings (ESM-2)**](tutorials/plm_embeddings.ipynb) | ⭐⭐ | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/ml_integration/plm_embeddings.ipynb) |
| [**Co-evolution Factory**](tutorials/coevolution_msa_factory.ipynb) | ⭐⭐⭐ | 35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/interactive_tutorials/coevolution_msa_factory.ipynb) |
| [**6D Orientogram Lab**](tutorials/orientogram_lab.ipynb) | ⭐⭐⭐ | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/ml_integration/orientogram_lab.ipynb) |
| [**Drug Discovery Pipeline**](tutorials/drug_discovery_pipeline.ipynb) | ⭐⭐⭐ | 35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-pdb/blob/master/examples/ml_integration/drug_discovery_pipeline.ipynb) |

### 🎓 Learning Paths

Choose a path based on your background and goals:

#### 🤖 **For ML Engineers**
*Build AI models with synthetic protein data*

1. **AI Protein Data Factory** (15 min) - Learn zero-copy data handover to PyTorch/JAX.
2. **Bulk Dataset Factory** (15 min) - Generate thousands of training samples.
3. **Hard Decoy Challenge** (35 min) - Create negative samples for robust training.
4. **PLM Embeddings (ESM-2)** (30 min) - Add evolutionary context as per-residue node features.

#### 🔬 **For Biophysicists**
*Understand structure, dynamics, and spectroscopy*

1. **NeRF Geometry Lab** (25 min) - Learn internal coordinate systems.
2. **Virtual NMR Spectrometer** (25 min) - Predict relaxation rates and chemical shifts.
3. **Protein Quality Assessment** (25 min) - Validate structure quality and geometry.
4. **GFP Molecular Forge** (30 min) - Explore chromophore chemistry.
5. **AlphaFold pLDDT vs NMR S²** (35 min) - Contrast AI rigidity with physical dynamics.

#### 💊 **For Drug Designers**
*Design and optimize therapeutic peptides*

1. **Drug Discovery Pipeline** (35 min) - End-to-end peptide library to lead selection.
2. **Macrocycle Design Lab** (20 min) - Create head-to-tail cyclic peptides.
3. **Bio-Active Hormone Lab** (20 min) - Model bioactive peptide hormones.
4. **Hard Decoy Challenge** (35 min) - Generate decoys for docking validation.

---

## Quick Visual Demo

Run this command to generate a **Leucine Zipper**, **minimize** its energy using OpenMM, and **visualize** it in your browser:

```bash
synth-pdb --sequence "LKELEKELEKELEKELEKELEKEL" --conformation alpha --minimize --visualize
```

## Citation

If you use `synth-pdb` in your research, please cite it:

```bibtex
@software{elkins_synth_pdb_2026,
  author = {Elkins, George},
  title = {synth-pdb: High-Performance Protein Structure Generator},
  url = {https://github.com/elkins/synth-pdb},
  version = {1.35.0},
  year = {2026}
}
```
