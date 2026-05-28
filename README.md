# 🧬 synth-pdb: High-Fidelity PDB Generation

[![PyPI version](https://img.shields.io/pypi/v/synth-pdb.svg)](https://pypi.org/project/synth-pdb/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/synth-pdb.svg)](https://pypi.org/project/synth-pdb/)
[![Tests](https://github.com/elkins/synth-pdb/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-pdb/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`synth-pdb` is a command-line tool and library for generating realistic, full-atomic Protein Data Bank (PDB) files with mixed secondary structures for benchmarking, education, and software testing.

---

### 🧪 For Structural Biologists
*   **Realistic Models:** Generates structures with physically sensible bond lengths, angles, and omega-torsions.
*   **Validation Benchmarks:** Create "perfect" ground-truth models to test your structure-validation tools or NMR assignment algorithms.

### 🤖 For Machine Learning Geeks
*   **Dataset Augmentation:** Generate thousands of unique protein-like topologies to train GNNs or Diffusion models.
*   **Fuzz Testing:** Create "corrupted" or edge-case PDB files to ensure your structural parsers are robust to experimental noise and non-standard residues.

---

## 🚀 Key Features

*   **Mixed Secondary Structure:** Generate realistic alpha-helices and beta-sheets in the same chain.
*   **Full Atomic Representation:** Includes all backbone and side-chain atoms (not just C-alpha).
*   **Deterministic Generation:** Reproduce any structure using a seed for rigorous benchmarking.

## 📦 Installation

```bash
pip install synth-pdb
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
