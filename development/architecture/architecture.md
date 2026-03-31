# Architecture

This document describes the high-level architecture of `synth-pdb`, a high-performance protein structure generator.

## Design Philosophy

`synth-pdb` is designed to be a modular, extensible, and high-performance tool for generating synthetic protein structures. It follows a "Physics-First" approach, where structures are built using fundamental geometric and biophysical principles rather than relying on structural templates.

## Core Components

The project is organized into several key modules, each responsible for a specific aspect of protein generation and analysis:

### 1. Structure Generation (`synth_pdb/generator.py`)
This is the heart of the tool. It handles:
- **Sequence Resolution:** Determining the amino acid sequence from user input (length, specific sequence, or plausible frequencies).
- **Backbone Construction:** Using the NeRF (Natural Extension Reference Frame) algorithm to build the N-CA-C backbone chain from internal coordinates (bond lengths, angles, and dihedrals).
- **Side-Chain Placement:** Adding side-chain atoms based on rotamer libraries and backbone-dependent distributions.

### 2. Geometry & Physics (`synth_pdb/geometry.py`, `synth_pdb/physics.py`)
- **Geometry:** Provides low-level mathematical functions for 3D transformations, dihedral angle calculations, and internal-to-Cartesian coordinate conversion.
- **Physics Engine:** Implements energy functions (clash detection, electrostatic interactions, hydrogen bonding) and optimization algorithms (energy minimization, simulated annealing) to refine structures.

### 3. NMR & Biophysics (`synth_pdb/nmr.py`, `synth_pdb/chemical_shifts.py`, `synth_pdb/rdc.py`, `synth_pdb/relaxation.py`)
These modules bridge the gap between static structures and experimental observables:
- **Chemical Shifts:** Predicts backbone and side-chain chemical shifts (supporting multiple predictors like SHIFTX2).
- **RDCs:** Calculates Residual Dipolar Couplings for different alignment tensors.
- **Relaxation:** Predicts NMR relaxation parameters (S², T1, T2, NOE) based on structural dynamics and tumbling.

### 4. Evolution & MSA (`synth_pdb/evolution.py`, `synth_pdb/msa.py`)
- Simulates protein evolution to generate Multiple Sequence Alignments (MSAs).
- Models co-evolutionary constraints to create realistic sequence variations that maintain structural integrity.

### 5. Validation (`synth_pdb/validator.py`)
Ensures that generated structures are biophysically plausible by checking:
- Steric clashes.
- Bond lengths and angles.
- Ramachandran distribution.
- Chirality.

## Data Flow

A typical structure generation workflow follows these steps:

1.  **Input Parsing:** `synth_pdb/main.py` parses CLI arguments.
2.  **Sequence Generation:** A sequence is generated or provided.
3.  **Backbone Building:** `generator.py` uses `geometry.py` to build the backbone based on secondary structure presets (e.g., alpha-helix, beta-sheet).
4.  **Side-Chain Addition:** Rotamers are added to the backbone.
5.  **Refinement:** `physics.py` and `packing.py` optimize the structure to resolve clashes and minimize energy.
6.  **Observables Calculation:** If requested, NMR observables (shifts, RDCs, etc.) are calculated.
7.  **Output:** The final structure is saved as a PDB file, and metadata is exported to JSON or NEF formats.

## Extensibility

The modular design allows researchers to easily plug in new:
- **Rotamer Libraries:** By updating `synth_pdb/data.py`.
- **Force Fields:** By extending `synth_pdb/physics.py`.
- **NMR Predictors:** By adding new modules that implement the predictor interface.
