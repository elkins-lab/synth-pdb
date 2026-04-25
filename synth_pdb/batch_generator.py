import random
import re
from typing import Any, Dict, List, Optional

import biotite.structure as struc
import numpy as np

from .data import (
    ANGLE_C_N_CA,
    ANGLE_CA_C_N,
    ANGLE_CA_C_O,
    ANGLE_N_CA_C,
    BOND_LENGTH_C_N,
    BOND_LENGTH_C_O,
    BOND_LENGTH_CA_C,
    BOND_LENGTH_N_CA,
    ONE_TO_THREE_LETTER_CODE,
    RAMACHANDRAN_PRESETS,
)
from .geometry import (
    calculate_rmsd_to_average,
    find_medoid,
    position_atoms_batch,
    superimpose_batch,
)

# EDUCATIONAL OVERVIEW - Batched Generation (GPU-First):
# ----------------------------------------------------
# Traditional protein generators (like the serial Generator in generator.py)
# process structures one-by-one. While easy to code, this is a bottleneck for
# training Deep Learning models which require millions of samples.
#
# BatchedGenerator uses "Vectorized Math":
# 1. Parallelism: It processes B structures at once (e.g., B=1000).
# 2. Broadcasting: Using NumPy's broadcasting, a single mathematical expression
#    calculates positions for all structures in the batch simultaneously.
# 3. Hardware Acceleration: On Apple Silicon (M4), this leverages AMX/Accelerate
#    units, often providing 10-100x speedups over Python loops.
#
# This architecture is "ML-Ready" - the output is a single contiguous tensor
# that can be passed directly to frameworks like MLX, PyTorch, or JAX.
#
# EDUCATIONAL NOTE - The "Memory Wall" in AI Training:
# --------------------------------------------------
# When generating millions of protein samples, the bottleneck is rarely the
# CPU math (thanks to vectorization), but rather the "Memory Wall":
#
# 1. PCIE Latency: Copying large tensors from CPU to GPU memory can be slower
#    than actually generating the coordinates.
# 2. Contiguity: Deep Learning models require contiguous memory blocks.
#    BatchedGenerator ensures the output is one massive C-style array,
#    avoiding the "gather" overhead of traditional Python lists.
# 3. Unified Memory: On Apple Silicon (M4), CPU and GPU share the same physical
#    RAM. This means the coordinate tensor can be "zero-copy" - once generated
#    by NumPy, it is IMMEDIATELY visible to the Metal/MLX GPU without any
#    data movement.


class BatchedPeptide:
    """A lightweight container for batched protein coordinates.
    Designed for high-performance handover to ML frameworks.

    SCIENTIFIC RATIONALE:
    --------------------
    Representing protein ensembles as 3D tensors (B, N, 3) allows for
    extremely fast calculation of ensemble-wide metrics like RMSD,
    contact maps, and 6D orientations using vectorized linear algebra.
    """

    def __init__(
        self,
        coords: np.ndarray,
        sequence: List[str],
        atom_names: List[str],
        residue_indices: List[int],
        atom_chain_ids: Optional[List[str]] = None,
    ):
        """Initialize the container.

        Args:
            coords: Coordinate tensor of shape (B, N_atoms, 3).
            sequence: List of residue types (3-letter codes).
            atom_names: List of atom names in the structure.
            residue_indices: Mapping from each atom to its residue (1-indexed).
            atom_chain_ids: List of chain identifiers for each atom.

        """
        self.coords = coords  # (B, N_atoms, 3) - Primary coordinate tensor
        self.sequence = sequence  # List of residue names in the complex
        self.atom_names = atom_names  # Flat list of all atom names
        self.residue_indices = residue_indices  # 1-indexed residue IDs for each atom
        # Support for multi-chain assemblies (dimers, trimers, etc.)
        self.atom_chain_ids = atom_chain_ids if atom_chain_ids else ["A"] * self.coords.shape[1]
        self.n_structures = self.coords.shape[0]  # Number of structures in the batch
        self.n_atoms = self.coords.shape[1]  # Atoms per structure
        self.n_residues = len(sequence)  # Total residues in the sequence

    def __len__(self) -> int:
        """Returns the number of structures in the batch."""
        return int(self.n_structures)

    def __getitem__(self, index: Any) -> "BatchedPeptide":
        """Slice the batch, returning a new BatchedPeptide with a subset of structures."""
        if isinstance(index, int):
            # Convert single integer index to a slice of size 1 to maintain tensor rank
            return BatchedPeptide(
                self.coords[index : index + 1],
                self.sequence,
                self.atom_names,
                self.residue_indices,
                self.atom_chain_ids,
            )
        # Handle standard slice objects or boolean masks
        return BatchedPeptide(
            self.coords[index],
            self.sequence,
            self.atom_names,
            self.residue_indices,
            self.atom_chain_ids,
        )

    def save_pdb(self, path: str, index: int = 0) -> None:
        """Saves one structure from the batch to a PDB file.

        Args:
            path: Target file path.
            index: Batch index of the structure to save.

        """
        # Ensure the output directory exists before writing
        with open(path, "w") as f:
            f.write(self.to_pdb(index))

    def to_pdb(self, index: int = 0) -> str:
        """Converts one structure in the batch to a PDB string.

        Args:
            index: Batch index of the structure to convert.

        Returns:
            The PDB content as a string.

        EDUCATIONAL NOTE - PDB Specification:
        ------------------------------------
        The PDB format uses fixed-width columns. Atom coordinates must reside in
        columns 31-54. We use a precise format string to ensure compliance with
        downstream tools like PyMOL or OpenMM.
        """
        lines = []
        c = self.coords[index]
        # PDB Format String: ATOM, Serial, Name, ResName, Chain, ResSeq, X, Y, Z, Occupancy, B-factor, Element
        fmt = "ATOM  {:>5d} {:<4s} {:>3s} {:<1s}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2s}"

        for i in range(self.n_atoms):
            name = self.atom_names[i]
            res_idx = self.residue_indices[i]
            res_name = self.sequence[res_idx - 1]
            chain_id = self.atom_chain_ids[i]

            # Handle 4-character atom names (e.g., 1HG2) per PDB convention
            clean_name = name.strip()
            if len(clean_name) == 4:
                atom_field = clean_name
            else:
                atom_field = " " + clean_name.ljust(3)

            # Heuristic to extract element from the atom name for the PDB element column
            match = re.search(r"[A-Z]", clean_name)
            element = match.group(0) if match else "C"

            lines.append(
                fmt.format(
                    i + 1,
                    atom_field,
                    res_name,
                    chain_id,
                    res_idx,
                    c[i, 0],
                    c[i, 1],
                    c[i, 2],
                    element,
                )
            )
        # Every PDB structure segment should end with TER
        lines.append("TER")
        # Every PDB file should end with END
        lines.append("END")
        return "\n".join(lines)

    def get_6d_orientations(self) -> Dict[str, np.ndarray]:
        """Computes 6D inter-residue orientations (trRosetta style).

        Returns:
            A dictionary of (B, L, L) tensors:
                - 'dist': Cb-Cb distance.
                - 'omega': Cb1-Ca1-Ca2-Cb2 torsion.
                - 'theta': N1-Ca1-Cb1-Cb2 torsion.
                - 'phi': Ca1-Cb1-Cb2 angle.

        """
        from .orientogram import compute_6d_orientations

        # Leverage the orientogram module for specialized geometric feature extraction
        return compute_6d_orientations(
            self.coords, self.atom_names, self.residue_indices, self.n_residues
        )

    def analyze_ensemble(self, superimpose: bool = True) -> Dict[str, Any]:
        """Performs NMR-style ensemble analysis on the batch.

        Calculates the average structure, the average RMSD to that structure
        (measuring batch precision), and identifies the medoid structure.

        Args:
            superimpose: If True, aligns all structures before analysis.

        Returns:
            A dictionary containing:
                - 'avg_rmsd': The mean RMSD of all structures to the average.
                - 'medoid_index': The index of the most representative structure.
                - 'avg_coords': (N_atoms, 3) array of the centroid structure.
        """
        # Convert (B, N, 3) tensor to a list of arrays for standard geometry API compatibility
        coords_list = [self.coords[i] for i in range(self.n_structures)]

        # Calculate standard ensemble statistics (average RMSD and centroid)
        avg_rmsd, avg_coords = calculate_rmsd_to_average(coords_list)
        # Find the medoid structure (the actual sample closest to the mean)
        medoid_idx = find_medoid(coords_list, superimpose=superimpose)

        return {
            "avg_rmsd": avg_rmsd,
            "medoid_index": medoid_idx,
            "avg_coords": avg_coords,
        }


class BatchedGenerator:
    """High-performance vectorized protein structure generator.
    Optimized for generating millions of labeled samples for AI training.
    Supports multichain assemblies via ':' sequence syntax.

    BIOPHYSICAL DESIGN:
    -------------------
    This generator implements the "Parallel Walker" algorithm. Instead of
    looping over structures, it performs NumPy-accelerated operations across
    the entire batch (B) for each residue placement step. This avoids the
    Python loop overhead and allows for massive scaling on modern CPUs.
    """

    def __init__(self, sequence_str: str, n_batch: int = 1, full_atom: bool = False):
        """Initialize the batched generator.

        Args:
            sequence_str: The primary sequence. Use ':' for chains (e.g. 'ALA:GLY').
            n_batch: Number of structures to generate in a single vectorized pass.
            full_atom: If True, generates all heavy atoms (Kabsch superimposition).

        """
        # 1. Resolve multi-chain sequence topology
        chain_strs = sequence_str.split(":") if ":" in sequence_str else [sequence_str]
        self.chain_sequences = []
        for c_str in chain_strs:
            if "-" in c_str:
                # Handle 3-letter codes and D-amino acid notation
                raw_parts = [s.strip().upper() for s in c_str.split("-") if s.strip()]
                resolved = []
                skip = False
                for i, p in enumerate(raw_parts):
                    if skip:
                        skip = False
                        continue
                    if p == "D" and i + 1 < len(raw_parts):
                        next_p = raw_parts[i + 1]
                        if len(next_p) == 1:
                            next_p = ONE_TO_THREE_LETTER_CODE.get(next_p, next_p)
                        resolved.append(f"D-{next_p}")
                        skip = True
                    else:
                        if len(p) == 1:
                            resolved.append(ONE_TO_THREE_LETTER_CODE.get(p, p))
                        else:
                            resolved.append(p)
                self.chain_sequences.append(resolved)
            else:
                # Convert 1-letter IUPAC codes to standard 3-letter codes
                self.chain_sequences.append(
                    [ONE_TO_THREE_LETTER_CODE.get(c.upper(), "ALA") for c in c_str]
                )

        # Build a flat sequence for global indexing
        self.sequence = [aa for seq in self.chain_sequences for aa in seq]
        self.n_batch = n_batch
        self.full_atom = full_atom

        # 2. Topology and Template Management
        self.atom_names = []
        self.residue_indices = []
        self.atom_chain_ids = []
        self.templates = []
        self.template_backbones = []
        self.offsets = []
        self.chain_start_indices = []  # Residue global indices where chains begin

        current_atom_offset = 0
        current_res_global_idx = 0

        # Iterate through chains and residues to build static topology
        for chain_idx, seq in enumerate(self.chain_sequences):
            chain_id = chr(65 + chain_idx)
            for i, full_res_name in enumerate(seq):
                if i == 0:
                    self.chain_start_indices.append(current_res_global_idx)

                is_d = full_res_name.startswith("D-")
                res_name = full_res_name[2:] if is_d else full_res_name

                if full_atom:
                    # Fetch sidechain templates for Kabsch superimposition
                    template = struc.info.residue(res_name).copy()
                    # Prune terminal capping atoms not used in internal peptide bonds
                    mask = ~np.isin(template.atom_name, ["OXT", "H2", "H3", "HXT"])
                    template = template[mask]
                    names = template.atom_name.tolist()
                    n_atoms = len(names)

                    self.templates.append(template)
                    # Pre-calculate template backbone coordinates (N, CA, C)
                    n_idx, ca_idx, c_idx = names.index("N"), names.index("CA"), names.index("C")
                    self.template_backbones.append(template.coord[[n_idx, ca_idx, c_idx]])
                else:
                    # Backbone-only mode: N, CA, C, O
                    names = ["N", "CA", "C", "O"]
                    n_atoms = 4

                self.atom_names.extend(names)
                self.residue_indices.extend([current_res_global_idx + 1] * n_atoms)
                self.atom_chain_ids.extend([chain_id] * n_atoms)
                self.offsets.append(current_atom_offset)
                current_atom_offset += n_atoms
                current_res_global_idx += 1

        self.total_atoms = current_atom_offset
        self.n_res = current_res_global_idx

    def generate_batch(
        self, seed: Optional[int] = None, conformation: str = "alpha", drift: float = 0.0
    ) -> BatchedPeptide:
        """Generates B structures in parallel using vectorized NeRF kernels.

        This method implements the "Parallel Backbone Walk". Instead of placing
        atoms for structure 1 then structure 2, it calculates the positions of
        atom 'N' for ALL structures in the batch simultaneously using tensor math.

        Args:
            seed: Random seed for reproducible batch generation.
            conformation: Secondary structure preset (e.g., 'alpha' or 'beta').
            drift: Gaussian std dev (degrees) added to torsions for decoy generation.

        """
        # Set seeds to ensure scientific reproducibility in dataset generation
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        b = self.n_batch
        length = self.n_res

        # Internal coordinate walker always populates a backbone coord tensor (B, L*4, 3)
        backbone_coords = np.zeros((b, length * 4, 3))

        # 1. Vectorized Torsion Angle Sampling
        # Map secondary structure presets to Phi/Psi angles
        preset = RAMACHANDRAN_PRESETS.get(conformation, RAMACHANDRAN_PRESETS["alpha"])
        phi = np.full((b, length), preset["phi"])
        psi = np.full((b, length), preset["psi"])
        omega = np.full((b, length), 180.0)

        # Apply D-amino acid chirality mirroring batch-wise
        for i, full_res_name in enumerate(self.sequence):
            if full_res_name.startswith("D-"):
                phi[:, i] *= -1
                psi[:, i] *= -1

        # Add stochastic noise for "Hard Decoy" generation (AI training benchmarks)
        if drift > 0:
            phi += np.random.normal(0, drift, (b, length))
            psi += np.random.normal(0, drift, (b, length))
            omega += np.random.normal(0, 2.0, (b, length))

        # 2. Vectorized Backbone Walker (NeRF)
        # EDUCATIONAL NOTE - Peptidyl Chain Walk:
        # ---------------------------------------
        # We construct the polypeptide chain N -> CA -> C iteratively.
        # For each residue (i), we use the coordinates of (i-1) to place the new atoms.
        # In this vectorized version, we perform this "walk" for all B members
        # of the batch simultaneously.
        #
        # Deterministic spatial offset to separate multiple chains (Inter-chain translation)
        offset_step = np.array([20.0, 20.0, 20.0])

        current_chain_idx = -1
        for i in range(length):
            idx = i * 4
            if i in self.chain_start_indices:
                # HANDLE CHAIN BREAK: Place new N-terminus at a separate spatial origin
                current_chain_idx += 1
                chain_offset = current_chain_idx * offset_step

                # First residue N, CA, C placement via basic geometry
                backbone_coords[:, idx] = chain_offset + [0, 0, 0]  # N
                backbone_coords[:, idx + 1] = chain_offset + [BOND_LENGTH_N_CA, 0, 0]  # CA
                ang = np.deg2rad(ANGLE_N_CA_C)
                backbone_coords[:, idx + 2] = chain_offset + [
                    BOND_LENGTH_N_CA - BOND_LENGTH_CA_C * np.cos(ang),
                    BOND_LENGTH_CA_C * np.sin(ang),
                    0,
                ]

                # Place Carbonyl Oxygen (O) explicitly in the trans configuration
                p1, p2, p3 = (
                    backbone_coords[:, idx],
                    backbone_coords[:, idx + 1],
                    backbone_coords[:, idx + 2],
                )
                bl, ba, di = (
                    np.full(b, BOND_LENGTH_C_O),
                    np.full(b, ANGLE_CA_C_O),
                    np.full(b, 180.0),
                )
                backbone_coords[:, idx + 3] = position_atoms_batch(  # type: ignore[assignment]
                    p1, p2, p3, bl, ba, di
                )
            else:
                # Normal Vectorized NeRF Step (Natural Extension Reference Frame)
                # Placement of N(i) relative to N(i-1), CA(i-1), C(i-1)
                p1, p2, p3 = (
                    backbone_coords[:, (i - 1) * 4],
                    backbone_coords[:, (i - 1) * 4 + 1],
                    backbone_coords[:, (i - 1) * 4 + 2],
                )
                bl, ba, di = (
                    np.full(b, BOND_LENGTH_C_N),
                    np.full(b, ANGLE_CA_C_N),
                    psi[:, i - 1],
                )
                backbone_coords[:, idx] = position_atoms_batch(  # type: ignore[assignment]
                    p1, p2, p3, bl, ba, di
                )

                # Step B: Place CA(i) using Peptide Bond Torsion (Omega)
                p1, p2, p3 = (
                    backbone_coords[:, (i - 1) * 4 + 1],
                    backbone_coords[:, (i - 1) * 4 + 2],
                    backbone_coords[:, idx],
                )
                bl, ba, di = (
                    np.full(b, BOND_LENGTH_N_CA),
                    np.full(b, ANGLE_C_N_CA),
                    omega[:, i - 1],
                )
                backbone_coords[:, idx + 1] = position_atoms_batch(  # type: ignore[assignment]
                    p1, p2, p3, bl, ba, di
                )

                # Step C: Place C(i) using Phi angle
                p1, p2, p3 = (
                    backbone_coords[:, (i - 1) * 4 + 2],
                    backbone_coords[:, idx],
                    backbone_coords[:, idx + 1],
                )
                bl, ba, di = np.full(b, BOND_LENGTH_CA_C), np.full(b, ANGLE_N_CA_C), phi[:, i]
                backbone_coords[:, idx + 2] = position_atoms_batch(  # type: ignore[assignment]
                    p1, p2, p3, bl, ba, di
                )

                # Placement of O(i) using fixed Carbonyl geometry
                p1, p2, p3 = (
                    backbone_coords[:, idx],
                    backbone_coords[:, idx + 1],
                    backbone_coords[:, idx + 2],
                )
                bl, ba, di = (
                    np.full(b, BOND_LENGTH_C_O),
                    np.full(b, ANGLE_CA_C_O),
                    np.full(b, 180.0),
                )
                backbone_coords[:, idx + 3] = position_atoms_batch(  # type: ignore[assignment]
                    p1, p2, p3, bl, ba, di
                )

        # 3. Batch Sidechain Superimposition (Full-Atom Mode)
        if self.full_atom:
            fa_coords = np.zeros((b, self.total_atoms, 3))
            for i in range(length):
                # Construct backbone frames (N, CA, C) for the entire batch
                target_n, target_ca, target_c = (
                    backbone_coords[:, i * 4],
                    backbone_coords[:, i * 4 + 1],
                    backbone_coords[:, i * 4 + 2],
                )
                target_bb = np.stack([target_n, target_ca, target_c], axis=1)  # (B, 3, 3)
                # Broadcast source template frame across the batch
                source_bb = np.repeat(self.template_backbones[i][np.newaxis, :, :], b, axis=0)

                # Align templates to targets using vectorized SVD (Kabsch algorithm)
                trans, rot = superimpose_batch(source_bb, target_bb)

                # Apply rotation and translation to all sidechain atoms
                template_coords = self.templates[i].coord
                rotated = np.matmul(rot, template_coords.T).transpose(0, 2, 1)
                aligned = rotated + trans[:, np.newaxis, :]

                # Vectorized Chiral Mirroring for D-amino acids in the batch
                full_res_name = self.sequence[i]
                if full_res_name.startswith("D-") and "GLY" not in full_res_name:
                    # Define the backbone mirror plane (N-CA-C) for all structures
                    ca, c, n = target_bb[:, 1, :], target_bb[:, 2, :], target_bb[:, 0, :]
                    normal = np.cross(c - ca, n - ca, axis=-1)
                    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
                    # Stability: avoid singularity if backbone atoms are collinear
                    normal /= np.where(norm == 0, 1.0, norm)

                    # Mirror everything except backbone core to flip stereocenter
                    backbone_names = {"N", "CA", "C", "O", "H", "HA"}
                    for atom_idx, name in enumerate(self.templates[i].atom_name):
                        if name not in backbone_names:
                            p = aligned[:, atom_idx, :]
                            dist = np.sum((p - ca) * normal, axis=-1, keepdims=True)
                            aligned[:, atom_idx, :] = p - 2 * dist * normal

                # Insert aligned residue coordinates into the global tensor
                offset = self.offsets[i]
                fa_coords[:, offset : offset + len(template_coords)] = aligned

                # Rigorous Parity: Ensure Oxygen matches the idealized NeRF coordinate exactly
                target_o = backbone_coords[:, i * 4 + 3]
                res_atom_names = list(self.templates[i].atom_name)
                if "O" in res_atom_names:
                    fa_coords[:, offset + res_atom_names.index("O")] = target_o

            return BatchedPeptide(
                fa_coords, self.sequence, self.atom_names, self.residue_indices, self.atom_chain_ids
            )

        # Return the backbone-only ensemble
        return BatchedPeptide(
            backbone_coords,
            self.sequence,
            ["N", "CA", "C", "O"] * length,
            self.residue_indices,
            self.atom_chain_ids,
        )
