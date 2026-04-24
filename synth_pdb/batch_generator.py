import random
from typing import Any, Dict, List, Optional

import biotite.structure as struc
import numpy as np

from .data import (
    ANGLE_C_N_CA,
    ANGLE_CA_C_N,
    ANGLE_N_CA_C,
    BOND_LENGTH_C_N,
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
    """

    def __init__(
        self,
        coords: np.ndarray,
        sequence: List[str],
        atom_names: List[str],
        residue_indices: List[int],
    ):
        """Initialize the container.

        Args:
            coords: Coordinate tensor of shape (B, N_atoms, 3).
            sequence: List of residue types (3-letter codes).
            atom_names: List of atom names in the structure.
            residue_indices: Mapping from each atom to its residue (1-indexed).

        """
        self.coords = coords  # (B, N_atoms, 3)
        self.sequence = sequence
        self.atom_names = atom_names
        self.residue_indices = residue_indices
        self.n_structures = self.coords.shape[0]
        self.n_atoms = self.coords.shape[1]
        self.n_residues = len(sequence)

    def __len__(self) -> int:
        return int(self.n_structures)

    def __getitem__(self, index: int) -> "BatchedPeptide":
        if isinstance(index, int):
            return BatchedPeptide(
                self.coords[index : index + 1], self.sequence, self.atom_names, self.residue_indices
            )
        return BatchedPeptide(
            self.coords[index], self.sequence, self.atom_names, self.residue_indices
        )

    def save_pdb(self, path: str, index: int = 0) -> None:
        """Saves one structure from the batch to a PDB file.

        Args:
            path: Target file path.
            index: Batch index of the structure to save.

        """
        with open(path, "w") as f:
            f.write(self.to_pdb(index))

    def to_pdb(self, index: int = 0) -> str:
        """Converts one structure in the batch to a PDB string.

        Args:
            index: Batch index of the structure to convert.

        Returns:
            The PDB content as a string.

        """
        lines = []
        c = self.coords[index]
        # PDB Format String: ATOM, Serial, Name, ResName, Chain, ResSeq, X, Y, Z, Occupancy, B-factor, Element
        fmt = "ATOM  {:>5d} {:<4s} {:>3s} A{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2s}"

        for i in range(self.n_atoms):
            name = self.atom_names[i]
            res_idx = self.residue_indices[i]
            res_name = self.sequence[res_idx - 1]

            # Atom names with 4 chars start at col 13, others at col 14
            clean_name = name.strip()
            if len(clean_name) == 4:
                atom_field = clean_name
            else:
                atom_field = " " + clean_name.ljust(3)

            # Element is the first non-numeric char of the stripped name
            import re

            match = re.search(r"[A-Z]", clean_name)
            element = match.group(0) if match else "C"

            lines.append(
                fmt.format(i + 1, atom_field, res_name, res_idx, c[i, 0], c[i, 1], c[i, 2], element)
            )
        lines.append("TER")
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
        # Convert List[np.ndarray] for the geometry API
        # self.coords is (B, N, 3)
        coords_list = [self.coords[i] for i in range(self.n_structures)]

        avg_rmsd, avg_coords = calculate_rmsd_to_average(coords_list)
        medoid_idx = find_medoid(coords_list, superimpose=superimpose)

        return {
            "avg_rmsd": avg_rmsd,
            "medoid_index": medoid_idx,
            "avg_coords": avg_coords,
        }


class BatchedGenerator:
    """High-performance vectorized protein structure generator.
    Optimized for generating millions of labeled samples for AI training.
    """

    def __init__(self, sequence_str: str, n_batch: int = 1, full_atom: bool = False):
        """Initialize the batched generator.

        Args:
            sequence_str: The primary sequence (e.g. 'ACDEF' or 'ALA-CYS-ASP').
            n_batch: Number of structures to generate in a single vectorized pass.
            full_atom: If True, generates all heavy atoms (Kabsch superimposition).

        """
        # Resolve sequence
        if "-" in sequence_str:
            raw_parts = [s.strip().upper() for s in sequence_str.split("-") if s.strip()]
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
            self.sequence = resolved
        else:
            self.sequence = [ONE_TO_THREE_LETTER_CODE.get(c.upper(), "ALA") for c in sequence_str]

        self.n_batch = n_batch
        self.n_res = len(self.sequence)
        self.full_atom = full_atom

        # Build Atom Topology & Load Templates
        self.atom_names = []
        self.residue_indices = []
        self.templates = []
        self.template_backbones = []
        self.offsets = []

        current_offset = 0
        for i, full_res_name in enumerate(self.sequence):
            is_d = full_res_name.startswith("D-")
            res_name = full_res_name[2:] if is_d else full_res_name

            if full_atom:
                # Get full-atom template from biotite (Always L-template)
                template = struc.info.residue(res_name).copy()

                # Remove terminal atoms (OXT, H2, H3) to match peptide chain logic
                # (Simple heuristic for now, matches generator.py logic)
                mask = ~np.isin(template.atom_name, ["OXT", "H2", "H3", "HXT"])
                template = template[mask]

                names = template.atom_name.tolist()
                n_atoms = len(names)

                # Ensure N, CA, C are present for superimposition
                if not all(a in names for a in ["N", "CA", "C"]):
                    raise ValueError(f"Template for {res_name} missing core backbone atoms.")

                self.templates.append(template)
                # Store (N, CA, C) coordinates of template for Kabsch (shape 3, 3)
                n_idx = names.index("N")
                ca_idx = names.index("CA")
                c_idx = names.index("C")
                self.template_backbones.append(template.coord[[n_idx, ca_idx, c_idx]])
            else:
                # Backbone only: N, CA, C, O
                names = ["N", "CA", "C", "O"]
                n_atoms = 4

            self.atom_names.extend(names)
            self.residue_indices.extend([i + 1] * n_atoms)
            self.offsets.append(current_offset)
            current_offset += n_atoms

        self.total_atoms = current_offset

    def generate_batch(
        self, seed: Optional[int] = None, conformation: str = "alpha", drift: float = 0.0
    ) -> BatchedPeptide:
        """Generates B structures in parallel.

        This method replaces the traditional per-residue loop with a "Batch Walk".
        Instead of placing atoms for structure 1, then structure 2... it places
        atom 'N' for ALL structures, then 'CA' for ALL structures, and so on.

        Args:
            seed: Random seed for reproducible batch generation.
            conformation: The secondary structure preset to use for all members.
            drift: Gaussian noise (std dev) in degrees. Use this to generate "hard decoys"
                   that challenge AI models with near-native but slightly incorrect geometry.

        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        b = self.n_batch
        length = self.n_res

        # We only generate backbone for now (N, CA, C, O) - 4 atoms per residue
        n_atoms = length * 4
        coords = np.zeros((b, n_atoms, 3))

        # 1. Place first residue (N, CA, C) at origin frame
        coords[:, 0] = [0, 0, 0]  # N
        coords[:, 1] = [BOND_LENGTH_N_CA, 0, 0]  # CA
        ang = np.deg2rad(ANGLE_N_CA_C)
        coords[:, 2] = [
            BOND_LENGTH_N_CA - BOND_LENGTH_CA_C * np.cos(ang),
            BOND_LENGTH_CA_C * np.sin(ang),
            0,
        ]

        # Resolve preset angles
        preset = RAMACHANDRAN_PRESETS.get(conformation, RAMACHANDRAN_PRESETS["alpha"])
        p_phi = preset["phi"]
        p_psi = preset["psi"]

        # Sample torsions for the entire batch (b, length)
        phi = np.full((b, length), p_phi)
        psi = np.full((b, length), p_psi)
        omega = np.full((b, length), 180.0)

        # Mirror phi/psi for D-amino acids
        for i, full_res_name in enumerate(self.sequence):
            if full_res_name.startswith("D-"):
                phi[:, i] *= -1
                psi[:, i] *= -1

        if drift > 0:
            phi += np.random.normal(0, drift, (b, length))
            psi += np.random.normal(0, drift, (b, length))
            omega += np.random.normal(0, 2.0, (b, length))  # Fixed small omega drift

        # EDUCATIONAL NOTE - Peptidyl Chain Walk:
        # We construct the chain N -> CA -> C iteratively.
        # For each residue (i), we use the coordinates of (i-1) to place the new atoms.

        from .data import ANGLE_CA_C_O, BOND_LENGTH_C_O

        for i in range(length):
            idx = i * 4
            if i == 0:
                # Place O(0) using N(0), CA(0), C(0)
                p1, p2, p3 = coords[:, 0], coords[:, 1], coords[:, 2]
                bl, ba, di = (
                    np.full(b, BOND_LENGTH_C_O),
                    np.full(b, ANGLE_CA_C_O),
                    np.full(b, 180.0),
                )
                coords[:, 3] = position_atoms_batch(p1, p2, p3, bl, ba, di)
            else:
                # Place N(i) using N(i-1), CA(i-1), C(i-1)
                p1, p2, p3 = (
                    coords[:, (i - 1) * 4],
                    coords[:, (i - 1) * 4 + 1],
                    coords[:, (i - 1) * 4 + 2],
                )
                bl, ba, di = np.full(b, BOND_LENGTH_C_N), np.full(b, ANGLE_CA_C_N), psi[:, i - 1]  # type: ignore[assignment]
                coords[:, idx] = position_atoms_batch(p1, p2, p3, bl, ba, di)  # type: ignore[assignment]

                # Place CA(i) using CA(i-1), C(i-1), N(i)
                p1, p2, p3 = coords[:, (i - 1) * 4 + 1], coords[:, (i - 1) * 4 + 2], coords[:, idx]
                bl, ba, di = np.full(b, BOND_LENGTH_N_CA), np.full(b, ANGLE_C_N_CA), omega[:, i - 1]  # type: ignore[assignment]
                coords[:, idx + 1] = position_atoms_batch(p1, p2, p3, bl, ba, di)  # type: ignore[assignment]

                # Place C(i) using C(i-1), N(i), CA(i)
                p1, p2, p3 = coords[:, (i - 1) * 4 + 2], coords[:, idx], coords[:, idx + 1]
                bl, ba, di = np.full(b, BOND_LENGTH_CA_C), np.full(b, ANGLE_N_CA_C), phi[:, i]  # type: ignore[assignment]
                coords[:, idx + 2] = position_atoms_batch(p1, p2, p3, bl, ba, di)  # type: ignore[assignment]

                # Place O(i) using N(i), CA(i), C(i)
                p1, p2, p3 = coords[:, idx], coords[:, idx + 1], coords[:, idx + 2]
                bl, ba, di = (
                    np.full(b, BOND_LENGTH_C_O),
                    np.full(b, ANGLE_CA_C_O),
                    np.full(b, 180.0),
                )
                coords[:, idx + 3] = position_atoms_batch(p1, p2, p3, bl, ba, di)

        # 3. Full-Atom Superimposition
        if self.full_atom:
            # Allocate full-atom coords
            fa_coords = np.zeros((b, self.total_atoms, 3))

            for i in range(length):
                # Target backbone frame (b, 3, 3) from the NeRF backbone
                # NeRF backbone order: N(0), CA(1), C(2), O(3)
                target_n = coords[:, i * 4]
                target_ca = coords[:, i * 4 + 1]
                target_c = coords[:, i * 4 + 2]
                target_bb = np.stack([target_n, target_ca, target_c], axis=1)  # (b, 3, 3)

                # Source backbone frame (3, 3)
                template_bb = self.template_backbones[i]
                # Broadcast template to batch: (b, 3, 3)
                source_bb = np.repeat(template_bb[np.newaxis, :, :], b, axis=0)

                # Align template to target
                trans, rot = superimpose_batch(source_bb, target_bb)

                # Apply rotation and translation to all atoms in template
                # template.coord: (N_res_atoms, 3)
                template_coords = self.templates[i].coord

                # (b, 3, 3) @ (N_res_atoms, 3)^T -> (b, 3, N_res_atoms)
                # Then transpose back to (b, N_res_atoms, 3)
                rotated = np.matmul(rot, template_coords.T).transpose(0, 2, 1)
                aligned = rotated + trans[:, np.newaxis, :]

                # EDUCATIONAL NOTE - D-amino acid chiral mirroring:
                # -------------------------------------------------
                # To convert an L-template to its D-form, we reflect all sidechain
                # atoms across the plane defined by the backbone (N, CA, C).
                # This must be done for EACH member of the batch independently.
                full_res_name = self.sequence[i]
                is_d = full_res_name.startswith("D-")
                if is_d and "GLY" not in full_res_name:
                    # Target backbone plane for this residue in each batch member
                    # target_bb: (b, 3, 3) -> N, CA, C
                    ca_coords_batch = target_bb[:, 1, :]
                    c_coords_batch = target_bb[:, 2, :]
                    n_coords_batch = target_bb[:, 0, :]

                    v1 = c_coords_batch - ca_coords_batch
                    v2 = n_coords_batch - ca_coords_batch
                    normal = np.cross(v1, v2, axis=-1)
                    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
                    norm = np.where(norm == 0, 1.0, norm)
                    normal /= norm  # (b, 3)

                    backbone_names = {"N", "CA", "C", "O", "H", "HA"}
                    res_atom_names = list(self.templates[i].atom_name)

                    for atom_idx, name in enumerate(res_atom_names):
                        if name not in backbone_names:
                            # p: (b, 3) coordinates for this atom in the batch
                            p = aligned[:, atom_idx, :]
                            w = p - ca_coords_batch
                            # Dot product batch-wise: (b, 3) * (b, 3) sum along -1
                            dist_to_plane = np.sum(w * normal, axis=-1, keepdims=True)
                            aligned[:, atom_idx, :] = p - 2 * dist_to_plane * normal

                # Write to global tensor
                offset = self.offsets[i]
                n_res_atoms = template_coords.shape[0]
                fa_coords[:, offset : offset + n_res_atoms] = aligned

                # EDUCATIONAL NOTE - Rigorous Backbone Overwrite:
                # To ensure perfect alignment with the serial generator (generator.py),
                # we explicitly overwrite the Oxygen (O) from the template with the
                # idealized NeRF coordinate.
                target_o = coords[:, i * 4 + 3]
                res_atom_names = self.atom_names[offset : offset + n_res_atoms]
                if "O" in res_atom_names:
                    o_local_idx = res_atom_names.index("O")
                    fa_coords[:, offset + o_local_idx] = target_o

            return BatchedPeptide(fa_coords, self.sequence, self.atom_names, self.residue_indices)

        return BatchedPeptide(
            coords, self.sequence, ["N", "CA", "C", "O"] * length, self.residue_indices
        )
