"""synth_pdb.quality.gnn.graph.
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Converts a PDB string into a PyTorch Geometric `Data` graph for the GNN quality scorer.

─────────────────────────────────────────────────────────────────────────────
EDUCATIONAL BACKGROUND — Why represent a protein as a graph?
─────────────────────────────────────────────────────────────────────────────

A conventional neural network expects a fixed-size vector (e.g. scikit-learn's
Random Forest receives [n_samples, n_features]).  Proteins are NOT fixed-size: a
20-residue peptide and a 500-residue enzyme need to be handled by the same model.

Graph Neural Networks (GNNs) solve this elegantly:

    Protein → Graph G = (V, E)
    V = {residues}                 ← nodes
    E = {spatial contacts}         ← edges

The model then operates over the graph topology, regardless of size.  This is the
same abstraction used by AlphaFold 2's "triangle attention", GVP-GNN, and most
modern structure models.

─────────────────────────────────────────────────────────────────────────────
GEOMETRIC VECTORIZATION — Why skip PDB parsing?
─────────────────────────────────────────────────────────────────────────────

In high-throughput structural biology (e.g., screening 100,000 ligands or
generating large-scale training sets), the PDB file format is a bottleneck.
PDB is a text-based, fixed-width format designed for human readability in
the 1970s. Converting internal coordinate tensors to PDB strings only to
re-parse them via Regex or Biotite introduces massive CPU latency.

By implementing `build_protein_graphs_from_batch`, we remain entirely in
the "Vectorized Domain" (NumPy/PyTorch), allowing the generation pipeline
to feed the GNN quality auditor at the speed of RAM rather than the speed
of Disk I/O.

─────────────────────────────────────────────────────────────────────────────
TOPOLOGY VS CARTESIAN — Why Graphs Win
─────────────────────────────────────────────────────────────────────────────

Standard 3D CNNs or MLPs treat protein coordinates as raw (x, y, z) vectors.
This is problematic because proteins are invariant to rotation and
translation. If you rotate a protein by 90°, the (x, y, z) values change
completely, but the *biology* remains identical.

Graphs solve this by encoding the structure as a network of **relative
spatial relationships**. By using pairwise distances and internal dihedral
angles, the GNN becomes **SE(3)-Invariant** by design. It sees the
structure exactly as a physics engine does: as a web of bonded and
non-bonded interactions.

─────────────────────────────────────────────────────────────────────────────
GRAPH STRUCTURE
─────────────────────────────────────────────────────────────────────────────

  Nodes  — one per residue, identified by Cα position.
  Edges  — bidirectional between every Cα pair within 8 Å of each other.
           8 Å is a biologically meaningful cutoff: it captures direct
           contacts (side-chain interactions, H-bonds) without connecting
           residues that are too distant to interact.

  Node features  (8-dimensional):
    [0] sin(φ)          Backbone torsion — periodic encoding avoids the
    [1] cos(φ)          ±180° discontinuity. Alpha helices cluster at
    [2] sin(ψ)          φ ≈ -60°, ψ ≈ -45°. Beta strands at φ ≈ -120°,
    [3] cos(ψ)          ψ ≈ +120°. The GNN learns these clusters without
                        us hard-coding them.
    [4] B-factor        Crystallographic temperature factor (flexibility).
                        High B-factor → atomic displacement → possible
                        disorder. Normalised to [0, 1].
    [5] seq_position    Normalised index (0 = N-term, 1 = C-term). Lets
                        the model distinguish terminal from buried residues.
    [6] is_N_terminus   One-hot flag — termini have different chemistry
    [7] is_C_terminus   (free NH3+ / COO-) and often higher B-factors.

  Edge features  (2-dimensional):
    [0] Cα–Cα distance  Physical distance in Å.  Nearby residues interact
                        more strongly — the GNN can learn to weight edges
                        by proximity.
    [1] seq_separation  |i − j|.  Distinguishes local contacts (i, i+3 in
                        a helix) from long-range contacts (cross-strand).
"""

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def build_protein_graph(pdb_content: str, ca_distance_threshold: float = 8.0) -> Any:
    """Parse *pdb_content* and return a :class:`torch_geometric.data.Data` object
    representing the protein as a residue-level contact graph.

    ── How this fits into the GNN pipeline ─────────────────────────────────
    The pipeline is:

        PDB string
            │
            ▼
        build_protein_graph()     ← YOU ARE HERE
            │  produces a PyG Data object:
            │     data.x          — node feature matrix [N, 8]
            │     data.edge_index — edge connectivity   [2, E]
            │     data.edge_attr  — edge features       [E, 2]
            │
            ▼
        ProteinGNN.forward()      — message passing over the graph
            │
            ▼
        log-softmax scores        — [batch_size, 2]  (Bad / Good)

    ────────────────────────────────────────────────────────────────────────

    Args:
        pdb_content: PDB-format string.
        ca_distance_threshold: Maximum Cα–Cα distance (Å) for an edge to be
            created. 8 Å is a standard contact-map cutoff in structural biology.

    Returns:
        torch_geometric.data.Data with attributes:
            - ``x``          : float32 node feature matrix [N, 8]
            - ``edge_index`` : long edge index [2, E]
            - ``edge_attr``  : float32 edge features [E, 2]
            - ``num_nodes``  : int N

    Raises:
        ValueError: If fewer than 2 residues with Cα atoms are found.
        ImportError: If torch or torch_geometric are not installed.

    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "torch and torch_geometric are required for the GNN quality scorer. "
            "Install with: pip install synth-pdb[gnn]"
        ) from exc

    # ------------------------------------------------------------------
    # Step 1 — Parse PDB to extract per-residue structural data
    # ------------------------------------------------------------------
    # We read raw ATOM records rather than using a heavy library so that
    # this module has only numpy as a dependency (PyG handles the tensor
    # construction below). This "Zero-Dependency" parsing ensures that
    # synth-pdb remains lightweight and fast for high-throughput tasks.
    residues = _parse_backbone(pdb_content)

    if len(residues) < 2:
        # A single residue has no contacts and no dihedrals; it cannot
        # form a graph structure for message passing.
        raise ValueError(
            f"Found only {len(residues)} Cα atom(s) in PDB. "
            "Need at least 2 residues to build a graph."
        )

    n = len(residues)
    logger.debug("Building graph for %d residues (threshold=%.1f Å)", n, ca_distance_threshold)

    # ------------------------------------------------------------------
    # Step 2 — Build node feature matrix  x  of shape [N, 8]
    # ------------------------------------------------------------------
    # Each row describes one residue.  The features are chosen to capture:
    #   • Backbone geometry  (φ/ψ dihedrals — most discriminative signal)
    #   • Flexibility        (B-factor)
    #   • Sequence context   (position, terminus flags)
    # This feature vector is the initial "embedding" that the GNN will
    # transform via successive layers of graph attention.
    # ------------------------------------------------------------------

    # Collect Cα coordinates — needed for the pairwise distance matrix.
    # Coordinates are stored in float32 for compatibility with PyTorch tensors.
    ca_coords = np.array([r["ca"] for r in residues], dtype=np.float32)  # [N, 3]

    # ── B-factor normalisation ────────────────────────────────────────
    # Raw B-factors vary across structures (e.g. 5–80 Å²).  We normalise
    # to [0, 1] within each structure so the model sees relative flexibility,
    # not absolute values that depend on the refinement protocol or
    # experimental resolution.
    b_factors = np.array([r["b_factor"] for r in residues], dtype=np.float32)
    b_range = b_factors.max() - b_factors.min()
    # Handle the degenerate case of constant B-factors (common in synthetic data)
    b_norm = (b_factors - b_factors.min()) / (b_range if b_range > 1e-6 else 1.0)

    # ── Sequence position ─────────────────────────────────────────────
    # Linspace gives a clean gradient 0→1 that the GNN can use to
    # distinguish N-terminal (0.0) from C-terminal (1.0) residues.
    # This breaks the "spatial-only" symmetry of the graph.
    seq_pos = np.linspace(0.0, 1.0, n, dtype=np.float32)

    # ── Dihedral sin/cos encoding ─────────────────────────────────────
    # Why sin AND cos, not just the raw angle?
    # Because -180° and +180° are the SAME conformation, but numerically
    # very far apart.  Encoding as (sin θ, cos θ) makes the representation
    # continuous and circular — a point on the unit circle.
    # This is the same trick used in positional encodings for Transformers.
    node_feats = np.zeros((n, 8), dtype=np.float32)
    for i, res in enumerate(residues):
        phi, psi = res["phi"], res["psi"]
        # Convert degrees to radians before sin/cos transform
        node_feats[i, 0] = math.sin(math.radians(phi)) if phi is not None else 0.0
        node_feats[i, 1] = math.cos(math.radians(phi)) if phi is not None else 0.0
        node_feats[i, 2] = math.sin(math.radians(psi)) if psi is not None else 0.0
        node_feats[i, 3] = math.cos(math.radians(psi)) if psi is not None else 0.0
        node_feats[i, 4] = b_norm[i]
        node_feats[i, 5] = seq_pos[i]
        node_feats[i, 6] = 1.0 if i == 0 else 0.0  # N-terminus flag
        node_feats[i, 7] = 1.0 if i == n - 1 else 0.0  # C-terminus flag

    # ------------------------------------------------------------------
    # Step 3 — Build edge index and edge features
    # ------------------------------------------------------------------
    # The edge index is a [2, E] tensor where:
    #   edge_index[0] = source residue indices
    #   edge_index[1] = destination residue indices
    #
    # This COO (coordinate) sparse format is what PyTorch Geometric expects.
    # We create BIDIRECTIONAL edges: if residue i contacts j, we add both
    # (i→j) AND (j→i).  This ensures every node can aggregate information
    # from all its neighbours during message passing.
    # ------------------------------------------------------------------

    # Compute pairwise Cα distance matrix using broadcasting: O(N²) memory
    # which is fine for the short peptides synth-pdb generates (≤ 100 res).
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # [N, N, 3]
    # Standard Euclidean norm along the XYZ dimension
    dist_matrix = np.sqrt((diff**2).sum(axis=-1))  # [N, N]

    src_list, dst_list, edge_attr_list = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                # No self-loops: a residue does not send messages to itself
                # (The GAT architecture handles self-attention via residual
                # skip connections, not explicit self-edges).
                continue
            d = dist_matrix[i, j]
            # Biologically meaningful cutoff (8 Å) captures H-bonds and VdW.
            if d < ca_distance_threshold:
                src_list.append(i)
                dst_list.append(j)
                # Edge feature vector: [distance, sequence_separation]
                # Sequence separation helps the model distinguish helical
                # contacts (separation=3 or 4) from cross-strand contacts.
                edge_attr_list.append([d, float(abs(i - j))])

    if not src_list:
        # Safety fallback: if all Cα atoms happened to be > 8 Å apart
        # (extremely distorted structure), connect sequential neighbours so
        # the graph is never edgeless. An edgeless graph breaks GNN pooling.
        logger.warning(
            "No edges found within %.1f Å threshold — adding sequential backbone edges.",
            ca_distance_threshold,
        )
        for i in range(n - 1):
            d = float(dist_matrix[i, i + 1])
            src_list.extend([i, i + 1])
            dst_list.extend([i + 1, i])
            # Default sequence separation of 1.0 for these "bonded" edges.
            edge_attr_list.extend([[d, 1.0], [d, 1.0]])

    # ------------------------------------------------------------------
    # Step 4 — Pack into a PyG Data object
    # ------------------------------------------------------------------
    # The final Data object is passed to ProteinGNN.forward().
    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


# ---------------------------------------------------------------------------
# Internal: PDB backbone parser
# ---------------------------------------------------------------------------


def _parse_backbone(pdb_content: str) -> list[dict]:
    """Extract per-residue Cα coordinates, B-factors, and backbone dihedrals
    from a raw PDB string.

    Returns a list of dicts (one per residue, sorted by chain then residue
    number):
        {"ca": [x, y, z], "b_factor": float, "phi": float|None, "psi": float|None}

    ── PDB format reminder ──────────────────────────────────────────────
    The Protein Data Bank (PDB) format is a fixed-width specification.
    Each ATOM line is exactly 80 characters long, and fields are defined
    by character columns rather than delimiters (like commas).
      cols  1– 6  Record type ("ATOM  ")
      cols 13–16  Atom name  ("CA  " for alpha carbon)
      cols 22–26  Residue sequence number
      cols 31–38  X coordinate (Å)
      cols 39–46  Y coordinate (Å)
      cols 47–54  Z coordinate (Å)
      cols 61–66  B-factor
    ─────────────────────────────────────────────────────────────────────
    """
    # Dictionary to store raw coordinates, keyed by (chain, res_num) tuple.
    # This allows us to handle multi-chain PDBs or non-sequential residue numbers.
    atom_records: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    b_records: dict[tuple[str, int], float] = {}

    # Line-by-line parsing bypasses heavy PDB libraries for speed.
    for line in pdb_content.splitlines():
        if not line.startswith("ATOM"):
            # Skip non-ATOM records (HEADER, HELIX, SHEET, CONECT, etc.)
            continue

        # Fixed-width field extraction
        atom_name = line[12:16].strip()
        chain = line[21].strip() or "A"  # Default to chain 'A' if empty

        try:
            # Parse numeric fields from specific column offsets.
            # Character slicing is O(1) and much faster than Regex for this format.
            res_num = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            # B-factors (occupancy) follow the coordinates.
            b = float(line[60:66]) if len(line) > 60 else 0.0
        except ValueError:
            # Skip malformed lines (e.g. invalid numbers)
            continue

        key = (chain, res_num)
        if key not in atom_records:
            atom_records[key] = {}
        # Store coordinates as double-precision for accurate dihedral math
        atom_records[key][atom_name] = np.array([x, y, z], dtype=np.float64)
        if atom_name == "CA":
            # Track B-factor for the alpha carbon node specifically
            b_records[key] = b

    if not atom_records:
        return []

    # Sort residues by (chain, residue number) to ensure the 1D backbone
    # connectivity is preserved during graph construction.
    sorted_keys = sorted(atom_records.keys())

    # ------------------------------------------------------------------
    # Compute backbone dihedrals φ and ψ
    # ------------------------------------------------------------------
    # These angles describe the rotation around the N-CA and CA-C bonds.
    # φ (phi) = dihedral(C_{i-1}, N_i, Cα_i, C_i)
    # ψ (psi) = dihedral(N_i, Cα_i, C_i, N_{i+1})
    #
    # These are undefined for the first (φ) and last (ψ) residues, where
    # we leave them as None → encoded as 0 in both sin/cos channels.
    # This is a soft indicator to the model that these are terminal residues.
    residues = []
    for idx, key in enumerate(sorted_keys):
        atoms = atom_records[key]
        if "CA" not in atoms:
            # Every residue node in our GNN MUST have a C-alpha coordinate
            continue

        phi = None
        psi = None

        # Look-behind for previous Carbonyl C (required for Phi)
        if idx > 0:
            prev_key = sorted_keys[idx - 1]
            prev = atom_records.get(prev_key, {})
            if "C" in prev and "N" in atoms and "CA" in atoms and "C" in atoms:
                phi = _dihedral(prev["C"], atoms["N"], atoms["CA"], atoms["C"])

        # Look-ahead for next Nitrogen N (required for Psi)
        if idx < len(sorted_keys) - 1:
            next_key = sorted_keys[idx + 1]
            nxt = atom_records.get(next_key, {})
            if "N" in atoms and "CA" in atoms and "C" in atoms and "N" in nxt:
                psi = _dihedral(atoms["N"], atoms["CA"], atoms["C"], nxt["N"])

        residues.append(
            {
                "ca": atoms["CA"].astype(np.float32),
                "b_factor": b_records.get(key, 0.0),
                "phi": phi,
                "psi": psi,
            }
        )

    return residues


def build_protein_graphs_from_batch(batch: Any, ca_distance_threshold: float = 8.0) -> list[Any]:
    """Build a list of PyG Data objects from a BatchedPeptide container.

    ─────────────────────────────────────────────────────────────────────────────
    HIGH-PERFORMANCE VECTORIZED PIPELINE
    ─────────────────────────────────────────────────────────────────────────────
    Traditional GNN scoring involves a bottleneck:
        (Tensor) → (PDB String) → (Parse PDB) → (Build Graph)

    This function implements a "Short-Circuit" path:
        (Tensor) → (Build Graph)

    By operating directly on the coordinate tensor, we bypass the expensive
    string formatting and regular expression parsing of the PDB format,
    enabling high-throughput quality auditing of generated ensembles.

    Args:
        batch: A :class:`synth_pdb.batch_generator.BatchedPeptide` object
            containing a coordinate tensor of shape (B, N_atoms, 3).
        ca_distance_threshold: Spatial cutoff (Å) for creating edges between
            residue nodes.

    Returns:
        A list of torch_geometric.data.Data objects, ready for GNN inference.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        # Graceful fallback if torch-geometric is not installed
        return []

    # 1. Topology Mapping — Identifying Backbone Nodes
    # ─────────────────────────────────────────────────────────────────────────
    # The BatchedPeptide container holds a flat coordinate tensor and a
    # static topology (atom_names). We use a mask to extract only the
    # Alpha Carbons (CA) which serve as the nodes in our residue graph.
    # Nodes in this graph represent individual amino acids.
    ca_mask = np.array([name == "CA" for name in batch.atom_names])
    if not np.any(ca_mask):
        # Safety check: if no CA atoms are present, we cannot form a protein graph
        return []

    # ca_coords shape: (Batch, Residues, 3)
    # This tensor holds the 3D positions of all backbone nodes for all B structures.
    ca_coords = batch.coords[:, ca_mask, :]
    b, length, _ = ca_coords.shape

    # 2. Geometric Feature Extraction
    # ─────────────────────────────────────────────────────────────────────────
    # We reconstruct the internal backbone reference frame (N, CA, C) for
    # every structure in the batch to calculate φ/ψ dihedrals.

    # Identify indices in the static atom list using list comprehensions.
    # These indices are consistent across all structures in the batch.
    n_idx = [i for i, name in enumerate(batch.atom_names) if name == "N"]
    ca_idx = [i for i, name in enumerate(batch.atom_names) if name == "CA"]
    c_idx = [i for i, name in enumerate(batch.atom_names) if name == "C"]

    graphs = []

    # Loop over batch members — even though we loop here, the heavy work
    # (distance matrices and dihedral math) uses NumPy's optimized C kernels.
    for i in range(b):
        # Extract the coordinate slice for this specific protein structure [Atoms, 3]
        struc_coords = batch.coords[i]

        # Node Feature Matrix [N, 8] — holds the inputs for the first GNN layer.
        node_feats = np.zeros((length, 8), dtype=np.float32)
        # Sequence position: helps the model distinguish termini from core residues.
        seq_pos = np.linspace(0.0, 1.0, length, dtype=np.float32)

        # Iterative Dihedral Calculation
        # These dihedrals (φ and ψ) are the most important descriptors of
        # local secondary structure quality (Ramachandran status).
        # φ (phi) = C(i-1)-N(i)-CA(i)-C(i)
        # ψ (psi) = N(i)-CA(i)-C(i)-N(i+1)
        for res_idx in range(length):
            phi = None
            psi = None

            # Phi calculation — requires the previous residue's Carbonyl C
            # This is undefined for the first residue (N-terminus).
            if res_idx > 0:
                p1 = struc_coords[c_idx[res_idx - 1]]
                p2 = struc_coords[n_idx[res_idx]]
                p3 = struc_coords[ca_idx[res_idx]]
                p4 = struc_coords[c_idx[res_idx]]
                phi = _dihedral(p1, p2, p3, p4)

            # Psi calculation — requires the next residue's Nitrogen N
            # This is undefined for the last residue (C-terminus).
            if res_idx < length - 1:
                p1 = struc_coords[n_idx[res_idx]]
                p2 = struc_coords[ca_idx[res_idx]]
                p3 = struc_coords[c_idx[res_idx]]
                p4 = struc_coords[n_idx[res_idx + 1]]
                psi = _dihedral(p1, p2, p3, p4)

            # Circular Encoding (Sin/Cos)
            # Neural networks struggle with discontinuities (±180°). By encoding
            # as a 2D point on the unit circle, we provide a smooth input space.
            node_feats[res_idx, 0] = math.sin(math.radians(phi)) if phi is not None else 0.0
            node_feats[res_idx, 1] = math.cos(math.radians(phi)) if phi is not None else 0.0
            node_feats[res_idx, 2] = math.sin(math.radians(psi)) if psi is not None else 0.0
            node_feats[res_idx, 3] = math.cos(math.radians(psi)) if psi is not None else 0.0
            # B-factors are set to 0.5 (neutral) for purely synthetic structures
            # unless explicitly simulated by the batch generator. High B-factors
            # usually indicate disorder or mobile regions.
            node_feats[res_idx, 4] = 0.5
            node_feats[res_idx, 5] = seq_pos[res_idx]
            # One-hot encoding of terminus status allows the model to learn
            # specific chemistry/geometries for peptide ends.
            node_feats[res_idx, 6] = 1.0 if res_idx == 0 else 0.0
            node_feats[res_idx, 7] = 1.0 if res_idx == length - 1 else 0.0

        # 3. Spatial Adjacency (Contact Graph)
        # ─────────────────────────────────────────────────────────────────────
        # We compute the pairwise distance matrix [L, L] between all CA atoms.
        # Edges are created for every residue pair within the distance threshold.
        # This interaction network encodes the 3D tertiary fold.
        struc_ca = ca_coords[i]
        diff = struc_ca[:, None, :] - struc_ca[None, :, :]
        # Vectorized Euclidean distance calculation using NumPy broadcasting.
        dist_matrix = np.sqrt((diff**2).sum(axis=-1))

        src_list, dst_list, edge_attr_list = [], [], []
        for u in range(length):
            for v in range(length):
                if u == v:
                    continue  # No self-loops
                d = dist_matrix[u, v]
                if d < ca_distance_threshold:
                    # Edge found within threshold!
                    src_list.append(u)
                    dst_list.append(v)
                    # Edge features: [Physical Distance, Sequence Separation Distance]
                    # Sequence distance |u-v| helps distinguish local vs long-range contacts.
                    edge_attr_list.append([d, float(abs(u - v))])

        # Safety Fallback: Sequential Continuity
        # If no spatial contacts are found (e.g., highly extended chain),
        # we add edges between adjacent residues in the sequence to ensure
        # the graph is connected for message passing (otherwise no GNN power).
        if not src_list:
            for u in range(length - 1):
                d = float(dist_matrix[u, u + 1])
                src_list.extend([u, u + 1])
                dst_list.extend([u + 1, u])
                edge_attr_list.extend([[d, 1.0], [d, 1.0]])

        # 4. PyTorch Geometric Packaging
        # ─────────────────────────────────────────────────────────────────────
        # Convert NumPy arrays to Torch Tensors and pack into the standard
        # Data container used by the PyG library.
        graphs.append(
            Data(
                x=torch.tensor(node_feats, dtype=torch.float),
                edge_index=torch.tensor([src_list, dst_list], dtype=torch.long),
                edge_attr=torch.tensor(edge_attr_list, dtype=torch.float),
                num_nodes=length,
            )
        )

    return graphs


def _dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Compute the dihedral angle (degrees) defined by four 3D points.

    Algorithm — the "two normal vectors" method:
    ─────────────────────────────────────────────
    Given four points p1–p4 defining three bond vectors b1, b2, b3:

        b1 = p2 - p1
        b2 = p3 - p2  ← central bond (the rotation axis)
        b3 = p4 - p3

    The dihedral is the angle between the planes (b1,b2) and (b2,b3):
        n1 = b1 × b2   ← normal to plane 1
        n2 = b2 × b3   ← normal to plane 2
        θ  = arccos(n1 · n2)

    Sign is determined by whether n1 and b3 point in the same direction
    (the "right-hand rule" convention used in IUPAC backbone dihedrals).
    """
    # 1. Plane normals calculation via cross product
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)  # vector perpendicular to the first plane
    n2 = np.cross(b2, b3)  # vector perpendicular to the second plane

    # 2. Vector Normalisation
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        # Handle collinear points (bond angle = 0 or 180).
        # In these cases, the plane is undefined and the dihedral is 0.0.
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # 3. Angular Separation calculation
    # Clamp to [-1, 1] to guard against floating-point drift before arccos
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))

    # 4. Sign determination (IUPAC Right-Hand Convention)
    # If the vector n1 and b3 point in the same direction (dot > 0),
    # the angle is positive. Otherwise negative.
    if np.dot(n1, b3) < 0:
        angle = -angle

    return angle
