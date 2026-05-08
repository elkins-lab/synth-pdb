"""synth_pdb.quality.gnn.gnn_classifier.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GNN-based protein structure quality classifier with global and per-residue outputs.

─────────────────────────────────────────────────────────────────────────────
DESIGN CONTRACT — Same API as ProteinQualityClassifier (RF)
─────────────────────────────────────────────────────────────────────────────

Both classifiers expose:
    predict(pdb_str) → (is_good: bool, probability: float, features: dict)

This lets downstream code swap between the RF and GNN model without changes::

    from synth_pdb.quality.classifier    import ProteinQualityClassifier   # RF
    from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier  # GNN

    clf = GNNQualityClassifier()       # or ProteinQualityClassifier()
    is_good, prob, feats = clf.predict(pdb_string)

Additionally, GNNQualityClassifier exposes a richer API::

    result = clf.score(pdb_string)
    # result.global_score       → float ∈ [0, 1]  (P(Good))
    # result.per_residue        → list[float]      (pLDDT per residue)
    # result.residue_labels     → list[str]        ("Very High"/"High"/"Uncertain"/"Low")
    # result.label              → str              ("High Quality" / "Low Quality")

─────────────────────────────────────────────────────────────────────────────
CHECKPOINT FORMAT (.pt)
─────────────────────────────────────────────────────────────────────────────

GNN weights are saved with torch.save() as a dict::

    {
      "state_dict"   : OrderedDict of parameter tensors,
      "node_features": int,    ← architecture metadata
      "edge_features": int,
      "hidden_dim"   : int,
      "num_classes"  : int,
    }

We store architecture metadata alongside weights so the model can be
re-instantiated without any external configuration file.  This is the
standard pattern for "self-describing" PyTorch checkpoints.

─────────────────────────────────────────────────────────────────────────────
HIGH-THROUGHPUT AUDITING — Vectorized Ensemble Scoring
─────────────────────────────────────────────────────────────────────────────

For large-scale structural genomics or generative AI tasks, scoring
structures individually is prohibitively slow. Each structure requires
a Python function call, PDB serialization, and a GPU kernel launch.

The `GNNQualityClassifier` solves this via the `score_batch` method.
By accepting a `BatchedPeptide` object, the classifier can process
thousands of structures in a single massive GPU operation. This
"vectorized auditing" allows synth-pdb to act as a real-time quality
filter for high-diversity secondary structure ensembles.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default checkpoint path — v2 has per-residue pLDDT head; falls back to v1
# These paths are resolved relative to the installed package directory.
_DEFAULT_CHECKPOINT_V2 = os.path.join(
    os.path.dirname(__file__), "..", "models", "gnn_quality_v2.pt"
)
_DEFAULT_CHECKPOINT_V1 = os.path.join(
    os.path.dirname(__file__), "..", "models", "gnn_quality_v1.pt"
)

# Feature names matching graph.py's node feature ordering.
# Used to build the feature dict returned by predict() — useful for debugging
# and for producing human-readable explanations of a prediction.
_FEATURE_NAMES = [
    "sin_phi",  # backbone dihedral φ — sine component
    "cos_phi",  # backbone dihedral φ — cosine component
    "sin_psi",  # backbone dihedral ψ — sine component
    "cos_psi",  # backbone dihedral ψ — cosine component
    "b_factor_norm",  # normalised crystallographic temperature factor
    "seq_position",  # normalised sequence position (0=N-term, 1=C-term)
    "is_n_terminus",  # 1 if this is the N-terminal residue, else 0
    "is_c_terminus",  # 1 if this is the C-terminal residue, else 0
]

# pLDDT colour-band thresholds, matching AlphaFold's colour scheme.
# This makes synth-pdb outputs directly comparable to AlphaFold confidence plots.
_PLDDT_BANDS = [
    (0.90, "Very High"),  # blue in AlphaFold
    (0.70, "High"),  # cyan
    (0.50, "Uncertain"),  # yellow
    (0.00, "Low"),  # orange
]


def _plddt_label(score: float) -> str:
    """Convert a pLDDT score ∈ [0, 1] to a human-readable confidence label.

    Confidence bands align with industry standards (MolProbity, CASP).
    """
    for threshold, label in _PLDDT_BANDS:
        if score >= threshold:
            return label
    return "Low"


@dataclass
class QualityScore:
    """Rich quality assessment result for a single protein structure.

    This object serves as the **Data Transfer Object (DTO)** between the
    internal GNN inference engine and the end-user. It encapsulates not just
    a binary "Good/Bad" label, but a high-resolution map of the structure's
    physical confidence.

    ── pLDDT: The Standard of Confidence ─────────────────────────────────────
    The `per_residue` scores are modeled after the **predicted Local Distance
    Difference Test (pLDDT)**, the primary confidence metric used by AlphaFold.
    Values ∈ [0, 1] represent the model's certainty that a residue is in its
    physically correct local environment.

    Attributes
    ----------
    global_score : float
        The "Whole-Protein" probability P(Good) ∈ [0, 1]. This is the output
         of the GNN's global pooling layer followed by a log-softmax.
         • 0.9 - 1.0: Extremely confident, well-folded model.
         • 0.5 - 0.9: Likely valid but may have minor local strains.
         • < 0.5    : "Low Quality" — likely contains unphysical geometry.
    label : str
        A human-readable categorical label ("High Quality" or "Low Quality")
        derived from the 0.5 global_score threshold.
    per_residue : list[float]
        The "Confidence Heatmap". Each float represents the pLDDT of an
        individual residue. Length equals the number of residues (Cα atoms).
        This is generated by the auxiliary regression head in v2 models.
    residue_labels : list[str]
        The AlphaFold-standardized categorical bands:
        • "Very High" (≥ 0.90) : Crystallographic-quality geometry.
        • "High"      (≥ 0.70) : Generally reliable backbone.
        • "Uncertain" (≥ 0.50) : Low-confidence loop or linker.
        • "Low"       (< 0.50) : Unphysical/Clashing region.
    features : dict[str, float]
        A dictionary of the mean input node features (sin_phi, cos_phi, etc.).
        This is provided for **Explainable AI (XAI)** — it helps researchers
        understand if a low score was triggered by bad dihedrals or high B-factors.
    n_residues : int
        The total number of nodes (amino acids) in the interaction graph.

    Examples
    --------
    >>> clf = GNNQualityClassifier()
    >>> result = clf.score(pdb_string)
    >>> print(f"Protein quality: {result.label} ({result.global_score:.1%})")
    >>> # Identify local errors
    >>> clashes = [i for i, s in enumerate(result.per_residue) if s < 0.5]
    >>> print(f"Detected {len(clashes)} problematic residues.")
    """

    global_score: float
    label: str
    per_residue: list[float] = field(default_factory=list)
    residue_labels: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    n_residues: int = 0


class GNNQualityClassifier:
    """GNN-based protein structure quality classifier.

    Predicts whether a PDB structure is "High Quality" (biophysically plausible,
    good Ramachandran geometry, no steric clashes) or "Low Quality".

    This classifier implements the Graph Neural Network (GNN) philosophy:
    instead of relying on manual features like 'clash counts', it learns to
    recognise local and global patterns of structural discordance directly
    from the residue interaction graph.

    ── When is a GNN better than a Random Forest? ─────────────────────────
    The RF classifier uses hand-crafted, per-structure summary statistics
    (e.g. "fraction of residues in favoured Ramachandran regions").

    The GNN works directly on the full residue interaction graph, so it can:
      • Learn WHICH specific contacts are problematic, not just aggregate counts
      • Capture spatial patterns (e.g. a single clashing i/i+4 contact pair)
      • Generalise to protein classes or contact patterns not seen in training
        (because the pattern recogniser is learned, not hand-engineered)

    ── Performance Trade-offs ─────────────────────────────────────────────
    • RF:  Extremely fast (~0.1 ms/sample), zero dependencies, best for simple screening.
    • GNN: Richer signal (~0.3 ms/sample), requires torch, best for deep quality assessment.
    ───────────────────────────────────────────────────────────────────────
    """

    def __init__(self, model_path: str | None = None):
        """Initialise the GNN quality classifier.

        The classifier follows a **'Lazy Loading'** pattern where the heavy GNN
        architecture and weights are only loaded when explicitly requested or
        during first inference. This prevents synth-pdb from crashing on
        systems where `torch` is not installed, as long as the user doesn't
        try to use the GNN features.

        ── Model Versioning ───────────────────────────────────────────────────
        • **v2 Models**: Include an auxiliary regression head for pLDDT.
        • **v1 Models**: Legacy global-only classification.
        ───────────────────────────────────────────────────────────────────────

        Args:
            model_path: Path to a .pt checkpoint written by GNNQualityClassifier.save().
                        If None, looks for the default bundled checkpoint (v2 first, v1 fallback).
                        If no checkpoint is found, initialises a random-weight model
                        (useful for testing graph construction without training).

        """
        # Internal model storage (Lazy loaded via self.load() or _init_fresh_model())
        self.model: Any | None = None

        # Keep track of where we loaded the weights from for provenance and auditing.
        self._model_path: str | None = None

        # Track if the model supports per-residue confidence (v2) or just global (v1).
        # v2 models have an auxiliary regression head that predicts local confidence.
        self._has_residue_head: bool = False

        # Track if weights were successfully loaded from a checkpoint.
        # This is used by unit tests to determine if accuracy assertions
        # are valid or if the model is in a random state.
        self._is_pretrained: bool = False

        if model_path:
            # User provided an explicit path — load it or die.
            # This allows researchers to use their own trained checkpoints (e.g. robust_final.pt).
            self.load(model_path)
        else:
            # Search for bundled pre-trained weights in the installation folder.
            # v2 (per-residue head) is the modern standard for synth-pdb.
            v2 = os.path.normpath(_DEFAULT_CHECKPOINT_V2)
            v1 = os.path.normpath(_DEFAULT_CHECKPOINT_V1)

            if os.path.exists(v2):
                # Standard case: load the best available model
                self.load(v2)
            elif os.path.exists(v1):
                # Fallback to legacy v1 model if v2 is missing (e.g. partial install)
                logger.info(
                    "v2 checkpoint not found, loading v1 (no per-residue pLDDT). "
                    "Run scripts/train_gnn_quality_filter.py to produce v2."
                )
                self.load(v1)
            else:
                # No weights found in models/ — this usually happens during
                # fresh development or minimal installs without weight assets.
                logger.info(
                    "No pre-trained GNN checkpoint found. "
                    "Classifier initialised with random weights. "
                    "Run scripts/train_gnn_quality_filter.py to train."
                )
                self._init_fresh_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, pdb_content: str) -> tuple[bool, float, dict[str, float]]:
        """Predict the quality of a PDB structure (Legacy Protocol).

        This method satisfies the `ProteinQualityClassifier` protocol,
        allowing the GNN to act as a **drop-in replacement** for the
        Random Forest (RF) classifier used in the core synth-pdb `Validator`.

        ── The API Bridge ───────────────────────────────────────────────
        By maintaining this exact signature, we enable "Polymorphic Quality":
        a generation script can accept ANY classifier instance and use it
        without knowing if it's a fast RF model or a deep GNN.

        ── Internal Pipeline ────────────────────────────────────────────
        1. **PDB Parsing**: Extract 3D Cα coordinates from ATOM records.
        2. **Graph Construction**: Build the spatial interaction graph (graph.py).
        3. **Message Passing**: Propagate geometric data across the graph edges.
        4. **Global Pooling**: Average node embeddings into a whole-protein vector.
        5. **Softmax Output**: Produce final probability P(Good).
        ─────────────────────────────────────────────────────────────────

        Args:
            pdb_content: Raw PDB string containing the structural model.

        Returns:
            is_good (bool): True if the structure passes the 0.5 quality cliff.
            probability (float): The linear confidence score [0, 1].
            features (dict): Summary statistics of the input geometric features.

        Raises:
            ImportError: If the 'gnn' optional dependencies are missing.
            ValueError: If the PDB contains fewer than 2 valid residues.

        """
        # score() handles the heavy lifting; predict() wraps it for the legacy API.
        # This ensures that both the categorical and rich assessment APIs
        # use the exact same weights and graph-building logic.
        result = self.score(pdb_content)
        # We unpack the rich result object into the (bool, float, dict) format
        # expected by the standard synth-pdb Validator interface.
        return (result.label == "High Quality"), result.global_score, result.features

    def score(self, pdb_content: str) -> QualityScore:
        """Score a PDB structure, returning a rich :class:`QualityScore` object.

        This is the **modern, rich API** for the quality system. While
        `predict()` returns a simple bool, `score()` returns the full
        pLDDT confidence map, allowing for fine-grained structural auditing.

        ── The Power of the v2 Model ──────────────────────────────────────────
        When a v2 checkpoint is loaded, this method activates the auxiliary
        **Per-Residue Head**. This head doesn't just judge the whole protein;
        it performs "structural surgery" to identify exactly which loop
        is strained or which residue is clashing.

        Args:
            pdb_content: PDB-format string representing the protein.

        Returns:
            A :class:`QualityScore` instance with global and residue-level metrics.

        Raises:
            ImportError: If PyTorch or PyG are not installed.
            ValueError: If the PDB has insufficient atoms to form a graph.

        """
        # PyTorch and PyG are heavy dependencies (~500MB). We only import
        # them inside this method to allow synth-pdb to maintain its
        # "Lean Core" philosophy. Users who only want PDB generation
        # don't need to install the deep learning stack.
        try:
            import torch
        except ImportError as exc:
            # Provide an actionable error message for the optional [gnn] extra.
            raise ImportError(
                "torch is required for GNNQualityClassifier. "
                "Install with: pip install synth-pdb[gnn]"
            ) from exc

        # Lazy runtime imports for PyTorch Geometric (PyG) utilities.
        # Batch is the container used to pack graphs into high-performance tensors.
        from torch_geometric.data import Batch
        from .graph import build_protein_graph

        # Step 1: Spatial Graph Reconstruction
        # ─────────────────────────────────────────────────────────────────────
        # Convert the raw PDB string into a Topological Interaction Graph.
        # Nodes = Residues, Edges = Contacts < 10 Å.
        graph = build_protein_graph(pdb_content)

        # Step 2: Batch Packaging
        # ─────────────────────────────────────────────────────────────────────
        # Even for one structure, we wrap the graph in a Batch object.
        # This is the standard input format for Graph Attention (GAT) layers,
        # as it includes the critical 'batch' mapping vector [Nodes, 1].
        batch = Batch.from_data_list([graph])

        # Step 3: Global Model Audit
        # ─────────────────────────────────────────────────────────────────────
        # Verify the weights were successfully loaded from the checkpoint.
        assert self.model is not None, "Model not loaded"

        # Evaluation Mode: Disables stochastic layers like Dropout and
        # BatchNormalization. This is essential for reproducible results.
        self.model.eval()

        # Inference Optimization: Disable the autograd engine (grad tracking).
        # This increases speed by 2x and saves memory by not building the
        # back-propagation graph.
        with torch.no_grad():
            if self._has_residue_head:
                # DUAL-OUTPUT MODE (v2 Architecture)
                # We retrieve both the pooled global log-probs and the
                # un-pooled node confidence map (pLDDT).
                log_probs, per_res_tensor = type(self.model).forward_with_node_embeddings(
                    self.model,
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                # Move tensor back to standard Python floats for the user.
                # Squeeze removes the redundant unit dimension: [Nodes, 1] -> [Nodes].
                per_residue = per_res_tensor.squeeze(-1).tolist()
            else:
                # LEGACY MODE (v1 Architecture)
                # Only the global whole-protein classification is available.
                # per_residue is returned as an empty list.
                log_probs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                per_residue = []

        # Step 4: Final Probability Calculation
        # ─────────────────────────────────────────────────────────────────────
        # The raw output is in "Log-Softmax" space [-inf, 0].
        # We apply the exponent to get linear probabilities [0, 1].
        # Index 1 corresponds to the 'Good' (Physically Valid) class.
        prob_good = float(log_probs.exp()[0, 1].item())

        # Categorize the prediction using the 0.5 decision boundary.
        label = "High Quality" if prob_good > 0.5 else "Low Quality"

        # Map the numeric pLDDT confidence floats to the AlphaFold color bands.
        # This provides immediate structural insight (e.g. blue = good, orange = bad).
        residue_labels = [_plddt_label(s) for s in per_residue]

        # Step 5: Feature Introspection (Explainability)
        # ─────────────────────────────────────────────────────────────────────
        # We extract the mean of each input node feature (e.g., mean phi angle).
        # This dictionary is returned so researchers can verify IF the GNN is
        # paying attention to the correct physical signals.
        node_feats = graph.x.numpy()
        feat_dict = {
            name: float(np.mean(node_feats[:, i])) for i, name in enumerate(_FEATURE_NAMES)
        }

        # Step 6: Package Result
        # ─────────────────────────────────────────────────────────────────────
        return QualityScore(
            global_score=prob_good,
            label=label,
            per_residue=per_residue,
            residue_labels=residue_labels,
            features=feat_dict,
            n_residues=graph.num_nodes,
        )

    def score_batch(self, batch: Any) -> list[QualityScore]:
        r"""Score an entire batch of structures in a single vectorized pass.

        ─────────────────────────────────────────────────────────────────────────────
        VECTORIZED ENSEMBLE AUDITING
        ─────────────────────────────────────────────────────────────────────────────
        While the standard ``score()`` method processes a single PDB string,
        this method is optimized for the **high-throughput generation pipeline**.
        It operates directly on the coordinate tensors of a ``BatchedPeptide``,
        eliminating the CPU-bound bottlenecks of string serialization.

        ── GPU Parallelism ──────────────────────────────────────────────────────────
        Instead of iterating structure-by-structure, we leverage PyTorch
        Geometric's **Graph Batching** mechanism. All $B$ structures are
        packed into a single disjoint Interaction Graph:
          1. $N_{total} = \sum N_i$ nodes are combined into a single matrix.
          2. Edge indices are shifted by node offsets to maintain connectivity.
          3. A single forward pass on the GPU evaluates the entire ensemble.

        This architecture is critical for screening large-scale "Bio-Active"
        libraries where thousands of candidates must be validated in seconds.

        Args:
            batch: A :class:`synth_pdb.batch_generator.BatchedPeptide` object
                containing $B$ protein structures.

        Returns:
            A list of $B$ :class:`QualityScore` objects, containing global and
            per-residue confidence for every member of the ensemble.
        """
        try:
            import torch
            from torch_geometric.data import Batch
        except ImportError:
            # Silent fallback — scoring will be skipped in downstream pipelines
            # if the optional 'gnn' dependencies are missing.
            return []

        from .graph import build_protein_graphs_from_batch

        # 1. Vectorized Graph Construction
        # ─────────────────────────────────────────────────────────────────────
        # We bypass PDB parsing and build graphs directly from the
        # (Batch, Atoms, 3) coordinate tensor using optimized NumPy kernels.
        graphs = build_protein_graphs_from_batch(batch)
        if not graphs:
            # No structures to score (empty batch)
            return []

        # 2. PyG Packing
        # ─────────────────────────────────────────────────────────────────────
        # Convert the list of individual graphs into a single large
        # 'Batch' object for parallel GPU processing.
        pyg_batch = Batch.from_data_list(graphs)

        # 3. Model & Device Synchronization
        # ─────────────────────────────────────────────────────────────────────
        # Ensure the model is loaded and in evaluation mode (disables dropout).
        assert self.model is not None, "Model not loaded"
        self.model.eval()

        # Identify the active device (CPU/CUDA/MPS) and move the entire packed
        # batch tensor to that device in one transfer.
        device = next(self.model.parameters()).device
        pyg_batch = pyg_batch.to(device)

        # 4. Forward Pass (Inference)
        # ─────────────────────────────────────────────────────────────────────
        # Gradient tracking is disabled to save VRAM and increase speed.
        with torch.no_grad():
            if self._has_residue_head:
                # v2 models: Retrieve both global log-probs and local pLDDT.
                # per_res_tensor shape: [TotalNodes, 1] — contains confidence
                # for every atom across all B structures.
                log_probs, per_res_tensor = type(self.model).forward_with_node_embeddings(
                    self.model,
                    pyg_batch.x,
                    pyg_batch.edge_index,
                    pyg_batch.edge_attr,
                    pyg_batch.batch,
                )
                per_residue_all = per_res_tensor.squeeze(-1).cpu().numpy()
            else:
                # v1 models: Legacy global-only scoring head.
                log_probs = self.model(
                    pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr, pyg_batch.batch
                )
                per_residue_all = None

        # 5. Result Post-Processing & Slicing
        # ─────────────────────────────────────────────────────────────────────
        # Convert log-probabilities back to linear probability space [0, 1].
        # probs_good[i] represents P(High Quality) for structure i.
        probs_good = log_probs.exp()[:, 1].cpu().numpy()

        results = []
        # ptr (pointer) array defines the residue boundaries for each graph
        # in the packed batch tensor. node_ptr[i] to node_ptr[i+1] is the slice.
        node_ptr = pyg_batch.ptr.cpu().numpy()

        for i in range(len(graphs)):
            prob_good = float(probs_good[i])
            label = "High Quality" if prob_good > 0.5 else "Low Quality"

            # Slice the per-residue pLDDT scores using the node pointers
            plddt = []
            res_lbls = []
            if per_residue_all is not None:
                # Extract the confidence values for this specific structure
                plddt = per_residue_all[node_ptr[i] : node_ptr[i + 1]].tolist()
                # Map to human-readable AlphaFold confidence bands
                res_lbls = [_plddt_label(s) for s in plddt]

            # Extract aggregate feature statistics (e.g. mean dihedrals)
            # for downstream transparency and debugging.
            node_feats = graphs[i].x.numpy()
            feat_dict = {
                name: float(np.mean(node_feats[:, j])) for j, name in enumerate(_FEATURE_NAMES)
            }

            # Wrap in structured QualityScore container
            results.append(
                QualityScore(
                    global_score=prob_good,
                    label=label,
                    per_residue=plddt,
                    residue_labels=res_lbls,
                    features=feat_dict,
                    n_residues=graphs[i].num_nodes,
                )
            )

        return results

    def save(self, path: str) -> None:
        """Save model weights and architecture config to a ``.pt`` checkpoint.

        The checkpoint is 'self-describing' — it includes the layer dimensions
        required to re-instantiate the `ProteinGNN` class without an external
        config or JSON file.  This follows the **'Single File Per Asset'**
        philosophy which simplifies deployment, model versioning, and
        cloud-based distribution of pre-trained structural auditors.

        ── Persistence Mechanism ──────────────────────────────────────────────
        We use standard `torch.save()`, which uses Python's `pickle` module
        internally. The payload includes:
          1. The `state_dict`: An OrderedDict mapping layer names to parameter
             tensors (weights and biases).
          2. Architectural Hyperparameters: `hidden_dim`, `node_features`, etc.
        ───────────────────────────────────────────────────────────────────────

        Args:
            path: Destination file path (should end in .pt).

        """
        # Ensure we have torch before trying to save.
        # This is a safety check for systems where torch is an optional extra.
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to save a GNN checkpoint.") from exc

        # Ensure destination directory exists before writing to prevent IOErrors.
        # We use abspath to handle relative paths provided by the user in the CLI.
        # This prevents crashes when users specify nested output directories.
        target_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(target_dir, exist_ok=True)

        # Verification check: we cannot save a non-existent or uninitialized model.
        # The internal self.model must be an instance of ProteinGNN.
        assert self.model is not None, "Model not loaded; nothing to save"

        # Compile the state dictionary (weights) and architecture metadata.
        # This ensures the model can be reconstructed on any machine
        # without needing to guess the hidden_dim or feature counts used.
        # We store the core hyper-parameters (node_features, hidden_dim, etc.)
        # directly in the checkpoint to ensure perfect reproducibility.
        payload = {
            "state_dict": self.model.state_dict(),
            "node_features": self.model.node_features,
            "edge_features": self.model.edge_features,
            "hidden_dim": self.model.hidden_dim,
            "num_classes": self.model.num_classes,
        }

        # Standard PyTorch serialization (uses pickle/zip under the hood).
        # We use standard torch.save() to remain compatible with standard tools
        # like Netron for model visualization.
        torch.save(payload, path)
        self._model_path = path
        logger.info("GNN checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights and re-configure architecture from a checkpoint.

        This method acts as a **Dynamic Factory**: it reads the metadata
        inside the checkpoint to determine the model's width and depth,
        builds the graph attention layers, and then injects the learned weights.
        This allows one classifier instance to seamlessly switch between
        different model variants (e.g. v1 vs v2) or generations.

        ── Portability (CPU/GPU) ──────────────────────────────────────────────
        Models trained on a GPU (CUDA) often contain tensors tied to that
        hardware. We use `map_location='cpu'` during the load phase to
        ensure the model can be deployed on standard workstations without
        specialized hardware, then move it to the active device later.
        ───────────────────────────────────────────────────────────────────────

        Args:
            path: Path to a .pt checkpoint written by GNNQualityClassifier.save().

        """
        # Runtime dependency check for torch.
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load a GNN checkpoint.") from exc

        # Lazy import of the GNN model class to minimize startup time.
        # ProteinGNN is the core Message Passing Neural Network architecture.
        from .model import ProteinGNN

        try:
            # Load into CPU memory by default for maximum compatibility across systems.
            # weights_only=False is used because our checkpoint is a custom dict
            # rather than a raw tensor stream.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            import typing

            # Instantiate model using dimensions stored inside the checkpoint file.
            # This dynamic reconstruction is key for 'self-describing' assets.
            # node_features and hidden_dim MUST match what was used during training.
            # If they mismatch, layer shapes will be incompatible with the state_dict.
            self.model = typing.cast(
                Any,
                ProteinGNN(
                    node_features=checkpoint["node_features"],
                    edge_features=checkpoint["edge_features"],
                    hidden_dim=checkpoint["hidden_dim"],
                    num_classes=checkpoint["num_classes"],
                ),
            )

            # Inject parameter tensors (weights/biases) into the instantiated layers.
            # This uses the standard state_dict mechanism for weight restoration.
            # This step performs the heavy transfer of floating-point weights.
            self.model.load_state_dict(checkpoint["state_dict"])

            # Switch to evaluation mode (essential: fixes dropout and batchnorm behavior).
            # Without this, the model might produce stochastic or incorrect results
            # due to active Dropout layers.
            self.model.eval()
            self._model_path = path
            self._is_pretrained = True

            # Detect if this checkpoint has the per-residue pLDDT head (v2).
            # We look for the existence of the specific linear layer parameters
            # that only exist in the expanded v2 architecture (residue_lin1).
            # This allows the classifier to gracefully handle both v1 and v2 files.
            self._has_residue_head = "residue_lin1.weight" in checkpoint["state_dict"]

            logger.info(
                "GNN classifier loaded from %s (per-residue head: %s)",
                path,
                self._has_residue_head,
            )
        except Exception as exc:
            # Critical error: model weights are essential for GNN inference.
            # We log the full stack trace to help researchers debug corrupted
            # assets or version mismatches in the architecture/state_dict.
            logger.error("Failed to load GNN checkpoint from %s: %s", path, exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_fresh_model(self) -> None:
        """Initialise a randomly-weighted model (v2 architecture).

        Used primarily for integration tests or when bootstrapping a new
        training run from scratch.  Uses standard synth-pdb dimensions.
        """
        import typing

        from .model import ProteinGNN

        # Default architecture for fresh training:
        # - 8 node features (dihedrals, sequence info, b-factors)
        # - 2 edge features (euclidean distance, distance bins)
        self.model = typing.cast(
            Any, ProteinGNN(node_features=8, edge_features=2, hidden_dim=64, num_classes=2)
        )

        # Start in eval mode for safety
        self.model.eval()

        # New models always include the residue head (standard for v2+)
        self._has_residue_head = True
        self._is_pretrained = False

    @property
    def is_pretrained(self) -> bool:
        """Return True if the model weights were loaded from a checkpoint."""
        return self._is_pretrained
