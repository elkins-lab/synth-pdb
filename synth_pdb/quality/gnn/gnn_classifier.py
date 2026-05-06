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
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default checkpoint path — v2 has per-residue pLDDT head; falls back to v1
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
    """Convert a pLDDT score ∈ [0, 1] to a human-readable confidence label."""
    for threshold, label in _PLDDT_BANDS:
        if score >= threshold:
            return label
    return "Low"


@dataclass
class QualityScore:
    """Rich quality assessment result for a single protein structure.

    Attributes
    ----------
    global_score : float
        P(Good) ∈ [0, 1].  Values near 1 indicate high confidence the
        structure is biophysically plausible.  Values near 0.5 indicate the
        model is uncertain.
    label : str
        "High Quality" if global_score > 0.5, else "Low Quality".
    per_residue : list[float]
        Per-residue pLDDT-like confidence scores ∈ [0, 1].  Length equals
        the number of residues with Cα atoms in the PDB.  Analogous to
        AlphaFold's per-residue pLDDT output.
    residue_labels : list[str]
        Human-readable confidence band for each residue:
        "Very High" (≥0.90), "High" (≥0.70), "Uncertain" (≥0.50), "Low" (<0.50).
    features : dict[str, float]
        Mean per-feature summary of the input graph's node feature matrix.
        Useful for debugging.
    n_residues : int
        Number of residues scored.

    Examples
    --------
    >>> clf = GNNQualityClassifier()
    >>> result = clf.score(pdb_string)
    >>> print(f"Global score: {result.global_score:.3f}  ({result.label})")
    >>> low_conf = [i for i, lbl in enumerate(result.residue_labels) if lbl == "Low"]
    >>> print(f"Low-confidence residues: {low_conf}")
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

        The classifier follows a 'lazy' loading pattern where the heavy GNN
        architecture and weights are only loaded when explicitly requested or
        during first inference.  This prevents synth-pdb from crashing on
        systems where torch is not installed, as long as the user doesn't
        try to use the GNN features.

        Args:
            model_path: Path to a .pt checkpoint written by GNNQualityClassifier.save().
                        If None, looks for the default bundled checkpoint (v2 first, v1 fallback).
                        If no checkpoint is found, initialises a random-weight model
                        (useful for testing graph construction without training).

        """
        # Internal model storage (Lazy loaded)
        self.model: Any | None = None

        # Keep track of where we loaded the weights from for provenance
        self._model_path: str | None = None

        # Track if the model supports per-residue confidence (v2) or just global (v1).
        # v2 models have an auxiliary regression head that predicts lDDT.
        self._has_residue_head: bool = False

        if model_path:
            # User provided an explicit path — load it or die.
            # This allows researchers to use their own trained checkpoints.
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
        """Predict the quality of a PDB structure.

        This method satisfies the `ProteinQualityClassifier` protocol,
        allowing it to be used as a drop-in replacement for the
        Random Forest classifier.

        ── Internal Pipeline ────────────────────────────────────────────
        1. Parse PDB → Extract Cα coords → Build spatial graph
        2. Featurise nodes (dihedrals, b-factors) and edges (distances)
        3. Pass through GNN layers (Message Passing + Global Pooling)
        4. Softmax → P(Good) probability
        ─────────────────────────────────────────────────────────────────

        Args:
            pdb_content: PDB-format string.

        Returns:
            is_good (bool): True if P(Good) > 0.5.
            probability (float): P(Good), in [0, 1].
            features (dict): Mean per-feature summary of the input graph's
                node feature matrix.  Useful for logging and introspection.

        Raises:
            ImportError: If torch or torch_geometric are not installed.
            ValueError: If the PDB contains too few residues to build a graph.

        """
        # score() handles the heavy lifting; predict() wraps it for the legacy API.
        # This ensures that both APIs stay in sync.
        result = self.score(pdb_content)
        return (result.label == "High Quality"), result.global_score, result.features

    def score(self, pdb_content: str) -> QualityScore:
        """Score a PDB structure, returning a rich :class:`QualityScore` object.

        This is the preferred API over ``predict()`` because it also returns
        per-residue pLDDT confidence scores (when the v2 checkpoint is loaded).

        Args:
            pdb_content: PDB-format string.

        Returns:
            QualityScore with global and per-residue quality information.

        Raises:
            ImportError: If torch or torch_geometric are not installed.
            ValueError: If the PDB contains too few residues to build a graph.

        """
        # PyTorch and PyG are heavy dependencies — we only import them here
        # to ensure that users without GNN support can still use the rest
        # of synth-pdb (e.g. core generation) without crashing on import.
        # This is the 'Optional Dependency' pattern which keeps the core slim.
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for GNNQualityClassifier. "
                "Install with: pip install synth-pdb[gnn]"
            ) from exc

        # Import PyG utilities only at runtime to minimize overhead.
        # Batch is used to group individual graphs into a single tensor structure.
        from torch_geometric.data import Batch
        from .graph import build_protein_graph

        # Step 1: Convert PDB text into a PyG Interaction Graph.
        # The graph connects residues within a spatial 10 Angstrom cutoff.
        # This graph representation is more physically meaningful than a
        # linear sequence because it explicitly models tertiary contacts.
        graph = build_protein_graph(pdb_content)

        # Step 2: Wrap in a batch object (the GNN model expects batched input).
        # Even for a single structure, PyG requires a Batch container to
        # correctly handle the node-to-graph mapping vector (batch).
        batch = Batch.from_data_list([graph])

        # Step 3: Inference mode setup.
        # Check that a model was successfully loaded during __init__.
        assert self.model is not None, "Model not loaded"

        # Set to evaluation mode to disable Dropout and Batch Normalization training.
        # This is critical for obtaining deterministic and correct inference results.
        self.model.eval()

        # Disable gradient tracking to speed up the forward pass and save memory.
        with torch.no_grad():
            if self._has_residue_head:
                # v2 checkpoint logic: dual-output forward pass.
                # We use forward_with_node_embeddings to retrieve both the
                # global classification log-probs and the node-level pLDDT.
                # This head was trained specifically to predict local lDDT.
                log_probs, per_res_tensor = type(self.model).forward_with_node_embeddings(
                    self.model,
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                # Convert the internal torch tensor back to a standard Python list.
                # Squeeze removes the redundant singleton dimension [N, 1] -> [N].
                per_residue = per_res_tensor.squeeze(-1).tolist()
            else:
                # v1 checkpoint logic: legacy global-only forward pass.
                # Node-level confidence is not available in older model variants.
                # We return an empty list to signal that pLDDT is not supported.
                log_probs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                per_residue = []

        # Step 4: Post-process raw model outputs for the user.
        # The raw log_probs tensor is [1, 2] -> [log P(Bad), log P(Good)].
        # We apply the exponent to convert log-space back to probability [0, 1].
        prob_good = float(log_probs.exp()[0, 1].item())

        # Determine the categorical label based on the 0.5 probability threshold.
        label = "High Quality" if prob_good > 0.5 else "Low Quality"

        # Map per-residue floats back to human-readable AlphaFold-style bands.
        # These bands: Very High (0.9), High (0.7), Uncertain (0.5), Low (<0.5)
        # allow for immediate visual interpretation of the model's confidence.
        residue_labels = [_plddt_label(s) for s in per_residue]

        # Step 5: Extract aggregate feature statistics (for debugging/XAI).
        # We compute the mean of each input node feature to observe which
        # signal (e.g. b-factors or sequence position) is dominant.
        node_feats = graph.x.numpy()
        feat_dict = {
            name: float(np.mean(node_feats[:, i])) for i, name in enumerate(_FEATURE_NAMES)
        }

        # Return a structured QualityScore object with all processed metrics.
        return QualityScore(
            global_score=prob_good,
            label=label,
            per_residue=per_residue,
            residue_labels=residue_labels,
            features=feat_dict,
            n_residues=graph.num_nodes,
        )

    def save(self, path: str) -> None:
        """Save model weights and architecture config to a ``.pt`` checkpoint.

        The checkpoint is 'self-describing' — it includes the layer dimensions
        required to re-instantiate the `ProteinGNN` class without an external
        config or JSON file.  This follows the 'single file per asset' philosophy
        which simplifies deployment and model versioning.

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
        # We use standard torch.save() to remain compatible with standard tools.
        torch.save(payload, path)
        self._model_path = path
        logger.info("GNN checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights and re-configure architecture from a checkpoint.

        This method acts as a factory: it reads the metadata in the checkpoint
        to determine the model's width and depth, builds the graph layers,
        and then injects the learned weights. This allows one classifier instance
        to seamlessly switch between different model variants or generations.

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
            # Even if trained on CUDA, we load to CPU for local inference.
            # weights_only=False is used because our checkpoint is a custom dict.
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
            self.model.load_state_dict(checkpoint["state_dict"])

            # Switch to evaluation mode (essential: fixes dropout and batchnorm behavior).
            # Without this, the model might produce stochastic or incorrect results.
            self.model.eval()
            self._model_path = path

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
