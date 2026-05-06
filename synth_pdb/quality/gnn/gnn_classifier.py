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

    Also supports per-residue pLDDT-like confidence scoring (requires a v2
    checkpoint trained with the per-residue auxiliary head).

    ── When is a GNN better than a Random Forest? ─────────────────────────
    The RF classifier uses hand-crafted, per-structure summary statistics
    (e.g. "fraction of residues in favoured Ramachandran regions").

    The GNN works directly on the full residue interaction graph, so it can:
      • Learn WHICH specific contacts are problematic, not just aggregate counts
      • Capture spatial patterns (e.g. a single clashing i/i+4 contact pair)
      • Generalise to protein classes or contact patterns not seen in training
        (because the pattern recogniser is learned, not hand-engineered)

    The trade-off is training time and interpretability:
      • RF:  ~0.03 s to train, ~0.1 ms/sample inference, instant setup
      • GNN: ~3 s to train,   ~0.3 ms/sample inference, richer features
    ────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, model_path: str | None = None):
        """Args:
        model_path: Path to a .pt checkpoint written by GNNQualityClassifier.save().
                    If None, looks for the default bundled checkpoint (v2 first, v1 fallback).
                    If no checkpoint is found, initialises a random-weight model
                    (useful for testing graph construction without training).

        """
        self.model: Any | None = None
        self._model_path: str | None = None
        self._has_residue_head: bool = False  # True if v2 checkpoint with pLDDT head

        if model_path:
            self.load(model_path)
        else:
            # Prefer v2 (per-residue head) over v1 (global only)
            v2 = os.path.normpath(_DEFAULT_CHECKPOINT_V2)
            v1 = os.path.normpath(_DEFAULT_CHECKPOINT_V1)
            if os.path.exists(v2):
                self.load(v2)
            elif os.path.exists(v1):
                logger.info(
                    "v2 checkpoint not found, loading v1 (no per-residue pLDDT). "
                    "Run scripts/train_gnn_quality_filter.py to produce v2."
                )
                self.load(v1)
            else:
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

        ── What happens inside ──────────────────────────────────────────
        1. build_protein_graph(pdb_content) → PyG Data object
        2. Batch.from_data_list([graph])    → single-element batch
        3. model.forward(...)               → log-probabilities [1, 2]
        4. .exp()[0, 1]                     → P(Good) ∈ [0, 1]
        5. > 0.5 threshold                  → is_good bool
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
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for GNNQualityClassifier. "
                "Install with: pip install synth-pdb[gnn]"
            ) from exc

        from torch_geometric.data import Batch

        from .graph import build_protein_graph

        graph = build_protein_graph(pdb_content)
        batch = Batch.from_data_list([graph])

        assert self.model is not None, "Model not loaded"
        self.model.eval()

        with torch.no_grad():
            if self._has_residue_head:
                # v2 checkpoint: dual-output forward pass
                log_probs, per_res_tensor = type(self.model).forward_with_node_embeddings(
                    self.model,
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                per_residue = per_res_tensor.squeeze(-1).tolist()
            else:
                # v1 checkpoint: global only
                log_probs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                per_residue = []

        prob_good = float(log_probs.exp()[0, 1].item())
        label = "High Quality" if prob_good > 0.5 else "Low Quality"
        residue_labels = [_plddt_label(s) for s in per_residue]

        node_feats = graph.x.numpy()
        feat_dict = {
            name: float(np.mean(node_feats[:, i])) for i, name in enumerate(_FEATURE_NAMES)
        }

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

        Args:
            path: Destination file path (should end in .pt).

        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to save a GNN checkpoint.") from exc

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        assert self.model is not None, "Model not loaded"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "node_features": self.model.node_features,
                "edge_features": self.model.edge_features,
                "hidden_dim": self.model.hidden_dim,
                "num_classes": self.model.num_classes,
            },
            path,
        )
        self._model_path = path
        logger.info("GNN checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from a ``.pt`` checkpoint.

        Args:
            path: Path to a .pt checkpoint written by GNNQualityClassifier.save().

        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load a GNN checkpoint.") from exc

        from .model import ProteinGNN

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            import typing

            self.model = typing.cast(
                Any,
                ProteinGNN(
                    node_features=checkpoint["node_features"],
                    edge_features=checkpoint["edge_features"],
                    hidden_dim=checkpoint["hidden_dim"],
                    num_classes=checkpoint["num_classes"],
                ),
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self._model_path = path

            # Detect if this checkpoint has the per-residue pLDDT head (v2)
            self._has_residue_head = "residue_lin1.weight" in checkpoint["state_dict"]
            logger.info(
                "GNN classifier loaded from %s (per-residue head: %s)",
                path,
                self._has_residue_head,
            )
        except Exception as exc:
            logger.error("Failed to load GNN checkpoint from %s: %s", path, exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_fresh_model(self) -> None:
        """Initialise a randomly-weighted model (v2 architecture with per-residue head)."""
        import typing

        from .model import ProteinGNN

        self.model = typing.cast(
            Any, ProteinGNN(node_features=8, edge_features=2, hidden_dim=64, num_classes=2)
        )
        self.model.eval()
        self._has_residue_head = True
