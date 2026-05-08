"""synth_pdb.score.
~~~~~~~~~~~~~~~~
Top-level, user-facing API for protein structure quality scoring.

This module provides a single-import interface to the GNN quality scorer,
hiding all internal complexity behind two clean functions::

    from synth_pdb.score import score_structure, score_batch

    # Score a single structure from a file path or PDB string
    result = score_structure("my_protein.pdb")
    print(f"Global quality: {result.global_score:.3f}  ({result.label})")
    print(f"Per-residue pLDDT: {result.per_residue}")

    # Score a batch (processes all structures in one call)
    results = score_batch(["prot1.pdb", "prot2.pdb", "prot3.pdb"])
    top = sorted(results, key=lambda r: r.global_score, reverse=True)

Design goals
------------
1. **Zero-friction**: works out of the box with the bundled pre-trained
   weights; no training required.
2. **Single import**: ``from synth_pdb.score import score_structure`` is all
   a user ever needs.
3. **Lazy loading**: PyTorch and torch_geometric are imported only when a
   scoring function is called — users without GPU/torch can still import
   the rest of synth_pdb.
4. **Backward compatibility**: the underlying ``GNNQualityClassifier`` is
   still available for users who need lower-level access.

Requirements
------------
    pip install synth-pdb[gnn]

or equivalently::

    pip install torch torch_geometric

"""

import logging
import os
from typing import Union

logger = logging.getLogger(__name__)

# Module-level singleton to avoid re-loading weights on every call.
# Lazily initialised on first use.
_classifier = None


def _get_classifier() -> "GNNQualityClassifier":  # type: ignore[name-defined]
    """Return the module-level GNNQualityClassifier singleton, loading it lazily."""
    global _classifier
    if _classifier is None:
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        _classifier = GNNQualityClassifier()
    return _classifier


def score_structure(
    source: Union[str, "os.PathLike[str]"],
    *,
    model_path: str | None = None,
) -> "QualityScore":  # type: ignore[name-defined]
    """Score a single protein structure and return a rich quality assessment.

    Parameters
    ----------
    source : str or path-like
        Either:
        • A file path ending in ``.pdb`` — the file is read automatically.
        • A raw PDB-format string (must start with ``ATOM`` or ``REMARK``).
    model_path : str, optional
        Path to a custom ``.pt`` checkpoint.  Defaults to the bundled
        pre-trained weights (``gnn_quality_v2.pt``).

    Returns
    -------
    QualityScore
        Dataclass with:
        ``global_score`` (float ∈ [0, 1]),
        ``label`` ("High Quality" / "Low Quality"),
        ``per_residue`` (list of per-residue pLDDT scores ∈ [0, 1]),
        ``residue_labels`` (list of "Very High"/"High"/"Uncertain"/"Low"),
        ``n_residues`` (int),
        ``features`` (dict of mean input features, for debugging).

    Raises
    ------
    FileNotFoundError
        If *source* looks like a file path but does not exist.
    ImportError
        If ``torch`` or ``torch_geometric`` are not installed.
    ValueError
        If the PDB content has fewer than 2 residues with Cα atoms.

    Examples
    --------
    >>> from synth_pdb.score import score_structure
    >>> result = score_structure("outputs/my_helix.pdb")
    >>> print(result.global_score, result.label)
    0.987 High Quality
    >>> low = [i for i, lbl in enumerate(result.residue_labels) if lbl == "Low"]
    >>> print(f"Low-confidence residues: {low}")
    Low-confidence residues: []

    """
    # ── Resolve source → PDB string ────────────────────────────────────
    source_str = os.fspath(source) if not isinstance(source, str) else source

    if source_str.strip().startswith(("ATOM", "REMARK", "HEADER", "MODEL")):
        # Treat as inline PDB content
        pdb_content = source_str
    else:
        # Treat as file path
        path = os.path.expanduser(source_str)
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDB file not found: {path}")
        with open(path) as fh:
            pdb_content = fh.read()

    # ── Load classifier (singleton) ────────────────────────────────────
    if model_path:
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        clf = GNNQualityClassifier(model_path=model_path)
    else:
        clf = _get_classifier()

    return clf.score(pdb_content)


def score_batch(
    sources: list[Union[str, "os.PathLike[str]"]],
    *,
    model_path: str | None = None,
) -> "list[QualityScore]":  # type: ignore[name-defined]
    """Score a list of protein structures efficiently using a shared model.

    The GNN weights are loaded once and reused for all structures, making
    this significantly faster than calling ``score_structure()`` in a loop
    when the list is large (avoids repeated checkpoint loading).

    Parameters
    ----------
    sources : list of str or path-like
        Each element is either a file path ending in ``.pdb`` or a raw PDB
        string.  Mixed lists are accepted.
    model_path : str, optional
        Path to a custom ``.pt`` checkpoint.

    Returns
    -------
    list[QualityScore]
        One result per input, in the same order as *sources*.

    Examples
    --------
    >>> from synth_pdb.score import score_batch
    >>> paths = ["helix.pdb", "strand.pdb", "random.pdb"]
    >>> results = score_batch(paths)
    >>> top = max(results, key=lambda r: r.global_score)
    >>> print(f"Best: {top.global_score:.3f}  ({top.label})")

    """
    if model_path:
        from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

        clf = GNNQualityClassifier(model_path=model_path)
    else:
        clf = _get_classifier()

    results = []
    for i, source in enumerate(sources):
        try:
            source_str = os.fspath(source) if not isinstance(source, str) else source

            if source_str.strip().startswith(("ATOM", "REMARK", "HEADER", "MODEL")):
                pdb_content = source_str
            else:
                path = os.path.expanduser(source_str)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"PDB file not found: {path}")
                with open(path) as fh:
                    pdb_content = fh.read()

            results.append(clf.score(pdb_content))
        except Exception as exc:
            logger.warning("score_batch: failed on item %d (%s): %s", i, source, exc)
            # Return a sentinel score so the list length always matches the input
            from synth_pdb.quality.gnn.gnn_classifier import QualityScore

            results.append(
                QualityScore(
                    global_score=float("nan"),
                    label="Error",
                    per_residue=[],
                    residue_labels=[],
                    features={},
                    n_residues=0,
                )
            )

    return results


# Re-export QualityScore at module level for convenience
from synth_pdb.quality.gnn.gnn_classifier import QualityScore  # noqa: E402

__all__ = ["score_structure", "score_batch", "QualityScore"]
