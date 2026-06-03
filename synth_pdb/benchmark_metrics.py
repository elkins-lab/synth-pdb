"""synth_pdb.benchmark_metrics.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure-numpy structural biology evaluation metrics for the AlphaFold benchmarking suite.

All functions operate on numpy arrays and have no optional dependencies,
making them trivially unit-testable and usable in headless environments.

-----------------------------------------------------------------------------
Metrics implemented
-----------------------------------------------------------------------------

TM-score
    Template Modelling score in [0, 1].  A TM-score > 0.5 indicates that two
    proteins share the same global topology regardless of RMSD.  It is the
    primary metric in CASP (Critical Assessment of Structure Prediction).
    Reference: Zhang & Skolnick (2004) *Proteins* 57, 702-710.

lDDT
    Local Distance Difference Test in [0, 1].  Evaluates per-residue local
    geometry without superposition.  Used by AlphaFold to report per-residue
    confidence (pLDDT) during training.
    Reference: Mariani et al. (2013) *Bioinformatics* 29, 2722-2728.

GDT-TS
    Global Distance Test - Total Score.  CASP standard metric computed as
    the average fraction of Calpha atoms within 1, 2, 4, 8 A of the reference.
    Reference: Zemla (2003) *Nucleic Acids Research* 31, 3370-3374.

shift_rmsd
    NMR Chemical Shift Root-Mean-Square Deviation.  Measures the agreement
    between predicted and reference 1H/13C/15N chemical shifts.

"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def superpose_kabsch(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Optimally superpose *mobile* onto *reference* using the Kabsch algorithm.

    The Kabsch algorithm finds the rotation matrix R that minimises RMSD
    between two sets of paired points by SVD-decomposing their cross-covariance
    matrix.

    -- Algorithm ------------------------------------------------------------
    1. Translate both sets to their centroids.
    2. Compute cross-covariance  H = mobile_centred.T @ ref_centred
    3. SVD: H = U Sum V^T
    4. R = V diag(1, 1, det(V U^T)) U^T   <- det term fixes reflections
    5. Apply R to mobile.
    -------------------------------------------------------------------------

    Parameters
    ----------
    mobile    : np.ndarray [N, 3]   Coordinates to rotate.
    reference : np.ndarray [N, 3]   Target coordinates.

    Returns
    -------
    rotated  : np.ndarray [N, 3]  Mobile after optimal superposition.
    rmsd     : float              Calpha-RMSD after superposition (A).

    """
    assert mobile.shape == reference.shape, "mobile and reference must have the same shape"
    n = len(mobile)

    # -- Step 1: Centre both sets ---------------------------------------
    mob_c = mobile - mobile.mean(axis=0)
    ref_c = reference - reference.mean(axis=0)

    # -- Step 2: Cross-covariance matrix -------------------------------
    h = mob_c.T @ ref_c  # [3, 3]

    # -- Step 3: SVD ---------------------------------------------------
    u, _s, vt = np.linalg.svd(h)

    # -- Step 4: Rotation matrix (with reflection correction) ----------
    # det(V U^T) is +1 for a proper rotation, -1 for a reflection.
    # We force a proper rotation by flipping the last singular vector.
    d = np.linalg.det(vt.T @ u.T)
    diag = np.diag([1.0, 1.0, d])
    rot = vt.T @ diag @ u.T  # [3, 3]

    # -- Step 5: Apply rotation -----------------------------------------
    rotated = mob_c @ rot.T + reference.mean(axis=0)

    # -- RMSD ----------------------------------------------------------
    diff = rotated - reference
    rmsd = float(math.sqrt(np.sum(diff**2) / n))

    return rotated, rmsd


def tm_score(
    ca_pred: np.ndarray,
    ca_ref: np.ndarray,
    *,
    normalise_by: int | None = None,
) -> float:
    """Compute TM-score between a predicted and reference Calpha trace.

    TM-score is defined as::

        TM = (1/L_ref) Sum_i  1 / (1 + (d_i / d0)^2)

    where ``d_i`` is the distance between the i-th pair of aligned Calpha atoms
    after optimal superposition, ``L_ref`` is the reference chain length, and
    ``d0`` is a length-normalising constant::

        d0 = 1.24 x (L_ref - 15)^(1/3) - 1.8    (for L_ref >= 22)
        d0 = 0.5                                   (for L_ref < 22)

    Parameters
    ----------
    ca_pred      : np.ndarray [N, 3]  Predicted Calpha coordinates.
    ca_ref       : np.ndarray [N, 3]  Reference Calpha coordinates.
    normalise_by : int, optional      If provided, normalise by this length
                                      rather than len(ca_ref).  Useful when
                                      comparing structures of different lengths.

    Returns
    -------
    float  TM-score in (0, 1].  Two random structures ~ 0.17; same fold >= 0.5.

    """
    assert ca_pred.shape == ca_ref.shape and ca_pred.ndim == 2 and ca_pred.shape[1] == 3
    n = len(ca_ref)
    l_ref = normalise_by if normalise_by is not None else n

    # -- d0 normalisation constant --------------------------------------
    # Formula: d0 = 1.24 * (L - 15)^(1/3) - 1.8  (Zhang & Skolnick, 2004)
    #
    # DEPARTURE FROM REFERENCE:
    # Zhang & Skolnick (2004) apply the closed-form formula for L >= 30 and
    # hard-code d0 = 0.5 for L < 30.  This implementation uses L >= 22 as the
    # threshold instead.  Rationale: for L in [22, 29] the formula gives values
    # in [0.50, 0.74] Å — already above the 0.5 floor — so applying it avoids
    # the abrupt discontinuity at L = 30 in the original paper.  The subsequent
    # max(d0, 0.5) clamp guarantees the floor is never violated regardless of L.
    # The practical effect on TM-score is negligible for L > 30 (the range where
    # the metric is most commonly applied) but is documented here for
    # reproducibility.  To match the original paper exactly, change 22 to 30.
    if l_ref >= 22:
        d0 = 1.24 * (l_ref - 15) ** (1.0 / 3.0) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.5)  # never let d0 go below 0.5 (matches paper's floor)

    # -- Superpose then compute per-residue distances -------------------
    rotated, _rmsd = superpose_kabsch(ca_pred, ca_ref)
    d = np.linalg.norm(rotated - ca_ref, axis=1)  # [N]

    # -- TM-score sum --------------------------------------------------
    tm = float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / l_ref)
    return tm


def lddt(
    ca_pred: np.ndarray,
    ca_ref: np.ndarray,
    *,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> np.ndarray:
    """Compute per-residue lDDT scores.

    lDDT measures how well the local distance geometry of a predicted structure
    agrees with the reference, without requiring superposition.

    For each residue i, we look at all pairs (i, j) where j is within
    ``inclusion_radius`` A of i in the *reference* structure.  We then count
    what fraction of those reference distances are preserved in the predicted
    structure within each threshold.

    Per-residue lDDT::

        lDDT_i = (1 / |T|) Sum_t  (fraction of distances within threshold t)

    where |T| = len(thresholds) = 4.

    Parameters
    ----------
    ca_pred          : np.ndarray [N, 3]  Predicted Calpha coordinates.
    ca_ref           : np.ndarray [N, 3]  Reference Calpha coordinates.
    inclusion_radius : float              Only pairs within this radius in the
                                          reference contribute (default 15 A).
    thresholds       : tuple of float     Distance preservation thresholds (A).

    Returns
    -------
    np.ndarray [N]  Per-residue lDDT in [0, 1].  Global lDDT = mean().

    """
    assert ca_pred.shape == ca_ref.shape
    n = len(ca_ref)

    # Pairwise distances in reference and predicted structures - O(N^2) memory
    ref_dists = np.sqrt(((ca_ref[:, None, :] - ca_ref[None, :, :]) ** 2).sum(axis=-1))
    pred_dists = np.sqrt(((ca_pred[:, None, :] - ca_pred[None, :, :]) ** 2).sum(axis=-1))

    # Mask: pairs within inclusion radius in reference, excluding self-pairs
    mask = (ref_dists < inclusion_radius) & (ref_dists > 0)

    per_residue = np.zeros(n, dtype=np.float64)

    for i in range(n):
        neighbours = np.where(mask[i])[0]
        if len(neighbours) == 0:
            per_residue[i] = 0.0
            continue

        ref_d = ref_dists[i, neighbours]
        pred_d = pred_dists[i, neighbours]
        diff = np.abs(pred_d - ref_d)

        # Fraction within each threshold
        fractions = [float(np.mean(diff < t)) for t in thresholds]
        per_residue[i] = float(np.mean(fractions))

    return per_residue.astype(np.float32)


def gdt_ts(
    ca_pred: np.ndarray,
    ca_ref: np.ndarray,
    *,
    cutoffs: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> float:
    """Compute GDT-TS (Global Distance Test - Total Score).

    GDT-TS is the average fraction of Calpha atoms placed within
    {1, 2, 4, 8} A of the reference after optimal superposition.
    It is the primary ranking metric used in CASP competitions.

    Parameters
    ----------
    ca_pred : np.ndarray [N, 3]  Predicted Calpha coordinates.
    ca_ref  : np.ndarray [N, 3]  Reference Calpha coordinates.
    cutoffs : tuple of float     Distance cutoffs in A (default CASP standard).

    Returns
    -------
    float  GDT-TS in [0, 1].  Perfect prediction = 1.0.

    """
    assert ca_pred.shape == ca_ref.shape
    rotated, _rmsd = superpose_kabsch(ca_pred, ca_ref)
    d = np.linalg.norm(rotated - ca_ref, axis=1)  # [N]

    fractions = [float(np.mean(d < c)) for c in cutoffs]
    return float(np.mean(fractions))


def shift_rmsd(
    pred_shifts: dict[str, np.ndarray],
    ref_shifts: dict[str, np.ndarray],
    *,
    nucleus_weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted chemical shift RMSD between predicted and reference shifts.

    Parameters
    ----------
    pred_shifts, ref_shifts : dict mapping nucleus -> np.ndarray [N]
        Nucleus keys are typically ``"H"``, ``"C"``, ``"N"`` (BMRB convention).
        Arrays must be the same length (one value per residue).

        .. note::
            The ``"C"`` key is intentionally ambiguous: it conflates Cα, Cβ,
            and carbonyl C′ shifts.  Those three nucleus types have very
            different inherent RMS errors (~1.0, ~1.5, and ~0.5 ppm,
            respectively).  For high-accuracy benchmarking, supply separate
            ``"CA"``, ``"CB"``, and ``"C"`` keys with per-nucleus weights.

    nucleus_weights : dict, optional
        Per-nucleus weighting.  The default values
        ``{"H": 1.0, "C": 0.25, "N": 0.1}`` are a **heuristic** chosen to
        reflect the relative measurement precision of each nucleus type in a
        typical solution-NMR experiment.  They are **not** derived from a
        published calibration (e.g., SPARTA+ or CamSol) and should not be
        cited as such.  Pass an explicit ``nucleus_weights`` dict if you need
        a specific literature-validated weighting scheme.

    Returns
    -------
    float  Weighted shift RMSD in ppm.  Lower is better.

    Examples
    --------
    >>> rmsd = shift_rmsd(
    ...     {"H": np.array([8.1, 8.2, 8.3])},
    ...     {"H": np.array([8.0, 8.1, 8.4])},
    ... )
    >>> print(f"{rmsd:.4f} ppm")
    0.1000 ppm

    """
    # Heuristic per-nucleus weights (not from SPARTA+ or any single reference).
    # H is weighted 1.0 because 1H shifts have the best experimental precision.
    # C (generic carbon) is downweighted to 0.25 to account for the larger
    # spread of reference-corrected 13C shifts (~1-2 ppm vs ~0.1 ppm for 1H).
    # N is downweighted to 0.1 because 15N shifts span a much wider range
    # (~120 ppm) and prediction errors are correspondingly larger.
    if nucleus_weights is None:
        nucleus_weights = {"H": 1.0, "C": 0.25, "N": 0.1}

    total_sq = 0.0
    total_weight = 0.0

    for nucleus, pred_arr in pred_shifts.items():
        ref_arr = ref_shifts.get(nucleus)
        if ref_arr is None:
            logger.warning("shift_rmsd: nucleus '%s' not in ref_shifts, skipping", nucleus)
            continue

        w = nucleus_weights.get(nucleus, 1.0)
        diff = np.asarray(pred_arr, dtype=np.float64) - np.asarray(ref_arr, dtype=np.float64)

        # Filter NaNs (missing assignments are common in experimental spectra)
        valid = np.isfinite(diff)
        if not valid.any():
            continue

        total_sq += w * float(np.sum(diff[valid] ** 2))
        total_weight += w * float(np.sum(valid))

    if total_weight < 1e-10:
        logger.warning("shift_rmsd: no valid (nucleus, residue) pairs found; returning nan")
        return float("nan")

    return float(math.sqrt(total_sq / total_weight))


def extract_ca_coords(pdb_content: str) -> np.ndarray:
    """Extract Calpha coordinates from a PDB string in residue order.

    A lightweight parser - uses only the standard library, no biotite.

    Parameters
    ----------
    pdb_content : str  Raw PDB-format string.

    Returns
    -------
    np.ndarray [N, 3]  Calpha coordinates in A.

    Raises
    ------
    ValueError  If fewer than 2 Calpha atoms are found.

    """
    coords = []
    seen: set[tuple[str, int]] = set()

    for line in pdb_content.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain = line[21].strip() or "A"
        try:
            res_num = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        key = (chain, res_num)
        if key not in seen:
            seen.add(key)
            coords.append([x, y, z])

    if len(coords) < 2:
        raise ValueError(f"Only {len(coords)} Calpha atoms found in PDB - need at least 2.")

    return np.array(coords, dtype=np.float32)
