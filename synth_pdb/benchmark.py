"""synth_pdb.benchmark.
~~~~~~~~~~~~~~~~~~~~~~
AlphaFold / ESMFold Benchmarking Suite for synth-pdb.

The benchmark asks a simple but powerful question:
  "Can AI structure prediction models predict synth-pdb synthetic structures?"

Because synth-pdb controls the ground-truth structure and its NMR chemical
shifts, the benchmark is perfectly objective — no ambiguity about which
experimental structure is "correct".

─────────────────────────────────────────────────────────────────────────────
SCIENTIFIC RATIONALE — Why benchmark against synthetic structures?
─────────────────────────────────────────────────────────────────────────────
Standard benchmarks for protein structure prediction (like CASP or CAMEO) rely
on experimental structures from the PDB. While essential, these have noise:
  • Resolution limits and refinement errors.
  • Conformational ensemble averaging in crystals/cryo-EM.
  • Missing loops or disordered regions.

The `synth-pdb` benchmark provides a **Perfect Control Group**:
  1. **Objective Truth**: We control every atomic coordinate exactly.
  2. **Zero Ambiguity**: NMR chemical shifts are predicted directly from
     the ground truth, eliminating experimental measurement error.
  3. **Targeted Stress**: We can generate structures that specifically
     challenge AI models (e.g., extremely long helices or unusual linkers).

─────────────────────────────────────────────────────────────────────────────
METRICS EXPLAINED — The Structural Biology Toolkit
─────────────────────────────────────────────────────────────────────────────
We use the same standards used to judge AlphaFold in CASP competitions:

  • **TM-score** (Template Modeling score): Measures global topology overlap.
    Ranges [0, 1]. >0.5 typically means the "same fold."
  • **GDT-TS** (Global Distance Test Total Score): Percentage of residues
    whose Cα positions are within 1, 2, 4, or 8 Å of the target.
  • **lDDT** (Local Distance Difference Test): Measures how well the local
    inter-atomic distances are preserved. Unlike RMSD, it doesn't require
    superposition, making it robust to hinge motions.

Quick start::

    from synth_pdb.benchmark import run_benchmark

    results = run_benchmark(n_structures=10, lengths=[20, 30])
    print(results.summary())
    results.to_csv("benchmark_results.csv")

Or with the CLI::

    python scripts/run_benchmark.py --predictor esmfold --n-structures 20

Supported predictors
--------------------
- **ESMFold** (default): Meta's sequence-to-structure model via HuggingFace
  ``transformers`` library.  No API key required.  ~700 MB model download on
  first use.  ``pip install transformers accelerate``
- **Custom callable**: Pass any ``predictor_fn(sequence: str) → pdb_str`` for
  full flexibility with any model (ColabFold, OmegaFold, RoseTTAFold, etc.)

Metrics computed
----------------
For each generated structure:

  - ``tm_score``   TM-score between ground-truth and predicted Cα trace
  - ``gdt_ts``     GDT-TS score
  - ``lddt``       Mean lDDT across all residues
  - ``rmsd``       Cα-RMSD after Kabsch superposition (Å)
  - ``shift_rmsd`` Chemical shift RMSD (ppm) — compares shifts predicted from
                   ground-truth structure vs. from the AI-predicted structure
  - ``gnn_score_ref``  GNN pLDDT of the ground-truth structure
  - ``gnn_score_pred`` GNN pLDDT of the AI-predicted structure

"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, cast

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StructureResult:
    """Quality metrics for a single (ground-truth, predicted) structure pair.

    This container tracks the "Delta" between what synth-pdb generated and
    what an AI model (like ESMFold) predicted. It captures both global
    geometry (TM-score) and local physics (NMR Chemical Shift RMSD).

    Attributes
    ----------
    sequence        : str    Amino acid sequence (single-letter).
    length          : int    Number of residues.
    conformation    : str    Ground-truth conformation type (e.g. "alpha").
    tm_score        : float  TM-score ∈ [0, 1]. High values indicate correct fold.
    gdt_ts          : float  GDT-TS ∈ [0, 1]. Captures global Cα trace fidelity.
    lddt_mean       : float  Mean lDDT ∈ [0, 1]. Captures local geometric accuracy.
    rmsd            : float  Cα-RMSD in Å after superposition.
    shift_rmsd      : float  NMR Chemical shift RMSD in ppm. A "Physical Audit":
                             measures if the AI-predicted structure "sounds" like
                              the ground truth structure under a virtual spectrometer.
    gnn_score_ref   : float  GNN quality score for the ground-truth structure.
    gnn_score_pred  : float  GNN quality score for the predicted structure.
    predictor_time_s: float  Wall-clock inference time for this structure (s).
    error           : str    Non-empty if prediction failed (e.g. CUDA OOM).
    """

    sequence: str = ""
    length: int = 0
    conformation: str = ""
    tm_score: float = float("nan")
    gdt_ts: float = float("nan")
    lddt_mean: float = float("nan")
    rmsd: float = float("nan")
    shift_rmsd: float = float("nan")
    gnn_score_ref: float = float("nan")
    gnn_score_pred: float = float("nan")
    predictor_time_s: float = float("nan")
    error: str = ""


@dataclass
class BenchmarkResults:
    """Aggregated results for a complete benchmarking run.

    Attributes
    ----------
    results       : list[StructureResult]  Per-structure results.
    predictor     : str                    Name of the predictor used.
    n_structures  : int                    Number of structures attempted.
    n_success     : int                    Structures successfully predicted.
    """

    results: list[StructureResult] = field(default_factory=list)
    predictor: str = "unknown"
    n_structures: int = 0
    n_success: int = 0

    def summary(self) -> str:
        """Return a formatted summary of the benchmark results."""
        good = [r for r in self.results if not r.error]
        if not good:
            return f"Benchmark '{self.predictor}': 0/{self.n_structures} succeeded."

        tm_scores = [r.tm_score for r in good if np.isfinite(r.tm_score)]
        gdt = [r.gdt_ts for r in good if np.isfinite(r.gdt_ts)]
        lddts = [r.lddt_mean for r in good if np.isfinite(r.lddt_mean)]
        rmsds = [r.rmsd for r in good if np.isfinite(r.rmsd)]
        shift = [r.shift_rmsd for r in good if np.isfinite(r.shift_rmsd)]
        gnn_pred = [r.gnn_score_pred for r in good if np.isfinite(r.gnn_score_pred)]

        lines = [
            f"━━ Benchmark: {self.predictor} ({self.n_success}/{self.n_structures} structures) ━━",
            f"  TM-score   mean={np.mean(tm_scores):.3f}  std={np.std(tm_scores):.3f}  min={np.min(tm_scores):.3f}  max={np.max(tm_scores):.3f}",
            f"  GDT-TS     mean={np.mean(gdt):.3f}  std={np.std(gdt):.3f}",
            f"  lDDT       mean={np.mean(lddts):.3f}  std={np.std(lddts):.3f}",
            f"  Cα-RMSD    mean={np.mean(rmsds):.2f} Å  std={np.std(rmsds):.2f} Å",
        ]
        if shift:
            lines.append(f"  Shift RMSD mean={np.mean(shift):.3f} ppm  std={np.std(shift):.3f} ppm")
        if gnn_pred:
            lines.append(f"  GNN pLDDT  mean={np.mean(gnn_pred):.3f} (predicted structures)")

        # Fraction with TM-score > 0.5 (same fold)
        same_fold = sum(1 for t in tm_scores if t > 0.5)
        lines.append(
            f"\n  Structures with TM-score > 0.5 (same fold): "
            f"{same_fold}/{len(tm_scores)} ({100*same_fold/len(tm_scores):.0f}%)"
        )
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Write full per-structure results to a CSV file."""
        import csv

        fields = [
            "sequence",
            "length",
            "conformation",
            "tm_score",
            "gdt_ts",
            "lddt_mean",
            "rmsd",
            "shift_rmsd",
            "gnn_score_ref",
            "gnn_score_pred",
            "predictor_time_s",
            "error",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.results:
                writer.writerow({k: getattr(r, k) for k in fields})
        logger.info("Benchmark results saved to %s", path)

    def to_dataframe(self) -> pd.DataFrame:  # type: ignore[name-defined]
        """Return results as a pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe().") from exc

        return pd.DataFrame(
            [{k: getattr(r, k) for k in StructureResult.__dataclass_fields__} for r in self.results]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Predictor backends
# ─────────────────────────────────────────────────────────────────────────────


def _load_esmfold_predictor() -> Callable[[str], str]:
    """Return a callable that runs ESMFold inference via HuggingFace transformers.

    ─────────────────────────────────────────────────────────────────────────────
    AI ARCHITECTURE — What is ESMFold?
    ─────────────────────────────────────────────────────────────────────────────
    Unlike AlphaFold 2, which uses a multiple sequence alignment (MSA) and a
    folding trunk, ESMFold is an **"End-to-End" language model**. It uses the
    internal representations of the ESM-2 protein language model to predict 3D
    coordinates directly from a single sequence.

    Pros: Extremely fast (seconds vs minutes for AlphaFold).
    Cons: Slightly lower accuracy on de novo folds or unusual geometries.

    The ESMFold model (~700 MB) is downloaded automatically on first call and
    cached by HuggingFace.  No API key required.

    Returns
    -------
    callable : str → str
        Accepts an amino acid sequence, returns a PDB-format string.

    Raises
    ------
    ImportError  If ``transformers`` is not installed.

    """
    try:
        from transformers import EsmForProteinFolding, EsmTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for ESMFold. "
            "Install with: pip install transformers accelerate"
        ) from exc

    logger.info("Loading ESMFold tokenizer and model (first load may take several minutes)...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model.eval()
    logger.info("ESMFold loaded.")

    try:
        import torch

        if torch.cuda.is_available():
            model = model.cuda()  # type: ignore[call-arg]
            logger.info("ESMFold moved to CUDA.")
    except ImportError:
        pass

    def predict(sequence: str) -> str:
        """Run ESMFold on a single sequence; return PDB string."""
        import torch

        with torch.no_grad():
            inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
            if next(model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)  # type: ignore[call-arg]

        # Convert ESMFold output to PDB
        from transformers.models.esm.openfold_utils.protein import Protein, to_pdb
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        pdb_protein = Protein(
            aatype=outputs["aatype"][0].cpu().numpy(),
            atom_positions=final_atom_positions[0].cpu().numpy(),
            atom_mask=outputs["atom37_atom_exists"][0].cpu().numpy(),
            residue_index=outputs["residue_index"][0].cpu().numpy() + 1,
            b_factors=outputs["plddt"][0].cpu().numpy(),
            chain_index=outputs.get("chain_index", None),
        )
        return str(to_pdb(pdb_protein))

    return predict


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark engine
# ─────────────────────────────────────────────────────────────────────────────


def run_benchmark(
    n_structures: int = 20,
    lengths: list[int] | None = None,
    conformations: list[str] | None = None,
    predictor: str | Callable[[str], str] = "esmfold",
    *,
    compute_shifts: bool = True,
    compute_gnn: bool = True,
    random_state: int = 42,
) -> BenchmarkResults:
    """Run the synth-pdb vs. AI structure prediction benchmark.

    Generates ``n_structures`` synthetic protein structures of varying length
    and secondary structure content, then asks the specified predictor to
    reconstruct each structure from sequence alone.  The prediction is
    evaluated against the ground-truth on TM-score, GDT-TS, lDDT, Cα-RMSD,
    and (optionally) NMR chemical shift RMSD.

    ─────────────────────────────────────────────────────────────────────────────
    BENCHMARK LIFECYCLE
    ─────────────────────────────────────────────────────────────────────────────
    For each structure in the trial:
      1. **Generate**: Create a ground-truth PDB with specific geometry.
      2. **Extract**: Convert the 3D structure to its 1D amino acid sequence.
      3. **Predict**: Blindly ask the AI model to predict the 3D structure.
      4. **Evaluate**: Compute TM-score, GDT-TS, and lDDT between (1) and (3).
      5. **Audit**: (Optional) Compare predicted NMR chemical shifts.

    This "Circular Validation" ensures the AI model is learning the true
    underlying physics of protein folding, not just memorizing the PDB.

    Parameters
    ----------
    n_structures : int
        Total number of test structures to generate.
    lengths : list[int], optional
        Pool of chain lengths to sample from (default: [20, 30, 50]).
        Lengths are sampled uniformly.
    conformations : list[str], optional
        Pool of conformations to sample from (default: ["alpha", "beta"]).
    predictor : str or callable
        "esmfold" to use the bundled ESMFold backend, or any callable
        ``predictor_fn(sequence: str) → pdb_str``.
    compute_shifts : bool
        Whether to compute NMR chemical shift RMSD (requires ``synth_pdb``
        chemical_shifts module).  Default True.
    compute_gnn : bool
        Whether to score both structures with the GNN quality scorer.
        Default True.
    random_state : int
        RNG seed for reproducible structure generation.

    Returns
    -------
    BenchmarkResults
        Call ``.summary()`` for a formatted text report or ``.to_csv()`` to
        export full results.

    Examples
    --------
    >>> from synth_pdb.benchmark import run_benchmark
    >>> results = run_benchmark(n_structures=10)
    >>> print(results.summary())

    """
    from synth_pdb.benchmark_metrics import (
        extract_ca_coords,
        gdt_ts as _gdt_ts,
        lddt as _lddt,
        shift_rmsd as _shift_rmsd,
        superpose_kabsch,
        tm_score as _tm_score,
    )
    from synth_pdb.generator import generate_pdb_content

    if lengths is None:
        lengths = [20, 30, 50]
    if conformations is None:
        conformations = ["alpha", "beta"]

    # ── Load predictor ─────────────────────────────────────────────────
    predictor_name: str
    predictor_fn: Callable[[str], str]
    if callable(predictor):
        predictor_fn = predictor
        predictor_name = getattr(predictor, "__name__", "custom")
    elif predictor == "esmfold":
        predictor_fn = _load_esmfold_predictor()
        predictor_name = "ESMFold"
    else:
        raise ValueError(f"Unknown predictor: {predictor!r}. Use 'esmfold' or a callable.")

    # ── Optional GNN scorer ────────────────────────────────────────────
    # The GNN scorer provides an independent physical audit of both the
    # reference and predicted structures. It requires torch and torch_geometric.
    gnn_clf = None
    if compute_gnn:
        try:
            from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier

            gnn_clf = GNNQualityClassifier()
        except ImportError:
            logger.warning(
                "torch/torch_geometric not available — GNN scoring disabled. "
                "Install with: pip install synth-pdb[gnn]"
            )

    # ── Optional chemical shift predictor ─────────────────────────────
    # This is the "Gold Standard" of structural validation in synth-pdb.
    # If the AI model predicts a structure with a high TM-score but a bad
    # Shift RMSD, it implies the global fold is correct but the local
    # backbone packaging or H-bonding is physically unrealistic.
    shift_fn = None
    if compute_shifts:
        try:
            from synth_pdb.chemical_shifts import predict_chemical_shifts as _predict_shifts

            shift_fn = _predict_shifts
        except ImportError:
            logger.warning("chemical_shifts module not available — shift RMSD disabled.")

    # ── Generate and evaluate ──────────────────────────────────────────
    rng = np.random.default_rng(random_state)
    benchmark_results = BenchmarkResults(predictor=predictor_name, n_structures=n_structures)

    for i in range(n_structures):
        length = int(rng.choice(lengths))
        conformation = str(rng.choice(conformations))
        result = StructureResult(length=length, conformation=conformation)

        logger.info(
            "Structure %d/%d  length=%d  conformation=%s",
            i + 1,
            n_structures,
            length,
            conformation,
        )

        try:
            # ── 1. Generate ground-truth structure ─────────────────────
            # We skip energy minimisation to ensure the ground truth is
            # mathematically perfect (exactly at Ramachandran centres).
            ref_pdb = cast(
                str,
                generate_pdb_content(
                    length=length,
                    conformation=conformation,
                    minimize_energy=False,
                ),
            )

            # ── 2. Extract sequence ────────────────────────────────────
            # The AI models only get the sequence — no coordinate hints allowed!
            sequence = _extract_sequence(ref_pdb)
            result.sequence = sequence

            # ── 3. Run predictor ───────────────────────────────────────
            t0 = time.perf_counter()
            pred_pdb = predictor_fn(sequence)
            result.predictor_time_s = time.perf_counter() - t0

            # ── 4. Structural metrics ──────────────────────────────────
            # We extract Cα coordinates (the "backbone trace") to compute
            # global alignment metrics.
            ca_ref = extract_ca_coords(ref_pdb)
            ca_pred = extract_ca_coords(pred_pdb)

            # Trim to the shorter of the two (in case of predictor truncation).
            n_align = min(len(ca_ref), len(ca_pred))
            ca_ref_a = ca_ref[:n_align]
            ca_pred_a = ca_pred[:n_align]

            # Compute standard CASP metrics
            result.tm_score = _tm_score(ca_pred_a, ca_ref_a)
            result.gdt_ts = _gdt_ts(ca_pred_a, ca_ref_a)
            per_res_lddt = _lddt(ca_pred_a, ca_ref_a)
            result.lddt_mean = float(np.mean(per_res_lddt))

            # RMSD after optimal Kabsch superposition (rigid-body alignment)
            _, result.rmsd = superpose_kabsch(ca_pred_a, ca_ref_a)

            # ── 5. Chemical shift RMSD ─────────────────────────────────
            if shift_fn is not None:
                try:
                    ref_shifts_raw = shift_fn(ref_pdb)
                    pred_shifts_raw = shift_fn(pred_pdb)
                    # Convert to nucleus → np.array format
                    ref_shifts = _shifts_to_arrays(ref_shifts_raw)
                    pred_shifts = _shifts_to_arrays(pred_shifts_raw)
                    result.shift_rmsd = _shift_rmsd(pred_shifts, ref_shifts)
                except Exception as se:
                    logger.warning("Shift RMSD failed for structure %d: %s", i, se)

            # ── 6. GNN quality scores ──────────────────────────────────
            if gnn_clf is not None:
                try:
                    ref_score = gnn_clf.score(ref_pdb)
                    pred_score = gnn_clf.score(pred_pdb)
                    result.gnn_score_ref = ref_score.global_score
                    result.gnn_score_pred = pred_score.global_score
                except Exception as ge:
                    logger.warning("GNN scoring failed for structure %d: %s", i, ge)

            benchmark_results.n_success += 1

        except Exception as exc:
            result.error = str(exc)
            logger.warning("Structure %d failed: %s", i + 1, exc, exc_info=True)

        benchmark_results.results.append(result)

    return benchmark_results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common non-standard → map to nearest standard
    "HIE": "H",
    "HID": "H",
    "HIP": "H",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}


def _extract_sequence(pdb_content: str) -> str:
    """Extract one-letter amino acid sequence from PDB ATOM records.

    We only look at Cα atoms to avoid duplicate residues from multiple
    atom records.  We use a dictionary keyed by (chain, res_num) to
    robustly handle non-sequential residue numbers in the PDB file.
    """
    seen: dict[tuple[str, int], str] = {}
    for line in pdb_content.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain = line[21].strip() or "A"
        try:
            res_num = int(line[22:26].strip())
            res_name = line[17:20].strip()
        except (ValueError, IndexError):
            continue
        key = (chain, res_num)
        if key not in seen:
            seen[key] = _THREE_TO_ONE.get(res_name, "X")

    ordered = sorted(seen.keys())
    return "".join(seen[k] for k in ordered)


def _shifts_to_arrays(shifts_data: object) -> dict[str, np.ndarray]:
    """Convert predict_shifts() output to nucleus → np.ndarray format.

    The ``synth_pdb.chemical_shifts.predict_shifts()`` function returns a
    dict or list of per-residue dicts.  This helper normalises it into the
    ``{"H": array, "C": array, "N": array}`` format expected by ``shift_rmsd``.

    This normalization is required because different chemical shift predictors
    (SPARTA+, SHIFTX2) return data in different shapes.
    """
    if isinstance(shifts_data, dict):
        # Already in nucleus → array format
        return {k: np.asarray(v, dtype=np.float32) for k, v in shifts_data.items()}

    if isinstance(shifts_data, list | np.ndarray):
        # List of per-residue dicts — extract H, C, N columns
        h_vals, c_vals, n_vals = [], [], []
        for res in shifts_data:
            if isinstance(res, dict):
                h_vals.append(res.get("H", float("nan")))
                c_vals.append(res.get("C", res.get("CA", float("nan"))))
                n_vals.append(res.get("N", float("nan")))
        return {
            "H": np.array(h_vals, dtype=np.float32),
            "C": np.array(c_vals, dtype=np.float32),
            "N": np.array(n_vals, dtype=np.float32),
        }

    logger.warning("_shifts_to_arrays: unrecognised format %s", type(shifts_data))
    return {}
