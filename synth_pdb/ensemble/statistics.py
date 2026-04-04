"""
Typed dataclasses for NMR ensemble statistics and quality assessment.

Provides structured, type-safe containers for the outputs of ensemble
analysis (RMSD, RMSF, medoid, well-defined regions) and an interpretive
quality tier drawn from the literature.

Quality thresholds:
    HIGH     : RMSD-to-mean < 1.0 Å  (Tejero et al. 2013)
    GOOD     : RMSD-to-mean < 2.0 Å
    MODERATE : RMSD-to-mean ≥ 2.0 Å

Reference:
    Tejero, R., Snyder, D., Mao, B., Aramini, J.M. & Montelione, G.T. (2013).
    PDBStat: A universal restraint converter and restraint analysis software
    package for protein NMR.
    *Journal of Biomolecular NMR*, **56**, 337-351.
    https://doi.org/10.1007/s10858-013-9732-3

Note:
    These dataclasses are pure Python (stdlib only).  They carry no
    dependencies on file formats, PDB parsers, or any scientific computing
    libraries - they are simple, serialisable value objects.
"""

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Thresholds (Tejero et al. 2013)
# ---------------------------------------------------------------------------

_HIGH_QUALITY_RMSD = 1.0  # Å
_GOOD_QUALITY_RMSD = 2.0  # Å
_WELL_DEFINED_PCT_HIGH = 80.0  # %
_WELL_DEFINED_PCT_GOOD = 70.0  # %
_WELL_DEFINED_PCT_MODERATE = 60.0  # %


@dataclass
class EnsembleStatistics:
    """
    Comprehensive, typed container for NMR ensemble quality statistics.

    Replaces the plain ``Dict[str, float]`` pattern with a type-safe
    dataclass that adds computed properties, ``to_dict``/``from_dict``
    round-trips, and a human-readable ``__str__``.

    Attributes:
        n_models: Number of models in the ensemble.
        n_residues: Number of residues per model.
        mean_pairwise_rmsd: Mean of all pairwise RMSDs (Å).
        median_pairwise_rmsd: Median of all pairwise RMSDs (Å).
        std_pairwise_rmsd: Standard deviation of pairwise RMSDs (Å).
        min_pairwise_rmsd: Minimum pairwise RMSD (Å).
        max_pairwise_rmsd: Maximum pairwise RMSD (Å).
        medoid_index: 0-based index of the medoid model.
        medoid_mean_rmsd: Mean RMSD of the medoid to all other models (Å).
        rmsd_to_mean: Mean RMSD-to-ensemble-mean (Å).  Primary precision
            indicator; thresholds from Tejero et al. 2013.
        mean_rmsf: Mean per-residue RMSF (Å).
        max_rmsf: Maximum per-residue RMSF (Å).
        well_defined_residues: Count of residues with RMSF < 1.0 Å.
        pct_well_defined: Percentage of well-defined residues (0-100).

    Examples:
        >>> from synth_pdb.ensemble.statistics import EnsembleStatistics
        >>> stats = EnsembleStatistics(
        ...     n_models=20, n_residues=91,
        ...     mean_pairwise_rmsd=0.85, rmsd_to_mean=0.72,
        ...     pct_well_defined=88.0,
        ... )
        >>> stats.precision
        'HIGH'
        >>> stats.overall_quality
        'Excellent quality NMR ensemble'
        >>> stats.is_ensemble
        True
    """

    # Required fields (no defaults) -----------------------------------------
    n_models: int
    n_residues: int

    # Pairwise RMSD statistics -----------------------------------------------
    mean_pairwise_rmsd: float = 0.0
    median_pairwise_rmsd: float = 0.0
    std_pairwise_rmsd: float = 0.0
    min_pairwise_rmsd: float = 0.0
    max_pairwise_rmsd: float = 0.0

    # Medoid information -----------------------------------------------------
    medoid_index: int = 0
    medoid_mean_rmsd: float = 0.0

    # RMSD to mean (primary precision metric) --------------------------------
    rmsd_to_mean: float = 0.0

    # RMSF statistics --------------------------------------------------------
    mean_rmsf: float = 0.0
    max_rmsf: float = 0.0

    # Well-defined region statistics -----------------------------------------
    well_defined_residues: int = 0
    pct_well_defined: float = 0.0

    # -----------------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------------

    @property
    def is_single_model(self) -> bool:
        """Return ``True`` if the ensemble contains exactly one model."""
        return self.n_models == 1

    @property
    def is_ensemble(self) -> bool:
        """Return ``True`` if the ensemble contains more than one model."""
        return self.n_models > 1

    @property
    def precision(self) -> str:
        """
        Precision tier based on RMSD-to-mean (Tejero et al. 2013).

        Returns:
            ``"HIGH"``     if ``rmsd_to_mean < 1.0`` Å,
            ``"GOOD"``     if ``rmsd_to_mean < 2.0`` Å,
            ``"MODERATE"`` otherwise.
        """
        if self.rmsd_to_mean < _HIGH_QUALITY_RMSD:
            return "HIGH"
        if self.rmsd_to_mean < _GOOD_QUALITY_RMSD:
            return "GOOD"
        return "MODERATE"

    @property
    def overall_quality(self) -> str:
        """
        Human-readable overall quality assessment.

        Combines :attr:`precision` with the percentage of well-defined
        residues to produce a single summary string.

        Returns:
            One of four quality strings ranging from
            ``"Excellent quality NMR ensemble"`` to
            ``"Ensemble may need refinement"``.
        """
        if self.precision == "HIGH" and self.pct_well_defined > _WELL_DEFINED_PCT_HIGH:
            return "Excellent quality NMR ensemble"
        if self.precision == "GOOD" and self.pct_well_defined > _WELL_DEFINED_PCT_GOOD:
            return "Good quality NMR ensemble"
        if self.precision == "MODERATE" or self.pct_well_defined > _WELL_DEFINED_PCT_MODERATE:
            return "Acceptable quality NMR ensemble"
        return "Ensemble may need refinement"

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise all fields to a plain dictionary.

        Useful for JSON export, backward compatibility with code that
        previously consumed ``Dict[str, float]`` from ensemble analysers,
        and for persistence.

        Returns:
            Dictionary containing every field of this dataclass.

        Examples:
            >>> stats = EnsembleStatistics(n_models=5, n_residues=30)
            >>> d = stats.to_dict()
            >>> d["n_models"]
            5
            >>> EnsembleStatistics.from_dict(d) == stats
            True
        """
        return {
            "n_models": self.n_models,
            "n_residues": self.n_residues,
            "mean_pairwise_rmsd": self.mean_pairwise_rmsd,
            "median_pairwise_rmsd": self.median_pairwise_rmsd,
            "std_pairwise_rmsd": self.std_pairwise_rmsd,
            "min_pairwise_rmsd": self.min_pairwise_rmsd,
            "max_pairwise_rmsd": self.max_pairwise_rmsd,
            "medoid_index": self.medoid_index,
            "medoid_mean_rmsd": self.medoid_mean_rmsd,
            "rmsd_to_mean": self.rmsd_to_mean,
            "mean_rmsf": self.mean_rmsf,
            "max_rmsf": self.max_rmsf,
            "well_defined_residues": self.well_defined_residues,
            "pct_well_defined": self.pct_well_defined,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnsembleStatistics":
        """
        Construct an instance from a plain dictionary.

        Extra keys in ``data`` are silently ignored, making this robust
        to version skew when deserialising persisted results.

        Args:
            data: Dictionary produced by :meth:`to_dict` (or any compatible
                mapping). Must contain at least ``n_models`` and
                ``n_residues``.

        Returns:
            New :class:`EnsembleStatistics` instance.

        Raises:
            KeyError: If ``n_models`` or ``n_residues`` are absent.

        Examples:
            >>> d = {"n_models": 20, "n_residues": 91, "rmsd_to_mean": 0.68}
            >>> stats = EnsembleStatistics.from_dict(d)
            >>> stats.precision
            'HIGH'
        """
        return cls(
            n_models=data["n_models"],
            n_residues=data["n_residues"],
            mean_pairwise_rmsd=data.get("mean_pairwise_rmsd", 0.0),
            median_pairwise_rmsd=data.get("median_pairwise_rmsd", 0.0),
            std_pairwise_rmsd=data.get("std_pairwise_rmsd", 0.0),
            min_pairwise_rmsd=data.get("min_pairwise_rmsd", 0.0),
            max_pairwise_rmsd=data.get("max_pairwise_rmsd", 0.0),
            medoid_index=data.get("medoid_index", 0),
            medoid_mean_rmsd=data.get("medoid_mean_rmsd", 0.0),
            rmsd_to_mean=data.get("rmsd_to_mean", 0.0),
            mean_rmsf=data.get("mean_rmsf", 0.0),
            max_rmsf=data.get("max_rmsf", 0.0),
            well_defined_residues=data.get("well_defined_residues", 0),
            pct_well_defined=data.get("pct_well_defined", 0.0),
        )

    def __str__(self) -> str:
        """Format statistics as a concise human-readable summary."""
        lines = [
            f"Ensemble Statistics ({self.n_models} models, {self.n_residues} residues)",
            f"  Precision      : {self.precision}",
            f"  RMSD to mean   : {self.rmsd_to_mean:.2f} Å",
            f"  Mean pw-RMSD   : {self.mean_pairwise_rmsd:.2f} Å",
            f"  Well-defined   : {self.pct_well_defined:.1f}%"
            f" ({self.well_defined_residues}/{self.n_residues} residues)",
            f"  Overall quality: {self.overall_quality}",
        ]
        return "\n".join(lines)


@dataclass
class QualityAssessment:
    """
    Interpretive quality assessment for an NMR ensemble.

    A lightweight companion to :class:`EnsembleStatistics` that carries the
    human-readable precision tier and overall quality string.

    Attributes:
        precision: Precision tier — ``"HIGH"``, ``"GOOD"``, ``"MODERATE"``,
            or ``"UNKNOWN"``.
        overall: Free-text overall quality description.

    Examples:
        >>> from synth_pdb.ensemble.statistics import QualityAssessment
        >>> qa = QualityAssessment(precision="HIGH",
        ...                        overall="Excellent quality NMR ensemble")
        >>> qa.to_dict()
        {'precision': 'HIGH', 'overall': 'Excellent quality NMR ensemble'}
    """

    precision: str
    overall: str

    def to_dict(self) -> dict[str, str]:
        """
        Serialise to a plain dictionary.

        Returns:
            ``{"precision": ..., "overall": ...}``
        """
        return {"precision": self.precision, "overall": self.overall}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "QualityAssessment":
        """
        Construct from a plain dictionary.

        Args:
            data: Must contain ``"precision"`` and ``"overall"`` keys.

        Returns:
            New :class:`QualityAssessment` instance.

        Raises:
            KeyError: If required keys are absent.
        """
        return cls(precision=data["precision"], overall=data["overall"])
