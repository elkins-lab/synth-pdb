#!/usr/bin/env python3

"""CLI entry point for the synth-pdb tool.

This module provides the main() function that serves as the command-line interface
for generating PDB files.
"""

import argparse
import datetime
import logging
import os
import sys
from typing import Any, cast

import numpy as np

from .decoys import DecoyGenerator
from .docking import DockingPrep
from .generator import generate_pdb_content
from .pdb_utils import assemble_pdb_content, extract_atomic_content, extract_header_records
from .validator import PDBValidator
from .viewer import view_structure_in_browser

# Get a logger for this module
logger = logging.getLogger(__name__)


class CLIFormatter(logging.Formatter):
    """Custom formatter to only show level name for WARNING and above."""

    def format(self, record: logging.LogRecord) -> str:
        # Use the standard formatting logic to resolve %s and other placeholders
        formatted_msg = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {formatted_msg}"
        return formatted_msg


def _build_command_string(args: argparse.Namespace) -> str:
    """Build a command string from parsed arguments for PDB header."""
    cmd_parts = ["synth-pdb"]
    if args.sequence:
        cmd_parts.append(f"--sequence {args.sequence}")
    else:
        cmd_parts.append(f"--length {args.length}")

    if args.plausible_frequencies:
        cmd_parts.append("--plausible-frequencies")
    if hasattr(args, "seed") and args.seed is not None:
        cmd_parts.append(f"--seed {args.seed}")

    if args.conformation != "alpha":  # Only add if not default
        cmd_parts.append(f"--conformation {args.conformation}")
    if hasattr(args, "structure") and args.structure:  # NEW: add structure if provided
        cmd_parts.append(f"--structure '{args.structure}'")
    if args.validate:
        cmd_parts.append("--validate")
    if args.guarantee_valid:
        cmd_parts.append("--guarantee-valid")
        cmd_parts.append(f"--max-attempts {args.max_attempts}")
    if args.best_of_N > 1:
        cmd_parts.append(f"--best-of-N {args.best_of_N}")
    if args.refine_clashes > 0:
        cmd_parts.append(f"--refine-clashes {args.refine_clashes}")
    if args.optimize:
        cmd_parts.append("--optimize")
    if args.minimize:
        cmd_parts.append("--minimize")
        cmd_parts.append(f"--forcefield {args.forcefield}")
        # Phase 1: Hardware Acceleration Overrides
        if getattr(args, "platform", None):
            cmd_parts.append(f"--platform {args.platform}")
        if getattr(args, "precision", None):
            cmd_parts.append(f"--precision {args.precision}")

    if args.cyclic:
        cmd_parts.append("--cyclic")

    # Phase 7/8/9 flags
    if args.gen_nef:
        cmd_parts.append("--gen-nef")
        cmd_parts.append(f"--noe-cutoff {args.noe_cutoff}")
        if args.nef_output:
            cmd_parts.append(f"--nef-output {args.nef_output}")

    if args.gen_relax:
        cmd_parts.append("--gen-relax")
        cmd_parts.append(f"--field {args.field}")
        cmd_parts.append(f"--tumbling-time {args.tumbling_time}")

    if args.gen_shifts:
        cmd_parts.append("--gen-shifts")
        if args.shift_output:
            cmd_parts.append(f"--shift-output {args.shift_output}")
        # Record the chosen predictor so the REMARK block captures it for reproducibility.
        # EDUCATIONAL NOTE - Scientific Reproducibility:
        # The PDB REMARK 3 block records all generation parameters.
        # Researchers reading a synth-pdb file can therefore re-run the exact same
        # command and obtain an identical structure, satisfying the FAIR data principles
        # (Findable, Accessible, Interoperable, Reusable).
        # We use getattr with a default so that bare argparse.Namespace objects created by
        # older tests that predate this flag do not raise AttributeError.
        shift_predictor = getattr(args, "shift_predictor", "shiftx2")
        if shift_predictor != "shiftx2":  # Only add if non-default to keep REMARKs concise
            cmd_parts.append(f"--shift-predictor {shift_predictor}")

    if getattr(args, "gen_cd", False):
        cmd_parts.append("--gen-cd")

    # Use getattr for all new Phase 9.6 attributes for backward compatibility with code
    # that creates bare Namespace objects (e.g. existing test_main_coverage.py tests).
    output_rdcs = getattr(args, "output_rdcs", None)
    if output_rdcs:
        cmd_parts.append(f"--output-rdcs {output_rdcs}")
        rdc_da = getattr(args, "rdc_da", 10.0)
        rdc_r = getattr(args, "rdc_r", 0.1)
        if rdc_da != 10.0:
            cmd_parts.append(f"--rdc-da {rdc_da}")
        if rdc_r != 0.1:
            cmd_parts.append(f"--rdc-r {rdc_r}")

    if args.output:
        cmd_parts.append(f"--output {args.output}")

    return " ".join(cmd_parts)


def main() -> None:
    """Main function to parse arguments and generate the PDB file."""
    parser = argparse.ArgumentParser(
        description="Generate a PDB file with a random linear amino acid sequence."
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Length of the amino acid sequence (number of residues). Default: 10 (or inferred from --structure if provided).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Output filename. If not provided, a default name will be generated.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdb",
        choices=["pdb", "cif", "bcif"],
        help="Output file format (pdb, cif, bcif). Default: pdb.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        const="",
        help="Natural language prompt to generate structures using an LLM.",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="local",
        choices=["local", "openai"],
        help="Backend to use for --prompt. 'local' downloads and runs a model locally. 'openai' requires OPENAI_API_KEY.",
    )

    parser.add_argument(
        "--sequence",
        type=str,
        help="Specify an amino acid sequence (e.g., 'AGV' or 'ALA-GLY-VAL'). Use ':' to separate multiple chains (e.g., 'ALA-GLY:SER-VAL'). Overrides random generation.",
    )
    parser.add_argument(
        "--plausible-frequencies",
        action="store_true",
        help="Use biologically plausible amino acid frequencies for random sequence generation (ignored if --sequence is provided).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks (bond lengths and angles, Ramachandran) on the generated PDB.",
    )
    parser.add_argument(
        "--guarantee-valid",
        action="store_true",
        help="If set, repeatedly generate PDB files until a valid one (no violations) is produced. Implies --validate. Will stop after --max-attempts if no valid PDB is found.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100,
        help="Maximum number of regeneration attempts when --guarantee-valid is set.",
    )
    parser.add_argument(
        "--best-of-N",
        type=int,
        default=1,
        help="Generate N PDBs, validate each, and select the one with the fewest violations. Implies --validate. Overrides --guarantee-valid.",
    )
    parser.add_argument(
        "--refine-clashes",
        type=int,
        default=0,  # Default to 0, meaning no refinement
        help="Number of iterations to refine generated PDB by minimally adjusting clashing atoms. Implies --validate. Applied after --guarantee-valid or --best-of-N selection.",
    )
    parser.add_argument(
        "--conformation",
        type=str,
        default="alpha",
        choices=["alpha", "beta", "ppii", "polyproline", "extended", "random"],
        help="Secondary structure conformation to generate. Options: alpha (default, alpha helix), beta (beta sheet), ppii (polyproline II), polyproline (alias for ppii), extended (stretched), random (random sampling).",
    )
    parser.add_argument(
        "--metal-ions",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="Mechanism for handling metal cofactors (e.g. Zinc). 'auto' (default) scans for binding motifs and inserts ions. 'none' disables this.",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default=None,
        help="Per-region conformation specification. Format: 'start-end:conformation,...'. Supports secondary structures (alpha, beta) and Turn types (typeI, typeII, typeVIII). Example: '1-10:alpha,11-14:typeII,15-20:beta'.",
    )
    parser.add_argument(
        "--restraints",
        type=str,
        help="Optional: Path to an NMR NOE restraint file (.nef or .restraints) for RPF validation.",
    )
    parser.add_argument(
        "--bmrb-id",
        type=str,
        help="Optional: BMRB ID to fetch experimental NOE restraints for validation.",
    )
    parser.add_argument(
        "--rdc-restraints",
        type=str,
        help="Optional: Path to an NMR RDC restraint file for Q-factor validation.",
    )
    parser.add_argument(
        "--shift-restraints",
        type=str,
        help="Optional: Path to a chemical shift restraint file for validation.",
    )
    parser.add_argument(
        "--scorecard",
        action="store_true",
        help="Generate a comprehensive Integrated Scientific Defense Scorecard (Physics + ML + NMR + Interface).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open generated structure in browser-based 3D viewer (uses 3Dmol.js). Interactive visualization with rotation, zoom, and style controls.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Monte Carlo side-chain optimization to minimize steric clashes (Advanced).",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Run physics-based energy minimization using OpenMM (Phase 2). Requires 'openmm' installed.",
    )
    parser.add_argument(
        "--cyclic",
        action="store_true",
        help="Generate a head-to-tail cyclic peptide. Implies --minimize and disables --cap-termini.",
    )
    parser.add_argument(
        "--forcefield",
        type=str,
        default="amber14-all.xml",
        help="Forcefield to use for minimization (default: amber14-all.xml).",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        choices=["CUDA", "Metal", "OpenCL", "CPU", "Reference"],
        help="Explicit OpenMM platform for hardware acceleration (default: auto-detect).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["single", "mixed", "double"],
        help="Numerical precision for GPU platforms (default: mixed for CUDA/OpenCL).",
    )

    parser.add_argument(
        "--solvent",
        type=str,
        default="obc2",
        choices=["obc2", "obc1", "gbn", "gbn2", "hct", "explicit"],
        help="Solvent model for energy minimization and MD.",
    )
    parser.add_argument(
        "--solvent-padding",
        type=float,
        default=1.0,
        help="Padding distance (nm) for the explicit water box (default: 1.0).",
    )
    parser.add_argument(
        "--keep-solvent",
        action="store_true",
        help="Retain explicit water (HOH) molecules in the final PDB. Default is to strip them.",
    )

    # Phase 17: Cryo-EM Density Maps
    parser.add_argument(
        "--resolution",
        type=float,
        default=3.0,
        help="Target resolution (Angstroms) for Cryo-EM density maps (for --mode cryo-em). Default 3.0A.",
    )
    parser.add_argument(
        "--mrc-output",
        type=str,
        help="Optional: Output MRC/CCP4 filename for Cryo-EM maps.",
    )

    # Phase 18: SAXS Curves
    parser.add_argument(
        "--q-max",
        type=float,
        default=0.5,
        help="Maximum scattering vector q (A^-1) for SAXS profiles (for --mode saxs). Default 0.5.",
    )
    parser.add_argument(
        "--saxs-output",
        type=str,
        help="Optional: Output .dat filename for synthetic SAXS profiles.",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="standard",
        choices=["standard", "kratky", "guinier", "all"],
        help="Type of SAXS plot to generate with --visualize. Options: 'standard', 'kratky', 'guinier', 'all'.",
    )

    # Phase 3: Research Utilities Arguments
    # Using 'mode' argument to distinguish workflows without breaking BC (default is 'generate')
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "decoys", "docking", "pymol", "dataset", "ai", "cryo-em", "saxs"],
        help="Operation mode: 'generate' (default) single structure, 'decoys' ensemble, 'docking' preparation (PQR), 'pymol' visualization script, 'dataset' bulk generation, 'ai' structure interpolation/clustering, 'cryo-em' density map generation, 'saxs' scattering curve simulation.",
    )
    parser.add_argument(
        "--n-decoys",
        type=int,
        default=10,
        help="Number of decoys to generate (for --mode decoys).",
    )
    parser.add_argument(
        "--rmsd-range",
        type=str,
        default="0.0-999.0",
        help="Target RMSD range in Angstroms 'min-max' (for --mode decoys).",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Enable 'hard decoy' mode (threading, shuffling, drift) for AI support.",
    )
    parser.add_argument(
        "--template-sequence",
        type=str,
        help="Sequence to use for backbone folding when threading (for --mode decoys).",
    )
    parser.add_argument(
        "--shuffle-sequence",
        action="store_true",
        help="Shuffle residue labels in the final decoy (for --mode decoys).",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.0,
        help="Maximum torsion angle drift in degrees to apply for conformational noise.",
    )
    parser.add_argument(
        "--input-pdb",
        type=str,
        help="Input PDB file path (required for --mode docking and --mode pymol).",
    )
    parser.add_argument(
        "--input-nef",
        type=str,
        help="Input NEF file path (required for --mode pymol).",
    )
    parser.add_argument(
        "--output-pml",
        type=str,
        help="Output PyMOL script path (for --mode pymol).",
    )

    # Phase 7: Synthetic NMR Data (NEF)
    parser.add_argument(
        "--gen-nef",
        action="store_true",
        help="Generate synthetic NMR data (NOE restraints) in NEF format.",
    )
    parser.add_argument(
        "--noe-cutoff",
        type=float,
        default=5.0,
        help="Distance cutoff (Angstroms) for synthetic NOEs (default 5.0).",
    )
    parser.add_argument(
        "--nef-output",
        type=str,
        help="Optional: Output NEF filename.",
    )
    parser.add_argument(
        "--gen-pymol",
        action="store_true",
        help="Generate a PyMOL script (.pml) to visualize the synthetic NEF restraints on the structure.",
    )

    # Phase 8: Synthetic Relaxation Data
    parser.add_argument(
        "--gen-relax",
        action="store_true",
        help="Generate synthetic NMR relaxation data (R1, R2, NOE) in NEF format.",
    )
    parser.add_argument(
        "--field",
        type=float,
        default=600.0,
        help="Proton Larmor frequency in MHz for relaxation calculation (default 600.0).",
    )
    parser.add_argument(
        "--tumbling-time",
        type=float,
        default=10.0,
        help="Global rotational correlation time (tau_m) in nanoseconds (default 10.0).",
    )

    # Phase 9: Synthetic Chemical Shifts
    parser.add_argument(
        "--gen-shifts",
        action="store_true",
        help="Generate synthetic Chemical Shift data (H, N, CA, CB, C) based on secondary structure.",
    )
    parser.add_argument(
        "--shift-output",
        type=str,
        help="Optional: Output NEF filename for chemical shifts.",
    )
    parser.add_argument(
        "--shift-predictor",
        type=str,
        default="shiftx2",
        choices=["shiftx2", "empirical"],
        help=(
            "Chemical shift predictor to use with --gen-shifts. "
            "EDUCATIONAL NOTE - Predictor Selection: "
            "'shiftx2' (default): prefers the SHIFTX2 external binary (Han et al., 2011, "
            "J Biomol NMR 50:43), falls back automatically to the empirical method if the "
            "SHIFTX2 binary is not installed. "
            "'empirical': uses the SPARTA+-style empirical table method directly (Shen & Bax, "
            "2010, J Biomol NMR 48:13), always available with no external dependencies. "
            "Use 'empirical' for reproducible CI/CD pipelines."
        ),
    )

    # Phase 9.5: J-Couplings
    parser.add_argument(
        "--gen-couplings",
        action="store_true",
        help="Generate synthetic 3J(HN-HA) scalar couplings based on phi angles.",
    )
    parser.add_argument(
        "--coupling-output",
        type=str,
        help="Optional: Output CSV filename for J-couplings.",
    )

    # Phase 9.6: Residual Dipolar Couplings (RDCs)
    # EDUCATIONAL NOTE - RDC Background:
    # RDCs encode the orientation of bondvectors (e.g., backbone N-H) relative to a
    # global alignment frame. They are fundamentally different from NOE distance
    # restraints: NOEs are local (short-range r^-6 distance contacts) while RDCs are
    # global (orientational information vs. the alignment tensor). The two observables
    # are therefore complementary - combining them dramatically improves NMR structure
    # accuracy. A groundbreaking study showed RDC-constrained calculations refined a
    # 100-ns MD ensemble to within 0.4 A of the crystal structure without additional NOEs
    # (Bewley & Clore, 2000, J Am Chem Soc, 122, 6009).
    #
    # The RDC formula (Tjandra & Bax, 1997, Science 278:1111):
    #   D(theta, phi) = Da * [(3cos^2theta - 1) + (3/2)*R*sin^2theta*cos(2phi)]
    # where Da is the axial component and R is the rhombicity of the alignment tensor.
    parser.add_argument(
        "--output-rdcs",
        type=str,
        help=(
            "Generate synthetic backbone N-H Residual Dipolar Coupling (RDC) data and "
            "export to a CSV file (columns: res_id, residue, RDC_NH_Hz). "
            "The structure must contain amide H atoms - use --minimize to add protons "
            "via OpenMM. Alignment tensor defaults: Da=10 Hz, R=0.1. "
            "Based on: Tjandra & Bax (1997), Science 278:1111."
        ),
    )
    parser.add_argument(
        "--rdc-da",
        type=float,
        default=10.0,
        help=(
            "Axial component of the alignment tensor Da in Hz (default: 10.0). "
            "Da controls the overall magnitude of the RDC values. "
            "Typical experimental range: 5-25 Hz for dilute liquid crystal or "
            "phage-based alignment media (Tjandra & Bax, 1997; Hansen et al., 2000, "
            "J Biomol NMR 14:85)."
        ),
    )
    parser.add_argument(
        "--rdc-r",
        type=float,
        default=0.1,
        help=(
            "Rhombicity R of the alignment tensor, 0 <= R <= 2/3 (default: 0.1). "
            "R=0 = axially symmetric tensor (simplest case; measurement in rod-like media). "
            "R=2/3 = maximum rhombicity. "
            "Increasing R breaks the degeneracy between bond vectors related by rotation "
            "about the tensor Z-axis, providing additional orientational information. "
            "Reference: Clore et al. (1998), J Magn Reson, 133, 216-221."
        ),
    )

    # Phase 9.7: Circular Dichroism (CD)
    parser.add_argument(
        "--gen-cd",
        action="store_true",
        help="Generate synthetic Circular Dichroism (CD) spectrum based on secondary structure.",
    )

    # Phase 10: Constraint Export
    parser.add_argument(
        "--export-constraints",
        type=str,
        help="Export contact map constraints for modeling (e.g. AlphaFold, CASP). Specify output filename.",
    )
    parser.add_argument(
        "--constraint-format",
        type=str,
        default="casp",
        choices=["casp", "csv"],
        help="Format for constraint export (default: casp). casp=RR format.",
    )
    parser.add_argument(
        "--constraint-cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff (Angstroms) for binary contacts (default: 8.0).",
    )

    # Phase 11: Torsion Export
    parser.add_argument(
        "--export-torsion",
        type=str,
        help="Export backbone torsion angles (Phi, Psi, Omega) to a file. Specify output filename.",
    )
    parser.add_argument(
        "--torsion-format",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Format for torsion angle export (default: csv).",
    )

    # Phase 12: Synthetic MSA (Evolution)
    parser.add_argument(
        "--gen-msa",
        action="store_true",
        help="Generate synthetic Multiple Sequence Alignment (MSA) via simulated evolution.",
    )
    parser.add_argument(
        "--msa-depth",
        type=int,
        default=100,
        help="Number of sequences to generate for MSA (default: 100).",
    )
    parser.add_argument(
        "--evolution-temp",
        type=float,
        default=1.5,
        help="Thermal Noise of MSA MCMC evolution (default: 1.5). Higher means more sequence divergence.",
    )

    # Phase 13: Distogram Export (AI Trinity #3)
    parser.add_argument(
        "--export-distogram",
        type=str,
        help="Export NxN Distance Matrix (Distogram) to a file. Specify output filename.",
    )
    parser.add_argument(
        "--distogram-format",
        type=str,
        choices=["json", "csv", "npz"],
        default="json",
        help="Format for distogram export (default: json).",
    )

    # Phase 14: Biophysical Realism (Capping & pH)
    parser.add_argument(
        "--cap-termini",
        action="store_true",
        help="Add N-terminal Acetyl (ACE) and C-terminal N-methylamide (NME) caps.",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for determining protonation states (default: 7.4). Affects Histidine (HIS/HIP/HIE).",
    )
    parser.add_argument(
        "--cis-proline-frequency",
        type=float,
        default=0.05,
        help="Probability (0.0-1.0) of Proline adopting the Cis conformation (default: 0.05).",
    )
    parser.add_argument(
        "--phosphorylation-rate",
        type=float,
        default=0.0,
        help="Probability (0.0-1.0) of Ser/Thr/Tyr phosphorylation (default: 0.0).",
    )

    # Phase 2: MD Equilibration
    parser.add_argument(
        "--equilibrate",
        action="store_true",
        help="Run Molecular Dynamics equilibration (at 300K) after minimization. Requires OpenMM.",
    )

    # Phase 15: Bulk Dataset Generation (ML)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate for the dataset (for --mode dataset). Default: 100.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum sequence length for dataset samples (for --mode dataset). Default: 10.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum sequence length for dataset samples (for --mode dataset). Default: 50.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of samples to split into training set (for --mode dataset). Default: 0.8.",
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=1000,
        help="Number of MD steps for equilibration (default: 1000 approx 2ps).",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="pdb",
        choices=["pdb", "npz"],
        help="Output format for dataset generation (default: pdb). 'npz' produces compressed arrays.",
    )

    # Phase 16: Structure Quality Filter (Random Forest classifier)
    parser.add_argument(
        "--quality-filter",
        "--include-ml",
        action="store_true",
        help="Enable Random Forest-based structure quality filtering. Rejects structures that score below the cutoff on geometric quality metrics (Ramachandran, clashes, bond lengths).",
    )
    parser.add_argument(
        "--quality-score-cutoff",
        type=float,
        default=0.5,
        help="Minimum confidence score (0.0-1.0) for structure quality filter (default: 0.5).",
    )
    parser.add_argument(
        "--ai-op",
        type=str,
        choices=["interpolate", "cluster"],
        help="Operation to perform in 'ai' mode.",
    )
    parser.add_argument(
        "--start-pdb",
        type=str,
        help="Start PDB file for interpolation.",
    )
    parser.add_argument(
        "--end-pdb",
        type=str,
        help="End PDB file for interpolation.",
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        help="Glob pattern for input PDB files (e.g., 'decoys/*.pdb') for clustering.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters to form in 'ai' clustering mode (default: 5).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps for interpolation (default: 10).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible structure generation. If set, the same command will produce the exact same PDB file.",
    )

    args = parser.parse_args()

    # Set the logging level based on user input
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")

    # Configure logging if not already configured (e.g., by pytest or another caller)
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(CLIFormatter())
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(log_level)
    logger.debug("Logging level set to %s.", args.log_level.upper())

    # Process Natural Language Prompt if provided
    if args.prompt is not None:
        if args.prompt == "":
            try:
                if sys.stdin.isatty():
                    print(
                        "Enter your natural language prompt (type 'exit' on a new line or press Ctrl+D to finish):",
                        file=sys.stderr,
                    )
                    lines = []
                    while True:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        if line.strip().lower() == "exit":
                            break
                        lines.append(line)
                    args.prompt = "".join(lines).strip()
                else:
                    args.prompt = sys.stdin.read().strip()

                if args.prompt:
                    logger.debug(f"Read prompt from stdin: {args.prompt}")
            except EOFError:
                pass

        if args.prompt:
            try:
                from .llm import LLMInterface

                llm = LLMInterface(backend=args.llm_backend)
                llm_args = llm.translate_prompt(args.prompt)

                # Inform the user what the prompt was translated into
                if llm_args:
                    args_str = " ".join(f"--{k} {v}" for k, v in llm_args.items())
                    logger.info(f"Translated prompt into: {args_str}")
                else:
                    logger.warning(
                        "No specific structural instructions identified in prompt. Using defaults."
                    )

                for key, value in llm_args.items():
                    setattr(args, key, value)
            except Exception as e:
                print(f"Failed to process natural language prompt: {e}", file=sys.stderr)
                sys.exit(1)

    if args.prompt:
        logger.info("Successfully translated prompt into command-line arguments.")

    logger.info("Starting PDB file generation process.")
    logger.debug(
        "Parsed arguments: length=%s, output='%s', sequence='%s', plausible_frequencies=%s, validate=%s",
        args.length,
        args.output,
        args.sequence,
        args.plausible_frequencies,
        args.validate,
    )

    # If --best-of-N is set, it overrides --guarantee-valid and implies --validate.
    if args.best_of_N > 1:
        args.validate = True
        args.guarantee_valid = False  # Disable guarantee-valid if best-of-N is used
        logger.info(
            f"--best-of-N is set to {args.best_of_N}. Generating multiple PDBs to find the one with fewest violations."
        )
    elif args.guarantee_valid:  # Only apply if best-of-N is not active
        args.validate = True
        logger.info("--guarantee-valid is set. Will attempt to generate a valid PDB.")

    if args.refine_clashes > 0:
        args.validate = True  # Refinement implies validation during initial generation
        logger.info(
            f"--refine-clashes is set to {args.refine_clashes}. Validation will be performed."
        )

    if args.cyclic:
        logger.info(
            "--cyclic is set. Enabling energy minimization for ring closure and disabling terminal caps."
        )
        args.minimize = True
        args.cap_termini = False

    # Validate length only if no sequence is provided
    if args.sequence is None and args.length is not None:
        if args.length <= 0:
            logger.error("Length must be a positive integer.")
            sys.exit(1)

    # Dispatch to specific modes if not generating a new structure
    if args.mode == "docking":
        if args.input_pdb:
            prep = DockingPrep(args.forcefield)
            pqr_file = args.output if args.output else "docking_prep.pqr"
            success = prep.write_pqr(args.input_pdb, pqr_file)
            if success:
                logger.info(f"Docking preparation complete. PQR file: {pqr_file}")
            else:
                logger.error("Docking preparation failed.")
                sys.exit(1)
            return
        elif not args.sequence and args.length is None and not args.structure:
            logger.error("Docking mode requires --input-pdb.")
            sys.exit(1)
        else:
            logger.info("Docking mode: No input PDB provided. Generating new structure first...")

    # Set default length if not provided and no sequence
    if args.sequence is None and args.length is None:
        # Check if we can infer length from structure parameter
        if args.structure:
            # Parse structure to find maximum residue number
            try:
                max_residue = 0
                for region in args.structure.split(","):
                    region = region.strip()
                    if ":" in region:
                        range_part = region.split(":", 1)[0]
                        if "-" in range_part:
                            _, end_str = range_part.split("-", 1)
                            end = int(end_str)
                            max_residue = max(max_residue, end)

                if max_residue > 0:
                    args.length = max_residue
                    logger.info(f"Inferred length={max_residue} from --structure parameter")
                else:
                    logger.error("Could not infer length from --structure parameter")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to parse --structure parameter: {e}")
                sys.exit(1)
        else:
            # No structure parameter, use default length of 10
            args.length = 10
            logger.debug("Using default length=10")

    if args.mode == "pymol":
        if not args.input_pdb or not args.input_nef or not args.output_pml:
            logger.error("PyMOL mode requires --input-pdb, --input-nef, and --output-pml.")
            sys.exit(1)

        from .nef_io import read_nef_restraints
        from .visualization import generate_pymol_script

        try:
            # Read restraints first (the function now expects a list)
            restraints = read_nef_restraints(args.input_nef)
            generate_pymol_script(args.input_pdb, restraints, args.output_pml)
            logger.info(f"PyMOL script generated successfully: {args.output_pml}")
        except Exception as e:
            logger.error(f"Failed to generate PyMOL script: {e}")
            sys.exit(1)
        return  # Exit after visualization generation

    if args.mode == "decoys":
        if not args.sequence and (args.length is None or args.length <= 0):
            logger.error("Decoy generation requires --sequence or a positive --length.")
            sys.exit(1)
            return

        target_sequence: Any = args.sequence
        if not target_sequence:
            # Generate random sequence if not provided
            from .generator import _get_random_sequence
            import random as rnd_local

            rng_local = rnd_local.Random(args.seed)

            res_list = _get_random_sequence(
                args.length or 10, args.plausible_frequencies, rng=rng_local
            )
            # target_sequence is passed to generate_ensemble
            target_sequence = res_list
            logger.info(f"Generated random sequence for decoys: {'-'.join(res_list)}")

        logger.info(f"Starting Decoy Ensemble Generation for: {target_sequence}")
        # Parse RMSD range
        rmsd_min, rmsd_max = 0.0, 999.0
        if args.rmsd_range:
            if "-" not in args.rmsd_range:
                logger.error(
                    f"Invalid RMSD range: {args.rmsd_range}. Use format MIN-MAX (e.g. 2.0-5.0)"
                )
                sys.exit(1)
                return
            try:
                min_s, max_s = args.rmsd_range.split("-")
                rmsd_min, rmsd_max = float(min_s), float(max_s)
            except ValueError:
                logger.error(
                    f"Invalid RMSD range: {args.rmsd_range}. Use format MIN-MAX (e.g. 2.0-5.0)"
                )
                sys.exit(1)
                return

        dg = DecoyGenerator()
        ensemble = dg.generate_ensemble(
            sequence=target_sequence,  # Pass original (list or str) to satisfy tests
            n_decoys=args.n_decoys,
            out_dir=args.output or "decoys",
            rmsd_min=rmsd_min,
            rmsd_max=rmsd_max,
            hard_mode=args.hard,
            template_sequence=args.template_sequence,
            shuffle_sequence=args.shuffle_sequence,
            drift=args.drift,
            seed=args.seed,
            optimize=args.optimize,
            minimize=args.minimize,
            forcefield=args.forcefield,
        )
        logger.info(f"Generated {len(ensemble)} decoys within {args.rmsd_range} A RMSD.")
        return

    if args.mode == "dataset":
        from .dataset import generate_balanced_dataset

        logger.info("Starting Bulk Dataset Generation...")
        generate_balanced_dataset(
            num_samples=args.num_samples,
            min_length=args.min_length,
            max_length=args.max_length,
            output_dir=args.output or "synth_dataset",
            train_ratio=args.train_ratio,
            output_format=args.dataset_format,
            seed=args.seed,
        )
        logger.info("Dataset generation complete.")
        return

    if args.mode == "ai":
        if args.ai_op == "interpolate":
            if not args.start_pdb or not args.end_pdb:
                logger.error("Interpolation requires --start-pdb and --end-pdb.")
                sys.exit(1)
                return
            from .quality.interpolate import interpolate_structures

            logger.info(f"Interpolating structures between {args.start_pdb} and {args.end_pdb}...")
            # signature: (start_pdb_path: str, end_pdb_path: str, steps: int, output_prefix: str)
            interpolate_structures(
                args.start_pdb, args.end_pdb, args.steps, args.output or "interpolation"
            )
            logger.info("Interpolation complete.")
            return

        elif args.ai_op == "cluster":
            if not args.input_pattern:
                logger.error("Clustering requires --input-pattern.")
                sys.exit(1)
                return
            from .quality.cluster import cluster_structures

            logger.info(f"Clustering structures matching pattern: {args.input_pattern}...")
            # signature: (input_pattern: str, n_clusters: int, output_dir: str, random_seed: int = 42)
            # Returns None, saves files to output_dir
            cluster_structures(
                args.input_pattern,
                args.n_clusters,
                args.output or "clusters",
                random_seed=args.seed if args.seed is not None else 42,
            )
            return
        else:
            logger.error("AI mode requires --ai-op {interpolate, cluster}")
            sys.exit(1)
            return

    if args.mode == "cryo-em":
        if not args.sequence and (args.length is None or args.length <= 0):
            logger.error("Cryo-EM mode requires --sequence or a positive --length.")
            sys.exit(1)
            return

        from .batch_generator import BatchedGenerator
        from .cryo_em import generate_density_map, save_mrc_file

        target_sequence = args.sequence
        if not target_sequence:
            from .generator import _get_random_sequence
            import random as rnd_local

            rng_local = rnd_local.Random(args.seed)

            res_list = _get_random_sequence(
                args.length or 10, args.plausible_frequencies, rng=rng_local
            )
            target_sequence = "-".join(res_list)

        logger.info(f"Generating Cryo-EM density map for sequence: {target_sequence}")

        # 1. Generate Ensemble (Ensemble-averaging makes for more realistic maps)
        bg = BatchedGenerator(target_sequence, n_batch=args.n_decoys, full_atom=True)
        batch = bg.generate_batch(seed=args.seed)
        stack = batch.to_stack()

        # 2. Simulate Map
        # default grid spacing 1.0
        grid, origin = generate_density_map(stack, resolution=args.resolution, grid_spacing=1.0)

        # 3. Save MRC
        mrc_file = args.mrc_output or "synthetic_density.mrc"
        save_mrc_file(mrc_file, grid, origin, spacing=1.0)
        logger.info(f"Cryo-EM density map saved to {mrc_file} at {args.resolution}A resolution.")

        # 4. Optional Visualization (Not yet implemented for maps)
        if args.visualize:
            logger.warning(
                "Density map visualization is not yet implemented in the browser viewer."
            )
        return

    if args.mode == "saxs":
        if not args.sequence and (args.length is None or args.length <= 0):
            logger.error("SAXS mode requires --sequence or a positive --length.")
            sys.exit(1)
            return

        from .batch_generator import BatchedGenerator
        from .saxs import (
            calculate_radius_of_gyration,
            calculate_saxs_profile,
            export_saxs_profile,
        )

        target_sequence = args.sequence
        if not target_sequence:
            from .generator import _get_random_sequence
            import random as rnd_local

            rng_local = rnd_local.Random(args.seed)

            res_list = _get_random_sequence(
                args.length or 10, args.plausible_frequencies, rng=rng_local
            )
            target_sequence = "-".join(res_list)

        logger.info(f"Generating SAXS profile for sequence: {target_sequence}")

        # 1. Generate Ensemble
        bg = BatchedGenerator(target_sequence, n_batch=args.n_decoys, full_atom=True)
        batch = bg.generate_batch(drift=args.drift or 3.0, seed=args.seed)

        # 2. Simulate SAXS (Averaged over ensemble)
        stack = batch.to_stack()
        q_vals = np.linspace(0.0, args.q_max, 51)
        all_intensities = []

        try:
            for i in range(len(stack)):
                _, intensity = calculate_saxs_profile(stack[i], q_max=args.q_max)
                all_intensities.append(intensity)

            if all_intensities:
                avg_intensity = np.mean(all_intensities, axis=0)
                out_file = args.saxs_output or "synthetic_saxs.dat"
                export_saxs_profile(q_vals, avg_intensity, out_file)
                logger.info(f"SAXS simulation complete. Profile saved to {out_file}")
            else:
                logger.error("SAXS simulation failed: No intensities calculated.")
                sys.exit(1)

            # 3. Optional Visualization
            if args.visualize:
                from .visualization_saxs import plot_saxs_results

                # Calculate average Rg for annotation if visualizing
                all_rg = [calculate_radius_of_gyration(stack[i]) for i in range(len(stack))]
                avg_rg = float(np.mean(all_rg))

                plot_file = out_file.replace(".dat", ".png")
                plot_saxs_results(
                    q_vals,
                    avg_intensity,
                    title=f"Synthetic SAXS ({target_sequence})",
                    output_path=plot_file,
                    plot_type=args.plot_type,
                    rg=avg_rg,
                )

                # Try to show the plot if in an interactive environment
                try:
                    import matplotlib.pyplot as plt

                    if plt.get_backend() != "agg":
                        plt.show()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"SAXS simulation failed: {e}")
            sys.exit(1)

        return

    length_for_generator = args.length if args.sequence is None else None

    final_content: str | bytes | None = None
    final_violations: list[str] = []
    min_violations_count = float("inf")

    generation_attempts = (
        1 if not args.guarantee_valid and args.best_of_N <= 1 else args.max_attempts
    )
    if args.best_of_N > 1:
        generation_attempts = args.best_of_N

    internal_format = "pdb" if args.format == "pdb" else "cif"
    for attempt_num in range(1, generation_attempts + 1):
        logger.info(f"Generation attempt {attempt_num}/{generation_attempts}.")
        current_content: str | bytes = ""
        current_violations = []

        try:
            # EDUCATIONAL NOTE - Polyglot Internal Pipeline:
            # While synth-pdb supports exporting to modern formats like mmCIF,
            # we must ensure that our internal validation and refinement loop
            # doesn't hit legacy PDB limits (like the +/-1000 A coordinate wall).
            # We use CIF as an internal intermediate if the final format is
            # non-PDB, and perform validation directly on the geometric AtomArray.
            generated_content = generate_pdb_content(
                length=length_for_generator,
                sequence_str=args.sequence,
                use_plausible_frequencies=args.plausible_frequencies,
                conformation=args.conformation,
                structure=args.structure,  # NEW: per-region conformation support
                optimize_sidechains=args.optimize,
                minimize_energy=args.minimize,
                forcefield=args.forcefield,
                solvent_model=args.solvent,
                solvent_padding=args.solvent_padding,
                keep_solvent=args.keep_solvent,
                seed=args.seed,
                ph=args.ph,
                cap_termini=args.cap_termini,
                equilibrate=args.equilibrate,
                equilibrate_steps=args.md_steps,
                metal_ions=args.metal_ions,
                cis_proline_frequency=args.cis_proline_frequency,
                phosphorylation_rate=args.phosphorylation_rate,
                cyclic=args.cyclic,
                platform=args.platform,
                precision=args.precision,
                output_format=internal_format,
            )
            current_content = generated_content

            if not current_content:
                logger.warning(
                    f"Failed to generate PDB content in attempt {attempt_num}. Skipping."
                )
                continue

            # -- Structural Validation ---------------------------------------
            if args.validate or args.quality_filter:
                from .generator import PeptideResult

                res = PeptideResult(current_content, format=internal_format)
                structure_obj = res.structure

                if args.validate:
                    logger.info("Performing structural validation checks...")
                    validator = PDBValidator(structure_obj)
                    validator.validate_all()
                    current_violations = validator.get_violations()
                    logger.debug(
                        f"Validator returned {len(current_violations)} violations for attempt {attempt_num}."
                    )

                if args.quality_filter:
                    try:
                        from .quality.classifier import ProteinQualityClassifier

                        classifier = ProteinQualityClassifier()
                        # Quality filter prefers PDB content for its internal parser
                        pdb_for_filter = res.pdb

                        is_good, prob, _ = classifier.predict(pdb_for_filter)

                        if prob < args.quality_score_cutoff:
                            logger.warning(
                                f"Quality Filter Reject (Attempt {attempt_num}): Score {prob:.2f} < {args.quality_score_cutoff}"
                            )
                            continue  # Retry
                        else:
                            logger.info(
                                f"Quality Filter Pass (Attempt {attempt_num}): Score {prob:.2f}"
                            )
                    except ImportError:
                        logger.warning(
                            "Quality Filter enabled but dependencies missing. Install `synth-pdb[ai]`. Skipping filter."
                        )
                    except Exception as e:
                        logger.warning(f"Quality Filter failed: {e}. Skipping.")

            if args.guarantee_valid:
                if not current_violations:
                    logger.info(
                        f"Successfully generated a valid PDB file after {attempt_num} attempts."
                    )
                    final_content = current_content
                    final_violations = current_violations
                    break  # Exit loop, valid structure found
                else:
                    logger.warning(
                        f"PDB generated in attempt {attempt_num} has {len(current_violations)} violations. Retrying..."
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("--- PDB Validation Report for failed attempt ---")
                        for violation in current_violations:
                            logger.debug(violation)
                        logger.debug("--- End Validation Report ---")
            elif args.best_of_N > 1:
                if len(current_violations) < min_violations_count:
                    min_violations_count = len(current_violations)
                    final_content = current_content
                    final_violations = current_violations
                    logger.info(
                        f"Attempt {attempt_num} yielded {len(current_violations)} violations (new minimum)."
                    )
                else:
                    logger.info(
                        f"Attempt {attempt_num} yielded {len(current_violations)} violations. Current minimum is {min_violations_count}."
                    )
            else:  # No guarantee-valid or best-of-N, just take the first one
                final_content = current_content
                final_violations = current_violations
                break

        except (ValueError, TypeError, RuntimeError, Exception) as e:
            logger.error(f"Error processing sequence during generation: {e}")
            sys.exit(1)

    # Parse structure definitions for highlighting in Viewer
    highlights = []
    if args.structure:
        try:
            # Format: "1-5:beta,6-9:typeII"
            parts = args.structure.split(",")
            for part in parts:
                if ":" in part:
                    res_range, s_type = part.split(":")
                    if "-" in res_range:
                        start_s, end_s = res_range.split("-")
                        start, end = int(start_s), int(end_s)

                        # Only highlight specific turns or interesting features
                        if (
                            "type" in s_type or "beta" in s_type and "turn" in s_type
                        ):  # e.g. typeII, beta-turn
                            highlights.append(
                                {
                                    "start": start,
                                    "end": end,
                                    "color": "purple",
                                    "style": "stick",  # Stick makes turn geometry visible
                                    "label": s_type,
                                }
                            )
                        elif "helix" in s_type or "alpha" in s_type:
                            highlights.append(
                                {
                                    "start": start,
                                    "end": end,
                                    "color": "magenta",
                                    "style": "cartoon",
                                    "label": "Alpha Helix",
                                }
                            )
        except Exception as e:
            logger.warning(f"Could not parse structure for highlighting: {e}")

    if final_content is None:
        logger.error(
            f"Failed to generate a suitable PDB file after {generation_attempts} attempts."
        )
        sys.exit(1)
        return

    # -- Internal State Management (Refinement & Header Preservation) --------
    preserved_ssbonds = None
    preserved_conects = None
    final_pdb_atomic_content = None

    if internal_format == "pdb" and final_content is not None:
        # Extract atomic content from the initially selected PDB for subsequent refinement or final assembly.
        final_pdb_atomic_content = extract_atomic_content(cast(str, final_content))

        # PRESERVE HEADER RECORDS (SSBOND)
        preserved_ssbonds = extract_header_records(cast(str, final_content), "SSBOND")
        preserved_conects = extract_header_records(cast(str, final_content), "CONECT")

        # Apply refinement if requested (Currently PDB-only)
        if args.refine_clashes > 0:
            args.validate = True  # Refinement implies validation
            logger.info(f"Starting steric clash refinement for {args.refine_clashes} iterations.")

            current_refined_atomic_content = final_pdb_atomic_content
            current_refined_violations = final_violations
            initial_violations_count = len(final_violations)

            for refine_iter in range(args.refine_clashes):
                logger.info(
                    f"Refinement iteration {refine_iter + 1}/{args.refine_clashes}. Violations: {len(current_refined_violations)}"
                )
                if not current_refined_violations:
                    logger.info("No violations remain, stopping refinement early.")
                    break

                # Parse atoms from current atomic PDB content
                parsed_atoms_for_refinement = PDBValidator._parse_pdb_atoms(
                    current_refined_atomic_content
                )

                # Apply steric clash tweak
                modified_atoms = PDBValidator._apply_steric_clash_tweak(parsed_atoms_for_refinement)

                # Use PDBValidator directly with modified atoms for validation
                temp_validator = PDBValidator(parsed_atoms=modified_atoms)
                try:
                    temp_validator.validate_all()
                    new_violations = temp_validator.get_violations()
                except Exception as e:
                    logger.debug(
                        f"Validation crashed during refinement iteration {refine_iter + 1}: {e}"
                    )
                    new_violations = current_refined_violations

                if len(new_violations) < len(current_refined_violations):
                    logger.info(
                        f"Refinement iteration {refine_iter + 1}: Reduced violations from {len(current_refined_violations)} to {len(new_violations)}."
                    )
                    # Update atomic content
                    current_refined_atomic_content = temp_validator.get_pdb_content()
                    current_refined_violations = new_violations
                else:
                    logger.info(
                        f"Refinement iteration {refine_iter + 1}: No further reduction in violations ({len(new_violations)}). Stopping refinement."
                    )
                    break

            final_pdb_atomic_content = current_refined_atomic_content
            final_violations = current_refined_violations
            if initial_violations_count > len(final_violations):
                logger.info(
                    f"Refinement process completed. Reduced total violations from {initial_violations_count} to {len(final_violations)}."
                )
            elif initial_violations_count == len(final_violations):
                logger.info(
                    f"Refinement process completed. No change in total violations ({len(final_violations)})."
                )
    else:
        # For non-PDB formats (e.g. CIF), we skip refinement for now
        if args.refine_clashes > 0:
            logger.warning("Steric clash refinement is currently only supported for PDB output.")

    # After successful generation (and optional validation)
    # -- Final Output Assembly --
    if final_content is not None:
        # Determine the sequence length for the final metadata
        final_sequence_length = args.length
        if args.sequence:
            final_sequence_length = len(args.sequence.replace("-", ""))
        elif args.length is None:
            # Infer from structure
            from .generator import PeptideResult

            res_inf = PeptideResult(final_content, format=internal_format)
            final_sequence_length = len(set(res_inf.structure.res_id))

        # Output filename generation
        if args.output:
            output_filename = args.output
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = args.format
            if args.sequence:
                sequence_tag = args.sequence.replace("-", "")[:10]
                output_filename = f"custom_peptide_{sequence_tag}_{timestamp}.{ext}"
            else:
                output_filename = f"random_linear_peptide_{args.length}_{timestamp}.{ext}"

        try:
            # -- Format-Specific Writing --
            final_to_write: str | bytes
            if args.format == "pdb":
                # For PDB, we use the standard assembler to add REMARKs
                cmd_string = _build_command_string(args)
                # We need atomic content only for assemble_pdb_content
                # If refinement was run, final_pdb_atomic_content is set.
                # Otherwise use final_content directly.
                atomic_content = (
                    final_pdb_atomic_content
                    if final_pdb_atomic_content is not None
                    else extract_atomic_content(cast(str, final_content))
                )
                final_to_write = assemble_pdb_content(
                    atomic_content,
                    final_sequence_length,
                    command_args=cmd_string,
                    extra_records=preserved_ssbonds,
                    conect_records=preserved_conects,
                )
                mode = "w"
            else:
                # For modern formats, we avoid the legacy assembler
                from .generator import PeptideResult

                res_final = PeptideResult(final_content, format=internal_format)
                final_to_write = res_final.get_content(args.format)
                mode = "wb" if isinstance(final_to_write, bytes) else "w"

            with open(output_filename, mode) as f:
                f.write(final_to_write)

            logger.info(
                f"Successfully generated {args.format.upper()} file: {os.path.abspath(output_filename)}"
            )

            # We need the structure for subsequent analytical steps
            from .generator import PeptideResult

            res_anal = PeptideResult(final_content, format=internal_format)
            structure = res_anal.structure

            # 1. Scorecard and Validation Reporting
            if args.scorecard or args.validate or final_violations:
                if final_violations:
                    logger.warning(
                        f"--- PDB Validation Report for {os.path.abspath(output_filename)} ---"
                    )
                    logger.warning(f"Final PDB has {len(final_violations)} violations.")
                    for violation in final_violations:
                        logger.warning(violation)
                    logger.warning("--- End Validation Report ---")
                elif args.validate:
                    logger.info(
                        f"No violations found in the final PDB for {os.path.abspath(output_filename)}."
                    )

                if args.scorecard:
                    logger.info("Generating Integrated Scientific Defense Scorecard...")

                    # Fetch NMR restraints if BMRB ID provided
                    input_nmr_restraints = None
                    if args.bmrb_id:
                        from .bmrb_api import BMRBAPI

                        input_nmr_restraints = BMRBAPI.fetch_restraints(args.bmrb_id)

                    validator = PDBValidator(structure)
                    report = validator.get_quality_report(
                        include_ml=args.quality_filter, nmr_restraints=input_nmr_restraints
                    )

                    print("\n" + "=" * 60)
                    print("[INFO]  INTEGRATED SCIENTIFIC DEFENSE SCORECARD")
                    print("=" * 60)

                    # Layer 1: Physics & Geometry
                    print(f"| {'PHYSICS & GEOMETRY':<35} | {'STATUS':<18} |")
                    print("-" * 60)
                    e_status = "[OK] PASS" if report["is_physically_plausible"] else "[FAIL] FAIL"
                    print(
                        f"| Potential Energy: {report['potential_energy_kj_mol']:>15.1f} kJ/mol | {e_status:<18} |"
                    )

                    z_val = report["geometric_z_scores"]["mean_bond_zscore"]
                    z_status = "[OK] PASS" if z_val < 3.0 else "[FAIL] FAIL"
                    print(f"| Mean Bond Z-Score: {z_val:>18.2f} | {z_status:<18} |")

                    ram_val = report["ramachandran_stats"]["favored_pct"]
                    ram_status = "[OK] PASS" if ram_val > 90.0 else "[WARN] WARN"
                    print(f"| Ramachandran Favored: {ram_val:>15.1f}% | {ram_status:<18} |")

                    rot_val = report["rotamer_stats"]["favored_rotamers_pct"]
                    rot_status = "[OK] PASS" if rot_val > 80.0 else "[WARN] WARN"
                    print(f"| Favored Rotamers: {rot_val:>19.1f}% | {rot_status:<18} |")

                    c_val = report["chirality_stats"]["l_amino_acid_pct"]
                    c_status = (
                        "[OK] PASS"
                        if report["chirality_stats"]["is_standard_biology"]
                        else "[ALIEN] ALIEN"
                    )
                    print(f"| Chirality (L-Biology): {c_val:>12.1f}% | {c_status:<18} |")

                    # Layer 2: Biophysics
                    print("-" * 60)
                    print(f"| {'BIOPHYSICAL REALISM':<35} | {'STATUS':<18} |")
                    print("-" * 60)
                    b_status = (
                        "[OK] PASS" if report["is_biophysically_plausible"] else "[FAIL] FAIL"
                    )
                    print(
                        f"| Hydrophobic Burial Ratio: {report['hydrophobic_burial_ratio']:>11.2f} | {b_status:<18} |"
                    )

                    # Layer 3: NMR (if applicable)
                    if "nmr_stats" in report:
                        print("-" * 60)
                        print(f"| {'NMR SPECTROSCOPIC FIDELITY':<35} | {'STATUS':<18} |")
                        print("-" * 60)
                        nmr = report["nmr_stats"]
                        n_status = (
                            "[OK] PASS" if nmr["noe_satisfaction_pct"] >= 90.0 else "[FAIL] FAIL"
                        )
                        print(
                            f"| NOE Satisfaction: {nmr['noe_satisfaction_pct']:>19.1f}% | {n_status:<18} |"
                        )

                    # Layer 4: AI/ML (if applicable)
                    if "ml_score" in report:
                        print("-" * 60)
                        print(f"| {'AI/GNN QUALITY FILTER':<35} | {'STATUS':<18} |")
                        print("-" * 60)
                        ml_status = "[OK] PASS" if report["ml_is_plausible"] else "[FAIL] FAIL"
                        print(
                            f"| ML Confidence Score: {report['ml_score']:>17.2f} | {ml_status:<18} |"
                        )

                    # Layer 5: Interface (if multi-chain)
                    if "interface_metrics" in report:
                        print("-" * 60)
                        print(f"| {'STRUCTURAL INTERACTOME (BSA)':<35} | {'STATUS':<18} |")
                        print("-" * 60)
                        int_m = report["interface_metrics"]
                        i_status = (
                            "[OK] PASS"
                            if int_m["is_interface_physically_plausible"]
                            else "[FAIL] FAIL"
                        )
                        print(
                            f"| Buried Surface Area: {int_m['buried_surface_area']:>15.1f} A^2 | {i_status:<18} |"
                        )

                    print("=" * 60)
                    final_def = report["is_overall_scientifically_defensible"]
                    overall_status = (
                        "[OK] SCIENTIFICALLY DEFENSIBLE" if final_def else "[FAIL] NOT DEFENSIBLE"
                    )
                    print(f"| OVERALL: {overall_status:^47} |")
                    print("=" * 60 + "\n")

                    if not final_def:
                        v_count = report.get("violation_count", 0)
                        logger.warning(f"Structure has {v_count} total violations.")
                        for violation in report.get("detailed_violations", [])[:5]:
                            logger.warning(f"  - {violation}")

            # Phase 7, 8, & 9 + 10: Synthetic NMR Data & Exports
            # We perform calculations first, so we can capture data (like restraints) for visualization if needed.
            generated_restraints = None  # To hold restraints for viewer

            if (
                args.gen_nef
                or args.restraints
                or args.rdc_restraints
                or args.shift_restraints
                or args.gen_relax
                or args.gen_shifts
                or args.gen_cd
                or args.gen_couplings
                or args.output_rdcs
                or args.export_constraints
                or args.export_torsion
                or args.gen_msa
                or args.export_distogram
            ):
                if args.mode != "generate":
                    logger.warning(
                        "Synthetic Data Generation is currently only supported in single structure 'generate' mode."
                    )
                else:
                    import io
                    import biotite.structure.io.pdb as pdb_io
                    from .chemical_shifts import (
                        calculate_shift_metrics,
                        predict_chemical_shifts,
                        read_shift_file,
                    )

                    # NEW IMPORTS for Export
                    from .contact import compute_contact_map
                    from .distogram import calculate_distogram, export_distogram
                    from .export import export_constraints
                    from .msa import generate_msa  # NEW: Physics based generator
                    from .nef_io import (
                        write_nef_chemical_shifts,
                        write_nef_file,
                        write_nef_relaxation,
                    )
                    from .nmr import (
                        calculate_rpf_score,
                        calculate_synthetic_noes,
                        read_restraint_file,
                    )
                    from .rdc import calculate_rdc_q_factor, calculate_rdcs, read_rdc_file
                    from .relaxation import calculate_relaxation_rates
                    from .torsion import calculate_torsion_angles, export_torsion_angles

                    # Sequence inference
                    from .data import L_TO_D_MAPPING, ONE_TO_THREE_LETTER_CODE

                    three_to_one = {v: k for k, v in ONE_TO_THREE_LETTER_CODE.items()}
                    # Add support for Histidine tautomers
                    three_to_one["HID"] = "H"
                    three_to_one["HIE"] = "H"
                    three_to_one["HIP"] = "H"
                    # Add support for PTMs
                    three_to_one["SEP"] = "S"
                    three_to_one["TPO"] = "T"
                    three_to_one["PTR"] = "Y"
                    # Add support for D-amino acids
                    for l_name, d_name in L_TO_D_MAPPING.items():
                        three_to_one[d_name] = three_to_one[l_name]

                    res_names = [
                        structure[structure.res_id == i][0].res_name
                        for i in sorted(set(structure.res_id))
                    ]
                    seq_str = "".join([three_to_one.get(r, "X") for r in res_names])

                    logger.info("Generating Synthetic Data...")

                    # We need the generated structure as an AtomArray
                    from .generator import PeptideResult

                    res_nmr = PeptideResult(final_content, format=internal_format)
                    structure = res_nmr.structure

                    # RPF Validation if restraints provided
                    if args.restraints:
                        logger.info(f"Performing RPF Validation against {args.restraints}...")
                        try:
                            target_restraints = read_restraint_file(args.restraints)
                            rpf_scores = calculate_rpf_score(structure, target_restraints)

                            # Print RPF Report
                            print("\n--- NMR RPF Validation Report ---")
                            print(f"Recall:    {rpf_scores.get('recall', 0.0):8.4f}")
                            print(f"Precision: {rpf_scores.get('precision', 0.0):8.4f}")
                            print(f"F-measure: {rpf_scores.get('f_measure', 0.0):8.4f}")
                            print("----------------------------------\n")

                        except Exception as e:
                            logger.error(f"RPF Validation failed: {e}")

                    # 1. NEF Generation (Phase 7)
                    if args.gen_nef:
                        if not np.any(structure.element == "H"):
                            logger.error(
                                "Structure has no hydrogens! NEF/Relaxation requires protons. Use --minimize."
                            )
                        else:
                            logger.info("Calculating NOE Restraints...")
                            restraints = calculate_synthetic_noes(structure, cutoff=args.noe_cutoff)
                            generated_restraints = restraints  # Capture for viewer

                            nef_filename = args.nef_output
                            if not nef_filename:
                                nef_filename = output_filename.replace(".pdb", ".nef")

                            write_nef_file(nef_filename, seq_str, restraints)
                            logger.info(
                                f"NEF Restraints generated: {os.path.abspath(nef_filename)}"
                            )

                            if args.gen_pymol:
                                from .visualization import generate_pymol_script

                                pml_filename = output_filename.replace(".pdb", ".pml")
                                generate_pymol_script(
                                    os.path.basename(output_filename), restraints, pml_filename
                                )
                                logger.info(
                                    f"PyMOL Visualization Script generated: {os.path.abspath(pml_filename)}"
                                )

                    # 2. Relaxation Data (Phase 8)
                    if args.gen_relax:
                        if not np.any(structure.element == "H"):
                            logger.error(
                                "Structure has no hydrogens! NEF/Relaxation requires protons. Use --minimize."
                            )
                        else:
                            logger.info(
                                f"Calculating synthetic relaxation rates ({args.field} MHz)..."
                            )
                            rates = calculate_relaxation_rates(
                                structure, field_mhz=args.field, tau_m_ns=args.tumbling_time
                            )
                            relax_filename = output_filename.replace(".pdb", "_relax.nef")
                            write_nef_relaxation(
                                relax_filename, seq_str, rates, field_freq_mhz=args.field
                            )
                            logger.info(
                                f"Relaxation data generated: {os.path.abspath(relax_filename)}"
                            )

                    # 3. Chemical Shifts (Phase 9)
                    if args.gen_shifts or args.shift_restraints:
                        logger.info("Predicting Chemical Shifts...")
                        # Map three-letter codes to one-letter for the shift engine
                        # (already done in seq_str)
                        use_shiftx2 = getattr(args, "shift_predictor", "shiftx2") == "shiftx2"
                        from .chemical_shifts import (
                            calculate_shift_metrics,
                            predict_chemical_shifts,
                            read_shift_file,
                        )

                        shifts = predict_chemical_shifts(structure, use_shiftx2=use_shiftx2)

                        shift_fn = args.shift_output
                        if not shift_fn and args.gen_shifts:
                            shift_fn = output_filename.replace(".pdb", "_shifts.nef")

                        if shift_fn:
                            from .nef_io import write_nef_chemical_shifts

                            write_nef_chemical_shifts(shift_fn, seq_str, shifts)
                            logger.info(f"Chemical shifts generated: {os.path.abspath(shift_fn)}")

                        # If shift restraints provided, calculate RMSD
                        if args.shift_restraints:
                            logger.info(
                                f"Validating shifts against experimental data: {args.shift_restraints}"
                            )
                            try:
                                exp_shifts = read_shift_file(args.shift_restraints)

                                # Align and group by atom type for per-atom metrics
                                atom_groups: dict[str, list[tuple[float, float]]] = {}
                                for item in exp_shifts:
                                    rid = item["res_id"]
                                    atom = item["atom_name"]
                                    val_obs = item["value"]

                                    # Match against any chain
                                    for c_id in shifts:
                                        if rid in shifts[c_id] and atom in shifts[c_id][rid]:
                                            val_calc = shifts[c_id][rid][atom]
                                            atype = atom[0]  # Group by element (C, N, H)
                                            if atype not in atom_groups:
                                                atom_groups[atype] = []
                                            atom_groups[atype].append((val_obs, val_calc))

                                if atom_groups:
                                    print("\n--- NMR Chemical Shift Validation Report ---")
                                    for atype, pairs in atom_groups.items():
                                        obs_arr = np.array([p[0] for p in pairs])
                                        calc_arr = np.array([p[1] for p in pairs])
                                        res_metrics = calculate_shift_metrics(obs_arr, calc_arr)
                                        print(
                                            f"{atype:4s} RMSD:        {res_metrics['rmsd']:8.4f} ppm"
                                        )
                                        print(
                                            f"{atype:4s} Correlation: {res_metrics['correlation']:8.4f}"
                                        )
                                    print("------------------------------------------\n")
                            except Exception as e:
                                logger.error(f"Shift validation failed: {e}")

                    # 3.7 Circular Dichroism (Phase 9.7)
                    if args.gen_cd:
                        # EDUCATIONAL NOTE - CD Background:
                        # Circular Dichroism (CD) measures the differential absorption
                        # of left and right circularly polarized light. In the far-UV
                        # (190-250 nm), it is the premier tool for measuring the
                        # overall secondary structure content of a protein sample.
                        #
                        # The physics is based on the interaction between amide
                        # chromophores. For a given conformation, we can synthesize
                        # the expected spectrum as a weighted average of basis
                        # spectra (Greenfield & Fasman, 1969, Biochemistry 8:4108):
                        #   [theta]total = f_helix * [theta]helix + f_sheet * [theta]sheet + f_coil * [theta]coil
                        logger.info("Simulating Circular Dichroism (CD) spectrum...")
                        from .cd_simulator import CDSimulator, validate_cd_against_literature

                        cd_sim = CDSimulator(structure)
                        cd_plot = output_filename.replace(".pdb", "_cd.png")
                        cd_sim.plot(save_path=cd_plot)

                        # Automated scientific validation against literature CD signatures.
                        cd_findings = validate_cd_against_literature(
                            cd_sim.fractions, cd_sim.get_spectrum(noise_level=0)
                        )
                        if cd_findings:
                            logger.info("--- Synthetic CD Validation Report ---")
                            for find_item in cd_findings:
                                logger.info(f"  {find_item}")
                            logger.info("--------------------------------------")

                        logger.info(
                            f"Synthetic CD spectrum plot saved to: {os.path.abspath(cd_plot)}"
                        )

                    # 4.5 J-Couplings (Phase 9.5)
                    if args.gen_couplings:
                        logger.info("Calculating HN-HA scalar couplings...")
                        from .coupling import predict_couplings_from_structure
                        from .torsion import calculate_torsion_angles

                        # predict_couplings_from_structure() returns
                        # {chain_id: {res_id: J_value}} with prolines stripped and
                        # D-amino acids phase-corrected. We iterate over the full
                        # residue list (from calculate_torsion_angles) so the CSV
                        # has one row per residue - prolines/N-term get NaN rather
                        # than being silently absent, which keeps the schema
                        # fixed-width for downstream consumers.
                        couplings_by_chain = predict_couplings_from_structure(structure)
                        flat_couplings: dict[int, float] = {}
                        for inner in couplings_by_chain.values():
                            flat_couplings.update(inner)

                        angles_list = calculate_torsion_angles(structure)
                        cp_fn = args.coupling_output or output_filename.replace(".pdb", "_j.csv")
                        with open(cp_fn, "w") as f:
                            f.write("res_id,residue,J_HN_HA\n")
                            for angle_data in angles_list:
                                rid = angle_data["res_id"]
                                res = angle_data["residue"]
                                jval = flat_couplings.get(rid, float("nan"))
                                f.write(f"{rid},{res},{jval:.4f}\n")
                        logger.info(f"Scalar couplings exported to: {os.path.abspath(cp_fn)}")

                    # 3.6 RDC Output (Phase 9.6)
                    output_rdcs = getattr(args, "output_rdcs", None)
                    if output_rdcs or args.rdc_restraints:
                        # EDUCATIONAL NOTE - RDC Calculation:
                        # We compute backbone N-H Residual Dipolar Couplings by:
                        #   1. Locating every backbone amide nitrogen (N) and its
                        #      associated amide proton (H) in the structure.
                        #   2. Computing the unit vector along each N-H bond.
                        #   3. Projecting that vector onto the alignment tensor
                        #      principal axis system (PAS) to get (theta, phi).
                        #   4. Applying the full RDC formula:
                        #        D = Da * [(3cos^2theta - 1) + 1.5*R*sin^2theta*cos(2phi)]
                        #      (Tjandra & Bax, 1997, Science 278:1111)
                        #
                        # Proline residues are automatically skipped because they
                        # lack a backbone amide proton (their nitrogen is a
                        # tertiary/secondary amine in the pyrrolidine ring).
                        from .rdc import (
                            calculate_rdc_q_factor,
                            calculate_rdcs,
                            export_rdcs,
                            read_rdc_file,
                        )

                        rdc_da = getattr(args, "rdc_da", 10.0)
                        rdc_r = getattr(args, "rdc_r", 0.1)
                        rdcs = calculate_rdcs(structure, da=rdc_da, r=rdc_r)

                        if output_rdcs:
                            export_rdcs(rdcs, output_rdcs, structure=structure)
                            logger.info(f"RDC data exported to: {os.path.abspath(output_rdcs)}")

                        # If RDC restraints provided, calculate Q-factor
                        if args.rdc_restraints:
                            logger.info(
                                f"Validating RDCs against experimental data: {args.rdc_restraints}"
                            )
                            try:
                                exp_rdc_list = read_rdc_file(args.rdc_restraints)
                                # Align exp vs calc
                                obs_vals = []
                                calc_vals = []
                                for item in exp_rdc_list:
                                    rid = item["res_1"]
                                    if rid in rdcs:
                                        obs_vals.append(item["value"])
                                        calc_vals.append(rdcs[rid])

                                if obs_vals:
                                    q_factor = calculate_rdc_q_factor(
                                        np.array(obs_vals), np.array(calc_vals)
                                    )
                                    print("\n--- NMR RDC Validation Report ---")
                                    print("=" * 40)
                                    print("[CD] RDC VALIDATION (Q-FACTOR)")
                                    print("=" * 40)
                                    print(
                                        f"| Q-factor: {q_factor:8.4f} "
                                        f"{'(EXCELLENT)' if q_factor < 0.2 else '(POOR)'} |"
                                    )
                                    print("=" * 40 + "\n")
                            except Exception as e:
                                logger.error(f"RDC validation failed: {e}")
                    # 5. Feature Export (Phase 11)
                    if args.export_constraints or args.export_torsion:
                        logger.info("Exporting structural features...")
                        if args.export_constraints:
                            from .contact import compute_contact_map
                            from .export import export_constraints

                            # Use user-specified cutoff (Phase 11)
                            cmap = compute_contact_map(
                                structure, method="ca", threshold=args.constraint_cutoff
                            )
                            constr_str = export_constraints(
                                cmap,
                                seq_str,
                                fmt=args.constraint_format,
                                threshold=args.constraint_cutoff,
                            )
                            with open(args.export_constraints, "w") as f:
                                f.write(constr_str)
                            logger.info(
                                f"Constraints exported to: {os.path.abspath(args.export_constraints)}"
                            )

                        if args.export_torsion:
                            from .torsion import calculate_torsion_angles, export_torsion_angles

                            angles = calculate_torsion_angles(structure)
                            export_torsion_angles(
                                angles, args.export_torsion, fmt=args.torsion_format
                            )
                            logger.info(
                                f"Torsion angles exported to: {os.path.abspath(args.export_torsion)}"
                            )

                    # 6. MSA Generation (Phase 12)
                    if args.gen_msa:
                        logger.info(
                            f"Generating Synthetic MSA (depth: {args.msa_depth}, temp: {args.evolution_temp})..."
                        )
                        # 1. Extract ground-truth contact map for constraints
                        cmap = compute_contact_map(
                            structure, method="ca", threshold=args.constraint_cutoff
                        )
                        # Convert Distogram to boolean map using the user-specified cutoff
                        bool_cmap = (cmap > 0) & (cmap <= args.constraint_cutoff)

                        # 2. Run Metropolis-Hastings Co-Evolution
                        from .msa import generate_msa

                        sequences = generate_msa(
                            base_sequence=seq_str,
                            contact_map=bool_cmap,
                            num_sequences=args.msa_depth,
                            temperature=args.evolution_temp,
                            steps_between_samples=100,
                        )

                        # 3. Write FASTA
                        msa_filename = (
                            args.output
                            if args.output and args.output.endswith(".fasta")
                            else output_filename.replace(".pdb", ".fasta")
                        )
                        with open(msa_filename, "w") as f:
                            for idx, sq in enumerate(sequences):
                                f.write(f">seq_{idx}\n{sq}\n")

                        logger.info(f"Synthetic MSA generated: {os.path.abspath(msa_filename)}")

                    # 7. Distogram Export (Phase 13)
                    if args.export_distogram:
                        from .distogram import calculate_distogram, export_distogram

                        matrix = calculate_distogram(structure)
                        export_distogram(matrix, args.export_distogram, fmt=args.distogram_format)
                        logger.info(
                            f"Distogram exported to: {os.path.abspath(args.export_distogram)}"
                        )

            # 8. Docking Post-Processing (PQR generation)
            if args.mode == "docking":
                logger.info("Generating PQR file for docking...")
                try:
                    prep = DockingPrep(args.forcefield)

                    # We need a PDB file for DockingPrep.
                    # If the primary output was PDB, use it.
                    # Otherwise, generate a temporary one.
                    pdb_for_prep = output_filename
                    temp_pdb = None

                    if args.format != "pdb":
                        import tempfile
                        from .generator import PeptideResult

                        res_tmp = PeptideResult(final_content, format=internal_format)
                        temp_pdb_content = res_tmp.get_content("pdb")
                        fd, temp_pdb = tempfile.mkstemp(suffix=".pdb")
                        with os.fdopen(fd, "w") as f:
                            f.write(cast(str, temp_pdb_content))
                        pdb_for_prep = temp_pdb

                    pqr_file = (
                        args.output
                        if (args.output and args.output.endswith(".pqr"))
                        else output_filename.rsplit(".", 1)[0] + ".pqr"
                    )

                    success = prep.write_pqr(pdb_for_prep, pqr_file)
                    if success:
                        logger.info(
                            f"Docking preparation complete. PQR file: {os.path.abspath(pqr_file)}"
                        )
                    else:
                        logger.error("Docking preparation failed.")

                    # Cleanup temporary PDB if created
                    if temp_pdb and os.path.exists(temp_pdb):
                        os.remove(temp_pdb)

                except Exception as e:
                    logger.error(f"Failed to generate PQR: {e}")

            # Open 3D viewer if requested (MOVED AFTER NMR calc to access generated_restraints)
            if args.visualize:
                if isinstance(final_to_write, bytes):
                    logger.warning(
                        "3D visualization is not supported for binary formats (BCIF). Skipping viewer."
                    )
                else:
                    logger.info("Opening 3D molecular viewer in browser...")
                    try:
                        view_structure_in_browser(
                            final_to_write,
                            filename=output_filename,
                            style="cartoon",
                            color="spectrum",
                            restraints=generated_restraints,  # Pass captured restraints
                            highlights=highlights,  # Pass beta-turn highlights
                            show_hbonds=True,
                        )
                    except Exception as e:
                        logger.error(f"Failed to open 3D viewer: {e}")

        except Exception as e:
            logger.error("An unexpected error occurred during file writing: %s", e)
            sys.exit(1)
    else:
        # If final_content is None
        logger.error("No suitable content was generated for writing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
