# EDUCATIONAL OVERVIEW - How Synthetic Protein Generation Works:
# -----------------------------------------------------------
# 1. Sequence Resolution: Determine the amino acid string (e.g., "ALA-GLY-SER").
#    - Sequences can be single chains or multiple chains (separated by ':').
#    - Natural protein sequences often have specific frequencies of amino acids.
# 2. Backbone Generation: Place N-CA-C-O atoms for each residue.
#    - Geometrically constructing the chain using Bond Lengths and Angles.
#    - Setting Dihedral Angles (Phi/Psi) to define secondary structure (Helix/Sheet).
#    - The NeRF algorithm (Natural Extension Reference Frame) is used for stability.
# 3. Side-Chain Placement: Add side-chain atoms (CB, CG...) based on Rotamer Libraries.
#    - Backbone-dependent rotamer libraries (Dunbrack) provide realistic conformations.
#    - Rotation about Chi angles is performed using Rodrigues' rotation formula.
# 4. Refinement (Optional):
#    - Packing: Optimize rotamers via Monte Carlo to avoid steric clashes.
#    - Physics: Use OpenMM to minimize the potential energy of the assembly.
# 5. Metadata: Fill in B-factors and Occupancy.
#    - X-ray: Represents thermal motion (temperature factors).
#    - NMR: Often used to represent local ensemble flexibility or Order Parameters (S2).

# BIOPHYSICAL PRINCIPLES:
# ----------------------
# - The Peptide Bond: A planar, trans-configured amide bond connecting residues.
# - The Ramachandran Plot: The conformational landscape defined by Phi and Psi angles.
# - Secondary Structure: Regular patterns of H-bonding (Helices, Sheets, Turns).
# - The Structural Interactome: Multi-chain assemblies that form functional machines.
# - Chiral Mirroring: Generating D-amino acids as reflections of standard L-isomers.

# Re-export for backward compatibility with tests
import io
import logging
import os
import random
import tempfile
from typing import Any, cast

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

from . import biophysics  # New Module
from .batch_generator import BatchedGenerator
from .data import (
    ALL_VALID_AMINO_ACIDS,
    AMINO_ACID_FREQUENCIES,
    ANGLE_C_N_CA,
    ANGLE_CA_C_N,
    ANGLE_CA_C_O,
    ANGLE_N_CA_C,
    BACKBONE_DEPENDENT_ROTAMER_LIBRARY,
    BETA_TURN_TYPES,
    BOND_LENGTH_C_N,
    BOND_LENGTH_C_O,
    BOND_LENGTH_CA_C,
    BOND_LENGTH_N_CA,
    L_TO_D_MAPPING,
    NON_STANDARD_RESIDUES,
    ONE_TO_THREE_LETTER_CODE,
    RAMACHANDRAN_PRESETS,
    RAMACHANDRAN_REGIONS,
    ROTAMER_LIBRARY,
    STANDARD_AMINO_ACIDS,
)
from .geometry import (
    calculate_dihedral_angle,
    position_atom_3d_from_internal_coords,
)
from .packing import optimize_sidechains as run_optimization
from .pdb_utils import (
    assemble_pdb_content,
    extract_atomic_content,
)
from .physics import EnergyMinimizer
from .relaxation import predict_order_parameters

# Re-export for backward compatibility with tests
_position_atom_3d_from_internal_coords = position_atom_3d_from_internal_coords

# Used in _build_peptide_chain to orient the first residue's C atom
ANGLE_N_CA_C_RAD = np.deg2rad(ANGLE_N_CA_C)

# Ideal Omega for trans peptide bond
OMEGA_TRANS = 180.0
OMEGA_VARIATION = 4.0  # degrees - adds thermal fluctuation to peptide bond

logger = logging.getLogger(__name__)


# This constant is used in test_generator.py for coordinate calculations.
CA_DISTANCE = 3.8  # Approximate C-alpha to C-alpha distance in Angstroms for a linear chain


def _calculate_bfactor(
    atom_name: str,
    residue_number: int,
    total_residues: int,
    residue_name: str,
    s2: float = 0.85,
    rng: random.Random | None = None,
) -> float:
    r"""Calculate realistic B-factor (temperature factor) derived from Order Parameter (S2).

    EDUCATIONAL NOTE - B-factors (Temperature Factors):
    ===================================================
    B-factors represent atomic displacement due to thermal motion and static disorder.
    They are measured in Ų (square Angstroms) and indicate atomic mobility.

    Physical Interpretation:
    - B = 8π²<u²> where <u²> is mean square displacement
    - Higher B-factor = more mobile/flexible atom
    - Lower B-factor = more rigid/constrained atom

    Typical Patterns in Real Protein Structures:
    1. Backbone vs Side Chains:
       - Backbone atoms (N, CA, C, O): 15-25 Ų (constrained by peptide bonds)
       - Side chain atoms (CB, CG, etc.): 20-35 Ų (more conformational freedom)

    2. Position Along Chain:
       In this synthetic generator, we simulate B-factors that follow these
       universal patterns of rigidity vs. flexibility.
       - Core residues: 10-20 Ų (buried, constrained)
       - Terminal residues: 30-50 Ų ("terminal fraying" - fewer constraints)

    3. Residue-Specific Effects:
       - Glycine: Higher (no side chain constraints, more flexible)
       - Proline: Lower (cyclic structure, rigid backbone)

    For Linear Peptides (our case):
    - No folding/packing, so all residues are "surface-like"
    - Terminal fraying effect is prominent
    - Backbone still more rigid than side chains

    4. NMR Perspective (Order Parameters):
       For NMR structures, the "B-factor" column is often repurposed.
       - It can store RMSD across the ensemble.
       - It can also represent the Order Parameter ($S^2$).
       - $S^2$ (from Lipari-Szabo) measures rigidity on ps-ns timescales.
       - $S^2 = 1.0 \rightarrow$ Rigid (Low B-factor).
       - $S^2 \approx 0.5 \rightarrow$ Flexible (High B-factor).

    In this generator, we explicitly link the geometric flexibility ($S^2$)
    to the crystallographic observable ($B$), creating a unified biophysical model.

    Args:
        atom_name: Atom name (e.g., 'N', 'CA', 'CB', 'CG')
        residue_number: Residue number (1-indexed)
        total_residues: Total number of residues in chain
        residue_name: Three-letter residue code (e.g., 'ALA', 'GLY')
        s2: Lipari-Szabo Order Parameter (0.0=Random, 1.0=Rigid). Default 0.85.

    Returns:
        B-factor value in Ų, rounded to 2 decimal places

    """
    # Define backbone atoms (more rigid due to peptide bond constraints)
    backbone_atoms = {"N", "CA", "C", "O", "H", "HA"}

    # Base physics: B-factor inversely related to Order Parameter
    # Calibration:
    # S2=1.00 -> B=5.0
    # S2=0.85 -> B=20.0 (Typical Core)
    # S2=0.50 -> B=55.0 (Typical Termini/Loop)

    # Simple linear map: B = M * (1 - S2) + C
    # Delta S2 = 0.35 (0.85->0.50) -> Delta B = 35 (20->55)
    # Slope M = 35/0.35 = 100
    # B = 100 * (1 - S2) + Offset
    # Check: S2=0.85 -> B = 100*0.15 = 15. Need 20. So Offset=5.

    base_bfactor = 100.0 * (1.0 - s2) + 5.0

    # Atom type adjustments
    if atom_name not in backbone_atoms:
        base_bfactor *= 1.5  # Side chains are more mobile than backbone

    # Residue-specific adjustments
    if residue_name == "GLY":
        base_bfactor += 5.0
    elif residue_name == "PRO":
        base_bfactor -= 3.0

    # Add small random variation
    _rng: Any = rng if rng is not None else random
    random_variation = float(_rng.uniform(-2.0, 2.0))

    # Calculate final B-factor
    bfactor = base_bfactor + random_variation

    # Clamp to realistic range (5-99 Ų)
    bfactor = max(5.0, min(99.0, bfactor))

    return round(bfactor, 2)


def _calculate_occupancy(
    atom_name: str,
    residue_number: int,
    total_residues: int,
    residue_name: str,
    bfactor: float,
    rng: random.Random | None = None,
) -> float:
    """Calculate realistic occupancy for an atom (0.85-1.00)."""
    backbone_atoms = {"N", "CA", "C", "O", "H", "HA"}

    # Base occupancy
    base_occupancy = 0.98 if atom_name in backbone_atoms else 0.95

    # Terminal disorder
    dist_from_n_term = (residue_number - 1) / max(total_residues - 1, 1)
    dist_from_c_term = (total_residues - residue_number) / max(total_residues - 1, 1)
    dist_from_nearest_term = min(dist_from_n_term, dist_from_c_term)
    terminal_factor = -0.10 * (1.0 - dist_from_nearest_term)

    # Residue-specific
    residue_factor = 0.0
    if residue_name in ["GLY", "SER", "ASN", "GLN"]:
        residue_factor = -0.03
    elif residue_name in ["PRO", "TRP", "PHE"]:
        residue_factor = +0.02

    # B-factor correlation
    normalized_bfactor = (bfactor - 5.0) / 55.0
    bfactor_correlation = -0.08 * normalized_bfactor

    # Random variation
    _rng: Any = rng if rng is not None else random
    random_variation = float(_rng.uniform(-0.01, 0.01))

    # Calculate and clamp
    occupancy = (
        base_occupancy + terminal_factor + residue_factor + bfactor_correlation + random_variation
    )
    occupancy = max(0.85, min(1.00, occupancy))

    return round(occupancy, 2)


# Helper function to create a minimal PDB ATOM line
def create_atom_line(
    atom_number: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str,
    alt_loc: str = "",
    insertion_code: str = "",
    temp_factor: float = 0.00,  # B-factor (temperature factor) in Ų
    occupancy: float = 1.00,  # Occupancy (fraction of molecules)
) -> str:
    """Create a PDB ATOM line.

    EDUCATIONAL NOTE - PDB ATOM Record Format:
    - temp_factor (columns 61-66): Atomic mobility/flexibility
    - occupancy (columns 55-60): Fraction of molecules with atom at this position
    See _calculate_bfactor() and _calculate_occupancy() for details.
    """
    return (
        f"ATOM  {atom_number: >5} {atom_name: <4}{alt_loc: <1}{residue_name: >3} {chain_id: <1}"
        f"{residue_number: >4}{insertion_code: <1}   "
        f"{x: >8.3f}{y: >8.3f}{z: >8.3f}{occupancy: >6.2f}{temp_factor: >6.2f}          "
        f"{element: >2}  "
    )


def _place_atom_with_dihedral(
    atom1: np.ndarray,
    atom2: np.ndarray,
    atom3: np.ndarray,
    bond_length: float,
    bond_angle: float,
    dihedral: float,
) -> np.ndarray:
    """Place a new atom using bond length, angle, and dihedral.

    Wrapper around position_atom_3d_from_internal_coords with clearer naming.
    """
    return cast(
        np.ndarray,
        position_atom_3d_from_internal_coords(
            atom1, atom2, atom3, bond_length, bond_angle, dihedral
        ),
    )


def _generate_random_amino_acid_sequence(
    length: int, use_plausible_frequencies: bool = False, rng: random.Random | None = None
) -> list[str]:
    """Generate a random amino acid sequence of a given length.

    If `use_plausible_frequencies` is True, uses frequencies from AMINO_ACID_FREQUENCIES.
    Otherwise, uses a uniform random distribution.

    Args:
        length: Number of residues to generate.
        use_plausible_frequencies: If True, uses biological distribution.
        rng: Optional local random generator.

    Returns:
        List of 3-letter amino acid codes.
    """
    # Use global random if no local rng provided (for backward compatibility)
    _rng: Any = rng if rng is not None else random

    # Handle potential non-standard types like numpy._NoValue
    if length is None or str(length) == "<no value>":
        return []

    try:
        length_int = int(length)
    except (TypeError, ValueError):
        logger.warning(f"Invalid sequence length: {length}. Returning empty list.")
        return []

    if length_int <= 0:
        return []

    if use_plausible_frequencies:
        amino_acids = list(AMINO_ACID_FREQUENCIES.keys())
        weights = list(AMINO_ACID_FREQUENCIES.values())
        return cast(list[str], _rng.choices(amino_acids, weights=weights, k=length))
    else:
        return [str(_rng.choice(STANDARD_AMINO_ACIDS)) for _ in range(length)]


def _detect_disulfide_bonds(peptide: struc.AtomArray) -> list:
    """Detect potential disulfide bonds between cysteine residues.

    EDUCATIONAL NOTE - Disulfide Bond Detection:
    ============================================
    Disulfide bonds form between two cysteine (CYS) residues when their
    sulfur atoms (SG) are close enough to form a covalent S-S bond.

    Detection Criteria:
    - Both residues must be CYS
    - SG-SG distance: 2.0-2.2 Å (slightly relaxed from ideal 2.0-2.1 Å)
    - Only report each pair once (avoid duplicates)

    Why Distance Matters:
    - < 2.0 Å: Too close (steric clash, not realistic)
    - 2.0-2.1 Å: Ideal disulfide bond distance
    - 2.1-2.2 Å: Acceptable (allows for flexibility)
    - > 2.2 Å: Too far (no covalent bond possible)

    Biological Context:
    - Disulfides stabilize protein structure
    - Common in extracellular proteins
    - Rare in cytoplasm (reducing environment)
    - Important for protein folding and stability

    Args:
        peptide: Biotite AtomArray structure

    Returns:
        List of tuples (res_id1, res_id2) representing disulfide bonds

    Example:
        >>> disulfides = _detect_disulfide_bonds(structure)
        >>> print(disulfides)
        [(3, 8), (12, 20)]  # CYS 3-8 and CYS 12-20 are bonded

    """
    disulfides: list[tuple[int, int]] = []

    # Find all CYS/CYX residues
    cys_residues = peptide[(peptide.res_name == "CYS") | (peptide.res_name == "CYX")]

    if len(cys_residues) < 2:
        return disulfides  # Need at least 2 CYS for a bond

    # Get unique residue IDs
    cys_res_ids = np.unique(cys_residues.res_id)

    # Check all pairs of CYS residues
    for i, res_id1 in enumerate(cys_res_ids):
        for res_id2 in cys_res_ids[i + 1 :]:  # Avoid duplicates
            # Get SG atoms for both residues
            sg1 = peptide[(peptide.res_id == res_id1) & (peptide.atom_name == "SG")]
            sg2 = peptide[(peptide.res_id == res_id2) & (peptide.atom_name == "SG")]

            if len(sg1) > 0 and len(sg2) > 0:
                # Calculate distance
                p1 = sg1[0].coord
                p2 = sg2[0].coord
                distance = np.sqrt(np.sum((p1 - p2) ** 2))

                # Check if within disulfide bond range
                # Minimized structures can be highly strained (especially small cyclic rings),
                # so we use a generous range [1.5, 3.0] to ensure the SSBOND record is generated
                # if the physics engine identified a bond.
                if 1.5 <= distance <= 3.0:
                    disulfides.append((int(res_id1), int(res_id2)))

    return disulfides


def _generate_ssbond_records(disulfides: list, chain_id: str = "A") -> str:
    """Generate SSBOND records for PDB header.

    EDUCATIONAL NOTE - PDB SSBOND Format:
    ====================================
    SSBOND records annotate disulfide bonds in PDB files.

    Format (PDB specification):
    SSBOND   1 CYS A    6    CYS A   11

    Columns:
    1-6:   "SSBOND"
    8-10:  Serial number (1, 2, 3, ...)
    12-14: Residue name 1 (always "CYS")
    16:    Chain ID 1
    18-21: Residue number 1 (right-justified)
    26-28: Residue name 2 (always "CYS")
    30:    Chain ID 2
    32-35: Residue number 2 (right-justified)

    Why This Matters:
    - Structure viewers use this to display bonds
    - Analysis tools use this for stability calculations
    - Essential for understanding protein structure
    - Part of standard PDB format

    Args:
        disulfides: List of (res_id1, res_id2) tuples
        chain_id: Chain identifier (default 'A')

    Returns:
        String containing SSBOND records (one per line)

    Example:
        >>> records = _generate_ssbond_records([(3, 8), (12, 20)], 'A')
        >>> print(records)
        SSBOND   1 CYS A    3    CYS A    8
        SSBOND   2 CYS A   12    CYS A   20

    """
    if not disulfides:
        return ""

    records = []
    for serial, (res_id1, res_id2) in enumerate(disulfides, 1):
        # Format according to PDB specification
        # SSBOND   1 CYS A    6    CYS A   11
        record = f"SSBOND{serial:4d} CYS {chain_id}{res_id1:5d}    CYS {chain_id}{res_id2:5d}"
        records.append(record)

    return "\n".join(records) + "\n" if records else ""


def _resolve_sequence(
    length: int | None,
    user_sequence_str: str | None = None,
    use_plausible_frequencies: bool = False,
    rng: random.Random | None = None,
) -> list[str]:
    """Resolve the amino acid sequence from user input or random generation.

    Args:
        length: Target length for random generation.
        user_sequence_str: Optional user-provided sequence (1-letter or 3-letter).
        use_plausible_frequencies: If True, uses biological frequencies for random.

    Returns:
        List of 3-letter amino acid codes.
    """
    if user_sequence_str:
        user_sequence_str_upper = user_sequence_str.upper()
        if "-" in user_sequence_str_upper:
            # Assume 3-letter code format like 'ALA-GLY-VAL' or 'D-ALA-GLY'
            # Note: We split by '-' but if somebody uses 'D-ALA', splitting by '-'
            # will give ['D', 'ALA']. This is ambiguous with 'ALA-GLY'.
            # Better logic: if there are any residues starting with 'D-',
            # we should preserve them.

            # Simple fix: join them back if they were part of a D- moiety
            raw_splits = [
                s.strip().upper() for s in user_sequence_str_upper.split("-") if s.strip()
            ]
            amino_acids = []
            skip_next = False
            for j, part in enumerate(raw_splits):
                if skip_next:
                    skip_next = False
                    continue
                if part == "D" and j + 1 < len(raw_splits):
                    next_p = raw_splits[j + 1]
                    if len(next_p) == 1:
                        next_p = ONE_TO_THREE_LETTER_CODE.get(next_p, next_p)
                    amino_acids.append(f"D-{next_p}")
                    skip_next = True
                elif len(part) == 4 and part.startswith("D"):
                    # Handle shorthand like 'DALA' -> 'D-ALA'
                    base = part[1:]
                    if base in ALL_VALID_AMINO_ACIDS:
                        amino_acids.append(f"D-{base}")
                    else:
                        amino_acids.append(part)
                else:
                    if len(part) == 1:
                        amino_acids.append(ONE_TO_THREE_LETTER_CODE.get(part, part))
                    else:
                        amino_acids.append(part)

            for aa in amino_acids:
                base_aa = aa[2:] if aa.startswith("D-") else aa
                if base_aa not in ALL_VALID_AMINO_ACIDS:
                    raise ValueError(f"Invalid 3-letter amino acid code: {aa}")
            return amino_acids
        elif len(user_sequence_str_upper) == 3 and user_sequence_str_upper in ALL_VALID_AMINO_ACIDS:
            # It's a single 3-letter amino acid code
            return [user_sequence_str_upper]
        else:
            # Assume 1-letter code format format 'AGV'
            amino_acids = []
            for one_letter_code in user_sequence_str_upper:
                if one_letter_code not in ONE_TO_THREE_LETTER_CODE:
                    raise ValueError(f"Invalid 1-letter amino acid code: {one_letter_code}")
                amino_acids.append(ONE_TO_THREE_LETTER_CODE[one_letter_code])
            return amino_acids
    else:
        actual_length = length if length is not None else 20
        return _generate_random_amino_acid_sequence(
            actual_length, use_plausible_frequencies=use_plausible_frequencies, rng=rng
        )


def _sample_ramachandran_angles(
    res_name: str, next_res_name: str | None = None, rng: random.Random | None = None
) -> tuple[float, float]:
    """Sample phi/psi angles from Ramachandran probability distribution.

    Uses residue-specific distributions for GLY and PRO, general distribution
    for all other amino acids. Samples from favored regions using weighted
    Gaussian distributions.

    New Feature: Pre-Proline Bias
    If next_res_name is 'PRO' and current residue is not GLY or PRO,
    uses a specific 'PRE_PRO' distribution (favors beta/extended).

    Args:
        res_name: Three-letter amino acid code
        next_res_name: (Optional) Code of the next residue
        rng: Optional local random generator.

    Returns:
        Tuple of (phi, psi) angles in degrees

    Reference:
        Lovell et al. (2003) Proteins: Structure, Function, and Bioinformatics

    """
    _rng: Any = rng if rng is not None else random

    # Get residue-specific or general distribution
    if res_name in RAMACHANDRAN_REGIONS:
        # GLY or PRO specific maps take precedence
        regions = RAMACHANDRAN_REGIONS[res_name]
    elif next_res_name == "PRO":
        # Pre-Proline effect!
        # Standard residues (ALA, VAL, etc) before Proline have restricted conformation.
        regions = RAMACHANDRAN_REGIONS.get("PRE_PRO", RAMACHANDRAN_REGIONS["general"])
    else:
        regions = RAMACHANDRAN_REGIONS["general"]

    # Get favored regions
    favored_regions = regions["favored"]
    weights = [r["weight"] for r in favored_regions]

    # Choose region based on weights
    chosen_region = _rng.choices(favored_regions, weights=weights, k=1)[0]

    # Sample angles from Gaussian around region center
    phi = _rng.normalvariate(chosen_region["phi"], chosen_region["std"])
    psi = _rng.normalvariate(chosen_region["psi"], chosen_region["std"])

    # Wrap to [-180, 180]
    phi = ((phi + 180) % 360) - 180
    psi = ((psi + 180) % 360) - 180

    return phi, psi


def _parse_structure_regions(structure_str: str, sequence_length: int) -> dict[int, str]:
    """Parse structure region specification into per-residue conformations.

    This function enables users to specify different secondary structure conformations
    for different regions of their peptide. This is crucial for creating realistic
    protein-like structures that have mixed secondary structures (e.g., helix-turn-sheet).

    EDUCATIONAL NOTE - Why This Matters:
    Real proteins don't have uniform secondary structure throughout. They typically
    have regions of alpha helices, beta sheets, turns, and loops. This function
    allows users to specify these regions explicitly, making the generated structures
    much more realistic and useful for educational demonstrations.

    Args:
        structure_str: Region specification string in format "start-end:conformation,..."
                      Example: "1-10:alpha,11-20:beta,21-30:random"
                      - Residue numbering is 1-indexed (first residue is 1)
                      - Conformations: alpha, beta, ppii, extended, random
                      - Multiple regions separated by commas
        sequence_length: Total number of residues in the sequence

    Returns:
        Dictionary mapping residue index (0-based) to conformation name.
        Only includes explicitly specified residues (gaps are allowed).

        EDUCATIONAL NOTE - Return Format:
        We use 0-based indexing internally (Python convention) even though
        the input uses 1-based indexing (PDB/biology convention). This is
        a common pattern in bioinformatics software.

    Raises:
        ValueError: If syntax is invalid, regions overlap, or ranges are out of bounds

    Examples:
        >>> _parse_structure_regions("1-10:alpha,11-20:beta", 20)
        {0: 'alpha', 1: 'alpha', ..., 9: 'alpha', 10: 'beta', ..., 19: 'beta'}

        >>> _parse_structure_regions("1-5:alpha,10-15:beta", 20)
        {0: 'alpha', ..., 4: 'alpha', 9: 'beta', ..., 14: 'beta'}
        # Note: Residues 6-9 and 16-20 are not in the dictionary (gaps allowed)

    EDUCATIONAL NOTE - Design Decisions:
    1. We allow gaps in coverage - unspecified residues will use the default conformation
    2. We strictly forbid overlaps - each residue can only have one conformation
    3. We validate all inputs before processing to give clear error messages

    """
    # Handle empty input - return empty dictionary (all residues will use default)
    if not structure_str:
        return {}

    # EDUCATIONAL NOTE - Data Structure Choice:
    # We use a dictionary to map residue indices to conformations because:
    # 1. Fast lookup: O(1) to check if a residue has a specified conformation
    # 2. Sparse representation: Only stores specified residues (memory efficient)
    # 3. Easy to check for overlaps: Just check if key already exists
    residue_conformations = {}

    # Split the input string by commas to get individual region specifications
    # Example: "1-10:alpha,11-20:beta" -> ["1-10:alpha", "11-20:beta"]
    regions = structure_str.split(",")

    # Process each region specification
    for region in regions:
        # Remove any leading/trailing whitespace for robustness
        # This allows users to write "1-10:alpha, 11-20:beta" (with spaces)
        region = region.strip()

        # VALIDATION STEP 1: Check for colon separator
        # Expected format: "start-end:conformation"
        if ":" not in region:
            raise ValueError(
                f"Invalid region syntax: '{region}'. "
                f"Expected format: 'start-end:conformation' (e.g., '1-10:alpha')"
            )

        # Split by colon to separate range from conformation
        # Example: "1-10:alpha" -> range_part="1-10", conformation="alpha"
        range_part, conformation = region.split(":", 1)

        # VALIDATION STEP 2: Check conformation name
        # Build list of valid conformations from presets plus 'random'
        valid_conformations = (
            list(RAMACHANDRAN_PRESETS.keys()) + ["random"] + list(BETA_TURN_TYPES.keys())
        )
        if conformation not in valid_conformations:
            raise ValueError(
                f"Invalid conformation '{conformation}'. "
                f"Valid options are: {', '.join(valid_conformations)}"
            )

        # VALIDATION STEP 3: Check for dash separator in range
        # Expected format: "start-end"
        if "-" not in range_part:
            raise ValueError(
                f"Invalid range syntax: '{range_part}'. Expected format: 'start-end' (e.g., '1-10')"
            )

        # Split range by dash to get start and end positions
        # Example: "1-10" -> start_str="1", end_str="10"
        start_str, end_str = range_part.split("-", 1)

        # VALIDATION STEP 4: Parse numbers
        # Try to convert strings to integers, give clear error if they're not numbers
        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError as err:
            raise ValueError(
                f"Invalid range numbers: '{range_part}'. "
                f"Start and end must be integers (e.g., '1-10')"
            ) from err

        # Special Check for Beta-Turns
        if conformation in BETA_TURN_TYPES:
            if (end - start + 1) != 4:
                raise ValueError(
                    f"Beta-turn '{conformation}' region {start}-{end} must be exactly 4 residues long. "
                    f"Got {end - start + 1}."
                )

        # VALIDATION STEP 5: Check range bounds
        # EDUCATIONAL NOTE - Why These Checks Matter:
        # 1. start < 1: PDB/biology uses 1-based indexing, so 0 or negative makes no sense
        # 2. end > sequence_length: Can't specify residues that don't exist
        # 3. start > end: Range would be backwards (e.g., "10-5"), which is nonsensical
        if start < 1 or end > sequence_length:
            raise ValueError(
                f"Range {start}-{end} is out of bounds for sequence length {sequence_length}. "
                f"Valid range is 1 to {sequence_length}"
            )
        if start > end:
            raise ValueError(
                f"Invalid range: start ({start}) is greater than end ({end}). "
                f"Range must be in format 'smaller-larger' (e.g., '1-10', not '10-1')"
            )

        # VALIDATION STEP 6: Check for overlaps and assign conformations
        # EDUCATIONAL NOTE - Why We Forbid Overlaps:
        # If residue 5 is specified as both "alpha" and "beta", which should we use?
        # Rather than making an arbitrary choice (like "last one wins"), we require
        # the user to be explicit and not specify overlapping regions.
        for res_idx in range(start - 1, end):  # Convert to 0-based indexing
            # Check if this residue was already specified in a previous region
            if res_idx in residue_conformations:
                # EDUCATIONAL NOTE - Error Message Design:
                # We convert back to 1-based indexing in the error message because
                # that's what the user specified. This makes errors easier to understand.
                raise ValueError(
                    f"Overlapping regions detected: residue {res_idx + 1} is specified "
                    f"in multiple regions. Each residue can only have one conformation."
                )

            # Assign the conformation to this residue (using 0-based indexing internally)
            residue_conformations[res_idx] = conformation

    # Return the mapping of residue indices to conformations
    # EDUCATIONAL NOTE - What Happens to Gaps:
    # If a residue index is not in this dictionary, the calling code will use
    # the default conformation specified by the --conformation parameter.
    # This allows users to specify only the interesting regions and let the
    # rest use a sensible default.
    return residue_conformations


def _resolve_conformation_map(
    sequence: list[str],
    conformation: str,
    structure: str | None,
) -> dict[int, str]:
    """Build per-residue conformation map from default and optional structure spec.

    Validates the default conformation, expands the optional per-region structure
    string into a 0-based residue-index dict, and fills gaps with the default.

    Args:
        sequence: List of 3-letter amino acid codes.
        conformation: Default secondary structure for all residues.
        structure: Optional per-region spec (e.g. ``"1-10:alpha,11-20:beta"``).

    Returns:
        Dict mapping 0-based residue index to conformation name.

    """
    sequence_length = len(sequence)
    # EDUCATIONAL NOTE - Input Validation:
    # We validate the default conformation early to give clear error messages.
    # Even if structure parameter overrides it for some residues, we need to
    # ensure the default is valid for any gaps or when structure is not provided.
    valid_conformations = list(RAMACHANDRAN_PRESETS.keys()) + ["random"]
    if conformation not in valid_conformations:
        raise ValueError(
            f"Invalid conformation '{conformation}'. "
            f"Valid options are: {', '.join(valid_conformations)}"
        )
    # EDUCATIONAL NOTE - Per-Residue Conformation Assignment:
    # We now support two modes:
    # 1. Uniform conformation (old behavior): All residues use same conformation
    # 2. Per-region conformation (new!): Different regions can have different conformations

    # Parse per-residue conformations if structure parameter is provided
    if structure:
        residue_conformations = _parse_structure_regions(structure, sequence_length)
        # EDUCATIONAL NOTE - Gap Handling:
        # If a residue is not specified in the structure parameter,
        # we use the default conformation. This allows users to specify
        # only the interesting regions and let the rest use a sensible default.
        for i in range(sequence_length):
            if i not in residue_conformations:
                residue_conformations[i] = conformation
    else:
        residue_conformations = dict.fromkeys(range(sequence_length), conformation)
    # EDUCATIONAL NOTE - Why We Don't Validate Conformations Here:
    # We already validated conformations in _parse_structure_regions(),
    # so we don't need to re-validate them here. The default conformation
    # will be validated when we actually use it below.
    return residue_conformations


def _build_peptide_chains(
    chain_sequences: list[list[str]],
    conformation: str,
    structure: str | None,
    cyclic: bool,
    cis_proline_frequency: float,
    drift: float,
    phi_list: list[float] | None,
    psi_list: list[float] | None,
    omega_list: list[float] | None,
    rng: random.Random,
) -> struc.AtomArray:
    """Construct coordinates for one or more polypeptide chains using NeRF geometry.

    This function performs the step-by-step \"walk\" along the protein backbone for
    each chain, placing each atom relative to the previous ones and assigning
    unique chain identifiers.

    Args:
        chain_sequences: List of lists of 3-letter amino acid codes.
        conformation: Default conformation used as fallback in gap detection.
        structure: Original structure spec string.
        cyclic: Whether to suppress terminal atoms for head-to-tail ring closure.
        cis_proline_frequency: Probability that a PRO residue adopts cis-omega.
        drift: Max uniform drift (degrees) added to phi/psi for hard-decoy mode.
        phi_list: Explicit phi angles (threaded backbone); overrides sampling.
        psi_list: Explicit psi angles (threaded backbone); overrides sampling.
        omega_list: Explicit omega angles; overrides sampling.
        rng: Seeded random generator for reproducible drift.

    Returns:
        Complete :class:`biotite.structure.AtomArray` with multiple chains.

    """
    complex_array = struc.AtomArray(0)
    chain_ids = [chr(65 + i) for i in range(len(chain_sequences))]  # A, B, C...

    # Spatial offset between chains to prevent overlap
    # We translate each subsequent chain by 20.0A in X, Y, and Z.
    offset_step = np.array([20.0, 20.0, 20.0])

    for chain_idx, sequence in enumerate(chain_sequences):
        chain_id = chain_ids[chain_idx]
        sequence_length = len(sequence)

        # Build the per-residue {index: conformation} map for this chain
        residue_conformations = _resolve_conformation_map(sequence, conformation, structure)

        current_chain_array = struc.AtomArray(0)
        residue_coordinates: dict[int, dict[str, np.ndarray]] = {}

        for i, full_res_name in enumerate(sequence):
            res_id = i + 1

            # EDUCATIONAL NOTE - D-Amino Acid Handling:
            # D-amino acids are the mirror images of the standard L-amino acids.
            is_d = full_res_name.startswith("D-")
            res_name = full_res_name[2:] if is_d else full_res_name

            # Determine conformation for this residue
            res_conformation = residue_conformations.get(i, conformation)
            if res_conformation == "alpha" and not structure:
                # Cyclic single chains get curved defaults if not specified
                if cyclic and len(chain_sequences) == 1:
                    res_conformation = "curved"

            # ── Backbone coordinate placement ───────────────────────────────────
            if i == 0:
                # First residue (N-terminus)
                n_coord = np.array([0.0, 0.0, 0.0])
                ca_coord = np.array([BOND_LENGTH_N_CA, 0.0, 0.0])

                angle_with_x = np.pi - ANGLE_N_CA_C_RAD
                c_x = ca_coord[0] + BOND_LENGTH_CA_C * np.cos(angle_with_x)
                c_y = ca_coord[1] + BOND_LENGTH_CA_C * np.sin(angle_with_x)
                c_coord = np.array([c_x, c_y, 0.0])

                if res_conformation in RAMACHANDRAN_PRESETS:
                    current_psi = RAMACHANDRAN_PRESETS[res_conformation]["psi"]
                elif res_conformation == "random":
                    next_res_name = sequence[i + 1] if i + 1 < sequence_length else None
                    _, current_psi = _sample_ramachandran_angles(res_name, next_res_name, rng=rng)
                elif res_conformation in BETA_TURN_TYPES:
                    current_psi = RAMACHANDRAN_PRESETS["extended"]["psi"]
                else:
                    current_psi = -47.0  # Default Alpha

                if is_d:
                    current_psi = -current_psi

            else:
                # Subsequent residues use internal-coordinate (NeRF) placement
                prev_res_idx = i - 1
                prev_coords = residue_coordinates[prev_res_idx]
                prev_n_coord = prev_coords["N"]
                prev_ca_coord = prev_coords["CA"]
                prev_c_coord = prev_coords["C"]

                next_full_res_name = sequence[i + 1] if i + 1 < sequence_length else None
                next_res_name = (
                    next_full_res_name[2:]
                    if next_full_res_name and next_full_res_name.startswith("D-")
                    else next_full_res_name
                )

                prev_conformation = residue_conformations.get(prev_res_idx, conformation)
                prev_is_d = sequence[prev_res_idx].startswith("D-")

                current_phi = None
                current_psi = None

                if prev_conformation in RAMACHANDRAN_PRESETS:
                    prev_psi = RAMACHANDRAN_PRESETS[prev_conformation]["psi"]
                elif prev_conformation in BETA_TURN_TYPES:
                    c_prev = prev_conformation
                    if residue_conformations.get(prev_res_idx - 1) != c_prev:
                        turn_pos = 1
                    elif residue_conformations.get(prev_res_idx - 2) != c_prev:
                        turn_pos = 2
                    elif residue_conformations.get(prev_res_idx - 3) != c_prev:
                        turn_pos = 3
                    else:
                        turn_pos = 4

                    turn_angles = BETA_TURN_TYPES[prev_conformation]
                    if turn_pos == 1:
                        prev_psi = RAMACHANDRAN_PRESETS["extended"]["psi"]
                    elif turn_pos == 2:
                        prev_psi = turn_angles[0][1]
                    elif turn_pos == 3:
                        prev_psi = turn_angles[1][1]
                    else:
                        prev_psi = RAMACHANDRAN_PRESETS["extended"]["psi"]
                elif prev_conformation == "random":
                    prev_base_res_name = sequence[prev_res_idx].replace("D-", "")
                    _, prev_psi = _sample_ramachandran_angles(prev_base_res_name, res_name, rng=rng)
                else:
                    prev_psi = RAMACHANDRAN_PRESETS["alpha"]["psi"]

                if prev_is_d:
                    prev_psi = -prev_psi

                # Place N(i)
                n_coord = _place_atom_with_dihedral(
                    prev_n_coord,
                    prev_ca_coord,
                    prev_c_coord,
                    BOND_LENGTH_C_N,
                    ANGLE_CA_C_N,
                    prev_psi,
                )

                # Place CA(i)
                omega_mean = OMEGA_TRANS
                if res_name == "PRO" and rng.random() < cis_proline_frequency:
                    omega_mean = 0.0

                if omega_list is not None and i > 0 and (i - 1) < len(omega_list):
                    omega = omega_list[i - 1]
                else:
                    omega = rng.normalvariate(omega_mean, OMEGA_VARIATION)

                ca_coord = _place_atom_with_dihedral(
                    prev_ca_coord, prev_c_coord, n_coord, BOND_LENGTH_N_CA, ANGLE_C_N_CA, omega
                )

                # Determine phi/psi for this residue
                if (
                    phi_list is not None
                    and i < len(phi_list)
                    and psi_list is not None
                    and i < len(psi_list)
                ):
                    current_phi = phi_list[i]
                    current_psi = psi_list[i]
                elif res_conformation in RAMACHANDRAN_PRESETS:
                    current_phi = RAMACHANDRAN_PRESETS[res_conformation]["phi"]
                    current_psi = RAMACHANDRAN_PRESETS[res_conformation]["psi"]
                elif res_conformation in BETA_TURN_TYPES:
                    c_curr = res_conformation
                    if residue_conformations.get(i - 1) != c_curr:
                        turn_pos = 1
                    elif residue_conformations.get(i - 2) != c_curr:
                        turn_pos = 2
                    elif residue_conformations.get(i - 3) != c_curr:
                        turn_pos = 3
                    else:
                        turn_pos = 4
                    turn_angles = BETA_TURN_TYPES[res_conformation]
                    if turn_pos == 1:
                        current_phi = RAMACHANDRAN_PRESETS["extended"]["phi"]
                        current_psi = RAMACHANDRAN_PRESETS["extended"]["psi"]
                    elif turn_pos == 2:
                        current_phi = turn_angles[0][0]
                        current_psi = turn_angles[0][1]
                    elif turn_pos == 3:
                        current_phi = turn_angles[1][0]
                        current_psi = turn_angles[1][1]
                    else:
                        current_phi = RAMACHANDRAN_PRESETS["extended"]["phi"]
                        current_psi = RAMACHANDRAN_PRESETS["extended"]["psi"]
                elif res_conformation == "random":
                    current_phi, current_psi = _sample_ramachandran_angles(
                        res_name, next_res_name, rng=rng
                    )
                else:
                    current_phi = RAMACHANDRAN_PRESETS["alpha"]["phi"]
                    current_psi = RAMACHANDRAN_PRESETS["alpha"]["psi"]

                if is_d:
                    if phi_list is None or i >= len(phi_list):
                        current_phi = -current_phi
                    if psi_list is None or i >= len(psi_list):
                        current_psi = -current_psi

                if drift > 0:
                    current_phi += rng.uniform(-drift, drift)
                    current_psi += rng.uniform(-drift, drift)
                    current_phi = ((current_phi + 180) % 360) - 180
                    current_psi = ((current_psi + 180) % 360) - 180

                c_coord = _place_atom_with_dihedral(
                    prev_c_coord, n_coord, ca_coord, BOND_LENGTH_CA_C, ANGLE_N_CA_C, current_phi
                )

            # ── Place Oxygen (O) explicitly ──
            o_dihedral = 180.0
            o_coord = _place_atom_with_dihedral(
                n_coord, ca_coord, c_coord, BOND_LENGTH_C_O, ANGLE_CA_C_O, o_dihedral
            )

            # ── Store coordinates for next iteration ────────────────────────────
            residue_coordinates[i] = {
                "N": n_coord,
                "CA": ca_coord,
                "C": c_coord,
                "O": o_coord,
            }

            # ── Biotite reference template ───────────────────────────────────────
            template_res_name = res_name
            if res_name in ["HID", "HIE", "HIP"]:
                template_res_name = "HIS"
            elif res_name in L_TO_D_MAPPING.values():
                d_to_l = {v: k for k, v in L_TO_D_MAPPING.items()}
                template_res_name = d_to_l[res_name]

            try:
                ref_res_template = struc.info.residue(template_res_name).copy()
            except KeyError:
                logger.warning(f"Residue {res_name} not found in Biotite info, skipping.")
                continue

            # Remove terminal atoms
            if i < sequence_length - 1 or (cyclic and len(chain_sequences) == 1):
                ref_res_template = ref_res_template[ref_res_template.atom_name != "OXT"]
                ref_res_template = ref_res_template[ref_res_template.atom_name != "HXT"]

            if i > 0 or (cyclic and len(chain_sequences) == 1):
                ref_res_template = ref_res_template[ref_res_template.atom_name != "H2"]
                ref_res_template = ref_res_template[ref_res_template.atom_name != "H3"]
                if res_name == "PRO":
                    ref_res_template = ref_res_template[ref_res_template.atom_name != "H"]

            # Rotamer selection
            rotamers = None
            if res_name in BACKBONE_DEPENDENT_ROTAMER_LIBRARY:
                if res_conformation in BACKBONE_DEPENDENT_ROTAMER_LIBRARY[res_name]:
                    rotamers = BACKBONE_DEPENDENT_ROTAMER_LIBRARY[res_name][res_conformation]
            if rotamers is None and res_name in ROTAMER_LIBRARY:
                rotamers = ROTAMER_LIBRARY[res_name]

            if rotamers:
                weights = [float(r.get("prob", 0.0)) for r in rotamers]  # type: ignore[arg-type]
                selected_rotamer = rng.choices(rotamers, weights=weights, k=1)[0]

                if "chi1" in selected_rotamer:
                    _chi1_val = selected_rotamer["chi1"]
                    chi1_target = _chi1_val[0] if isinstance(_chi1_val, list) else float(_chi1_val)

                    gamma_atom_name = None
                    for candidate in ["CG", "CG1", "OG", "OG1", "SG"]:
                        if len(ref_res_template[ref_res_template.atom_name == candidate]) > 0:
                            gamma_atom_name = candidate
                            break

                    if gamma_atom_name:
                        ca_atom = ref_res_template[ref_res_template.atom_name == "CA"][0]
                        cb_atom = ref_res_template[ref_res_template.atom_name == "CB"][0]
                        n_atom = ref_res_template[ref_res_template.atom_name == "N"][0]
                        g_atom = ref_res_template[ref_res_template.atom_name == gamma_atom_name][0]

                        current_chi1 = calculate_dihedral_angle(
                            n_atom.coord, ca_atom.coord, cb_atom.coord, g_atom.coord
                        )
                        diff_deg = chi1_target - current_chi1

                        backbone_names = {"N", "CA", "C", "O", "H", "HA", "CB"}
                        for atom_idx in range(len(ref_res_template)):
                            if ref_res_template.atom_name[atom_idx] not in backbone_names:
                                p = ref_res_template.coord[atom_idx]
                                v = cb_atom.coord - ca_atom.coord
                                v /= np.linalg.norm(v)
                                alpha = np.deg2rad(diff_deg)
                                cos_a = np.cos(alpha)
                                sin_a = np.sin(alpha)
                                rel_p = p - ca_atom.coord
                                rotated_p = (
                                    rel_p * cos_a
                                    + np.cross(v, rel_p) * sin_a
                                    + v * np.dot(v, rel_p) * (1 - cos_a)
                                )
                                ref_res_template.coord[atom_idx] = rotated_p + ca_atom.coord

            # ── Superimpose template onto constructed backbone frame ─────────────
            template_backbone_n = ref_res_template[ref_res_template.atom_name == "N"]
            template_backbone_ca = ref_res_template[ref_res_template.atom_name == "CA"]
            template_backbone_c = ref_res_template[ref_res_template.atom_name == "C"]
            mobile_backbone_from_template = (
                template_backbone_n + template_backbone_ca + template_backbone_c
            )

            if len(mobile_backbone_from_template) != 3:
                raise ValueError(
                    f"Reference residue template for {res_name} is missing required "
                    f"backbone atoms (N, CA, C) for superimposition. "
                    f"Found atoms: {list(mobile_backbone_from_template.atom_name)}"
                )

            target_backbone_constructed = struc.array(
                [
                    struc.Atom(
                        n_coord,
                        atom_name="N",
                        res_id=res_id,
                        res_name=res_name,
                        element="N",
                        hetero=False,
                    ),
                    struc.Atom(
                        ca_coord,
                        atom_name="CA",
                        res_id=res_id,
                        res_name=res_name,
                        element="C",
                        hetero=False,
                    ),
                    struc.Atom(
                        c_coord,
                        atom_name="C",
                        res_id=res_id,
                        res_name=res_name,
                        element="C",
                        hetero=False,
                    ),
                ]
            )

            _, transformation = struc.superimpose(
                target_backbone_constructed, mobile_backbone_from_template
            )
            transformed_res = ref_res_template
            transformed_res.coord = transformation.apply(transformed_res.coord)
            transformed_res.hetero[:] = False

            o_mask = transformed_res.atom_name == "O"
            if o_mask.any():
                transformed_res.coord[o_mask] = o_coord

            # D-amino acid chiral mirroring
            if is_d and res_name != "GLY":
                vec1 = c_coord - ca_coord
                vec2 = n_coord - ca_coord
                normal = np.cross(vec1, vec2)
                normal /= np.linalg.norm(normal)
                backbone_names = {"N", "CA", "C"}
                for atom_idx in range(len(transformed_res)):
                    if transformed_res.atom_name[atom_idx] not in backbone_names:
                        p = transformed_res.coord[atom_idx]
                        w = p - ca_coord
                        dist_to_plane = np.dot(w, normal)
                        transformed_res.coord[atom_idx] = p - 2 * dist_to_plane * normal

            # ── Assign residue metadata ──────────────────────────────────────────
            transformed_res.res_id[:] = res_id
            if is_d:
                transformed_res.res_name[:] = L_TO_D_MAPPING.get(res_name, res_name)
            else:
                transformed_res.res_name[:] = res_name
            transformed_res.chain_id[:] = chain_id

            if i == 0:
                current_chain_array = transformed_res.copy()
            else:
                current_chain_array += transformed_res

        # Apply spatial offset to the whole chain
        current_chain_array.coord += chain_idx * offset_step

        if chain_idx == 0:
            complex_array = current_chain_array.copy()
        else:
            complex_array += current_chain_array

    return complex_array


def _apply_biophysical_mods(
    peptide: struc.AtomArray,
    optimize_sidechains: bool,
    cap_termini: bool,
    cyclic: bool,
    ph: float,
    metal_ions: str,
    rng: random.Random | None = None,
) -> tuple[struc.AtomArray, list[dict]]:
    """Apply post-construction biophysical modifications in-place on a copy.

    Order of application matches the original generator logic:
    1. Monte Carlo sidechain optimization (if requested)
    2. ACE/NME terminal capping (disabled for cyclic peptides)
    3. pH-dependent protonation state titration
    4. Automatic metal-ion injection (Zn coordination motifs)

    Args:
        peptide: Input AtomArray.
        optimize_sidechains: Run Monte Carlo rotamer packing.
        cap_termini: Add ACE/NME caps (ignored when ``cyclic=True``).
        cyclic: Skip capping for cyclic peptides.
        ph: Solution pH for histidine protonation state assignment.
        metal_ions: ``'auto'`` detects Zn coordination sites; any other value skips.

    Returns:
        Tuple[struc.AtomArray, List[Dict]]: Modified AtomArray and coordination sites.

    """
    # EDUCATIONAL NOTE - Biophysical Realism (Phase 2):
    # We apply chemical modifications after geometric construction/packing but BEFORE
    # energy minimization. Correct protonation states (pH) are critical for
    # correct electrostatics in the forcefield.

    # EDUCATIONAL NOTE - Side-Chain Optimization:
    # If requested, run Monte Carlo optimization to fix steric clashes.
    # This is "Phase 1" of biophysical realism.
    if optimize_sidechains:
        logger.info("Running side-chain optimization...")
        peptide = run_optimization(peptide)

    # 1. Terminal Capping (ACE/NME) — cyclic peptides are naturally capped.
    if cap_termini and not cyclic:
        peptide = biophysics.cap_termini(peptide)

    # 2. pH Titration (Protonation States)
    peptide = biophysics.apply_ph_titration(peptide, ph=ph, rng=rng)

    # EDUCATIONAL NOTE - Metal Ion Coordination (Phase 15):
    # Inorganic cofactors like Zinc (Zn2+) are automatically detected.
    # If a coordination motif is found (Cys/His clusters), the ion is
    # injected and harmonic constraints are applied in the physics module.
    sites: list[dict] = []
    if metal_ions == "auto":
        from .cofactors import add_metal_ion, find_metal_binding_sites

        sites = find_metal_binding_sites(peptide)
        for site in sites:
            peptide = add_metal_ion(peptide, site)

    return peptide, sites


def _do_energy_minimization(
    peptide: struc.AtomArray,
    sequence: list[str],
    forcefield: str,
    minimization_k: float,
    minimization_max_iter: int,
    cyclic: bool,
    equilibrate: bool,
    equilibrate_steps: int,
    solvent_model: str,
    solvent_padding: float,
    keep_solvent: bool,
    coordination: list[dict] | None = None,
) -> tuple[str | None, struc.AtomArray]:
    """Run OpenMM energy minimization (or MD equilibration) and return results.

    Writes a temporary PDB, invokes :class:`.EnergyMinimizer`, and reads the
    optimized structure back.  PTM residue names (SEP, TPO, PTR) that were
    reverted to their parent types for forcefield compatibility are restored.

    Args:
        peptide: Un-minimized AtomArray.
        sequence: Original amino-acid sequence list (used for PTM restoration).
        forcefield: Forcefield XML name passed to OpenMM (e.g. ``'amber14-all.xml'``).
        minimization_k: Energy convergence tolerance (kJ/mol).
        minimization_max_iter: Max minimization iterations (0 = until convergence).
        cyclic: If ``True``, caps termini before writing temp PDB for OpenMM.
        equilibrate: Run MD equilibration instead of pure minimization.
        equilibrate_steps: Number of 2 fs MD steps for equilibration.
        coordination: Optional metal coordination sites.

    Returns:
        ``(atomic_and_ter_content, updated_peptide)`` where *atomic_and_ter_content*
        is the raw PDB string from OpenMM (or ``None`` on failure) and
        *updated_peptide* is the (possibly updated) AtomArray.

    """
    # EDUCATIONAL NOTE - Energy Minimization (Phase 2):
    # OpenMM requires a file-based interaction for easy topology handling from PDB.
    # So we write the current state to a temp file, minimize it, and read it back.
    logger.info("Running energy minimization (OpenMM)...")
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_pdb_path = os.path.join(tmpdirname, "pre_min.pdb")
            output_pdb_path = os.path.join(tmpdirname, "minimized.pdb")

            # CRITICAL Fix for OpenMM: cyclic peptides need terminal caps so
            # Amber template-matching works, then physics.py prunes them.
            if cyclic:
                peptide_to_save = biophysics.cap_termini(peptide)
            else:
                peptide_to_save = peptide[peptide.element != "H"]

            pdb_file_write = pdb.PDBFile()
            pdb_file_write.set_structure(peptide_to_save)
            pdb_file_write.write(input_pdb_path)

            minimizer = EnergyMinimizer(
                forcefield_name=forcefield, solvent_model=solvent_model, box_size=solvent_padding
            )
            current_disulfides = _detect_disulfide_bonds(peptide)

            if equilibrate:
                logger.info(
                    f"Running MD Equilibration ({equilibrate_steps} steps). "
                    "This includes minimization."
                )
                success = minimizer.equilibrate(
                    input_pdb_path,
                    output_pdb_path,
                    steps=equilibrate_steps,
                    cyclic=cyclic,
                    disulfides=current_disulfides,
                )
            else:
                success = minimizer.add_hydrogens_and_minimize(
                    input_pdb_path,
                    output_pdb_path,
                    max_iterations=minimization_max_iter,
                    tolerance=minimization_k,
                    cyclic=cyclic,
                    disulfides=current_disulfides,
                    coordination=coordination,
                )

            if not success:
                logger.error("Minimization failed. Returning un-minimized structure.")
                return None, peptide

            logger.info("Minimization/Equilibration successful.")
            with open(output_pdb_path) as f:
                atomic_and_ter_content = f.read()

            pdb_file_read = pdb.PDBFile.read(output_pdb_path)
            peptide = pdb_file_read.get_structure(model=1)

            # EDUCATIONAL NOTE - Explicit Solvent Pruning:
            # When researchers request explicit solvent (TIP3P), OpenMM simulates the peptide
            # floating in a dense box of thousands of water molecules.
            # While this is biophysically accurate during minimization, leaving 10,000+
            # waters in a PDB makes it extraordinarily difficult to parse or feed into AI models.
            #
            # Thus, if the user did not specify --keep-solvent, we programmatically strip
            # out the explicit water molecules (HOH residue name) here, leaving only the
            # high-quality, physics-refined protein coordinates.
            if solvent_model == "explicit" and not keep_solvent:
                logger.info(
                    "Stripping thousands of explicit water molecules (HOH) from the final structure for a cleaner PDB..."
                )
                peptide = peptide[peptide.res_name != "HOH"]

                # We also need to regenerate the atomic_and_ter_content from the pruned peptide,
                # otherwise the raw OpenMM output string (with full water box) will be saved!
                peptide.atom_id = np.arange(1, peptide.array_length() + 1)
                pdb_file_out = pdb.PDBFile()
                pdb_file_out.set_structure(peptide)
                string_io = io.StringIO()
                pdb_file_out.write(string_io)
                atomic_and_ter_content = string_io.getvalue()

            # RESTORE PTM NAMES (Fix for "Missing Orange Balls"):
            # OpenMM reverted SEP→SER etc.; restore for downstream viewers.
            # We must rename in BOTH the AtomArray (peptide.res_name) AND in the
            # raw PDB string (atomic_and_ter_content) — the final output is assembled
            # from the string, so only patching the array leaves PTMs silently absent.
            try:
                unique_res_ids = np.unique(peptide.res_id)
                n_seq = len(sequence)
                n_min = len(unique_res_ids)
                start_offset = 0
                if n_min > 0:
                    first_res_id_local = unique_res_ids[0]
                    mask_first = peptide.res_id == first_res_id_local
                    if np.any(mask_first):
                        first_res_name_local = peptide.res_name[mask_first][0]
                        if first_res_name_local == "ACE":
                            logger.info("Detected N-terminal ACE cap. Applying start offset of 1.")
                            start_offset = 1

                # Build a mapping: PDB residue-number → new PTM name
                ptm_rename_map: dict = {}  # {res_id_int: "SEP"/"TPO"/...}
                if n_min >= n_seq + start_offset:
                    for idx, res_name_target in enumerate(sequence):
                        rid = unique_res_ids[idx + start_offset]
                        if res_name_target in NON_STANDARD_RESIDUES:
                            mask = peptide.res_id == rid
                            peptide.res_name[mask] = res_name_target
                            ptm_rename_map[int(rid)] = res_name_target

                # Apply the same renames to the raw PDB string so the final
                # assembled output contains the correct PTM residue names.
                if ptm_rename_map:
                    patched_lines = []
                    for line in atomic_and_ter_content.splitlines():
                        if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
                            try:
                                line_res_id = int(line[22:26].strip())
                                if line_res_id in ptm_rename_map:
                                    new_name = f"{ptm_rename_map[line_res_id]:>3}"
                                    line = line[:17] + new_name + line[20:]
                            except (ValueError, IndexError):
                                pass
                        patched_lines.append(line)
                    atomic_and_ter_content = "\n".join(patched_lines) + "\n"

            except Exception as ptm_err:
                logger.warning(f"Failed to restore PTM names: {ptm_err}")

            return atomic_and_ter_content, peptide

    except Exception as e:
        logger.error(f"Error during minimization workflow: {e}")
        return None, peptide


def _assemble_pdb_output(
    peptide: struc.AtomArray,
    atomic_and_ter_content: str | None,
    sequence_length: int,
    cyclic: bool,
    rng: random.Random | None = None,
) -> str:
    """Assemble the final PDB string with realistic B-factors, occupancy, and records.

    If *atomic_and_ter_content* is ``None`` (no minimization was run), generates
    the raw PDB from the Biotite AtomArray first.  Then post-processes every
    ATOM/HETATM line to insert order-parameter-derived B-factors and occupancy,
    adds a TER record if missing, generates CONECT records for cyclic and
    disulfide bonds, and delegates final assembly to :func:`.assemble_pdb_content`.

    Args:
        peptide: Final AtomArray (minimized or raw).
        atomic_and_ter_content: Raw ATOM-block PDB string from OpenMM, or ``None``.
        sequence_length: Number of residues (used for PDB header metadata).
        cyclic: Whether to add a head-to-tail CONECT record.

    Returns:
        Complete PDB file as a string.

    """
    # ── Step 1: Initialize Atomic Block ─────────────────────────────────────
    # If no minimization was performed, we convert the AtomArray directly to PDB format.
    if atomic_and_ter_content is None:
        peptide.atom_id = np.arange(1, peptide.array_length() + 1)
        pdb_file_out = pdb.PDBFile()
        pdb_file_out.set_structure(peptide)
        string_io = io.StringIO()
        pdb_file_out.write(string_io)
        atomic_and_ter_content = string_io.getvalue()

    # EDUCATIONAL NOTE - Adding Realistic B-factors:
    # Biotite sets all B-factors to 0.00 by default. We post-process the PDB string
    # to replace these with realistic values based on atom type, position, and residue type.
    # This makes the output look more professional and realistic.

    # EDUCATIONAL NOTE - Adding Realistic Occupancy:
    # Similarly, biotite sets all occupancy values to 1.00. We calculate realistic
    # occupancy values (0.85-1.00) that correlate with B-factors and reflect disorder.
    protein_res_ids = peptide.res_id[~peptide.hetero]
    protein_res_ids_set = set(protein_res_ids)
    max_protein_res_id = np.max(protein_res_ids) if len(protein_res_ids) > 0 else 0
    total_residues = np.max(peptide.res_id) if len(peptide.res_id) > 0 else 0

    # Predict Order Parameters (S2) to derive B-factors from geometric flexibility.

    s2_map = predict_order_parameters(peptide)

    # Sanitize: Extract only atomic content to avoid header duplication
    atomic_and_ter_content = extract_atomic_content(atomic_and_ter_content)

    processed_lines = []
    n_term_serial = None
    c_term_serial = None
    sg_serials: dict[int, int] = {}
    serial = 0  # tracks last serial of any ATOM/HETATM line
    last_atom_serial = 0  # tracks last serial of ATOM (polymer) lines only
    last_protein_atom_line: str | None = None  # last ATOM line text

    # ── Step 2: Line-by-Line Refinement ────────────────────────────────────
    # We iterate over every ATOM line to apply biophysical metadata.
    for line in atomic_and_ter_content.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            res_num = int(line[22:26].strip())

            # EDUCATIONAL NOTE - HETATM vs ATOM:
            # OpenMM's PDB writer may output non-standard amino acids (like PTMs)
            # as HETATM. We forcefully convert them back to ATOM if they are part
            # of the main polymer chain to ensure compatibility with standard viewers.
            if res_num in protein_res_ids_set and line.startswith("HETATM"):
                line = "ATOM  " + line[6:]

            serial = int(line[6:11].strip())
            if line.startswith("ATOM"):
                last_atom_serial = serial
                last_protein_atom_line = line
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()

            # Identify terminal atoms for CONECT record generation in cyclic peptides.
            if cyclic:
                if res_num == 1 and atom_name == "N":
                    n_term_serial = serial
                if res_num == max_protein_res_id and atom_name == "C":
                    c_term_serial = serial

            # Track sulfur atom serials for disulfide bond CONECT records.
            if (res_name == "CYS" or res_name == "CYX") and atom_name == "SG":
                sg_serials[res_num] = serial

            # Calculate and inject realistic B-factor and Occupancy values.
            current_s2 = s2_map.get(res_num, 0.85)
            bfactor = _calculate_bfactor(
                atom_name, res_num, total_residues, res_name, s2=current_s2, rng=rng
            )
            occupancy = _calculate_occupancy(
                atom_name, res_num, total_residues, res_name, bfactor, rng=rng
            )

            # Patch the PDB line with the new floating-point values.
            line = line[:54] + f"{occupancy:6.2f}" + f"{bfactor:6.2f}" + line[66:]

        processed_lines.append(line)

    atomic_and_ter_content = "\n".join(processed_lines) + "\n"

    # ── Step 3: PDB Termination (TER) ──────────────────────────────────────
    # Ensure a proper TER record exists to separate polymer from ligands/solvent.
    lines = atomic_and_ter_content.strip().splitlines()
    if not lines:
        logger.error("Generated PDB content is empty! Falling back to raw sequence string.")
        return atomic_and_ter_content

    has_ter = any(line_str.startswith("TER") for line_str in lines)
    if not has_ter and last_protein_atom_line is not None:
        # PDB standard: TER follows the last atom of the polymer chain.
        ter_atom_num = last_atom_serial + 1
        ter_res_name = last_protein_atom_line[17:20].strip()
        ter_chain_id = last_protein_atom_line[21].strip()
        ter_res_num = int(last_protein_atom_line[22:26].strip())

        # Build a standard 80-character TER record.
        ter_record = (
            f"TER   {ter_atom_num: >5}      {ter_res_name: >3} "
            f"{ter_chain_id: <1}{ter_res_num: >4}"
        ).ljust(80)

        # Splicing: Insert TER immediately after the polymer ATOM block.
        insert_lines = atomic_and_ter_content.strip().splitlines()
        last_atom_idx = max(
            i for i, line_str in enumerate(insert_lines) if line_str.startswith("ATOM")
        )
        insert_lines.insert(last_atom_idx + 1, ter_record)
        atomic_and_ter_content = "\n".join(insert_lines) + "\n"

    # Ensure all lines are padded to 80 characters for legacy PDB compatibility.
    padded_lines = [line.ljust(80) for line in atomic_and_ter_content.splitlines()]
    final_atomic_content_block = "\n".join(padded_lines).strip()

    # ── Step 4: Connectivity (CONECT & SSBOND) ─────────────────────────────
    # Generate records for cyclic peptides and disulfide bonds.
    conect_records = []
    if cyclic and n_term_serial and c_term_serial:
        conect_records.append(f"CONECT{n_term_serial:5d}{c_term_serial:5d}".ljust(80))

    disulfides = _detect_disulfide_bonds(peptide)
    # SSBOND records specifically annotate covalent sulfur-sulfur bridges.
    ssbond_records = _generate_ssbond_records(disulfides, chain_id="A")

    if disulfides:
        for r1, r2 in disulfides:
            s1 = sg_serials.get(r1)
            s2 = sg_serials.get(r2)
            if s1 and s2:
                # Disulfide connectivity requires bidirectional records.
                conect_records.append(f"CONECT{s1:5d}{s2:5d}".ljust(80))
                conect_records.append(f"CONECT{s2:5d}{s1:5d}".ljust(80))

    conect_block = "\n".join(conect_records)
    if conect_block:
        conect_block += "\n"

    # Final assembly using the project-standard utility.
    return assemble_pdb_content(
        final_atomic_content_block,
        sequence_length,
        command_args=None,
        extra_records=ssbond_records if ssbond_records else None,
        conect_records=conect_block if conect_block else None,
    )


def generate_pdb_content(
    length: int | None = None,
    sequence_str: str | None = None,
    use_plausible_frequencies: bool = False,
    conformation: str = "alpha",
    structure: str | None = None,
    optimize_sidechains: bool = False,
    minimize_energy: bool = False,
    forcefield: str = "amber14-all.xml",
    solvent_model: str = "obc2",
    solvent_padding: float = 1.0,
    keep_solvent: bool = False,
    seed: int | None = None,
    ph: float = 7.4,
    cap_termini: bool = False,
    equilibrate: bool = False,
    equilibrate_steps: int = 1000,
    metal_ions: str = "auto",
    minimization_k: float = 10.0,  # Tolerance
    minimization_max_iter: int = 0,  # 0 = unlimited
    cis_proline_frequency: float = 0.05,  # Probability of Cis-Proline (0.05 = 5%)
    phosphorylation_rate: float = 0.0,  # Probability of S/T/Y phosphorylation
    cyclic: bool = False,  # Head-to-Tail cyclization
    drift: float = 0.0,  # Torsion angle perturbation in degrees
    phi_list: list[float] | None = None,  # Explicit Phi angles
    psi_list: list[float] | None = None,  # Explicit Psi angles
    omega_list: list[float] | None = None,  # Explicit Omega angles
) -> str:
    r"""Generates PDB content for a linear or cyclic peptide chain.

    EDUCATIONAL NOTE - New Feature: Cyclic Peptides
    Cyclic peptides have their N-terminus bonded to their C-terminus.
    This modification increases metabolic stability and is common in
    therapeutic peptides (e.g., Cyclosporin).

    Args:
        length: Number of residues (ignored if sequence_str provided)
        sequence_str: Explicit amino acid sequence (1-letter or 3-letter codes)
        use_plausible_frequencies: Use biologically realistic amino acid frequencies
        conformation: Default secondary structure conformation.
                     Options: 'alpha', 'beta', 'ppii', 'extended', 'random'
                     Default: 'alpha' (alpha helix)
                     Used for all residues if structure is not provided,
                     or for residues not specified in structure parameter.
        structure: Per-region conformation specification (NEW!)
                  Format: "start-end:conformation,start-end:conformation,..."
                  Example: "1-10:alpha,11-15:random,16-30:beta"
                  If provided, overrides conformation for specified regions.
                  Unspecified residues use the default conformation parameter.
        drift: Maximum random perturbation applied to phi/psi angles (degrees).
               Used for "hard decoy" generation to create near-native conformations.
        optimize_sidechains: Run Monte Carlo side-chain optimization
        minimize_energy: Run OpenMM energy minimization (REQUIRED for cyclic closure)
        forcefield: Forcefield to use for minimization
        seed: Random seed for reproducible generation
        ph: pH for titration
        cap_termini: Add ACE/NME caps (Disabled if cyclic=True)
        equilibrate: Run MD equilibration
        equilibrate_steps: Number of MD steps
        metal_ions: Handle metal ions
        minimization_k: Tolerance
        minimization_max_iter: Max iterations
        cis_proline_frequency: Frequency of cis-proline
        phosphorylation_rate: Frequency of phosphorylation
        cyclic: Whether to generate a cyclic peptide (Head-to-Tail)

    Returns:
        str: Complete PDB file content

    Raises:
        ValueError: If invalid conformation name or structure syntax provided

    EDUCATIONAL NOTE - Why Per-Region Conformations Matter:
    Real proteins have mixed secondary structures. For example:
    - Zinc fingers: beta sheets + alpha helices
    - Immunoglobulins: multiple beta sheets connected by loops
    - Helix-turn-helix motifs: two alpha helices connected by a turn
    This feature allows users to create these realistic structures.

    EDUCATIONAL NOTE - Macrocyclization (Cyclic Peptides):
    -----------------------------------------------------
    Cyclic peptides (macrocycles) are chains where the N-terminus and C-terminus
    are covalently linked. This has profound biological implications:
    1. Metabolic Stability: Resistance to exopeptidases that chew protein ends.
    2. Binding Affinity: By "locking" the molecule into a specific shape,
       the entropic penalty of binding to a target is greatly reduced.
    3. Bioavailability: Many legendary drugs (like Cyclosporine A) are macrocycles.

    EDUCATIONAL NOTE - Multi-Chain Complex Generation (Phase 16):
    ----------------------------------------------------------
    A major frontier in structural biology is the study of the "Interactome"—how
    individual proteins assemble into complexes. This generator supports the creation
    of dimers, trimers, and larger multimers by accepting multiple sequences
    separated by a colon (':').

    Spatial Rationale:
    When generating complexes, each chain is placed in its own coordinate frame.
    We apply a deterministic spatial offset to each chain based on its index
    to ensure that they do not overlap initially. This allows users to then
    perform energy minimization or manual docking to find the native interface.

    EDUCATIONAL NOTE - Hard Decoy Support (AI Training):
    ---------------------------------------------------
    This generator includes specialized parameters for "Hard Decoy" generation:
    1. **Torsion Drift (`drift`)**: Adds controlled Gaussian noise to ideal $\\phi/\\psi$
       angles. This simulates "near-native" local structural errors that
       challenge the resolution of AI scoring functions.
    2. **Threading (`phi_list`, `psi_list`, `omega_list`)**: Allows constructing
       one sequence using the backbone torsion angles of another. This maps a
       "wrong" sequence to a "right" fold, a key test for discriminative models.

    """
    if seed is not None and str(seed) != "<no value>":
        logger.info(f"Setting random seed to {seed} for reproducibility.")
        random.seed(seed)
        np.random.seed(seed)

    # Use localized random generator for better control and thread-safety
    # Ensure seed is None if it's <no value> for random.Random initialization
    actual_seed = seed if str(seed) != "<no value>" else None
    rng = random.Random(actual_seed)

    # 1. Resolve sequences for one or more chains
    chain_sequences = []
    if sequence_str and ":" in sequence_str:
        # Multichain sequence string: "ALA:GLY"
        chain_strs = sequence_str.split(":")
        for c_str in chain_strs:
            if not c_str.strip():
                continue
            chain_sequences.append(
                _resolve_sequence(
                    length=None,
                    user_sequence_str=c_str,
                    use_plausible_frequencies=use_plausible_frequencies,
                    rng=rng,
                )
            )
    else:
        # Single chain
        chain_sequences = [
            _resolve_sequence(
                length=length,
                user_sequence_str=sequence_str,
                use_plausible_frequencies=use_plausible_frequencies,
                rng=rng,
            )
        ]

    # EDUCATIONAL NOTE - Post-Translational Modifications (PTMs):
    # Phosphorylation converts SER/THR/TYR to SEP/TPO/PTR before geometry build.
    if phosphorylation_rate > 0:
        for chain_idx, sequence in enumerate(chain_sequences):
            modified_sequence = []
            for aa in sequence:
                if aa == "SER" and rng.random() < phosphorylation_rate:
                    modified_sequence.append("SEP")
                elif aa == "THR" and rng.random() < phosphorylation_rate:
                    modified_sequence.append("TPO")
                elif aa == "TYR" and rng.random() < phosphorylation_rate:
                    modified_sequence.append("PTR")
                else:
                    modified_sequence.append(aa)
            chain_sequences[chain_idx] = modified_sequence

    if not any(chain_sequences):
        if sequence_str is not None and len(sequence_str) == 0:
            raise ValueError("Provided sequence string cannot be empty.")
        raise ValueError(
            "Length must be a positive integer when no sequence is provided "
            "and no valid sequence string is given."
        )

    # Build backbone + sidechains for all chains via NeRF geometry
    peptide = _build_peptide_chains(
        chain_sequences,
        conformation,
        structure,
        cyclic,
        cis_proline_frequency,
        drift,
        phi_list,
        psi_list,
        omega_list,
        rng,
    )

    # Sidechain optimization, terminal capping, pH titration, metal ions
    peptide, sites = _apply_biophysical_mods(
        peptide,
        optimize_sidechains,
        cap_termini,
        cyclic,
        ph,
        metal_ions,
        rng=rng,
    )

    # Optional energy minimization / MD equilibration via OpenMM
    atomic_and_ter_content: str | None = None
    if minimize_energy:
        # We pass a flattened sequence for energy minimization (topology)
        flat_sequence = []
        for seq in chain_sequences:
            flat_sequence.extend(seq)

        atomic_and_ter_content, peptide = _do_energy_minimization(
            peptide,
            flat_sequence,
            forcefield,
            minimization_k,
            minimization_max_iter,
            cyclic,
            equilibrate,
            equilibrate_steps,
            solvent_model,
            solvent_padding,
            keep_solvent,
            coordination=sites,
        )

    # Assemble final PDB with B-factors, occupancy, TER, CONECT records
    total_res_count = sum(len(s) for s in chain_sequences)
    return _assemble_pdb_output(peptide, atomic_and_ter_content, total_res_count, cyclic, rng=rng)


class PeptideGenerator:
    """Object-oriented wrapper for protein structure generation.

    This class provides a stateful interface for generating synthetic
    protein structures. It allows users to pre-configure generation
    parameters (like forcefields or PTM rates) and then generate
    multiple structures from the same configuration.

    EDUCATIONAL RATIONALE:
    Encapsulating the generation logic in a class makes it easier
    to manage complex experiments, such as generating an ensemble
    of decoys with varying levels of torsion drift.
    """

    def __init__(self, sequence: str = "ALA-GLY-SER", **kwargs: Any) -> None:
        """Initialize the generator with a target sequence and config."""
        self.sequence = sequence
        self.config = kwargs
        self._last_result: PeptideResult | None = None

    def generate(self, **overrides: Any) -> "PeptideResult":
        """Generates the protein structure and returns a Result object.

        Supports on-the-fly overrides for any configuration parameter.
        """
        # Merge init config with call-time overrides
        call_config = {**self.config, **overrides}

        # Call the functional generator which handles multichain splitting
        pdb_content = generate_pdb_content(sequence_str=self.sequence, **call_config)

        # Package into a Result object for easy access to structures
        self._last_result = PeptideResult(pdb_content)
        return self._last_result

    def generate_ensemble(self, n_models: int = 10, as_stack: bool = True, **overrides: Any) -> Any:
        """Generates a structural ensemble using the high-performance BatchedGenerator.

        Args:
            n_models: Number of models to generate.
            as_stack: If True, returns a Biotite AtomArrayStack. If False,
                      returns a list of PeptideResult objects.
            **overrides: On-the-fly configuration overrides.

        Returns:
            biotite.structure.AtomArrayStack or List[PeptideResult].
        """
        config = {**self.config, **overrides}
        conformation = config.get("conformation", "alpha")
        drift = config.get("drift", 0.0)
        seed = config.get("seed")

        # BatchedGenerator is 10-100x faster for ensembles than looping generate()
        bg = BatchedGenerator(self.sequence, n_batch=n_models, full_atom=True)
        batch = bg.generate_batch(seed=seed, conformation=conformation, drift=drift)

        if as_stack:
            stack = batch.to_stack()
            # Explicitly clear templates to free memory
            bg.clear_cache()
            import gc

            gc.collect()
            return stack

        # Fallback to list of results for backward compatibility or unique formatting
        results = []
        for i in range(n_models):
            results.append(PeptideResult(batch.to_pdb(i)))

        # Explicitly clear templates to free memory
        bg.clear_cache()
        import gc

        gc.collect()
        return results


class PeptideResult:
    """Container for generation outputs, providing easy access to
    PDB strings, Biotite structures, and metadata.

    BIOPHYSICAL SIGNIFICANCE:
    Storing the result in a structured container allows for seamless
    integration with downstream analysis modules like NMR predictors
    or interface validators.
    """

    def __init__(self, pdb_content: str) -> None:
        """Initialize the result container with a PDB-formatted string."""
        self.pdb = pdb_content
        self._structure = None

    @property
    def structure(self) -> struc.AtomArray:
        """Returns the structure as a Biotite AtomArray (cached).

        This property lazily parses the PDB string into a high-level
        biotite structure object for geometric analysis.
        """
        if self._structure is None:
            # Defensive check for empty or invalid PDB content
            if not self.pdb or ("ATOM" not in self.pdb and "HETATM" not in self.pdb):
                logger.error("PeptideResult contains no atomic data. Returning empty AtomArray.")
                self._structure = struc.AtomArray(0)
                return self._structure

            # We use io.StringIO to avoid unnecessary disk I/O
            import io

            from biotite.structure.io.pdb import PDBFile

            f = PDBFile.read(io.StringIO(self.pdb))

            # Additional check: ensure at least one model exists in the PDB
            if f.get_model_count() == 0:
                logger.error("PDB content has 0 models. Returning empty AtomArray.")
                self._structure = struc.AtomArray(0)
            else:
                self._structure = f.get_structure(model=1)
        return self._structure

    def save(self, path: str) -> None:
        """Saves the generated PDB content to a permanent file on disk."""
        with open(path, "w") as f:
            f.write(self.pdb)

    def __repr__(self) -> str:
        """Structured representation of the peptide result."""
        return f"<PeptideResult: {len(self.pdb)} chars, {self.structure.array_length()} atoms>"
