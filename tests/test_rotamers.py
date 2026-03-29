from collections import Counter

import synth_pdb.generator


def get_chi1_angle(peptide, res_id, res_name="VAL"):
    """Helper to calculate Chi1 angle (N-CA-CB-Gamma) from a generated structure."""
    # Determine Gamma atom name based on residue
    gamma_map = {
        "VAL": "CG1",  # or CG2
        "ILE": "CG1",
        "THR": "OG1",
        "LEU": "CG",
        "PHE": "CG",
        "TYR": "CG",
        "TRP": "CG",
        "LYS": "CG",
        "SER": "OG",
        "CYS": "SG",
        "MET": "CG",  # standard
        "ASP": "CG",
        "GLU": "CG",
        "GLN": "CG",
        "ARG": "CG",
        "ASN": "CG",
        "HIS": "CG",
    }
    g_name = gamma_map.get(res_name, "CG")

    # Get atoms
    try:
        n = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "N")][0]
        ca = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CA")][0]
        cb = peptide[(peptide.res_id == res_id) & (peptide.atom_name == "CB")][0]
        gamma = peptide[(peptide.res_id == res_id) & (peptide.atom_name == g_name)][0]
    except IndexError:
        return None

    # Calculate dihedral
    from synth_pdb.geometry import calculate_dihedral_angle

    angle = calculate_dihedral_angle(n.coord, ca.coord, cb.coord, gamma.coord)
    return angle


def sample_rotamer_distribution(conformation, res_name="VAL", n_samples=30):
    """Generates N peptides of Length 1 (Just the residue) with a specific conformation
    and returns the list of observed Chi1 angles.
    """
    angles = []
    seq = f"ALA-{res_name}-ALA"

    for _i in range(n_samples):
        pdb_content = synth_pdb.generator.generate_pdb_content(
            sequence_str=seq, conformation=conformation, minimize_energy=False  # Speed up
        )

        import io

        import biotite.structure.io.pdb as pdb

        pdb_file = pdb.PDBFile.read(io.StringIO(pdb_content))
        structure = pdb_file.get_structure(model=1)

        # Get Chi1 of Residue 2
        angle = get_chi1_angle(structure, 2, res_name)
        if angle is not None:
            angles.append(angle)

    return angles


def classify_rotamer(angle):
    """Classify angle into g+ (60), g- (-60/300), t (180)."""
    # Normalize to [-180, 180]
    angle = ((angle + 180) % 360) - 180

    if -120 < angle <= 0:
        return "g-"  # around -60
    elif 0 < angle <= 120:
        return "g+"  # around 60
    else:
        return "t"  # around 180


def test_val_rotamer_dependence_on_structure():
    """Verify that the rotamer distribution is different for Alpha Helix vs Beta Sheet (Valine)."""
    # 1. Sample Alpha Helix
    alpha_angles = sample_rotamer_distribution("alpha", res_name="VAL", n_samples=30)
    alpha_counts = Counter([classify_rotamer(a) for a in alpha_angles])

    # 2. Sample Beta Sheet
    beta_angles = sample_rotamer_distribution("beta", res_name="VAL", n_samples=30)
    beta_counts = Counter([classify_rotamer(a) for a in beta_angles])

    print(f"\nVAL Alpha Counts: {alpha_counts}")
    print(f"VAL Beta Counts:  {beta_counts}")

    # Fractions of Trans
    alpha_trans = alpha_counts["t"] / 30.0
    beta_trans = beta_counts["t"] / 30.0

    # Expectation: Beta loves Trans (0.40), Alpha hates it (0.05)
    # Generic: Trans is 0.20
    # Difference should be large
    assert (
        abs(alpha_trans - beta_trans) > 0.15
    ), f"VAL: Distributions similar. Alpha={alpha_trans}, Beta={beta_trans}"


def test_leu_rotamer_dependence_on_structure():
    """Verify LEU rotamer dependence. Expecting failure until implemented."""
    # 1. Sample Alpha Helix
    alpha_angles = sample_rotamer_distribution("alpha", res_name="LEU", n_samples=30)
    alpha_counts = Counter([classify_rotamer(a) for a in alpha_angles])

    # 2. Sample Beta Sheet
    beta_angles = sample_rotamer_distribution("beta", res_name="LEU", n_samples=30)
    beta_counts = Counter([classify_rotamer(a) for a in beta_angles])

    print(f"\nLEU Alpha Counts: {alpha_counts}")
    print(f"LEU Beta Counts:  {beta_counts}")

    # Generic LEU: g- (0.85 approx?), t (0.10)
    # Backbone Dependent Plan:
    # Alpha: g- (0.90+), t (very rare)
    # Beta: g- (0.60), t (0.30+)

    alpha_trans = alpha_counts["t"] / 30.0
    beta_trans = beta_counts["t"] / 30.0

    # This assertion should FAIL if both use Generic Library
    assert (
        abs(alpha_trans - beta_trans) > 0.15
    ), f"LEU: Distributions similar. Alpha={alpha_trans}, Beta={beta_trans}"


def test_phe_rotamer_dependence_on_structure():
    """Verify PHE (Aromatic) rotamer dependence. Expecting failure until implemented."""
    # 1. Sample Alpha Helix
    alpha_angles = sample_rotamer_distribution("alpha", res_name="PHE", n_samples=30)
    alpha_counts = Counter([classify_rotamer(a) for a in alpha_angles])

    # 2. Sample Beta Sheet
    beta_angles = sample_rotamer_distribution("beta", res_name="PHE", n_samples=30)
    beta_counts = Counter([classify_rotamer(a) for a in beta_angles])

    print(f"\nPHE Alpha Counts: {alpha_counts}")
    print(f"PHE Beta Counts:  {beta_counts}")

    # Generic PHE: g- (0.60), t (0.30)
    # Backbone Dependent Plan:
    # Alpha: g- (0.90), t (rare) - Aromatics clash heavily with helix
    # Beta: t is allowed/favored

    alpha_trans = alpha_counts["t"] / 30.0
    beta_trans = beta_counts["t"] / 30.0

    assert (
        abs(alpha_trans - beta_trans) > 0.15
    ), f"PHE: Distributions similar. Alpha={alpha_trans}, Beta={beta_trans}"


def test_arg_rotamer_dependence():
    """Verify ARG rotamer dependence (Long/Charged)."""
    alpha_angles = sample_rotamer_distribution("alpha", res_name="ARG", n_samples=30)
    alpha_counts = Counter([classify_rotamer(a) for a in alpha_angles])

    beta_angles = sample_rotamer_distribution("beta", res_name="ARG", n_samples=30)
    beta_counts = Counter([classify_rotamer(a) for a in beta_angles])

    # ARG in helix: heavily confined to g-
    # ARG in sheet: extended 't' rotamer is much more common
    alpha_trans = alpha_counts["t"] / 30.0
    beta_trans = beta_counts["t"] / 30.0

    print(f"ARG Alpha Trans: {alpha_trans}, Beta Trans: {beta_trans}")
    assert abs(alpha_trans - beta_trans) > 0.15, "ARG: Expected divergence between Alpha and Beta"


def test_glu_rotamer_dependence():
    """Verify GLU rotamer dependence (similar to ARG)."""
    alpha_angles = sample_rotamer_distribution("alpha", res_name="GLU", n_samples=30)
    alpha_counts = Counter([classify_rotamer(a) for a in alpha_angles])

    beta_angles = sample_rotamer_distribution("beta", res_name="GLU", n_samples=30)
    beta_counts = Counter([classify_rotamer(a) for a in beta_angles])

    alpha_trans = alpha_counts["t"] / 30.0
    beta_trans = beta_counts["t"] / 30.0

    print(f"GLU Alpha Trans: {alpha_trans}, Beta Trans: {beta_trans}")
    assert abs(alpha_trans - beta_trans) > 0.15, "GLU: Expected divergence between Alpha and Beta"


def test_his_rotamer_dependence():
    """Verify HIS rotamer dependence (Aromatic-like)."""
    alpha_angles = sample_rotamer_distribution("alpha", res_name="HIS", n_samples=30)
    alpha_counts = Counter([classify_rotamer(a) for a in alpha_angles])

    beta_angles = sample_rotamer_distribution("beta", res_name="HIS", n_samples=30)
    beta_counts = Counter([classify_rotamer(a) for a in beta_angles])

    # Aromatics avoid 't' in Helix due to clash with i-3
    alpha_trans = alpha_counts["t"] / 30.0
    beta_trans = beta_counts["t"] / 30.0

    print(f"HIS Alpha Trans: {alpha_trans}, Beta Trans: {beta_trans}")
    assert abs(alpha_trans - beta_trans) > 0.15, "HIS: Expected divergence between Alpha and Beta"
