#!/usr/bin/env python3
"""
Refactor generate_pdb_content() in generator.py.

Extracts 5 private helpers with identical logic to the original,
then rewrites generate_pdb_content as a ~50-line orchestrator.
No logic is changed — only scoping and naming.
"""
import ast
import sys
from pathlib import Path

GENERATOR = Path(__file__).parent.parent / "synth_pdb" / "generator.py"
original = GENERATOR.read_text()

# ─────────────────────────────────────────────────────────────────────────────
# NEW HELPER FUNCTIONS (inserted before generate_pdb_content)
# ─────────────────────────────────────────────────────────────────────────────

HELPERS = r'''

def _resolve_conformation_map(
    sequence: List[str],
    conformation: str,
    structure: Optional[str],
) -> Dict[int, str]:
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
    valid_conformations = list(RAMACHANDRAN_PRESETS.keys()) + ['random']
    if conformation not in valid_conformations:
        raise ValueError(
            f"Invalid conformation '{conformation}'. "
            f"Valid options are: {', '.join(valid_conformations)}"
        )
    if structure:
        residue_conformations = _parse_structure_regions(structure, sequence_length)
        for i in range(sequence_length):
            if i not in residue_conformations:
                residue_conformations[i] = conformation
    else:
        residue_conformations = {i: conformation for i in range(sequence_length)}
    return residue_conformations


def _build_peptide_chain(
    sequence: List[str],
    residue_conformations: Dict[int, str],
    conformation: str,
    structure: Optional[str],
    cyclic: bool,
    cis_proline_frequency: float,
    drift: float,
    phi_list: Optional[List[float]],
    psi_list: Optional[List[float]],
    omega_list: Optional[List[float]],
    rng: random.Random,
) -> struc.AtomArray:
    """Build the backbone + sidechain AtomArray for the full peptide.

    Places N, CA, C atoms residue-by-residue via NeRF (internal-coordinate)
    geometry using the conformation map, Ramachandran distributions, beta-turn
    definitions and rotamer libraries.  Handles D-amino acids by chirality
    mirroring and applies hard-decoy torsion drift when requested.

    Args:
        sequence: List of 3-letter amino acid codes (possibly with D- prefix).
        residue_conformations: Per-residue conformation dict (0-based index).
        conformation: Default conformation used as fallback in gap detection.
        structure: Original structure spec string (used only for ``not structure``
            short-circuit logic; pass ``None`` when not specified).
        cyclic: Whether to suppress terminal atoms for head-to-tail ring closure.
        cis_proline_frequency: Probability that a PRO residue adopts cis-omega.
        drift: Max uniform drift (degrees) added to phi/psi for hard-decoy mode.
        phi_list: Explicit phi angles (threaded backbone); overrides sampling.
        psi_list: Explicit psi angles (threaded backbone); overrides sampling.
        omega_list: Explicit omega angles; overrides sampling.
        rng: Seeded random generator for reproducible drift.

    Returns:
        Complete :class:`biotite.structure.AtomArray` with chain_id ``"A"``.
    """
    sequence_length = len(sequence)
    peptide = struc.AtomArray(0)
    residue_coordinates: Dict[int, Dict[str, np.ndarray]] = {}

    for i, full_res_name in enumerate(sequence):
        res_id = i + 1

        # EDUCATIONAL NOTE - D-Amino Acid Handling:
        # D-amino acids are the mirror images of the standard L-amino acids.
        is_d = full_res_name.startswith("D-")
        res_name = full_res_name[2:] if is_d else full_res_name

        # Determine conformation for this residue
        res_conformation = residue_conformations.get(i, conformation)
        if res_conformation == 'alpha' and not structure:
            cys_count = sum(1 for aa in sequence if "CYS" in aa.upper() or "DCY" in aa.upper())
            if cyclic or cys_count >= 2:
                res_conformation = 'curved'

        # Compatibility alias for rotamer selection logic downstream
        current_conformation = res_conformation

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
                current_psi = RAMACHANDRAN_PRESETS[res_conformation]['psi']
            elif res_conformation == 'random':
                next_res_name = sequence[i + 1] if i + 1 < sequence_length else None
                _, current_psi = _sample_ramachandran_angles(res_name, next_res_name)
            elif res_conformation in BETA_TURN_TYPES:
                current_psi = RAMACHANDRAN_PRESETS['extended']['psi']
            else:
                current_psi = -47.0  # Default Alpha

        else:
            # Subsequent residues use internal-coordinate (NeRF) placement
            prev_res_idx = i - 1
            prev_coords = residue_coordinates[prev_res_idx]
            prev_n_coord = prev_coords['N']
            prev_ca_coord = prev_coords['CA']
            prev_c_coord = prev_coords['C']

            next_full_res_name = sequence[i + 1] if i + 1 < sequence_length else None
            next_res_name = (
                next_full_res_name[2:]
                if next_full_res_name and next_full_res_name.startswith("D-")
                else next_full_res_name
            )

            prev_conformation = residue_conformations.get(prev_res_idx, conformation)

            current_phi = None
            current_psi = None

            # Handle Beta-Turns explicitly
            pass

            if prev_conformation in RAMACHANDRAN_PRESETS:
                prev_psi = RAMACHANDRAN_PRESETS[prev_conformation]['psi']
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
                    prev_psi = RAMACHANDRAN_PRESETS['extended']['psi']
                elif turn_pos == 2:
                    prev_psi = turn_angles[0][1]
                elif turn_pos == 3:
                    prev_psi = turn_angles[1][1]
                else:
                    prev_psi = RAMACHANDRAN_PRESETS['extended']['psi']
            elif prev_conformation == 'random':
                prev_full_res_name = sequence[prev_res_idx]
                prev_base_res_name = (
                    prev_full_res_name[2:]
                    if prev_full_res_name.startswith("D-")
                    else prev_full_res_name
                )
                _, prev_psi = _sample_ramachandran_angles(prev_base_res_name, res_name)
            else:
                prev_psi = RAMACHANDRAN_PRESETS['alpha']['psi']

            # Place N(i)
            n_coord = _place_atom_with_dihedral(
                prev_n_coord, prev_ca_coord, prev_c_coord,
                BOND_LENGTH_C_N, ANGLE_CA_C_N, prev_psi
            )

            # Place CA(i) — sample omega (including cis-proline)
            omega_mean = OMEGA_TRANS
            if res_name == 'PRO' and random.random() < cis_proline_frequency:
                omega_mean = 0.0

            if omega_list is not None and i > 0 and (i - 1) < len(omega_list):
                omega = omega_list[i - 1]
            else:
                omega = np.random.normal(omega_mean, OMEGA_VARIATION)

            ca_coord = _place_atom_with_dihedral(
                prev_ca_coord, prev_c_coord, n_coord,
                BOND_LENGTH_N_CA, ANGLE_C_N_CA, omega
            )

            # Determine phi/psi for this residue
            if (
                phi_list is not None and i < len(phi_list)
                and psi_list is not None and i < len(psi_list)
            ):
                current_phi = phi_list[i]
                current_psi = psi_list[i]
            elif res_conformation in RAMACHANDRAN_PRESETS:
                current_phi = RAMACHANDRAN_PRESETS[res_conformation]['phi']
                current_psi = RAMACHANDRAN_PRESETS[res_conformation]['psi']
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
                    current_phi = RAMACHANDRAN_PRESETS['extended']['phi']
                    current_psi = RAMACHANDRAN_PRESETS['extended']['psi']
                elif turn_pos == 2:
                    current_phi = turn_angles[0][0]
                    current_psi = turn_angles[0][1]
                elif turn_pos == 3:
                    current_phi = turn_angles[1][0]
                    current_psi = turn_angles[1][1]
                else:
                    current_phi = RAMACHANDRAN_PRESETS['extended']['phi']
                    current_psi = RAMACHANDRAN_PRESETS['extended']['psi']
            elif res_conformation == 'random':
                current_phi, current_psi = _sample_ramachandran_angles(res_name, next_res_name)
            else:
                current_phi = RAMACHANDRAN_PRESETS['alpha']['phi']
                current_psi = RAMACHANDRAN_PRESETS['alpha']['psi']

            # Apply hard-decoy torsion drift
            if drift > 0:
                current_phi += rng.uniform(-drift, drift)
                current_psi += rng.uniform(-drift, drift)
                current_phi = ((current_phi + 180) % 360) - 180
                current_psi = ((current_psi + 180) % 360) - 180

            # Place C(i)
            # D-amino acids invert the chirality by negating phi/psi before placement
            if is_d:
                current_phi *= -1.0
                current_psi *= -1.0

            c_coord = _place_atom_with_dihedral(
                prev_c_coord, n_coord, ca_coord,
                BOND_LENGTH_CA_C, ANGLE_N_CA_C, current_phi
            )

        # ── Store coordinates for next iteration ────────────────────────────
        residue_coordinates[i] = {
            'N': n_coord,
            'CA': ca_coord,
            'C': c_coord,
        }

        # Store Psi for next iteration (kept for readability; recomputed in loop)
        prev_psi = current_psi

        # ── Biotite reference template ───────────────────────────────────────
        # CRITICAL FIX: Always use .copy() — residue() returns a cached template.
        ref_res_template = struc.info.residue(res_name).copy()

        # Remove terminal atoms that are incompatible with internal peptide bonds
        if i < len(sequence) - 1 or cyclic:
            ref_res_template = ref_res_template[ref_res_template.atom_name != "OXT"]
            ref_res_template = ref_res_template[ref_res_template.atom_name != "HXT"]

        if i > 0 or cyclic:
            ref_res_template = ref_res_template[ref_res_template.atom_name != "H2"]
            ref_res_template = ref_res_template[ref_res_template.atom_name != "H3"]
            # PROLINE FIX: Internal/Cyclic Proline has NO amide hydrogen.
            if res_name == "PRO":
                ref_res_template = ref_res_template[ref_res_template.atom_name != "H"]

        # ── Rotamer selection ────────────────────────────────────────────────
        rotamers = None
        if res_name in BACKBONE_DEPENDENT_ROTAMER_LIBRARY:
            if current_conformation in BACKBONE_DEPENDENT_ROTAMER_LIBRARY[res_name]:
                rotamers = BACKBONE_DEPENDENT_ROTAMER_LIBRARY[res_name][current_conformation]
        if rotamers is None and res_name in ROTAMER_LIBRARY:
            rotamers = ROTAMER_LIBRARY[res_name]

        if rotamers:
            weights = [r.get('prob', 0.0) for r in rotamers]
            selected_rotamer = random.choices(rotamers, weights=weights, k=1)[0]

            if 'chi1' in selected_rotamer:
                chi1_target = selected_rotamer["chi1"][0]
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

        target_backbone_constructed = struc.array([
            struc.Atom(n_coord, atom_name="N", res_id=res_id, res_name=res_name, element="N", hetero=False),
            struc.Atom(ca_coord, atom_name="CA", res_id=res_id, res_name=res_name, element="C", hetero=False),
            struc.Atom(c_coord, atom_name="C", res_id=res_id, res_name=res_name, element="C", hetero=False),
        ])

        _, transformation = struc.superimpose(target_backbone_constructed, mobile_backbone_from_template)
        transformed_res = ref_res_template
        transformed_res.coord = transformation.apply(transformed_res.coord)

        # ── D-amino acid chiral mirroring ─────────────────────────────────
        if is_d and res_name != "GLY":
            vec1 = c_coord - ca_coord
            vec2 = n_coord - ca_coord
            normal = np.cross(vec1, vec2)
            normal /= np.linalg.norm(normal)
            backbone_names = {"N", "CA", "C", "O", "H", "HA"}
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
        transformed_res.chain_id[:] = "A"

        if i == 0:
            peptide = transformed_res.copy()
        else:
            peptide += transformed_res

        # Store for next iteration (dead code, residue_coordinates takes precedence)
        prev_n_coord = n_coord
        prev_ca_coord = ca_coord
        prev_c_coord = c_coord

    # Ensure global chain_id is 'A'
    peptide.chain_id = np.array(["A"] * peptide.array_length(), dtype="U1")
    return peptide


def _apply_biophysical_mods(
    peptide: struc.AtomArray,
    optimize_sidechains: bool,
    cap_termini: bool,
    cyclic: bool,
    ph: float,
    metal_ions: str,
) -> struc.AtomArray:
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
        Modified AtomArray (may be the same object or a new one with ions).
    """
    # EDUCATIONAL NOTE - Side-Chain Optimization:
    # If requested, run Monte Carlo optimization to fix steric clashes.
    if optimize_sidechains:
        logger.info("Running side-chain optimization...")
        peptide = run_optimization(peptide)

    # 1. Terminal Capping (ACE/NME) — cyclic peptides are naturally capped.
    if cap_termini and not cyclic:
        peptide = biophysics.cap_termini(peptide)

    # 2. pH Titration (Protonation States)
    peptide = biophysics.apply_ph_titration(peptide, ph=ph)

    # EDUCATIONAL NOTE - Metal Ion Coordination:
    # Inorganic cofactors like Zn2+ are automatically detected and injected.
    if metal_ions == 'auto':
        from .cofactors import find_metal_binding_sites, add_metal_ion
        sites = find_metal_binding_sites(peptide)
        for site in sites:
            peptide = add_metal_ion(peptide, site)

    return peptide


def _do_energy_minimization(
    peptide: struc.AtomArray,
    sequence: List[str],
    forcefield: str,
    minimization_k: float,
    minimization_max_iter: int,
    cyclic: bool,
    equilibrate: bool,
    equilibrate_steps: int,
) -> Tuple[Optional[str], struc.AtomArray]:
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

    Returns:
        ``(atomic_and_ter_content, updated_peptide)`` where *atomic_and_ter_content*
        is the raw PDB string from OpenMM (or ``None`` on failure) and
        *updated_peptide* is the (possibly updated) AtomArray.
    """
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

            minimizer = EnergyMinimizer(forcefield_name=forcefield)
            current_disulfides = _detect_disulfide_bonds(peptide)

            if equilibrate:
                logger.info(
                    f"Running MD Equilibration ({equilibrate_steps} steps). "
                    "This includes minimization."
                )
                success = minimizer.equilibrate(
                    input_pdb_path, output_pdb_path,
                    steps=equilibrate_steps, cyclic=cyclic,
                    disulfides=current_disulfides,
                )
            else:
                success = minimizer.add_hydrogens_and_minimize(
                    input_pdb_path, output_pdb_path,
                    max_iterations=minimization_max_iter,
                    tolerance=minimization_k,
                    cyclic=cyclic,
                    disulfides=current_disulfides,
                )

            if not success:
                logger.error("Minimization failed. Returning un-minimized structure.")
                return None, peptide

            logger.info("Minimization/Equilibration successful.")
            with open(output_pdb_path, 'r') as f:
                atomic_and_ter_content = f.read()

            pdb_file_read = pdb.PDBFile.read(output_pdb_path)
            peptide = pdb_file_read.get_structure(model=1)

            # RESTORE PTM NAMES (Fix for "Missing Orange Balls"):
            # OpenMM reverted SEP→SER etc.; restore for downstream viewers.
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
                            logger.info(
                                "Detected N-terminal ACE cap. Applying start offset of 1."
                            )
                            start_offset = 1
                if n_min >= n_seq + start_offset:
                    for idx, res_name_target in enumerate(sequence):
                        rid = unique_res_ids[idx + start_offset]
                        if res_name_target in ['SEP', 'TPO', 'PTR', 'HIE', 'HID', 'HIP']:
                            mask = peptide.res_id == rid
                            peptide.res_name[mask] = res_name_target
            except Exception as ptm_err:
                logger.warning(f"Failed to restore PTM names: {ptm_err}")

            return atomic_and_ter_content, peptide

    except Exception as e:
        logger.error(f"Error during minimization workflow: {e}")
        return None, peptide


def _assemble_pdb_output(
    peptide: struc.AtomArray,
    atomic_and_ter_content: Optional[str],
    sequence_length: int,
    cyclic: bool,
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
    if atomic_and_ter_content is None:
        peptide.atom_id = np.arange(1, peptide.array_length() + 1)
        pdb_file_out = pdb.PDBFile()
        pdb_file_out.set_structure(peptide)
        string_io = io.StringIO()
        pdb_file_out.write(string_io)
        atomic_and_ter_content = string_io.getvalue()

    # EDUCATIONAL NOTE - Adding Realistic B-factors & Occupancy:
    # Biotite sets B-factors to 0.00 and occupancy to 1.00 by default.
    # We post-process to insert physically meaningful values derived from S2.
    total_residues = len(set(peptide.res_id))
    s2_map = predict_order_parameters(peptide)

    # Sanitize: Extract only atomic content to avoid header duplication
    atomic_and_ter_content = extract_atomic_content(atomic_and_ter_content)

    processed_lines = []
    n_term_serial = None
    c_term_serial = None
    sg_serials: Dict[int, int] = {}
    serial = 0  # Initialize to avoid UnboundLocalError

    for line in atomic_and_ter_content.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            serial = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_num = int(line[22:26].strip())

            if cyclic:
                if res_num == 1 and atom_name == "N":
                    n_term_serial = serial
                if res_num == total_residues and atom_name == "C":
                    c_term_serial = serial

            if (res_name == "CYS" or res_name == "CYX") and atom_name == "SG":
                sg_serials[res_num] = serial

            current_s2 = s2_map.get(res_num, 0.85)
            bfactor = _calculate_bfactor(atom_name, res_num, total_residues, res_name, s2=current_s2)
            occupancy = _calculate_occupancy(atom_name, res_num, total_residues, res_name, bfactor)
            line = line[:54] + f"{occupancy:6.2f}" + f"{bfactor:6.2f}" + line[66:]

        processed_lines.append(line)

    atomic_and_ter_content = "\n".join(processed_lines) + "\n"

    # Ensure TER record exists at the end
    lines = atomic_and_ter_content.strip().splitlines()
    if not lines:
        logger.error("Generated PDB content is empty! Falling back to raw sequence string.")
        return atomic_and_ter_content

    last_line = lines[-1]
    if last_line.startswith("ATOM") or last_line.startswith("HETATM"):
        last_atom = peptide[-1]
        ter_atom_num = serial + 1
        ter_record = (
            f"TER   {ter_atom_num: >5}      {last_atom.res_name: >3} "
            f"{last_atom.chain_id: <1}{last_atom.res_id: >4}"
        ).ljust(80)
        atomic_and_ter_content = atomic_and_ter_content.strip() + "\n" + ter_record + "\n"

    # Pad to 80 chars
    padded_lines = [line.ljust(80) for line in atomic_and_ter_content.splitlines()]
    final_atomic_content_block = "\n".join(padded_lines).strip()

    # Generate CONECT records for visualization
    conect_records = []
    if cyclic and n_term_serial and c_term_serial:
        conect_records.append(f"CONECT{n_term_serial:5d}{c_term_serial:5d}".ljust(80))

    disulfides = _detect_disulfide_bonds(peptide)
    ssbond_records = _generate_ssbond_records(disulfides, chain_id='A')

    if disulfides:
        for r1, r2 in disulfides:
            s1 = sg_serials.get(r1)
            s2 = sg_serials.get(r2)
            if s1 and s2:
                conect_records.append(f"CONECT{s1:5d}{s2:5d}".ljust(80))
                conect_records.append(f"CONECT{s2:5d}{s1:5d}".ljust(80))

    conect_block = "\n".join(conect_records)
    if conect_block:
        conect_block += "\n"

    return assemble_pdb_content(
        final_atomic_content_block,
        sequence_length,
        command_args=None,
        extra_records=ssbond_records if ssbond_records else None,
        conect_records=conect_block if conect_block else None,
    )

'''  # end HELPERS

# ─────────────────────────────────────────────────────────────────────────────
# NEW generate_pdb_content ORCHESTRATOR (replaces the 900-line body)
# ─────────────────────────────────────────────────────────────────────────────

NEW_ORCHESTRATOR = r"""
    if seed is not None:
        logger.info(f"Setting random seed to {seed} for reproducibility.")
        random.seed(seed)
        np.random.seed(seed)

    # Use localized random generator for better control and thread-safety
    rng = random.Random(seed)

    sequence = _resolve_sequence(
        length=length,
        user_sequence_str=sequence_str,
        use_plausible_frequencies=use_plausible_frequencies,
    )

    # EDUCATIONAL NOTE - Post-Translational Modifications (PTMs):
    # Phosphorylation converts SER/THR/TYR to SEP/TPO/PTR before geometry build.
    if phosphorylation_rate > 0:
        modified_sequence = []
        for aa in sequence:
            if aa == 'SER' and random.random() < phosphorylation_rate:
                modified_sequence.append('SEP')
            elif aa == 'THR' and random.random() < phosphorylation_rate:
                modified_sequence.append('TPO')
            elif aa == 'TYR' and random.random() < phosphorylation_rate:
                modified_sequence.append('PTR')
            else:
                modified_sequence.append(aa)
        sequence = modified_sequence

    if not sequence:
        if sequence_str is not None and len(sequence_str) == 0:
            raise ValueError("Provided sequence string cannot be empty.")
        raise ValueError(
            "Length must be a positive integer when no sequence is provided "
            "and no valid sequence string is given."
        )

    sequence_length = len(sequence)

    # Build the per-residue {index: conformation} map (validates inputs)
    residue_conformations = _resolve_conformation_map(sequence, conformation, structure)

    # Build backbone + sidechains via NeRF geometry
    peptide = _build_peptide_chain(
        sequence, residue_conformations, conformation, structure, cyclic,
        cis_proline_frequency, drift, phi_list, psi_list, omega_list, rng,
    )

    # Sidechain optimization, terminal capping, pH titration, metal ions
    peptide = _apply_biophysical_mods(
        peptide, optimize_sidechains, cap_termini, cyclic, ph, metal_ions,
    )

    # Optional energy minimization / MD equilibration via OpenMM
    atomic_and_ter_content: Optional[str] = None
    if minimize_energy:
        atomic_and_ter_content, peptide = _do_energy_minimization(
            peptide, sequence, forcefield,
            minimization_k, minimization_max_iter,
            cyclic, equilibrate, equilibrate_steps,
        )

    # Assemble final PDB with B-factors, occupancy, TER, CONECT records
    return _assemble_pdb_output(peptide, atomic_and_ter_content, sequence_length, cyclic)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Perform the patch
# ─────────────────────────────────────────────────────────────────────────────

START_MARKER = "def generate_pdb_content("
END_MARKER = "\nclass PeptideGenerator:"

start_idx = original.find(START_MARKER)
end_idx = original.find(END_MARKER)

assert start_idx != -1, f"ERROR: Could not find '{START_MARKER}' in {GENERATOR}"
assert end_idx != -1, f"ERROR: Could not find '{END_MARKER}' in {GENERATOR}"

# Everything before generate_pdb_content (the 5 helpers go right before it)
prefix = original[:start_idx]

# Extract just the function signature + docstring, stopping right before the body code.
# The docstring ends and the body begins after the triple-quote-close.
func_def_and_docstring = original[start_idx:end_idx]

# We need to keep the function signature lines (def + args) and the docstring.
# Find where the docstring ends (closing """) and the real body starts.
# The body starts after the first occurrence of the closing triple-quote of the docstring.
docstring_start = func_def_and_docstring.find('"""')
docstring_end = func_def_and_docstring.find('"""', docstring_start + 3) + 3
signature_and_docstring = func_def_and_docstring[:docstring_end]

# The suffix is everything from class PeptideGenerator onwards
suffix = original[end_idx:]

new_content = (
    prefix
    + HELPERS.lstrip("\n")  # 5 helper functions
    + "\n"
    + signature_and_docstring  # original def + docstring preserved verbatim
    + "\n"
    + NEW_ORCHESTRATOR  # new 50-line body
    + suffix
)

# Syntax-check before writing
try:
    ast.parse(new_content)
except SyntaxError as e:
    print(f"SYNTAX ERROR in generated code: {e}")
    sys.exit(1)

GENERATOR.write_text(new_content)
print(f"SUCCESS: {GENERATOR} patched.")
print("  Helpers inserted before generate_pdb_content.")
print(f"  Old function body replaced with {len(NEW_ORCHESTRATOR.splitlines())} line orchestrator.")
