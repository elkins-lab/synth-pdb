#!/usr/bin/env python3
"""
Refactor EnergyMinimizer._run_simulation() in physics.py.

Extracts 4 private methods with identical logic, reducing _run_simulation
to a ~70-line orchestrator.  No logic is changed — only scoping.
"""
import ast
import sys
from pathlib import Path

PHYSICS = Path(__file__).parent.parent / "synth_pdb" / "physics.py"
original = PHYSICS.read_text()

# ─────────────────────────────────────────────────────────────────────────────
# NEW PRIVATE METHODS (inserted before _run_simulation)
# ─────────────────────────────────────────────────────────────────────────────

NEW_METHODS = r'''
    def _preprocess_pdb_for_simulation(self, input_path, cyclic, disulfides_param):
        """Load and sanitize the input PDB for OpenMM; return OpenMM topology/positions.

        Performs PTM residue renaming (SEP→SER, etc.), HETATM ion stripping,
        optional OXT dummy insertion for cyclic peptides, and cyclic CONECT
        removal.  Loads the modified PDB into OpenMM and applies standard bond
        generation and cyclic bond surgery.

        Args:
            input_path: Path to the input PDB file.
            cyclic: Whether to apply cyclic-peptide preprocessing.
            disulfides_param: Initial disulfide list from caller (used to seed
                ``added_bonds``; note the SSBOND detection later resets this).

        Returns:
            Tuple ``(topology, positions, hetatm_lines, original_metadata)``
            where *topology* and *positions* come from OpenMM's PDBFile loader.

        Raises:
            Exception: Any failure in file I/O or OpenMM loading is re-raised
                so that the caller's try/except can log and return ``None``.
        """
        import tempfile, os, numpy as np

        ptm_map = {
            'SEP': 'SER', 'TPO': 'THR', 'PTR': 'TYR',
            'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS',
            'DAL': 'ALA', 'DAR': 'ARG', 'DAN': 'ASN', 'DAS': 'ASP', 'DCY': 'CYS',
            'DGL': 'GLU', 'DGN': 'GLN', 'DHI': 'HIS', 'DIL': 'ILE', 'DLE': 'LEU',
            'DLY': 'LYS', 'DME': 'MET', 'DPH': 'PHE', 'DPR': 'PRO', 'DSE': 'SER',
            'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR', 'DVA': 'VAL',
        }
        ptm_atom_names = ["P", "O1P", "O2P", "O3P"]

        original_metadata: dict = {}
        modified_lines: list = []
        hetatm_lines: list = []
        last_res_key = None
        first_res_id = None
        last_res_id = None

        if os.path.exists(input_path):
            with open(input_path, 'r') as f:
                pdb_lines = f.readlines()

            atom_lines = [l for l in pdb_lines if l.startswith("ATOM")]
            first_res_id = atom_lines[0][22:26].strip() if atom_lines else None
            last_res_id  = atom_lines[-1][22:26].strip() if atom_lines else None

            n_term_serial, c_term_serial = None, None
            c_coords, c_line_template = None, None
            if cyclic and atom_lines:
                for line in atom_lines:
                    res_id   = line[22:26].strip()
                    atom_name = line[12:16].strip()
                    if res_id == first_res_id and atom_name == "N":
                        n_term_serial = line[6:11].strip()
                    if res_id == last_res_id and atom_name == "C":
                        c_term_serial = line[6:11].strip()
                        c_coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                        c_line_template = line

            for line in pdb_lines:
                if line.startswith("CONECT") and cyclic and n_term_serial and c_term_serial:
                    parts = line.split()
                    if len(parts) >= 3:
                        if (
                            (parts[1] == n_term_serial and parts[2] == c_term_serial)
                            or (parts[1] == c_term_serial and parts[2] == n_term_serial)
                        ):
                            print(f"DEBUG: Skipping cyclic CONECT: {line.strip()}")
                            continue

                if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
                    res_name  = line[17:20].strip()
                    res_id    = line[22:26].strip()
                    chain_id  = line[21] if len(line) > 21 else " "
                    res_key   = (res_id, chain_id)
                    atom_name = line[12:16].strip()

                    if res_key != last_res_key:
                        last_res_key = res_key
                        original_metadata[res_key] = {"name": res_name, "id": res_id}

                    res_name_upper = line[17:20].strip().upper()
                    if res_name_upper in ['ZN', 'FE', 'MG', 'CA', 'NA', 'CL']:
                        hetatm_lines.append(line)
                        logger.info(f"Restoring lost HETATM: {res_name_upper}")
                        continue

                    if res_name in ptm_map:
                        new_name = ptm_map[res_name]
                        line = line[:17] + f"{new_name: >3}" + line[20:]
                        if res_name in ['SEP', 'TPO', 'PTR'] and len(line) >= 16:
                            if atom_name in ptm_atom_names:
                                continue
                modified_lines.append(line)

            # Add dummy OXT for cyclic peptides to satisfy C-terminal templates
            if cyclic and last_res_id and c_line_template:
                insert_idx = -1
                for idx, line in enumerate(modified_lines):
                    if line.startswith("ATOM") and line[22:26].strip() == last_res_id:
                        insert_idx = idx + 1
                if insert_idx != -1:
                    x, y, z = c_coords
                    res_name_c = c_line_template[17:20]
                    res_id_full = c_line_template[21:26]
                    oxt_line = (
                        f"ATOM   9999  OXT {res_name_c} {res_id_full}    "
                        f"{x+1.2:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\n"
                    )
                    modified_lines.insert(insert_idx, oxt_line)
                    logger.info(
                        f"Added temporary OXT to residue {last_res_id} "
                        f"(Renamed: {res_name_c.strip()})"
                    )

            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tf:
                tf.writelines(modified_lines)
                temp_input_path = tf.name

            pdb_obj = app.PDBFile(temp_input_path)
            topology, positions = pdb_obj.topology, pdb_obj.positions
            try:
                os.unlink(temp_input_path)
            except Exception:
                pass
        else:
            pdb_obj = app.PDBFile(input_path)
            topology, positions = pdb_obj.topology, pdb_obj.positions

        topology.createStandardBonds()
        topology.createDisulfideBonds(positions)

        # Surgically remove head-to-tail bond so addHydrogens doesn't fail
        if cyclic:
            bonds_to_remove = []
            res_list = list(topology.residues())
            if len(res_list) >= 2:
                first_res, last_res = res_list[0], res_list[-1]
                for bond in topology.bonds():
                    if (
                        (bond[0].residue == first_res and bond[1].residue == last_res)
                        or (bond[0].residue == last_res and bond[1].residue == first_res)
                    ):
                        if (
                            (bond[0].name == "N" and bond[1].name == "C")
                            or (bond[0].name == "C" and bond[1].name == "N")
                        ):
                            bonds_to_remove.append(bond)
            if bonds_to_remove:
                new_bonds = [b for b in topology._bonds if b not in bonds_to_remove]
                topology._bonds = new_bonds
                logger.info(f"Surgically removed {len(bonds_to_remove)} cyclic head-to-tail bonds.")
            else:
                logger.debug("No cyclic bonds found in topology to remove.")

        return topology, positions, hetatm_lines, original_metadata

    def _setup_openmm_modeller(
        self, topology, positions, add_hydrogens, cyclic, coordination_param, atom_list
    ):
        """Build the OpenMM Modeller, apply H handling, detect disulfides and salt bridges.

        Steps:
        1. Optionally announce cyclic restraint intent.
        2. Heuristic backbone stitching (missing C–N peptide bonds).
        3. Strip existing hydrogens if ``add_hydrogens`` is True.
        4. Detect candidate disulfide bonds by S–S proximity.
        5. Detect salt bridges via biotite structure analysis.
        6. Add hydrogens via ``Modeller.addHydrogens``.
        7. Weld cyclic topology (post-H) and clean up terminal atoms.
        8. Rename bonded CYS → CYX, delete SG hydrogens.

        Args:
            topology: OpenMM :class:`Topology` from preprocessing.
            positions: OpenMM positions.
            add_hydrogens: Whether to add hydrogens with Modeller.
            cyclic: Whether this is a cyclic peptide.
            coordination_param: Caller-supplied coordination restraint list
                (may be mutated with metal-site restraints).
            atom_list: Current atom list (passed through / re-derived here).

        Returns:
            ``(modeller, added_bonds, salt_bridge_restraints,
              coordination_restraints, atom_list)``
        """
        import numpy as np, io as _io
        import biotite.structure.io.pdb as biotite_pdb

        coordination_restraints = coordination_param if coordination_param is not None else []
        salt_bridge_restraints: list = []

        if cyclic:
            logger.info("Cyclizing peptide via harmonic restraints (Restraint-First approach).")

        modeller = app.Modeller(topology, positions)

        # Heuristic backbone stitching
        try:
            residues = list(modeller.topology.residues())
            existing_bonds = set(
                frozenset([b[0].index, b[1].index]) for b in modeller.topology.bonds()
            )
            for i in range(len(residues) - 1):
                res1, res2 = residues[i], residues[i + 1]
                c_s = next((a for a in res1.atoms() if a.name == 'C'), None)
                n_s = next((a for a in res2.atoms() if a.name == 'N'), None)
                if c_s and n_s and frozenset([c_s.index, n_s.index]) not in existing_bonds:
                    logger.debug(
                        f"Stitching missing backbone bond: "
                        f"{res1.name}{res1.id} -> {res2.name}{res2.id}"
                    )
                    modeller.topology.addBond(c_s, n_s)
        except Exception as e:
            logger.warning(f"Robust stitching failed: {e}")

        # Hydrogen stripping
        added_bonds: list = []
        if add_hydrogens:
            try:
                modeller.delete([
                    a for a in modeller.topology.atoms()
                    if a.element is not None and a.element.symbol == "H"
                ])
            except Exception as e:
                logger.debug(f"H deletion failed: {e}")

        # Disulfide bond detection by SG proximity
        try:
            cys_residues = [
                r for r in modeller.topology.residues()
                if r.name in ('CYS', 'CYX')
            ]
            res_to_sg = {
                r.index: [a for a in r.atoms() if a.name == 'SG'][0]
                for r in cys_residues
                if any(a.name == 'SG' for a in r.atoms())
            }
            potential_bonds = []
            for i in range(len(cys_residues)):
                r1 = cys_residues[i]
                s1 = res_to_sg.get(r1.index)
                if not s1:
                    continue
                for j in range(i + 1, len(cys_residues)):
                    r2 = cys_residues[j]
                    s2 = res_to_sg.get(r2.index)
                    if not s2:
                        continue
                    p1 = np.array(modeller.positions[s1.index].value_in_unit(unit.angstroms))
                    p2 = np.array(modeller.positions[s2.index].value_in_unit(unit.angstroms))
                    d_a = np.sqrt(np.sum((p1 - p2) ** 2))
                    if d_a < SSBOND_CAPTURE_RADIUS:
                        potential_bonds.append((d_a, r1, r2, s1, s2))
            potential_bonds.sort(key=lambda x: x[0])
            bonded_indices: set = set()
            for d, r1, r2, s1, s2 in potential_bonds:
                if r1.index in bonded_indices or r2.index in bonded_indices:
                    continue
                modeller.topology.addBond(s1, s2)
                added_bonds.append((str(r1.id).strip(), str(r2.id).strip()))
                bonded_indices.add(r1.index)
                bonded_indices.add(r2.index)
        except Exception as e:
            logger.warning(f"SSBOND failed: {e}")

        # Salt bridge + metal coordination detection via biotite
        try:
            from .cofactors import find_metal_binding_sites
            from .biophysics import find_salt_bridges
            tmp_io = _io.StringIO()
            app.PDBFile.writeFile(modeller.topology, modeller.positions, tmp_io)
            tmp_io.seek(0)
            try:
                b_struc_raw = biotite_pdb.PDBFile.read(tmp_io)
                try:
                    b_struc = b_struc_raw.get_structure(model=1)
                except Exception:
                    b_struc = (
                        b_struc_raw.get_structure()[0]
                        if hasattr(b_struc_raw.get_structure(), "__getitem__")
                        else b_struc_raw.get_structure()
                    )
            except Exception:
                b_struc = None

            if b_struc is not None:
                sites = find_metal_binding_sites(b_struc)
            logger.debug(f"DEBUG: Found {len(sites)} metal binding sites.")
            for site in sites:
                i_idx = -1
                for atom in atom_list:
                    if atom.residue.name == site["type"]:
                        i_idx = atom.index
                        break
                if i_idx != -1:
                    for l_idx in site["ligand_indices"]:
                        l_at = b_struc[l_idx]
                        for atom in atom_list:
                            if (
                                int(atom.residue.id) == int(l_at.res_id)
                                and atom.name == l_at.atom_name
                            ):
                                coordination_restraints.append((i_idx, atom.index))
                                break

            # Salt bridges
            try:
                salt_bridges = find_salt_bridges(b_struc, cutoff=5.0)
                logger.info(
                    f"DEBUG: Found {len(salt_bridges) if salt_bridges else 0} salt bridges"
                )
                if salt_bridges:
                    current_atoms = list(modeller.topology.atoms())
                    for br in salt_bridges:
                        ia, ib = -1, -1
                        for atom in current_atoms:
                            if (
                                str(atom.residue.id).strip() == str(br["res_ia"]).strip()
                                and atom.name == br["atom_a"]
                            ):
                                ia = atom.index
                            if (
                                str(atom.residue.id).strip() == str(br["res_ib"]).strip()
                                and atom.name == br["atom_b"]
                            ):
                                ib = atom.index
                        if ia != -1 and ib != -1:
                            salt_bridge_restraints.append((ia, ib, br["distance"] / 10.0))
            except Exception as e:
                logger.debug(f"Internal salt bridge detection failed: {e}")
        except Exception as e:
            logger.warning(f"Metadata/SaltBridge detection failed: {e}")

        # Add hydrogens
        if add_hydrogens:
            modeller.addHydrogens(self.forcefield, pH=7.0)

        # Post-hydrogen cyclic weld
        if cyclic:
            try:
                res = list(modeller.topology.residues())
                if len(res) >= 2:
                    res1, resN = res[0], res[-1]
                    c_at = next((a for a in resN.atoms() if a.name == 'C'), None)
                    n_at = next((a for a in res1.atoms() if a.name == 'N'), None)
                    if c_at and n_at:
                        modeller.topology.addBond(c_at, n_at)
                        logger.info(
                            f"Welded cyclic link in Topology: "
                            f"{resN.name}{resN.id} -> {res1.name}{res1.id}"
                        )
                        to_delete = []
                        for a in resN.atoms():
                            if a.name in ["OXT", "OT1", "OT2", "HXT"]:
                                to_delete.append(a)
                        n_hyds = [
                            a for a in res1.atoms()
                            if a.name in ["H1", "H2", "H3", "H"]
                        ]
                        if len(n_hyds) > 1:
                            sorted_hyds = sorted(n_hyds, key=lambda x: x.name)
                            to_delete.extend(sorted_hyds[1:])
                        if to_delete:
                            modeller.delete(to_delete)
                            logger.info(f"Purged {len(to_delete)} terminal atoms for cyclic closure.")
            except Exception as e:
                logger.debug(f"Cyclic welding failed: {e}")

        # CYX renaming + HG deletion for bonded cysteines
        if added_bonds:
            hg_to_delete = []
            res_map = {str(r.id).strip(): r for r in modeller.topology.residues()}
            for id1, id2 in added_bonds:
                for rid in [id1, id2]:
                    res = res_map.get(rid)
                    if res and res.name == 'CYS':
                        res.name = 'CYX'
                        hg_to_delete.extend([a for a in res.atoms() if a.name == 'HG'])
            if hg_to_delete:
                modeller.delete(hg_to_delete)

        # Refresh atom_list after all modeller modifications
        atom_list = list(modeller.topology.atoms())

        return modeller, added_bonds, salt_bridge_restraints, coordination_restraints, atom_list

    def _build_simulation_context(
        self, modeller, cyclic, added_bonds, salt_bridge_restraints,
        coordination_restraints, atom_list
    ):
        """Create OpenMM System + Simulation, apply forces, return context objects.

        Wraps system creation (implicit/explicit solvent), cyclic terminal
        ghosting, shadow-cap zeroing, pull forces for ring closure and
        disulfide formation, and coordination restraints.

        Args:
            modeller: Fully prepared :class:`app.Modeller`.
            cyclic: Whether to apply terminal ghosting and pull forces.
            added_bonds: List of ``(id1, id2)`` disulfide residue-ID pairs.
            salt_bridge_restraints: ``[(ia, ib, r0_nm), ...]`` from detection.
            coordination_restraints: ``[(i_idx, l_idx), ...]`` metal restraints.
            atom_list: Atom list at system-creation time (for index mapping).

        Returns:
            ``(simulation, system, n_idx, c_idx, topology, positions)``
            where *n_idx*/*c_idx* are the N/C-terminus atom indices used for
            the cyclic pull force (``-1`` for non-cyclic structures).
        """
        import numpy as np
        topology, positions = modeller.topology, modeller.positions

        # System creation
        try:
            current_constraints = None if cyclic else app.HBonds
            if self.solvent_model == 'explicit':
                logger.info(
                    f"Adding explicit solvent (TIP3P water) with a "
                    f"{self.box_size} nm padding..."
                )
                modeller.addSolvent(
                    self.forcefield, model='tip3p',
                    padding=self.box_size, ionicStrength=0.1 * unit.molar,
                )
                topology = modeller.topology
                positions = modeller.positions
                if os.getenv("SYNTH_PDB_DEBUG_SAVE_INTERMEDIATE") == "1":
                    with open("intermediate_debug.pdb", 'w') as f:
                        app.PDBFile.writeFile(topology, positions, f)
                system = self.forcefield.createSystem(
                    topology,
                    nonbondedMethod=app.PME,
                    nonbondedCutoff=1.0 * unit.nanometers,
                    constraints=app.HBonds,
                )
            else:
                system, topology, positions = self._create_system_robust(
                    topology, current_constraints, modeller=modeller
                )
        except Exception as e:
            logger.error(f"Initial system creation failed despite robustness. Error: {e}")
            system = self.forcefield.createSystem(
                topology, nonbondedMethod=app.NoCutoff, constraints=None
            )

        # Cyclic terminal ghosting + shadow cap zeroing
        n_idx, c_idx = -1, -1
        if cyclic:
            try:
                nb_force = next(
                    f for f in system.getForces() if isinstance(f, mm.NonbondedForce)
                )
                residues = list(topology.residues())
                if len(residues) >= 2:
                    res1 = residues[0]
                    resN = residues[-1]
                    ats_first = list(res1.atoms())
                    ats_last  = list(resN.atoms())
                    logger.info(
                        f"Ghosting entire residues {res1.name}{res1.id} and "
                        f"{resN.name}{resN.id} for closure."
                    )
                    for a1 in ats_first:
                        for a2 in ats_last:
                            nb_force.addException(a1.index, a2.index, 0.0, 0.1, 0.0, replace=True)

                logger.info("De-physicizing capping residues (Shadow Caps) to allow closure.")
                top_atoms = list(topology.atoms())
                for force in system.getForces():
                    if isinstance(force, mm.HarmonicBondForce):
                        for i in range(force.getNumBonds()):
                            p1, p2, r0, k = force.getBondParameters(i)
                            if (
                                top_atoms[p1].residue.name in ['ACE', 'NME']
                                or top_atoms[p2].residue.name in ['ACE', 'NME']
                            ):
                                force.setBondParameters(i, p1, p2, r0, 0.0)
                    elif isinstance(force, mm.HarmonicAngleForce):
                        for i in range(force.getNumAngles()):
                            p1, p2, p3, theta, k = force.getAngleParameters(i)
                            if any(top_atoms[p].residue.name in ['ACE', 'NME'] for p in [p1, p2, p3]):
                                force.setAngleParameters(i, p1, p2, p3, theta, 0.0)
                    elif isinstance(force, mm.PeriodicTorsionForce):
                        for i in range(force.getNumTorsions()):
                            p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                            if any(top_atoms[p].residue.name in ['ACE', 'NME'] for p in [p1, p2, p3, p4]):
                                force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0.0)

                logger.info("Excised non-bonded interactions between termini for cyclic closure.")
            except Exception as e:
                logger.warning(f"Failed to excise terminal interactions: {e}")

        # Coordination restraints
        if coordination_restraints:
            f = mm.CustomBondForce("0.5*k*(r-r0)^2")
            f.addGlobalParameter(
                "k", 50000.0 * unit.kilojoules_per_mole / unit.nanometer ** 2
            )
            f.addPerBondParameter("r0")
            new_ats = list(topology.atoms())
            for i_o, l_o in coordination_restraints:
                oi, ol = atom_list[i_o], atom_list[l_o]
                ni, nl = -1, -1
                for a in new_ats:
                    if a.residue.id == oi.residue.id and a.name == oi.name:
                        ni = a.index
                    if a.residue.id == ol.residue.id and a.name == ol.name:
                        nl = a.index
                if ni != -1 and nl != -1:
                    f.addBond(
                        ni, nl,
                        [(0.23 if new_ats[nl].name == "SG" else 0.21) * unit.nanometers],
                    )
            system.addForce(f)

        # Pull forces for cyclic closure and disulfide formation
        if cyclic or added_bonds:
            pull_force = mm.CustomBondForce("0.5*k_pull*(r-r0)^2")
            pull_force.addGlobalParameter(
                "k_pull",
                100000000.0 * unit.kilojoules_per_mole / unit.nanometer ** 2,
            )
            pull_force.addPerBondParameter("r0")

            if cyclic:
                solvent_names = [
                    'HOH', 'WAT', 'SOL', 'TIP3', 'POP', 'NA', 'CL',
                    'ZN', 'FE', 'MG', 'CA',
                ]
                real_residues = [
                    r for r in list(topology.residues())
                    if r.name.strip().upper() not in (['ACE', 'NME'] + solvent_names)
                ]
                if real_residues:
                    r_first, r_last = real_residues[0], real_residues[-1]
                    logger.info(
                        f"CYCLIC: Termini identified as "
                        f"{r_first.name}{r_first.id} and {r_last.name}{r_last.id}"
                    )
                    for a in r_first.atoms():
                        if a.name == 'N':
                            n_idx = a.index
                            break
                    for a in r_last.atoms():
                        if a.name == 'C':
                            c_idx = a.index
                            break
                    logger.info(f"CYCLIC: Indices: N={n_idx}, C={c_idx}")

                if n_idx != -1 and c_idx != -1:
                    pull_force.addBond(n_idx, c_idx, [0.133 * unit.nanometers])
                    logger.info(f"Added massive cyclic pull force: {n_idx} -- {c_idx}")

                    # Ghost the welded topological bond
                    for force in system.getForces():
                        if isinstance(force, mm.HarmonicBondForce):
                            for i in range(force.getNumBonds()):
                                p1, p2, r0, k = force.getBondParameters(i)
                                if (p1 == n_idx and p2 == c_idx) or (p1 == c_idx and p2 == n_idx):
                                    force.setBondParameters(i, p1, p2, r0, 0.0)
                        elif isinstance(force, mm.HarmonicAngleForce):
                            for i in range(force.getNumAngles()):
                                p1, p2, p3, theta, k = force.getAngleParameters(i)
                                if (
                                    any(p == n_idx for p in [p1, p2, p3])
                                    and any(p == c_idx for p in [p1, p2, p3])
                                ):
                                    force.setAngleParameters(i, p1, p2, p3, theta, 0.0)
                        elif isinstance(force, mm.PeriodicTorsionForce):
                            for i in range(force.getNumTorsions()):
                                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                                if (
                                    any(p == n_idx for p in [p1, p2, p3, p4])
                                    and any(p == c_idx for p in [p1, p2, p3, p4])
                                ):
                                    force.setTorsionParameters(
                                        i, p1, p2, p3, p4, periodicity, phase, 0.0
                                    )

            # Disulfide pull
            if added_bonds:
                for id1, id2 in added_bonds:
                    s1, s2 = -1, -1
                    for res in topology.residues():
                        if str(res.id).strip() == id1:
                            for a in res.atoms():
                                if a.name == 'SG':
                                    s1 = a.index
                                    break
                        if str(res.id).strip() == id2:
                            for a in res.atoms():
                                if a.name == 'SG':
                                    s2 = a.index
                                    break
                    if s1 != -1 and s2 != -1:
                        pull_force.addBond(s1, s2, [0.203 * unit.nanometers])

            try:
                num_bonds = pull_force.getNumBonds()
                has_bonds = (num_bonds > 0) if isinstance(num_bonds, int) else False
            except Exception:
                has_bonds = False
            if has_bonds:
                system.addForce(pull_force)

        # Build Simulation
        integrator = mm.LangevinIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds
        )
        platform = None
        props: dict = {}
        for name in ['CUDA', 'Metal', 'OpenCL']:
            try:
                platform = mm.Platform.getPlatformByName(name)
                if name in ['CUDA', 'OpenCL']:
                    props = {'Precision': 'mixed'}
                logger.info(f"Using OpenMM Platform: {name}")
                break
            except Exception:
                continue

        if platform:
            try:
                simulation = app.Simulation(topology, system, integrator, platform, props)
            except Exception:
                platform = None
        if not platform:
            simulation = app.Simulation(topology, system, integrator)

        simulation.context.setPositions(positions)

        return simulation, system, n_idx, c_idx, topology, positions

    def _finalize_output(
        self, output_path, simulation, cyclic, added_bonds,
        coordination_restraints, hetatm_lines, original_metadata, atom_list
    ):
        """Write the post-simulation structure to *output_path*.

        Handles macrocycle terminal-atom cleanup, restores original residue
        names (PTMs, D-amino acids), writes SSBOND records, appends CONECT
        records for disulfides and metal coordination, and re-inserts stripped
        HETATM ion lines.

        Args:
            output_path: Destination PDB path.
            simulation: Active :class:`app.Simulation` after minimization.
            cyclic: Whether to run cyclic post-processing.
            added_bonds: Detected disulfide pairs as ``(id1, id2)`` strings.
            coordination_restraints: Metal coordination atom-index pairs.
            hetatm_lines: Ion HETATM lines stripped during preprocessing.
            original_metadata: ``{(res_id, chain_id): {"name": ..., "id": ...}}``
                used to restore renamed residues.
            atom_list: Atom list at system-creation time (for CONECT mapping).
        """
        import io as _io

        state = simulation.context.getState(getPositions=True)
        pos   = state.getPositions()
        final_topology  = simulation.topology
        final_positions = pos

        with open(output_path, 'w') as f:
            # Macrocycle Cleanup
            if cyclic:
                try:
                    logger.info("Cleaning up terminal atoms for cyclic peptide output...")
                    mod_modeller = app.Modeller(final_topology, final_positions)
                    residues = list(mod_modeller.topology.residues())
                    if residues:
                        to_prune_caps = [
                            a for r in residues
                            if r.name in ['ACE', 'NME']
                            for a in r.atoms()
                        ]
                        if to_prune_caps:
                            mod_modeller.delete(to_prune_caps)
                            final_topology  = mod_modeller.topology
                            final_positions = mod_modeller.positions
                            residues = list(final_topology.residues())

                        solvent_names = [
                            'HOH', 'WAT', 'SOL', 'TIP3', 'POP',
                            'NA', 'CL', 'ZN', 'FE', 'MG', 'CA',
                        ]
                        amino_residues = [
                            r for r in residues
                            if r.name.strip().upper() not in (['ACE', 'NME'] + solvent_names)
                        ]

                        if amino_residues:
                            res1, resN = amino_residues[0], amino_residues[-1]
                            to_prune = []

                            n1 = next((a for a in res1.atoms() if a.name == 'N'), None)
                            if n1:
                                h_on_n1 = [
                                    a for a in res1.atoms()
                                    if a.element is not None
                                    and a.element.symbol == 'H'
                                    and any(
                                        b.atom1 == n1 or b.atom2 == n1
                                        for b in final_topology.bonds()
                                        if a == b.atom1 or a == b.atom2
                                    )
                                ]
                                if len(h_on_n1) == 1:
                                    h_on_n1[0].name = 'H'

                            oxt = next((a for a in resN.atoms() if a.name == 'OXT'), None)
                            if oxt:
                                to_prune.append(oxt)
                            if to_prune:
                                mod_modeller.delete(to_prune)
                                final_topology  = mod_modeller.topology
                                final_positions = mod_modeller.positions
                except Exception as e:
                    logger.warning(f"Macrocycle cleanup failed: {e}")

            if len(final_positions) == 0:
                logger.error("OpenMM returned empty positions! Topology might be corrupted.")
                return

            # Restore original residue names and IDs
            for res in final_topology.residues():
                res_key = (str(res.id).strip(), res.chain.id)
                if res_key in original_metadata:
                    res.name = original_metadata[res_key]["name"]
                    res.id   = original_metadata[res_key]["id"]

            if added_bonds:
                for s, (id1, id2) in enumerate(added_bonds, 1):
                    try:
                        f.write(
                            f"SSBOND{s:4d} CYS A {int(id1):4d}    "
                            f"CYS A {int(id2):4d}                          \n"
                        )
                    except Exception:
                        pass

            # Build PDB buffer
            pdb_buffer = _io.StringIO()
            app.PDBFile.writeFile(final_topology, final_positions, pdb_buffer)
            pdb_lines = pdb_buffer.getvalue().split('\n')

            # Force CONECT for disulfides
            extra_conects = []
            for bond in final_topology.bonds():
                a1, a2 = bond.atom1, bond.atom2
                if a1.name == 'SG' and a2.name == 'SG':
                    extra_conects.append((a1.index + 1, a2.index + 1))
            for id1, id2 in coordination_restraints:
                extra_conects.append((id1 + 1, id2 + 1))

            final_lines = []
            for line in pdb_lines:
                if line.startswith('END') or line.startswith('CONECT'):
                    continue
                if line.strip():
                    final_lines.append(line)

            for ci1, ci2 in extra_conects:
                final_lines.append(f"CONECT{ci1:5d}{ci2:5d}")
                final_lines.append(f"CONECT{ci2:5d}{ci1:5d}")

            # Restore SSBOND records (after HEADER/TITLE)
            if added_bonds:
                insert_idx = 0
                for idx, l in enumerate(final_lines):
                    if l.startswith(("HEADER", "TITLE", "COMPND")):
                        insert_idx = idx + 1
                for s, (id1, id2) in enumerate(added_bonds, 1):
                    final_lines.insert(
                        insert_idx,
                        f"SSBOND{s:4d} CYS A {int(id1):4d}    "
                        f"CYS A {int(id2):4d}                          ",
                    )

            # Restore stripped ions
            if hetatm_lines:
                for line in hetatm_lines:
                    res_name = line[17:20].strip().upper()
                    logger.debug(f"Appending restored HETATM: {res_name}")
                    final_lines.append(line.strip())

            final_lines.append("END")
            f.write("\n".join(final_lines) + "\n")

'''  # end NEW_METHODS

# ─────────────────────────────────────────────────────────────────────────────
# NEW _run_simulation ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

NEW_RUN_SIM = r'''        """Internal engine. Returns final_energy if successful, else None."""
        logger.info(f"Processing physics for {input_path} (cyclic={cyclic})...")
        import tempfile, os, numpy as np

        # ── Stage 1: PDB preprocessing ──────────────────────────────────────
        try:
            topology, positions, hetatm_lines, original_metadata = \
                self._preprocess_pdb_for_simulation(input_path, cyclic, disulfides)
            atom_list = list(topology.atoms())
        except Exception as e:
            logger.error(f"PDB Pre-processing failed: {e}")
            return None

        # ── Stage 2: Modeller setup (H, SSBOND, salt-bridge, cyclic weld) ──
        try:
            coordination_param = coordination if coordination is not None else []
            modeller, added_bonds, salt_bridge_restraints, coordination_restraints, atom_list = \
                self._setup_openmm_modeller(
                    topology, positions, add_hydrogens, cyclic,
                    coordination_param, atom_list,
                )

            # ── Stage 3: System + forces + Simulation context ───────────────
            simulation, system, n_idx, c_idx, topology, positions = \
                self._build_simulation_context(
                    modeller, cyclic, added_bonds,
                    salt_bridge_restraints, coordination_restraints, atom_list,
                )

            # Health check
            if len(list(topology.atoms())) == 0:
                logger.error("Health Check Failed: Topology has 0 atoms!")
                if len(positions) == 0:
                    logger.error("OpenMM returned empty positions! Topology might be corrupted.")
                return None

            # Single-point energy calculation (bypass minimization)
            if max_iterations < 0:
                logger.info(
                    "Single-point energy calculation (max_iterations < 0). "
                    "Skipping minimization."
                )
                state = simulation.context.getState(getEnergy=True)
                return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

            # ── Stage 4: Minimization / equilibration ───────────────────────
            logger.info(f"Minimizing (Tolerance={tolerance} kJ/mol, MaxIter={max_iterations})...")
            if cyclic or added_bonds or salt_bridge_restraints:
                cyc_iter = 0
                logger.info(
                    "Macrocycle/Disulfide Optimization: "
                    "Running unlimited iterations for closure."
                )

                if salt_bridge_restraints:
                    sb_force = mm.CustomBondForce("0.5*k_sb*(r-r0)^2")
                    sb_force.addGlobalParameter(
                        "k_sb",
                        10000.0 * unit.kilojoules_per_mole / unit.nanometer ** 2,
                    )
                    sb_force.addPerBondParameter("r0")
                    new_ats = list(topology.atoms())
                    for ao, bo, r0 in salt_bridge_restraints:
                        oa, ob = atom_list[ao], atom_list[bo]
                        na, nb = -1, -1
                        for a in new_ats:
                            if (
                                str(a.residue.id).strip() == str(oa.residue.id).strip()
                                and a.name == oa.name
                            ):
                                na = a.index
                            if (
                                str(a.residue.id).strip() == str(ob.residue.id).strip()
                                and a.name == ob.name
                            ):
                                nb = a.index
                        if na != -1 and nb != -1:
                            sb_force.addBond(na, nb, [r0 * unit.nanometers])
                    system.addForce(sb_force)

                simulation.minimizeEnergy(
                    maxIterations=cyc_iter,
                    tolerance=(tolerance * 0.1) * unit.kilojoule / (unit.mole * unit.nanometer),
                )

                if cyclic:
                    logger.info(
                        "Thermal Jiggling: Applying random perturbation to break deadlocks."
                    )
                    try:
                        state = simulation.context.getState(getPositions=True)
                        try:
                            pos = state.getPositions(asNumpy=True)
                        except Exception:
                            pos = state.getPositions()
                        if len(pos) > 0:
                            pos_np = (
                                np.array(pos.value_in_unit(unit.nanometers))
                                if hasattr(pos, 'value_in_unit')
                                else np.array(pos)
                            )
                            noise = np.random.normal(0, 0.05, (len(pos_np), 3))
                            simulation.context.setPositions((pos_np + noise) * unit.nanometers)
                            simulation.minimizeEnergy(
                                maxIterations=cyc_iter,
                                tolerance=(tolerance * 0.1) * unit.kilojoule / (unit.mole * unit.nanometer),
                            )
                    except Exception as e:
                        logger.debug(f"Thermal jiggling failed (likely mocked): {e}")

                    logger.info(
                        "Iterative Closure: Reinforcing pull force and refining geometry."
                    )
                    simulation.minimizeEnergy(
                        maxIterations=0,
                        tolerance=(tolerance * 0.01) * unit.kilojoule / (unit.mole * unit.nanometer),
                    )

                    logger.info(
                        "Final Constraint Refinement: Forcing 1.33A closure via reinitialization."
                    )
                    system.addConstraint(n_idx, c_idx, 0.133 * unit.nanometers)
                    simulation.context.reinitialize(preserveState=True)
                    simulation.minimizeEnergy(
                        maxIterations=0,
                        tolerance=(tolerance * 0.001) * unit.kilojoule / (unit.mole * unit.nanometer),
                    )
            else:
                simulation.minimizeEnergy(
                    maxIterations=max_iterations,
                    tolerance=tolerance * unit.kilojoule / (unit.mole * unit.nanometer),
                )

            # Post-minimization health check
            final_state = simulation.context.getState(getPositions=True, getEnergy=True)
            final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

            try:
                final_pos = simulation.context.getState(getPositions=True).getPositions()
                if hasattr(final_pos, 'value_in_unit'):
                    check_pos = np.array(final_pos.value_in_unit(unit.nanometers))
                    if check_pos.size > 0 and np.any(np.isnan(check_pos)):
                        logger.error("Health Check Failed: Atomic Coordinates contain NaNs!")
                        return None
            except Exception:
                logger.debug("Health check (isnan) skipped due to non-standard context.")

            try:
                val_energy = float(final_energy)
                if val_energy > 1e6:
                    logger.warning(
                        f"Health Check Warning: High Potential Energy "
                        f"({val_energy:.2e} kJ/mol). Structure may contain severe clashes."
                    )
                if np.isnan(val_energy):
                    logger.error("Health Check Failed: Potential Energy is NaN!")
                    return None
            except Exception:
                pass

            # Equilibration steps
            if equilibration_steps > 0:
                simulation.step(equilibration_steps)

            # ── Stage 5: Write output PDB ────────────────────────────────────
            self._finalize_output(
                output_path, simulation, cyclic, added_bonds,
                coordination_restraints, hetatm_lines, original_metadata, atom_list,
            )
            return final_energy

        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            return None
'''

# ─────────────────────────────────────────────────────────────────────────────
# Perform the patch
# ─────────────────────────────────────────────────────────────────────────────

START_MARKER = "    def _run_simulation(self, input_path, output_path"
END_MARKER = '\n        except Exception as e:\n            logger.error(f"Simulation failed: {e}", exc_info=True)\n            return None\n'

start_idx = original.find(START_MARKER)
assert start_idx != -1, "ERROR: Could not find _run_simulation method"

# Find end of the method: the closing of the big try/except
end_idx = original.find(END_MARKER, start_idx)
assert end_idx != -1, "ERROR: Could not find end of _run_simulation"
end_idx += len(END_MARKER)  # include the closing lines

# Find the _run_simulation signature line(s) up to and including the docstring
prefix = original[:start_idx]

# Locate the docstring inside _run_simulation
doc_start = original.find('"""', start_idx)
doc_end = original.find('"""', doc_start + 3) + 3
run_sim_signature_and_doc = original[start_idx:doc_end]

suffix = original[end_idx:]

new_content = (
    prefix
    + NEW_METHODS  # 4 new private methods
    + "\n"
    + run_sim_signature_and_doc  # original signature + docstring
    + "\n"
    + NEW_RUN_SIM  # new ~120-line orchestrator body
    + "\n"
    + suffix
)

# Syntax-check before writing
try:
    ast.parse(new_content)
except SyntaxError as e:
    print(f"SYNTAX ERROR in generated code: {e}")
    sys.exit(1)

PHYSICS.write_text(new_content)
print(f"SUCCESS: {PHYSICS} patched.")
print("  4 new EnergyMinimizer methods inserted before _run_simulation.")
print("  Old _run_simulation body replaced with ~120-line orchestrator.")
