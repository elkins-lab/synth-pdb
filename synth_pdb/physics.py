import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    app = None
    mm = None
    unit = None
import os

import numpy as np

# Constants
# SSBOND_CAPTURE_RADIUS determines the maximum distance (in Angstroms) between two Sulfur atoms
# for them to be considered as a potential disulfide bond.
# Linear chains being cyclized can have terminals > 15A apart initially.
SSBOND_CAPTURE_RADIUS = 18.0

logger = logging.getLogger(__name__)

# Global cache for ForceField objects to prevent memory leaks and speed up initialization.
# Loading XML forcefield files repeatedly is expensive and can accumulate memory.
_FORCEFIELD_CACHE: Dict[Tuple[str, ...], "app.ForceField"] = {}


class EnergyMinimizer:
    """Performs energy minimization on molecular structures using OpenMM.

    ### Educational Note: What is Energy Minimization?
    --------------------------------------------------
    Proteins fold into specific 3D shapes to minimize their "Gibbs Free Energy".
    A generated structure (like one built from simple geometry) often has "clashes"
    where atoms are too close (high Van der Waals repulsion) or bond angles are strained.

    Energy Minimization is like rolling a ball down a hill. The "Energy Landscape"
    represents the potential energy of the protein as a function of all its atom coordinates.
    The algorithm moves atoms slightly to reduce this energy, finding a local minimum
    where the structure is physically relaxed.

    ### Educational Note - Metal Coordination in Physics:
    -----------------------------------------------------
    Metal ions like Zinc (Zn2+) are not "bonded" in the same covalent sense as Carbon-Carbon
    bonds in classical forcefields. Instead, they are typically modeled as point charges
    held by electrostatics and Van der Waals forces.

    In this tool, we automatically detect potential coordination sites (like Zinc Fingers).
    To maintain the geometry during minimization, we apply Harmonic Constraints
    that act like springs, tethering the Zinc to its ligands (Cys/His).
    We also deprotonate coordinating Cysteines to represent the thiolate state.

    ### NMR Perspective:
    In NMR structure calculation (e.g., CYANA, XPLOR-NIH), minimization is often part of
    "Simulated Annealing". Structures are calculated to satisfy experimental restraints
    (NOEs, J-couplings) and then energy-minimized to ensure good geometry.
    This module performs that final "geometry regularization" step.
    """

    def __init__(
        self,
        forcefield_name: str = "amber14-all.xml",
        solvent_model: str = "app.OBC2",
        box_size: float = 1.0,
        disable_cache: bool = False,
    ) -> None:
        """Initialize the Minimizer with a Forcefield and Solvent Model.

        Args:
            forcefield_name: The "rulebook" for how atoms interact.
                             'amber14-all.xml' describes protein atoms (parameters for bond lengths,
                             angles, charges, and VdW radii).
            solvent_model:   How water is simulated.
                             'explicit' will use a TIP3P water box (High Fidelity).
                             'app.OBC2' is an "Implicit Solvent" model (High Performance).
            box_size:        The padding distance (in nm) for the explicit solvent box.
                             Default 1.0 nm ensures the protein doesn't see its own image.
            disable_cache:   If True, reloads ForceField from scratch (used for testing).

        ### EDUCATIONAL NOTE - Explicit vs. Implicit Solvent:
        ---------------------------------------------------
        1. **Explicit Solvent (TIP3P)**:
           Every water molecule (H2O) is simulated as a rigid 3-site model. This captures the
           "Enthalpic" and "Entropic" costs of cavity formation and hydrogen bonding.

           *Deep Dive*: TIP3P is the "standard" but modern simulations often use TIP4P/Ew
           for better electrostatic performance.

        2. **Implicit Solvent (Generalized Born / OBC)**:
           Also known as "Born Solvation". The cost of moving an ion from vacuum (ε=1)
           to water (ε=80) is estimated by the **Born Equation**:

           ΔG_solv = - (q^2 / 2r) * (1 - 1/ε)

           In proteins, each atom has a unique "Effective Born Radius" based on how buried
           it is. Surface atoms feel the full ε=80, while core atoms are shielded.
           The **OBC2 (Onufriev-Bashford-Case)** model is a refined version that
           parameterizes these radii to match explicit solvent behavior closely.

        """
        if not HAS_OPENMM:
            return

        # Normalize string inputs from CLI (e.g. "obc2") to OpenMM names ("app.OBC2")
        if isinstance(solvent_model, str) and solvent_model.lower() in [
            "obc2",
            "obc1",
            "gbn",
            "gbn2",
            "hct",
        ]:
            name_map = {"obc2": "OBC2", "obc1": "OBC1", "gbn": "GBn", "gbn2": "GBn2", "hct": "HCT"}
            solvent_model = f"app.{name_map[solvent_model.lower()]}"

        # Robust Validation
        valid_implicit = ["app.OBC2", "app.OBC1", "app.GBn", "app.GBn2", "app.HCT"]
        if (
            solvent_model != "explicit"
            and solvent_model not in valid_implicit
            and not hasattr(app, str(solvent_model).split(".")[-1])
        ):
            logger.warning(f"Unknown solvent model '{solvent_model}'. Defaulting to 'explicit'.")
            solvent_model = "explicit"

        if box_size <= 0:
            raise ValueError("box_size must be positive (nm).")

        if solvent_model == "explicit" and box_size <= 1.0:
            logger.warning(
                f"Explicit solvent box_size ({box_size} nm) is dangerously small. "
                f"OpenMM requires the box to be at least twice the nonbonded cutoff (1.0 nm). "
                f"Increasing box_size to 1.1 nm to prevent NonbondedForce creation errors."
            )
            box_size = 1.1

        self.forcefield_name = forcefield_name
        self.water_model = "amber14/tip3pfb.xml"
        self.solvent_model = solvent_model
        self.box_size = box_size * unit.nanometers
        ff_files = [self.forcefield_name]

        if self.solvent_model == "explicit":
            ff_files.append(self.water_model)
        else:
            solvent_xml_map = {
                app.OBC2: "implicit/obc2.xml",
                app.OBC1: "implicit/obc1.xml",
                app.GBn: "implicit/gbn.xml",
                app.GBn2: "implicit/gbn2.xml",
                app.HCT: "implicit/hct.xml",
            }
            # Resolve if passed as string or object
            self.implicit_solvent_enum = (
                solvent_model
                if not isinstance(solvent_model, str)
                else getattr(app, str(solvent_model).split(".")[-1], None)
            )
            if self.implicit_solvent_enum in solvent_xml_map:
                ff_files.append(solvent_xml_map[self.implicit_solvent_enum])
                # The solvent is fully configured via the XML file above.
                # Setting implicit_solvent_enum to None prevents _create_system_robust
                # from also passing it as a createSystem() kwarg, which modern OpenMM
                # rejects as an unused argument (triggering a warning + retry every run).
                self.implicit_solvent_enum = None

        try:
            # Use global cache to avoid redundant loading (unless disabled for tests)
            if not disable_cache:
                ff_key = tuple(sorted(ff_files))
                if ff_key not in _FORCEFIELD_CACHE:
                    logger.debug(f"Loading ForceField: {ff_key}")
                    _FORCEFIELD_CACHE[ff_key] = app.ForceField(*ff_files)
                self.forcefield = _FORCEFIELD_CACHE[ff_key]
            else:
                self.forcefield = app.ForceField(*ff_files)
        except Exception as e:
            logger.error(f"Failed to load forcefield: {e}")
            raise

    def minimize(
        self,
        pdb_file_path: str,
        output_path: str,
        max_iterations: int = 0,
        tolerance: float = 10.0,
        cyclic: bool = False,
        disulfides: Optional[List] = None,
        coordination: Optional[List] = None,
    ) -> bool:
        """Run energy minimization to regularize geometry and resolve clashes.

        Uses OpenMM with implicit solvent (OBC2) and the AMBER forcefield.
        This provides a "physically valid" structure by moving atoms into their
        local energy minimum.

        ### EDUCATIONAL NOTE - Anatomy of a Forcefield:
        -------------------------------------------
        A forcefield (like Amber14) approximates the potential energy (U) of a
        molecule as a sum of four main terms:

        U = U_bond + U_angle + U_torsion + [U_vdw + U_elec]

        1. Bonded Terms (Springs):
           - U_bond/U_angle: Atoms behave like balls on springs. Pushing them
             away from ideal (equilibrium) lengths/angles costs energy.
           - U_torsion: Rotation around bonds is restricted by periodic potential
             wells (e.g., the preference for trans vs cis).
        2. Non-Bonded Terms (Distant Neighbors):
           - U_vdw (Lennard-Jones): Models Steric Repulsion (don't overlap!) and
             London Dispersion (subtle attraction).
           - U_elec (Coulomb): Attraction between opposite charges (e.g., a
             Salt Bridge) and repulsion between like charges.

        Minimization is the process of finding the coordinate set where $dU/dX = 0$.

        Args:
            pdb_file_path: Input PDB path.
            output_path: Output PDB path.
            max_iterations: Limit steps (0 = until convergence).
            tolerance: Target energy convergence threshold (kJ/mol).
            cyclic: Whether to apply head-to-tail peptide bond constraints.
            disulfides: Optional list of (res1, res2) indices for SSBOND constraints.
            coordination: Optional list of (ion_name, [res_indices]) for metal constraints.

        ### Educational Note - Computational Efficiency:
        ----------------------------------------------
        Energy Minimization is an O(N^2) or O(N log N) operation depending on the method.
        Starting with a structure that satisfies Ramachandran constraints (from `validator.py`)
        can reduce convergence time by 10-50x compared to minimizing a random coil.

        Effectively, the validator acts as a "pre-minimizer", placing atoms in the
        correct basin of attraction so the expensive physics engine only needs to
        perform local optimization.

        ### NMR Realism:
        In NMR structure calculation (e.g., CYANA/XPLOR), we often use "Simulated Annealing"
        to find low energy states. `minimize` is a simpler, gradient-based version
        of this process. It ensures bond lengths and angles are correct before
        performing more complex MD.

        Returns:
            True if successful.

        """
        if not HAS_OPENMM:
            logger.error("Cannot minimize: OpenMM not found.")
            return False
        res = self._run_simulation(
            pdb_file_path,
            output_path,
            add_hydrogens=False,
            max_iterations=max_iterations,
            tolerance=tolerance,
            cyclic=cyclic,
            disulfides=disulfides,
            coordination=coordination,
        )
        return res is not None

    def equilibrate(
        self,
        pdb_file_path: str,
        output_path: str,
        steps: int = 1000,
        cyclic: bool = False,
        disulfides: Optional[List] = None,
        coordination: Optional[List] = None,
    ) -> bool:
        """Run Thermal Equilibration (MD) at 300K.

        Args:
            pdb_file_path: Input PDB/File path.
            output_path: Output PDB path.
            steps: Number of MD steps (2 fs per step). 1000 steps = 2 ps.
            cyclic: Whether to apply head-to-tail peptide bond constraints.
            disulfides: Optional list of (res1, res2) indices for SSBOND constraints.
            coordination: Optional list of (ion_name, [res_indices]) for metal constraints.

        ### Educational Note - Thermal Equilibration:
        -------------------------------------------
        After finding a local energy minimum (where atoms are perfectly still at 0 K),
        we need to bring the system up to "room temperature" (300 K).

        We "heat" the system by assigning random velocities to all atoms according
        to a Maxwell-Boltzmann distribution for 300 K. We then simulate the Newtonian
        equations of motion over time (F = ma).

        This step allows the protein to "settle" and find a stable dynamic average
        structure rather than being trapped in a rigid unnatural minimum. In NMR,
        the true structure is an ensemble of these room-temperature states, not
        a single frozen snapshot.

        Returns:
            True if successful.

        """
        if not HAS_OPENMM:
            logger.error("Cannot equilibrate: OpenMM not found.")
            return False
        res = self._run_simulation(
            pdb_file_path,
            output_path,
            add_hydrogens=True,
            equilibration_steps=steps,
            cyclic=cyclic,
            disulfides=disulfides,
            coordination=coordination,
        )
        return res is not None

    def add_hydrogens_and_minimize(
        self,
        pdb_file_path: str,
        output_path: str,
        max_iterations: int = 0,
        tolerance: float = 10.0,
        cyclic: bool = False,
        disulfides: Optional[List] = None,
        coordination: Optional[List] = None,
    ) -> bool:
        """Robust minimization pipeline: Adds Hydrogens -> Creates/Minimizes System -> Saves Result.

        ### Why Add Hydrogens?
        X-ray crystallography often doesn't resolve hydrogen atoms because they have very few electrons.
        However, Molecular Dynamics forcefields (like Amber) are explicitly "All-Atom". They REQUIRE
        hydrogens to calculate bond angles and electrostatics (h-bonds) correctly.

        ### NMR Perspective:
        Unlike X-ray, NMR relies entirely on the magnetic spin of protons (H1). Hydrogens are
        the "eyes" of NMR. Correctly placing them is critical not just for physics but for
        predicting NOEs (Nuclear Overhauser Effects) which depend on H-H distances.
        We use `app.Modeller` to "guess" the standard positions of hydrogens at specific pH (7.0).

        Args:
            pdb_file_path: Input PDB path.
            output_path: Output PDB path.
            max_iterations: Limit steps (0 = until convergence).
            tolerance: Target energy convergence threshold (kJ/mol).
            cyclic: Whether to apply head-to-tail peptide bond constraints.
            disulfides: Optional list of (res1, res2) indices for SSBOND constraints.
            coordination: Optional list of (ion_name, [res_indices]) for metal constraints.

        Returns:
            True if successful.

        """
        if not HAS_OPENMM:
            logger.error("Cannot add hydrogens: OpenMM not found.")
            return False
        res = self._run_simulation(
            pdb_file_path,
            output_path,
            add_hydrogens=True,
            max_iterations=max_iterations,
            tolerance=tolerance,
            cyclic=cyclic,
            disulfides=disulfides,
            coordination=coordination,
        )
        return res is not None

    def calculate_energy(
        self, input_data: Union[str, Any], cyclic: bool = False
    ) -> Optional[float]:
        """Calculates the potential energy of a structure.

        Args:
            input_data: Can be a PDB file path, a PDB string, or a PeptideResult object.
            cyclic: Whether the peptide is cyclic.

        Returns:
            float: Potential energy in kJ/mol.

        """
        if not HAS_OPENMM:
            return 0.0

        # Handle different input types
        pdb_path = None
        temp_file = None

        import tempfile

        try:
            if (
                isinstance(input_data, str)
                and input_data.endswith(".pdb")
                and os.path.exists(input_data)
            ):
                pdb_path = input_data
            else:
                # Treat as PDB content or object with .pdb property
                content = input_data.pdb if hasattr(input_data, "pdb") else str(input_data)
                temp_file = tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False)
                temp_file.write(content)
                temp_file.close()
                pdb_path = temp_file.name

            # Use a dummy output path as we don't care about the result
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "energy_calc.pdb")
                # We use _run_simulation with max_iterations=1 to just get the initial state's energy?
                # Actually, _run_simulation usually minimizes.
                # To get the energy WITHOUT moving atoms, we need a "0-step" simulation.
                # I'll update _run_simulation to handle max_iterations=0 correctly or
                # just use the energy from the first step.
                # Actually, I'll pass a special flag or just use max_iterations=0 and handle it.
                # For now, let's assume _run_simulation returns the energy if we add a return value.
                # Wait, I didn't see _run_simulation return energy.
                # I'll add a 'return_energy' parameter to _run_simulation.
                return self._run_simulation(pdb_path, out_path, max_iterations=-1, cyclic=cyclic)
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass

    def _create_system_robust(
        self, topology: Any, constraints: Any, modeller: Optional[Any] = None
    ) -> Tuple[Any, Any, Any]:
        """Creates an OpenMM system, with robust fallbacks for template mismatches
        and incompatible forcefield arguments. Returns (system, topology, positions).
        """
        if not hasattr(self, "_suppressed_args"):
            self._suppressed_args: set[str] = set()

        sys_kwargs = {"nonbondedMethod": app.NoCutoff, "constraints": constraints}
        if (
            self.implicit_solvent_enum is not None
            and "implicitSolvent" not in self._suppressed_args
        ):
            sys_kwargs["implicitSolvent"] = self.implicit_solvent_enum

        current_topo = topology
        current_pos = modeller.positions if modeller else None

        def _try_create(topo: Any, **kwargs: Any) -> Any:
            nonlocal current_topo, current_pos
            try:
                system = self.forcefield.createSystem(topo, **kwargs)
                return system, topo, (modeller.positions if modeller else None)
            except Exception as e:
                msg = str(e)
                # Fallback 1: Forcefield doesn't support an argument (e.g. implicitSolvent)
                if "was specified to createSystem() but was never used" in msg:
                    for arg in ["implicitSolvent"]:
                        if arg in msg and arg in kwargs:
                            logger.warning(
                                f"Forcefield does not support {arg}. Retrying without it and suppressing for future calls..."
                            )
                            self._suppressed_args.add(arg)
                            del kwargs[arg]
                            return cast(Tuple[Any, Any, Any], _try_create(topo, **kwargs))

                # Fallback 2: Template mismatch (Hydrogen issues)
                if "No template found" in msg and modeller is not None:
                    try:
                        logger.warning(
                            f"Template mismatch: {msg}. Attempting re-protonation repair..."
                        )
                        # Strip and re-add hydrogens
                        h_atoms = [
                            a
                            for a in modeller.topology.atoms()
                            if a.element and a.element.symbol == "H"
                        ]
                        if h_atoms:
                            modeller.delete(h_atoms)
                        modeller.addHydrogens(self.forcefield)
                        current_topo = modeller.topology
                        current_pos = modeller.positions
                        return cast(Tuple[Any, Any, Any], _try_create(current_topo, **kwargs))
                    except Exception as repair_e:
                        logger.warning(f"Repair failed: {repair_e}")

                raise e

        try:
            return cast(Tuple[Any, Any, Any], _try_create(current_topo, **sys_kwargs))
        except Exception as final_e:
            logger.warning(
                f"Robust system creation failed, final fallback to no constraints: {final_e}"
            )
            sys = self.forcefield.createSystem(
                current_topo, nonbondedMethod=app.NoCutoff, constraints=None
            )
            return (sys, current_topo, current_pos)

    def _preprocess_pdb_for_simulation(
        self, input_path: str, cyclic: bool, disulfides_param: Optional[List]
    ) -> Tuple[Any, Any, List[str], Dict[Any, Any]]:
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
        import os
        import tempfile

        # EDUCATIONAL NOTE - PDB PRE-PROCESSING (OpenMM Template Fix):
        # -----------------------------------------------------------
        # OpenMM's standard forcefields (amber14-all) are highly optimized for wild-type
        # human proteins but frequently lack templates for:
        # 1. Phosphorylated residues (SEP, TPO, PTR)
        # 2. Histidine tautomers (HIE, HID) named explicitly in the input.
        # 3. D-Amino Acids (DAL, DPH, etc.) - These require L-analog templates.
        #
        # To prevent "No template found" errors, we surgically rename residues to
        # their standard counterparts BEFORE loading. We preserve the original
        # identity in `original_metadata` for final restoration.
        ptm_map = {
            "SEP": "SER",
            "TPO": "THR",
            "PTR": "TYR",
            "HIE": "HIS",
            "HID": "HIS",
            "HIP": "HIS",
            "DAL": "ALA",
            "DAR": "ARG",
            "DAN": "ASN",
            "DAS": "ASP",
            "DCY": "CYS",
            "DGL": "GLU",
            "DGN": "GLN",
            "DHI": "HIS",
            "DIL": "ILE",
            "DLE": "LEU",
            "DLY": "LYS",
            "DME": "MET",
            "DPH": "PHE",
            "DPR": "PRO",
            "DSE": "SER",
            "DTH": "THR",
            "DTR": "TRP",
            "DTY": "TYR",
            "DVA": "VAL",
        }
        ptm_atom_names = ["P", "O1P", "O2P", "O3P"]

        original_metadata: dict = {}
        modified_lines: list = []
        hetatm_lines: list = []
        last_res_key = None
        first_res_id = None
        last_res_id = None

        if os.path.exists(input_path):
            with open(input_path) as f:
                pdb_lines = f.readlines()

            atom_lines = [line for line in pdb_lines if line.startswith("ATOM")]
            first_res_id = atom_lines[0][22:26].strip() if atom_lines else None
            last_res_id = atom_lines[-1][22:26].strip() if atom_lines else None

            # EDUCATIONAL NOTE - Cyclic CONECT Stripping:
            # OpenMM's PDB reader creates CONECT records for all explicit bonds.
            # For cyclic peptides, the head-to-tail bond is already encoded as
            # a CONECT (written by generator.py). We must remove it here so
            # addHydrogens later does not see a conflicting terminal N–C bond.
            n_term_serial, c_term_serial = None, None
            c_coords, c_line_template = None, None
            if cyclic and atom_lines:
                for line in atom_lines:
                    res_id = line[22:26].strip()
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
                        if (parts[1] == n_term_serial and parts[2] == c_term_serial) or (
                            parts[1] == c_term_serial and parts[2] == n_term_serial
                        ):
                            print(f"DEBUG: Skipping cyclic CONECT: {line.strip()}")
                            continue

                if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
                    res_name = line[17:20].strip()
                    res_id = line[22:26].strip()
                    chain_id = line[21] if len(line) > 21 else " "
                    res_key = (res_id, chain_id)
                    atom_name = line[12:16].strip()

                    if res_key != last_res_key:
                        last_res_key = res_key
                        original_metadata[res_key] = {"name": res_name, "id": res_id}

                    res_name_upper = line[17:20].strip().upper()
                    # EDUCATIONAL NOTE - Ion Stripping:
                    # Ions like Zn2+, Fe2+, Mg2+ crash Modeller.addHydrogens()
                    # because they have no hydrogen template. We stash them in
                    # hetatm_lines and re-append them after minimization.
                    if res_name_upper in ["ZN", "FE", "MG", "CA", "NA", "CL"]:
                        hetatm_lines.append(line)
                        logger.info(f"Restoring lost HETATM: {res_name_upper}")
                        continue

                    if res_name in ptm_map:
                        new_name = ptm_map[res_name]
                        line = line[:17] + f"{new_name: >3}" + line[20:]
                        if res_name in ["SEP", "TPO", "PTR"] and len(line) >= 16:
                            if atom_name in ptm_atom_names:
                                continue
                modified_lines.append(line)

            # EDUCATIONAL NOTE - Dummy OXT Insertion:
            # OpenMM amber14 residue templates for C-termini require an OXT
            # oxygen to match the "C_TERM" patch. Cyclic peptides lack this atom
            # so we add a temporary OXT positioned ~1.2 Å from the terminal C.
            # physics.py _finalize_output() will delete it after minimization.
            # Add dummy OXT for cyclic peptides to satisfy C-terminal templates
            if cyclic and last_res_id and c_line_template:
                insert_idx = -1
                for idx, line in enumerate(modified_lines):
                    if line.startswith("ATOM") and line[22:26].strip() == last_res_id:
                        insert_idx = idx + 1
                if insert_idx != -1 and c_coords is not None:
                    x, y, z = c_coords
                    res_name_c = c_line_template[17:20]
                    res_id_full = c_line_template[21:26]
                    oxt_line = (
                        f"ATOM   9999  OXT {res_name_c} {res_id_full}    "
                        f"{x + 1.2:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\n"
                    )
                    modified_lines.insert(insert_idx, oxt_line)
                    logger.info(
                        f"Added temporary OXT to residue {last_res_id} "
                        f"(Renamed: {res_name_c.strip()})"
                    )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tf:
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

        # EDUCATIONAL NOTE - Topological Validation:
        # ------------------------------------------
        # OpenMM's PDB reader can sometimes skip bonds if they aren't explicitly
        # in CONECT records or deviate too far from their ideal length.
        # We force bond generation to ensure standard residues have
        # all internal bonds defined, which is required for template matching.
        topology.createStandardBonds()
        topology.createDisulfideBonds(positions)

        # Surgically remove head-to-tail bond so addHydrogens doesn't fail
        if cyclic:
            bonds_to_remove = []
            res_list = list(topology.residues())
            if len(res_list) >= 2:
                first_res, last_res = res_list[0], res_list[-1]
                for bond in topology.bonds():
                    if (bond[0].residue == first_res and bond[1].residue == last_res) or (
                        bond[0].residue == last_res and bond[1].residue == first_res
                    ):
                        if (bond[0].name == "N" and bond[1].name == "C") or (
                            bond[0].name == "C" and bond[1].name == "N"
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
        self,
        topology: Any,
        positions: Any,
        add_hydrogens: bool,
        cyclic: bool,
        coordination_param: Optional[List],
        atom_list: List[Any],
    ) -> Tuple[Any, List, List, List, List[Any]]:
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
        import io as _io

        import biotite.structure.io.pdb as biotite_pdb

        coordination_restraints: list = []
        salt_bridge_restraints: list = []

        # EDUCATIONAL NOTE: We do NOT add the bond to the Topology here.
        # Adding it here causes OpenMM's template-matcher to fail ("Too many external bonds").
        # Instead, we use massive harmonic restraints to CLOSE the ring physically.
        if cyclic:
            logger.info("Cyclizing peptide via harmonic restraints (Restraint-First approach).")

        modeller = app.Modeller(topology, positions)

        # EDUCATIONAL NOTE - Topology Bridging (welding the ring):
        # --------------------------------------------------------
        # Adding the head-to-tail bond to the Topology triggers OpenMM's Amber
        # template matcher to look for non-existent cyclic templates or
        # specialized patches, leading to "Too many external bonds" errors.
        # We use Restraints instead (applied later in _build_simulation_context).
        #
        # EDUCATIONAL NOTE - Robust Backbone Stitching (Heuristic Bonding):
        # -----------------------------------------------------------------
        # When building de novo structures, standard PDB-to-Topology builders
        # often miss local connectivity. We implement a "Heuristic Welder" that
        # looks for missing C-N peptide bonds based on proximity. If two
        # sequential residues are close but unbonded, we manually weld their
        # backbone atoms to ensure a continuous, force-propagating chain.
        # Heuristic backbone stitching
        try:
            residues = list(modeller.topology.residues())
            existing_bonds = {
                frozenset([b[0].index, b[1].index]) for b in modeller.topology.bonds()
            }
            for i in range(len(residues) - 1):
                res1, res2 = residues[i], residues[i + 1]
                # Only stitch if they are on the same chain
                if res1.chain.id != res2.chain.id:
                    continue

                c_s = next((a for a in res1.atoms() if a.name == "C"), None)
                n_s = next((a for a in res2.atoms() if a.name == "N"), None)
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
                modeller.delete(
                    [
                        a
                        for a in modeller.topology.atoms()
                        if a.element is not None and a.element.symbol == "H"
                    ]
                )
            except Exception as e:
                logger.debug(f"H deletion failed: {e}")

        # EDUCATIONAL NOTE - The SSBOND Capture Radius:
        # ---------------------------------------------
        # Unlike distance-based bonding in simple geometry, physical disulfide
        # formation is highly sensitive to the S-S distance (~2.03 Å).
        # We use a large "Capture Radius" (SSBOND_CAPTURE_RADIUS) to detect
        # potential pairs in un-optimized structures, then allow the "Mega-Pull"
        # to bring them into the ideal covalent distance.
        # Disulfide bond detection by SG proximity
        try:
            cys_residues = [r for r in modeller.topology.residues() if r.name in ("CYS", "CYX")]
            res_to_sg = {
                r.index: [a for a in r.atoms() if a.name == "SG"][0]
                for r in cys_residues
                if any(a.name == "SG" for a in r.atoms())
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
            potential_bonds.sort(key=lambda x: float(x[0]))

            bonded_indices: set = set()
            for _d, r1, r2, s1, s2 in potential_bonds:
                if r1.index in bonded_indices or r2.index in bonded_indices:
                    continue
                modeller.topology.addBond(s1, s2)
                added_bonds.append((str(r1.id).strip(), str(r2.id).strip()))
                bonded_indices.add(r1.index)
                bonded_indices.add(r2.index)
        except Exception as e:
            logger.warning(f"SSBOND failed: {e}")

        # EDUCATIONAL NOTE - Salt Bridges & Electrostatics:
        # -------------------------------------------------
        # A Salt Bridge is an electrostatic attraction between a cationic sidechain
        # (e.g. Lysine/Arginine) and an anionic one (Aspartate/Glutamate).
        # Forcefields model these naturally via Coulomb's law, but in vacuum
        # simulations, the attraction can be artificially weak or slow to form.
        # We apply harmonic "Bungee" restraints to help these bridges snap together.
        # Salt bridge + metal coordination detection via biotite
        try:
            from .biophysics import find_salt_bridges
            from .cofactors import find_metal_binding_sites

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
                # We use BOTH caller-supplied sites (if any) and internally detected ones.
                # Convert caller-supplied site dictionaries into the list of atom-index pairs
                # that _build_simulation_context expects.

                # 1. Process caller sites (from generator)
                if coordination_param:
                    for site in coordination_param:
                        i_idx_at = -1
                        # Find the ion atom index in the topology
                        for atom in atom_list:
                            if atom.residue.name == site["type"]:
                                i_idx_at = atom.index
                                break
                        if i_idx_at != -1:
                            # Map ligand indices from b_struc to topology
                            for l_idx in site["ligand_indices"]:
                                l_at = b_struc[l_idx]
                                for atom in atom_list:
                                    if (
                                        int(atom.residue.id) == int(l_at.res_id)
                                        and atom.name == l_at.atom_name
                                    ):
                                        coordination_restraints.append((i_idx_at, atom.index))
                                        break

                # 2. Add internally detected sites (if not already covered)
                internal_sites = find_metal_binding_sites(b_struc)
                for site in internal_sites:
                    i_idx_at = -1
                    for atom in atom_list:
                        if atom.residue.name == site["type"]:
                            i_idx_at = atom.index
                            break
                    if i_idx_at != -1:
                        for l_idx in site["ligand_indices"]:
                            l_at = b_struc[l_idx]
                            for atom in atom_list:
                                if (
                                    int(atom.residue.id) == int(l_at.res_id)
                                    and atom.name == l_at.atom_name
                                ):
                                    pair = (i_idx_at, atom.index)
                                    if pair not in coordination_restraints:
                                        coordination_restraints.append(pair)
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

        # EDUCATIONAL NOTE - Why Add Hydrogens?
        # X-ray crystallography often doesn't resolve hydrogen atoms because they
        # have very few electrons. However, Molecular Dynamics forcefields (like
        # Amber) are explicitly "All-Atom". They REQUIRE hydrogens to calculate
        # bond angles and electrostatics (h-bonds) correctly.
        #
        # NMR Perspective: Unlike X-ray, NMR relies entirely on the magnetic spin
        # of protons (H1). Correctly placing them is critical not just for physics
        # but for predicting NOEs (Nuclear Overhauser Effects) which depend on
        # H-H distances. We use `app.Modeller` to "guess" the standard positions
        # of hydrogens at specific pH (7.0).
        # Add hydrogens
        if add_hydrogens:
            modeller.addHydrogens(self.forcefield, pH=7.0)

        # Post-hydrogen cyclic weld
        if cyclic:
            try:
                res = list(modeller.topology.residues())
                if len(res) >= 2:
                    res1, res_n = res[0], res[-1]
                    c_at = next((a for a in res_n.atoms() if a.name == "C"), None)
                    n_at = next((a for a in res1.atoms() if a.name == "N"), None)
                    if c_at and n_at:
                        modeller.topology.addBond(c_at, n_at)
                        logger.info(
                            f"Welded cyclic link in Topology: "
                            f"{res_n.name}{res_n.id} -> {res1.name}{res1.id}"
                        )
                        to_delete = []
                        for a in res_n.atoms():
                            if a.name in ["OXT", "OT1", "OT2", "HXT"]:
                                to_delete.append(a)
                        n_hyds = [a for a in res1.atoms() if a.name in ["H1", "H2", "H3", "H"]]
                        if len(n_hyds) > 1:
                            sorted_hyds = sorted(n_hyds, key=lambda x: str(x.name))
                            to_delete.extend(sorted_hyds[1:])
                        if to_delete:
                            modeller.delete(to_delete)
                            logger.info(
                                f"Purged {len(to_delete)} terminal atoms for cyclic closure."
                            )
            except Exception as e:
                logger.debug(f"Cyclic welding failed: {e}")

        # EDUCATIONAL NOTE - CYX Renaming & Thiol Stripping:
        # -------------------------------------------------
        # In classical forcefields, a standard Cysteine (CYS) has a thiol group (-SH).
        # When a disulfide bond (S-S) forms, two hydrogens are LOST.
        # OpenMM's Amber forcefield uses a separate residue template ('CYX') for
        # these bonded cysteines. We must rename the residues AND manually delete
        # the HG atoms, or the physics engine will see a "template mismatch" error.
        # CYX renaming + HG deletion for bonded cysteines
        if added_bonds:
            hg_to_delete = []
            res_map = {str(r.id).strip(): r for r in modeller.topology.residues()}
            for id1, id2 in added_bonds:
                for rid in [id1, id2]:
                    residue_obj = res_map.get(rid)
                    if residue_obj and residue_obj.name == "CYS":
                        residue_obj.name = "CYX"
                        hg_to_delete.extend([a for a in residue_obj.atoms() if a.name == "HG"])
            if hg_to_delete:
                modeller.delete(hg_to_delete)

        # EDUCATIONAL NOTE - Atom Index Refresh:
        # After Modeller operations (addHydrogens, delete, addBond), all atom
        # indices may shift. We rebuild atom_list from the final topology so that
        # coordination and salt-bridge restraint index mappings remain correct.
        # Refresh atom_list after all modeller modifications
        atom_list = list(modeller.topology.atoms())

        return modeller, added_bonds, salt_bridge_restraints, coordination_restraints, atom_list

    def _build_simulation_context(
        self,
        modeller: Any,
        cyclic: bool,
        added_bonds: List,
        salt_bridge_restraints: List,
        coordination_restraints: List,
        atom_list: List[Any],
        positions: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, int, int, Any, Any]:
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
            positions: Optional positions override.

        Returns:
            ``(simulation, system, integrator, n_idx, c_idx, topology, positions)``
            where *n_idx*/*c_idx* are the N/C-terminus atom indices used for
            the cyclic pull force (``-1`` for non-cyclic structures).
        """
        topology, positions = modeller.topology, modeller.positions

        # EDUCATIONAL NOTE - System Creation & Solvent Handling:
        # ----------------------------------------------------
        # The `createSystem` method is the heaviest computation here. It maps every atom in our
        # Topology to a set of parameters (charge, radius, mass) defined in the Amber XML files.
        #
        # For implicit solvent (like OBC2), it also calculates the 'Born Radii' for every atom,
        # which determines how shielded they are from the water dielectric.

        # A forcefield (like Amber14) approximates the potential energy (U) of a
        # molecule as a sum of four main terms:
        #   U = U_bond + U_angle + U_torsion + [U_vdw + U_elec]
        # Minimization finds the coordinate set where dU/dX = 0.
        # System creation
        try:
            current_constraints = None if cyclic else app.HBonds
            if self.solvent_model == "explicit":
                logger.info(
                    f"Adding explicit solvent (TIP3P water) with a {self.box_size} nm padding..."
                )
                modeller.addSolvent(
                    self.forcefield,
                    model="tip3p",
                    padding=self.box_size,
                    ionicStrength=0.1 * unit.molar,
                )
                topology = modeller.topology
                positions = modeller.positions
                if os.getenv("SYNTH_PDB_DEBUG_SAVE_INTERMEDIATE") == "1":
                    with open("intermediate_debug.pdb", "w") as f:
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

        # EDUCATIONAL NOTE - The "Nuclear Option" & "Shadow Caps":
        # -------------------------------------------------------
        # Closing a ring is a physical paradox for most forcefields. The N and C
        # termini are parameterized as charged ions that repel each other violently.
        #
        # 1. Terminal Ghosting: We surgically disable all non-bonded interactions
        #    between the first and last residues. They can now pass through each other
        #    without steric or electrostatic resistance.
        #
        # 2. Shadow Caps: To satisfy OpenMM's template requirements, we temporarily
        #    attached ACE/NME dummy residues. Here, we zero out ALL their forces.
        #    They allow the simulation to run but contribute nothing to the energy,
        #    leaving the path clear for the "Mega-Pull" to snap the ring shut.
        # EDUCATIONAL NOTE - Constraints and Macrocycles:
        # For macrocycles, we temporarily DISABLE all constraints (like HBonds)
        # to allow the chain to bend freely into a ring during the pull phase.
        # We also use vacuum (NoCutoff) to maximize closure speed.
        # Cyclic terminal ghosting + shadow cap zeroing
        n_idx, c_idx = -1, -1
        if cyclic:
            try:
                nb_force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
                residues = list(topology.residues())
                if len(residues) >= 2:
                    res1 = residues[0]
                    res_n = residues[-1]
                    ats_first = list(res1.atoms())
                    ats_last = list(res_n.atoms())
                    logger.info(
                        f"Ghosting entire residues {res1.name}{res1.id} and "
                        f"{res_n.name}{res_n.id} for closure."
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
                            if top_atoms[p1].residue.name in ["ACE", "NME"] or top_atoms[
                                p2
                            ].residue.name in ["ACE", "NME"]:
                                force.setBondParameters(i, p1, p2, r0, 0.0)
                    elif isinstance(force, mm.HarmonicAngleForce):
                        for i in range(force.getNumAngles()):
                            p1, p2, p3, theta, k = force.getAngleParameters(i)
                            if any(
                                top_atoms[p].residue.name in ["ACE", "NME"] for p in [p1, p2, p3]
                            ):
                                force.setAngleParameters(i, p1, p2, p3, theta, 0.0)
                    elif isinstance(force, mm.PeriodicTorsionForce):
                        for i in range(force.getNumTorsions()):
                            p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                            if any(
                                top_atoms[p].residue.name in ["ACE", "NME"]
                                for p in [p1, p2, p3, p4]
                            ):
                                force.setTorsionParameters(
                                    i, p1, p2, p3, p4, periodicity, phase, 0.0
                                )

                logger.info("Excised non-bonded interactions between termini for cyclic closure.")
            except Exception as e:
                logger.warning(f"Failed to excise terminal interactions: {e}")

        # Coordination restraints
        if coordination_restraints:
            force_ext = mm.CustomBondForce("0.5*k*(r-r0)^2")
            force_ext.addGlobalParameter(
                "k", 50000.0 * unit.kilojoules_per_mole / unit.nanometer**2
            )
            force_ext.addPerBondParameter("r0")
            new_ats = list(topology.atoms())
            for i_o, l_o in coordination_restraints:
                # Ensure indices are integers
                i_o, l_o = int(i_o), int(l_o)
                oi, ol = atom_list[i_o], atom_list[l_o]
                ni, nl = -1, -1
                for a in new_ats:
                    if a.residue.id == oi.residue.id and a.name == oi.name:
                        ni = a.index
                    if a.residue.id == ol.residue.id and a.name == ol.name:
                        nl = a.index
                if ni != -1 and nl != -1:
                    force_ext.addBond(
                        ni,
                        nl,
                        [(0.23 if new_ats[nl].name == "SG" else 0.21) * unit.nanometers],
                    )
            system.addForce(force_ext)

        # EDUCATIONAL NOTE - Harmonic "Pull" Restraints & Hard Constraints:
        # -----------------------------------------------------------------
        # To bridge the gap between N and C termini, we use two levels of force:
        # 1. Harmonic Pull: A massive "spring" (100M kJ/mol/nm²) that treats the
        #    termini like two magnets. It provides a global gradient that pulls
        #    the structure toward closure.
        # 2. Hard Constraint: A specialized OpenMM constraint that FIXES the
        #    distance at exactly 1.33 Angstroms. While the pull force gets us close,
        #    the constraint ensures the final "weld" satisfies the perfect geometry
        #    required by downstream NMR tools.
        #
        # EDUCATIONAL NOTE - Why we avoid adding a hard constraint initially:
        # If the termini are far apart, a hard constraint crashes the system.
        # The 100M kJ magnet (pull_force) will get us to 1.33Å first.
        # Pull forces for cyclic closure and disulfide formation
        if cyclic or added_bonds:
            pull_force = mm.CustomBondForce("0.5*k_pull*(r-r0)^2")
            pull_force.addGlobalParameter(
                "k_pull",
                100000000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
            )
            pull_force.addPerBondParameter("r0")

            if cyclic:
                solvent_names = [
                    "HOH",
                    "WAT",
                    "SOL",
                    "TIP3",
                    "POP",
                    "NA",
                    "CL",
                    "ZN",
                    "FE",
                    "MG",
                    "CA",
                ]
                real_residues = [
                    r
                    for r in list(topology.residues())
                    if r.name.strip().upper() not in (["ACE", "NME"] + solvent_names)
                ]
                if real_residues:
                    r_first, r_last = real_residues[0], real_residues[-1]
                    logger.info(
                        f"CYCLIC: Termini identified as "
                        f"{r_first.name}{r_first.id} and {r_last.name}{r_last.id}"
                    )
                    for a in r_first.atoms():
                        if a.name == "N":
                            n_idx = a.index
                            break
                    for a in r_last.atoms():
                        if a.name == "C":
                            c_idx = a.index
                            break
                    logger.info(f"CYCLIC: Indices: N={n_idx}, C={c_idx}")

                if n_idx != -1 and c_idx != -1:
                    pull_force.addBond(n_idx, c_idx, [0.133 * unit.nanometers])
                    logger.info(f"Added massive cyclic pull force: {n_idx} -- {c_idx}")

                    # GHOSTING THE TOPOLOGICAL BOND:
                    # Since we welded the ring in the topology for templates,
                    # we must zero out its physical forces so the pull-magnet works.
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
                                if any(p == n_idx for p in [p1, p2, p3]) and any(
                                    p == c_idx for p in [p1, p2, p3]
                                ):
                                    force.setAngleParameters(i, p1, p2, p3, theta, 0.0)
                        elif isinstance(force, mm.PeriodicTorsionForce):
                            for i in range(force.getNumTorsions()):
                                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(
                                    i
                                )
                                if any(p == n_idx for p in [p1, p2, p3, p4]) and any(
                                    p == c_idx for p in [p1, p2, p3, p4]
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
                                if a.name == "SG":
                                    s1 = a.index
                                    break
                        if str(res.id).strip() == id2:
                            for a in res.atoms():
                                if a.name == "SG":
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

        # EDUCATIONAL NOTE - Simulation Setup:
        # We prefer hardware-accelerated platforms (CUDA > Metal > OpenCL) for
        # speed. If none are available, OpenMM falls back to the CPU reference
        # platform, which is correct but slower. Mixed precision is used for
        # GPU platforms to balance accuracy and throughput.
        #
        # EDUCATIONAL NOTE - Thermodynamic Ensembles & Integrators:
        # ---------------------------------------------------------
        # When we simulate a protein, we must choose which thermodynamic variables to hold constant.
        # The choice of "Ensemble" dictates how the integrator manages the system over time.
        #
        # 1. NVE (Microcanonical Ensemble):
        #    - Constant: Number of particles (N), Volume (V), Energy (E).
        #    - Integrator: Verlet Integrator.
        #    - Realism: Poor for lab conditions, but conserves energy perfectly.
        #
        # 2. NVT (Canonical Ensemble) <-- **What we use here!**:
        #    - Constant: Number of particles (N), Volume (V), Temperature (T).
        #    - Integrator: Langevin Integrator (or Andersen thermostat).
        #    - Realism: Good. The Langevin collision frequency (friction, 1.0/ps here) mimics the viscosity
        #               of water, randomly kicking atoms to maintain kinetic energy (T=300K) while
        #               dragging on them to prevent explosions.
        #
        # 3. NPT (Isothermal-Isobaric Ensemble):
        #    - Constant: Number of particles (N), Pressure (P), Temperature (T).
        #    - Integrator: Langevin Integrator + Monte Carlo Barostat.
        #    - Realism: Best for replicating a test tube on a lab bench.
        #
        # Note on Timesteps: We use a 2.0 femtosecond (0.002 ps) timestep.
        # Bonds involving hydrogen vibrate with a period of ~10 fs.
        # A 2 fs timestep is only stable because we passed `app.HBonds` to `constraints` earlier,
        # which rigidly locks all R-H bond lengths, removing the fastest vibrations from the system.
        # Build Simulation
        integrator = mm.LangevinIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds
        )
        platform = None
        props: dict = {}
        for name in ["CUDA", "Metal", "OpenCL"]:
            try:
                platform = mm.Platform.getPlatformByName(name)
                if name in ["CUDA", "OpenCL"]:
                    props = {"Precision": "mixed"}
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

        return simulation, system, integrator, n_idx, c_idx, topology, positions

    def _finalize_output(
        self,
        output_path: str,
        simulation: Any,
        cyclic: bool,
        added_bonds: List,
        coordination_restraints: List,
        hetatm_lines: List[str],
        original_metadata: Dict[Any, Any],
        atom_list: List[Any],
    ) -> Optional[bool]:
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
        pos = state.getPositions()
        final_topology = simulation.topology
        final_positions = pos

        with open(output_path, "w") as f:
            # Macrocycle Cleanup
            if cyclic:
                try:
                    logger.info("Cleaning up terminal atoms for cyclic peptide output...")
                    mod_modeller = app.Modeller(final_topology, final_positions)
                    residues = list(mod_modeller.topology.residues())
                    if residues:
                        to_prune_caps = [
                            a for r in residues if r.name in ["ACE", "NME"] for a in r.atoms()
                        ]
                        if to_prune_caps:
                            mod_modeller.delete(to_prune_caps)
                            final_topology = mod_modeller.topology
                            final_positions = mod_modeller.positions
                            residues = list(final_topology.residues())

                        solvent_names = [
                            "HOH",
                            "WAT",
                            "SOL",
                            "TIP3",
                            "POP",
                            "NA",
                            "CL",
                            "ZN",
                            "FE",
                            "MG",
                            "CA",
                        ]
                        amino_residues = [
                            r
                            for r in residues
                            if r.name.strip().upper() not in (["ACE", "NME"] + solvent_names)
                        ]

                        if amino_residues:
                            res1, res_n = amino_residues[0], amino_residues[-1]
                            to_prune = []

                            n1 = next((a for a in res1.atoms() if a.name == "N"), None)
                            if n1:
                                h_on_n1 = [
                                    a
                                    for a in res1.atoms()
                                    if a.element is not None
                                    and a.element.symbol == "H"
                                    and any(
                                        b.atom1 == n1 or b.atom2 == n1
                                        for b in final_topology.bonds()
                                        if a == b.atom1 or a == b.atom2
                                    )
                                ]
                                if len(h_on_n1) == 1:
                                    h_on_n1[0].name = "H"
                            oxt = next((a for a in res_n.atoms() if a.name == "OXT"), None)
                            if oxt:
                                to_prune.append(oxt)
                            if to_prune:
                                mod_modeller.delete(to_prune)
                                final_topology = mod_modeller.topology
                                final_positions = mod_modeller.positions
                except Exception as e:
                    logger.warning(f"Macrocycle cleanup failed: {e}")

            if len(final_positions) == 0:
                logger.error("OpenMM returned empty positions! Topology might be corrupted.")
                return False

            # EDUCATIONAL NOTE - The Importance of Metadata Restoration:
            # --------------------------------------------------------------
            # During the preprocessing steps, we violently mutated the input structure
            # to make it compatible with the Amber forcefield:
            # 1. Phosphorylation (SEP, TPO) -> Dephosphorylated L-amino acids.
            # 2. D-Amino Acids (DAL, DTR) -> L-Amino Acids (ALA, TRP).
            # 3. Metal Ions (Zn2+, Ca2+) -> Stripped and stored.
            # 4. Cyclic Peptides -> Bonded, with terminal oxygens purged.
            #
            # If we exported the file right now, the user would lose all their special
            # chemistry. This `_finalize_output` step carefully puts everything back
            # the way it was, using the `original_metadata` dictionary as a guide.
            # This ensures that down-stream analysis pipelines (like PyMOL, CYANA, or 3Dmol.js)
            # see the correct chemical identities, even though OpenMM treated them
            # as standard amino acids for the minimization.
            #
            # EDUCATIONAL NOTE - Serialization:
            # -------------------------------------------------------
            # After physics completes, we must "tidy up" our synthetic hack.
            # We prune the "Shadow Caps" (ACE/NME) and any extra terminal hydration
            # protons (H1, H2, H3, OXT) that Modeller added. We rename the remaining
            # amide proton to 'H' to satisfy canonical PDB naming. Finally, we
            # project the original residue names and IDs back onto the physics-optimized
            # coordinates, bridging the gap between molecular physics and structural
            # metadata (PTMs, D-amino acids).
            # Restore original residue names and IDs
            for res in final_topology.residues():
                res_key = (str(res.id).strip(), res.chain.id)
                if res_key in original_metadata:
                    res.name = original_metadata[res_key]["name"]
                    res.id = original_metadata[res_key]["id"]

            # EDUCATIONAL NOTE - Disulfide Mapping:
            # -------------------------------------
            # OpenMM's PDBFile writer doesn't output SSBOND records automatically.
            # We must explicitly write them to the PDB Header so that parsers
            # (and visualizers like PyMOL) know that the SG atoms are covalently linked,
            # rather than just displaying them as physically close.

            # Build PDB buffer
            # EDUCATIONAL NOTE - PDB Atom Sorting:
            # ------------------------------------
            # The Protein Data Bank (PDB) format is heavily standardized. Many parsers
            # will crash if atoms are out of order, or if CONECT records reference
            # non-existent serial numbers. We use a precise formatting string to ensure
            # the output precisely matches the PDB v3.3 spec.
            pdb_buffer = _io.StringIO()
            app.PDBFile.writeFile(final_topology, final_positions, pdb_buffer)
            pdb_lines = pdb_buffer.getvalue().split("\n")

            # EDUCATIONAL NOTE - CONECT Records & Visualization:
            # CONECT records are critical for molecular viewers (PyMOL, Chimera)
            # to draw covalent bonds that OpenMM's PDB writer may not emit
            # automatically for non-standard connections (SS bonds, metal–ligand).
            # We enumerate them from the final topology and write both directions.
            # Force CONECT for disulfides
            extra_conects = []
            for bond in final_topology.bonds():
                a1, a2 = bond.atom1, bond.atom2
                if a1.name == "SG" and a2.name == "SG":
                    extra_conects.append((a1.index + 1, a2.index + 1))
            for id1, id2 in coordination_restraints:
                extra_conects.append((id1 + 1, id2 + 1))

            final_lines = []
            for line in pdb_lines:
                if line.startswith("END") or line.startswith("CONECT"):
                    continue
                if line.strip():
                    final_lines.append(line)

            for ci1, ci2 in extra_conects:
                final_lines.append(f"CONECT{ci1:5d}{ci2:5d}")
                final_lines.append(f"CONECT{ci2:5d}{ci1:5d}")

            # Restore SSBOND records (after HEADER/TITLE)
            if added_bonds:
                insert_idx = 0
                for idx, line in enumerate(final_lines):
                    if line.startswith(("HEADER", "TITLE", "COMPND")):
                        insert_idx = idx + 1
                for s, (id1, id2) in enumerate(added_bonds, 1):
                    final_lines.insert(
                        insert_idx,
                        f"SSBOND{s:4d} CYS A {int(id1):4d}    "
                        f"CYS A {int(id2):4d}                          ",
                    )

            # EDUCATIONAL NOTE - HETATM Restoration:
            # Metal ions were stripped before addHydrogens() (they crash it).
            # Re-append them verbatim at the end of the PDB file so downstream
            # tools (viewers, NMR shift predictors) can see them correctly.
            # Restore stripped ions
            if hetatm_lines:
                for line in hetatm_lines:
                    res_name = line[17:20].strip().upper()
                    logger.debug(f"Appending restored HETATM: {res_name}")
                    final_lines.append(line.strip())

            final_lines.append("END")
            f.write("\n".join(final_lines) + "\n")
        return True

    def _run_simulation(
        self,
        input_path: str,
        output_path: str,
        max_iterations: int = 0,
        tolerance: float = 10.0,
        add_hydrogens: bool = True,
        equilibration_steps: int = 0,
        cyclic: bool = False,
        disulfides: Optional[List] = None,
        coordination: Optional[List] = None,
    ) -> Optional[float]:
        """Internal engine. Returns final_energy if successful, else None."""
        """Internal engine. Returns final_energy if successful, else None."""
        logger.info(f"Processing physics for {input_path} (cyclic={cyclic})...")

        # ── Stage 1: PDB preprocessing ──────────────────────────────────────
        try:
            (
                topology,
                positions,
                hetatm_lines,
                original_metadata,
            ) = self._preprocess_pdb_for_simulation(input_path, cyclic, disulfides)
            atom_list = list(topology.atoms())
        except Exception as e:
            logger.error(f"PDB Pre-processing failed: {e}")
            return None

        # ── Stage 2: Modeller setup (H, SSBOND, salt-bridge, cyclic weld) ──
        try:
            coordination_param = coordination if coordination is not None else []
            (
                modeller,
                added_bonds,
                salt_bridge_restraints,
                coordination_restraints,
                atom_list,
            ) = self._setup_openmm_modeller(
                topology,
                positions,
                add_hydrogens,
                cyclic,
                coordination_param,
                atom_list,
            )

            # ── Stage 3: System + forces + Simulation context ───────────────
            simulation: Any
            system: Any
            integrator: Any
            n_idx: int
            c_idx: int

            (
                simulation,
                system,
                integrator,
                n_idx,
                c_idx,
                topology,
                positions,
            ) = self._build_simulation_context(
                modeller,
                cyclic,
                added_bonds,
                salt_bridge_restraints,
                coordination_restraints,
                atom_list,
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
                    "Single-point energy calculation (max_iterations < 0). Skipping minimization."
                )
                state = simulation.context.getState(getEnergy=True)
                return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

            # ── Stage 4: Minimization / equilibration ───────────────────────
            logger.info(f"Minimizing (Tolerance={tolerance} kJ/mol, MaxIter={max_iterations})...")
            if cyclic or added_bonds or salt_bridge_restraints:
                cyc_iter = 0
                logger.info(
                    "Macrocycle/Disulfide Optimization: Running unlimited iterations for closure."
                )

                if salt_bridge_restraints:
                    sb_force = mm.CustomBondForce("0.5*k_sb*(r-r0)^2")
                    sb_force.addGlobalParameter(
                        "k_sb",
                        10000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
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
                    # EDUCATIONAL NOTE - Thermal Jiggling (Simulated Annealing):
                    # ---------------------------------------------------------
                    # Sometimes a linear sequence gets "deadlocked" in a
                    # high-energy conformation that prevents closure.
                    # We apply a brief burst of random motion (perturbation)
                    # followed by another minimization to "jiggle" the
                    # molecule into a closable state.
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
                                if hasattr(pos, "value_in_unit")
                                else np.array(pos)
                            )
                            noise = np.random.normal(0, 0.05, (len(pos_np), 3))
                            simulation.context.setPositions((pos_np + noise) * unit.nanometers)
                            simulation.minimizeEnergy(
                                maxIterations=cyc_iter,
                                tolerance=(tolerance * 0.1)
                                * unit.kilojoule
                                / (unit.mole * unit.nanometer),
                            )
                    except Exception as e:
                        logger.debug(f"Thermal jiggling failed (likely mocked): {e}")

                    logger.info("Iterative Closure: Reinforcing pull force and refining geometry.")
                    simulation.minimizeEnergy(
                        maxIterations=0,
                        tolerance=(tolerance * 0.01)
                        * unit.kilojoule
                        / (unit.mole * unit.nanometer),
                    )

                    logger.info(
                        "Final Constraint Refinement: Forcing 1.33A closure via reinitialization."
                    )
                    system.addConstraint(n_idx, c_idx, 0.133 * unit.nanometers)
                    simulation.context.reinitialize(preserveState=True)
                    simulation.minimizeEnergy(
                        maxIterations=0,
                        tolerance=(tolerance * 0.001)
                        * unit.kilojoule
                        / (unit.mole * unit.nanometer),
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
                if hasattr(final_pos, "value_in_unit"):
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

            # EDUCATIONAL NOTE - Thermal Equilibration (MD):
            # ----------------------------------------------
            # Minimization only finds a "Static Minimum" (0 Kelvin).
            # Real proteins are dynamic. Running MD steps (Langevin Dynamics)
            # resolves clashes and satisfies entropy-driven structural preferences.
            # Equilibration steps
            if equilibration_steps > 0:
                simulation.step(equilibration_steps)

            # ── Stage 5: Write output PDB ────────────────────────────────────
            write_ok = self._finalize_output(
                output_path,
                simulation,
                cyclic,
                added_bonds,
                coordination_restraints,
                hetatm_lines,
                original_metadata,
                atom_list,
            )
            if write_ok is False:
                return None

            res_energy = float(final_energy)

            # Explicitly clean up OpenMM objects to prevent memory leaks.
            # Python's GC sometimes fails to reclaim OpenMM's C++ resources
            # unless the context is explicitly destroyed.
            try:
                del simulation.context
                del simulation
                del system
                del integrator
                del modeller
            except Exception as cleanup_err:
                logger.debug(f"OpenMM cleanup failed (non-critical): {cleanup_err}")

            return res_energy

        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            return None


def simulate_trajectory(
    pdb_content: str,
    temperature_kelvin: float = 300.0,
    steps: int = 1000,
    report_interval: int = 20,
) -> List[str]:
    """Runs a short Molecular Dynamics simulation in implicit solvent and returns a list of PDB trajectory frames.

    Args:
        pdb_content: Complete string representation of the starting PDB.
        temperature_kelvin: Simulation temperature.
        steps: Total number of 2fs integration steps to run.
        report_interval: How many steps between saving a frame to the trajectory.

    Returns:
        A list of PDB formatted strings, one for each recorded frame.

    """
    import io
    import os
    import tempfile

    import openmm as mm
    from openmm import app, unit

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        f.write(pdb_content.encode("utf-8"))
        temp_pdb = f.name

    try:
        pdb = app.PDBFile(temp_pdb)
        forcefield = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
        system = forcefield.createSystem(
            pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )
        integrator = mm.LangevinMiddleIntegrator(
            temperature_kelvin * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds
        )
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()

        trajectory = []

        # Save frame 0
        state = simulation.context.getState(getPositions=True)
        out_io = io.StringIO()
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), out_io)
        trajectory.append(out_io.getvalue())

        n_frames = steps // report_interval
        for _ in range(n_frames):
            simulation.step(report_interval)
            state = simulation.context.getState(getPositions=True)
            out_io = io.StringIO()
            app.PDBFile.writeFile(simulation.topology, state.getPositions(), out_io)
            trajectory.append(out_io.getvalue())

        return trajectory
    except Exception as e:
        logger.error(f"simulate_trajectory failed: {e}")
        return []
    finally:
        if os.path.exists(temp_pdb):
            os.unlink(temp_pdb)
