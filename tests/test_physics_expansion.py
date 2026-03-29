from unittest.mock import MagicMock, patch

import synth_pdb.physics


class TestPhysicsExpansion:
    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.os.path.exists")
    def test_calculate_energy_with_file_path(self, mock_exists):
        """Test calculate_energy using a mocked existing PDB file."""
        mock_exists.return_value = True
        minimizer = synth_pdb.physics.EnergyMinimizer()

        with patch.object(minimizer, "_run_simulation") as mock_run:
            mock_run.return_value = -123.45
            energy = minimizer.calculate_energy("dummy_input.pdb", cyclic=True)

            assert energy == -123.45
            mock_run.assert_called_once()
            called_args, called_kwargs = mock_run.call_args
            assert called_args[0] == "dummy_input.pdb"
            assert called_kwargs["max_iterations"] == -1
            assert called_kwargs["cyclic"] is True

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    def test_calculate_energy_with_string_content(self):
        """Test calculate_energy using raw PDB string content."""
        minimizer = synth_pdb.physics.EnergyMinimizer()

        mock_pdb_str = "ATOM      1  N   ALA A   1       0.000   0.000   0.000"

        with patch.object(minimizer, "_run_simulation") as mock_run:
            mock_run.return_value = 50.0
            energy = minimizer.calculate_energy(mock_pdb_str, cyclic=False)

            assert energy == 50.0
            mock_run.assert_called_once()
            called_args, called_kwargs = mock_run.call_args
            # Should be a temporary file
            assert "tmp" in called_args[0] or "temp" in called_args[0]
            assert called_kwargs["max_iterations"] == -1
            assert called_kwargs["cyclic"] is False

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    def test_calculate_energy_with_object(self):
        """Test calculate_energy using an object that has a .pdb property."""
        minimizer = synth_pdb.physics.EnergyMinimizer()

        mock_obj = MagicMock()
        mock_obj.pdb = "ATOM      1  N   ALA A   1       0.000   0.000   0.000"

        with patch.object(minimizer, "_run_simulation") as mock_run:
            mock_run.return_value = 100.0
            energy = minimizer.calculate_energy(mock_obj, cyclic=False)

            assert energy == 100.0
            mock_run.assert_called_once()

    def test_calculate_energy_no_openmm(self):
        """calculate_energy should return 0.0 if OpenMM is missing."""
        with patch("synth_pdb.physics.HAS_OPENMM", False):
            minimizer = synth_pdb.physics.EnergyMinimizer()
            assert minimizer.calculate_energy("test.pdb") == 0.0

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_create_system_robust_suppress_implicit(self, mock_app):
        """Test _create_system_robust retry logic when implicitSolvent fails."""
        minimizer = synth_pdb.physics.EnergyMinimizer()
        minimizer.forcefield = MagicMock()

        # Raise exception on first call (with implicitSolvent), succeed on second
        def create_system_side_effect(topo, **kwargs):
            if "implicitSolvent" in kwargs:
                raise Exception("implicitSolvent was specified to createSystem() but was never used")
            return "mock_system"

        minimizer.forcefield.createSystem.side_effect = create_system_side_effect

        minimizer.implicit_solvent_enum = "mock_enum"

        mock_modeller = MagicMock()
        mock_modeller.positions = [0, 1, 2]

        system, topo, pos = minimizer._create_system_robust("mock_topo", None, mock_modeller)

        assert system == "mock_system"
        assert topo == "mock_topo"
        assert pos == [0, 1, 2]
        assert "implicitSolvent" in minimizer._suppressed_args

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_create_system_robust_template_mismatch(self, mock_app):
        """Test _create_system_robust fallback for 'No template found'."""
        minimizer = synth_pdb.physics.EnergyMinimizer()
        minimizer.forcefield = MagicMock()
        minimizer.implicit_solvent_enum = None

        call_count = {"count": 0}

        def create_system_side_effect(topo, **kwargs):
            if call_count["count"] == 0:
                call_count["count"] += 1
                raise Exception("No template found for residue")
            return "mock_system_repaired"

        minimizer.forcefield.createSystem.side_effect = create_system_side_effect

        mock_modeller = MagicMock()
        mock_modeller.positions = [0, 1, 2]

        mock_atom_h = MagicMock()
        mock_atom_h.element.symbol = "H"
        mock_atom_c = MagicMock()
        mock_atom_c.element.symbol = "C"
        mock_modeller.topology.atoms.return_value = [mock_atom_h, mock_atom_c]

        mock_repaired_topo = MagicMock()

        def side_effect_addHydrogens(ff):
            mock_modeller.topology = mock_repaired_topo

        mock_modeller.addHydrogens.side_effect = side_effect_addHydrogens

        system, topo, pos = minimizer._create_system_robust("mock_topo", None, mock_modeller)

        assert system == "mock_system_repaired"
        # modeller.delete should have been called with the [mock_atom_h]
        mock_modeller.delete.assert_called_once_with([mock_atom_h])
        mock_modeller.addHydrogens.assert_called_once_with(minimizer.forcefield)

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_create_system_robust_final_fallback(self, mock_app):
        """Test _create_system_robust total crash fallback."""
        minimizer = synth_pdb.physics.EnergyMinimizer()
        minimizer.forcefield = MagicMock()
        minimizer.implicit_solvent_enum = None

        # Always throw an exception that doesn't match specific fallbacks
        minimizer.forcefield.createSystem.side_effect = [
            Exception("Severe segmentation fault simulation explosion"),
            "fallback_system"
        ]

        mock_modeller = MagicMock()

        system, topo, pos = minimizer._create_system_robust("mock_topo", None, mock_modeller)

        assert system == "fallback_system"
        # Check that it called once for the test, and once for the fallback
        assert minimizer.forcefield.createSystem.call_count == 2
        # The fallback call should have nonbondedMethod=app.NoCutoff
        args, kwargs = minimizer.forcefield.createSystem.call_args_list[1]
        assert kwargs.get("nonbondedMethod") == mock_app.NoCutoff
        assert kwargs.get("constraints") is None

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    @patch("synth_pdb.physics.mm")
    def test_coordination_restraints_application(self, mock_mm, mock_app):
        """Test the application of CustomBondForce for coordination restraints."""
        minimizer = synth_pdb.physics.EnergyMinimizer()

        # Mock Modeller and System setup
        mock_modeller = MagicMock()

        # Atom list setup
        mock_atom_i = MagicMock()
        mock_atom_i.name = "ZN"
        mock_atom_i.residue.id = "100"
        mock_atom_i.index = 0

        mock_atom_l = MagicMock()
        mock_atom_l.name = "SG"
        mock_atom_l.residue.id = "50"
        mock_atom_l.index = 1

        mock_modeller.topology.atoms.return_value = iter([mock_atom_i, mock_atom_l])
        atom_list = [mock_atom_i, mock_atom_l]

        coordination_restraints = [(0, 1)]  # Indices in atom_list

        # Build simulation context
        minimizer.forcefield = MagicMock()
        mock_system = MagicMock()
        mock_app.Simulation.return_value = MagicMock()

        minimizer._create_system_robust = MagicMock(return_value=(mock_system, mock_modeller.topology, mock_modeller.positions))

        simulation, system, n_idx, c_idx, topo, pos = minimizer._build_simulation_context(
            mock_modeller, False, [], [], coordination_restraints, atom_list
        )

        # Assert CustomBondForce was created and added
        mock_mm.CustomBondForce.assert_any_call("0.5*k*(r-r0)^2")
        assert mock_system.addForce.called

    @patch("synth_pdb.physics.HAS_OPENMM", True)
    @patch("synth_pdb.physics.app")
    def test_finalize_output_cyclic_cleanup(self, mock_app):
        """Test the post-simulation cleanup of cyclic termini in _finalize_output."""
        minimizer = synth_pdb.physics.EnergyMinimizer()

        mock_simulation = MagicMock()
        mock_state = MagicMock()
        mock_simulation.context.getState.return_value = mock_state
        mock_state.getPositions.return_value = [0.0]

        # We need to simulate the Modeller deleting term atoms
        mock_modeller = MagicMock()
        mock_app.Modeller.return_value = mock_modeller
        mock_modeller.positions = [0.0]

        mock_res_nme = MagicMock()
        mock_res_nme.name = "NME"
        mock_atom_nme = MagicMock()
        mock_res_nme.atoms.return_value = [mock_atom_nme]

        mock_res_ala = MagicMock()
        mock_res_ala.name = "ALA"
        mock_atom_n = MagicMock()
        mock_atom_n.name = "N"

        # We need an H explicitly connected to N
        mock_atom_h = MagicMock()
        mock_atom_h.element.symbol = "H"
        mock_atom_h.name = "H1"

        mock_res_ala.atoms.return_value = [mock_atom_n, mock_atom_h]

        # We need a bond
        mock_bond = MagicMock()
        mock_bond.atom1 = mock_atom_n
        mock_bond.atom2 = mock_atom_h

        # Last residue
        mock_res_tyr = MagicMock()
        mock_res_tyr.name = "TYR"
        mock_atom_oxt = MagicMock()
        mock_atom_oxt.name = "OXT"
        mock_res_tyr.atoms.return_value = [mock_atom_oxt]

        mock_modeller.topology.residues.return_value = [mock_res_nme, mock_res_ala, mock_res_tyr]
        mock_modeller.topology.bonds.return_value = [mock_bond]

        # mock internal file io
        with patch("builtins.open", MagicMock()):
            with patch("synth_pdb.physics.app.PDBFile.writeFile"):
                minimizer._finalize_output(
                    "dummy.pdb",
                    mock_simulation,
                    cyclic=True,
                    added_bonds=[],
                    coordination_restraints=[],
                    hetatm_lines=[],
                    original_metadata={},
                    atom_list=[]
                )

        # Check pruning
        mock_modeller.delete.assert_any_call([mock_atom_nme])
        assert mock_atom_h.name == "H"
        mock_modeller.delete.assert_any_call([mock_atom_oxt])
