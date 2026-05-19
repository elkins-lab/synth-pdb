import os
import tempfile

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
import pytest

from synth_pdb.generator import generate_pdb_content
from synth_pdb.physics import HAS_OPENMM, EnergyMinimizer


@pytest.mark.skipif(
    not HAS_OPENMM, reason="Solvent consistency tests require OpenMM physics engine"
)
class TestSolventConsistency:
    """
    Test suite to verify that different solvent models produce
    consistent and physically plausible results.
    """

    def test_solvent_energy_and_rmsd_stability(self) -> None:
        """
        Compare TRP-CAGE minimization across OBC2, GBn2, and Explicit solvent.
        Verify that none of the models 'explode' and energies are in expected ranges.
        """
        sequence = "NLYIQWLKDGGPSSGRPPPS"  # TRP-CAGE
        seed = 42

        results = {}
        models = ["obc2", "gbn2", "explicit"]

        # Pre-generate coordinates to ensure starting from same point
        # We use minimize_energy=False first to get the raw geometric build
        raw_pdb = generate_pdb_content(sequence_str=sequence, seed=seed, minimize_energy=False)

        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp_in:
            tmp_in.write(raw_pdb)
            tmp_in_path = tmp_in.name

        try:
            for model in models:
                with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp_out:
                    tmp_out_path = tmp_out.name

                # Use EnergyMinimizer directly to ensure we get the energy return
                minimizer = EnergyMinimizer(solvent_model=model)

                # Explicit solvent needs more iterations to reach a stable state
                # because of the large number of degrees of freedom in the water box.
                iters = 1000 if model == "explicit" else 500

                # _run_simulation returns the energy
                energy = minimizer._run_simulation(
                    tmp_in_path,
                    tmp_out_path,
                    add_hydrogens=True,
                    max_iterations=iters,
                    tolerance=10.0,
                )

                # Load result to check RMSD
                f = pdb.PDBFile.read(tmp_out_path)
                structure = f.get_structure(model=1)

                # Filter to protein only (exclude water/ions) for RMSD
                # This ensures consistent atom count even if explicit solvent adds molecules
                protein_mask = ~structure.hetero
                structure = structure[protein_mask]

                results[model] = {"energy": energy, "structure": structure}
                os.remove(tmp_out_path)

            # --- Verification ---

            # 1. All models should return a finite energy
            for model in models:
                e = results[model]["energy"]
                assert e is not None
                assert np.isfinite(e)

                if model == "explicit":
                    # SCIENTIFIC NOTE: Explicit solvent potential energy depends on the
                    # number of water molecules and the specific platform/version of OpenMM.
                    # While it is typically very negative, certain conditions or incomplete
                    # minimization (1000 steps is still few) might result in positive values.
                    # We just ensure it hasn't "exploded" (unphysically high energy).
                    assert e < 1e6, f"Explicit solvent model produced unphysically high energy: {e}"
                else:
                    # Implicit models should consistently produce negative energy for stable proteins
                    assert e < 0, f"Implicit model {model} produced positive energy: {e}"

            # 2. RMSD between models should be reasonable (< 2.0A)
            # They should all find a similar local minimum for TRP-CAGE
            ref_struc = results["obc2"]["structure"]
            for model in ["gbn2", "explicit"]:
                comp_struc = results[model]["structure"]
                # Align before RMSD
                _, transformation = struc.superimpose(ref_struc, comp_struc)
                aligned_comp = transformation.apply(comp_struc)
                rmsd = struc.rmsd(ref_struc, aligned_comp)

                assert rmsd < 2.0, f"RMSD between obc2 and {model} too high: {rmsd:.3f} A"

            # 3. Explicit solvent usually produces lower energy due to many more interactions
            # but since we calculate potential energy of the WHOLE system,
            # explicit will be massive (-10^5 to -10^6).
            # Implicit models should be comparable.
            assert abs(results["obc2"]["energy"] - results["gbn2"]["energy"]) < 2000.0

        finally:
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)

    def test_solvent_validation_logic(self) -> None:
        """Test the normalization and validation of solvent names in EnergyMinimizer."""
        # Test string normalization
        m1 = EnergyMinimizer(solvent_model="OBC2")
        assert m1.solvent_model == "app.OBC2"

        m2 = EnergyMinimizer(solvent_model="gbn2")
        assert m2.solvent_model == "app.GBn2"

        # Test unknown fallback
        m3 = EnergyMinimizer(solvent_model="junk_model")
        assert m3.solvent_model == "explicit"

    def test_explicit_solvent_padding(self) -> None:
        """Verify that explicit solvent box size is handled correctly."""
        # Box size too small warning check (should increase to 1.1)
        m = EnergyMinimizer(solvent_model="explicit", box_size=0.5)
        assert m.box_size.value_in_unit(EnergyMinimizer().box_size.unit) == 1.1
