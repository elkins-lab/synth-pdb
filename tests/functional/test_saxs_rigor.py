import numpy as np
import biotite.structure as struc
import pytest
import os
from synth_pdb.saxs import get_form_factor, calculate_saxs_profile, calculate_radius_of_gyration
from synth_pdb.visualization_saxs import plot_saxs_results


class TestSAXSRigor:
    """Scientific rigor tests for SAXS simulation based on peer-reviewed standards."""

    def test_atomic_form_factors_at_zero_q(self) -> None:
        """Verify form factors converge to atomic number Z at q=0 (Waasmaier & Kirfel 1995)."""
        q_zero = np.array([0.0])

        # Peer-reviewed Z values
        benchmarks = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}

        print("\n[SAXS Rigor] Verifying Form Factors at q=0:")
        for elem, z_expected in benchmarks.items():
            f_0 = get_form_factor(elem, q_zero)[0]
            print(f"  - {elem}: Calculated={f_0:.4f}, Expected={z_expected}")
            # Tolerance 0.1 to account for Gaussian approximation residuals
            assert np.abs(f_0 - z_expected) < 0.1

    def test_ubiquitin_rg_internal_consistency(self) -> None:
        """Verify that Rg from scattering curve matches Rg from coordinates (Internal Consistency)."""
        pdb_path = "examples/1UBQ.pdb"
        if not os.path.exists(pdb_path):
            pytest.skip("1UBQ.pdb example not found.")

        import biotite.structure.io.pdb as pdb_io

        pdb_file = pdb_io.PDBFile.read(pdb_path)
        structure = pdb_file.get_structure(model=1)
        # Select monomeric Chain A
        structure = structure[(structure.chain_id == "A") & (~structure.hetero)]

        # 1. Direct Rg from coordinates
        rg_coord = calculate_radius_of_gyration(structure)

        # 2. Scattering Rg from simulated curve (Guinier)
        # Low q range is critical for Guinier: q*Rg < 1.3
        q_max_guinier = 1.3 / rg_coord
        q, intensity = calculate_saxs_profile(
            structure, q_min=0.0, q_max=q_max_guinier, n_points=50, include_solvent=False
        )

        # Perform Guinier fit: ln I vs q^2
        q2 = q**2
        ln_i = np.log(intensity)
        slope, _ = np.polyfit(q2, ln_i, 1)
        rg_estimated = np.sqrt(-3 * slope)

        print("\n[SAXS Rigor] Ubiquitin Consistency Check:")
        print(f"  - Coordinate Rg:  {rg_coord:.3f} A")
        print(f"  - Scattering Rg:  {rg_estimated:.3f} A")

        # Rigorous match within 2%
        assert np.abs(rg_estimated - rg_coord) / rg_coord < 0.02

    def test_kratky_folding_signature(self) -> None:
        """Verify Kratky plot distinguishes between folded and disordered states (Kratky 1949)."""
        from synth_pdb.generator import generate_pdb_content
        import io
        import biotite.structure.io.pdb as pdb_io

        # 1. Folded State: Perfect Alpha Helix (50 res)
        # Use fixed seed for reproducibility
        pdb_alpha = generate_pdb_content(
            length=50, conformation="alpha", minimize_energy=False, seed=42
        )
        struct_alpha = pdb_io.PDBFile.read(io.StringIO(pdb_alpha)).get_structure(model=1)

        # 2. Disordered State: High-Drift Ensemble
        # Use fixed seed for reproducibility
        pdb_random = generate_pdb_content(
            length=50, conformation="random", drift=90.0, minimize_energy=False, seed=42
        )
        struct_random = pdb_io.PDBFile.read(io.StringIO(pdb_random)).get_structure(model=1)

        # We use a wider q-range to see the intensity divergence at high q
        q = np.linspace(0.01, 1.0, 50)
        _, i_alpha = calculate_saxs_profile(
            struct_alpha, q_min=0.01, q_max=1.0, n_points=50, include_solvent=False
        )
        _, i_random = calculate_saxs_profile(
            struct_random, q_min=0.01, q_max=1.0, n_points=50, include_solvent=False
        )

        # Compute Kratky values: q^2 * I(q)
        k_alpha = (q**2) * i_alpha
        k_random = (q**2) * i_random

        # 1. Folded Signature: Should peak and then decay significantly
        peak_idx = np.argmax(k_alpha)
        assert k_alpha[-1] < 0.5 * k_alpha[peak_idx], "Folded helix should decay at high q"

        # 2. Disordered Signature: Should stay high relative to folded at high q
        # Normalize by I(0) for a fair comparison of relative shapes
        ratio_alpha = k_alpha[-1] / i_alpha[0]
        ratio_random = k_random[-1] / i_random[0]

        print("\n[SAXS Rigor] Kratky Signature (q=1.0):")
        print(f"  - Folded Helix (Normalized high-q):    {ratio_alpha:.6f}")
        print(f"  - Disordered Coil (Normalized high-q): {ratio_random:.6f}")

        assert (
            ratio_random > ratio_alpha
        ), "Disordered state should have higher relative high-q intensity"
