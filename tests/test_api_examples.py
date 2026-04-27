import io
import numpy as np
import biotite.structure.io.pdb as pdb_io
from synth_pdb.generator import PeptideGenerator
from synth_pdb.chemical_shifts import predict_chemical_shifts
from synth_pdb.saxs import calculate_saxs_profile
from synth_pdb.cryo_em import generate_density_map


class TestAPIExamples:
    """Test suite for Python API examples documented in Science Guides."""

    def test_multimodal_api_example(self) -> None:
        """Verify the integrated multimodal Python API workflow."""
        # 1. Structure Generation
        gen = PeptideGenerator("MQIFVKTLTGK")
        # generate_ensemble returns an AtomArrayStack by default (as_stack=True)
        ensemble = gen.generate_ensemble(n_models=5)
        assert len(ensemble) == 5

        # 2. Local Observables (NMR)
        # predict_chemical_shifts expects an AtomArray
        shifts = predict_chemical_shifts(ensemble[0])
        assert "CA" in shifts["A"][1]

        # 3. Global Shape (SAXS)
        q, intensity = calculate_saxs_profile(ensemble[0])
        assert len(q) == len(intensity)
        assert intensity[0] > 0

        # 4. 3D Volume (Cryo-EM)
        density, origin = generate_density_map(ensemble, resolution=4.0)
        assert density.ndim == 3
        assert len(origin) == 3
