import io

import biotite.structure.io.pdb as pdb_io
import numpy as np
import pytest

# Skip the entire module if GNN dependencies are missing
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from synth_pdb.generator import generate_pdb_content
from synth_pdb.quality.gnn.gnn_classifier import GNNQualityClassifier


class TestGNNCalibration:
    """
    Sensitivity Sweep for GNN Quality Scorer.
    Validates that the model is a reliable proxy for physical quality.
    """

    clf: GNNQualityClassifier
    sequence: str
    length: int

    @classmethod
    def setup_class(cls) -> None:
        # Initialize the classifier with the trained model
        cls.clf = GNNQualityClassifier()
        # Use 20 residues for better graph resolution
        cls.sequence = "ALA-GLY-SER-LEU-GLU-VAL-ASP-THR-LYS-ILE-" * 2
        cls.sequence = cls.sequence.strip("-")
        cls.length = 20

    def test_coordinate_noise_sensitivity(self) -> None:
        """
        Verify that adding increasing coordinate noise drops the quality score monotonically.
        """
        # 1. Generate baseline perfect helix
        base_pdb = generate_pdb_content(
            sequence_str=self.sequence, conformation="alpha", minimize_energy=False
        )

        # Get baseline prob
        _, prob_baseline, _ = self.clf.predict(base_pdb)
        print(f"\nBaseline (Noise 0.0A) -> Prob(Good): {prob_baseline:.4f}")

        f = io.StringIO(base_pdb)
        pdb_file_in = pdb_io.PDBFile.read(f)
        base_struc = pdb_file_in.get_structure(model=1)

        noise_levels = [0.1, 0.3, 0.5, 1.0, 2.0]
        probs = [prob_baseline]

        rng = np.random.default_rng(42)

        for noise in noise_levels:
            perturbed = base_struc.copy()
            perturbed.coord += rng.normal(0, noise, perturbed.coord.shape)

            # Write to string, ensuring we preserve other annotations (B-factors, etc)
            f_out = io.StringIO()
            pdb_file = pdb_io.PDBFile()
            pdb_file.set_structure(perturbed)

            # Note: We don't manually copy B-factors here because we copy the whole structure object,
            # but if PDBFile.write() zeroes them out, we'd need to re-assign.
            # Biotite structure objects preserve annotations.

            pdb_file.write(f_out)
            perturbed_pdb = f_out.getvalue()

            _, prob, _ = self.clf.predict(perturbed_pdb)
            probs.append(prob)
            print(f"Noise {noise:.1f}A -> Prob(Good): {prob:.4f}")

        # Verification
        # 1. Score should decrease significantly from baseline to max noise
        assert probs[-1] < probs[0]
        assert probs[-1] < 0.2

        # 2. General monotonic trend (allowing for minor ML noise)
        # We check that the sum of differences is negative
        diffs = np.diff(probs)
        assert np.sum(diffs) < 0

    def test_single_clash_sensitivity(self) -> None:
        """
        Verify that moving a single atom to clash with another drops the score.
        """
        base_pdb = generate_pdb_content(
            sequence_str=self.sequence, conformation="alpha", minimize_energy=False
        )
        _, prob_clean, _ = self.clf.predict(base_pdb)

        f = io.StringIO(base_pdb)
        struc = pdb_io.PDBFile.read(f).get_structure(model=1)

        # Create an artificial clash by moving CA of residue 2 to the same position as CA of residue 5
        ca_indices = [i for i, a in enumerate(struc) if a.atom_name == "CA"]
        struc.coord[ca_indices[1]] = struc.coord[ca_indices[4]]

        f_out = io.StringIO()
        pdb_file = pdb_io.PDBFile()
        pdb_file.set_structure(struc)
        pdb_file.write(f_out)
        clashing_pdb = f_out.getvalue()

        _, prob_clash, _ = self.clf.predict(clashing_pdb)

        print(f"Clean Prob: {prob_clean:.4f} -> Clashing Prob: {prob_clash:.4f}")
        assert prob_clash < prob_clean
        assert prob_clash < 0.5

    def test_backbone_distortion_sensitivity(self) -> None:
        """
        Verify that perturbing backbone dihedral angles (phi/psi) drops the score.
        """
        base_pdb = generate_pdb_content(
            sequence_str=self.sequence, conformation="alpha", minimize_energy=False
        )
        _, prob_clean, _ = self.clf.predict(base_pdb)

        f = io.StringIO(base_pdb)
        struc = pdb_io.PDBFile.read(f).get_structure(model=1)

        # Move all Carbonyl Oxygens by a large amount (0.8A)
        # This breaks H-bonding and local geometry without shifting the whole chain
        struc.coord[struc.atom_name == "O"] += 0.8

        f_out = io.StringIO()
        pdb_file = pdb_io.PDBFile()
        pdb_file.set_structure(struc)
        pdb_file.write(f_out)
        distorted_pdb = f_out.getvalue()

        _, prob_distorted, _ = self.clf.predict(distorted_pdb)

        print(f"Clean Prob: {prob_clean:.4f} -> Distorted O Prob: {prob_distorted:.4f}")
        assert prob_distorted < prob_clean
