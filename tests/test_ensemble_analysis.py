from synth_pdb.batch_generator import BatchedGenerator


def test_ensemble_analysis_medoid():
    """Test that analyze_ensemble correctly identifies a medoid structure."""
    bg = BatchedGenerator("AAAAA", n_batch=5)
    # Generate structures with some drift to ensure they are unique
    batch = bg.generate_batch(seed=42, drift=2.0)

    analysis = batch.analyze_ensemble(superimpose=True)

    assert "avg_rmsd" in analysis
    assert "medoid_index" in analysis
    assert "avg_coords" in analysis

    assert 0 <= analysis["medoid_index"] < 5
    assert analysis["avg_coords"].shape == (batch.n_atoms, 3)
    assert analysis["avg_rmsd"] >= 0.0


def test_ensemble_analysis_identical():
    """Test that identical structures yield 0 RMSD."""
    bg = BatchedGenerator("AAAAA", n_batch=2)
    # Generate identical structures by using drift=0
    batch = bg.generate_batch(seed=42, drift=0.0)

    analysis = batch.analyze_ensemble(superimpose=True)

    # Due to floating point precision and superimposition, it might not be EXACTLY 0,
    # but it should be very close.
    assert analysis["avg_rmsd"] < 1e-5


def test_ensemble_analysis_different_drift():
    """Test that higher drift leads to higher average RMSD."""
    bg1 = BatchedGenerator("AAAAA", n_batch=10)
    batch1 = bg1.generate_batch(seed=42, drift=1.0)
    analysis1 = batch1.analyze_ensemble(superimpose=True)

    bg2 = BatchedGenerator("AAAAA", n_batch=10)
    batch2 = bg2.generate_batch(seed=42, drift=5.0)
    analysis2 = batch2.analyze_ensemble(superimpose=True)

    assert analysis2["avg_rmsd"] > analysis1["avg_rmsd"]
