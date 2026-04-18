from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Helper to create the mocks
def get_mocks():
    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    # Ensure torch.device returns the input so it's a simple string-like mock
    mock_torch.device.side_effect = lambda x: x
    return mock_torch, mock_transformers

@pytest.fixture
def mocked_plm_deps():
    mock_torch, mock_transformers = get_mocks()
    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
        yield mock_torch, mock_transformers

# We can import it at top level because it doesn't have top-level torch/transformers imports
from synth_pdb.plm import ESM2Embedder


def test_esm2_embedder_lazy_loading(mocked_plm_deps):
    """Verify that model and tokenizer are only loaded when embed() is called."""
    embedder = ESM2Embedder()
    assert embedder._model is None
    assert embedder._tokenizer is None

    # Mock the internal _load_model to avoid actually using the mocked transformers in this test
    with patch.object(embedder, "_load_model") as mock_load:
        with patch.object(embedder, "_run_model", return_value=np.zeros((5, 320))):
            embedder.embed("ACDEF")
            mock_load.assert_called_once()

def test_esm2_embedder_embedding_dim_property(mocked_plm_deps):
    """Verify that embedding_dim returns known defaults or triggers load."""
    embedder = ESM2Embedder(model_name="facebook/esm2_t6_8M_UR50D")
    assert embedder.embedding_dim == 320

    embedder_large = ESM2Embedder(model_name="facebook/esm2_t12_35M_UR50D")
    assert embedder_large.embedding_dim == 480

def test_mean_embed(mocked_plm_deps):
    """Verify mean pooling logic (L, D) -> (D,)."""
    embedder = ESM2Embedder()
    # Mock embed to return a known matrix
    mock_emb = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=np.float32)

    with patch.object(embedder, "embed", return_value=mock_emb):
        mean = embedder.mean_embed("AA")
        # (1+3)/2 = 2.0, (2+4)/2 = 3.0
        np.testing.assert_array_almost_equal(mean, [2.0, 3.0])
        assert mean.shape == (2,)

def test_sequence_similarity(mocked_plm_deps):
    """Verify cosine similarity calculation."""
    embedder = ESM2Embedder()

    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)

    with patch.object(embedder, "mean_embed", side_effect=[v1, v2]):
        sim = embedder.sequence_similarity("A", "G")
        assert sim == 0.0

    v3 = np.array([1.0, 1.0], dtype=np.float32)
    with patch.object(embedder, "mean_embed", side_effect=[v3, v3]):
        sim = embedder.sequence_similarity("A", "A")
        assert pytest.approx(sim) == 1.0

def test_embed_structure_integration(mocked_plm_deps):
    """Verify that embed_structure correctly extracts sequence from Biotite AtomArray."""
    import biotite.structure as struc

    atom1 = struc.Atom([0,0,0], res_id=1, res_name="ALA", atom_name="CA", element="C")
    atom2 = struc.Atom([0,0,0], res_id=2, res_name="GLY", atom_name="CA", element="C")
    structure = struc.array([atom1, atom2])

    embedder = ESM2Embedder()
    with patch.object(embedder, "embed", return_value=np.zeros((2, 320))) as mock_embed:
        embedder.embed_structure(structure)
        mock_embed.assert_called_once_with("AG")

def test_run_model_internal_logic(mocked_plm_deps):
    """Verify the internal _run_model correctly interacts with mocked torch/transformers."""
    embedder = ESM2Embedder()

    # Setup mocks for the internal pipeline
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    embedder._model = mock_model
    embedder._tokenizer = mock_tokenizer
    # Mock _device to avoid AttributeError
    embedder._device = "cpu"

    # Mock tokenizer output: it returns a dict-like object with .to() on its values
    mock_tensor = MagicMock()
    # .to() returns another mock (the "moved" tensor)
    mock_moved_tensor = MagicMock()
    mock_tensor.to.return_value = mock_moved_tensor

    mock_inputs = {"input_ids": mock_tensor}
    mock_tokenizer.return_value = mock_inputs

    # Mock model output (last_hidden_state)
    # Shape should be (1, L+2, D)
    L = 3
    D = 320
    mock_output = MagicMock()

    # Create a fake tensor that behaves like a numpy array when sliced
    fake_tensor_data = np.random.rand(1, L + 2, D).astype(np.float32)

    # Mock the slicing logic: outputs.last_hidden_state[:, 1:-1, :]
    mock_hidden_state = MagicMock()
    # The __getitem__ should return a mock that has .squeeze(0) and then .cpu().numpy()
    mock_sliced_hidden = MagicMock()
    mock_hidden_state.__getitem__.return_value = mock_sliced_hidden

    mock_squeezed = MagicMock()
    mock_sliced_hidden.squeeze.return_value = mock_squeezed

    # .cpu().numpy()
    mock_cpu = MagicMock()
    mock_squeezed.cpu.return_value = mock_cpu
    mock_cpu.numpy.return_value = fake_tensor_data[0, 1:-1, :]

    mock_output.last_hidden_state = mock_hidden_state
    mock_model.return_value = mock_output

    result = embedder._run_model("ABC")

    assert result.shape == (L, D)
    np.testing.assert_array_equal(result, fake_tensor_data[0, 1:-1, :])

    # Verify the chain of calls
    mock_tokenizer.assert_called_once_with("ABC", return_tensors="pt", add_special_tokens=True)
    mock_tensor.to.assert_called_with("cpu")
    mock_model.assert_called_once()
