import math
import pytest
import torch
from transformer.model.modules import Embedding

VOCAB_SIZE = 10
D_MODEL = 4
INP_X = torch.LongTensor([[1, 2, 3], [4, 5, 6]])

@pytest.fixture
def emb_layer():
    return Embedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL)

class TestEmbeddingConstruction:
    def test_weight_shape(self, emb_layer):
        assert emb_layer.emb.weight.shape == (VOCAB_SIZE, D_MODEL)

class TestScaling:
    def test_scaling_factor_value(self, emb_layer):
        assert emb_layer.scaling_factor == math.sqrt(D_MODEL)

    def test_scaling_applied_to_embedding(self, emb_layer):
        out = emb_layer(INP_X)
        expected_out = emb_layer.emb(INP_X) * math.sqrt(D_MODEL)
        assert torch.allclose(out, expected_out)

class TestEmbeddingForward:
    @pytest.mark.parametrize('x,expected_shape', [
        ([1, 2, 3], [3, D_MODEL]), # unbatched, sequence_length=3
        ([[1, 2, 3], [4, 5, 6]], [2, 3, D_MODEL]) # batch=2, sequence_length=3
    ])
    def test_output_shape(self, emb_layer, x, expected_shape):
        assert emb_layer(torch.LongTensor(x)).shape == torch.Size(expected_shape)

class TestPropertiesPreservedPostScaling:
    """nn.Embedding guarantees these; assert our sqrt(d_model) scaling doesn't break them."""
    def test_output_is_float(self, emb_layer):
        assert emb_layer(INP_X).dtype == torch.float32

    def test_different_ids_give_different_vectors(self, emb_layer):
        out = emb_layer(torch.LongTensor([0, 1, 1]))
        assert not torch.allclose(out[0], out[1])

    def test_same_ids_give_same_vectors(self, emb_layer):
        out = emb_layer(torch.LongTensor([0, 1, 1]))
        assert torch.allclose(out[1], out[2])