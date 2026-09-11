import math

import pytest
import torch

from tests.model._dims import BATCH, D_MODEL, SEQ_LEN, VOCAB_SIZE
from transformer.model.modules import Embedding


@pytest.fixture
def x():
    return torch.randint(size=(BATCH, SEQ_LEN), low=0, high=VOCAB_SIZE)


@pytest.fixture
def x_with_known_ids():
    return torch.LongTensor([0, 1, 1])


@pytest.fixture
def emb_layer():
    return Embedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL)


class TestEmbeddingConstruction:
    def test_weight_shape(self, emb_layer):
        assert emb_layer.emb.weight.shape == (VOCAB_SIZE, D_MODEL)


class TestScaling:
    def test_scaling_factor_value(self, emb_layer):
        assert emb_layer.scaling_factor == math.sqrt(D_MODEL)

    def test_scaling_applied_to_embedding(self, emb_layer, x):
        out = emb_layer(x)
        expected_out = emb_layer.emb(x) * math.sqrt(D_MODEL)
        assert torch.allclose(out, expected_out)


class TestEmbeddingForward:
    @pytest.mark.parametrize(
        'ids,expected_shape',
        [
            ([1, 2, 3], [3, D_MODEL]),  # unbatched, sequence_length=3
            ([[1, 2, 3], [4, 5, 6]], [2, 3, D_MODEL]),  # batch=2, sequence_length=3
        ],
    )
    def test_output_shape(self, emb_layer, ids, expected_shape):
        assert emb_layer(torch.LongTensor(ids)).shape == torch.Size(expected_shape)


class TestPropertiesPreservedPostScaling:
    """nn.Embedding guarantees these; assert our sqrt(d_model) scaling doesn't break them."""

    def test_output_is_float(self, emb_layer, x):
        assert emb_layer(x).dtype == torch.float32

    def test_different_ids_give_different_vectors(self, emb_layer, x_with_known_ids):
        out = emb_layer(x_with_known_ids)
        assert not torch.allclose(out[0], out[1])

    def test_same_ids_give_same_vectors(self, emb_layer, x_with_known_ids):
        out = emb_layer(x_with_known_ids)
        assert torch.allclose(out[1], out[2])
