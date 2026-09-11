import math

import pytest
import torch

from transformer.model.modules import PositionalEncoding

SEQ_LEN = 5
MAX_SEQ_LEN = 10
D_MODEL = 4

@pytest.fixture
def pe_layer():
    return PositionalEncoding(max_seq_len=MAX_SEQ_LEN, d_model=D_MODEL)

class TestPositionalEncodingConstruction:
    def test_odd_dmodel_raises_error(self):
        with pytest.raises(ValueError):
            PositionalEncoding(max_seq_len=MAX_SEQ_LEN, d_model=5)

    def test_compare_to_naive_pe(self, pe_layer):
        # compare output of log-based computation, against the original formula from the paper
        def reference_pe():
            pe = torch.zeros(MAX_SEQ_LEN, D_MODEL)
            for pos in range(MAX_SEQ_LEN):
                for _2i in range(0, D_MODEL, 2):
                    denom = 10000 ** (_2i / D_MODEL)
                    pe[pos, _2i] = math.sin(pos / denom)
                    pe[pos, _2i+1] = math.cos(pos / denom)
            return pe
        
        assert torch.allclose(pe_layer.pe, reference_pe())

    def test_pe_is_registered(self, pe_layer):
        assert list(pe_layer.parameters()) == []
        assert 'pe' in dict(pe_layer.named_buffers())

    def test_pos_zero(self, pe_layer):
        # for the zero-th row, i.e, when pos=0, the (pos/10000^(2i/d_model)) term becomes 0, 
        # which collapses the pe[pos] value to sin(0) = 0 for even elements, and cos(0) = 1 for odd elements
        assert torch.allclose(pe_layer.pe[0, 0::2], torch.zeros(D_MODEL//2))
        assert torch.allclose(pe_layer.pe[0, 1::2], torch.ones(D_MODEL//2))

class TestPositionalEncodingForward:
    def test_output_shape(self, pe_layer):
        assert pe_layer(seq_len=SEQ_LEN).shape == (SEQ_LEN, D_MODEL)

    def test_output_slice(self, pe_layer):
        assert torch.equal(pe_layer(seq_len=SEQ_LEN), pe_layer.pe[:SEQ_LEN])
        assert torch.equal(pe_layer(seq_len=SEQ_LEN), pe_layer(seq_len=MAX_SEQ_LEN)[:SEQ_LEN])

    @pytest.mark.parametrize('seq_len', [-1, 0, MAX_SEQ_LEN+1]) 
    def test_out_of_bounds_seq_len_raises_error(self, pe_layer, seq_len):
        with pytest.raises(ValueError):
            pe_layer(seq_len=seq_len) # check that it raises an error when given a value greater than max_seq_len