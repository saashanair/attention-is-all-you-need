from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.model._dims import BATCH, D_FF, D_KQ, D_MODEL, D_V, HEADS, MAX_SEQ_LEN, N_ENC, P_DROP, SL_SRC, VOCAB_SIZE_SRC
from tests.model._mask_helpers import get_padding_mask
from transformer.model.encoder import Encoder, EncoderLayer
from transformer.model.modules import (
    AddAndNorm,
    Embedding,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)


@pytest.fixture
def x_enclayer():
    return torch.randn((BATCH, SL_SRC, D_MODEL))


@pytest.fixture
def enc_layer():
    return EncoderLayer(d_model=D_MODEL, d_kq=D_KQ, d_v=D_V, h=HEADS, d_ff=D_FF, p_drop=P_DROP)


@pytest.fixture
def x_enc():
    return torch.randint(size=(BATCH, SL_SRC), low=0, high=VOCAB_SIZE_SRC)


@pytest.fixture
def enc():
    return Encoder(
        n_enc=N_ENC,
        vocab_size=VOCAB_SIZE_SRC,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        d_kq=D_KQ,
        d_v=D_V,
        h=HEADS,
        d_ff=D_FF,
        p_drop=P_DROP,
    )


class TestEncoderLayerConstruction:
    def test_submodules_registered(self, enc_layer):
        modules = dict(enc_layer.named_modules())
        assert isinstance(modules['mha'], MultiHeadAttention)
        assert isinstance(modules['norm1'], AddAndNorm)
        assert isinstance(modules['ffnn'], PositionwiseFeedForward)
        assert isinstance(modules['norm2'], AddAndNorm)


class TestEncoderLayerForward:
    def test_output_shape(self, enc_layer, x_enclayer):
        assert enc_layer(x=x_enclayer, mask=None).shape == (BATCH, SL_SRC, D_MODEL)

    def test_output_shape_with_mask(self, enc_layer, x_enclayer):
        mask = get_padding_mask(sl=SL_SRC)
        assert enc_layer(x=x_enclayer, mask=mask).shape == (BATCH, SL_SRC, D_MODEL)

    def test_eval_mode_is_deterministic(self, enc_layer, x_enclayer):
        enc_layer.eval()
        assert torch.allclose(enc_layer(x=x_enclayer), enc_layer(x=x_enclayer))

    def test_batch_independence(self, enc_layer):
        x1 = torch.randn((1, SL_SRC, D_MODEL))
        x2 = torch.randn((1, SL_SRC, D_MODEL))

        enc_layer.eval()
        out_stacked = enc_layer(x=torch.cat((x1, x2)))

        assert torch.allclose(out_stacked[0], enc_layer(x=x1))
        assert torch.allclose(out_stacked[1], enc_layer(x=x2))


class TestEncoderLayerWiring:
    def test_self_attention_wiring(self, enc_layer, x_enclayer):
        mask = get_padding_mask(sl=SL_SRC)

        # use MagicMock to wrap around the actual mha.forward call, to track the inputs and outputs for inspection
        with patch.object(enc_layer.mha, 'forward', wraps=enc_layer.mha.forward) as spy:
            enc_layer(x=x_enclayer, mask=mask)
            kwargs = spy.call_args.kwargs
            assert (
                kwargs['x_q'] is x_enclayer
            )  # use 'is' to check that the same tensor is passed, not just value equality
            assert kwargs['x_kv'] is x_enclayer
            assert kwargs['mask'] is mask

    def test_residual_wiring(self, enc_layer, x_enclayer):
        real_norm1 = enc_layer.norm1.forward
        captured_outs = {}

        def spying_on_norm1(residual_x, sublayer_x):
            out = real_norm1(residual_x=residual_x, sublayer_x=sublayer_x)
            captured_outs['norm1'] = out
            return out

        with (
            patch.object(enc_layer.norm1, 'forward', side_effect=spying_on_norm1) as spy_norm1,
            patch.object(enc_layer.norm2, 'forward', wraps=enc_layer.norm2.forward) as spy_norm2,
        ):
            enc_layer(x=x_enclayer)
            assert spy_norm1.call_args.kwargs['residual_x'] is x_enclayer
            assert spy_norm2.call_args.kwargs['residual_x'] is captured_outs['norm1']


class TestEncoderConstruction:
    def test_submodules_registered(self, enc):
        modules = dict(enc.named_modules())
        assert isinstance(modules['emb'], Embedding)
        assert isinstance(modules['pe'], PositionalEncoding)
        assert isinstance(modules['dropout'], nn.Dropout)
        assert isinstance(modules['enc'], nn.ModuleList)

        assert all(isinstance(el, EncoderLayer) for el in modules['enc'])
        assert len(modules['enc']) == N_ENC

    def test_dropout_p(self, enc):
        assert enc.dropout.p == P_DROP


class TestEncoderForward:
    def test_output_shape(self, enc, x_enc):
        assert enc(x=x_enc).shape == (BATCH, SL_SRC, D_MODEL)

    def test_output_shape_with_mask(self, enc, x_enc):
        mask = get_padding_mask(sl=SL_SRC)
        assert enc(x=x_enc, mask=mask).shape == (BATCH, SL_SRC, D_MODEL)

    def test_eval_mode_is_deterministic(self, enc, x_enc):
        enc.eval()
        assert torch.allclose(enc(x=x_enc), enc(x=x_enc))


class TestEncoderWiring:
    def test_embedding_and_pos_encoding_combined(self, enc, x_enc):
        enc.eval()

        emb_stub = torch.full(size=(BATCH, SL_SRC, D_MODEL), fill_value=1.0)
        pe_stub = torch.full(size=(SL_SRC, D_MODEL), fill_value=2.0)

        with (
            patch.object(enc.emb, 'forward', return_value=emb_stub),
            patch.object(enc.pe, 'forward', return_value=pe_stub),
            patch.object(enc.enc[0], 'forward', wraps=enc.enc[0].forward) as spy,
        ):
            enc(x=x_enc)
            assert torch.equal(spy.call_args.kwargs['x'], emb_stub + pe_stub)

    def test_enc_layers_are_connected(self, enc, x_enc):
        assert len(enc.enc) >= 2, 'needs atleast two encoder layers for this test'

        mask = get_padding_mask(sl=SL_SRC)
        first_enclayer = enc.enc[0].forward

        captured_output = {}

        def spying_on_enclayer1(x, mask):
            out = first_enclayer(x=x, mask=mask)
            captured_output['enclayer1'] = out
            return out

        with (
            patch.object(enc.enc[0], 'forward', side_effect=spying_on_enclayer1) as spy_enclayer1,
            patch.object(enc.enc[1], 'forward', wraps=enc.enc[1].forward) as spy_enclayer2,
        ):
            enc(x=x_enc, mask=mask)

            assert spy_enclayer1.call_args.kwargs['mask'] is mask
            assert spy_enclayer2.call_args.kwargs['mask'] is mask

            assert spy_enclayer2.call_args.kwargs['x'] is captured_output['enclayer1']
