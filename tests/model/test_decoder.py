from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.model._dims import (
    BATCH,
    D_FF,
    D_KQ,
    D_MODEL,
    D_V,
    HEADS,
    MAX_SEQ_LEN,
    N_DEC,
    P_DROP,
    SL_SRC,
    SL_TGT,
    VOCAB_SIZE_TGT,
)
from tests.model._mask_helpers import get_causal_mask, get_padding_mask
from transformer.model.decoder import Decoder, DecoderLayer
from transformer.model.modules import (
    AddAndNorm,
    Embedding,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)


@pytest.fixture
def x_enc():
    return torch.randn((BATCH, SL_SRC, D_MODEL))


@pytest.fixture
def x_declayer():
    return torch.randn((BATCH, SL_TGT, D_MODEL))


@pytest.fixture
def dec_layer():
    return DecoderLayer(d_model=D_MODEL, d_kq=D_KQ, d_v=D_V, h=HEADS, d_ff=D_FF, p_drop=P_DROP)


@pytest.fixture
def x_dec():
    return torch.randint(size=(BATCH, SL_TGT), low=0, high=VOCAB_SIZE_TGT)


@pytest.fixture
def dec():
    return Decoder(
        n_dec=N_DEC,
        vocab_size=VOCAB_SIZE_TGT,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        d_kq=D_KQ,
        d_v=D_V,
        h=HEADS,
        d_ff=D_FF,
        p_drop=P_DROP,
    )


class TestDecoderLayerConsutruction:
    def test_submodules_registered(self, dec_layer):
        modules = dict(dec_layer.named_modules())
        assert isinstance(modules['masked_mha'], MultiHeadAttention)
        assert isinstance(modules['norm1'], AddAndNorm)
        assert isinstance(modules['cross_mha'], MultiHeadAttention)
        assert isinstance(modules['norm2'], AddAndNorm)
        assert isinstance(modules['ffnn'], PositionwiseFeedForward)
        assert isinstance(modules['norm3'], AddAndNorm)


class TestDecoderLayerForward:
    def test_output_shape(self, dec_layer, x_declayer, x_enc):
        assert dec_layer(x=x_declayer, x_enc=x_enc).shape == (BATCH, SL_TGT, D_MODEL)

    def test_output_shape_with_mask(self, dec_layer, x_declayer, x_enc):
        self_mask = get_causal_mask(SL_TGT)
        cross_mask = get_padding_mask(SL_SRC)

        assert dec_layer(x=x_declayer, x_enc=x_enc, self_mask=self_mask, cross_mask=cross_mask).shape == (
            BATCH,
            SL_TGT,
            D_MODEL,
        )

    def test_eval_mode_is_deterministic(self, dec_layer, x_declayer, x_enc):
        dec_layer.eval()
        assert torch.allclose(dec_layer(x=x_declayer, x_enc=x_enc), dec_layer(x=x_declayer, x_enc=x_enc))

    def test_batch_independence(self, dec_layer, x_enc):
        x1 = torch.randn((1, SL_TGT, D_MODEL))
        x2 = torch.randn((1, SL_TGT, D_MODEL))

        x_enc = x_enc[:2, :, :].clone()
        x_enc1, x_enc2 = x_enc[:1], x_enc[1:2]

        dec_layer.eval()
        out_stacked = dec_layer(x=torch.cat((x1, x2)), x_enc=x_enc)

        assert torch.allclose(out_stacked[0], dec_layer(x=x1, x_enc=x_enc1))
        assert torch.allclose(out_stacked[1], dec_layer(x=x2, x_enc=x_enc2))


class TestDecoderLayerWiring:
    def test_attention_wiring(self, dec_layer, x_declayer, x_enc):
        self_mask = get_causal_mask(sl=SL_TGT) | get_padding_mask(sl=SL_TGT)
        cross_mask = get_padding_mask(sl=SL_SRC)

        real_norm1 = dec_layer.norm1.forward
        captured_outs = {}

        def spy_on_norm1(residual_x, sublayer_x):
            out = real_norm1(residual_x=residual_x, sublayer_x=sublayer_x)
            captured_outs['norm1'] = out
            return out

        with (
            patch.object(dec_layer.masked_mha, 'forward', wraps=dec_layer.masked_mha.forward) as spy_masked_mha,
            patch.object(dec_layer.norm1, 'forward', side_effect=spy_on_norm1),
            patch.object(dec_layer.cross_mha, 'forward', wraps=dec_layer.cross_mha.forward) as spy_cross_mha,
        ):
            dec_layer(x=x_declayer, x_enc=x_enc, self_mask=self_mask, cross_mask=cross_mask)

            assert spy_masked_mha.call_args.kwargs['x_q'] is x_declayer
            assert spy_masked_mha.call_args.kwargs['x_kv'] is x_declayer
            assert spy_masked_mha.call_args.kwargs['mask'] is self_mask

            assert spy_cross_mha.call_args.kwargs['x_q'] is captured_outs['norm1']
            assert spy_cross_mha.call_args.kwargs['x_kv'] is x_enc
            assert spy_cross_mha.call_args.kwargs['mask'] is cross_mask

    def test_residual_wiring(self, dec_layer, x_declayer, x_enc):
        real_norms = {
            'norm1': dec_layer.norm1.forward,
            'norm2': dec_layer.norm2.forward,
        }
        captured_outs = {}

        def spy_on_norm1(residual_x, sublayer_x):
            out = real_norms['norm1'](residual_x=residual_x, sublayer_x=sublayer_x)
            captured_outs['norm1'] = out
            return out

        def spy_on_norm2(residual_x, sublayer_x):
            out = real_norms['norm2'](residual_x=residual_x, sublayer_x=sublayer_x)
            captured_outs['norm2'] = out
            return out

        with (
            patch.object(dec_layer.norm1, 'forward', side_effect=spy_on_norm1) as spy_norm1,
            patch.object(dec_layer.norm2, 'forward', side_effect=spy_on_norm2) as spy_norm2,
            patch.object(dec_layer.norm3, 'forward', wraps=dec_layer.norm3.forward) as spy_norm3,
        ):
            dec_layer(x=x_declayer, x_enc=x_enc)

            assert spy_norm1.call_args.kwargs['residual_x'] is x_declayer
            assert spy_norm2.call_args.kwargs['residual_x'] is captured_outs['norm1']
            assert spy_norm3.call_args.kwargs['residual_x'] is captured_outs['norm2']


class TestDecoderConstruction:
    def test_submodules_registered(self, dec):
        modules = dict(dec.named_modules())
        assert isinstance(modules['emb'], Embedding)
        assert isinstance(modules['pe'], PositionalEncoding)
        assert isinstance(modules['dropout'], nn.Dropout)
        assert isinstance(modules['dec'], nn.ModuleList)

        assert all(isinstance(dl, DecoderLayer) for dl in modules['dec'])
        assert len(modules['dec']) == N_DEC

    def test_dropout_p(self, dec):
        assert dec.dropout.p == P_DROP


class TestDecoderForward:
    def test_output_shape(self, dec, x_dec, x_enc):
        assert dec(x=x_dec, x_enc=x_enc).shape == (BATCH, SL_TGT, D_MODEL)

    def test_output_shape_with_mask(self, dec, x_dec, x_enc):
        self_mask = get_causal_mask(sl=SL_TGT) | get_padding_mask(sl=SL_TGT)
        cross_mask = get_padding_mask(sl=SL_SRC)
        assert dec(x=x_dec, x_enc=x_enc, self_mask=self_mask, cross_mask=cross_mask).shape == (BATCH, SL_TGT, D_MODEL)

    def test_eval_mode_is_deterministic(self, dec, x_dec, x_enc):
        dec.eval()
        assert torch.allclose(dec(x=x_dec, x_enc=x_enc), dec(x=x_dec, x_enc=x_enc))


class TestDecoderWiring:
    def test_embedding_and_pos_encoding_combined(self, dec, x_dec, x_enc):
        dec.eval()

        emb_stub = torch.full(size=(BATCH, SL_TGT, D_MODEL), fill_value=1.0)
        pe_stub = torch.full(size=(SL_TGT, D_MODEL), fill_value=2.0)

        with (
            patch.object(dec.emb, 'forward', return_value=emb_stub),
            patch.object(dec.pe, 'forward', return_value=pe_stub),
            patch.object(dec.dec[0], 'forward', wraps=dec.dec[0].forward) as spy,
        ):
            dec(x=x_dec, x_enc=x_enc)
            assert torch.equal(spy.call_args.kwargs['x'], emb_stub + pe_stub)

    def test_dec_layers_are_connected(self, dec, x_dec, x_enc):
        assert len(dec.dec) >= 2, 'needs atleast two decoder layers for this test'

        self_mask = get_causal_mask(sl=SL_TGT) | get_padding_mask(sl=SL_TGT)
        cross_mask = get_padding_mask(sl=SL_SRC)

        first_declayer = dec.dec[0].forward

        captured_output = {}

        def spying_on_declayer1(x, x_enc, self_mask, cross_mask):
            out = first_declayer(x=x, x_enc=x_enc, self_mask=self_mask, cross_mask=cross_mask)
            captured_output['declayer1'] = out
            return out

        with (
            patch.object(dec.dec[0], 'forward', side_effect=spying_on_declayer1) as spy_declayer1,
            patch.object(dec.dec[1], 'forward', wraps=dec.dec[1].forward) as spy_declayer2,
        ):
            dec(x=x_dec, x_enc=x_enc, self_mask=self_mask, cross_mask=cross_mask)

            assert spy_declayer1.call_args.kwargs['self_mask'] is self_mask
            assert spy_declayer1.call_args.kwargs['cross_mask'] is cross_mask

            assert spy_declayer2.call_args.kwargs['self_mask'] is self_mask
            assert spy_declayer2.call_args.kwargs['cross_mask'] is cross_mask

            assert spy_declayer2.call_args.kwargs['x'] is captured_output['declayer1']
