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
    N_ENC,
    P_DROP,
    SL_SRC,
    SL_TGT,
    VOCAB_SIZE,
    VOCAB_SIZE_SRC,
    VOCAB_SIZE_TGT,
)
from tests.model._mask_helpers import get_causal_mask, get_padding_mask
from transformer.model.decoder import Decoder
from transformer.model.encoder import Encoder
from transformer.model.transformer import Transformer, TransformerConfig

ARCH_CFG = TransformerConfig(
    n_enc=N_ENC,
    n_dec=N_DEC,
    max_seq_len=MAX_SEQ_LEN,
    d_model=D_MODEL,
    d_kq=D_KQ,
    d_v=D_V,
    h=HEADS,
    d_ff=D_FF,
    p_drop=P_DROP,
)


# === fixtures for model with shared embeddings ===
@pytest.fixture
def model():
    return Transformer(src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE, shared_embeddings=True, cfg=ARCH_CFG)


@pytest.fixture
def x_src_shared_emb():
    return torch.randint(size=(BATCH, SL_SRC), low=0, high=VOCAB_SIZE)


@pytest.fixture
def x_tgt_shared_emb():
    return torch.randint(size=(BATCH, SL_TGT), low=0, high=VOCAB_SIZE)


# === fixtures for model with shared embeddings ===
@pytest.fixture
def model_independent_emb():
    return Transformer(
        src_vocab_size=VOCAB_SIZE_SRC, tgt_vocab_size=VOCAB_SIZE_TGT, shared_embeddings=False, cfg=ARCH_CFG
    )


@pytest.fixture
def x_src_independent_emb():
    return torch.randint(size=(BATCH, SL_SRC), low=0, high=VOCAB_SIZE_SRC)


@pytest.fixture
def x_tgt_independent_emb():
    return torch.randint(size=(BATCH, SL_TGT), low=0, high=VOCAB_SIZE_TGT)


# === shared fixtures ===
@pytest.fixture
def src_pad_mask():
    return get_padding_mask(sl=SL_SRC).expand((BATCH, -1))  # (batch, sl_src)


@pytest.fixture
def tgt_pad_mask():
    return get_padding_mask(sl=SL_TGT).expand((BATCH, -1))  # (batch, sl_tgt)


class TestTransformerConfig:
    def test_transformer_arch_config_construction(self):
        assert ARCH_CFG.n_enc == N_ENC
        assert ARCH_CFG.n_dec == N_DEC

    @pytest.mark.parametrize('d_model_to_test, expect_error', [(256, False), (255, True)])
    def test_d_model_is_even_validation(self, d_model_to_test, expect_error):
        if expect_error:
            with pytest.raises(ValueError):
                TransformerConfig(d_model=d_model_to_test)


class TestTransformerConstruction:
    def test_submodules_registered(self, model):
        modules = dict(model.named_modules())
        assert isinstance(modules['encoder'], Encoder)
        assert isinstance(modules['decoder'], Decoder)
        assert isinstance(modules['linear'], nn.Linear)
        assert 'causal_mask' in dict(model.named_buffers())  # check causal_mask is registered in the buffer
        assert (
            'causal_mask' not in model.state_dict()
        )  # check causal_mask is not persistent, i.e. not stored in the state_dict

    def test_submodule_shapes(self, model_independent_emb):
        assert model_independent_emb.encoder.emb.emb.weight.shape == (VOCAB_SIZE_SRC, D_MODEL)
        assert model_independent_emb.decoder.emb.emb.weight.shape == (VOCAB_SIZE_TGT, D_MODEL)
        assert model_independent_emb.linear.weight.shape == (VOCAB_SIZE_TGT, D_MODEL)

    @pytest.mark.parametrize(
        'shared_emb, vs_src, vs_tgt, expect_error',
        [(True, 100, 100, False), (True, 100, 200, True), (False, 100, 200, False)],
    )
    def test_shared_embeddings_validation(self, shared_emb, vs_src, vs_tgt, expect_error):
        if expect_error:
            with pytest.raises(ValueError):
                Transformer(src_vocab_size=vs_src, tgt_vocab_size=vs_tgt, shared_embeddings=shared_emb, cfg=ARCH_CFG)

    def test_causal_mask_at_construction(self, model):
        assert model.causal_mask.shape == (MAX_SEQ_LEN, MAX_SEQ_LEN)
        assert isinstance(model.causal_mask, torch.Tensor)
        assert model.causal_mask.dtype == torch.bool

    def test_shared_embeddings_tie_encoder_and_decoder(self, model):
        assert torch.equal(model.decoder.emb.emb.weight, model.encoder.emb.emb.weight)
        assert torch.equal(model.linear.weight, model.encoder.emb.emb.weight)
        assert torch.equal(model.linear.weight, model.decoder.emb.emb.weight)

    def test_independent_embeddings_dont_tie_encoder_and_decoder(self, model_independent_emb):
        assert not torch.equal(
            model_independent_emb.decoder.emb.emb.weight, model_independent_emb.encoder.emb.emb.weight
        )
        assert not torch.equal(model_independent_emb.linear.weight, model_independent_emb.encoder.emb.emb.weight)
        assert torch.equal(model_independent_emb.linear.weight, model_independent_emb.decoder.emb.emb.weight)


class TestTransformerForward:
    def test_output_shape_with_shared_embeddings(
        self, model, x_src_shared_emb, x_tgt_shared_emb, src_pad_mask, tgt_pad_mask
    ):
        assert model(
            src=x_src_shared_emb, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask
        ).shape == (BATCH, SL_TGT, VOCAB_SIZE)

    def test_output_shape_with_independent_embeddings(
        self, model_independent_emb, x_src_independent_emb, x_tgt_independent_emb, src_pad_mask, tgt_pad_mask
    ):
        assert model_independent_emb(
            src=x_src_independent_emb, tgt=x_tgt_independent_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask
        ).shape == (BATCH, SL_TGT, VOCAB_SIZE_TGT)

    def test_causal_mask_maps_to_tgt(self, model):
        cm = model._compute_causal_mask(mask_size=SL_TGT)
        cm_expected = get_causal_mask(sl=SL_TGT)
        assert cm.shape == cm_expected.shape
        assert torch.equal(cm, cm_expected)

    def test_eval_mode_is_deterministic(self, model, x_src_shared_emb, x_tgt_shared_emb, src_pad_mask, tgt_pad_mask):
        model.eval()
        assert torch.allclose(
            model(src=x_src_shared_emb, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask),
            model(src=x_src_shared_emb, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask),
        )

    def test_gradient_flow_with_shared_embeddings(
        self, model, x_src_shared_emb, x_tgt_shared_emb, src_pad_mask, tgt_pad_mask
    ):
        logits = model(src=x_src_shared_emb, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        logits.sum().backward()

        assert model.encoder.emb.emb.weight.grad is not None
        assert model.decoder.emb.emb.weight.grad is model.encoder.emb.emb.weight.grad
        assert model.linear.weight.grad is model.encoder.emb.emb.weight.grad

    def test_gradient_flow_with_independent_embeddings(
        self, model_independent_emb, x_src_independent_emb, x_tgt_independent_emb, src_pad_mask, tgt_pad_mask
    ):
        logits = model_independent_emb(
            src=x_src_independent_emb, tgt=x_tgt_independent_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask
        )
        logits.sum().backward()

        assert model_independent_emb.encoder.emb.emb.weight.grad is not None
        assert model_independent_emb.decoder.emb.emb.weight.grad is not None
        assert (
            model_independent_emb.decoder.emb.emb.weight.grad is not model_independent_emb.encoder.emb.emb.weight.grad
        )
        assert model_independent_emb.linear.weight.grad is model_independent_emb.decoder.emb.emb.weight.grad


class TestTransformerWiring:
    def test_attention_wiring(self, model, x_src_shared_emb, x_tgt_shared_emb, src_pad_mask, tgt_pad_mask):
        enc_mask_expected = src_pad_mask.unsqueeze(1).unsqueeze(1)
        self_mask_expected = get_causal_mask(sl=SL_TGT) | get_padding_mask(sl=SL_TGT).expand((BATCH, -1)).unsqueeze(
            1
        ).unsqueeze(1)

        real_enc = model.encoder.forward
        captured_outs = {}

        def spy_on_enc(x, mask):
            out = real_enc(x=x, mask=mask)
            captured_outs['enc'] = out
            return out

        with (
            patch.object(model.encoder, 'forward', side_effect=spy_on_enc) as spy_encoder,
            patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as spy_decoder,
        ):
            model(src=x_src_shared_emb, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)

            assert spy_encoder.call_args.kwargs['x'] is x_src_shared_emb
            assert torch.equal(spy_encoder.call_args.kwargs['mask'], enc_mask_expected)

            assert spy_decoder.call_args.kwargs['x'] is x_tgt_shared_emb
            assert spy_decoder.call_args.kwargs['x_enc'] is captured_outs['enc']
            assert torch.equal(spy_decoder.call_args.kwargs['self_mask'], self_mask_expected)
            assert torch.equal(spy_decoder.call_args.kwargs['cross_mask'], enc_mask_expected)


class TestTransformerMasking:
    def test_src_padding(self, model, x_src_shared_emb, x_tgt_shared_emb, src_pad_mask, tgt_pad_mask):
        # test that padding in fact does ignore the output, but perturbing x_src_shared_emb at a location masked by src_pad_mask
        model.eval()

        baseline = model(
            src=x_src_shared_emb, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask
        )

        x_src_perturbed = x_src_shared_emb.clone()
        x_src_perturbed[:, -1] = (
            x_src_perturbed[:, -1] + 1
        ) % VOCAB_SIZE  # increase the last position by 1 within vocab_size

        perturbed = model(
            src=x_src_perturbed, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask
        )

        assert torch.equal(baseline, perturbed)

    def test_tgt_padding(self, model, x_src_shared_emb, x_tgt_shared_emb, src_pad_mask, tgt_pad_mask):
        model.eval()
        model.causal_mask.fill_(False)  # switch "off" the causal mask for this test

        baseline = model(
            src=x_src_shared_emb, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask
        )

        x_tgt_perturbed = x_tgt_shared_emb.clone()
        x_tgt_perturbed[:, -1] = (x_tgt_perturbed[:, -1] + 1) % VOCAB_SIZE

        perturbed = model(
            src=x_src_shared_emb, tgt=x_tgt_perturbed, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask
        )

        # every row except the one we perturbed should now be immune -- purely on padding, since causal offers nothing
        assert torch.equal(baseline[:, :-1, :], perturbed[:, :-1, :])

    def test_causal_mask_blocks_future_leakage(
        self, model, x_src_shared_emb, x_tgt_shared_emb, src_pad_mask, tgt_pad_mask
    ):
        model.eval()
        tgt_pad_mask_off = torch.zeros_like(tgt_pad_mask)
        idx_of_token_to_perturb = -1

        baseline = model(
            src=x_src_shared_emb, tgt=x_tgt_shared_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask_off
        )

        x_tgt_perturbed = x_tgt_shared_emb.clone()
        x_tgt_perturbed[:, idx_of_token_to_perturb] = (x_tgt_perturbed[:, idx_of_token_to_perturb] + 1) % VOCAB_SIZE

        perturbed = model(
            src=x_src_shared_emb, tgt=x_tgt_perturbed, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask_off
        )

        assert torch.equal(baseline[:, :idx_of_token_to_perturb, :], perturbed[:, :idx_of_token_to_perturb, :])
