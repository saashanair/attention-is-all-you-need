import math

import pytest
import torch
from torch import nn

from tests.model._dims import BATCH, D_KQ, D_MODEL, D_V, HEADS, SL_KV, SL_Q
from transformer.model.modules import MultiHeadAttention


def _get_mask_1d(blocked_keys: list[int], sl: int):
    m = torch.zeros(sl, dtype=torch.bool)
    m[blocked_keys] = True
    return m  # (sl,)


def _get_mask_2d(blocked_pairs: list[tuple[int, int]], sl_rows: int, sl_cols: int):
    m = torch.zeros(sl_rows, sl_cols, dtype=torch.bool)
    for row, col in blocked_pairs:
        m[row, col] = True
    return m  # (sl_rows, sl_cols)


@pytest.fixture
def x_q():
    return torch.randn((BATCH, SL_Q, D_MODEL))


@pytest.fixture
def x_kv():
    return torch.randn((BATCH, SL_KV, D_MODEL))


@pytest.fixture
def mha_layer():
    return MultiHeadAttention(d_model=D_MODEL, d_kq=D_KQ, d_v=D_V, h=HEADS)


class TestMultiHeadAttentionConstruction:
    def test_submodules_registered(self, mha_layer):
        modules = dict(mha_layer.named_modules())
        assert isinstance(modules['w_q'], nn.Linear)
        assert isinstance(modules['w_k'], nn.Linear)
        assert isinstance(modules['w_v'], nn.Linear)
        assert isinstance(modules['w_o'], nn.Linear)

    def test_param_shapes(self, mha_layer):
        # weight.shape == (out_features, in_features)
        assert mha_layer.w_q.weight.shape == (D_KQ * HEADS, D_MODEL)
        assert mha_layer.w_k.weight.shape == (D_KQ * HEADS, D_MODEL)
        assert mha_layer.w_v.weight.shape == (D_V * HEADS, D_MODEL)
        assert mha_layer.w_o.weight.shape == (D_MODEL, D_V * HEADS)

    def test_scaling_factor_value(self, mha_layer):
        assert mha_layer.scaling_factor == math.sqrt(D_KQ)


class TestMultiHeadAttentionForward:
    def test_self_attention_output_shape(self, mha_layer, x_q):
        out = mha_layer(x_q=x_q, x_kv=x_q)
        assert out.shape == (BATCH, SL_Q, D_MODEL)

    def test_masked_self_attention_output_shape(self, mha_layer, x_q):
        out = mha_layer(x_q=x_q, x_kv=x_q, mask=_get_mask_1d([-2, -1], sl=SL_Q))
        assert out.shape == (BATCH, SL_Q, D_MODEL)

    def test_cross_attention_output_shape(self, mha_layer, x_q, x_kv):
        out = mha_layer(x_q=x_q, x_kv=x_kv)
        assert out.shape == (BATCH, SL_Q, D_MODEL)

    def test_masked_cross_attention_output_shape(self, mha_layer, x_q, x_kv):
        out = mha_layer(x_q=x_q, x_kv=x_kv, mask=_get_mask_1d([-2, -1], sl=SL_KV))
        assert out.shape == (BATCH, SL_Q, D_MODEL)

    def test_attention_output_shape_with_independent_dims(self, x_kv):
        mha = MultiHeadAttention(d_model=D_MODEL, d_kq=3, d_v=5, h=2)  # D_MODEL=4, d_kq*h != d_model, d_kq != d_v
        assert mha(x_q=x_kv, x_kv=x_kv).shape == (BATCH, SL_KV, D_MODEL)

    def test_backward_populates_grads(self, mha_layer, x_kv):
        # when backward() is called, .grad of the parameters go from being None, to a value
        assert all(p.grad is None for p in mha_layer.parameters())

        mha_layer(x_q=x_kv, x_kv=x_kv).sum().backward()
        assert all(p.grad is not None for p in mha_layer.parameters())

    @pytest.mark.parametrize(
        'mask, perturb_keys',
        [
            pytest.param(
                _get_mask_1d([8, 9], sl=SL_KV), [8, 9], id='1d_perturb_masked'
            ),  # bump the same keys the mask blocks -> output must not move
            pytest.param(
                _get_mask_1d([8, 9], sl=SL_KV), [0, 4], id='1d_perturb_visible'
            ),  # bump keys the mask lets through -> output must move
            pytest.param(
                _get_mask_2d([(0, 5), (1, 5), (2, 5)], sl_rows=SL_Q, sl_cols=SL_KV), [5], id='2d_per_query'
            ),  # key 5 blocked for rows 0-2 only -> rows 0-2 immune, rows 3-4 not
        ],
    )
    def test_mask_excludes_disallowed_keys(self, mha_layer, x_q, x_kv, mask, perturb_keys):
        # We cannot hand compute attention output, so this is a metamorphic test:
        # bump some KEY columns and assert exactly which output rows move.
        # Contract: bumping key j can change output row i only if the mask lets
        # query i attend to key j.

        full = mask.expand(
            SL_Q, SL_KV
        )  # (SL_Q, SL_KV) view; a 1d mask broadcasts up over queries, a 2d mask is unchanged
        perturb = torch.tensor(perturb_keys)  # the key columns we bump

        out1 = mha_layer(x_q=x_q, x_kv=x_kv, mask=mask)
        assert (
            not out1.isnan().any()
        )  # none of these masks block every key for a query, so no row softmaxes to all -inf -> NaN

        x_kv2 = x_kv.clone()
        x_kv2[:, perturb, :] += 1.0  # fixed +1 on the perturbed key/value rows; x_q is left alone
        out2 = mha_layer(x_q=x_q, x_kv=x_kv2, mask=mask)

        # immune[i]: are ALL the bumped keys blocked for query i? -> row i cannot be affected.
        # full[:, perturb] picks the bumped columns from every row; .all(dim=-1) ANDs across them.
        immune = full[:, perturb].all(dim=-1)  # (SL_Q,); boolean
        # unchanged[i]: did output row i stay identical between out1 and out2?
        # .all(dim=2) = whole feature vector matches; .all(dim=0) = matches for every batch element.
        unchanged = torch.isclose(out1, out2, atol=1e-6).all(dim=2).all(dim=0)  # (SL_Q,); boolean

        # a row is unchanged exactly when it is immune: masked keys have no effect,
        # and (contrapositive) visible keys genuinely do
        assert torch.equal(unchanged, immune)
