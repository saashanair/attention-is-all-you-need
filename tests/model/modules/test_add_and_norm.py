import pytest
import torch
import torch.nn as nn
from transformer.model.modules import AddAndNorm

BATCH = 3
SEQ_LEN = 5
D_MODEL = 4
P_DROP = 0.1

@pytest.fixture
def add_and_norm_layer():
    return AddAndNorm(d_model=D_MODEL, p_drop=P_DROP)

@pytest.fixture
def x_residual():
    return torch.randn((BATCH, SEQ_LEN, D_MODEL))

@pytest.fixture
def x_sublayer():
    return torch.randn((BATCH, SEQ_LEN, D_MODEL))

class TestAddAndNormConstruction:
    def test_submodules_registered(self, add_and_norm_layer):
        modules = dict(add_and_norm_layer.named_modules())
        assert isinstance(modules['ln'], nn.LayerNorm)
        assert isinstance(modules['dropout'], nn.Dropout)

    def test_dropout_p(self, add_and_norm_layer):
        assert add_and_norm_layer.dropout.p == P_DROP

    def test_layer_norm_shape(self, add_and_norm_layer):
        assert add_and_norm_layer.ln.normalized_shape == (D_MODEL,)

class TestAddAndNormForward:
    def test_output_shape(self, add_and_norm_layer, x_residual, x_sublayer):
        assert add_and_norm_layer(x_residual, x_sublayer).shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_dropout_in_train_mode(self, add_and_norm_layer, x_residual, x_sublayer):
        add_and_norm_layer.train()
        assert not torch.allclose(add_and_norm_layer(x_residual, x_sublayer), add_and_norm_layer(x_residual, x_sublayer))

    def test_dropout_in_eval_mode(self, add_and_norm_layer, x_residual, x_sublayer):
        add_and_norm_layer.eval()
        assert torch.allclose(add_and_norm_layer(x_residual, x_sublayer), add_and_norm_layer(x_residual, x_sublayer))

    def test_add_norm_computation(self, add_and_norm_layer, x_residual, x_sublayer):
        add_and_norm_layer.eval() # turn off dropout => dropout(x_sublayer) == x_sublayer

        out = add_and_norm_layer(x_residual, x_sublayer)
        expected = add_and_norm_layer.ln(x_residual + x_sublayer)
        assert torch.allclose(out, expected)

    def test_dropout_only_applied_to_sublayer(self, x_residual, x_sublayer):
        an_layer = AddAndNorm(d_model=D_MODEL, p_drop=1.0) # special case: when p_drop=1, all neurons relating to the sublayer branch should be dropped
        an_layer.train() # ensure that the layer is in train mode, i.e. that dropout is being applied
        
        out = an_layer(x_residual, x_sublayer)
        expected = an_layer.ln(x_residual)
        assert torch.allclose(out, expected)