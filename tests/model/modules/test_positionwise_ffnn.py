import pytest
import torch
from torch import nn

from transformer.model.modules import PositionwiseFeedForward

BATCH = 10
SEQ_LEN = 3
D_MODEL = 4
D_FF = 16
P_DROP = 0.1

@pytest.fixture
def inp_x():
    # treating this as a fixture, rather than a constant defined above
    # ensures that each test gets its own fresh, independent input tensor
    # so state (autograd graph, requires_grad, in-place edits etc) does not
    # leak between tests
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)

@pytest.fixture
def ffnn_layer():
    return PositionwiseFeedForward(d_model=D_MODEL, d_ff=D_FF, p_drop=P_DROP)

class TestPositionwiseFFNNConstruction:
    def test_submodules_registered(self, ffnn_layer):
        modules = dict(ffnn_layer.named_modules())
        assert isinstance(modules['l1'], nn.Linear)
        assert isinstance(modules['l2'], nn.Linear)
        assert isinstance(modules['dropout'], nn.Dropout)

    def test_param_shapes(self, ffnn_layer):
        # weight.shape == (out_features, in_features)
        assert ffnn_layer.l1.weight.shape == (D_FF, D_MODEL)
        assert ffnn_layer.l2.weight.shape == (D_MODEL, D_FF)

    def test_dropout_p(self, ffnn_layer):
        assert ffnn_layer.dropout.p == P_DROP

class TestPositionwiseFFNNForward:
    @pytest.mark.parametrize("x_shape", [(D_MODEL,), (SEQ_LEN, D_MODEL), (BATCH, SEQ_LEN, D_MODEL)])
    def test_output_shape(self, ffnn_layer, x_shape):
        x = torch.randn(x_shape)
        assert ffnn_layer(x).shape == x_shape

    def test_dropout_active_in_train_mode(self, ffnn_layer, inp_x):
        ffnn_layer.train()
        assert not torch.allclose(ffnn_layer(inp_x), ffnn_layer(inp_x))

    def test_eval_mode_is_deterministic(self, ffnn_layer, inp_x):
        # in eval mode, ensure there is no randomness => forward pass on the same input, should result in same output
        ffnn_layer.eval()
        assert torch.allclose(ffnn_layer(inp_x), ffnn_layer(inp_x))

    def test_backward_populates_grads(self, ffnn_layer, inp_x):
        # when backward() is called, .grad of the parameters go from being None, to a value
        assert all(p.grad is None for p in ffnn_layer.parameters())

        ffnn_layer(inp_x).sum().backward()
        assert all(p.grad is not None for p in ffnn_layer.parameters())