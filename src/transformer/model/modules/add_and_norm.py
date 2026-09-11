import torch
from torch import nn


class AddAndNorm(nn.Module):
    def __init__(self, d_model: int, p_drop: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, residual_x: torch.Tensor, sublayer_x: torch.Tensor) -> torch.Tensor:
        # input (residual_x, sublayer_x): (batch, sequence_length, d_model)
        # output: (batch, sequence_length, d_model)
        return self.ln(residual_x + self.dropout(sublayer_x))

    