import torch
import torch.nn.functional as F
from torch import nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p_drop: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.p_drop = p_drop

        self.l1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.l2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (x) -> (batch, sequence_length, d_model)
        # middle-layer -> (batch, sequence_length, d_ff)
        # output -> (batch, sequence_length, d_model)

        x = F.relu(self.l1(x))
        x = self.dropout(x) # the paper does not mention a dropout here, but the TF implementation of the paper adds it in
        x = self.l2(x)
        return x
