import math
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scaling_factor = math.sqrt(d_model)
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape (x) -> (batch, sequence_length), and each item is the token id
        # output shape -> (batch, sequence_length, d_model)
        return self.emb(x) * self.scaling_factor