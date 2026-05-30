import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention()
        self.feedforward = FeedForward()

        self.norm1 = nn.LayerNorm(self.attention.d_model)
        self.norm2 = nn.LayerNorm(self.attention.d_model)

    def forward(self, x, mask=None):

        # Attention block
        x = x + self.attention(
            self.norm1(x),
            mask
        )

        # Feed-forward block
        x = x + self.feedforward(
            self.norm2(x)
        )

        return x