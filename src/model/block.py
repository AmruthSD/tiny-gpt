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

    def forward(self, x, kv_cache=None):

        attn_out, new_kv_cache = self.attention(
            self.norm1(x),
            kv_cache
        )

        # Attention block
        x = x + attn_out

        # Feed-forward block
        x = x + self.feedforward(
            self.norm2(x)
        )

        return x, new_kv_cache