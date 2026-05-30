import torch.nn as nn

from src.config import (
    d_model,
    vocab_size,
    num_layers
)

from src.model.embeddings import Embeddings
from src.model.block import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embeddings = Embeddings()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

    def forward(self, x, mask=None):

        # token ids -> embeddings
        x = self.embeddings(x)

        # transformer blocks
        for block in self.blocks:
            x = block(
                x,
                mask
            )

        # final normalization
        x = self.final_norm(x)

        # project to vocabulary
        logits = self.lm_head(x)

        return logits