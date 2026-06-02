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

    def forward(self, x, kv_cache=None):

        x = self.embeddings(x)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        new_kv_cache = []

        for block, block_cache in zip(self.blocks, kv_cache):

            x, updated_cache = block(
                x,
                block_cache
            )

            new_kv_cache.append(updated_cache)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, new_kv_cache