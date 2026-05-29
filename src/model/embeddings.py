import torch
import torch.nn as nn

from src.config import vocab_size, d_model, max_seq_len

class Embeddings(nn.Module):
    def __init__(
        self,
        vocab_size = vocab_size,
        d_model = d_model,
        max_seq_len = max_seq_len
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )
        
        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

    def forward(self, x):

        batch_size, seq_len = x.shape
        positions = torch.arange(
            seq_len,
            device=x.device
        )

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        
        x = token_embeddings + position_embeddings

        return x