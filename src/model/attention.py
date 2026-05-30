import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    d_model,
    num_heads,
    dropout
)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(
            d_model,
            d_model
        )

        self.key = nn.Linear(
            d_model,
            d_model
        )

        self.value = nn.Linear(
            d_model,
            d_model
        )

        self.output = nn.Linear(
            d_model,
            d_model
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        batch_size, seq_len, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / (self.head_dim ** 0.5)

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(attention)

        out = torch.matmul(
            attention,
            v
        )

        out = out.transpose(1, 2)
        out = out.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        out = self.output(out)

        return out