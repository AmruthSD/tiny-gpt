import torch.nn as nn

from src.config import (
    d_model,
    dropout
)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                d_model,
                4 * d_model
            ),

            nn.GELU(),

            nn.Linear(
                4 * d_model,
                d_model
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)