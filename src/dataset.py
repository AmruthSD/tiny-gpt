import numpy as np
import torch

from torch.utils.data import Dataset

from src.config import context_length


class TinyStoriesDataset(Dataset):
    def __init__(self, path):
        self.tokens = np.load(path)

    def __len__(self):
        return len(self.tokens) - context_length

    def __getitem__(self, idx):

        x = self.tokens[
            idx : idx + context_length
        ]

        y = self.tokens[
            idx + 1 : idx + context_length + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )