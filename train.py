import torch
import torch.nn as nn
import math

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    batch_size,
    learning_rate,
    epochs,
    train_path,
    val_path
)

from src.dataset import TinyStoriesDataset
from src.model.model import TransformerModel


def evaluate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            logits = logits.view(
                -1,
                logits.size(-1)
            )

            y = y.view(-1)

            loss = criterion(
                logits,
                y
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    train_dataset = TinyStoriesDataset(
        train_path
    )

    val_dataset = TinyStoriesDataset(
        val_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = TransformerModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for x, y in pbar:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            logits = logits.view(
                -1,
                logits.size(-1)
            )

            y = y.view(-1)

            loss = criterion(
                logits,
                y
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss
            / len(train_loader)
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        perplexity = math.exp(val_loss)

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}\n"
        )

        print(
            f"Perplexity: {perplexity:.2f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            f"checkpoints/epoch_{epoch+1}.pt"
        )

if __name__ == "__main__":
    train()