import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from src.tokenizer import BPETokenizer
from src.config import vocab_size, train_split


DATA_DIR = "data"

TRAIN_PATH = os.path.join(DATA_DIR, "train.npy")
VAL_PATH = os.path.join(DATA_DIR, "val.npy")

TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")


def prepare_dataset(
    vocab_size: int = vocab_size,
    train_split: float = train_split,
    dataset_fraction: str = "1%"
):
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading TinyStories dataset...")
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=f"train[:{dataset_fraction}]"
    )

    texts = dataset["text"]

    print("Training tokenizer...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)

    tokenizer.train(texts)

    tokenizer.save(TOKENIZER_PATH)

    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    print(f"Actual vocab size: {tokenizer.vocab_size_actual}")

    print("Tokenizing dataset...")

    all_tokens = []

    for text in tqdm(texts):
        ids = tokenizer.encode(text)

        # Add EOS token between stories
        ids.append(3)  # [EOS]

        all_tokens.extend(ids)

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    split_idx = int(len(all_tokens) * train_split)

    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")

    np.save(TRAIN_PATH, train_tokens)
    np.save(VAL_PATH, val_tokens)

    print(f"Saved train dataset to {TRAIN_PATH}")
    print(f"Saved val dataset to {VAL_PATH}")


if __name__ == "__main__":
    prepare_dataset()