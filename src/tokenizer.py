from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

from src.config import vocab_size


class BPETokenizer:
    def __init__(self, vocab_size: int = vocab_size):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, texts):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        )

        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        self.tokenizer.save(path)

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)

    @property
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()
