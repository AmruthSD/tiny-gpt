import torch

from src.config import (
    checkpoint_path,
    context_length,
    max_new_tokens,
    temperature
)

from src.tokenizer import BPETokenizer
from src.model.model import TransformerModel

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = TransformerModel().to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

tokenizer = BPETokenizer()

tokenizer.load(
    "data/tokenizer.json"
)

prompt = input(
    "Prompt: "
)

tokens = tokenizer.encode(
    prompt
)

with torch.no_grad():

    for _ in range(max_new_tokens):

        x = torch.tensor(
            [tokens[-context_length:]],
            dtype=torch.long,
            device=device
        )

        logits = model(x)

        logits = logits[
            0,
            -1
        ]

        logits = logits / temperature

        probs = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        ).item()

        tokens.append(
            next_token
        )

        if next_token == 3:
            break

generated_text = tokenizer.decode(
    tokens
)

print("\nGenerated Text:\n")
print(generated_text)