import torch
import time

from src.config import (
    checkpoint_path,
    context_length,
    max_new_tokens,
    temperature
)

from src.tokenizer import BPETokenizer
from src.model.model import TransformerModel

def generate_without_cache(tokens):

    tokens = tokens.copy()

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            x = torch.tensor(
                [tokens[-context_length:]],
                dtype=torch.long,
                device=device
            )

            logits, _ = model(x)

            logits = logits[0, -1]

            logits = logits / temperature

            probs = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            ).item()

            tokens.append(next_token)

            if next_token == 3:
                break

    elapsed = time.perf_counter() - start

    return tokens, elapsed

def generate_with_cache(tokens):

    tokens = tokens.copy()

    start = time.perf_counter()

    with torch.no_grad():

        cache = None

        x = torch.tensor(
            [tokens[-context_length:]],
            dtype=torch.long,
            device=device
        )

        logits, cache = model(
            x,
            kv_cache=None
        )

        for _ in range(max_new_tokens):

            logits = logits[0, -1]

            logits = logits / temperature

            probs = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            ).item()

            tokens.append(next_token)

            if next_token == 3:
                break

            x = torch.tensor(
                [[next_token]],
                dtype=torch.long,
                device=device
            )

            logits, cache = model(
                x,
                kv_cache=cache
            )

    elapsed = time.perf_counter() - start

    return tokens, elapsed

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

tokens_no_cache, time_no_cache = generate_without_cache(tokens)

tokens_cache, time_cache = generate_with_cache(tokens)
        
generated_text_cache = tokenizer.decode(
    tokens_cache
)
print("\nGenerated Text with cache:\n")
print(generated_text_cache)

generated_text_no_cache = tokenizer.decode(
    tokens_no_cache
)
print("\nGenerated Text with no cache:\n")
print(generated_text_no_cache)


print(f"No cache: {time_no_cache:.4f}s")
print(f"With cache: {time_cache:.4f}s")

print(
    f"Speedup: {time_no_cache / time_cache:.2f}x"
)
