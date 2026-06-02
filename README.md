# tiny-gpt

A GPT-style language model built from scratch in PyTorch to understand how modern Large Language Models work internally. This project was not created to compete with state-of-the-art models or achieve the best possible benchmark scores. Instead, the goal was to learn how transformer-based language models work under the hood by implementing them from first principles rather than relying on high-level APIs or pre-trained models.

This project focuses on understanding and implementing those concepts directly, including tokenization, attention mechanisms, transformer blocks, training, and autoregressive text generation. The model is trained on the TinyStories dataset, which provides a lightweight and accessible dataset for experimenting with language model training while keeping the focus on learning the architecture itself.

- Tokenization using Byte Pair Encoding (BPE)
- Token and positional embeddings
- Multi-Head Self-Attention
- Causal masking
- Feed-forward networks
- Residual connections
- Layer normalization
- Autoregressive text generation
- KV Cache for efficient inference
- Temperature, Top-K, and Top-P sampling

The objective is educational: to develop a deeper understanding of transformer architectures by building every major component from scratch.

---

## Features

- GPT-style decoder-only transformer
- Custom BPE tokenizer
- Training pipeline in PyTorch
- Text generation
- KV Cache support for faster inference
- Temperature sampling
- Top-K sampling
- Top-P (nucleus) sampling
- Configurable architecture and training parameters

---

## Project Structure

```text
.
├── src
│   ├── model
│   │   ├── attention.py
│   │   ├── block.py
│   │   ├── embeddings.py
│   │   ├── feedforward.py
│   │   └── model.py
│   │
│   ├── config.py
│   ├── dataset.py
│   └── tokenizer.py
│
├── datasetLoader.py
├── train.py
├── generate.py
├── requirements.txt
```

### File Overview

| File                       | Purpose                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| `datasetLoader.py`         | Downloads and prepares the TinyStories dataset                     |
| `src/tokenizer.py`         | BPE tokenizer training, encoding, and decoding                     |
| `src/dataset.py`           | Dataset preparation and token loading                              |
| `src/config.py`            | Central configuration for model, training, and generation settings |
| `src/model/embeddings.py`  | Token and positional embeddings                                    |
| `src/model/attention.py`   | Multi-head causal self-attention and KV cache logic                |
| `src/model/feedforward.py` | Transformer feed-forward network                                   |
| `src/model/block.py`       | Transformer block implementation                                   |
| `src/model/model.py`       | Full GPT-style model definition                                    |
| `train.py`                 | Model training entry point                                         |
| `generate.py`              | Text generation and inference                                      |

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

Most project settings can be modified through:

```text
src/config.py
```

This file contains parameters such as:

- Model dimensions
- Number of transformer layers
- Number of attention heads
- Context length
- Batch size
- Learning rate
- Training epochs
- Generation settings
- Sampling parameters
- Dataset locations
- Checkpoint paths

Feel free to experiment with these values to understand how different architectural and training choices affect model behavior.

---

## Training

To load the dataset:

```bash
python3 datasetLoader.py
```

To train the model:

```bash
python3 train.py
```

Model checkpoints will be saved in a new folder /componenets.

---

## Text Generation

To generate text using a trained checkpoint:

```bash
python3 generate.py
```

Generation settings such as:

- Temperature
- Top-K
- Top-P
- Maximum tokens

can be adjusted through `config.py`.

---

## Example Outputs

After training with 1% of the TinyStories dataset for 2 epochs.

Train Loss: 2.5003

Val Loss: 2.4839

Perplexity: 11.99

### Prompt

Prompt: Once upon a time

### Generated Output

Generated Text with cache:

Once upon a time , there was a time there was a big , there was time , but she thought they could be a time to get a little girl is a time there was a fun and saw a big and a beautiful and one day , she was a time . The bunny was a big bird was a happy . His mom called the park . They were a girl to use the cat was so much bigger than the kitchen , not like she wanted to use the girl . He did as he saw how happy because she

Generated Text with no cache:

Once upon a time , there was a little girl named Lily . She loved to skip and play with her friends . One day , she met a little girl named Lily . She saw a pretty bird in a pretty garden with many flowers . Lily thought it was pretty and pretty flowers . One day , she decided to play with her friends . They saw a shiny ball . The bird was sitting on a branch in the sky . It was white and fluffy , and it was very cold . Lily wanted to see the bird , but

No cache: 0.2938s

With cache: 0.0725s

Speedup: 4.05x

---
