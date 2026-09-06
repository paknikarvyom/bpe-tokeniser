# BPE Tokeniser v1

A from-scratch **Byte-Pair Encoding** (BPE) tokeniser written in pure Python — no external dependencies.

## What is BPE?

Byte-Pair Encoding is a subword tokenisation algorithm widely used in modern language models (GPT-2, GPT-4, LLaMA, etc.). It starts with a base vocabulary of individual bytes and iteratively merges the most frequent adjacent pair of tokens until a target vocabulary size is reached.

## Features

- **Byte-level base vocabulary** (256 tokens) — handles any UTF-8 text, no unknown tokens
- **Pre-tokenisation** splits on word boundaries, contractions, digits, and punctuation
- **Interactive CLI** — paste your own training corpus, set vocab size, and encode/decode in a loop
- **Tokenised output** — see your training text segmented into learned tokens with inline IDs
- **Save / Load** — persist learned merges to a plain-text file and reload later
- **Zero dependencies** — only uses the Python standard library

## Quick Start

```bash
python3 v1/tokeniser-v1.py
```

You will be prompted to:

1. **Paste training text** (blank line to finish)
2. **Set vocabulary size** (default 300)
3. **View the tokenised training text** — the full input segmented into BPE tokens with their IDs
4. **Encode / decode** any text interactively via the `>>>` prompt

### Example Output

```
Tokenised training text
════════════════════════════════════════════════════════════
  [To:260]|[ be:258]|[,:44]|[ or:262]|[ not:265]|[ to:266]|[ be:258]|[,:44]
  (8 tokens)

>>> Byte-pair encoding
  Tokens (5): [66, 275, 45, 112, 277]
  Token string: [B:66]|[yte:275]|[-:45]|[p:112]|[air:277]
```

## Programmatic Usage

```python
from v1.tokeniser_v1 import BPETokeniser

tok = BPETokeniser()

# Train on a corpus
tok.train("your training text here ...", vocab_size=500, verbose=True)

# Encode
ids = tok.encode("hello world")

# Decode
text = tok.decode(ids)  # "hello world"

# Save / Load
tok.save("my_tokeniser.bpe")
tok.load("my_tokeniser.bpe")
```

## Algorithm

1. Start with 256 single-byte tokens (0x00–0xFF).
2. Pre-tokenise the corpus into word-like chunks.
3. Count every adjacent token pair across all chunks.
4. Merge the most frequent pair → assign a new token ID.
5. Repeat until `vocab_size` is reached.

Encoding applies the learned merges in priority order.  
Decoding concatenates the byte strings each token ID maps to.

## Project Structure

```
.
├── README.md
├── LICENSE
└── v1/
    └── tokeniser-v1.py    # BPE tokeniser implementation + interactive CLI
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
