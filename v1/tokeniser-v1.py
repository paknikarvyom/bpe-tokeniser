"""
Byte-Pair Encoding (BPE) Tokeniser — v1

Train on raw text to learn a compact subword vocabulary,
then encode / decode arbitrary strings using that vocabulary.

Algorithm overview
──────────────────
1. Start with a base vocabulary of 256 single-byte tokens (0x00–0xFF).
2. Scan the training corpus (represented as a list of byte sequences) and
   count every adjacent pair of tokens.
3. Merge the most frequent pair into a single new token and record the
   merge rule.
4. Repeat until the desired vocabulary size is reached.

Encoding applies the learned merges in priority order; decoding simply
concatenates the byte strings that each token ID maps to.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


class BPETokeniser:
    """A minimal, from-scratch Byte-Pair Encoding tokeniser."""

    # Pre-tokenisation regex — splits on whitespace boundaries, punctuation,
    # and digits so that merges stay within "word-like" chunks.
    _SPLIT_PAT = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d"""   # common contractions
        r"""| ?\w+"""                      # optional leading space + word
        r"""| ?\d+"""                      # optional leading space + digits
        r"""| ?[^\s\w]+"""                 # optional leading space + punctuation
        r"""|\s+"""                        # remaining whitespace
        , re.UNICODE,
    )

    def __init__(self) -> None:
        # ── Vocabulary ──────────────────────────────────────────────
        # vocab maps token_id → bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # merges is an ordered list of (pair → new_id) learned during training
        self.merges: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self, text: str, vocab_size: int = 512, verbose: bool = False) -> None:
        """Learn BPE merges from *text* until *vocab_size* tokens exist.

        Parameters
        ----------
        text : str
            The raw training corpus.
        vocab_size : int
            Target vocabulary size (must be > 256).
        verbose : bool
            If ``True``, print each merge as it is learned.
        """
        assert vocab_size > 256, "vocab_size must be > 256 (the 256 byte tokens are the base)"
        num_merges = vocab_size - 256

        # Pre-tokenise into chunks so merges don't cross word boundaries.
        chunks = re.findall(self._SPLIT_PAT, text)

        # Convert each chunk to a list of byte-valued token IDs.
        # We keep one list per *unique* chunk and a count of how often it
        # appeared — this makes pair counting much faster on large corpora.
        chunk_counts: dict[tuple[int, ...], int] = Counter()
        for chunk in chunks:
            ids = tuple(chunk.encode("utf-8"))
            chunk_counts[ids] += 1

        # Mutable version: list-of-lists so we can merge in-place.
        splits: list[tuple[list[int], int]] = [
            (list(ids), count) for ids, count in chunk_counts.items()
        ]

        for i in range(num_merges):
            # 1. Count every adjacent pair, weighted by chunk frequency.
            pair_counts: dict[tuple[int, int], int] = {}
            for ids, count in splits:
                for j in range(len(ids) - 1):
                    pair = (ids[j], ids[j + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count

            if not pair_counts:
                break  # nothing left to merge

            # 2. Pick the most frequent pair.
            best_pair = max(pair_counts, key=pair_counts.get)  # type: ignore[arg-type]

            # 3. Assign it a new token ID.
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {best_pair} → {new_id}  "
                    f"({self.vocab[new_id]!r})  freq={pair_counts[best_pair]}"
                )

            # 4. Apply the merge to every chunk.
            splits = [
                (self._merge_pair(ids, best_pair, new_id), count)
                for ids, count in splits
            ]

    # ------------------------------------------------------------------ #
    #  Encoding                                                            #
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> list[int]:
        """Encode *text* into a list of token IDs."""
        chunks = re.findall(self._SPLIT_PAT, text)
        token_ids: list[int] = []
        for chunk in chunks:
            ids = list(chunk.encode("utf-8"))
            # Apply every learned merge in priority order.
            for pair, new_id in self.merges.items():
                ids = self._merge_pair(ids, pair, new_id)
            token_ids.extend(ids)
        return token_ids

    # ------------------------------------------------------------------ #
    #  Decoding                                                            #
    # ------------------------------------------------------------------ #

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of *pair* in *ids* with *new_id*."""
        merged: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save merges to a plain-text file so the tokeniser can be reloaded."""
        with open(path, "w") as f:
            f.write(f"bpe v1\n")
            for (a, b), new_id in self.merges.items():
                f.write(f"{a} {b} {new_id}\n")

    def load(self, path: str) -> None:
        """Load merges from a file previously written by :meth:`save`."""
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        with open(path) as f:
            header = f.readline().strip()
            assert header == "bpe v1", f"unexpected header: {header!r}"
            for line in f:
                a, b, new_id = map(int, line.split())
                self.merges[(a, b)] = new_id
                self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"BPETokeniser(vocab_size={len(self.vocab)}, merges={len(self.merges)})"

    def token_to_str(self, token_id: int) -> str:
        """Return a human-readable representation of a single token."""
        return self.vocab[token_id].decode("utf-8", errors="replace")


# ====================================================================== #
#  Demo                                                                    #
# ====================================================================== #

if __name__ == "__main__":
    tok = BPETokeniser()

    # ── Collect training text ───────────────────────────────────
    print("═" * 60)
    print("  BPE Tokeniser — Training")
    print("═" * 60)
    print("Enter/paste your training text below.")
    print("(Type a blank line to finish)\n")

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)

    training_text = "\n".join(lines)

    if not training_text.strip():
        print("No training text provided — exiting.")
        raise SystemExit(1)

    # ── Vocab size ──────────────────────────────────────────────
    raw = input("\nVocab size (default 300): ").strip()
    vocab_size = int(raw) if raw else 300

    # ── Train ───────────────────────────────────────────────────
    print(f"\nTraining on {len(training_text)} chars, target vocab_size={vocab_size} …\n")
    tok.train(training_text, vocab_size=vocab_size, verbose=True)
    print(f"\n{tok}\n")

    # ── Show training text as token string ──────────────────────
    training_ids = tok.encode(training_text)
    token_str = "|".join(f"[{tok.token_to_str(tid)}:{tid}]" for tid in training_ids)
    print("═" * 60)
    print("  Tokenised training text")
    print("═" * 60)
    print(f"  {token_str}")
    print(f"  ({len(training_ids)} tokens)\n")

    # ── Encode / Decode loop ────────────────────────────────────
    print("═" * 60)
    print("  Encode / Decode")
    print("═" * 60)
    print("Enter text to tokenise (blank line to quit):\n")

    while True:
        try:
            test = input(">>> ")
        except EOFError:
            break
        if not test:
            break

        encoded = tok.encode(test)
        decoded = tok.decode(encoded)

        print(f"  Tokens ({len(encoded)}): {encoded}")
        print(f"  Decoded: {decoded!r}")
        print(f"  Round-trip OK: {test == decoded}")
        print("  Token details:")
        for tid in encoded:
            print(f"    {tid:>5d}  →  {tok.token_to_str(tid)!r}")

        # Show the full input segmented into its token pieces.
        token_str = "|".join(f"[{tok.token_to_str(tid)}]" for tid in encoded)
        print(f"  Token string: {token_str}")
        print()
