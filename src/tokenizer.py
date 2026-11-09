"""
Kourosh - Character-Level Tokenizer
====================================
Converts Persian text <-> integer token IDs at the character level.

Design:
    - Fixed vocabulary built from a training corpus
    - Four special tokens: <pad>, <unk>, <bos>, <eos>
    - Handles ZWNJ (نیم‌فاصله) and Persian digits/punctuation
    - Round-trip guarantee: decode(encode(text)) == text (for known chars)

Later, Kourosh's Transformer will consume these integer IDs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


# ---------- Special tokens ----------
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class CharTokenizer:
    """Character-level tokenizer for Persian (or any alphabet-based) text."""

    def __init__(self) -> None:
        # token -> id  and  id -> token
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        # Convenient ID shortcuts for special tokens (filled after build/load)
        self.pad_id: int = -1
        self.unk_id: int = -1
        self.bos_id: int = -1
        self.eos_id: int = -1

        self._built = False

    # --------------------------------------------------
    # Construction
    # --------------------------------------------------
    def build(self, text: str | Iterable[str]) -> "CharTokenizer":
        """Build vocabulary from a corpus (single string or iterable of strings)."""
        chars: set[str] = set()
        if isinstance(text, str):
            chars = set(text)
        else:
            for line in text:
                chars.update(line)

        # Always include special tokens first (stable, low IDs)
        tokens = list(SPECIAL_TOKENS) + sorted(chars - set(SPECIAL_TOKENS))

        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]

        self._built = True
        return self

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # --------------------------------------------------
    # Encode / Decode
    # --------------------------------------------------
    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Convert text to a list of token IDs.

        Unknown characters are mapped to <unk>.
        Optionally wraps the sequence with <bos> / <eos>.
        """
        if not self._built:
            raise RuntimeError("Tokenizer is not built. Call .build() or .load() first.")

        ids = [self.token_to_id.get(ch, self.unk_id) for ch in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        """Convert token IDs back to text. Special tokens are skipped by default."""
        if not self._built:
            raise RuntimeError("Tokenizer is not built. Call .build() or .load() first.")

        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        out: list[str] = []
        for i in ids:
            tok = self.id_to_token.get(int(i), UNK_TOKEN)
            if skip_special and int(i) in special_ids:
                continue
            out.append(tok)
        return "".join(out)

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save tokenizer to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id,
            "special_tokens": SPECIAL_TOKENS,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load tokenizer from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        tok = cls()
        tok.token_to_id = {k: int(v) for k, v in payload["token_to_id"].items()}
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        tok.pad_id = tok.token_to_id[PAD_TOKEN]
        tok.unk_id = tok.token_to_id[UNK_TOKEN]
        tok.bos_id = tok.token_to_id[BOS_TOKEN]
        tok.eos_id = tok.token_to_id[EOS_TOKEN]
        tok._built = True
        return tok

    # --------------------------------------------------
    # Debug / info
    # --------------------------------------------------
    def __repr__(self) -> str:
        status = "built" if self._built else "unbuilt"
        return f"CharTokenizer(vocab_size={self.vocab_size}, {status})"


# ==================================================
# Demo: run directly to see it in action
# ==================================================
if __name__ == "__main__":
    from src.config import CHECKPOINTS_DIR

    # A small sample of Persian text (Kourosh will later train on much more)
    corpus = (
        "کوروش بزرگ بنیان‌گذار امپراتوری هخامنشی بود. "
        "او نخستین منشور حقوق بشر را نوشت. "
        "پارسه، پاسارگاد و شوش از شهرهای مهم آن دوران بودند."
    )

    tok = CharTokenizer().build(corpus)
    print("=" * 55)
    print("Kourosh - Character Tokenizer Demo")
    print("=" * 55)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Special IDs: pad={tok.pad_id}, unk={tok.unk_id}, bos={tok.bos_id}, eos={tok.eos_id}")
    print(f"\nVocabulary (first 30): {[tok.id_to_token[i] for i in range(min(30, tok.vocab_size))]}")

    sample = "کوروش بزرگ"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    back = tok.decode(ids)
    print(f"\nSample text:  {sample!r}")
    print(f"Encoded IDs:  {ids}")
    print(f"Decoded back: {back!r}")
    assert back == sample, "Round-trip failed!"
    print("Round-trip:   OK ✅")

    # Unknown character test
    weird = "کوروش 🙂"   # emoji is not in vocab -> <unk>
    weird_ids = tok.encode(weird)
    print(f"\nUnknown test: {weird!r} -> {weird_ids} (last ID = <unk>)")

    # Save / load
    out_path = CHECKPOINTS_DIR / "char_tokenizer.json"
    tok.save(out_path)
    loaded = CharTokenizer.load(out_path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.decode(loaded.encode(sample)) == sample
    print(f"\nSaved & reloaded from: {out_path}")
    print("All checks passed ✅")