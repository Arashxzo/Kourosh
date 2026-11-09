"""
Kourosh - Transformer Architecture (from scratch)
==================================================
A decoder-only Transformer (GPT-style) implemented from scratch in PyTorch.
This is the heart of Kourosh: the module that will eventually learn to
generate Persian text.

Components implemented here:
    1. CausalMultiHeadSelfAttention  — how each token attends to previous ones
    2. FeedForward                   — per-token MLP transformation
    3. TransformerBlock              — Attention + FFN with residual + LayerNorm (pre-norm)
    4. KouroshLM                     — full language model: embeddings + N blocks + LM head

Design choices (aligned with modern LLMs like GPT-2/GPT-3):
    - Pre-norm residual connections (more stable training)
    - GELU activation
    - Learned positional embeddings
    - Weight tying between token embedding and LM head
    - Causal mask (no peeking at future tokens)

Tensor convention: all tensors are batch-first — (batch, seq_len, dim).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# 1. Causal Multi-Head Self-Attention
# ==================================================
class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (lower-triangular) mask.

    Each token can attend only to itself and to earlier tokens — never to
    future tokens. This is what makes an autoregressive language model
    (like GPT or Kourosh) able to generate text left-to-right.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # One fused projection for Q, K, V (efficient + fewer params)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer: (1, 1, T, T), 1 = allowed, 0 = forbidden
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv(x)                    # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)       # each (B, T, C)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, H, T, T)
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v                                     # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.resid_dropout(self.proj(out))


# ==================================================
# 2. Feed-Forward Network (per-token MLP)
# ==================================================
class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation, applied identically to each token."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================================================
# 3. Transformer Block (pre-norm)
# ==================================================
class TransformerBlock(nn.Module):
    """One Transformer block: Attention + FFN, each wrapped in a residual.

    Uses pre-norm (LayerNorm *before* each sublayer) — the standard in modern
    LLMs because it trains more stably at depth than the original post-norm.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int,
                 dropout: float, max_seq_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalMultiHeadSelfAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ==================================================
# 4. Full Language Model
# ==================================================
class KouroshLM(nn.Module):
    """Kourosh: a decoder-only Transformer language model.

    Forward returns raw logits (B, T, vocab_size).
    If `targets` is provided, also returns the cross-entropy loss
    (next-token prediction), ignoring positions where targets == ignore_index.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share embedding and output projection matrices
        # (saves params, improves generalization — GPT-2 style)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation: sample one token at a time."""
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = topk_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < threshold, float("-inf"), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if eos_id is not None and (next_id == eos_id).all():
                break
        return input_ids


# ==================================================
# Demo: verify architecture
# ==================================================
if __name__ == "__main__":
    from src.config import DEVICE, SEED
    from src.utils import set_seed

    print("=" * 55)
    print("Kourosh - Transformer Architecture Test")
    print("=" * 55)

    set_seed(SEED)

    # Small model for a quick architecture check
    model = KouroshLM(
        vocab_size=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        max_seq_len=32,
        dropout=0.0,
    ).to(DEVICE)

    print(f"Model parameters: {model.num_parameters():,}")

    # Forward pass (inference)
    input_ids = torch.randint(0, 64, (2, 16), device=DEVICE)
    logits, _ = model(input_ids)
    print(f"Input shape:  {tuple(input_ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}  (expected: (2, 16, 64))")
    assert logits.shape == (2, 16, 64)

    # Forward pass (with targets -> loss)
    targets = torch.randint(0, 64, (2, 16), device=DEVICE)
    _, loss = model(input_ids, targets=targets)
    print(f"Loss (random targets): {loss.item():.4f}  (≈ ln(64) = {math.log(64):.4f})")

    # Causal-mask test: changing a future token must NOT affect earlier predictions
    x1 = torch.tensor([[5, 6, 7, 8, 9]], device=DEVICE)
    x2 = torch.tensor([[5, 6, 7, 8, 3]], device=DEVICE)  # only last token differs
    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)
    # First 4 positions must be identical
    diff = (logits1[:, :4] - logits2[:, :4]).abs().max().item()
    print(f"Causal mask diff (should be 0): {diff:.2e}")
    assert diff < 1e-6, "Causal mask is leaking future information!"

    # Generation test (untrained model produces random tokens — that's fine)
    prompt = torch.tensor([[1, 2, 3]], device=DEVICE)
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=10)
    print(f"Generate: {prompt.shape} -> {out.shape}  (expected: (1, 8))")
    assert out.shape == (1, 8)

    print("\nAll architecture checks passed ✅")
    print("Kourosh's brain is wired. Next: train it on real Persian text.")