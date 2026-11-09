"""
Kourosh - Training Loop (Fast CPU Version)
==========================================
Bypasses PyTorch DataLoader for lightning-fast random batching.
Tuned to finish in minutes, not hours.
"""
import time

import torch
from tqdm import tqdm

from src.config import SEED, DEVICE, CHECKPOINTS_DIR, RAW_DATA_DIR
from src.utils import set_seed
from src.tokenizer import CharTokenizer
from src.transformer import KouroshLM


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    """Sample a random batch of (x, y) from a 1D tensor of token IDs."""
    # Pick random starting indices
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    # Stack the sequences
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    set_seed(SEED)

    # ---------- Load corpus ----------
    text = (RAW_DATA_DIR / "corpus.txt").read_text(encoding="utf-8")
    print(f"Corpus: {len(text):,} characters ({len(text)/1e6:.2f} MB)")

    tokenizer = CharTokenizer().build(text)
    ids = tokenizer.encode(text)
    print(f"Vocab size: {tokenizer.vocab_size}  |  Tokens: {len(ids):,}")

    split = int(len(ids) * 0.95)
    train_ids, val_ids = ids[:split], ids[split:]

    # Convert to tensors ONCE (this is the key optimization)
    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long)

    # ---------- Hyperparameters (Optimized for CPU) ----------
    BLOCK_SIZE = 128
    BATCH_SIZE = 64
    EPOCHS = 4
    MAX_STEPS_PER_EPOCH = 400
    MAX_VAL_STEPS = 50

    # Slightly smaller model so CPU calculates faster
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    HIDDEN_DIM = 512
    DROPOUT = 0.1
    LR = 5e-4
    WEIGHT_DECAY = 0.05

    # ---------- Model ----------
    model = KouroshLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        max_seq_len=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = MAX_STEPS_PER_EPOCH * EPOCHS
    warmup_steps = max(50, total_steps // 20)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------- Training ----------
    best_val = float("inf")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "max_seq_len": BLOCK_SIZE,
    }

    print("\n" + "=" * 65)
    print(f"Training: {EPOCHS} epochs, {MAX_STEPS_PER_EPOCH} steps/epoch")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        pbar = tqdm(range(MAX_STEPS_PER_EPOCH), desc=f"Epoch {epoch}/{EPOCHS}", ncols=90, leave=False)
        for step in pbar:
            x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            
            _, loss = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
            
        train_loss /= MAX_STEPS_PER_EPOCH

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(MAX_VAL_STEPS):
                x, y = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
                _, loss = model(x, targets=y)
                val_loss += loss.item()
        val_loss /= MAX_VAL_STEPS

        dt = time.time() - t0
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "config": config},
                CHECKPOINTS_DIR / "kourosh.pt",
            )
            tokenizer.save(CHECKPOINTS_DIR / "kourosh_tokenizer.json")
            improved = "  ✓ saved"

        print(f"Epoch {epoch}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f} | {dt/60:.1f} min{improved}")

    print("\n" + "=" * 65)
    print(f"Best val loss: {best_val:.4f}")
    print(f"Model:     {CHECKPOINTS_DIR / 'kourosh.pt'}")
    print("=" * 65)


if __name__ == "__main__":
    main()