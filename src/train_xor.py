"""
Kourosh - XOR Training
=======================
Train an MLP to solve XOR: a classic non-linearly-separable problem that a
single neuron cannot solve. Demonstrates:
    - nn.Module usage (model.py)
    - PyTorch training loop (zero_grad -> backward -> step)
    - Working on DEVICE (cpu/cuda)
    - Logging, checkpointing, and visualization
"""
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import SEED, DEVICE, CHECKPOINTS_DIR
from src.utils import set_seed
from src.model import MLP


# ---------- Data: XOR ----------
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

X, y = X.to(DEVICE), y.to(DEVICE)


def main() -> None:
    set_seed(SEED)

    # ---------- Model, optimizer, loss ----------
    model = MLP(input_dim=2, hidden_dim=16, output_dim=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCELoss()

    EPOCHS = 5000
    losses = []

    # ---------- Training loop ----------
    pbar = tqdm(range(EPOCHS), desc="Training XOR", ncols=80)
    for epoch in pbar:
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    # ---------- Evaluation ----------
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu()

    print("\n" + "=" * 50)
    print("Kourosh - MLP XOR Training")
    print("=" * 50)
    print(f"Final loss: {losses[-1]:.6f}\n")
    print("Predictions:")
    correct = 0
    for xi, yi, pi in zip(X.cpu(), y.cpu(), preds):
        is_ok = (pi.item() > 0.5) == (yi.item() > 0.5)
        mark = "OK " if is_ok else "ERR"
        if is_ok:
            correct += 1
        print(f"  {mark} input={xi.tolist()}  target={int(yi.item())}  pred={pi.item():.4f}")
    print(f"\nCorrect: {correct}/4")

    # ---------- Save loss curve ----------
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINTS_DIR / "mlp_xor_loss.png"
    plt.figure(figsize=(9, 4))
    plt.plot(losses, color="#0033a0", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss (log scale)")
    plt.title("Kourosh - MLP XOR Training")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nLoss curve: {out_path}")

    # ---------- Save model checkpoint ----------
    ckpt_path = CHECKPOINTS_DIR / "mlp_xor.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()