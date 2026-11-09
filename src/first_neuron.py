"""
Kourosh - First Neuron
=======================
A single artificial neuron learning the AND logic gate from scratch.
Demonstrates the full learning cycle:
    forward pass -> loss -> backpropagation -> gradient descent update
No PyTorch here: pure NumPy, so you can see every number.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SEED, CHECKPOINTS_DIR
from src.utils import set_seed


# ---------- 1. Data: AND gate ----------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)


# ---------- 2. Initialize weights ----------
set_seed(SEED)
n_inputs = X.shape[1]
W = np.random.randn(n_inputs, 1) * 0.1
b = np.zeros((1, 1))


# ---------- 3. Helper functions ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true, y_pred):
    eps = 1e-8
    return -np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    )


# ---------- 4. Training loop ----------
EPOCHS = 5000
LR = 0.5
losses = []

for epoch in range(EPOCHS):
    z = X @ W + b
    y_pred = sigmoid(z)
    loss = bce_loss(y, y_pred)
    losses.append(loss)

    dz = y_pred - y
    dW = X.T @ dz / X.shape[0]
    db = np.mean(dz, axis=0, keepdims=True)

    W -= LR * dW
    b -= LR * db


# ---------- 5. Report ----------
print("=" * 50)
print("Kourosh - First Neuron (AND gate)")
print("=" * 50)
print(f"Final loss: {losses[-1]:.6f}")
print(f"W = {W.ravel()}")
print(f"b = {b.ravel()}")
print()
print("Predictions:")
for xi, yi, pi in zip(X, y, y_pred):
    mark = "OK " if (pi[0] > 0.5) == (yi[0] > 0.5) else "ERR"
    print(f"  {mark} input={xi.tolist()}  target={int(yi[0])}  pred={pi[0]:.4f}")


# ---------- 6. Save loss curve ----------
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = CHECKPOINTS_DIR / "first_neuron_loss.png"

plt.figure(figsize=(9, 4))
plt.plot(losses, color="#c8102e", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Kourosh - First Neuron Training (AND gate)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_path, dpi=120)
plt.close()

print()
print(f"Loss curve saved to: {out_path}")