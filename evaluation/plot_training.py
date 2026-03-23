import numpy as np
import matplotlib.pyplot as plt
import os

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

CURVE_FILE = f"{EOS_BASE}/models/training_curves.npz"
PLOT_DIR = f"{EOS_BASE}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

if not os.path.exists(CURVE_FILE):
    raise FileNotFoundError(f"Training curve file not found: {CURVE_FILE}")

data = np.load(CURVE_FILE)

required = ["train_loss", "val_loss", "train_acc", "val_acc"]
for k in required:
    if k not in data:
        raise KeyError(f"Missing key '{k}' in training_curves.npz")

epochs = np.arange(1, len(data["train_loss"]) + 1)

# =============================
# Loss Curve
# =============================

plt.figure(figsize=(7,5))
plt.plot(epochs, data["train_loss"], marker="o", label="Train")
plt.plot(epochs, data["val_loss"], marker="s", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PNN Training / Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/loss_curve.png")
plt.close()

# =============================
# Accuracy Curve
# =============================

plt.figure(figsize=(7,5))
plt.plot(epochs, data["train_acc"], marker="o", label="Train")
plt.plot(epochs, data["val_acc"], marker="s", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("PNN Training / Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/accuracy_curve.png")
plt.close()

print("Saved training curves.")

