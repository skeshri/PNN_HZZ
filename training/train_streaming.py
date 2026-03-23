import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
from torch.utils.data import DataLoader

from training.model import PNN
from training.chunk_dataset import BalancedChunkIterableDataset

# =====================================================
# CONFIGURATION
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"
MODEL_DIR = f"{EOS_BASE}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# TRAIN SPLIT
# -------------------------

TRAIN_CHUNK_DIRS = {
    0: [f"{EOS_BASE}/chunks/VBF/train/"],
    1: [f"{EOS_BASE}/chunks/ggF/train/"],
    2: [
        f"{EOS_BASE}/chunks/background/DY/train/",
        f"{EOS_BASE}/chunks/background/TOP/train/",
        f"{EOS_BASE}/chunks/background/RARE/train/",
        f"{EOS_BASE}/chunks/background/VV/train/",
    ],
}

# -------------------------
# VALIDATION SPLIT
# -------------------------

VAL_CHUNK_DIRS = {
    0: [f"{EOS_BASE}/chunks/VBF/val/"],
    1: [f"{EOS_BASE}/chunks/ggF/val/"],
    2: [
        f"{EOS_BASE}/chunks/background/DY/val/",
        f"{EOS_BASE}/chunks/background/TOP/val/",
        f"{EOS_BASE}/chunks/background/RARE/val/",
        f"{EOS_BASE}/chunks/background/VV/val/",
    ],
}

# -------------------------
# Hyperparameters
# -------------------------

BATCH_SIZE = 512
EPOCHS = 25
STEPS_PER_EPOCH = 3000
VAL_STEPS = 1000

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 0.7])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD NORMALIZATION
# =====================================================

norm = np.load("preprocessing/norm.npz")

mean = torch.tensor(norm["mean"], dtype=torch.float32).to(DEVICE)
std  = torch.tensor(norm["std"], dtype=torch.float32).to(DEVICE)

n_features = len(mean)

print("==============================================")
print("Training PNN")
print(f"Device: {DEVICE}")
print(f"Features: {n_features}")
print("==============================================")

# =====================================================
# MODEL
# =====================================================

model = PNN(n_features=n_features).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=CLASS_WEIGHTS.to(DEVICE)
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# =====================================================
# DATASETS
# =====================================================

train_dataset = BalancedChunkIterableDataset(
    chunk_dirs=TRAIN_CHUNK_DIRS,
    batch_size=BATCH_SIZE,
    shuffle_chunks=True
)

val_dataset = BalancedChunkIterableDataset(
    chunk_dirs=VAL_CHUNK_DIRS,
    batch_size=BATCH_SIZE,
    shuffle_chunks=False
)

train_loader = DataLoader(train_dataset, batch_size=None, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=None)

# =====================================================
# CHECKPOINT HANDLING
# =====================================================

def find_latest_checkpoint():

    ckpts = glob.glob(f"{MODEL_DIR}/checkpoint_epoch_*.pt")

    if len(ckpts) == 0:
        return None

    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    return ckpts[-1]


train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

start_epoch = 0

latest_ckpt = find_latest_checkpoint()

if latest_ckpt is not None:

    print(f"Resuming from checkpoint: {latest_ckpt}")

    ckpt = torch.load(latest_ckpt, map_location=DEVICE)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    train_loss_hist = ckpt["train_loss"]
    train_acc_hist = ckpt["train_acc"]
    val_loss_hist = ckpt["val_loss"]
    val_acc_hist = ckpt["val_acc"]

    start_epoch = ckpt["epoch"]

    print(f"Resuming from epoch {start_epoch}")

# =====================================================
# TRAINING LOOP
# =====================================================

for epoch in range(start_epoch, EPOCHS):

    # ------------------------------
    # TRAIN
    # ------------------------------

    model.train()

    running_loss = 0
    correct = 0
    total = 0
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for step, (X, m, y) in enumerate(train_loader):

        if step >= STEPS_PER_EPOCH:
            break

        X = X.to(DEVICE)
        m = m.to(DEVICE).unsqueeze(1)
        y = y.to(DEVICE)

        X = (X - mean) / std

        #logits = model(X, m)

        #loss = criterion(logits, y)

        #loss.backward()

        #optimizer.step()
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(X, m)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / STEPS_PER_EPOCH
    train_acc = correct / total

    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)

    # ------------------------------
    # VALIDATION
    # ------------------------------

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for step, (X, m, y) in enumerate(val_loader):

            if step >= VAL_STEPS:
                break

            X = X.to(DEVICE)
            m = m.to(DEVICE).unsqueeze(1)
            y = y.to(DEVICE)

            X = (X - mean) / std

            logits = model(X, m)

            loss = criterion(logits, y)

            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_loss /= VAL_STEPS
    val_acc = val_correct / val_total

    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
    )

    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================

    checkpoint_path = f"{MODEL_DIR}/checkpoint_epoch_{epoch+1}.pt"

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss_hist,
            "train_acc": train_acc_hist,
            "val_loss": val_loss_hist,
            "val_acc": val_acc_hist,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved → {checkpoint_path}")

    # also save curves continuously

    np.savez(
        f"{MODEL_DIR}/training_curves.npz",
        train_loss=train_loss_hist,
        train_acc=train_acc_hist,
        val_loss=val_loss_hist,
        val_acc=val_acc_hist,
    )

# =====================================================
# FINAL MODEL
# =====================================================

torch.save(
    model.state_dict(),
    f"{MODEL_DIR}/pnn_hzz_streaming_balanced.pt",
)

print("\n==============================================")
print("Training completed successfully")
print("Final model saved")
print("==============================================")
