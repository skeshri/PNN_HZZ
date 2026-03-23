import numpy as np
import glob
import os

# =====================================================
# CONFIG
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

TRAIN_CHUNK_DIRS = [
    f"{EOS_BASE}/chunks/ggF/train",
    f"{EOS_BASE}/chunks/VBF/train",
    f"{EOS_BASE}/chunks/background/DY/train",
    f"{EOS_BASE}/chunks/background/TOP/train",
    f"{EOS_BASE}/chunks/background/VV/train",
    f"{EOS_BASE}/chunks/background/RARE/train",
]

MAX_EVENTS = 2_000_000   # cap for stability

# =====================================================
# INITIALIZE
# =====================================================

events_seen = 0
sum_x = None
sum_x2 = None
n_features = None

print("==============================================")
print(" Computing normalization (TRAIN split only) ")
print("==============================================")

# =====================================================
# LOOP OVER TRAIN CHUNKS
# =====================================================

for d in TRAIN_CHUNK_DIRS:

    if not os.path.exists(d):
        print(f"WARNING: directory missing → {d}")
        continue

    files = sorted(glob.glob(f"{d}/chunk_*.npz"))

    print(f"Scanning {d} → {len(files)} files")

    for f in files:

        with np.load(f) as data:
            X = data["X"]

            if sum_x is None:
                n_features = X.shape[1]
                sum_x = np.zeros(n_features, dtype=np.float64)
                sum_x2 = np.zeros(n_features, dtype=np.float64)

                print(f"Detected {n_features} input features")

            # Safety check
            if X.shape[1] != n_features:
                raise RuntimeError(
                    f"Feature mismatch in {f} "
                    f"({X.shape[1]} vs {n_features})"
                )

            sum_x += X.sum(axis=0)
            sum_x2 += (X ** 2).sum(axis=0)
            events_seen += len(X)

        if events_seen >= MAX_EVENTS:
            break

    if events_seen >= MAX_EVENTS:
        break

# =====================================================
# FINALIZE
# =====================================================

if events_seen == 0:
    raise RuntimeError("No training events found for normalization.")

mean = sum_x / events_seen
var = sum_x2 / events_seen - mean**2
std = np.sqrt(np.maximum(var, 1e-12))

# =====================================================
# SAVE
# =====================================================

os.makedirs("preprocessing", exist_ok=True)

out = "preprocessing/norm.npz"
np.savez(
    out,
    mean=mean.astype(np.float32),
    std=std.astype(np.float32),
)

print("==============================================")
print(f"Saved normalization to {out}")
print(f"Used {events_seen} training events")
print(f"Number of features = {n_features}")
print("==============================================")
