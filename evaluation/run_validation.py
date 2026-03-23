import torch
import numpy as np
import glob
import os
import argparse

from training.model import PNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="Streaming PNN validation (split-safe)")
parser.add_argument(
    "--mass",
    type=float,
    required=True,
    help="Mass hypothesis in GeV"
)
parser.add_argument(
    "--split",
    type=str,
    default="val",
    choices=["train", "val", "test"],
    help="Dataset split to evaluate"
)
args = parser.parse_args()

MASS = args.mass
MASS_NORM = (MASS - 1000.0) / 500.0
MASS_TAG = int(MASS)
SPLIT = args.split

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

VAL_DIRS = [
    f"{EOS_BASE}/chunks/ggF/{SPLIT}",
    f"{EOS_BASE}/chunks/VBF/{SPLIT}",
    f"{EOS_BASE}/chunks/background/DY/{SPLIT}",
    f"{EOS_BASE}/chunks/background/TOP/{SPLIT}",
    f"{EOS_BASE}/chunks/background/VV/{SPLIT}",
    f"{EOS_BASE}/chunks/background/RARE/{SPLIT}",
]

#MODEL_PATH = f"{EOS_BASE}/models/pnn_hzz_streaming.pt"
MODEL_PATH = f"{EOS_BASE}/models/checkpoint_epoch_7.pt"
NORM_PATH  = "preprocessing/norm.npz"

OUTDIR = f"{EOS_BASE}/models"
os.makedirs(OUTDIR, exist_ok=True)

OUTFILE = f"{OUTDIR}/validation_diagnostics_m{MASS_TAG}_{SPLIT}.npz"

# =====================================================
# LOAD NORMALIZATION
# =====================================================

norm = np.load(NORM_PATH)
mean = torch.tensor(norm["mean"], dtype=torch.float32).to(DEVICE)
std  = torch.tensor(norm["std"],  dtype=torch.float32).to(DEVICE)

n_features = len(mean)

# =====================================================
# LOAD MODEL
# =====================================================

model = PNN(n_features=n_features).to(DEVICE)
#model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#model.eval()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("==============================================")
print(f" Running validation")
print(f" Mass        : {MASS} GeV")
print(f" Split       : {SPLIT}")
print("==============================================")

# =====================================================
# COLLECT FILES
# =====================================================

all_files = []

for d in VAL_DIRS:
    if not os.path.exists(d):
        print(f"WARNING: directory does not exist: {d}")
        continue

    files = sorted(glob.glob(f"{d}/chunk_*.npz"))
    all_files.extend(files)

if len(all_files) == 0:
    raise RuntimeError(f"No chunk files found for split '{SPLIT}'")

# =====================================================
# FIRST PASS: count total events
# =====================================================

total_events = 0

for f in all_files:
    with np.load(f) as data:
        total_events += len(data["y"])

print(f"Total events in split '{SPLIT}': {total_events}")

if total_events == 0:
    raise RuntimeError("Split contains zero events")

# =====================================================
# Allocate memory-mapped arrays
# =====================================================

tmp_dir = f"{OUTDIR}/tmp_val_m{MASS_TAG}_{SPLIT}"
os.makedirs(tmp_dir, exist_ok=True)

P_VBF = np.memmap(f"{tmp_dir}/P_VBF.dat", dtype="float32", mode="w+", shape=(total_events,))
P_ggF = np.memmap(f"{tmp_dir}/P_ggF.dat", dtype="float32", mode="w+", shape=(total_events,))
P_bkg = np.memmap(f"{tmp_dir}/P_bkg.dat", dtype="float32", mode="w+", shape=(total_events,))
y_true = np.memmap(f"{tmp_dir}/y_true.dat", dtype="int32", mode="w+", shape=(total_events,))
y_pred = np.memmap(f"{tmp_dir}/y_pred.dat", dtype="int32", mode="w+", shape=(total_events,))
mass_arr = np.memmap(f"{tmp_dir}/mass.dat", dtype="float32", mode="w+", shape=(total_events,))

# =====================================================
# SECOND PASS: run inference
# =====================================================

offset = 0

with torch.no_grad():
    for f in all_files:

        with np.load(f) as data:

            X = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)
            y = data["y"]

            # Normalize
            X = (X - mean) / std

            m = torch.full(
                (len(X), 1),
                MASS_NORM,
                dtype=torch.float32,
                device=DEVICE
            )

            logits = model(X, m)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            n = len(y)

            P_VBF[offset:offset+n] = probs[:, 0]
            P_ggF[offset:offset+n] = probs[:, 1]
            P_bkg[offset:offset+n] = probs[:, 2]
            y_true[offset:offset+n] = y
            y_pred[offset:offset+n] = preds
            mass_arr[offset:offset+n] = MASS

            offset += n

            print(f"Processed {offset}/{total_events}")

# Flush
P_VBF.flush()
P_ggF.flush()
P_bkg.flush()
y_true.flush()
y_pred.flush()
mass_arr.flush()

# =====================================================
# FINAL SAVE (compressed)
# =====================================================

np.savez_compressed(
    OUTFILE,
    P_VBF=np.array(P_VBF),
    P_ggF=np.array(P_ggF),
    P_bkg=np.array(P_bkg),
    y_true=np.array(y_true),
    y_pred=np.array(y_pred),
    mass=np.array(mass_arr),
)

print("==============================================")
print(f" Validation saved to {OUTFILE}")
print("==============================================")

