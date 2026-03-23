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

parser = argparse.ArgumentParser(description="Streaming PNN inference")
parser.add_argument(
    "--mass",
    type=float,
    required=True,
    help="Mass hypothesis in GeV (e.g. 500, 800, 1500)"
)
parser.add_argument(
    "--split",
    type=str,
    default="test",
    choices=["train", "val", "test"],
    help="Dataset split to run inference on"
)
args = parser.parse_args()

MASS = args.mass
MASS_NORM = (MASS - 1000.0) / 500.0
SPLIT = args.split

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

CHUNK_DIRS = [
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

OUTDIR = f"{EOS_BASE}/evaluation"
os.makedirs(OUTDIR, exist_ok=True)

OUTFILE = f"{OUTDIR}/pnn_output_m{int(MASS)}_{SPLIT}.npz"

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
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()

#model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#model.eval()

print(f"Running inference:")
print(f"  Mass = {MASS} GeV")
print(f"  Split = {SPLIT}")

# =====================================================
# COLLECT FILES
# =====================================================

all_files = []

for d in CHUNK_DIRS:
    files = sorted(glob.glob(f"{d}/chunk_*.npz"))
    all_files.extend(files)

if len(all_files) == 0:
    raise RuntimeError("No chunk files found for inference.")

print(f"Found {len(all_files)} chunk files")

# =====================================================
# FIRST PASS: count total events
# =====================================================

total_events = 0
for f in all_files:
    data = np.load(f)
    total_events += len(data["y"])

print(f"Total events to process: {total_events}")

# =====================================================
# Allocate memmaps
# =====================================================

tmp_dir = f"{OUTDIR}/tmp_inference_m{int(MASS)}_{SPLIT}"
os.makedirs(tmp_dir, exist_ok=True)

P_VBF = np.memmap(f"{tmp_dir}/P_VBF.dat", dtype="float32", mode="w+", shape=(total_events,))
P_ggF = np.memmap(f"{tmp_dir}/P_ggF.dat", dtype="float32", mode="w+", shape=(total_events,))
P_bkg = np.memmap(f"{tmp_dir}/P_bkg.dat", dtype="float32", mode="w+", shape=(total_events,))
labels = np.memmap(f"{tmp_dir}/labels.dat", dtype="int32", mode="w+", shape=(total_events,))

# =====================================================
# INFERENCE LOOP
# =====================================================

offset = 0

with torch.no_grad():
    for f in all_files:

        data = np.load(f)

        X = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)
        y = data["y"]

        X = (X - mean) / std

        m = torch.full(
            (len(X), 1),
            MASS_NORM,
            dtype=torch.float32,
            device=DEVICE
        )

        logits = model(X, m)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        n = len(y)

        P_VBF[offset:offset+n] = probs[:, 0]
        P_ggF[offset:offset+n] = probs[:, 1]
        P_bkg[offset:offset+n] = probs[:, 2]
        labels[offset:offset+n] = y

        offset += n
        print(f"Processed {offset}/{total_events}")

# Flush to disk
P_VBF.flush()
P_ggF.flush()
P_bkg.flush()
labels.flush()

# =====================================================
# FINAL SAVE (COMPRESSED)
# =====================================================

np.savez_compressed(
    OUTFILE,
    P_VBF=np.array(P_VBF),
    P_ggF=np.array(P_ggF),
    P_bkg=np.array(P_bkg),
    label=np.array(labels),
)

print(f"\n✅ Inference saved to {OUTFILE}")
print(f"Total events processed: {total_events}")

