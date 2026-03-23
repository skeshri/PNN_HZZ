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

parser = argparse.ArgumentParser(description="PNN inference (DATA)")
parser.add_argument("--mass", type=float, required=True)
args = parser.parse_args()

MASS = args.mass
MASS_NORM = (MASS - 1000.0) / 500.0

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

CHUNK_DIR = f"{EOS_BASE}/chunks_data"

MODEL_PATH = f"{EOS_BASE}/models/checkpoint_epoch_7.pt"
NORM_PATH  = "preprocessing/norm.npz"

OUTDIR = f"{EOS_BASE}/evaluation_data"
os.makedirs(OUTDIR, exist_ok=True)

OUTFILE = f"{OUTDIR}/pnn_data_m{int(MASS)}.npz"

# =====================================================
# LOAD NORMALIZATION
# =====================================================

norm = np.load(NORM_PATH)
mean = torch.tensor(norm["mean"], dtype=torch.float32).to(DEVICE)
std  = torch.tensor(norm["std"],  dtype=torch.float32).to(DEVICE)

# =====================================================
# LOAD MODEL
# =====================================================

model = PNN(n_features=len(mean)).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(
    checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
)
model.eval()

# =====================================================
# FILES
# =====================================================

files = sorted(glob.glob(f"{CHUNK_DIR}/chunk_*.npz"))

if len(files) == 0:
    raise RuntimeError("No data chunks found")

print(f"Found {len(files)} files")

# =====================================================
# INFERENCE
# =====================================================

all_P_VBF, all_P_ggF, all_P_bkg = [], [], []

with torch.no_grad():
    for f in files:

        data = np.load(f)
        X = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)

        # Normalize (VERY IMPORTANT)
        X = (X - mean) / std

        m = torch.full((len(X), 1), MASS_NORM, device=DEVICE)

        probs = torch.softmax(model(X, m), dim=1).cpu().numpy()

        all_P_VBF.append(probs[:, 0])
        all_P_ggF.append(probs[:, 1])
        all_P_bkg.append(probs[:, 2])

        print(f"Processed {len(X)} events")

# =====================================================
# SAVE
# =====================================================

np.savez_compressed(
    OUTFILE,
    P_VBF=np.concatenate(all_P_VBF),
    P_ggF=np.concatenate(all_P_ggF),
    P_bkg=np.concatenate(all_P_bkg),
)

print(f"\n✅ Saved: {OUTFILE}")
