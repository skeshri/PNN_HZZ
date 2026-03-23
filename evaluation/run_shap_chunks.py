import os
import glob
import argparse
import numpy as np
import torch
import shap

from training.model import PNN
from evaluation.shap_wrapper import PNNWrapper

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="Run SHAP on PNN (robust split-aware)")
parser.add_argument("--mass", type=int, required=True)
parser.add_argument("--nsamples", type=int, default=2000)
parser.add_argument("--nbackground", type=int, default=2000)
args = parser.parse_args()

MASS = float(args.mass)
MASS_NORM = (MASS - 1000.0) / 500.0

NSAMPLES = args.nsamples
NBACKGROUND = args.nbackground

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

#MODEL_FILE = f"{EOS_BASE}/models/pnn_hzz_streaming.pt"
MODEL_PATH = f"{EOS_BASE}/models/checkpoint_epoch_7.pt"
NORM_PATH  = "preprocessing/norm.npz"

VAL_DIRS = [
    f"{EOS_BASE}/chunks/ggF/val",
    f"{EOS_BASE}/chunks/VBF/val",
    f"{EOS_BASE}/chunks/background/DY/val",
    f"{EOS_BASE}/chunks/background/TOP/val",
    f"{EOS_BASE}/chunks/background/VV/val",
    f"{EOS_BASE}/chunks/background/RARE/val",
]

BKG_VAL_DIRS = [
    f"{EOS_BASE}/chunks/background/DY/val",
    f"{EOS_BASE}/chunks/background/TOP/val",
    f"{EOS_BASE}/chunks/background/VV/val",
    f"{EOS_BASE}/chunks/background/RARE/val",
]

OUT_FILE = f"{EOS_BASE}/models/shap_values_m{int(MASS)}.npz"

# =====================================================
# LOAD NORMALIZATION
# =====================================================

norm = np.load(NORM_PATH)
mean = torch.tensor(norm["mean"], dtype=torch.float32, device=DEVICE)
std  = torch.tensor(norm["std"],  dtype=torch.float32, device=DEVICE)

# =====================================================
# COLLECT VALIDATION FILES
# =====================================================

val_files = []
for d in VAL_DIRS:
    val_files.extend(sorted(glob.glob(f"{d}/chunk_*.npz")))

if len(val_files) == 0:
    raise RuntimeError("No validation chunk files found.")

print(f"Found {len(val_files)} validation chunk files")

rng = np.random.default_rng(12345)
rng.shuffle(val_files)

# =====================================================
# COUNT TOTAL VALIDATION EVENTS
# =====================================================

total_val_events = 0
for f in val_files:
    data = np.load(f)
    total_val_events += len(data["X"])

if total_val_events < NSAMPLES:
    print(f"Warning: only {total_val_events} validation events available.")
    print(f"Using {total_val_events} instead of requested {NSAMPLES}.")
    NSAMPLES = total_val_events

# =====================================================
# SAMPLE EVENTS TO EXPLAIN
# =====================================================

X_explain_list = []
collected = 0

for f in val_files:
    if collected >= NSAMPLES:
        break

    data = np.load(f)
    X_chunk = data["X"]

    remaining = NSAMPLES - collected
    take = min(len(X_chunk), remaining)

    idx = rng.choice(len(X_chunk), size=take, replace=False)
    X_explain_list.append(X_chunk[idx])
    collected += take

X_explain_np = np.concatenate(X_explain_list)

X_explain = torch.tensor(
    X_explain_np,
    dtype=torch.float32,
    device=DEVICE
)

X_explain = (X_explain - mean) / std

print(f"Events to explain: {len(X_explain)}")

# =====================================================
# COUNT AVAILABLE BACKGROUND EVENTS (FIRST PASS)
# =====================================================

bkg_files = []
for d in BKG_VAL_DIRS:
    bkg_files.extend(sorted(glob.glob(f"{d}/chunk_*.npz")))

if len(bkg_files) == 0:
    raise RuntimeError("No background validation chunk files found")

total_background_events_found = 0

for f in bkg_files:
    data = np.load(f)
    y = data["y"]
    total_background_events_found += np.sum(y == 2)

available_bkg = total_background_events_found

if available_bkg == 0:
    raise RuntimeError("Zero background events available for SHAP.")

if available_bkg < NBACKGROUND:
    print(f"Warning: only {available_bkg} background events available.")
    print(f"Using {available_bkg} instead of requested {NBACKGROUND}.")
    NBACKGROUND = available_bkg

# =====================================================
# SAMPLE BACKGROUND EVENTS (SECOND PASS)
# =====================================================

rng.shuffle(bkg_files)

X_bkg_list = []
collected_bkg = 0

for f in bkg_files:
    if collected_bkg >= NBACKGROUND:
        break

    data = np.load(f)
    X_chunk = data["X"]
    y_chunk = data["y"]

    mask = (y_chunk == 2)
    if not np.any(mask):
        continue

    X_bkg_chunk = X_chunk[mask]

    remaining = NBACKGROUND - collected_bkg
    take = min(len(X_bkg_chunk), remaining)

    idx = rng.choice(len(X_bkg_chunk), size=take, replace=False)
    X_bkg_list.append(X_bkg_chunk[idx])
    collected_bkg += take

X_bkg_np = np.concatenate(X_bkg_list)

X_bkg = torch.tensor(
    X_bkg_np,
    dtype=torch.float32,
    device=DEVICE
)

X_bkg = (X_bkg - mean) / std

print(f"Background reference size: {len(X_bkg)}")

# =====================================================
# LOAD MODEL
# =====================================================

model = PNN(n_features=X_explain.shape[1]).to(DEVICE)
#model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
#model.eval()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()
wrapped_model = PNNWrapper(model, MASS_NORM, DEVICE)

# =====================================================
# SHAP EXPLAINER
# =====================================================

explainer = shap.GradientExplainer(wrapped_model, X_bkg)

shap_values = explainer.shap_values(X_explain)
shap_values = np.array(shap_values)

print("DEBUG: shap_values.shape =", shap_values.shape)

if shap_values.ndim != 3:
    raise RuntimeError(f"Unexpected SHAP shape: {shap_values.shape}")

shap_vbf = shap_values[:, :, 0]
shap_ggf = shap_values[:, :, 1]
shap_bkg = shap_values[:, :, 2]

# =====================================================
# SAVE OUTPUT
# =====================================================

np.savez(
    OUT_FILE,
    shap_vbf=shap_vbf,
    shap_ggf=shap_ggf,
    shap_bkg=shap_bkg,
    X=X_explain.cpu().numpy()
)

print("==========================================")
print(f" SHAP completed for mH = {int(MASS)} GeV")
print(f" Explained events     : {len(X_explain)}")
print(f" Background reference : {len(X_bkg)}")
print(f" Output saved to      : {OUT_FILE}")
print("==========================================")

