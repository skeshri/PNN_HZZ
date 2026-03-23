import numpy as np
import os
import argparse

from evaluation.confusion_utils import plot_confusion

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="Confusion matrix for one mass")
parser.add_argument(
    "--mass",
    type=int,
    required=True,
    help="Mass hypothesis in GeV (e.g. 500, 800, 1500)"
)
parser.add_argument(
    "--split",
    default="val",
    choices=["train", "val", "test"],
    help="Dataset split (default: val)"
)

args = parser.parse_args()

MASS = args.mass
SPLIT = args.split

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

# Future-proof filename
INFILE = f"{EOS_BASE}/models/validation_diagnostics_m{MASS}_{SPLIT}.npz"

# Backward compatibility (if old naming still used)
if not os.path.exists(INFILE):
    INFILE = f"{EOS_BASE}/models/validation_diagnostics_m{MASS}.npz"

OUTDIR = f"{EOS_BASE}/plots/confusion_m{MASS}"
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

if not os.path.exists(INFILE):
    raise FileNotFoundError(f"Validation file not found: {INFILE}")

data = np.load(INFILE)

required_keys = ["y_true", "y_pred"]
for k in required_keys:
    if k not in data:
        raise KeyError(f"Missing key '{k}' in {INFILE}")

y_true = data["y_true"]
y_pred = data["y_pred"]

print(f"Loaded {len(y_true)} events from {INFILE}")

class_names = ["VBF", "ggF", "Background"]

# =====================================================
# PLOT CONFUSION MATRIX
# =====================================================

outname = f"{OUTDIR}/confusion_matrix_{SPLIT}_m{MASS}.png"

plot_confusion(
    y_true,
    y_pred,
    class_names,
    title=f"Confusion Matrix ({SPLIT}, mH = {MASS} GeV)",
    outname=outname,
    normalize="true"
)

print(f"Saved {outname}")

