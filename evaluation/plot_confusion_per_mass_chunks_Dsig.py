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
    help="Dataset split"
)

parser.add_argument(
    "--cut",
    type=float,
    default=None,
    help="Optional D_sig cut (e.g. 0.7)"
)

args = parser.parse_args()

MASS = args.mass
SPLIT = args.split
CUT = args.cut

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

INFILE = f"{EOS_BASE}/models/validation_diagnostics_m{MASS}_{SPLIT}.npz"

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

required_keys = [
    "y_true",
    "y_pred",
    "P_VBF",
    "P_ggF",
    "P_bkg"
]

for k in required_keys:
    if k not in data:
        raise KeyError(f"Missing key '{k}' in {INFILE}")

y_true = data["y_true"]
y_pred = data["y_pred"]
P_VBF  = data["P_VBF"]
P_ggF  = data["P_ggF"]

print(f"Loaded {len(y_true)} events")

# =====================================================
# APPLY D_SIG CUT
# =====================================================

if CUT is not None:

    D_sig = P_VBF + P_ggF

    mask = D_sig > CUT

    print(f"Applying D_sig > {CUT}")
    print(f"Events passing cut: {mask.sum()} / {len(mask)}")

    y_true = y_true[mask]
    y_pred = y_pred[mask]

else:
    print("No D_sig cut applied")

# =====================================================
# CLASS NAMES
# =====================================================

class_names = ["VBF", "ggF", "Background"]

# =====================================================
# OUTPUT NAME
# =====================================================

if CUT is None:
    outname = f"{OUTDIR}/confusion_matrix_{SPLIT}_m{MASS}.png"
else:
    outname = f"{OUTDIR}/confusion_matrix_{SPLIT}_m{MASS}_Dsig{CUT}.png"

# =====================================================
# PLOT
# =====================================================

plot_confusion(
    y_true,
    y_pred,
    class_names,
    title=f"Confusion Matrix ({SPLIT}, mH = {MASS} GeV, D_sig>{CUT})",
    outname=outname,
    normalize="true"
)

print(f"Saved {outname}")
