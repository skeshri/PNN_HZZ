import numpy as np
import os
import argparse

from evaluation.confusion_utils import plot_confusion

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(
    description="Confusion matrix after D_sig cut"
)
parser.add_argument(
    "--mass",
    type=int,
    required=True,
    help="Mass hypothesis in GeV (e.g. 500, 800, 1500)"
)
parser.add_argument(
    "--cut",
    type=float,
    default=0.7,
    help="Cut value on D_sig = P_VBF + P_ggF (default: 0.7)"
)
args = parser.parse_args()

MASS = args.mass
CUT  = args.cut

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

INFILE = f"{EOS_BASE}/models/validation_diagnostics_m{MASS}.npz"
OUTDIR = f"{EOS_BASE}/plots/confusion_m{MASS}"
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

if not os.path.exists(INFILE):
    raise FileNotFoundError(f"Validation file not found: {INFILE}")

data = np.load(INFILE)

required_keys = ["y_true", "y_pred", "P_VBF", "P_ggF"]
for k in required_keys:
    if k not in data:
        raise KeyError(f"Missing key '{k}' in {INFILE}")

y_true = data["y_true"]
y_pred = data["y_pred"]
P_VBF  = data["P_VBF"]
P_ggF  = data["P_ggF"]

# =====================================================
# DISCRIMINANT & CUT
# =====================================================

D_sig = P_VBF + P_ggF
mask  = (D_sig > CUT)

n_total = len(y_true)
n_pass  = np.sum(mask)

print("\n==============================================")
print(f"Mass hypothesis: mH = {MASS} GeV")
print(f"D_sig cut       : {CUT}")
print(f"Total events    : {n_total}")
print(f"Events passing  : {n_pass}")

if n_pass < 50:
    print(
        f"Skipping confusion matrix (only {n_pass} events pass cut)"
    )
    exit(0)

# =====================================================
# PHYSICS DIAGNOSTICS
# =====================================================

class_names = ["VBF", "ggF", "Background"]

# --- Per-class yields
print("\nPer-class yields after cut:")

for cls, name in enumerate(class_names):
    n_cls = np.sum((y_true == cls) & mask)
    print(f"  {name:10s}: {n_cls}")

# --- Background fraction
n_bkg = np.sum((y_true == 2) & mask)
frac_bkg = n_bkg / n_pass

print(f"\nBackground fraction after cut: {frac_bkg:.4f}")

# --- Signal efficiency
sig_total = np.sum(y_true != 2)
sig_pass  = np.sum((y_true != 2) & mask)

sig_eff = sig_pass / sig_total

print(f"Signal efficiency after cut  : {sig_eff:.4f}")

print("==============================================\n")

# =====================================================
# PLOT CONFUSION MATRIX
# =====================================================

outname = (
    f"{OUTDIR}/confusion_matrix_Dsig_gt_{CUT:.2f}_m{MASS}.png"
)

plot_confusion(
    y_true[mask],
    y_pred[mask],
    class_names,
    title=f"Confusion Matrix (D_sig > {CUT}, mH = {MASS} GeV)",
    outname=outname,
    normalize="true",
)

print(f"Saved {outname}")

