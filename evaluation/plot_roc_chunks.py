import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
import os
import re

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="PNN ROC plots")
parser.add_argument(
    "--mass",
    type=int,
    required=True,
    help="Mass hypothesis in GeV"
)
parser.add_argument(
    "--split",
    type=str,
    default="test",
    choices=["train", "val", "test"],
    help="Dataset split"
)
args = parser.parse_args()

MASS = args.mass
SPLIT = args.split

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

INPUT_FILE = f"{EOS_BASE}/evaluation/pnn_output_m{MASS}_{SPLIT}.npz"
OUTDIR = f"{EOS_BASE}/plots/roc_m{MASS}_{SPLIT}"
os.makedirs(OUTDIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

mass_text = f"mH = {MASS} GeV"

# =====================================================
# LOAD DATA
# =====================================================

data = np.load(INPUT_FILE)

required_keys = ["P_VBF", "P_ggF", "P_bkg", "label"]
for k in required_keys:
    if k not in data:
        raise KeyError(f"Missing key '{k}' in {INPUT_FILE}")

P_VBF = data["P_VBF"]
P_ggF = data["P_ggF"]
P_bkg = data["P_bkg"]
y     = data["label"]

print(f"Loaded {len(y)} events")

# =====================================================
# DERIVED DISCRIMINANTS
# =====================================================

D_sig = P_VBF + P_ggF

# =====================================================
# 1) SIGNAL VS BACKGROUND ROC
# =====================================================

sig_mask = (y != 2)
bkg_mask = (y == 2)

if np.sum(sig_mask) > 0 and np.sum(bkg_mask) > 0:

    y_sig = sig_mask.astype(int)

    fpr_sig, tpr_sig, _ = roc_curve(y_sig, D_sig)
    auc_sig = auc(fpr_sig, tpr_sig)

    plt.figure(figsize=(6,6))
    plt.plot(
        fpr_sig, tpr_sig, lw=2,
        label=f"AUC = {auc_sig:.3f}"
    )
    plt.plot([0,1], [0,1], "k--", lw=1)
    plt.xlabel("Background efficiency")
    plt.ylabel("Signal efficiency")
    plt.title(f"Signal vs Background\n{mass_text} ({SPLIT})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_sig = f"{OUTDIR}/roc_signal_vs_background.png"
    plt.savefig(out_sig)
    plt.close()

else:
    print("Skipping Signal vs Background ROC")
    auc_sig = float("nan")

# =====================================================
# 2) VBF VS GGF ROC
# =====================================================

mask_sig = (y != 2)

if np.sum(mask_sig & (y == 0)) > 0 and np.sum(mask_sig & (y == 1)) > 0:

    P_VBF_s = P_VBF[mask_sig]
    P_ggF_s = P_ggF[mask_sig]
    y_sigproc = (y[mask_sig] == 0).astype(int)

    D_vbf = P_VBF_s / (P_VBF_s + P_ggF_s + 1e-6)

    fpr_vbf, tpr_vbf, _ = roc_curve(y_sigproc, D_vbf)
    auc_vbf = auc(fpr_vbf, tpr_vbf)

    plt.figure(figsize=(6,6))
    plt.plot(
        fpr_vbf, tpr_vbf, lw=2,
        label=f"AUC = {auc_vbf:.3f}"
    )
    plt.plot([0,1], [0,1], "k--", lw=1)
    plt.xlabel("ggF efficiency")
    plt.ylabel("VBF efficiency")
    plt.title(f"VBF vs ggF\n{mass_text} ({SPLIT})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_vbf = f"{OUTDIR}/roc_vbf_vs_ggf.png"
    plt.savefig(out_vbf)
    plt.close()

else:
    print("Skipping VBF vs ggF ROC")
    auc_vbf = float("nan")

# =====================================================
# SUMMARY
# =====================================================

print("\n========== ROC SUMMARY ==========")
print(f"Mass: {MASS} GeV")
print(f"Split: {SPLIT}")
print(f"AUC (Signal vs Background): {auc_sig:.4f}")
print(f"AUC (VBF vs ggF): {auc_vbf:.4f}")
print(f"Plots saved in: {OUTDIR}")

