import numpy as np
import os
import argparse

from evaluation.confusion_utils import plot_confusion

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="Confusion matrix per jet bin")
parser.add_argument(
    "--mass",
    type=int,
    required=True,
    help="Mass hypothesis in GeV (e.g. 500, 800, 1500)"
)
args = parser.parse_args()

MASS = args.mass

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

required_keys = ["y_true", "y_pred"]
for k in required_keys:
    if k not in data:
        raise KeyError(f"Missing key '{k}' in {INFILE}")

if "nJets" not in data:
    raise KeyError(
        f"'nJets' not found in {INFILE}. "
        "Jet-bin confusion matrices require nJets to be stored during validation."
    )

y_true = data["y_true"]
y_pred = data["y_pred"]
nJets  = data["nJets"]

class_names = ["VBF", "ggF", "Background"]

# =====================================================
# DEFINE JET BINS
# =====================================================

jet_bins = {
    "0jet":  (nJets == 0),
    "1jet":  (nJets == 1),
    "2jet+": (nJets >= 2),
}

# =====================================================
# PLOT CONFUSION MATRICES
# =====================================================

print(f"Producing jet-bin confusion matrices for mH = {MASS} GeV")

for name, mask in jet_bins.items():
    n_events = np.sum(mask)

    if n_events < 50:
        print(f"Skipping {name} (only {n_events} events)")
        continue

    outname = f"{OUTDIR}/confusion_matrix_{name}_m{MASS}.png"

    plot_confusion(
        y_true[mask],
        y_pred[mask],
        class_names,
        title=f"Confusion Matrix ({name}, mH = {MASS} GeV)",
        outname=outname,
        normalize="true",
    )

    print(f"Saved {outname}")

