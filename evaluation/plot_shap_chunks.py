import numpy as np
import shap
import matplotlib.pyplot as plt
import argparse
import os

from evaluation.feature_names import FEATURE_NAMES

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="Plot SHAP summaries for PNN")

parser.add_argument(
    "--mass",
    type=int,
    required=True,
    help="Mass hypothesis in GeV (e.g. 500, 800, 1500)"
)

parser.add_argument(
    "--top",
    type=int,
    default=None,
    help="Plot only top-N most important features (default: all physical features)"
)

parser.add_argument(
    "--clip",
    type=float,
    default=None,
    help="Clip SHAP values to [-clip, clip] for better visualization"
)

args = parser.parse_args()

MASS = args.mass
TOP_N = args.top
CLIP = args.clip

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

INFILE = f"{EOS_BASE}/models/shap_values_m{MASS}.npz"
OUTDIR = f"{EOS_BASE}/plots/shap_m{MASS}"
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

if not os.path.exists(INFILE):
    raise FileNotFoundError(f"SHAP file not found: {INFILE}")

data = np.load(INFILE)

required_keys = ["X", "shap_vbf", "shap_ggf", "shap_bkg"]
for k in required_keys:
    if k not in data:
        raise KeyError(f"Missing key '{k}' in {INFILE}")

X = data["X"]
shap_vbf = data["shap_vbf"]
shap_ggf = data["shap_ggf"]
shap_bkg = data["shap_bkg"]

# =====================================================
# SANITY CHECKS
# =====================================================

print("DEBUG shapes:")
print("X.shape        =", X.shape)
print("shap_vbf.shape =", shap_vbf.shape)
print("n_features     =", len(FEATURE_NAMES))

assert X.shape == shap_vbf.shape, "SHAP/X shape mismatch"

if X.shape[1] != len(FEATURE_NAMES):
    raise RuntimeError(
        f"FEATURE_NAMES mismatch: {len(FEATURE_NAMES)} vs {X.shape[1]}"
    )

# =====================================================
# REMOVE MASK FEATURES (_valid)
# =====================================================

keep_indices = []
filtered_feature_names = []

for i, name in enumerate(FEATURE_NAMES):
    if not name.endswith("_valid"):
        keep_indices.append(i)
        filtered_feature_names.append(name)

print(f"Removing mask features → plotting {len(filtered_feature_names)} variables")

X = X[:, keep_indices]
shap_vbf = shap_vbf[:, keep_indices]
shap_ggf = shap_ggf[:, keep_indices]
shap_bkg = shap_bkg[:, keep_indices]

# =====================================================
# OPTIONAL CLIPPING (for visual stability)
# =====================================================

if CLIP is not None:
    print(f"Clipping SHAP values to [-{CLIP}, {CLIP}]")
    shap_vbf = np.clip(shap_vbf, -CLIP, CLIP)
    shap_ggf = np.clip(shap_ggf, -CLIP, CLIP)
    shap_bkg = np.clip(shap_bkg, -CLIP, CLIP)

# =====================================================
# DETERMINE DISPLAY COUNT
# =====================================================

if TOP_N is None:
    MAX_DISPLAY = len(filtered_feature_names)
else:
    MAX_DISPLAY = TOP_N

# =====================================================
# PLOTTING FUNCTION
# =====================================================

def plot_summary(shap_vals, title, outfile_base):

    # ---- Beeswarm summary plot ----
    shap.summary_plot(
        shap_vals,
        X,
        feature_names=filtered_feature_names,
        show=False,
        plot_size=(8, 6),
        max_display=MAX_DISPLAY,
        sort=True
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{outfile_base}_beeswarm.png", dpi=300)
    plt.close()

    # ---- Bar importance plot ----
    shap.summary_plot(
        shap_vals,
        X,
        feature_names=filtered_feature_names,
        plot_type="bar",
        show=False,
        max_display=MAX_DISPLAY,
        sort=True
    )

    plt.title(title + " (Feature Importance)")
    plt.tight_layout()
    plt.savefig(f"{outfile_base}_bar.png", dpi=300)
    plt.close()

# =====================================================
# PRODUCE PLOTS
# =====================================================

plot_summary(
    shap_vbf,
    f"SHAP – VBF (mH = {MASS} GeV)",
    f"{OUTDIR}/shap_vbf_m{MASS}"
)

plot_summary(
    shap_ggf,
    f"SHAP – ggF (mH = {MASS} GeV)",
    f"{OUTDIR}/shap_ggf_m{MASS}"
)

plot_summary(
    shap_bkg,
    f"SHAP – Background (mH = {MASS} GeV)",
    f"{OUTDIR}/shap_bkg_m{MASS}"
)

print("========================================")
print(" SHAP summary plots saved successfully ")
print(f" Directory: {OUTDIR}")
print("========================================")
