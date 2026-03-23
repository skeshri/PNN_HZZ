import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from evaluation.feature_names import FEATURE_NAMES

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

# =====================================================
# LOAD FILES
# =====================================================

mc_file = glob.glob(f"{EOS_BASE}/chunks/background/*/test/chunk_*.npz")[0]
data_file = glob.glob(f"{EOS_BASE}/chunks_data/chunk_*.npz")[0]

mc = np.load(mc_file)
data = np.load(data_file)

X_mc = mc["X"]
X_data = data["X"]

# =====================================================
# SANITY CHECK
# =====================================================

if X_mc.shape[1] != len(FEATURE_NAMES):
    raise RuntimeError(
        f"Feature mismatch: {X_mc.shape[1]} vs {len(FEATURE_NAMES)}"
    )

# =====================================================
# OUTPUT
# =====================================================

OUTDIR = "plots_feature_check"
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# LOOP
# =====================================================

for i, name in enumerate(FEATURE_NAMES):

    # Skip validity flags (optional but recommended)
    if name.endswith("_valid"):
        continue

    plt.figure(figsize=(6,4))

    plt.hist(
        X_mc[:, i],
        bins=50,
        density=True,
        histtype="stepfilled",
        alpha=0.3,
        label="MC"
    )

    plt.hist(
        X_data[:, i],
        bins=50,
        density=True,
        histtype="step",
        linewidth=2,
        label="Data"
    )

    plt.xlabel(name)
    plt.ylabel("Density")
    plt.title(f"{name} (Data vs MC)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Clean filename
    safe_name = name.replace("/", "_").replace(" ", "_")

    plt.savefig(f"{OUTDIR}/{i:02d}_{safe_name}.png")
    plt.close()

print("✅ Feature comparison done")
