import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mass", type=int, required=True)
args = parser.parse_args()

MASS = args.mass

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

INFILE = f"{EOS_BASE}/evaluation_data/pnn_data_m{MASS}.npz"
OUTDIR = f"{EOS_BASE}/plots_data/m{MASS}"
os.makedirs(OUTDIR, exist_ok=True)

data = np.load(INFILE)

P_VBF = data["P_VBF"]
P_ggF = data["P_ggF"]
P_bkg = data["P_bkg"]

# =====================================================
# Derived
# =====================================================

D_sig = P_VBF + P_ggF
D_vbf = P_VBF / (P_VBF + P_ggF + 1e-6)

# =====================================================
# Plot function
# =====================================================

def plot(var, name):

    plt.figure()
    plt.hist(var, bins=50, density=True, histtype="step")
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.title(f"{name} (Data, mH={MASS})")
    plt.grid(True)
    plt.savefig(f"{OUTDIR}/{name}.png")
    plt.close()

# =====================================================
# Plots
# =====================================================

plot(P_VBF, "P_VBF")
plot(P_ggF, "P_ggF")
plot(P_bkg, "P_bkg")
plot(D_sig, "D_sig")
plot(D_vbf, "D_vbf")

print("✅ Plots saved")
