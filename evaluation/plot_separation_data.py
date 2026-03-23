import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="Separation plots (DATA)")
parser.add_argument("--mass", type=int, required=True)
parser.add_argument("--norm", choices=["events", "density"], default="density")
args = parser.parse_args()

MASS = args.mass
norm_mode = args.norm

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

INFILE = f"{EOS_BASE}/evaluation_data/pnn_data_m{MASS}.npz"
OUTDIR = f"{EOS_BASE}/plots_data/separation_m{MASS}"
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# LOAD
# =====================================================

data = np.load(INFILE)

P_VBF = data["P_VBF"]
P_ggF = data["P_ggF"]
P_bkg = data["P_bkg"]

print(f"Loaded {len(P_VBF)} events")

# =====================================================
# DISCRIMINANTS
# =====================================================

D_sig = P_VBF + P_ggF
D_vbf = P_VBF / (P_VBF + P_ggF + 1e-6)

bins = np.linspace(0, 1, 50)

# =====================================================
# NORMALIZATION
# =====================================================

def hist_kwargs(values):
    if norm_mode == "events":
        return {"weights": np.ones_like(values) / len(values)}
    else:
        return {"density": True}

ylabel = "Fraction of events" if norm_mode == "events" else "Density"

# =====================================================
# 1) SIGNAL-LIKE DISCRIMINANT
# =====================================================

plt.figure(figsize=(7,5))

plt.hist(D_sig, bins=bins, histtype="step", linewidth=2,
         label="Data", **hist_kwargs(D_sig))

plt.xlabel(r"$D_{\mathrm{sig}} = P_{\mathrm{VBF}} + P_{\mathrm{ggF}}$")
plt.ylabel(ylabel)
plt.title(f"Data distribution (mH = {MASS} GeV)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(f"{OUTDIR}/D_sig.png")
plt.close()

# =====================================================
# 2) VBF FRACTION
# =====================================================

plt.figure(figsize=(7,5))

plt.hist(D_vbf, bins=bins, histtype="step", linewidth=2,
         label="Data", **hist_kwargs(D_vbf))

plt.xlabel(r"$D_{\mathrm{VBF}} = P_{\mathrm{VBF}}/(P_{\mathrm{VBF}}+P_{\mathrm{ggF}})$")
plt.ylabel(ylabel)
plt.title(f"VBF score (mH = {MASS} GeV)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(f"{OUTDIR}/D_vbf.png")
plt.close()

print("✅ Separation plots saved")
