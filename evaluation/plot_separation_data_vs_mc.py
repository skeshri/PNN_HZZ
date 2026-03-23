import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="Data vs MC separation plot")
parser.add_argument(
    "--mass",
    type=int,
    required=True,
    help="Mass hypothesis (e.g. 500, 800, 1500)"
)
parser.add_argument(
    "--outdir",
    type=str,
    default=".",
    help="Output directory"
)

args = parser.parse_args()

MASS = args.mass
OUTDIR = args.outdir

os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

mc_file = f"{EOS_BASE}/evaluation/pnn_output_m{MASS}_test.npz"
data_file = f"{EOS_BASE}/evaluation_data/pnn_data_m{MASS}.npz"

# =====================================================
# LOAD
# =====================================================

if not os.path.exists(mc_file):
    raise FileNotFoundError(f"MC file not found: {mc_file}")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")

mc = np.load(mc_file)
data = np.load(data_file)

# =====================================================
# DISCRIMINANTS
# =====================================================

D_sig_mc = mc["P_VBF"] + mc["P_ggF"]
D_sig_data = data["P_VBF"] + data["P_ggF"]

bins = np.linspace(0, 1, 50)

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(7,5))

plt.hist(
    D_sig_mc,
    bins=bins,
    density=True,
    histtype="stepfilled",
    alpha=0.3,
    label="MC"
)

plt.hist(
    D_sig_data,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Data"
)

plt.xlabel(r"$D_{\mathrm{sig}} = P_{\mathrm{VBF}} + P_{\mathrm{ggF}}$")
plt.ylabel("Density")
plt.title(f"Data vs MC (mH = {MASS} GeV)")
plt.legend()
plt.grid(True)
plt.tight_layout()

outname = f"{OUTDIR}/data_vs_mc_Dsig_m{MASS}.png"
plt.savefig(outname)
plt.close()

print(f"✅ Saved: {outname}")
