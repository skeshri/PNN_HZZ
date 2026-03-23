import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="PNN separation plots")
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
parser.add_argument(
    "--norm",
    choices=["events", "density"],
    default="events",
    help="Histogram normalization"
)
args = parser.parse_args()

MASS = args.mass
SPLIT = args.split
norm_mode = args.norm

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

INPUT_FILE = f"{EOS_BASE}/evaluation/pnn_output_m{MASS}_{SPLIT}.npz"
OUTDIR = f"{EOS_BASE}/plots/separation_m{MASS}_{SPLIT}"
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
# MASKS & DISCRIMINANTS
# =====================================================

sig_mask = (y != 2)
bkg_mask = (y == 2)
vbf_mask = (y == 0)
ggf_mask = (y == 1)

D_sig = P_VBF + P_ggF
D_vbf = P_VBF / (P_VBF + P_ggF + 1e-6)

bins = np.linspace(0, 1, 41)

# =====================================================
# NORMALIZATION HANDLER
# =====================================================

def hist_kwargs(values):
    if len(values) == 0:
        return None
    if norm_mode == "events":
        return {
            "weights": np.ones_like(values) / len(values),
            "density": False
        }
    else:
        return {"density": True}

ylabel = (
    "Fraction of events" if norm_mode == "events"
    else "Probability density"
)

# =====================================================
# 1) SIGNAL VS BACKGROUND
# =====================================================

if np.sum(sig_mask) > 0 and np.sum(bkg_mask) > 0:

    plt.figure(figsize=(7,5))

    plt.hist(
        D_sig[bkg_mask],
        bins=bins,
        histtype="stepfilled",
        alpha=0.35,
        label="Background",
        **hist_kwargs(D_sig[bkg_mask])
    )

    plt.hist(
        D_sig[sig_mask],
        bins=bins,
        histtype="step",
        linewidth=2,
        label="Signal (ggF + VBF)",
        **hist_kwargs(D_sig[sig_mask])
    )

    plt.xlabel(r"$D_{\mathrm{sig}} = P_{\mathrm{VBF}} + P_{\mathrm{ggF}}$")
    plt.ylabel(ylabel)
    plt.title(
        f"Signal vs Background\n"
        f"{mass_text} ({SPLIT}, {norm_mode})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_sig = f"{OUTDIR}/separation_signal_vs_background_{norm_mode}.png"
    plt.savefig(out_sig)
    plt.close()

    print(f"Saved {out_sig}")

else:
    print("Skipping Signal vs Background separation")

# =====================================================
# 2) VBF VS GGF
# =====================================================

if np.sum(vbf_mask) > 0 and np.sum(ggf_mask) > 0:

    plt.figure(figsize=(7,5))

    plt.hist(
        D_vbf[ggf_mask],
        bins=bins,
        histtype="stepfilled",
        alpha=0.35,
        label="ggF",
        **hist_kwargs(D_vbf[ggf_mask])
    )

    plt.hist(
        D_vbf[vbf_mask],
        bins=bins,
        histtype="step",
        linewidth=2,
        label="VBF",
        **hist_kwargs(D_vbf[vbf_mask])
    )

    plt.xlabel(
        r"$D_{\mathrm{VBF}} = "
        r"P_{\mathrm{VBF}}/(P_{\mathrm{VBF}}+P_{\mathrm{ggF}})$"
    )
    plt.ylabel(ylabel)
    plt.title(
        f"VBF vs ggF\n"
        f"{mass_text} ({SPLIT}, {norm_mode})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_vbf = f"{OUTDIR}/separation_vbf_vs_ggf_{norm_mode}.png"
    plt.savefig(out_vbf)
    plt.close()

    print(f"Saved {out_vbf}")

else:
    print("Skipping VBF vs ggF separation")

