import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser(description="PNN separation plots")

parser.add_argument("--mass", type=int, required=True)
parser.add_argument("--split", default="test", choices=["train","val","test"])

parser.add_argument(
    "--norm",
    choices=["events","density"],
    default="events"
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
norm_mode = args.norm
CUT = args.cut

# =====================================================
# PATHS
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

INPUT_FILE = f"{EOS_BASE}/evaluation/pnn_output_m{MASS}_{SPLIT}.npz"
OUTDIR = f"{EOS_BASE}/plots/separation_m{MASS}_{SPLIT}"
os.makedirs(OUTDIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)

mass_text = f"mH = {MASS} GeV"

# =====================================================
# LOAD DATA
# =====================================================

data = np.load(INPUT_FILE)

P_VBF = data["P_VBF"]
P_ggF = data["P_ggF"]
P_bkg = data["P_bkg"]
y     = data["label"]

print(f"Loaded {len(y)} events")

# =====================================================
# DISCRIMINANTS
# =====================================================

D_sig = P_VBF + P_ggF
D_vbf = P_VBF / (P_VBF + P_ggF + 1e-6)

# =====================================================
# MASKS
# =====================================================

sig_mask = (y != 2)
bkg_mask = (y == 2)
vbf_mask = (y == 0)
ggf_mask = (y == 1)

bins = np.linspace(0,1,41)

# =====================================================
# NORMALIZATION HANDLER
# =====================================================

def hist_kwargs(values):

    if norm_mode == "events":
        return {
            "weights": np.ones_like(values) / len(values),
            "density": False
        }

    return {"density": True}

ylabel = (
    "Fraction of events"
    if norm_mode=="events"
    else "Probability density"
)

# =====================================================
# 1) SIGNAL VS BACKGROUND
# =====================================================

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

plt.xlabel(r"$D_{sig} = P_{VBF}+P_{ggF}$")
plt.ylabel(ylabel)

plt.title(
    f"Signal vs Background\n"
    f"{mass_text} ({SPLIT})"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

out = f"{OUTDIR}/separation_signal_vs_background_{norm_mode}.png"
plt.savefig(out)
plt.close()

print("Saved", out)

# =====================================================
# 2) VBF VS GGF (FULL SAMPLE)
# =====================================================

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

plt.xlabel(r"$D_{VBF} = P_{VBF}/(P_{VBF}+P_{ggF})$")
plt.ylabel(ylabel)

plt.title(
    f"VBF vs ggF\n"
    f"{mass_text} ({SPLIT})"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

out = f"{OUTDIR}/separation_vbf_vs_ggf_{norm_mode}.png"
plt.savefig(out)
plt.close()

print("Saved", out)

# =====================================================
# 3) VBF VS GGF AFTER D_SIG CUT
# =====================================================

if CUT is not None:

    print(f"Applying D_sig > {CUT}")

    mask = D_sig > CUT

    vbf_mask_cut = (y==0) & mask
    ggf_mask_cut = (y==1) & mask

    print(
        "Events after cut:",
        np.sum(mask),
        "/",
        len(mask)
    )

    plt.figure(figsize=(7,5))

    plt.hist(
        D_vbf[ggf_mask_cut],
        bins=bins,
        histtype="stepfilled",
        alpha=0.35,
        label="ggF",
        **hist_kwargs(D_vbf[ggf_mask_cut])
    )

    plt.hist(
        D_vbf[vbf_mask_cut],
        bins=bins,
        histtype="step",
        linewidth=2,
        label="VBF",
        **hist_kwargs(D_vbf[vbf_mask_cut])
    )

    plt.xlabel(r"$D_{VBF} = P_{VBF}/(P_{VBF}+P_{ggF})$")
    plt.ylabel(ylabel)

    plt.title(
        f"VBF vs ggF after $D_{{sig}}>{CUT}$\n"
        f"{mass_text} ({SPLIT})"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = f"{OUTDIR}/separation_vbf_vs_ggf_Dsig{CUT}_{norm_mode}.png"
    plt.savefig(out)
    plt.close()

    print("Saved", out)
