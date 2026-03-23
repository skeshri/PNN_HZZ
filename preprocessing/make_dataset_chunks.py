import uproot
import numpy as np
import glob
import os
import random

# =====================================================
# CONFIGURATION
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"

CHUNK_SIZE = 100_000
EOS_CHUNK_DIR = f"{EOS_BASE}/chunks"

MASS_POINTS = [500, 800, 1500]

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SPLIT_FRAC = {
    "train": 0.70,
    "val":   0.15,
    "test":  0.15,
}

LABELS = {
    "VBF": 0,
    "ggF": 1,
    "bkg": 2
}

# =====================================================
# ROOT BRANCHES
# =====================================================

ROOT_BRANCHES = [
    "pTL1","etaL1","phiL1",
    "pTL2","etaL2","phiL2",
    "massZ1","pTZ1","etaZ1",
    "PuppiMET_pt",
    "HZZ2l2nu_ZZmT",
    "HZZ2l2nu_ZZpT",
    "HZZ2l2qNu_nJets",

    # VBF jet variables
    "HZZ2l2nu_VBFjet1_pT",
    "HZZ2l2nu_VBFjet2_pT",
    "HZZ2l2nu_VBFjet1_eta",
    "HZZ2l2nu_VBFjet2_eta",
    "HZZ2l2nu_VBFdEta_jj",
    "HZZ2l2nu_VBFdPhi_jj",
    "HZZ2l2nu_VBFdR_jj",
    "HZZ2l2nu_VBFdijet_mass",
    "HZZ2l2nu_VBFdijet_pT",
    "HZZ2l2nu_minDPhi_METAK4",
]

# =====================================================
# FEATURE BUILDERS
# =====================================================

def build_engineered_features(data):

    # Lepton variables
    data["deltaEta_ll"] = np.abs(data["etaL1"] - data["etaL2"])

    dphi = data["phiL1"] - data["phiL2"]
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    data["deltaPhi_ll"] = np.abs(dphi)

    data["deltaR_ll"] = np.sqrt(
        (data["etaL1"] - data["etaL2"])**2 + dphi**2
    )

    data["pt1_by_pt2"] = data["pTL1"] / np.clip(data["pTL2"], 1e-6, None)
    data["eta1_by_eta2"] = data["etaL1"] / np.clip(data["etaL2"], 1e-6, None)
    data["zpt1_by_met"] = data["pTZ1"] / np.clip(data["PuppiMET_pt"], 1e-6, None)

    # -------------------------------------------------
    # VBF topology variables
    # -------------------------------------------------

    nJets = data["HZZ2l2qNu_nJets"]

    pt1 = data["HZZ2l2nu_VBFjet1_pT"]
    pt2 = data["HZZ2l2nu_VBFjet2_pT"]
    eta1 = data["HZZ2l2nu_VBFjet1_eta"]
    eta2 = data["HZZ2l2nu_VBFjet2_eta"]
    etaZ = data["etaZ1"]
    dEta_jj = data["HZZ2l2nu_VBFdEta_jj"]

    # Jet pT asymmetry
    denom = np.clip(pt1 + pt2, 1e-6, None)
    data["jet_pt_asym"] = np.abs(pt1 - pt2) / denom

    # HT
    data["HT_jets"] = pt1 + pt2

    # Δη(Z, jets)
    data["dEta_Z_j1"] = np.abs(etaZ - eta1)
    data["dEta_Z_j2"] = np.abs(etaZ - eta2)

    # Normalized centrality
    denom = np.clip(np.abs(dEta_jj), 1e-6, None)
    data["centrality_norm"] = (
        etaZ - 0.5*(eta1 + eta2)
    ) / denom


# =====================================================
# PROCESS EVENT
# =====================================================

BASE_FEATURES = [
    "pTL1","etaL1","phiL1",
    "pTL2","etaL2","phiL2",
    "massZ1","pTZ1","etaZ1",
    "PuppiMET_pt",
    "HZZ2l2nu_ZZmT",
    "HZZ2l2nu_ZZpT",
    "HZZ2l2qNu_nJets",
    "deltaEta_ll",
    "deltaPhi_ll",
    "deltaR_ll",
    "pt1_by_pt2",
    "eta1_by_eta2",
    "zpt1_by_met",
]

MASKED_FEATURES = [
    "HZZ2l2nu_VBFdijet_mass",
    "HZZ2l2nu_VBFdEta_jj",
    "HZZ2l2nu_VBFdPhi_jj",
    "HZZ2l2nu_VBFdR_jj",
    "HZZ2l2nu_VBFdijet_pT",
    "HZZ2l2nu_minDPhi_METAK4",

    # NEW
    "jet_pt_asym",
    "HT_jets",
    "dEta_Z_j1",
    "dEta_Z_j2",
    "centrality_norm",
]

def process_event(data, i):

    x = []

    for f in BASE_FEATURES:
        x.append(data[f][i])

    nJets = data["HZZ2l2qNu_nJets"][i]

    for raw in MASKED_FEATURES:

        val = data[raw][i]

        valid = (nJets >= 2)

        if not valid:
            x.extend([0.0, 0.0])
        else:
            x.extend([val, 1.0])

    return x


# =====================================================
# CORE PROCESSOR
# =====================================================

def process_root_files(files, label, outbase):

    for split in SPLIT_FRAC:
        os.makedirs(f"{outbase}/{split}", exist_ok=True)

    chunk_id = {k:0 for k in SPLIT_FRAC}

    for fname in sorted(files):

        print(f"Processing {fname}")

        with uproot.open(fname) as f:

            tree = f["Events"]
            n_events = tree.num_entries

            for start in range(0, n_events, CHUNK_SIZE):

                stop = min(start + CHUNK_SIZE, n_events)

                data = tree.arrays(
                    ROOT_BRANCHES,
                    entry_start=start,
                    entry_stop=stop,
                    library="np"
                )

                build_engineered_features(data)

                X, y, m = [], [], []

                for i in range(len(data[ROOT_BRANCHES[0]])):

                    x_evt = process_event(data, i)

                    for mh in MASS_POINTS:
                        X.append(x_evt)
                        y.append(label)
                        m.append(mh)

                if len(X) == 0:
                    continue

                X = np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.int64)
                m = np.array(m, dtype=np.int64)

                n = len(X)
                indices = np.random.permutation(n)

                n_train = int(n * SPLIT_FRAC["train"])
                n_val   = int(n * SPLIT_FRAC["val"])

                split_indices = {
                    "train": indices[:n_train],
                    "val": indices[n_train:n_train+n_val],
                    "test": indices[n_train+n_val:]
                }

                for split, idx in split_indices.items():

                    if len(idx) == 0:
                        continue

                    outname = f"{outbase}/{split}/chunk_{chunk_id[split]:05d}.npz"

                    np.savez(outname, X=X[idx], y=y[idx], m=m[idx])

                    chunk_id[split] += 1


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    process_root_files(
        glob.glob(f"{EOS_BASE}/data/ggF/*.root"),
        LABELS["ggF"],
        f"{EOS_CHUNK_DIR}/ggF"
    )

    process_root_files(
        glob.glob(f"{EOS_BASE}/data/VBF/*.root"),
        LABELS["VBF"],
        f"{EOS_CHUNK_DIR}/VBF"
    )

    for bkg in ["DY","TOP","VV","RARE"]:

        process_root_files(
            glob.glob(f"{EOS_BASE}/data/background/{bkg}/*.root"),
            LABELS["bkg"],
            f"{EOS_CHUNK_DIR}/background/{bkg}"
        )

    print("\nDataset creation completed.")
