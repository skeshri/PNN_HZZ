import os
import glob
import numpy as np
import uproot

# =====================================================
# CONFIG
# =====================================================

EOS_BASE = "/eos/user/s/skeshri/Analysis/PNN_PyTorch"
EOS_CHUNK_DIR = f"{EOS_BASE}/chunks_data"

CHUNK_SIZE = 50000

MASS_POINTS = [500, 800, 1500]

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

# =====================================================
# ROOT BRANCHES (same as MC)
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
# FEATURE BUILDING
# =====================================================

def build_engineered_features(data):

    # -------------------------
    # Lepton features
    # -------------------------
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

    # -------------------------
    # VBF topology features
    # -------------------------
    pt1 = data["HZZ2l2nu_VBFjet1_pT"]
    pt2 = data["HZZ2l2nu_VBFjet2_pT"]
    eta1 = data["HZZ2l2nu_VBFjet1_eta"]
    eta2 = data["HZZ2l2nu_VBFjet2_eta"]
    etaZ = data["etaZ1"]
    dEta_jj = data["HZZ2l2nu_VBFdEta_jj"]

    denom = np.clip(pt1 + pt2, 1e-6, None)
    data["jet_pt_asym"] = np.abs(pt1 - pt2) / denom

    data["HT_jets"] = pt1 + pt2

    data["dEta_Z_j1"] = np.abs(etaZ - eta1)
    data["dEta_Z_j2"] = np.abs(etaZ - eta2)

    denom = np.clip(np.abs(dEta_jj), 1e-6, None)
    data["centrality_norm"] = (
        etaZ - 0.5*(eta1 + eta2)
    ) / denom

    # -------------------------
    # NaN / Inf protection
    # -------------------------
    for k in data:
        data[k] = np.nan_to_num(data[k], nan=0.0, posinf=1e6, neginf=-1e6)


# =====================================================
# FEATURES (must match training)
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
    "jet_pt_asym",
    "HT_jets",
    "dEta_Z_j1",
    "dEta_Z_j2",
    "centrality_norm",
]

# =====================================================
# EVENT PROCESSOR
# =====================================================

def process_event(data, i):

    x = []

    # Base features
    for f in BASE_FEATURES:
        x.append(data[f][i])

    nJets = data["HZZ2l2qNu_nJets"][i]

    pt1 = data["HZZ2l2nu_VBFjet1_pT"][i]
    pt2 = data["HZZ2l2nu_VBFjet2_pT"][i]

    for raw in MASKED_FEATURES:

        val = data[raw][i]

        valid = (
            (nJets >= 2) and
            (pt1 > 0) and
            (pt2 > 0)
        )

        if not valid:
            x.extend([0.0, 0.0])
        else:
            x.extend([val, 1.0])

    return x


# =====================================================
# CORE PROCESSOR (DATA ONLY)
# =====================================================

def process_data(files, outbase):

    os.makedirs(outbase, exist_ok=True)

    chunk_id = 0

    for fname in sorted(files):

        print(f"Processing {fname}")

        with uproot.open(fname) as f:

            tree = f["Events"]

            # Branch validation
            available = tree.keys()
            missing = [b for b in ROOT_BRANCHES if b not in available]
            if missing:
                raise RuntimeError(f"Missing branches: {missing}")

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

                X, y, m, w = [], [], [], []

                for i in range(len(data[ROOT_BRANCHES[0]])):

                    x_evt = process_event(data, i)

                    for mh in MASS_POINTS:
                        X.append(x_evt)
                        y.append(-1)          # dummy label
                        m.append(mh)
                        w.append(1.0)         # unit weight

                if len(X) == 0:
                    continue

                X = np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.int64)
                m = np.array(m, dtype=np.int64)
                w = np.array(w, dtype=np.float32)

                outname = f"{outbase}/chunk_{chunk_id:05d}.npz"

                np.savez(outname, X=X, y=y, m=m, w=w)

                chunk_id += 1


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    process_data(
        glob.glob(f"{EOS_BASE}/data/data/egamma/*.root"),
        EOS_CHUNK_DIR
    )

    print("\nReal data processing completed.")
