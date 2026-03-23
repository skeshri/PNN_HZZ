# ============================================================
# Feature name definition for PNN (ALIGNED WITH PREPROCESSING)
# ============================================================

"""
Feature ordering MUST match preprocessing.

Total features = 41
  - 19 base features
  - 11 masked features × (value + validity flag)
"""

# ------------------------------------------------------------
# Base features (including engineered ones)
# ------------------------------------------------------------

BASE_FEATURES = [
    "pTL1","etaL1","phiL1",
    "pTL2","etaL2","phiL2",
    "massZ1","pTZ1","etaZ1",
    "PuppiMET_pt",
    "HZZ2l2nu_ZZmT",
    "HZZ2l2nu_ZZpT",
    "HZZ2l2qNu_nJets",

    # Engineered (lepton-level)
    "deltaEta_ll",
    "deltaPhi_ll",
    "deltaR_ll",
    "pt1_by_pt2",
    "eta1_by_eta2",
    "zpt1_by_met",
]

# ------------------------------------------------------------
# Masked features (EXACTLY as in preprocessing)
# ------------------------------------------------------------

MASKED_FEATURES = [
    # Raw VBF features
    "HZZ2l2nu_VBFdijet_mass",
    "HZZ2l2nu_VBFdEta_jj",
    "HZZ2l2nu_VBFdPhi_jj",
    "HZZ2l2nu_VBFdR_jj",
    "HZZ2l2nu_VBFdijet_pT",
    "HZZ2l2nu_minDPhi_METAK4",

    # Engineered VBF/topology features
    "jet_pt_asym",
    "HT_jets",
    "dEta_Z_j1",
    "dEta_Z_j2",
    "centrality_norm",
]

# ------------------------------------------------------------
# Build feature list
# ------------------------------------------------------------

FEATURE_NAMES = []

FEATURE_NAMES.extend(BASE_FEATURES)

for name in MASKED_FEATURES:
    FEATURE_NAMES.append(name)
    FEATURE_NAMES.append(f"{name}_valid")

# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------

EXPECTED_N_FEATURES = 41

if len(FEATURE_NAMES) != EXPECTED_N_FEATURES:
    raise RuntimeError(
        f"Feature count mismatch: {len(FEATURE_NAMES)} vs expected {EXPECTED_N_FEATURES}"
    )

# ------------------------------------------------------------
# Debug print
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Total features:", len(FEATURE_NAMES))
    for i, f in enumerate(FEATURE_NAMES):
        print(f"{i:02d} : {f}")
