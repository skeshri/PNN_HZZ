import argparse
import glob
import os
import random
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import uproot

from training.model import PNN


# =====================================================
# Feature configuration (must match training)
# =====================================================

ROOT_BRANCHES = [
    "pTL1", "etaL1", "phiL1",
    "pTL2", "etaL2", "phiL2",
    "massZ1", "pTZ1", "etaZ1",
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

BASE_FEATURES = [
    "pTL1", "etaL1", "phiL1",
    "pTL2", "etaL2", "phiL2",
    "massZ1", "pTZ1", "etaZ1",
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


def build_engineered_features(data: Dict[str, np.ndarray]) -> None:
    """Build engineered features in-place (matching training/preprocessing)."""

    data["deltaEta_ll"] = np.abs(data["etaL1"] - data["etaL2"])

    dphi = data["phiL1"] - data["phiL2"]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    data["deltaPhi_ll"] = np.abs(dphi)

    data["deltaR_ll"] = np.sqrt(
        (data["etaL1"] - data["etaL2"]) ** 2 + dphi ** 2
    )

    data["pt1_by_pt2"] = data["pTL1"] / np.clip(data["pTL2"], 1e-6, None)
    data["eta1_by_eta2"] = data["etaL1"] / np.clip(data["etaL2"], 1e-6, None)
    data["zpt1_by_met"] = data["pTZ1"] / np.clip(data["PuppiMET_pt"], 1e-6, None)

    pt1 = data["HZZ2l2nu_VBFjet1_pT"]
    pt2 = data["HZZ2l2nu_VBFjet2_pT"]
    eta1 = data["HZZ2l2nu_VBFjet1_eta"]
    eta2 = data["HZZ2l2nu_VBFjet2_eta"]
    eta_z = data["etaZ1"]
    d_eta_jj = data["HZZ2l2nu_VBFdEta_jj"]

    data["jet_pt_asym"] = np.abs(pt1 - pt2) / np.clip(pt1 + pt2, 1e-6, None)
    data["HT_jets"] = pt1 + pt2
    data["dEta_Z_j1"] = np.abs(eta_z - eta1)
    data["dEta_Z_j2"] = np.abs(eta_z - eta2)
    data["centrality_norm"] = (eta_z - 0.5 * (eta1 + eta2)) / np.clip(np.abs(d_eta_jj), 1e-6, None)

    for k in list(data.keys()):
        data[k] = np.nan_to_num(data[k], nan=0.0, posinf=1e6, neginf=-1e6)


def build_feature_matrix(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Construct model input matrix with masked features and mask bits."""

    x_base = np.column_stack([data[f] for f in BASE_FEATURES])

    n_jets = data["HZZ2l2qNu_nJets"]
    valid = n_jets >= 2

    masked_columns: List[np.ndarray] = []
    for feature in MASKED_FEATURES:
        value = np.where(valid, data[feature], 0.0).astype(np.float32)
        mask = valid.astype(np.float32)
        masked_columns.append(value)
        masked_columns.append(mask)

    x_masked = np.column_stack(masked_columns)
    x = np.concatenate([x_base.astype(np.float32), x_masked.astype(np.float32)], axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def parse_signal_mass_from_filename(path: str, allowed_masses: Sequence[int]) -> Optional[int]:
    """Extract mass hypothesis from filename by regex; keep only allowed masses if provided."""

    name = os.path.basename(path)
    candidates: List[int] = []

    # Prefer explicit mass tokens such as mH500, MH-800, mass1500, M_500...
    explicit_patterns = [
        r"(?:mH|MH|mass|M)[-_]?(\d{3,4})",
        r"To(\d{3,4})",  # common HEP naming pattern
    ]
    for pattern in explicit_patterns:
        candidates.extend(int(m) for m in re.findall(pattern, name))

    # Fallback to any standalone 3/4-digit number if explicit forms were absent.
    if not candidates:
        candidates = [int(m) for m in re.findall(r"(?<!\d)(\d{3,4})(?!\d)", name)]

    if allowed_masses:
        for c in candidates:
            if c in allowed_masses:
                return c

    return candidates[0] if candidates else None


def infer_file_kind(path: str) -> str:
    lower = path.lower()
    signal_tokens = (
        "ggf",
        "gluglu",   # e.g. GluGluHToZZTo2L2Nu_M500_...
        "vbf",      # e.g. VBF_HToZZTo2L2Nu_M500_...
        "htozz",    # signal convention in this production
        "signal",
    )
    if any(token in lower for token in signal_tokens):
        return "signal"
    return "background"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update ROOT ntuples with PNN scores.")
    parser.add_argument("--input", required=True, help="Input ROOT file glob (quoted).")
    parser.add_argument("--output-dir", required=True, help="Output directory for scored ROOT files.")
    parser.add_argument("--model", required=True, help="Path to trained model/checkpoint (.pt).")
    parser.add_argument("--norm", default="preprocessing/norm.npz", help="Path to normalization npz.")
    parser.add_argument("--tree", default="Events", help="TTree name.")
    parser.add_argument("--masses", default="500,800,1500", help="Comma-separated allowed mass points.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Events per processing chunk.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed (background mass assignment).")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise RuntimeError(f"No ROOT files matched pattern: {args.input}")

    masses = [int(x.strip()) for x in args.masses.split(",") if x.strip()]
    if not masses:
        raise RuntimeError("No masses provided.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm = np.load(args.norm)
    mean = torch.tensor(norm["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm["std"], dtype=torch.float32, device=device)

    model = PNN(n_features=len(mean)).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.eval()

    print("=" * 70)
    print(f"Found {len(files)} files")
    print(f"Device: {device}")
    print(f"Allowed masses: {masses}")
    print("=" * 70)

    for fpath in files:
        file_kind = infer_file_kind(fpath)
        signal_mass = parse_signal_mass_from_filename(fpath, masses) if file_kind == "signal" else None

        if file_kind == "signal" and signal_mass is None:
            raise RuntimeError(
                f"Could not extract signal mass from filename: {fpath}. "
                f"Ensure filename contains one of: {masses}"
            )

        out_path = os.path.join(args.output_dir, os.path.basename(fpath))
        print(f"\nProcessing: {fpath}")
        print(f"Kind: {file_kind}")
        if signal_mass is not None:
            print(f"Signal mass from filename: {signal_mass}")
        print(f"Output: {out_path}")

        with uproot.open(fpath) as fin:
            tree = fin[args.tree]
            n_events = tree.num_entries

            missing = [b for b in ROOT_BRANCHES if b not in tree.keys()]
            if missing:
                raise RuntimeError(f"Missing required branches in {fpath}: {missing}")

            with uproot.recreate(out_path) as fout:
                out_tree = None

                for start in range(0, n_events, args.chunk_size):
                    stop = min(start + args.chunk_size, n_events)

                    chunk = tree.arrays(entry_start=start, entry_stop=stop, library="np")

                    feature_data = tree.arrays(
                        ROOT_BRANCHES,
                        entry_start=start,
                        entry_stop=stop,
                        library="np",
                    )

                    build_engineered_features(feature_data)
                    x_np = build_feature_matrix(feature_data)

                    n_chunk = len(x_np)
                    if file_kind == "signal":
                        masses_chunk = np.full(n_chunk, signal_mass, dtype=np.float32)
                    else:
                        masses_chunk = np.random.choice(masses, size=n_chunk).astype(np.float32)

                    m_norm = ((masses_chunk - 1000.0) / 500.0).astype(np.float32)

                    with torch.no_grad():
                        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
                        x_t = (x_t - mean) / std

                        m_t = torch.tensor(m_norm, dtype=torch.float32, device=device).unsqueeze(1)
                        probs = torch.softmax(model(x_t, m_t), dim=1).cpu().numpy().astype(np.float32)

                    p_vbf = probs[:, 0]
                    p_ggf = probs[:, 1]
                    p_bkg = probs[:, 2]
                    d_sig = (p_vbf + p_ggf).astype(np.float32)
                    d_vbf = (p_vbf / np.clip(d_sig, 1e-6, None)).astype(np.float32)

                    chunk["PNN_mass"] = masses_chunk
                    chunk["PNN_P_VBF"] = p_vbf
                    chunk["PNN_P_ggF"] = p_ggf
                    chunk["PNN_P_bkg"] = p_bkg
                    chunk["PNN_D_sig"] = d_sig
                    chunk["PNN_D_vbf"] = d_vbf

                    if out_tree is None:
                        # Let uproot infer robust branch types from first chunk
                        # (more resilient than manually mapping dtype for all branches).
                        fout[args.tree] = chunk
                        out_tree = fout[args.tree]
                    else:
                        out_tree.extend(chunk)
                    print(f"  Processed {stop}/{n_events}")

        print("Done.")

    print("\nAll files processed successfully.")


if __name__ == "__main__":
    main()
