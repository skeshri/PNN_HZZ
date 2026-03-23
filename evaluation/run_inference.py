# evaluation/run_inference.py
import torch
import numpy as np
from training.model import PNN
from training.dataset import PNN_Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/pnn_hzz.pt"
DATASET_PATH = "preprocessing/dataset_raw.npz"
NORM_PATH = "preprocessing/norm.npz"

# -----------------------------------
# Choose mass hypothesis here
# -----------------------------------
MASS_HYPOTHESIS = 800.0   # GeV
#MASS_HYPOTHESIS = 650.0   # GeV


# -----------------------------------
# Load dataset (with normalization)
# -----------------------------------
dataset = PNN_Dataset(DATASET_PATH, norm_file=NORM_PATH)

X = dataset.X.to(DEVICE)
y = dataset.y.numpy()

# Override mass parameter
m = torch.full((len(dataset), 1),
               (MASS_HYPOTHESIS - 1000.0) / 500.0,
               device=DEVICE)

# -----------------------------------
# Load model
# -----------------------------------
model = PNN(n_features=X.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------------
# Run inference
# -----------------------------------
with torch.no_grad():
    probs = model(X, m).cpu().numpy()

# -----------------------------------
# Save outputs
# -----------------------------------
np.savez(
    f"evaluation/pnn_output_m{int(MASS_HYPOTHESIS)}.npz",
    P_VBF=probs[:, 0],
    P_ggF=probs[:, 1],
    P_bkg=probs[:, 2],
    label=y
)

print(f"Inference done for mH = {MASS_HYPOTHESIS} GeV")

