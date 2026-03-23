import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset

# ============================================================
# Load physics statistics (ONCE)
# ============================================================

EVENTS_JSON = "events_6Feb.json"

with open(EVENTS_JSON) as f:
    EVENT_INFO = json.load(f)

# ------------------------------------------------------------
# Build background file → normalized physics weights
# ------------------------------------------------------------

BKG_WEIGHTS = {}
total_bkg = 0.0

for entry in EVENT_INFO:
    fname = os.path.basename(entry["file"])
    n = entry["entries"]

    if any(
        key in fname
        for key in [
            "DY", "drellyan", "TT", "WZ", "ZZ", "WW",
            "ST", "WJets"
        ]
    ):
        BKG_WEIGHTS[fname] = n
        total_bkg += n

# Normalize
for k in BKG_WEIGHTS:
    BKG_WEIGHTS[k] /= total_bkg


# ============================================================
# Dataset
# ============================================================

class BalancedChunkIterableDataset(IterableDataset):
    """
    Physics-aware balanced streaming dataset.

    Labels:
      0 → VBF
      1 → ggF
      2 → Background
    """

    def __init__(
        self,
        chunk_dirs,            # dict: {label: list[str]}
        batch_size,
        shuffle_chunks=True,
    ):
        self.chunk_dirs = chunk_dirs
        self.batch_size = batch_size
        self.shuffle_chunks = shuffle_chunks

        self.labels = list(chunk_dirs.keys())

        # ----------------------------------------------------
        # Collect chunk files per label (support multiple dirs)
        # ----------------------------------------------------
        self.chunk_files = {}

        for label, dirs in chunk_dirs.items():
            if not isinstance(dirs, (list, tuple)):
                raise TypeError(
                    f"chunk_dirs[{label}] must be a list of directories"
                )

            files = []
            for d in dirs:
                files.extend(glob.glob(f"{d}/chunk_*.npz"))

            if len(files) == 0:
                raise RuntimeError(
                    f"No chunk files found for label {label} in {dirs}"
                )

            self.chunk_files[label] = sorted(files)

    # --------------------------------------------------------
    # Infinite event stream per class
    # --------------------------------------------------------
    def _event_stream(self, label):
        """Infinite generator of events for one class"""

        files = self.chunk_files[label].copy()

        # Background → physics-weighted file sampling
        if label == 2:
            weights = [
                BKG_WEIGHTS.get(os.path.basename(f), 1e-12)
                for f in files
            ]
        else:
            weights = None

        while True:

            # ----------------------------
            # Background (random, physics-weighted)
            # ----------------------------
            if label == 2:
                f = random.choices(files, weights=weights, k=1)[0]
                data = np.load(f)

                X = data["X"]
                y = data["y"]
                m = data["m"]

                idx = np.random.randint(len(X))
                yield X[idx], m[idx], y[idx]

            # ----------------------------
            # Signal (cycle deterministically)
            # ----------------------------
            else:
                if self.shuffle_chunks:
                    random.shuffle(files)

                for f in files:
                    data = np.load(f)

                    X = data["X"]
                    y = data["y"]
                    m = data["m"]

                    perm = np.random.permutation(len(X))
                    X, y, m = X[perm], y[perm], m[perm]

                    for i in range(len(X)):
                        yield X[i], m[i], y[i]

    # --------------------------------------------------------
    # Balanced batch iterator
    # --------------------------------------------------------
    def __iter__(self):

        streams = {
            label: self._event_stream(label)
            for label in self.labels
        }

        # Enforce balanced batches (1/3 each)
        n_vbf = self.batch_size // 3
        n_ggf = self.batch_size // 3
        n_bkg = self.batch_size - n_vbf - n_ggf

        while True:
            X_batch, m_batch, y_batch = [], [], []

            for _ in range(n_vbf):
                x, m, y = next(streams[0])
                X_batch.append(x)
                m_batch.append(m)
                y_batch.append(y)

            for _ in range(n_ggf):
                x, m, y = next(streams[1])
                X_batch.append(x)
                m_batch.append(m)
                y_batch.append(y)

            for _ in range(n_bkg):
                x, m, y = next(streams[2])
                X_batch.append(x)
                m_batch.append(m)
                y_batch.append(y)

            perm = np.random.permutation(len(X_batch))

            yield (
                torch.tensor(np.array(X_batch)[perm], dtype=torch.float32),
                torch.tensor(np.array(m_batch)[perm], dtype=torch.float32),
                torch.tensor(np.array(y_batch)[perm], dtype=torch.long),
            )

