#!/bin/bash

echo "===== Job started on $(date) ====="
echo "Running on: $(hostname)"
echo "Current dir: $(pwd)"

# Safety
set -e
set -o pipefail

# Activate Conda ONLY
source /eos/home-s/skeshri/miniconda3/etc/profile.d/conda.sh
conda activate pnn_hzz

# Go to project root
cd /afs/cern.ch/work/s/skeshri/Analysis/Sumit/PNN_HZZ

# Run training
python -m training.train_streaming

echo "===== Job finished on $(date) ====="

