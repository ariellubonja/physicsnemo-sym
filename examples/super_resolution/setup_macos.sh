#!/bin/bash
set -e

git clone -b apple-silicon-training https://github.com/ariellubonja/physicsnemo-sym.git
cd physicsnemo-sym

python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install -e .
pip install givernylocal

cd examples/super_resolution
python super_resolution.py

# BEFORE RUNNING:
# 1. Edit examples/super_resolution/conf/config.yaml and set your JHTDB access_token
# 2. If your Mac has less than 64 GB RAM, reduce domain_size (e.g. 64) and n_train (e.g. 32)
#    in config.yaml to avoid running Out of Memory
