# Super Resolution Example — macOS (Apple Silicon) Setup Guide

This guide gets the 3D super-resolution example running on an Apple Silicon Mac (M1/M2/M3) using **CPU-only** PyTorch. Training is slower than on a GPU but works for exploring the code and running short experiments.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3) with at least 16 GB RAM
- Python 3.10+
- A JHTDB access token (get one at http://turbulence.pha.jhu.edu/)

## Setup

```bash
# 1. Clone and enter the repo
cd /path/to/physicsnemo-sym

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch (CPU backend — no CUDA needed)
pip install torch torchvision torchaudio

# 4. Install physicsnemo-sym in editable mode
#    The CUDA extension will be skipped automatically on non-CUDA systems.
pip install -e .

# 5. Install givernylocal for JHTDB data download
pip install givernylocal
```

## Configure JHTDB Access

Edit `examples/super_resolution/conf/config.yaml` and set your JHTDB access token:

```yaml
jhtdb:
  access_token: "your-token-here"
```

## Run

```bash
cd examples/super_resolution
python super_resolution.py
```

The script will download training data from JHTDB on the first run (takes a few minutes), then start training the SRResNet model (16x16x16 -> 64x64x64 super-resolution, 4x scaling factor).

Training runs at ~11s per step on CPU. With 1000 max_steps, expect ~3 hours for a full run.

## Config Defaults (tuned for 16 GB RAM)

The config is already set for macOS-friendly sizes:

| Parameter      | Value | Notes                                      |
|----------------|-------|--------------------------------------------|
| `domain_size`  | 64    | 128 requires 32-64 GB RAM                  |
| `n_train`      | 32    | 32 samples x 64^3 x 3 x 4 bytes ~ 0.1 GB  |
| `n_valid`      | 4     |                                             |
| `batch_size`   | 1     | Keeps activation memory low on CPU          |
| `max_steps`    | 1000  |                                             |

To scale up on a machine with more RAM (32-64 GB), increase `domain_size` to 128,
`n_train` to 512, and `batch_size` to 4 (the original GPU defaults).

## What Was Changed for macOS Support

Three changes were made to the codebase:

1. **`setup.py`** — CUDA extension build is wrapped in try/except so it's skipped when CUDA isn't available. The extension is only used for AMP float16 gradient scaling, which isn't needed on CPU.

2. **`physicsnemo/sym/distributed/manager.py`** — Unconditional `torch.cuda.device()` and `torch.cuda.empty_cache()` calls are now guarded behind `torch.cuda.is_available()`.

3. **`physicsnemo/sym/__init__.py`** — `torch.cuda.nvtx.range_push` and `range_pop` are replaced with no-ops when CUDA is not available, since NVTX profiling annotations are used throughout the training loop.

## Troubleshooting

- **`pip install -e .` fails**: Make sure you're using the version of this repo with the macOS fixes (check that `setup.py` has the try/except around `cuda_extension()`).
- **Import errors**: Run `python -c "import physicsnemo.sym"` to verify the package installed correctly.
- **Out of memory / system becomes unresponsive**: Reduce `domain_size` in the config. The 3D SRResNet's activation memory during forward/backward scales as O(domain_size^3). On Apple Silicon, CPU RAM is shared with the system, so OOM manifests as system-wide lag before the process gets killed (signal 137).
