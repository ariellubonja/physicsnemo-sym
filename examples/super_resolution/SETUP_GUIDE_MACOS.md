# Super Resolution Example — macOS (Apple Silicon) Setup Guide

This guide gets the 3D super-resolution example running on an Apple Silicon Mac (M1/M2/M3) using **CPU-only** PyTorch. Training will be slower than on a GPU, but it works for exploring the code and running short experiments.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- A JHTDB access token (get one at http://turbulence.pha.jhu.edu/)

## Setup

```bash
# 1. Clone and enter the repo
cd /path/to/physicsnemo-sym

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

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

The script will download training data from JHTDB, then start training the SRResNet model (32x32x32 -> 128x128x128 super-resolution).

## Faster Demo Run (Optional)

For a quicker test, you can reduce the dataset size and training steps. Edit `conf/config.yaml`:

```yaml
training:
  max_steps: 100        # default is much higher
  domain_size: 32       # smaller domain = faster per step
```

## What Was Changed for macOS Support

Two small changes were made to the codebase:

1. **`setup.py`** — CUDA extension build is wrapped in try/except so it's skipped when CUDA isn't available. The extension is only used for AMP float16 gradient scaling, which isn't needed on CPU.

2. **`physicsnemo/sym/distributed/manager.py`** — Unconditional `torch.cuda.device()` and `torch.cuda.empty_cache()` calls are now guarded behind `torch.cuda.is_available()`.

## Troubleshooting

- **`pip install -e .` fails**: Make sure you're using the version of this repo with the macOS fixes (check that `setup.py` has the try/except around `cuda_extension()`).
- **Import errors**: Run `python -c "import physicsnemo.sym"` to verify the package installed correctly.
- **Out of memory**: Reduce `domain_size` in the config. CPU RAM is shared with the system on Apple Silicon.
