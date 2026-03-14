# Super Resolution Experiments

## Current work (branch: more-upscaling)

Investigating "deeper networks" for 8x turbulence super-resolution on JHTDB isotropic turbulence data.

### Key finding: residual block depth doesn't matter

Depth ablation (8x SR, all at 20K steps except 0-block at 5K):
| Blocks | L2 Error | Training Loss |
|--------|----------|--------------|
| 0      | 0.1604   | 0.024        |
| 2      | 0.1358   | 0.014        |
| 4      | 0.1359   | 0.014        |
| 8      | 0.1390   | 0.015        |
| 16     | 0.1402   | 0.018        |
| 24     | 0.1422   | 0.014        |

0 blocks is 18% worse, but 2-24 blocks are identical. Residual blocks operate at 16^3 where the receptive field saturates at 2 blocks.

### Current experiment: refinement layers between upsampling stages

The real bottleneck: the sub-pixel upsampling path has one Conv3d per 2x stage with zero processing at intermediate resolutions.

Architecture change (controlled by `custom.n_refine_per_stage`):
```
Before: 16^3 -> SubPixel -> 32^3 -> SubPixel -> 64^3 -> SubPixel -> 128^3
After:  16^3 -> SubPixel -> 32^3 -> [Conv+BN+PReLU x N] -> SubPixel -> 64^3 -> [Conv+BN+PReLU x N] -> SubPixel -> 128^3
```

Implementation: monkey-patch in `super_resolution.py` uses `_N_REFINE_PER_STAGE` global, set from `cfg.custom.n_refine_per_stage`. Same pattern in `inference_compare_depth.py`.

### Training commands (not yet run):
```bash
python super_resolution.py arch.super_res.scaling_factor=8 custom.n_refine_per_stage=1 training.max_steps=5000
python super_resolution.py arch.super_res.scaling_factor=8 custom.n_refine_per_stage=2 training.max_steps=5000
```

### Key files
| File | Role |
|------|------|
| `super_resolution.py` | Training script with monkey-patch |
| `conf/config.yaml` | Hydra config |
| `inference_compare_depth.py` | Comparison inference across depths/refinement |
| `inference_plot_raw_fields.py` | Raw fields visualization (all scaling factors) |
| `plot_scree.py` | Scree plots (`--mode scaling` or `--mode depth`) |
| `inference_utils.py` | TKE spectrum, continuity, P-Q-R functions |

### Trained model checkpoints (in outputs/):
- `super_resolution/` - 4x baseline (20K steps)
- `arch.super_res.scaling_factor=8/` - 8x baseline, 8 blocks (20K)
- `arch.super_res.scaling_factor=16/` - 16x baseline (20K)
- `arch.super_res.scaling_factor=32/` - 32x baseline (20K)
- `arch.super_res.n_resid_blocks={0,2,4,16,24},arch.super_res.scaling_factor=8/` - depth ablations
