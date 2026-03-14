# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare 8x super-resolution models with different depths (n_resid_blocks).

Loads baseline (8 blocks), deeper-16, and deeper-24 models, runs inference,
and produces:
  1. Raw field comparison plot (side-by-side slices)
  2. TKE spectrum overlay
"""

import glob
import math
import torch
import numpy as np
from torch import nn
from physicsnemo.sym.hydra.utils import compose
from physicsnemo.sym.hydra.amp import register_amp_configs
from physicsnemo.sym.key import Key
from physicsnemo.sym.hydra import instantiate_arch

from inference_utils import compute_tke_spectrum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Monkey-patch SRResNet to allow any power-of-2 scaling factor
import physicsnemo.models.srrn.super_res_net as _srn
from physicsnemo.models.srrn.super_res_net import (
    ConvolutionalBlock3d,
    ResidualConvBlock3d,
    SubPixel_ConvolutionalBlock3d,
    MetaData,
)
from physicsnemo.models.layers import get_activation


def _patched_init(
    self,
    in_channels,
    out_channels,
    large_kernel_size=7,
    small_kernel_size=3,
    conv_layer_size=32,
    n_resid_blocks=8,
    scaling_factor=8,
    activation_fn="prelu",
):
    _srn.Module.__init__(self, meta=MetaData())
    self.var_dim = 1
    if isinstance(activation_fn, str):
        activation_fn = get_activation(activation_fn)
    scaling_factor = int(scaling_factor)
    if scaling_factor < 2 or (scaling_factor & (scaling_factor - 1)) != 0:
        raise ValueError("The scaling factor must be a power of 2, >= 2!")
    self.conv_block1 = ConvolutionalBlock3d(
        in_channels=in_channels,
        out_channels=conv_layer_size,
        kernel_size=large_kernel_size,
        batch_norm=False,
        activation_fn=activation_fn,
    )
    self.residual_blocks = nn.Sequential(
        *[
            ResidualConvBlock3d(
                n_layers=2,
                kernel_size=small_kernel_size,
                conv_layer_size=conv_layer_size,
                activation_fn=activation_fn,
            )
            for _ in range(n_resid_blocks)
        ]
    )
    self.conv_block2 = ConvolutionalBlock3d(
        in_channels=conv_layer_size,
        out_channels=conv_layer_size,
        kernel_size=small_kernel_size,
        batch_norm=True,
    )
    n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
    self.subpixel_convolutional_blocks = nn.Sequential(
        *[
            SubPixel_ConvolutionalBlock3d(
                kernel_size=small_kernel_size,
                conv_layer_size=conv_layer_size,
                scaling_factor=2,
            )
            for _ in range(n_subpixel_convolution_blocks)
        ]
    )
    self.conv_block3 = ConvolutionalBlock3d(
        in_channels=conv_layer_size,
        out_channels=out_channels,
        kernel_size=large_kernel_size,
        batch_norm=False,
    )


_srn.SRResNet.__init__ = _patched_init

register_amp_configs()

# --- Experiment definitions ---
experiments = [
    {
        "label": "0 blocks",
        "overrides": [
            "arch.super_res.scaling_factor=8",
            "arch.super_res.n_resid_blocks=0",
        ],
        "ckpt_dir": "./outputs/arch.super_res.n_resid_blocks=0,arch.super_res.scaling_factor=8,training.max_steps=5000/super_resolution/",
    },
    {
        "label": "2 blocks",
        "overrides": [
            "arch.super_res.scaling_factor=8",
            "arch.super_res.n_resid_blocks=2",
        ],
        "ckpt_dir": "./outputs/arch.super_res.n_resid_blocks=2,arch.super_res.scaling_factor=8/super_resolution/",
    },
    {
        "label": "4 blocks",
        "overrides": [
            "arch.super_res.scaling_factor=8",
            "arch.super_res.n_resid_blocks=4",
        ],
        "ckpt_dir": "./outputs/arch.super_res.n_resid_blocks=4,arch.super_res.scaling_factor=8/super_resolution/",
    },
    {
        "label": "8 blocks",
        "overrides": ["arch.super_res.scaling_factor=8"],
        "ckpt_dir": "./outputs/arch.super_res.scaling_factor=8/super_resolution/",
    },
    {
        "label": "16 blocks",
        "overrides": [
            "arch.super_res.scaling_factor=8",
            "arch.super_res.n_resid_blocks=16",
        ],
        "ckpt_dir": "./outputs/arch.super_res.n_resid_blocks=16,arch.super_res.scaling_factor=8/super_resolution/",
    },
    {
        "label": "24 blocks",
        "overrides": [
            "arch.super_res.scaling_factor=8",
            "arch.super_res.n_resid_blocks=24",
        ],
        "ckpt_dir": "./outputs/arch.super_res.n_resid_blocks=24,arch.super_res.scaling_factor=8/super_resolution/",
    },
]

# --- Load data ---
sf = 8
domain_size = 128
path = "./datasets/jhtdb_valid"
hr_files = sorted(glob.glob(path + "/*_step_1_1_1.npy"))

high_res = []
valid_files = []
for file in hr_files:
    data = np.load(file)
    if data.shape[0] != domain_size:
        continue
    high_res.append(np.rollaxis(data, -1, 0))
    valid_files.append(file)

U_hr_true = torch.from_numpy(np.stack(high_res, axis=0)).to(torch.float)

lr_step = f"step_{sf}_{sf}_{sf}"
low_res = []
for file in valid_files:
    data_low_res = np.load(file.replace("step_1_1_1", lr_step))
    low_res.append(np.rollaxis(data_low_res, -1, 0))
U_lr = torch.from_numpy(np.stack(low_res, axis=0)).to(torch.float)

print(f"Loaded {len(high_res)} validation samples at {domain_size}^3")
print(f"  8x low-res: {list(U_lr.shape)}")

# --- Model inference ---
U_pred = {}

for exp in experiments:
    label = exp["label"]
    cfg = compose(
        config_path="conf",
        config_name="config",
        overrides=exp["overrides"],
    )
    model = instantiate_arch(
        input_keys=[Key("U_lr", size=3)],
        output_keys=[Key("U", size=3)],
        cfg=cfg.arch.super_res,
    )
    model.make_node(name="super_res")

    print(f"Loading {label} from: {exp['ckpt_dir']}")
    model.load(exp["ckpt_dir"])
    model.eval().to(device)

    preds = []
    with torch.inference_mode():
        for i in range(U_lr.shape[0]):
            out = model({"U_lr": U_lr[i : i + 1].to(device)})
            preds.append(out["U"].cpu())
    U_pred[label] = torch.cat(preds, dim=0)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  pred shape: {list(U_pred[label].shape)}")

# --- Plot 1: Raw field comparison ---
import matplotlib.pyplot as plt

hr_mid = U_hr_true.shape[-1] // 2
lr_mid = U_lr.shape[-1] // 2

sample_indices = [0, 4, 8, 12]
exp_labels = [exp["label"] for exp in experiments]
col_labels = ["Ground Truth"] + exp_labels + ["Low Res"]
n_cols = len(col_labels)

fig, axs = plt.subplots(len(sample_indices), n_cols, figsize=(6 * n_cols, 5 * len(sample_indices)))

for row, i in enumerate(sample_indices):
    # Ground truth
    axs[row, 0].imshow(U_hr_true.numpy()[i, 0, ..., hr_mid], cmap="RdBu_r")
    # Model predictions
    for j, label in enumerate(exp_labels):
        axs[row, j + 1].imshow(U_pred[label].numpy()[i, 0, ..., hr_mid], cmap="RdBu_r")
    # Low res
    axs[row, n_cols - 1].imshow(U_lr.numpy()[i, 0, ..., lr_mid], cmap="RdBu_r")
    # Y-axis only on left column
    for col in range(1, n_cols):
        axs[row, col].set_yticks([])
    # X-axis only on bottom row
    if row < len(sample_indices) - 1:
        for col in range(n_cols):
            axs[row, col].set_xticks([])
    # Titles only on top row
    if row == 0:
        for col, label in enumerate(col_labels):
            axs[row, col].set_title(label)

plt.tight_layout()
plt.savefig("depth_comparison_fields.png", dpi=150)
print("Saved depth_comparison_fields.png")

# --- Plot 2: TKE spectrum comparison ---
fig, ax = plt.subplots(figsize=(10, 7))

# Ground truth spectrum (average over samples)
tke_spectra_gt = []
for i in range(U_hr_true.shape[0]):
    _, _, wn, tke = compute_tke_spectrum(U_hr_true.numpy()[i])
    tke_spectra_gt.append(tke)
tke_mean_gt = np.mean(tke_spectra_gt, axis=0)
ax.plot(wn, tke_mean_gt, "k-", linewidth=2, label="Ground Truth")

# Each model's spectrum
colors = ["tab:blue", "tab:orange", "tab:green"]
for idx, (label, color) in enumerate(zip(exp_labels, colors)):
    tke_spectra = []
    for i in range(U_pred[label].shape[0]):
        _, _, wn, tke = compute_tke_spectrum(U_pred[label].numpy()[i])
        tke_spectra.append(tke)
    tke_mean = np.mean(tke_spectra, axis=0)
    ax.plot(wn, tke_mean, color=color, linewidth=1.5, label=label)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Wavenumber k")
ax.set_ylabel("E(k)")
ax.set_title("TKE Spectrum: 8x SR Depth Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("depth_comparison_tke.png", dpi=150)
print("Saved depth_comparison_tke.png")

# --- Print L2 relative errors ---
print("\n=== L2 Relative Error (per sample, then mean) ===")
for label in exp_labels:
    errors = []
    for i in range(U_hr_true.shape[0]):
        err = torch.norm(U_pred[label][i] - U_hr_true[i]) / torch.norm(U_hr_true[i])
        errors.append(err.item())
    print(f"  {label}: {np.mean(errors):.4f} +/- {np.std(errors):.4f}")
