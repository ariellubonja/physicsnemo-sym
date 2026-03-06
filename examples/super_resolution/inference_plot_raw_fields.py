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

"""Extract of inference_analysis.ipynb: load all models and plot raw fields."""

import glob
import math
import torch
import numpy as np
from torch import nn
from physicsnemo.sym.hydra.utils import compose
from physicsnemo.sym.hydra.amp import register_amp_configs
from physicsnemo.sym.key import Key
from physicsnemo.sym.hydra import instantiate_arch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Monkey-patch SRResNet to allow any power-of-2 scaling factor (needed for 16x, 32x)
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
scaling_factors = [4, 8, 16, 32]

# --- Load data ---
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

# Load low-res data for each scaling factor
U_lr = {}
for sf in scaling_factors:
    lr_step = f"step_{sf}_{sf}_{sf}"
    low_res = []
    for file in valid_files:
        data_low_res = np.load(file.replace("step_1_1_1", lr_step))
        low_res.append(np.rollaxis(data_low_res, -1, 0))
    U_lr[sf] = torch.from_numpy(np.stack(low_res, axis=0)).to(torch.float)

print(f"Loaded {len(high_res)} validation samples at {domain_size}^3")
for sf in scaling_factors:
    print(f"  {sf}x low-res: {list(U_lr[sf].shape)}")

# --- Model inference ---
U_pred = {}

for sf in scaling_factors:
    cfg = compose(
        config_path="conf",
        config_name="config",
        overrides=[f"arch.super_res.scaling_factor={sf}"],
    )
    model = instantiate_arch(
        input_keys=[Key("U_lr", size=3)],
        output_keys=[Key("U", size=3)],
        cfg=cfg.arch.super_res,
    )
    model.make_node(name="super_res")

    ckpt_dir = (
        "./outputs/super_resolution/"
        if sf == 4
        else f"./outputs/arch.super_res.scaling_factor={sf}/super_resolution/"
    )
    print(f"Loading {sf}x model from: {ckpt_dir}")
    model.load(ckpt_dir)
    model.eval().to(device)

    preds = []
    with torch.inference_mode():
        for i in range(U_lr[sf].shape[0]):
            out = model({"U_lr": U_lr[sf][i : i + 1].to(device)})
            preds.append(out["U"].cpu())
    U_pred[sf] = torch.cat(preds, dim=0)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  pred shape: {list(U_pred[sf].shape)}")

# --- Plot raw fields ---
import matplotlib.pyplot as plt

U_lr_ref = U_lr[4]
lr_mid = U_lr_ref.shape[-1] // 2
hr_mid = U_hr_true.shape[-1] // 2

sample_indices = [0, 4, 8, 12]
labels = ["Ground Truth", "4x SR", "8x SR", "16x SR", "32x SR", "Low Res (4x input)"]

fig, axs = plt.subplots(len(sample_indices), 6, figsize=(30, 20))

for row, i in enumerate(sample_indices):
    axs[row, 0].imshow(U_hr_true.numpy()[i, 0, ..., hr_mid], cmap="RdBu_r")
    for j, sf in enumerate([4, 8, 16, 32]):
        axs[row, j + 1].imshow(U_pred[sf].numpy()[i, 0, ..., hr_mid], cmap="RdBu_r")
    axs[row, 5].imshow(U_lr_ref.numpy()[i, 0, ..., lr_mid], cmap="RdBu_r")
    for col in range(1, 6):
        axs[row, col].set_yticks([])
    if row < len(sample_indices) - 1:
        for col in range(6):
            axs[row, col].set_xticks([])
    if row == 0:
        for col, label in enumerate(labels):
            axs[row, col].set_title(label)

plt.tight_layout()
plt.savefig("raw_fields.png", dpi=150)
print("Saved raw_fields.png")
plt.show()
