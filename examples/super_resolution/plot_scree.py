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

"""Generate scree plots (loss vs training step) from TensorBoard logs.

Supports two modes:
  --mode scaling   Compare models across scaling factors (4x, 8x, 16x, 32x)
  --mode depth     Compare 8x models across residual block counts (0, 2, 4, 8, 16, 24)
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def read_loss(run_dir):
    """Read Train/loss_aggregated from TensorBoard events in run_dir."""
    event_files = glob.glob(os.path.join(run_dir, "**", "*events*"), recursive=True)
    if not event_files:
        return None, None
    ea = event_accumulator.EventAccumulator(
        event_files[0],
        size_guidance={event_accumulator.TENSORS: 0},
    )
    ea.Reload()
    tag = "Train/loss_aggregated"
    if tag not in ea.Tags()["tensors"]:
        return None, None
    events = ea.Tensors(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.tensor_proto.float_val[0] for e in events])
    return steps, values


def find_runs_by_scaling(output_dir):
    """Auto-detect runs by scaling factor. Returns list of (label, run_dir)."""
    runs = []
    # Default 4x run (no scaling_factor override in dir name)
    default_dir = os.path.join(output_dir, "super_resolution")
    if os.path.isdir(default_dir):
        runs.append(("4x", default_dir))
    for d in sorted(os.listdir(output_dir)):
        m = re.match(r"^arch\.super_res\.scaling_factor=(\d+)$", d)
        if m:
            sf = int(m.group(1))
            run_dir = os.path.join(output_dir, d, "super_resolution")
            if os.path.isdir(run_dir):
                runs.append((f"{sf}x", run_dir))
    return runs


def find_runs_by_depth(output_dir):
    """Auto-detect 8x runs with different n_resid_blocks. Returns list of (label, run_dir).
    When multiple runs exist for the same block count, pick the one with the most event data."""
    candidates = {}  # n_blocks -> list of run_dirs
    for d in sorted(os.listdir(output_dir)):
        # Default 8x baseline (n_resid_blocks=8, no override in dir name)
        if d == "arch.super_res.scaling_factor=8":
            run_dir = os.path.join(output_dir, d, "super_resolution")
            if os.path.isdir(run_dir):
                candidates.setdefault(8, []).append(run_dir)
            continue
        # Runs with explicit n_resid_blocks override and scaling_factor=8
        m_blocks = re.search(r"n_resid_blocks=(\d+)", d)
        m_sf = re.search(r"scaling_factor=8", d)
        if m_blocks and m_sf:
            n_blocks = int(m_blocks.group(1))
            run_dir = os.path.join(output_dir, d, "super_resolution")
            if os.path.isdir(run_dir):
                candidates.setdefault(n_blocks, []).append(run_dir)
    # For each block count, pick the run with the most event data
    runs = []
    for n_blocks in sorted(candidates):
        best_dir = None
        best_count = -1
        for run_dir in candidates[n_blocks]:
            event_files = glob.glob(os.path.join(run_dir, "**", "*events*"), recursive=True)
            count = len(event_files)
            # Use file size as tiebreaker (more training = bigger event file)
            size = sum(os.path.getsize(f) for f in event_files) if event_files else 0
            if size > best_count:
                best_count = size
                best_dir = run_dir
        if best_dir:
            runs.append((f"{n_blocks} blocks", best_dir))
    return runs


def main():
    parser = argparse.ArgumentParser(description="Generate scree plots from SR training logs")
    parser.add_argument(
        "--mode", choices=["scaling", "depth"], default="scaling",
        help="Compare across scaling factors or across depth (default: scaling)",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Base output directory (default: outputs)",
    )
    parser.add_argument(
        "--save-dir", default="scree_plots",
        help="Directory to save plots (default: scree_plots)",
    )
    args = parser.parse_args()

    if args.mode == "scaling":
        runs = find_runs_by_scaling(args.output_dir)
        title = "Training Loss Comparison Across Scaling Factors"
        filename = "scree_combined.png"
    else:
        runs = find_runs_by_depth(args.output_dir)
        title = "Training Loss: 8x SR Depth Comparison"
        filename = "scree_depth.png"

    if not runs:
        print(f"No runs found in {args.output_dir} for mode={args.mode}")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, run_dir in runs:
        steps, values = read_loss(run_dir)
        if steps is None:
            print(f"  No events for {label}, skipping")
            continue
        ax.plot(steps, values, linewidth=1.5, label=label)
        print(f"  {label}: {len(steps)} points, final loss = {values[-1]:.4f}")

    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
