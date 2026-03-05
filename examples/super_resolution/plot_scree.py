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

"""Generate scree plots (loss vs training step) from TensorBoard logs for
super-resolution models trained at different scaling factors."""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard_events(path):
    """Read TensorBoard event files and return dict of {tag: {step, value}}."""
    event_files = glob.glob(os.path.join(path, "*events*"))
    assert event_files, f"No event files found in {path}"

    ea = event_accumulator.EventAccumulator(
        event_files[0],
        size_guidance={event_accumulator.TENSORS: 0},
    )
    ea.Reload()
    tags = ea.Tags()["tensors"]

    data = {}
    for tag in tags:
        if tag == "config/text_summary":
            continue
        events = ea.Tensors(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.tensor_proto.float_val[0] for e in events])
        data[tag] = {"step": steps, "value": values}
    return data


def find_events_dir(base_output_dir, scaling_factor):
    """Locate the TensorBoard events directory for a given scaling factor."""
    run_dir = os.path.join(
        base_output_dir,
        f"arch.super_res.scaling_factor={scaling_factor}",
        "super_resolution",
    )
    if not os.path.isdir(run_dir):
        return None
    # Events are typically in a subdirectory or directly in run_dir
    # Search recursively for event files
    event_files = glob.glob(os.path.join(run_dir, "**", "*events*"), recursive=True)
    if not event_files:
        return None
    # Return the directory containing the first event file
    return os.path.dirname(event_files[0]) + "/"


def plot_single_model(data, scaling_factor, save_dir):
    """Plot loss curves for a single model."""
    os.makedirs(save_dir, exist_ok=True)

    keys = list(data.keys())
    train_loss_keys = [k for k in keys if "Train" in k and "loss" in k.lower()]
    val_keys = [k for k in keys if "Validators" in k or "Valid" in k]
    plot_keys = train_loss_keys + val_keys

    if not plot_keys:
        # Fallback: plot all keys except learning rate
        plot_keys = [k for k in keys if "learning_rate" not in k]

    fig, axs = plt.subplots(len(plot_keys), 1, figsize=(8, 4 * len(plot_keys)))
    if len(plot_keys) == 1:
        axs = [axs]

    for i, key in enumerate(plot_keys):
        axs[i].plot(data[key]["step"], data[key]["value"], linewidth=1)
        axs[i].set_yscale("log")
        axs[i].set_xlabel("Training Step")
        axs[i].set_ylabel("Loss")
        axs[i].set_title(f"{key} ({scaling_factor}x)")
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"scree_{scaling_factor}x.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_combined(all_data, save_dir):
    """Plot combined comparison of Train/loss_aggregated across scaling factors."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for sf, data in sorted(all_data.items()):
        loss_key = "Train/loss_aggregated"
        if loss_key not in data:
            # Try to find any loss key
            loss_key = next((k for k in data if "loss" in k.lower()), None)
        if loss_key is None:
            print(f"  Warning: no loss key found for {sf}x, skipping")
            continue
        ax.plot(data[loss_key]["step"], data[loss_key]["value"],
                label=f"{sf}x", linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss Comparison Across Scaling Factors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "scree_combined.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate scree plots from SR training logs")
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Base output directory containing run subdirectories (default: outputs)",
    )
    parser.add_argument(
        "--scaling-factors", nargs="+", type=int, default=None,
        help="Scaling factors to plot (default: auto-detect from output dir)",
    )
    parser.add_argument(
        "--save-dir", default="scree_plots",
        help="Directory to save plots (default: scree_plots)",
    )
    args = parser.parse_args()

    # Auto-detect scaling factors if not specified
    if args.scaling_factors is None:
        args.scaling_factors = []
        if os.path.isdir(args.output_dir):
            for d in os.listdir(args.output_dir):
                if d.startswith("arch.super_res.scaling_factor="):
                    try:
                        sf = int(d.split("=")[1])
                        args.scaling_factors.append(sf)
                    except ValueError:
                        pass
        args.scaling_factors.sort()

    if not args.scaling_factors:
        print("No scaling factors found. Check --output-dir or specify --scaling-factors.")
        sys.exit(1)

    print(f"Scaling factors: {args.scaling_factors}")

    all_data = {}
    for sf in args.scaling_factors:
        events_dir = find_events_dir(args.output_dir, sf)
        if events_dir is None:
            print(f"  No events found for {sf}x, skipping")
            continue
        print(f"  Reading {sf}x from {events_dir}")
        data = read_tensorboard_events(events_dir)
        all_data[sf] = data
        plot_single_model(data, sf, args.save_dir)

    if len(all_data) > 1:
        plot_combined(all_data, args.save_dir)

    print("Done!")


if __name__ == "__main__":
    main()
