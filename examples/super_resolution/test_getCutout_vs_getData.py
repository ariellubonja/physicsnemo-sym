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

"""
Compare getCutout vs getData for low-res queries on isotropic1024coarse.

getCutout: uses 1-based integer grid indices + server-side strides
getData:   uses float physical coordinates + interpolation methods

For high-res (stride=1), both should return identical values.
For low-res (stride=4), we want to verify whether getData at the exact
strided grid-point coordinates matches getCutout with stride=4.
"""

import numpy as np

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout, getData

# --- Configuration -----------------------------------------------------------
TOKEN = "edu.jhu.pha.turbulence.testing-201406"
DATASET_TITLE = "isotropic1024coarse"
FIELD = "velocity"
TIME_STEP = 1  # integer snapshot index (1-based)

# Physical constants for isotropic1024coarse
PHYSICAL_DOMAIN = 2 * np.pi
N_GRID = 1024
DX = PHYSICAL_DOMAIN / N_GRID
DT = 0.002  # physical time per snapshot

# Small box: 16x16x16 starting at grid index 100
# (small enough for the testing token's 4096-point limit)
START = np.array([100, 100, 100], dtype=int)
DOMAIN_SIZE = 16
END = START + DOMAIN_SIZE - 1  # [115, 115, 115]

LR_FACTOR = 4

# --- Setup -------------------------------------------------------------------
loader = turb_dataset(
    dataset_title=DATASET_TITLE,
    output_path="/tmp/jhtdb_test",
    auth_token=TOKEN,
)


def grid_index_to_physical(i):
    """Convert 1-based grid index to physical coordinate."""
    return (i - 1) * DX


def build_points_from_grid_indices(x_indices, y_indices, z_indices):
    """Build (N, 3) array of physical coordinates from 1-based grid indices."""
    x_phys = grid_index_to_physical(x_indices)
    y_phys = grid_index_to_physical(y_indices)
    z_phys = grid_index_to_physical(z_indices)
    # meshgrid with 'ij' indexing to match getCutout's (x, y, z) ordering
    xx, yy, zz = np.meshgrid(x_phys, y_phys, z_phys, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return points


def cutout_to_array(result):
    """Extract numpy array from getCutout xarray result."""
    return result.to_array()[0].values  # shape: (z, y, x, 3)


# =============================================================================
# Test 1: HIGH-RES (stride=1) — getCutout vs getData
# =============================================================================
print("=" * 70)
print("TEST 1: HIGH-RES (stride=1)")
print("=" * 70)

# getCutout: stride=1
axes_ranges_hr = np.array(
    [
        [START[0], END[0]],
        [START[1], END[1]],
        [START[2], END[2]],
        [TIME_STEP, TIME_STEP],
    ]
)
strides_hr = np.array([1, 1, 1, 1], dtype=int)

print("\nCalling getCutout (high-res, stride=1)...")
cutout_hr = getCutout(loader, FIELD, axes_ranges_hr, strides_hr)
cutout_hr_arr = cutout_to_array(cutout_hr)
print(f"  getCutout result shape: {cutout_hr_arr.shape}")
# getCutout returns (z, y, x, 3)

# getData: query at exact grid-point coordinates, stride=1
x_indices_hr = np.arange(START[0], END[0] + 1, 1)
y_indices_hr = np.arange(START[1], END[1] + 1, 1)
z_indices_hr = np.arange(START[2], END[2] + 1, 1)
points_hr = build_points_from_grid_indices(x_indices_hr, y_indices_hr, z_indices_hr)
physical_time = (TIME_STEP - 1) * DT

print(f"\nCalling getData (high-res, {len(points_hr)} points)...")
print(f"  physical_time = {physical_time}")
print(f"  x range: [{points_hr[:, 0].min():.6f}, {points_hr[:, 0].max():.6f}]")
print(f"  y range: [{points_hr[:, 1].min():.6f}, {points_hr[:, 1].max():.6f}]")
print(f"  z range: [{points_hr[:, 2].min():.6f}, {points_hr[:, 2].max():.6f}]")

getdata_hr = getData(
    loader,
    FIELD,
    physical_time,
    "none",  # temporal_method
    "none",  # spatial_method (no interpolation = nearest grid point)
    "field",  # spatial_operator
    points_hr,
)
# getData returns list of DataFrames; one per timepoint
df_hr = getdata_hr[0]
print(f"  getData result shape: {df_hr.shape}")
print(f"  getData columns: {list(df_hr.columns)}")

# Reshape getData result to match getCutout's (z, y, x, 3) layout
# getData returns in the order points were supplied (x-major from meshgrid 'ij')
# which is (nx, ny, nz) = (16, 16, 16), then we need to match getCutout's (z, y, x)
nx_hr = len(x_indices_hr)
ny_hr = len(y_indices_hr)
nz_hr = len(z_indices_hr)

getdata_hr_arr = df_hr.values.reshape(nx_hr, ny_hr, nz_hr, 3)
# getCutout returns (z, y, x, 3), getData meshgrid 'ij' gives (x, y, z, 3)
# Transpose to (z, y, x, 3)
getdata_hr_arr_reordered = getdata_hr_arr.transpose(2, 1, 0, 3)

print("\n--- HIGH-RES COMPARISON ---")
print(f"  getCutout shape: {cutout_hr_arr.shape}")
print(f"  getData shape (reordered): {getdata_hr_arr_reordered.shape}")
print(f"  Arrays equal: {np.array_equal(cutout_hr_arr, getdata_hr_arr_reordered)}")
print(f"  Max abs diff: {np.max(np.abs(cutout_hr_arr - getdata_hr_arr_reordered))}")
print(
    f"  Allclose (atol=1e-6): {np.allclose(cutout_hr_arr, getdata_hr_arr_reordered, atol=1e-6)}"
)

# Print a few sample values
print("\n  Sample values (first 3 points):")
print(f"    getCutout[0,0,0,:] = {cutout_hr_arr[0, 0, 0, :]}")
print(f"    getData  [0,0,0,:] = {getdata_hr_arr_reordered[0, 0, 0, :]}")
print(f"    getCutout[0,0,1,:] = {cutout_hr_arr[0, 0, 1, :]}")
print(f"    getData  [0,0,1,:] = {getdata_hr_arr_reordered[0, 0, 1, :]}")

# =============================================================================
# Test 2: LOW-RES (stride=4) — getCutout vs getData
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: LOW-RES (stride=4)")
print("=" * 70)

# getCutout: stride=4
strides_lr = np.array([LR_FACTOR, LR_FACTOR, LR_FACTOR, 1], dtype=int)

print("\nCalling getCutout (low-res, stride=4)...")
cutout_lr = getCutout(loader, FIELD, axes_ranges_hr, strides_lr)
cutout_lr_arr = cutout_to_array(cutout_lr)
print(f"  getCutout result shape: {cutout_lr_arr.shape}")

# getData: query at strided grid-point coordinates
x_indices_lr = np.arange(START[0], END[0] + 1, LR_FACTOR)
y_indices_lr = np.arange(START[1], END[1] + 1, LR_FACTOR)
z_indices_lr = np.arange(START[2], END[2] + 1, LR_FACTOR)
points_lr = build_points_from_grid_indices(x_indices_lr, y_indices_lr, z_indices_lr)

print(f"\nCalling getData (low-res, {len(points_lr)} points, stride=4 grid coords)...")
print(f"  x grid indices: {x_indices_lr}")
print(f"  y grid indices: {y_indices_lr}")
print(f"  z grid indices: {z_indices_lr}")
print(f"  x physical: {grid_index_to_physical(x_indices_lr)}")

getdata_lr = getData(
    loader,
    FIELD,
    physical_time,
    "none",  # temporal_method
    "none",  # spatial_method
    "field",  # spatial_operator
    points_lr,
)
df_lr = getdata_lr[0]
print(f"  getData result shape: {df_lr.shape}")

nx_lr = len(x_indices_lr)
ny_lr = len(y_indices_lr)
nz_lr = len(z_indices_lr)

getdata_lr_arr = df_lr.values.reshape(nx_lr, ny_lr, nz_lr, 3)
getdata_lr_arr_reordered = getdata_lr_arr.transpose(2, 1, 0, 3)

print("\n--- LOW-RES COMPARISON ---")
print(f"  getCutout shape: {cutout_lr_arr.shape}")
print(f"  getData shape (reordered): {getdata_lr_arr_reordered.shape}")
print(f"  Arrays equal: {np.array_equal(cutout_lr_arr, getdata_lr_arr_reordered)}")
print(f"  Max abs diff: {np.max(np.abs(cutout_lr_arr - getdata_lr_arr_reordered))}")
print(
    f"  Allclose (atol=1e-6): {np.allclose(cutout_lr_arr, getdata_lr_arr_reordered, atol=1e-6)}"
)

# Print sample values
print("\n  Sample values:")
print(f"    getCutout[0,0,0,:] = {cutout_lr_arr[0, 0, 0, :]}")
print(f"    getData  [0,0,0,:] = {getdata_lr_arr_reordered[0, 0, 0, :]}")
print(f"    getCutout[0,0,1,:] = {cutout_lr_arr[0, 0, 1, :]}")
print(f"    getData  [0,0,1,:] = {getdata_lr_arr_reordered[0, 0, 1, :]}")

# =============================================================================
# Test 3: Cross-check — do low-res getCutout values match subsampled high-res?
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: LOW-RES getCutout vs SUBSAMPLED HIGH-RES getCutout")
print("=" * 70)

# getCutout with stride=4 should give the same values as every-4th-point from stride=1
cutout_hr_subsampled = cutout_hr_arr[::LR_FACTOR, ::LR_FACTOR, ::LR_FACTOR, :]
print(f"  High-res subsampled shape: {cutout_hr_subsampled.shape}")
print(f"  Low-res getCutout shape:   {cutout_lr_arr.shape}")
print(f"  Arrays equal: {np.array_equal(cutout_hr_subsampled, cutout_lr_arr)}")
print(f"  Max abs diff: {np.max(np.abs(cutout_hr_subsampled - cutout_lr_arr))}")

# =============================================================================
# Test 4: Cross-check — do low-res getData values match subsampled high-res getData?
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: LOW-RES getData vs SUBSAMPLED HIGH-RES getData")
print("=" * 70)

getdata_hr_subsampled = getdata_hr_arr_reordered[
    ::LR_FACTOR, ::LR_FACTOR, ::LR_FACTOR, :
]
print(f"  High-res getData subsampled shape: {getdata_hr_subsampled.shape}")
print(f"  Low-res getData shape:             {getdata_lr_arr_reordered.shape}")
print(
    f"  Arrays equal: {np.array_equal(getdata_hr_subsampled, getdata_lr_arr_reordered)}"
)
print(
    f"  Max abs diff: {np.max(np.abs(getdata_hr_subsampled - getdata_lr_arr_reordered))}"
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(
    f"  High-res getCutout vs getData match: {np.allclose(cutout_hr_arr, getdata_hr_arr_reordered, atol=1e-6)}"
)
print(
    f"  Low-res  getCutout vs getData match: {np.allclose(cutout_lr_arr, getdata_lr_arr_reordered, atol=1e-6)}"
)
print(
    f"  Low-res getCutout = subsampled high-res getCutout: {np.array_equal(cutout_hr_subsampled, cutout_lr_arr)}"
)
print(
    f"  Low-res getData = subsampled high-res getData: {np.array_equal(getdata_hr_subsampled, getdata_lr_arr_reordered)}"
)
