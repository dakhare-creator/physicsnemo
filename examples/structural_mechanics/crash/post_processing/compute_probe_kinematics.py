# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
Derives and plots velocity and acceleration from averaged position data over two probe point sets
('driver' and 'passenger'), producing four curves per plot:
- Driver (Ground Truth), Driver (Predicted), Passenger (Ground Truth), Passenger (Predicted).
"""

import os
import re
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt


def extract_timestep_from_path(file_path: str) -> str:
    filename = os.path.basename(file_path)
    numbers = re.findall(r"\d+", filename)
    return numbers[-1] if numbers else "0"


def parse_point_set(spec: str) -> List[int]:
    """
    Parse a comma/space-separated list of integers and inclusive ranges like '70658-70659'.
    Example: '70658-70659, 70664, 70676-70679' -> [70658, 70659, 70664, 70676, 70677, 70678, 70679]
    """
    if not spec:
        return []
    ids: List[int] = []
    for token in re.split(r"[,\s]+", spec.strip()):
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if b < a:
                a, b = b, a
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(token))
    return sorted(set(ids))


def load_averaged_series(
    vtp_dir: str, point_ids: List[int], dt: float, position_array: str
) -> pd.DataFrame | None:
    """
    Load VTPs from vtp_dir, average positions over point_ids for the given position_array ('prediction' or 'exact'),
    and return a DataFrame with Time, Position, Velocity, Acceleration (per axis).
    """
    if not os.path.isdir(vtp_dir):
        print(f"❌ Error: Directory not found: {vtp_dir}")
        return None

    try:
        vtp_files: Dict[int, str] = {
            int(extract_timestep_from_path(f)): os.path.join(vtp_dir, f)
            for f in os.listdir(vtp_dir)
            if f.lower().endswith(".vtp")
        }
    except (ValueError, TypeError):
        print(
            f"❌ Error: Could not extract integer timesteps from filenames in {vtp_dir}."
        )
        return None

    if not vtp_files:
        print(f"❌ Error: No .vtp files found in {vtp_dir}")
        return None

    if not point_ids:
        print("❌ Error: Empty point set provided.")
        return None

    sorted_timesteps = sorted(vtp_files.keys())
    rows = []
    for step in sorted_timesteps:
        filepath = vtp_files[step]
        try:
            mesh = pv.read(filepath, progress_bar=False)
            if position_array not in mesh.point_data:
                continue

            arr = mesh.point_data[position_array]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            valid_ids = [pid for pid in point_ids if 0 <= pid < mesh.n_points]
            if not valid_ids:
                continue

            subset = arr[valid_ids]  # (k, dim)
            avg = subset.mean(axis=0)  # (dim,)
            avg3 = np.zeros(3, dtype=float)
            avg3[: min(3, avg.shape[0])] = avg[: min(3, avg.shape[0])]
            avg3 /= 1000.0  # convert to m

            rows.append(
                {
                    "Time (s)": step * dt,
                    "Timestep": step,
                    "Position_X": float(avg3[0]),
                    "Position_Y": float(avg3[1]),
                    "Position_Z": float(avg3[2]),
                }
            )
        except Exception as e:
            print(f"❌ Error processing file {filepath}: {e}")

    if not rows:
        print("Could not extract any averaged position data.")
        return None

    df = pd.DataFrame(rows).sort_values(by="Time (s)").reset_index(drop=True)

    # Derive velocity and acceleration using central differences
    for axis in ["X", "Y", "Z"]:
        df[f"Velocity_{axis}"] = np.gradient(df[f"Position_{axis}"], df["Time (s)"])
        df[f"Acceleration_{axis}"] = np.gradient(df[f"Velocity_{axis}"], df["Time (s)"])

    df.fillna(0, inplace=True)
    return df


def discover_runs(parent_dir: str) -> List[str]:
    """
    Return sorted list of directories under parent_dir that contain at least one .vtp file.
    Used to find all sample/run subdirectories for aggregation.
    """
    if not parent_dir or not os.path.isdir(parent_dir):
        return []
    runs: List[str] = []
    for root, _dirs, files in os.walk(parent_dir):
        if any(f.lower().endswith(".vtp") for f in files):
            runs.append(root)
    return sorted(set(runs))


def compute_squared_error_series(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame
) -> pd.DataFrame | None:
    """
    Compute squared L2 error at each time for position, velocity, and acceleration.
    Returns a DataFrame with Time (s), Pos_sqerr, Vel_sqerr, Acc_sqerr (scalar per time).
    """
    common_time = np.intersect1d(gt_df["Time (s)"].values, pred_df["Time (s)"].values)
    if common_time.size == 0:
        return None
    gt_aligned = gt_df.set_index("Time (s)").loc[common_time].sort_index()
    pred_aligned = pred_df.set_index("Time (s)").loc[common_time].sort_index()
    pos_cols = [c for c in gt_aligned.columns if c.startswith("Position_")]
    vel_cols = [c for c in gt_aligned.columns if c.startswith("Velocity_")]
    acc_cols = [c for c in gt_aligned.columns if c.startswith("Acceleration_")]
    if not (pos_cols and vel_cols and acc_cols):
        return None
    pos_gt = gt_aligned[pos_cols].values
    pos_pred = pred_aligned[pos_cols].values
    vel_gt = gt_aligned[vel_cols].values
    vel_pred = pred_aligned[vel_cols].values
    acc_gt = gt_aligned[acc_cols].values
    acc_pred = pred_aligned[acc_cols].values
    pos_sqerr = np.sum((pos_pred - pos_gt) ** 2, axis=1)
    vel_sqerr = np.sum((vel_pred - vel_gt) ** 2, axis=1)
    acc_sqerr = np.sum((acc_pred - acc_gt) ** 2, axis=1)
    return pd.DataFrame(
        {
            "Time (s)": common_time,
            "Pos_sqerr": pos_sqerr,
            "Vel_sqerr": vel_sqerr,
            "Acc_sqerr": acc_sqerr,
        }
    )


def compute_mse_over_samples(
    list_of_sqerr: List[pd.DataFrame],
) -> Optional[Tuple[float, float, float]]:
    """
    Given per-sample squared-error DataFrames (Pos_sqerr, Vel_sqerr, Acc_sqerr),
    return MSE over all samples and all time: (mse_pos, mse_vel, mse_acc).
    """
    if not list_of_sqerr:
        return None
    all_pos = np.concatenate([df["Pos_sqerr"].values for df in list_of_sqerr])
    all_vel = np.concatenate([df["Vel_sqerr"].values for df in list_of_sqerr])
    all_acc = np.concatenate([df["Acc_sqerr"].values for df in list_of_sqerr])
    return (
        float(np.mean(all_pos)),
        float(np.mean(all_vel)),
        float(np.mean(all_acc)),
    )


def plot_kinematics(
    driver_gt: pd.DataFrame,
    driver_pred: pd.DataFrame,
    passenger_gt: pd.DataFrame,
    passenger_pred: pd.DataFrame,
    output_plot: str,
):
    """
    2x3 plots; each subplot shows two curves (GT vs Pred) for X only:
    - Row 0: Driver (red solid = GT, red dashed = Pred)
    - Row 1: Passenger (blue solid = GT, blue dashed = Pred)
    Columns: Displacement X, Velocity X, Acceleration X
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    fig.suptitle(
        "Kinematics (x-direction ): Driver (top) vs Passenger (bottom) toe pan| Ground Truth vs Predicted",
        fontsize=16,
    )

    def get_limits(dfs: List[pd.DataFrame], col: str):
        vals = np.concatenate([df[col].to_numpy() for df in dfs if col in df])
        if vals.size == 0:
            return (-1, 1)
        vmin, vmax = float(vals.min()), float(vals.max())
        margin = (vmax - vmin) * 0.05
        if margin == 0:
            margin = 1.0
        return (vmin - margin, vmax + margin)

    # Shared limits per row (driver vs passenger) for each X component
    pos_lim_driver = get_limits([driver_gt, driver_pred], "Position_X")
    vel_lim_driver = get_limits([driver_gt, driver_pred], "Velocity_X")
    acc_lim_driver = get_limits([driver_gt, driver_pred], "Acceleration_X")

    pos_lim_pass = get_limits([passenger_gt, passenger_pred], "Position_X")
    vel_lim_pass = get_limits([passenger_gt, passenger_pred], "Velocity_X")
    acc_lim_pass = get_limits([passenger_gt, passenger_pred], "Acceleration_X")

    components = [
        ("Position_X", "Displacement (m)"),
        ("Velocity_X", "Velocity (m/s)"),
        ("Acceleration_X", "Acceleration (m/s²)"),
    ]

    # Row 0: Driver (red)
    for j, (comp, label) in enumerate(components):
        ax = axes[0, j]
        ax.plot(
            driver_gt["Time (s)"],
            driver_gt[comp],
            color="red",
            linewidth=2,
            label="Driver GT",
        )
        ax.plot(
            driver_pred["Time (s)"],
            driver_pred[comp],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Driver Pred",
        )
        if comp.startswith("Position"):
            ax.set_ylim(pos_lim_driver)
        elif comp.startswith("Velocity"):
            ax.set_ylim(vel_lim_driver)
        else:
            ax.set_ylim(acc_lim_driver)
        ax.set_title(label, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    # Row 1: Passenger (blue)
    for j, (comp, label) in enumerate(components):
        ax = axes[1, j]
        ax.plot(
            passenger_gt["Time (s)"],
            passenger_gt[comp],
            color="blue",
            linewidth=2,
            label="Passenger GT",
        )
        ax.plot(
            passenger_pred["Time (s)"],
            passenger_pred[comp],
            color="blue",
            linestyle="--",
            linewidth=2,
            label="Passenger Pred",
        )
        if comp.startswith("Position"):
            ax.set_ylim(pos_lim_pass)
        elif comp.startswith("Velocity"):
            ax.set_ylim(vel_lim_pass)
        else:
            ax.set_ylim(acc_lim_pass)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_plot, dpi=300)
    print(f"\n📈 Plot saved to: {output_plot}")
    plt.show()


def compute_probe_kinematics_single_sample():
    parser = argparse.ArgumentParser(
        description="Plot derived kinematics (Driver & Passenger | GT vs Pred) from averaged point positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help='Directory with predicted VTP files ("prediction" array).',
    )
    parser.add_argument(
        "--exact_dir",
        type=str,
        required=True,
        help='Directory with ground truth VTP files ("exact" array).',
    )
    parser.add_argument(
        "--driver_points",
        type=str,
        required=True,
        help='Driver point IDs/ranges (e.g., "70658-70659,70664,70676-70679").',
    )
    parser.add_argument(
        "--passenger_points",
        type=str,
        required=True,
        help="Passenger point IDs/ranges.",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0, help="Time step size Δt in seconds."
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="driver_passenger_gt_pred_kinematics.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--save_csv", action="store_true", help="Save the processed data to CSV files."
    )
    args = parser.parse_args()

    driver_ids = parse_point_set(args.driver_points)
    passenger_ids = parse_point_set(args.passenger_points)

    # Load series
    print("--- Loading Driver (Ground Truth) ---")
    driver_gt = load_averaged_series(
        args.exact_dir, driver_ids, args.dt, position_array="exact"
    )
    if driver_gt is None:
        return

    print("--- Loading Driver (Predicted) ---")
    driver_pred = load_averaged_series(
        args.pred_dir, driver_ids, args.dt, position_array="prediction"
    )
    if driver_pred is None:
        return

    print("--- Loading Passenger (Ground Truth) ---")
    passenger_gt = load_averaged_series(
        args.exact_dir, passenger_ids, args.dt, position_array="exact"
    )
    if passenger_gt is None:
        return

    print("--- Loading Passenger (Predicted) ---")
    passenger_pred = load_averaged_series(
        args.pred_dir, passenger_ids, args.dt, position_array="prediction"
    )
    if passenger_pred is None:
        return

    # Compute normalized squared error for driver
    common = driver_gt.columns.intersection(driver_pred.columns)
    cols = common[
        ~common.str.contains("Time", case=False)
    ]  # drop any col containing 'time'

    # normalized error: (gt - pred) / mean(|gt|)
    sqerr = (driver_gt[cols] - driver_pred[cols]).abs()
    denom = driver_gt[cols].abs().mean(axis=0).replace(0, np.nan)  # avoid div-by-zero
    norm_sqerr = sqerr.div(denom, axis=1)

    # rename: e.g., Position_X -> Position_error_X
    norm_sqerr.columns = [
        f"{a}_error_{b}" for a, b in (c.split("_", 1) for c in norm_sqerr.columns)
    ]

    # print mean for each normalized error column (driver)
    driver_norm_means = norm_sqerr.mean()
    print("\nDriver normalized error means:")
    print(
        f"  Position_error: {(driver_norm_means['Position_error_X'] + driver_norm_means['Position_error_Y'] + driver_norm_means['Position_error_Z']) / 3}"
    )
    print(
        f"  Velocity_error: {(driver_norm_means['Velocity_error_X'] + driver_norm_means['Velocity_error_Y'] + driver_norm_means['Velocity_error_Z']) / 3}"
    )
    print(
        f"  Acceleration_error: {(driver_norm_means['Acceleration_error_X'] + driver_norm_means['Acceleration_error_Y'] + driver_norm_means['Acceleration_error_Z']) / 3}"
    )

    # append to driver_gt (optional: fill NaNs if any denom was 0)
    driver_gt = pd.concat([driver_gt, norm_sqerr.fillna(0)], axis=1)

    # Compute normalized squared error for passenger
    common = passenger_gt.columns.intersection(passenger_pred.columns)
    cols = common[
        ~common.str.contains("Time", case=False)
    ]  # drop any col containing 'time'

    # normalized error: (gt - pred) / mean(|gt|)
    sqerr = (passenger_gt[cols] - passenger_pred[cols]).abs()
    denom = (
        passenger_gt[cols].abs().mean(axis=0).replace(0, np.nan)
    )  # avoid div-by-zero
    norm_sqerr = sqerr.div(denom, axis=1)

    # rename: e.g., Position_X -> Position_error_X
    norm_sqerr.columns = [
        f"{a}_error_{b}" for a, b in (c.split("_", 1) for c in norm_sqerr.columns)
    ]

    # print mean for each normalized error column (passenger)
    passenger_norm_means = norm_sqerr.mean()
    print("\nPassenger normalized error means:")
    print(
        f"  Position_error: {(passenger_norm_means['Position_error_X'] + passenger_norm_means['Position_error_Y'] + passenger_norm_means['Position_error_Z']) / 3}"
    )
    print(
        f"  Velocity_error: {(passenger_norm_means['Velocity_error_X'] + passenger_norm_means['Velocity_error_Y'] + passenger_norm_means['Velocity_error_Z']) / 3}"
    )
    print(
        f"  Acceleration_error: {(passenger_norm_means['Acceleration_error_X'] + passenger_norm_means['Acceleration_error_Y'] + passenger_norm_means['Acceleration_error_Z']) / 3}"
    )

    # append to driver_gt (optional: fill NaNs if any denom was 0)
    passenger_gt = pd.concat([passenger_gt, norm_sqerr.fillna(0)], axis=1)

    if args.save_csv:
        driver_gt.to_csv(
            "driver_ground_truth_kinematics.csv", index=False, float_format="%.6e"
        )
        driver_pred.to_csv(
            "driver_predicted_kinematics.csv", index=False, float_format="%.6e"
        )
        passenger_gt.to_csv(
            "passenger_ground_truth_kinematics.csv", index=False, float_format="%.6e"
        )
        passenger_pred.to_csv(
            "passenger_predicted_kinematics.csv", index=False, float_format="%.6e"
        )
        print("\n💾 Saved CSVs for driver/passenger (GT/Pred)")

    plot_kinematics(
        driver_gt, driver_pred, passenger_gt, passenger_pred, args.output_plot
    )


def compute_probe_kinematics_all_samples():
    parser = argparse.ArgumentParser(
        description="Compute MSE (position, velocity, acceleration) at probe locations over all samples and time; "
        "optionally plot GT vs Pred per sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predicted_parent",
        type=str,
        help="Parent directory whose subdirs contain predicted VTP files (one subdir per sample).",
        default=None,
    )
    parser.add_argument(
        "--exact_parent",
        type=str,
        help="Parent directory whose subdirs contain ground truth VTP files (same relative paths as predicted_parent).",
        default=None,
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help='Single-sample: directory with predicted VTP files ("prediction" array). Ignored if predicted_parent is set.',
    )
    parser.add_argument(
        "--exact_dir",
        type=str,
        default=None,
        help='Single-sample: directory with exact VTP files ("exact" array). Ignored if exact_parent is set.',
    )
    parser.add_argument(
        "--driver_points",
        type=str,
        required=True,
        help='Driver point IDs/ranges (e.g., "70658-70659,70664,70676-70679").',
    )
    parser.add_argument(
        "--passenger_points",
        type=str,
        required=True,
        help="Passenger point IDs/ranges.",
    )
    parser.add_argument(
        "--dt", type=float, default=5e-3, help="Time step size Δt in seconds."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory where all plots are stored (GT vs Pred per sample + mean ± 2*std summary).",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save probe MSE (position, velocity, acceleration) to CSV.",
    )
    args = parser.parse_args()

    driver_ids = parse_point_set(args.driver_points)
    passenger_ids = parse_point_set(args.passenger_points)

    os.makedirs(args.output_path, exist_ok=True)

    run_pairs: List[tuple] = []
    if args.predicted_parent and args.exact_parent:
        pred_runs = discover_runs(args.predicted_parent)
        exact_by_rel = {
            os.path.relpath(d, args.exact_parent): d
            for d in discover_runs(args.exact_parent)
        }
        for pred_run in pred_runs:
            rel = os.path.relpath(pred_run, args.predicted_parent)
            exact_run = exact_by_rel.get(rel)
            if exact_run is not None:
                run_pairs.append((pred_run, exact_run))
    elif args.pred_dir and args.exact_dir:
        if os.path.isdir(args.pred_dir) and os.path.isdir(args.exact_dir):
            run_pairs = [(args.pred_dir, args.exact_dir)]
    if not run_pairs:
        print(
            "❌ No sample directories found. Set --predicted_parent and --exact_parent, or --pred_dir and --exact_dir."
        )
        return

    print(
        f"Found {len(run_pairs)} sample(s). Computing squared errors and GT vs Pred plots..."
    )
    driver_sqerrs: List[pd.DataFrame] = []
    passenger_sqerrs: List[pd.DataFrame] = []
    predicted_parent = args.predicted_parent or ""

    for idx, (pred_dir, exact_dir) in enumerate(run_pairs):
        driver_gt = load_averaged_series(
            exact_dir, driver_ids, args.dt, position_array="exact"
        )
        driver_pred = load_averaged_series(
            pred_dir, driver_ids, args.dt, position_array="prediction"
        )
        passenger_gt = load_averaged_series(
            exact_dir, passenger_ids, args.dt, position_array="exact"
        )
        passenger_pred = load_averaged_series(
            pred_dir, passenger_ids, args.dt, position_array="prediction"
        )
        if driver_gt is None or driver_pred is None:
            continue
        if passenger_gt is None or passenger_pred is None:
            continue
        dr_sq = compute_squared_error_series(driver_gt, driver_pred)
        pass_sq = compute_squared_error_series(passenger_gt, passenger_pred)
        if dr_sq is not None:
            driver_sqerrs.append(dr_sq)
        if pass_sq is not None:
            passenger_sqerrs.append(pass_sq)

        # Unique name for this sample: relative path from parent, sanitized for filename
        if predicted_parent:
            sample_rel = os.path.relpath(pred_dir, predicted_parent)
        else:
            sample_rel = str(idx)
        sample_name = sample_rel.replace(os.sep, "_").replace(" ", "_")
        if not sample_name.strip():
            sample_name = f"sample_{idx}"
        sample_plot_path = os.path.join(
            args.output_path, f"kinematics_gt_pred_{sample_name}.png"
        )
        plot_kinematics(
            driver_gt, driver_pred, passenger_gt, passenger_pred, sample_plot_path
        )

    if not driver_sqerrs or not passenger_sqerrs:
        print(
            "❌ Could not compute squared errors for any sample (missing or misaligned data)."
        )
        return

    mse_driver = compute_mse_over_samples(driver_sqerrs)
    mse_passenger = compute_mse_over_samples(passenger_sqerrs)
    if mse_driver is None or mse_passenger is None:
        return

    print("\n--- MSE at probe locations (over all samples and all time) ---")
    print("Driver probe:")
    print(f"  Position MSE:      {mse_driver[0]:.6e}")
    print(f"  Velocity MSE:     {mse_driver[1]:.6e}")
    print(f"  Acceleration MSE: {mse_driver[2]:.6e}")
    print("Passenger probe:")
    print(f"  Position MSE:      {mse_passenger[0]:.6e}")
    print(f"  Velocity MSE:     {mse_passenger[1]:.6e}")
    print(f"  Acceleration MSE: {mse_passenger[2]:.6e}")

    if args.save_csv:
        mse_df = pd.DataFrame(
            [
                {
                    "probe": "driver",
                    "position_mse": mse_driver[0],
                    "velocity_mse": mse_driver[1],
                    "acceleration_mse": mse_driver[2],
                },
                {
                    "probe": "passenger",
                    "position_mse": mse_passenger[0],
                    "velocity_mse": mse_passenger[1],
                    "acceleration_mse": mse_passenger[2],
                },
            ]
        )
        mse_df.to_csv(
            os.path.join(args.output_path, "probe_mse.csv"),
            index=False,
            float_format="%.6e",
        )
        print("\n💾 Saved probe MSE to probe_mse.csv.")


if __name__ == "__main__":
    # compute_probe_kinematics_single_sample()
    compute_probe_kinematics_all_samples()
