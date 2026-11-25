#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch GFM kinematic time-series creation, grouped by kinematic type, with 3D Euclidean magnitude.

Input:
    A folder of calibrated GFM / faceunits CSV files.

Output layout:
    out_dir/
      position/
      displacement/
      velocity/
      acceleration/
      cumulative/
      meta/
      batch_kinematic_summary.csv

Each kinematic CSV preserves requested passthrough columns such as timestamp/time_s
and contains:
    - all original XYZ kinematic channels
    - optional Euclidean 3D magnitude per landmark stem:
        STEM_3D = sqrt(STEM_X^2 + STEM_Y^2 + STEM_Z^2)

Derivative modes:
    frame, default:
        velocity = position.diff()
        acceleration = velocity.diff()
        units are input_units/frame and input_units/frame^2

    time:
        velocity = d(position)/dt
        acceleration = d(velocity)/dt
        units are input_units/second and input_units/second^2

The default frame mode matches the earlier validation_pipeline.py logic.

# 1. DRY RUN — confirm it sees all 75 files and resolves time_s
python -m wckdsmat.kinematics_grouped \
  --in-dir  ./data/lp_hn/faceunits/051926 \
  --out-dir ./data/articulator_coordination/kinematic_data \
  --pattern "*_lp_hn.csv" \
  --keep-cols time_s,timestamp,frame \
  --time-col time_s \
  --continue-on-error \
  --dry-run

  # 2. REAL RUN — same args, minus --dry-run
python -m wckdsmat.kinematics_grouped \
  --in-dir  ./data/lp_hn/faceunits/051926 \
  --out-dir ./data/articulator_coordination/kinematic_data \
  --pattern "*_lp_hn.csv" \
  --keep-cols time_s,timestamp,frame \
  --time-col time_s \
  --continue-on-error

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


XYZ_RE = re.compile(r"^(?P<stem>.+)_(?P<axis>[XYZxyz])$")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def resolve_keep_columns(df: pd.DataFrame, requested: Iterable[str]) -> tuple[list[str], list[str]]:
    """
    Case-insensitive passthrough column resolution.

    Returns:
        kept original column names, missing requested names.
    """
    lower_to_original = {str(c).lower(): str(c) for c in df.columns}
    kept: list[str] = []
    missing: list[str] = []

    for name in requested:
        original = lower_to_original.get(str(name).lower())
        if original is None:
            missing.append(str(name))
        elif original not in kept:
            kept.append(original)

    return kept, missing


def find_xyz_columns(df: pd.DataFrame) -> list[str]:
    """Return columns ending in _X/_Y/_Z, preserving dataframe order."""
    return [str(c) for c in df.columns if XYZ_RE.match(str(c))]


def stem_axis_map(columns: Iterable[str]) -> dict[str, dict[str, str]]:
    """
    Build mapping:
        stem -> {"X": col_x, "Y": col_y, "Z": col_z}
    """
    out: dict[str, dict[str, str]] = {}
    for col in columns:
        m = XYZ_RE.match(str(col))
        if not m:
            continue
        stem = m.group("stem")
        axis = m.group("axis").upper()
        out.setdefault(stem, {})[axis] = str(col)
    return out


def complete_xyz_stems(columns: Iterable[str]) -> list[str]:
    m = stem_axis_map(columns)
    return sorted(stem for stem, axes in m.items() if all(a in axes for a in ("X", "Y", "Z")))


def displacement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input CSV has 0 rows, so displacement/velocity cannot be computed.")
    return df - df.iloc[0]


def velocity_frame(df: pd.DataFrame) -> pd.DataFrame:
    v = df.diff()
    if len(v):
        v.iloc[0] = 0
    return v


def acceleration_frame(v: pd.DataFrame) -> pd.DataFrame:
    a = v.diff()
    if len(a):
        a.iloc[0] = 0
    return a


def cumulative(df: pd.DataFrame) -> pd.DataFrame:
    d = df.diff()
    if len(d):
        d.iloc[0] = 0
    return d.abs().cumsum()


def numeric_time_seconds(df: pd.DataFrame, time_col: str) -> np.ndarray:
    if time_col not in df.columns:
        raise ValueError(f"--time-col {time_col!r} was requested but is not present in this file.")
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(t).sum() < 2:
        raise ValueError(f"Time column {time_col!r} has fewer than 2 finite values.")
    return t


def velocity_time(df: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """
    Timestamp-normalized derivative dX/dt.
    Handles nonuniform dt by dividing each frame difference by local dt.
    """
    dt = np.diff(t)
    out = df.diff()

    # Avoid divide-by-zero/negative/invalid intervals.
    valid_dt = np.isfinite(dt) & (dt > 0)
    dt_safe = np.where(valid_dt, dt, np.nan)

    if len(out):
        out.iloc[0] = 0
    if len(out) > 1:
        out.iloc[1:] = out.iloc[1:].to_numpy(dtype=float) / dt_safe[:, None]

    # If bad timestamps created NaNs, keep them as NaN rather than silently inventing derivatives.
    return out


def acceleration_time(v: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    dt = np.diff(t)
    out = v.diff()

    valid_dt = np.isfinite(dt) & (dt > 0)
    dt_safe = np.where(valid_dt, dt, np.nan)

    if len(out):
        out.iloc[0] = 0
    if len(out) > 1:
        out.iloc[1:] = out.iloc[1:].to_numpy(dtype=float) / dt_safe[:, None]

    return out


def add_euclidean_3d_magnitudes(df: pd.DataFrame, xyz_source_columns: Iterable[str]) -> pd.DataFrame:
    """
    Add one 3D Euclidean magnitude column per complete XYZ landmark stem.

    Example:
        LBH_hn_X, LBH_hn_Y, LBH_hn_Z -> LBH_hn_3D

    For displacement/velocity/acceleration:
        This is the vector magnitude of that kinematic quantity.

    For cumulative:
        Because cumulative is coordinate-wise absolute travel, STEM_3D is the Euclidean
        magnitude of the cumulative coordinate-wise travel vector.
    """
    out = df.copy()
    mapping = stem_axis_map(xyz_source_columns)

    insertions: dict[str, np.ndarray] = {}
    for stem in sorted(mapping):
        axes = mapping[stem]
        if not all(a in axes for a in ("X", "Y", "Z")):
            continue

        xcol, ycol, zcol = axes["X"], axes["Y"], axes["Z"]
        if not all(c in out.columns for c in (xcol, ycol, zcol)):
            continue

        x = pd.to_numeric(out[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(out[ycol], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(out[zcol], errors="coerce").to_numpy(dtype=float)
        insertions[f"{stem}_3D"] = np.sqrt(x * x + y * y + z * z)

    for col, values in insertions.items():
        out[col] = values

    return out


def with_passthrough(keep_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepend passthrough columns to a kinematic dataframe.
    """
    if keep_df.empty or keep_df.shape[1] == 0:
        return data_df
    return pd.concat([keep_df.reset_index(drop=True), data_df.reset_index(drop=True)], axis=1)


def output_paths(out_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "position": out_dir / "position" / f"{stem}_gfm_position.csv",
        "displacement": out_dir / "displacement" / f"{stem}_gfm_displacement.csv",
        "velocity": out_dir / "velocity" / f"{stem}_gfm_velocity.csv",
        "acceleration": out_dir / "acceleration" / f"{stem}_gfm_acceleration.csv",
        "cumulative": out_dir / "cumulative" / f"{stem}_gfm_cumulative.csv",
        "meta": out_dir / "meta" / f"{stem}_kinematic_meta.json",
    }


def process_one_file(
    in_path: Path,
    out_dir: Path,
    keep_cols_requested: list[str],
    derivative_mode: str,
    time_col: str | None,
    add_3d: bool,
    overwrite: bool,
) -> dict[str, object]:
    df = pd.read_csv(in_path)

    if len(df) == 0:
        raise ValueError("Input CSV has 0 rows, so kinematic time-series cannot be computed.")

    xyz_cols = find_xyz_columns(df)
    if not xyz_cols:
        raise ValueError("No XYZ columns found. Expected columns ending in _X, _Y, or _Z.")

    xyz_df = df[xyz_cols].apply(pd.to_numeric, errors="coerce")

    kept_cols, missing_cols = resolve_keep_columns(df, keep_cols_requested)
    keep_df = df[kept_cols].copy() if kept_cols else pd.DataFrame(index=df.index)

    pos = xyz_df.copy()
    disp = displacement(pos)

    if derivative_mode == "frame":
        vel = velocity_frame(pos)
        acc = acceleration_frame(vel)
        used_time_col = None
    elif derivative_mode == "time":
        if not time_col:
            raise ValueError("--time-derivatives requires --time-col.")
        t = numeric_time_seconds(df, time_col)
        vel = velocity_time(pos, t)
        acc = acceleration_time(vel, t)
        used_time_col = time_col
    else:
        raise ValueError(f"Unknown derivative mode: {derivative_mode}")

    cum = cumulative(pos)

    products = {
        "position": pos,
        "displacement": disp,
        "velocity": vel,
        "acceleration": acc,
        "cumulative": cum,
    }

    euclidean_3d_columns: dict[str, list[str]] = {}
    if add_3d:
        products = {
            name: add_euclidean_3d_magnitudes(product, xyz_cols)
            for name, product in products.items()
        }
        for name, product in products.items():
            euclidean_3d_columns[name] = [c for c in product.columns if str(c).endswith("_3D")]
    else:
        euclidean_3d_columns = {name: [] for name in products}

    products = {
        name: with_passthrough(keep_df, product)
        for name, product in products.items()
    }

    paths = output_paths(out_dir, in_path.stem)

    for subdir in ("position", "displacement", "velocity", "acceleration", "cumulative", "meta"):
        ensure_dir(out_dir / subdir)

    for name in ("position", "displacement", "velocity", "acceleration", "cumulative"):
        path = paths[name]
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to replace existing outputs.")
        products[name].to_csv(path, index=False)

    complete_stems = complete_xyz_stems(xyz_cols)

    units = {
        "position": "input units, e.g. faceunits",
        "displacement": "input units from first frame",
        "velocity": "input units/frame" if derivative_mode == "frame" else "input units/second",
        "acceleration": "input units/frame^2" if derivative_mode == "frame" else "input units/second^2",
        "cumulative": "cumulative absolute coordinate-wise travel in input units",
        "euclidean_3D": "sqrt(X^2 + Y^2 + Z^2) in the corresponding kinematic units",
    }

    meta = {
        "input": str(in_path),
        "n_frames": int(len(df)),
        "n_xyz_columns": int(len(xyz_cols)),
        "n_landmark_stems": int(len(complete_stems)),
        "xyz_columns": xyz_cols,
        "complete_xyz_stems": complete_stems,
        "add_euclidean_3d": bool(add_3d),
        "euclidean_3d_columns": euclidean_3d_columns,
        "requested_keep_columns": keep_cols_requested,
        "kept_columns": kept_cols,
        "missing_keep_columns": missing_cols,
        "outputs": {k: str(v) for k, v in paths.items()},
        "output_layout": "grouped_by_kinematic_type",
        "derivative_mode": derivative_mode,
        "time_col": used_time_col,
        "units": units,
        "output_columns_begin_with": {
            name: list(products[name].columns[:12])
            for name in products
        },
    }

    meta_path = paths["meta"]
    if meta_path.exists() and not overwrite:
        raise FileExistsError(f"{meta_path} exists. Use --overwrite to replace existing outputs.")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "input": str(in_path),
        "status": "ok",
        "n_frames": len(df),
        "n_xyz_columns": len(xyz_cols),
        "n_landmark_stems": len(complete_stems),
        "n_3d_columns_per_output": len(euclidean_3d_columns.get("position", [])),
        "kept_columns": ",".join(kept_cols),
        "missing_keep_columns": ",".join(missing_cols),
        "derivative_mode": derivative_mode,
        "time_col": used_time_col,
        **{f"output_{k}": str(v) for k, v in paths.items()},
    }


def find_files(in_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    globber = in_dir.rglob if recursive else in_dir.glob
    return sorted(p for p in globber(pattern) if p.is_file() and p.suffix.lower() == ".csv")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch-create grouped GFM kinematic time-series with optional 3D Euclidean magnitudes."
    )
    p.add_argument("--in-dir", required=True, help="Folder containing input calibrated GFM/faceunits CSV files.")
    p.add_argument("--out-dir", required=True, help="Folder where grouped kinematic outputs will be written.")
    p.add_argument("--pattern", default="*.csv", help='Input glob pattern, default "*.csv".')
    p.add_argument("--recursive", action="store_true", help="Search input directory recursively.")

    p.add_argument(
        "--keep-cols",
        default="timestamp",
        help=(
            "Comma-separated passthrough columns to prepend to every output. "
            "Default: timestamp. Example: timestamp,time_s,frame"
        ),
    )
    p.add_argument(
        "--time-col",
        default="timestamp",
        help="Timestamp column used only with --time-derivatives. Default: timestamp.",
    )
    p.add_argument(
        "--time-derivatives",
        action="store_true",
        help="Compute velocity/acceleration using dt from --time-col instead of frame differences.",
    )

    p.add_argument(
        "--no-3d-magnitude",
        action="store_true",
        help="Disable Euclidean 3D magnitude columns. By default, STEM_3D columns are added.",
    )

    p.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    p.add_argument("--dry-run", action="store_true", help="Print files that would be processed without writing outputs.")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing after a file-level error. Errors are written to the batch summary.",
    )

    args = p.parse_args()

    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    keep_cols = parse_csv_list(args.keep_cols)
    derivative_mode = "time" if args.time_derivatives else "frame"
    add_3d = not args.no_3d_magnitude

    files = find_files(in_dir, args.pattern, args.recursive)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {in_dir} matching {args.pattern!r}.")

    print(f"Found {len(files)} file(s).")
    print(f"Output layout: grouped by kinematic type under {out_dir}")
    print(f"Passthrough columns: {keep_cols if keep_cols else 'none'}")
    print(f"Euclidean 3D magnitude columns: {'on' if add_3d else 'off'}")

    if args.dry_run:
        for path in files:
            print(f"DRY RUN: {path}")
        return

    ensure_dir(out_dir)

    rows: list[dict[str, object]] = []
    ok = 0
    errors = 0

    for path in files:
        print(f"\nProcessing: {path}")
        try:
            row = process_one_file(
                in_path=path,
                out_dir=out_dir,
                keep_cols_requested=keep_cols,
                derivative_mode=derivative_mode,
                time_col=args.time_col,
                add_3d=add_3d,
                overwrite=args.overwrite,
            )
            rows.append(row)
            ok += 1
            print("  wrote grouped outputs")
            if row.get("n_3d_columns_per_output", 0):
                print(f"  added {row['n_3d_columns_per_output']} Euclidean 3D channel(s) per output")
        except Exception as exc:
            errors += 1
            print(f"  ERROR: {exc}")
            rows.append({
                "input": str(path),
                "status": "error",
                "error": str(exc),
            })
            if not args.continue_on_error:
                break

    summary_path = out_dir / "batch_kinematic_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(f"\nDone. ok={ok}, errors={errors}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
