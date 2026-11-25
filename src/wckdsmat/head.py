#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch compute head-centroid X/Y/Z signals from *_lp landmark columns.

For each input CSV, compute:

    head_centroid_lp_X = mean([LANDMARK_lp_X ...])
    head_centroid_lp_Y = mean([LANDMARK_lp_Y ...])
    head_centroid_lp_Z = mean([LANDMARK_lp_Z ...])

Default landmarks: LBH,LTH,RBH,RTH
Default time column kept: timestamp

Optional --also-velocity writes centroid velocity columns:
    head_centroid_lp_v_X, head_centroid_lp_v_Y, head_centroid_lp_v_Z, head_centroid_lp_v_3D

By default velocity is framewise diff, matching the earlier kinematic pipeline.
Use --time-derivatives for units/second velocity from timestamp differences.

python -m wckdsmat.head \
  --in-dir ./data/lp_hn/faceunits/051926 \
  --out-dir ./data/head_centroid/052026 \
  --pattern "*_lp_hn.csv" \
  --time-col timestamp \
  --landmarks LBH,LTH,RBH,RTH \
  --overwrite \
  --continue-on-error
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


AXES = ("X", "Y", "Z")


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_ci(df: pd.DataFrame, name: str) -> str | None:
    lower_to_original = {str(c).lower(): str(c) for c in df.columns}
    return lower_to_original.get(str(name).lower())


def resolve_keep_cols(df: pd.DataFrame, requested: list[str]) -> tuple[list[str], list[str]]:
    kept, missing = [], []
    for col in requested:
        found = find_ci(df, col)
        if found is None:
            missing.append(col)
        else:
            kept.append(found)
    return kept, missing


def auto_detect_lp_landmarks(df: pd.DataFrame, coord_type: str) -> list[str]:
    pattern = re.compile(
        rf"^(?P<stem>.+?)_{re.escape(coord_type)}_(?P<axis>[XYZ])$",
        flags=re.IGNORECASE,
    )
    found: dict[str, set[str]] = {}
    original: dict[str, str] = {}
    for col in df.columns:
        m = pattern.match(str(col))
        if not m:
            continue
        stem = m.group("stem")
        key = stem.lower()
        found.setdefault(key, set()).add(m.group("axis").upper())
        original.setdefault(key, stem)
    return sorted(original[k] for k, axes in found.items() if set(AXES).issubset(axes))


def resolve_lp_xyz(
    df: pd.DataFrame,
    landmarks: list[str],
    coord_type: str,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    resolved: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for lm in landmarks:
        axis_map: dict[str, str] = {}
        for axis in AXES:
            wanted = f"{lm}_{coord_type}_{axis}"
            found = find_ci(df, wanted)
            if found is None:
                missing.append(wanted)
            else:
                axis_map[axis] = found
        if all(a in axis_map for a in AXES):
            resolved[lm] = axis_map
    return resolved, missing


def compute_centroid(
    df: pd.DataFrame,
    resolved: dict[str, dict[str, str]],
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = pd.DataFrame(index=df.index)
    used_cols_by_axis: dict[str, list[str]] = {}
    for axis in AXES:
        cols = [axis_map[axis] for axis_map in resolved.values()]
        used_cols_by_axis[axis] = cols
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")
        out[f"{prefix}_{axis}"] = numeric.mean(axis=1, skipna=True)

    # This is centroid position magnitude, included only as a convenience.
    out[f"{prefix}_3D"] = np.sqrt(
        out[f"{prefix}_X"] ** 2 + out[f"{prefix}_Y"] ** 2 + out[f"{prefix}_Z"] ** 2
    )
    return out, used_cols_by_axis


def frame_velocity(position_df: pd.DataFrame, pos_prefix: str, vel_prefix: str) -> pd.DataFrame:
    vel = pd.DataFrame(index=position_df.index)
    for axis in AXES:
        col = f"{vel_prefix}_{axis}"
        vel[col] = position_df[f"{pos_prefix}_{axis}"].diff()
        if len(vel):
            vel.loc[vel.index[0], col] = 0.0
    vel[f"{vel_prefix}_3D"] = np.sqrt(
        vel[f"{vel_prefix}_X"] ** 2 + vel[f"{vel_prefix}_Y"] ** 2 + vel[f"{vel_prefix}_Z"] ** 2
    )
    return vel


def time_velocity(
    position_df: pd.DataFrame,
    time_series: pd.Series,
    pos_prefix: str,
    vel_prefix: str,
) -> pd.DataFrame:
    t = pd.to_numeric(time_series, errors="coerce").to_numpy(dtype=float)
    dt = np.diff(t, prepend=np.nan)
    dt[~np.isfinite(dt) | (dt <= 0)] = np.nan

    vel = pd.DataFrame(index=position_df.index)
    for axis in AXES:
        x = pd.to_numeric(position_df[f"{pos_prefix}_{axis}"], errors="coerce").to_numpy(dtype=float)
        dx = np.diff(x, prepend=np.nan)
        v = dx / dt
        if len(v):
            v[0] = 0.0
        vel[f"{vel_prefix}_{axis}"] = v

    vel[f"{vel_prefix}_3D"] = np.sqrt(
        vel[f"{vel_prefix}_X"] ** 2 + vel[f"{vel_prefix}_Y"] ** 2 + vel[f"{vel_prefix}_Z"] ** 2
    )
    return vel


def prepend(passthrough: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    if passthrough.empty:
        return signal_df.reset_index(drop=True)
    return pd.concat([passthrough.reset_index(drop=True), signal_df.reset_index(drop=True)], axis=1)


def process_one(
    csv_path: Path,
    out_dir: Path,
    landmarks: list[str],
    coord_type: str,
    keep_cols: list[str],
    centroid_name: str,
    min_landmarks: int,
    also_velocity: bool,
    time_derivatives: bool,
    overwrite: bool,
) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV has 0 rows.")

    actual_landmarks = auto_detect_lp_landmarks(df, coord_type) if not landmarks else landmarks
    resolved, missing_lm_cols = resolve_lp_xyz(df, actual_landmarks, coord_type)

    if len(resolved) < min_landmarks:
        raise ValueError(
            f"Found {len(resolved)} complete {coord_type} XYZ landmark triplets; "
            f"need at least {min_landmarks}. Resolved={sorted(resolved)}. "
            f"Missing examples={missing_lm_cols[:12]}"
        )

    kept_cols, missing_keep_cols = resolve_keep_cols(df, keep_cols)
    passthrough = df[kept_cols].copy() if kept_cols else pd.DataFrame(index=df.index)

    centroid_df, used_cols_by_axis = compute_centroid(df, resolved, centroid_name)

    position_dir = ensure_dir(out_dir / "centroid_position")
    meta_dir = ensure_dir(out_dir / "meta")

    stem = csv_path.stem
    pos_path = position_dir / f"{stem}_{centroid_name}_position.csv"
    if pos_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {pos_path}")
    prepend(passthrough, centroid_df).to_csv(pos_path, index=False)

    outputs = {"centroid_position": str(pos_path)}
    vel_path = ""
    velocity_units = None

    if also_velocity:
        velocity_dir = ensure_dir(out_dir / "centroid_velocity")
        vel_prefix = f"{centroid_name}_v"
        if time_derivatives:
            if not kept_cols:
                raise ValueError("--time-derivatives requires a valid --time-col/--keep-cols column.")
            vel_df = time_velocity(centroid_df, df[kept_cols[0]], centroid_name, vel_prefix)
            velocity_units = "input units/second"
        else:
            vel_df = frame_velocity(centroid_df, centroid_name, vel_prefix)
            velocity_units = "input units/frame"

        vel_path_obj = velocity_dir / f"{stem}_{centroid_name}_velocity.csv"
        if vel_path_obj.exists() and not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {vel_path_obj}")
        prepend(passthrough, vel_df).to_csv(vel_path_obj, index=False)
        vel_path = str(vel_path_obj)
        outputs["centroid_velocity"] = vel_path

    meta = {
        "input": str(csv_path),
        "n_rows": int(len(df)),
        "coordinate_type": coord_type,
        "requested_landmarks": actual_landmarks,
        "used_landmarks": sorted(resolved.keys()),
        "n_used_landmarks": int(len(resolved)),
        "min_landmarks": int(min_landmarks),
        "centroid_name": centroid_name,
        "centroid_columns": list(centroid_df.columns),
        "used_columns_by_axis": used_cols_by_axis,
        "requested_keep_columns": keep_cols,
        "kept_columns": kept_cols,
        "missing_keep_columns": missing_keep_cols,
        "missing_landmark_columns": missing_lm_cols,
        "also_velocity": bool(also_velocity),
        "time_derivatives": bool(time_derivatives),
        "velocity_units": velocity_units,
        "outputs": outputs,
    }
    meta_path = meta_dir / f"{stem}_{centroid_name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    outputs["meta"] = str(meta_path)

    return {
        "file": csv_path.name,
        "status": "ok",
        "n_rows": int(len(df)),
        "coordinate_type": coord_type,
        "n_used_landmarks": int(len(resolved)),
        "used_landmarks": ",".join(sorted(resolved.keys())),
        "kept_columns": ",".join(kept_cols),
        "missing_keep_columns": ",".join(missing_keep_cols),
        "position_output": str(pos_path),
        "velocity_output": vel_path,
        "meta_output": str(meta_path),
        "error": "",
    }


def find_files(in_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = in_dir.rglob(pattern) if recursive else in_dir.glob(pattern)
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() == ".csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch compute head centroid X/Y/Z from *_lp landmark columns.")
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pattern", default="*.csv")
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--landmarks", default="LBH,LTH,RBH,RTH", help="Comma list, or 'auto'. Default: LBH,LTH,RBH,RTH")
    p.add_argument("--coord-type", default="lp", help="Default: lp, for columns like LBH_lp_X")
    p.add_argument("--centroid-name", default="head_centroid_lp")
    p.add_argument("--min-landmarks", type=int, default=3)

    p.add_argument("--time-col", default="timestamp", help="Primary passthrough time column. Default: timestamp")
    p.add_argument("--keep-cols", default=None, help="Additional comma-separated passthrough columns.")

    p.add_argument("--also-velocity", action="store_true")
    p.add_argument("--time-derivatives", action="store_true", help="With --also-velocity, compute dX/dt instead of frame diff.")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    in_dir = Path(args.in_dir).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser())

    landmarks = [] if str(args.landmarks).strip().lower() == "auto" else parse_csv_list(args.landmarks)

    keep_cols = []
    if args.time_col:
        keep_cols.append(args.time_col)
    for col in parse_csv_list(args.keep_cols):
        if col.lower() not in {x.lower() for x in keep_cols}:
            keep_cols.append(col)

    files = find_files(in_dir, args.pattern, args.recursive)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {in_dir} matching {args.pattern!r}")

    print(f"Found {len(files)} file(s).")
    print(f"Input: {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Landmarks: {'auto' if not landmarks else ','.join(landmarks)}")
    print(f"Coordinate type: {args.coord_type}")

    rows: list[dict[str, object]] = []

    for csv_path in files:
        print(f"\nProcessing: {csv_path}")
        if args.dry_run:
            row = {
                "file": csv_path.name,
                "status": "dry_run",
                "n_rows": np.nan,
                "coordinate_type": args.coord_type,
                "n_used_landmarks": np.nan,
                "used_landmarks": "auto" if not landmarks else ",".join(landmarks),
                "kept_columns": ",".join(keep_cols),
                "missing_keep_columns": "",
                "position_output": str(out_dir / "centroid_position" / f"{csv_path.stem}_{args.centroid_name}_position.csv"),
                "velocity_output": str(out_dir / "centroid_velocity" / f"{csv_path.stem}_{args.centroid_name}_velocity.csv") if args.also_velocity else "",
                "meta_output": str(out_dir / "meta" / f"{csv_path.stem}_{args.centroid_name}_meta.json"),
                "error": "",
            }
            rows.append(row)
            print("  dry run only")
            continue

        try:
            row = process_one(
                csv_path=csv_path,
                out_dir=out_dir,
                landmarks=landmarks,
                coord_type=args.coord_type,
                keep_cols=keep_cols,
                centroid_name=args.centroid_name,
                min_landmarks=args.min_landmarks,
                also_velocity=args.also_velocity,
                time_derivatives=args.time_derivatives,
                overwrite=args.overwrite,
            )
            rows.append(row)
            print(f"  wrote: {row['position_output']}")
            if row["velocity_output"]:
                print(f"  wrote: {row['velocity_output']}")
        except Exception as exc:
            row = {
                "file": csv_path.name,
                "status": "error",
                "n_rows": np.nan,
                "coordinate_type": args.coord_type,
                "n_used_landmarks": np.nan,
                "used_landmarks": "",
                "kept_columns": "",
                "missing_keep_columns": "",
                "position_output": "",
                "velocity_output": "",
                "meta_output": "",
                "error": str(exc),
            }
            rows.append(row)
            print(f"  ERROR: {exc}")
            if not args.continue_on_error:
                break

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "batch_head_centroid_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nDone. ok={(summary['status'] == 'ok').sum()}, dry_run={(summary['status'] == 'dry_run').sum()}, errors={(summary['status'] == 'error').sum()}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
