#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 python WCKDSMAT/batch_gfm_imll_timeseries_case_insensitive.py \
  --input ./data/lp_hn_om \
  --out ./data/lp_hn_om/speech \
  --pattern "*.csv" \
  --jobs 3

batch_gfm_imll_timeseries.py

Batch extraction of speech/lower-lip time series from GFM calibrated +
head-normalized CSV files.

This script is adapted from the working Stage-B validation/Bandini logic:
  - validation_pipeline.py: xyz-root detection, position/displacement/velocity
    time-series construction, safe_derivative()
  - bandini_full_pipeline.py: derived articulator imll = mll - jaw_mid,
    where jaw_mid = 0.5 * (lj + rj)

Expected input
--------------
One or more CSV files containing:
  - time_s column, or a frame column that can be converted to time_s
  - xyz triplets for anatomical stems, e.g.:
        mll_x, mll_y, mll_z
        lj_x,  lj_y,  lj_z
        rj_x,  rj_y,  rj_z

The input should already be calibrated and head-normalized. This script does
not perform spatial alignment or head normalization.

Main outputs per input file
---------------------------
<out>/<relative_input_stem>/time_series/
  gfm_position.csv
  gfm_displacement.csv
  gfm_velocity.csv
  imll_position.csv
  imll_displacement.csv
  imll_velocity.csv
  speech_landmarks_long.csv

Definitions
-----------
position:
  The input calibrated/head-normalized coordinates.

displacement:
  Position zeroed to the first valid frame, per landmark/stem.

velocity:
  Derivative of position with respect to time_s.

jaw_mid:
  0.5 * (lj + rj)

imll:
  mll - jaw_mid

Because inputs are assumed to be head-normalized, imll represents middle lower
lip motion isolated relative to the jaw midpoint in the head-normalized frame.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_ROOTS = {"COM", "com", "jaw_mid", "imll"}
DEFAULT_REQUIRED_IMLL_STEMS = ("mll", "lj", "rj")


def normalize_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase so landmark/time/frame lookup is case-insensitive.

    This makes variants like MLL_X, mll_X, Mll_x, Time_S, and FRAME all resolve
    to mll_x, time_s, and frame. Raises if two columns collapse to the same
    lowercase name, because that would make lookup ambiguous.
    """
    lower_cols = [str(c).strip().lower() for c in df.columns]
    duplicates = sorted({c for c in lower_cols if lower_cols.count(c) > 1})
    if duplicates:
        raise ValueError(
            "Case-insensitive column normalization produced duplicate columns: "
            + ", ".join(duplicates)
        )
    out = df.copy()
    out.columns = lower_cols
    return out


# -----------------------------------------------------------------------------
# Core utilities adapted from validation_pipeline.py
# -----------------------------------------------------------------------------
def safe_derivative(arr: np.ndarray, time_s: Sequence[float], order: int = 2) -> np.ndarray:
    """
    Compute derivative over a possibly irregular time axis.

    Duplicate timestamps are removed for the gradient calculation, then the
    derivative is interpolated back onto the original time vector.
    """
    t = np.asarray(time_s, dtype=float)
    arr = np.asarray(arr, dtype=float)

    if t.ndim != 1:
        raise ValueError("time_s must be 1D.")
    if len(t) != len(arr):
        raise ValueError(f"time_s length ({len(t)}) does not match array length ({len(arr)}).")

    t_unique, idx = np.unique(t, return_index=True)
    arr_unique = arr[idx]

    if len(t_unique) < 2:
        raise ValueError("Need at least two unique timestamps to compute velocity.")

    if len(t_unique) < len(t):
        print(f"[warn] safe_derivative: removed {len(t) - len(t_unique)} duplicated timestamps")

    if not np.all(np.diff(t_unique) > 0):
        raise ValueError("time_s is not strictly increasing after duplicate removal.")

    edge_order = min(order, len(t_unique) - 1)
    deriv_clean = np.gradient(arr_unique, t_unique, axis=0, edge_order=edge_order)

    deriv_interp = np.zeros_like(arr, dtype=float)
    if arr.ndim == 1:
        deriv_interp = np.interp(t, t_unique, deriv_clean, left=np.nan, right=np.nan)
    else:
        for d in range(arr.shape[1]):
            deriv_interp[:, d] = np.interp(
                t, t_unique, deriv_clean[:, d], left=np.nan, right=np.nan
            )
    return deriv_interp


def detect_xyz_roots(cols: Iterable[str], exclude_roots: Optional[Iterable[str]] = None) -> list[str]:
    """Return stems that have complete _x/_y/_z triplets."""
    exclude = {str(x).strip().lower() for x in (exclude_roots or [])}
    colset = {str(c).strip().lower() for c in cols}
    roots = set()

    for c in colset:
        if not c.endswith("_x"):
            continue
        root = c[:-2]
        if root in exclude:
            continue
        if f"{root}_y" in colset and f"{root}_z" in colset:
            roots.add(root)

    return sorted(roots)


def xyz_block(df: pd.DataFrame, stem: str) -> np.ndarray:
    """Return T x 3 xyz block for one stem."""
    return np.vstack([
        df[f"{stem}_x"].to_numpy(float),
        df[f"{stem}_y"].to_numpy(float),
        df[f"{stem}_z"].to_numpy(float),
    ]).T


# -----------------------------------------------------------------------------
# Derived articulator logic adapted from bandini_full_pipeline.py
# -----------------------------------------------------------------------------
def add_imll_columns(
    df: pd.DataFrame,
    mll_stem: str = "mll",
    lj_stem: str = "lj",
    rj_stem: str = "rj",
    out_stem: str = "imll",
    strict: bool = True,
) -> pd.DataFrame:
    """
    Add jaw_mid_x/y/z and imll_x/y/z.

    jaw_mid = 0.5 * (lj + rj)
    imll = mll - jaw_mid
    """
    needed = [f"{s}_{ax}" for s in (mll_stem, lj_stem, rj_stem) for ax in ("x", "y", "z")]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        msg = (
            "Cannot compute imll. Missing required columns: "
            + ", ".join(missing)
        )
        if strict:
            raise ValueError(msg)
        print(f"[warn] {msg}")
        return df

    out = df.copy()
    for ax in ("x", "y", "z"):
        out[f"jaw_mid_{ax}"] = 0.5 * (out[f"{lj_stem}_{ax}"] + out[f"{rj_stem}_{ax}"])
        out[f"{out_stem}_{ax}"] = out[f"{mll_stem}_{ax}"] - out[f"jaw_mid_{ax}"]

    # Add 3D magnitude for imll and jaw_mid when the table represents a time series.
    # For imll, the 3D value is magnitude of the vector imll_x/y/z.
    # For jaw_mid, the 3D value is the excursion magnitude of jaw_mid from first frame.
    imll_xyz = xyz_block(out, out_stem)
    out[f"{out_stem}_3d"] = np.linalg.norm(imll_xyz, axis=1)

    jaw_mid_xyz = xyz_block(out, "jaw_mid")
    out["jaw_mid_3d"] = np.linalg.norm(jaw_mid_xyz - jaw_mid_xyz[:1, :], axis=1)

    return out


def subset_columns(df: pd.DataFrame, stems: Sequence[str], include_3d: bool = True) -> pd.DataFrame:
    """Return time_s plus xyz(/3d) columns for selected stems that exist."""
    cols = ["time_s"] if "time_s" in df.columns else []
    for stem in stems:
        for ax in ("x", "y", "z"):
            c = f"{stem}_{ax}"
            if c in df.columns:
                cols.append(c)
        c3 = f"{stem}_3d"
        if include_3d and c3 in df.columns:
            cols.append(c3)
    return df.loc[:, cols].copy()


# -----------------------------------------------------------------------------
# Time axis and file discovery
# -----------------------------------------------------------------------------
def ensure_time_s(df: pd.DataFrame, fs: float, frame_col: Optional[str] = None) -> pd.DataFrame:
    """Ensure a time_s column exists."""
    out = df.copy()

    if "time_s" in out.columns:
        out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
        return out

    candidate_frame_cols = []
    if frame_col:
        candidate_frame_cols.append(str(frame_col).strip().lower())
    candidate_frame_cols.extend(["frame", "frame_idx", "frame_index"])

    for c in candidate_frame_cols:
        if c in out.columns:
            frame = pd.to_numeric(out[c], errors="coerce").to_numpy(float)
            # Preserve actual frame numbering if available, but start time at zero.
            frame0 = np.nanmin(frame)
            out.insert(0, "time_s", (frame - frame0) / float(fs))
            return out

    out.insert(0, "time_s", np.arange(len(out), dtype=float) / float(fs))
    return out


def discover_inputs(input_root: Path, pattern: str, recursive: bool) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    globber = input_root.rglob if recursive else input_root.glob
    files = sorted(p for p in globber(pattern) if p.is_file())
    return files


def output_dir_for_file(infile: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        rel_stem = infile.stem
    else:
        try:
            rel = infile.relative_to(input_root)
        except ValueError:
            rel = Path(infile.name)
        rel_stem = rel.with_suffix("")
    return output_root / rel_stem / "time_series"


# -----------------------------------------------------------------------------
# Main per-file processing
# -----------------------------------------------------------------------------
def build_timeseries(
    df: pd.DataFrame,
    roots: Sequence[str],
    time_s: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build position, displacement, and velocity tables for selected roots."""
    if not roots:
        raise ValueError("No xyz landmark roots selected.")

    pos = np.stack([
        np.stack([
            df[f"{r}_x"].to_numpy(float),
            df[f"{r}_y"].to_numpy(float),
            df[f"{r}_z"].to_numpy(float),
        ], axis=1)
        for r in roots
    ], axis=1)  # T x N x 3

    exc = pos - pos[:1, :, :]
    mag_exc = np.linalg.norm(exc, axis=2)

    vel = safe_derivative(pos.reshape(len(time_s), -1), time_s)
    vel = vel.reshape(pos.shape)
    vel_speed = np.linalg.norm(vel, axis=2)

    pos_df = pd.DataFrame({"time_s": time_s})
    disp_df = pd.DataFrame({"time_s": time_s})
    vel_df = pd.DataFrame({"time_s": time_s})

    for i, r in enumerate(roots):
        pos_df[f"{r}_x"] = pos[:, i, 0]
        pos_df[f"{r}_y"] = pos[:, i, 1]
        pos_df[f"{r}_z"] = pos[:, i, 2]
        pos_df[f"{r}_3d"] = mag_exc[:, i]

        disp_df[f"{r}_x"] = exc[:, i, 0]
        disp_df[f"{r}_y"] = exc[:, i, 1]
        disp_df[f"{r}_z"] = exc[:, i, 2]
        disp_df[f"{r}_3d"] = mag_exc[:, i]

        vel_df[f"{r}_x"] = vel[:, i, 0]
        vel_df[f"{r}_y"] = vel[:, i, 1]
        vel_df[f"{r}_z"] = vel[:, i, 2]
        vel_df[f"{r}_3d"] = vel_speed[:, i]

    return pos_df, disp_df, vel_df


def to_long(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Convert wide time-series table to long format."""
    value_cols = [c for c in df.columns if c != "time_s"]
    rows = []
    pat = re.compile(r"(.+)_([xyz]|3d)$")

    for c in value_cols:
        m = pat.match(c)
        if not m:
            continue
        stem, axis = m.group(1), m.group(2)
        tmp = pd.DataFrame({
            "time_s": df["time_s"].to_numpy(),
            "signal": signal,
            "stem": stem,
            "axis": axis,
            "value": df[c].to_numpy(),
        })
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["time_s", "signal", "stem", "axis", "value"])
    return pd.concat(rows, ignore_index=True)


def process_one(
    infile: Path,
    input_root: Path,
    output_root: Path,
    pattern: str,
    fs: float,
    frame_col: Optional[str],
    include_roots: Optional[set[str]],
    exclude_roots: set[str],
    mll_stem: str,
    lj_stem: str,
    rj_stem: str,
    strict_imll: bool,
    overwrite: bool,
) -> Path:
    outdir = output_dir_for_file(infile, input_root, output_root)
    done_marker = outdir / ".done"
    if done_marker.exists() and not overwrite:
        print(f"[skip] {infile} -> already complete")
        return outdir

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    df = normalize_columns_case_insensitive(df)

    mll_stem = str(mll_stem).strip().lower()
    lj_stem = str(lj_stem).strip().lower()
    rj_stem = str(rj_stem).strip().lower()

    df = ensure_time_s(df, fs=fs, frame_col=frame_col)
    time_s = df["time_s"].to_numpy(float)

    roots = detect_xyz_roots(df.columns, exclude_roots=exclude_roots)
    if include_roots:
        roots = [r for r in roots if r in include_roots]

    # Ensure required imll stems are present even if the user filters roots.
    for required in (mll_stem, lj_stem, rj_stem):
        if required not in roots and all(f"{required}_{ax}" in df.columns for ax in ("x", "y", "z")):
            roots.append(required)
    roots = sorted(set(roots))

    missing_imll = [
        f"{stem}_{ax}"
        for stem in (mll_stem, lj_stem, rj_stem)
        for ax in ("x", "y", "z")
        if f"{stem}_{ax}" not in df.columns
    ]
    if missing_imll and strict_imll:
        raise ValueError(f"{infile}: missing required imll source columns: {missing_imll}")

    pos_df, disp_df, vel_df = build_timeseries(df, roots=roots, time_s=time_s)

    # Add jaw_mid and imll consistently to all three signal types.
    pos_df = add_imll_columns(pos_df, mll_stem, lj_stem, rj_stem, strict=strict_imll)
    disp_df = add_imll_columns(disp_df, mll_stem, lj_stem, rj_stem, strict=strict_imll)
    vel_df = add_imll_columns(vel_df, mll_stem, lj_stem, rj_stem, strict=strict_imll)

    # Full GFM-wide outputs.
    pos_df.to_csv(outdir / "gfm_position.csv", index=False)
    disp_df.to_csv(outdir / "gfm_displacement.csv", index=False)
    vel_df.to_csv(outdir / "gfm_velocity.csv", index=False)

    # Focused derived-articulator outputs.
    derived_stems = ["jaw_mid", "imll"]
    subset_columns(pos_df, derived_stems).to_csv(outdir / "imll_position.csv", index=False)
    subset_columns(disp_df, derived_stems).to_csv(outdir / "imll_displacement.csv", index=False)
    subset_columns(vel_df, derived_stems).to_csv(outdir / "imll_velocity.csv", index=False)

    # Long format for easier plotting/modeling.
    long_df = pd.concat([
        to_long(pos_df, "pos"),
        to_long(disp_df, "disp"),
        to_long(vel_df, "vel"),
    ], ignore_index=True)
    long_df.insert(0, "source_file", str(infile))
    long_df.to_csv(outdir / "speech_landmarks_long.csv", index=False)

    done_marker.write_text("ok\n", encoding="utf-8")
    print(f"[ok] {infile} -> {outdir}")
    return outdir


def parse_root_list(value: Optional[str]) -> Optional[set[str]]:
    if value is None or not value.strip():
        return None
    return {x.strip().lower() for x in value.split(",") if x.strip()}


def run_many(files: Sequence[Path], jobs: int, fn):
    if jobs <= 1 or len(files) <= 1:
        for f in files:
            fn(f)
        return

    errors = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(fn, f): f for f in files}
        for fut in as_completed(futures):
            try:
                fut.result()
            except BaseException as e:
                errors.append((futures[fut], e))
                print(f"[err] {futures[fut]}: {e}")

    if errors:
        first_file, first_err = errors[0]
        raise RuntimeError(f"Failed on {first_file}: {first_err}") from first_err


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Batch-generate GFM speech/lower-lip time series from calibrated head-normalized CSV files."
    )
    ap.add_argument("--input", required=True, help="Input CSV file or root directory.")
    ap.add_argument("--out", required=True, help="Output root directory.")
    ap.add_argument("--pattern", default="*.csv", help="Filename glob when --input is a directory. Default: *.csv")
    ap.add_argument("--no-recursive", action="store_true", help="Do not recursively search input directories.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers. Default: 1")
    ap.add_argument("--fs", type=float, default=30.0, help="Sampling rate used if time_s is absent. Default: 30")
    ap.add_argument("--frame-col", default=None, help="Optional frame column to convert to time_s if time_s is absent.")
    ap.add_argument(
        "--include-roots",
        default=None,
        help="Optional comma-separated landmark stems to keep, e.g. mll,lj,rj,mul,mur. Required imll stems are always retained if present.",
    )
    ap.add_argument(
        "--exclude-roots",
        default=",".join(sorted(DEFAULT_EXCLUDE_ROOTS)),
        help="Comma-separated stems to exclude from generic root detection. Default excludes COM, com, jaw_mid, imll.",
    )
    ap.add_argument("--mll-stem", default="mll", help="Middle lower lip stem. Default: mll")
    ap.add_argument("--lj-stem", default="lj", help="Left jaw stem. Default: lj")
    ap.add_argument("--rj-stem", default="rj", help="Right jaw stem. Default: rj")
    ap.add_argument(
        "--allow-missing-imll",
        action="store_true",
        help="Do not fail if mll/lj/rj columns are missing; skip imll columns instead.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs even if a .done marker exists.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.out).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    include_roots = parse_root_list(args.include_roots)
    exclude_roots = parse_root_list(args.exclude_roots) or set()

    files = discover_inputs(input_root, pattern=args.pattern, recursive=not args.no_recursive)
    if not files:
        raise FileNotFoundError(f"No input files found under {input_root} with pattern {args.pattern}")

    print(f"[info] input    = {input_root}")
    print(f"[info] output   = {output_root}")
    print(f"[info] files    = {len(files)}")
    print(f"[info] jobs     = {max(1, args.jobs)}")
    print(f"[info] fs       = {args.fs}")
    print(f"[info] imll     = {str(args.mll_stem).lower()} - 0.5*({str(args.lj_stem).lower()}+{str(args.rj_stem).lower()})")

    def fn(f: Path):
        return process_one(
            infile=f,
            input_root=input_root,
            output_root=output_root,
            pattern=args.pattern,
            fs=args.fs,
            frame_col=args.frame_col,
            include_roots=include_roots,
            exclude_roots=exclude_roots,
            mll_stem=args.mll_stem,
            lj_stem=args.lj_stem,
            rj_stem=args.rj_stem,
            strict_imll=not args.allow_missing_imll,
            overwrite=args.overwrite,
        )

    run_many(files, max(1, args.jobs), fn)
    print("[done]")


if __name__ == "__main__":
    main()
