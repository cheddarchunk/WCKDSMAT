#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1: Spatial alignment of CTX to GFM.

WHAT IT DOES (important):
- Takes CTX at (e.g.) 120 Hz and GFM at (e.g.) 15 Hz
- Resamples CTX onto the GFM time base (so CTX becomes 15 Hz)
- Fits a rigid transform (Kabsch) using basis landmarks on the resampled CTX vs GFM
- Applies transform to *all shared stems* and writes paired 15 Hz outputs.

PATCHES (vs your GOOD 30 Hz script):
- Robust time handling: sort by time, collapse duplicate timestamps (mean)
- Auto output suffix based on target (GFM) sampling rate: final_tracking_15Hz, 30Hz, etc.
- Optional --fs-label override if you want forced naming
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_time_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "time" in c.lower():
            return c
    raise ValueError(f"No time column found in {list(df.columns)}")


def estimate_fs(t) -> float:
    t = np.asarray(t, float)
    if len(t) < 2:
        return np.nan
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return np.nan
    return 1.0 / np.median(dt)


def fs_label_from_time(t, default="unkHz") -> str:
    fs = estimate_fs(t)
    if not np.isfinite(fs) or fs <= 0:
        return default
    # Round sensibly: 29.97 -> 30, 14.98 -> 15
    return f"{int(round(fs))}Hz"


def coerce_time_monotonic(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    """
    Ensure time is strictly increasing for interpolation:
    - convert to float
    - sort by time
    - collapse duplicates by mean (numeric columns only)
    """
    d = df.copy()
    d[tcol] = d[tcol].astype(float)

    # Sort
    d = d.sort_values(tcol, kind="mergesort")

    # Collapse duplicates (mean). Keep only numeric columns (including time).
    num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    d = d[num_cols].groupby(tcol, as_index=False).mean()

    # After grouping, tcol is the first column and is strictly increasing.
    return d


def extract_xyz_triplets(cols, exclude_stems=("time", "com")):
    """
    Return stems that have complete _x/_y/_z triplets.
    Case-insensitive. No required prefix.
    """
    tmp = {}
    for c in cols:
        cl = c.lower()
        if not cl.endswith(("_x", "_y", "_z")):
            continue
        stem, ax = cl.rsplit("_", 1)

        # Exclude metadata stems
        if any(stem == e or stem.endswith(e) for e in exclude_stems):
            continue

        tmp.setdefault(stem, set()).add(ax.upper())

    return sorted(s for s, axes in tmp.items() if axes == {"X", "Y", "Z"})


def _colmap(df: pd.DataFrame) -> dict:
    """lowercase->actual column name mapping (last win)."""
    return {c.lower(): c for c in df.columns}


def xyz(df: pd.DataFrame, stem_lower: str) -> np.ndarray:
    """
    Return (N,3) array for the given stem (already lowercase),
    pulling columns <stem>_x, <stem>_y, <stem>_z case-insensitively.
    """
    m = _colmap(df)
    for ax in ("x", "y", "z"):
        if f"{stem_lower}_{ax}" not in m:
            raise KeyError(f"Missing column: {stem_lower}_{ax}")
    X = df[m[f"{stem_lower}_x"]].to_numpy(float)
    Y = df[m[f"{stem_lower}_y"]].to_numpy(float)
    Z = df[m[f"{stem_lower}_z"]].to_numpy(float)
    return np.stack([X, Y, Z], axis=1)


def canonical_out_stem(stem_lower: str) -> str:
    s = stem_lower.lower()
    if s.startswith("hn_"):
        s = s[3:]
    return s


def get_basis_cols(df: pd.DataFrame, base: str) -> tuple[str, str, str]:
    """
    For a base like 'rth', accept either:
      - rth_x / rth_y / rth_z
      - hn_rth_x / hn_rth_y / hn_rth_z
    in a case-insensitive way.
    Returns actual column names (cx, cy, cz).
    """
    m = _colmap(df)
    base = base.lower()

    def pick(ax):
        for cand in (f"{base}_{ax}", f"hn_{base}_{ax}"):
            if cand in m:
                return m[cand]
        raise KeyError(f"Missing basis '{base}' axis '{ax}' (tried {base}_{ax} and hn_{base}_{ax})")

    return pick("x"), pick("y"), pick("z")


def build_basis_per_frame(df: pd.DataFrame, bases: list[str]) -> np.ndarray:
    """
    Returns (T, n_bases, 3)
    """
    arrs = []
    for b in bases:
        cx, cy, cz = get_basis_cols(df, b)
        pts = np.stack(
            [df[cx].to_numpy(float), df[cy].to_numpy(float), df[cz].to_numpy(float)],
            axis=1
        )  # (T,3)
        arrs.append(pts)
    return np.stack(arrs, axis=1)  # (T,n_bases,3)


def rigid_fit(A: np.ndarray, B: np.ndarray):
    """
    Solve A->B rigid transform using Kabsch.
    A, B: (N,3) with finite values.
    """
    A0 = A - A.mean(axis=0)
    B0 = B - B.mean(axis=0)
    H = A0.T @ B0

    if not np.isfinite(H).all():
        raise ValueError("Non-finite values in Kabsch covariance matrix H (NaN/inf in basis data).")

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = B.mean(axis=0) - R @ A.mean(axis=0)
    return R, t


def fit_rigid_from_bases(ctx_rs: pd.DataFrame, gfm: pd.DataFrame, bases: list[str], min_valid_frames: int = 5):
    """
    Build (T,n_bases,3) for each stream, mask invalid frames, then fit.
    """
    A_pf = build_basis_per_frame(ctx_rs, bases)  # (T,n_bases,3)
    B_pf = build_basis_per_frame(gfm,    bases)  # (T,n_bases,3)

    okA = np.isfinite(A_pf).all(axis=(1, 2))
    okB = np.isfinite(B_pf).all(axis=(1, 2))
    ok = okA & okB

    n_ok = int(ok.sum())
    if n_ok < min_valid_frames:
        raise ValueError(
            f"Not enough valid frames for rigid fit: {n_ok}/{len(ok)} frames have finite XYZ for all bases "
            f"in BOTH streams."
        )

    A = A_pf[ok].reshape(-1, 3)
    B = B_pf[ok].reshape(-1, 3)

    if np.linalg.norm(np.std(A, axis=0)) == 0 or np.linalg.norm(np.std(B, axis=0)) == 0:
        raise ValueError("Basis points have ~zero variance after filtering (degenerate geometry).")

    print(f"[info] Rigid fit using {n_ok} valid frames (stacked points = {A.shape[0]}).")
    return rigid_fit(A, B)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--gfm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", default="rigid", choices=["rigid"])
    ap.add_argument("--bases", nargs="+", required=True)
    ap.add_argument("--min-valid-frames", type=int, default=5)
    ap.add_argument("--fs-label", default=None,
                    help="Override output suffix label (e.g., '15Hz'). If omitted, inferred from GFM time base.")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    ctx = pd.read_csv(args.ctx)
    gfm = pd.read_csv(args.gfm)

    tcol_ctx = find_time_column(ctx)
    tcol_gfm = find_time_column(gfm)

    # Make time safe for interpolation + alignment
    ctx = coerce_time_monotonic(ctx, tcol_ctx)
    gfm = coerce_time_monotonic(gfm, tcol_gfm)

    t_ctx = ctx[tcol_ctx].to_numpy(float)
    t_gfm = gfm[tcol_gfm].to_numpy(float)

    print(f"[info] Estimated CTX fs ≈ {estimate_fs(t_ctx):0.2f} Hz")
    print(f"[info] Estimated GFM fs ≈ {estimate_fs(t_gfm):0.2f} Hz")

    # Shared XYZ stems (prefix-agnostic)
    ctx_stems = extract_xyz_triplets(ctx.columns)
    gfm_stems = extract_xyz_triplets(gfm.columns)
    common = sorted(set(ctx_stems) & set(gfm_stems))
    if not common:
        raise ValueError("No common XYZ triplets between CTX and GFM.")

    # Resample CTX onto GFM time base (THIS is what makes output CTX = 15 Hz if GFM is 15 Hz)
    t_target = t_gfm
    ctx_rs = pd.DataFrame({"time_s": t_target})

    # Interpolate every non-time numeric column
    for c in ctx.columns:
        if c == tcol_ctx:
            continue
        ctx_rs[c] = np.interp(t_target, t_ctx, ctx[c].astype(float).to_numpy())

    bases = [b.lower() for b in args.bases]

    # Robust fit
    R, t = fit_rigid_from_bases(ctx_rs, gfm, bases, min_valid_frames=args.min_valid_frames)

    # Apply transform to CTX for all shared stems
    ctx_aligned = pd.DataFrame({"time_s": ctx_rs["time_s"]})
    for stem in common:
        pts = xyz(ctx_rs, stem)  # (N,3)
        pts_aligned = (R @ pts.T).T + t
        outstem = canonical_out_stem(stem)
        ctx_aligned[f"{outstem}_x"] = pts_aligned[:, 0]
        ctx_aligned[f"{outstem}_y"] = pts_aligned[:, 1]
        ctx_aligned[f"{outstem}_z"] = pts_aligned[:, 2]

    # Canonicalize GFM too (same time base = 15 Hz)
    gfm_aligned = pd.DataFrame({"time_s": gfm[tcol_gfm].astype(float)})
    for stem in common:
        pts = xyz(gfm, stem)
        outstem = canonical_out_stem(stem)
        gfm_aligned[f"{outstem}_x"] = pts[:, 0]
        gfm_aligned[f"{outstem}_y"] = pts[:, 1]
        gfm_aligned[f"{outstem}_z"] = pts[:, 2]

    # Write outputs
    ctx_base = Path(args.ctx).stem
    gfm_base = Path(args.gfm).stem

    out_ctx = outdir / f"{ctx_base}_aligned.csv"
    out_gfm = outdir / f"{gfm_base}_aligned.csv"

    ctx_aligned.to_csv(out_ctx, index=False)
    gfm_aligned.to_csv(out_gfm, index=False)
    print(f"[out] {out_ctx}")
    print(f"[out] {out_gfm}")

    # Final suffix based on TARGET (GFM) fs
    label = args.fs_label if args.fs_label else fs_label_from_time(t_target, default="unkHz")

    final_ctx = outdir / f"{ctx_base}_final_tracking_{label}.csv"
    final_gfm = outdir / f"{gfm_base}_final_tracking_{label}.csv"

    ctx_aligned.to_csv(final_ctx, index=False)
    gfm_aligned.to_csv(final_gfm, index=False)

    print(f"[out] Final aligned CTX saved: {final_ctx}")
    print(f"[out] Resampled GFM saved:    {final_gfm}")
    print("[done] Spatial alignment completed successfully.")


if __name__ == "__main__":
    main()
