#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bandini-style CTX vs GFM kinematic validation (WIDE FORMAT).

Inputs (in --tsdir):
    ctx_position.csv
    gfm_position.csv
    ctx_displacement.csv
    gfm_displacement.csv
    ctx_velocity.csv
    gfm_velocity.csv
    (optional) ctx_jawlip.csv
               gfm_jawlip.csv

Each TS CSV contains:
    <stem>_x, <stem>_y, <stem>_z, ...

Outputs (in --outdir):
    error_summary.csv  (wide-format)

PATCH:
  Adds derived articulator "imll" (isolated middle lower lip):
      jaw_mid = 0.5*(lj + rj)
      imll    = mll - jaw_mid
  for each stream (ctx/gfm) and each signal file (pos/disp/vel/acc),
  then includes imll in the per-stem metrics (x,y,z,3D).

PATCH (2026-01-27):
  Adds peak-lag cross-correlation metrics:
    - CC_peak
    - CC_peak_lag_samp
    - CC_peak_lag_ms
  using mean-centered, variance-normalized cross-correlation across lags.
  New CLI args:
    --fs (Hz, default 30.0)
    --max-lag-ms (default 500.0)
"""

import argparse
import os
import re
import numpy as np
import pandas as pd


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def load_csv(tsdir, name):
    path = os.path.join(tsdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)



def stem_of(col):
    """Convert e.g. mul_x → mul"""
    m = re.match(r"(.+)_([xyz])$", col)
    if not m:
        return None
    return m.group(1).lower()


def xyz_block(df, stem):
    """Return Nx3 array for a stem."""
    return np.vstack([
        df[f"{stem}_x"].to_numpy(float),
        df[f"{stem}_y"].to_numpy(float),
        df[f"{stem}_z"].to_numpy(float),
    ]).T


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def urmse(e):
    m = float(np.mean(e))
    return float(np.sqrt(max(0.0, np.mean(e ** 2) - m * m)))


def cc0(a, b):
    """Zero-lag Pearson correlation."""
    a = a.astype(float)
    b = b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def peak_cc_lag(a, b, fs=None, max_lag_samples=None):
    """
    Peak normalized cross-correlation across lags (mean-centered, variance-normalized).

    Returns:
      (lag_samples, lag_ms, cc_peak)

    Sign convention:
      +lag means b lags behind a (best alignment when b is shifted earlier by lag).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    if len(a) != len(b):
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]

    a = a - np.nanmean(a)
    b = b - np.nanmean(b)

    # If either signal is constant (or effectively so), correlation is undefined.
    if not np.isfinite(np.nanstd(a)) or not np.isfinite(np.nanstd(b)) or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan, np.nan, np.nan

    # Replace any NaNs with 0 *after* mean-centering (NaNs become "missing -> neutral").
    # If you prefer dropping NaNs instead, we can do that, but this is the most stable
    # for fixed-length streams.
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)

    cc = np.correlate(a, b, mode="full")
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0 or not np.isfinite(denom):
        return np.nan, np.nan, np.nan
    cc = cc / denom

    lags = np.arange(-len(a) + 1, len(a))

    if max_lag_samples is not None:
        max_lag_samples = int(max(0, max_lag_samples))
        mask = np.abs(lags) <= max_lag_samples
        lags = lags[mask]
        cc = cc[mask]

    if len(cc) == 0:
        return np.nan, np.nan, np.nan

    idx = int(np.argmax(cc))
    lag_samp = int(lags[idx])
    cc_peak = float(cc[idx])

    lag_ms = np.nan
    if fs is not None and np.isfinite(fs) and fs > 0:
        lag_ms = float(lag_samp / fs * 1000.0)

    return lag_samp, lag_ms, cc_peak


# -------------------------------------------------------------
# Robust range helper (Cortex-grounded, default 5–95%)
# -------------------------------------------------------------
def robust_range(arr, lo_pct=5.0, hi_pct=95.0):
    """
    Compute robust range as hi_pct - lo_pct percentile.
    Assumes arr is 1D, in mm.
    """
    if arr is None or len(arr) == 0:
        return np.nan
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    lo = np.nanpercentile(arr, lo_pct)
    hi = np.nanpercentile(arr, hi_pct)
    return float(hi - lo)


def range_agreement(rng_ctx, rng_gfm, floor_mm=0.5, eps=1e-12):
    """
    Compute range agreement metrics with a cortex-based validity floor.
    Returns (valid, diff, ratio, log_ratio).
    """
    if not np.isfinite(rng_ctx) or not np.isfinite(rng_gfm):
        return False, np.nan, np.nan, np.nan

    valid = bool(rng_ctx >= floor_mm)

    diff = float(rng_gfm - rng_ctx)

    ratio = float(rng_gfm / rng_ctx) if valid and rng_ctx > 0 else np.nan
    log_ratio = float(np.log10((rng_gfm + eps) / (rng_ctx + eps))) if valid else np.nan

    return valid, diff, ratio, log_ratio

def axis_metrics(A, B, lo_pct=5.0, hi_pct=95.0, range_floor_mm=0.5, fs=None, max_lag_samples=None):
    """
    Compute RMSE/uRMSE/Bias/Range/CC0 per axis.
    A: test system (e.g., FaceMesh), mm
    B: cortex (ground truth), mm

    Adds RMSE_peak_lag, uRMSE_peak_lag, Bias_peak_lag, n_overlap_peak_lag.
    """
    results = []
    axes = ["x", "y", "z"]

    for i, ax in enumerate(axes):
        a = A[:, i]
        b = B[:, i]
        # axis_metrics expects A=test, B=cortex -> e = cortex - test
        e = b - a

        # zero-lag metrics (legacy)
        R = rmse(e)
        U = urmse(e)
        bias = float(np.nanmean(e))

        rng_ctx = robust_range(b, lo_pct=lo_pct, hi_pct=hi_pct)
        rng_gfm = robust_range(a, lo_pct=lo_pct, hi_pct=hi_pct)

        valid, diff, ratio, log_ratio = range_agreement(
            rng_ctx, rng_gfm, floor_mm=range_floor_mm
        )

        rng = rng_ctx
        c = cc0(a, b)
        lag_samp, lag_ms, c_peak = peak_cc_lag(a, b, fs=fs, max_lag_samples=max_lag_samples)

        # --- compute peak-lag aligned RMSE/uRMSE/bias
        if np.isfinite(lag_samp):
            a_al, b_al = _align_by_lag(a, b, lag_samp)
            peak_metrics = _rmse_metrics(a_al, b_al)
            rmse_peak = peak_metrics["rmse"]
            urmse_peak = peak_metrics["urmse"]
            bias_peak = peak_metrics["bias"]
            n_overlap = peak_metrics["n"]
        else:
            rmse_peak = np.nan
            urmse_peak = np.nan
            bias_peak = np.nan
            n_overlap = 0

        results.append({
            "axis": ax,
            "RMSE": R,
            "uRMSE": U,
            "Bias": bias,

            "Range": rng,
            "Range_ctx": rng_ctx,
            "Range_gfm": rng_gfm,
            "Range_valid": valid,
            "Range_diff": diff,
            "Range_ratio": ratio,
            "Range_log_ratio": log_ratio,
"CC0": c,
            "CC_peak": c_peak,
            "CC_peak_lag_samp": lag_samp,
            "CC_peak_lag_ms": lag_ms,

            # NEW peak-lag metrics
            "RMSE_peak_lag": rmse_peak,
            "uRMSE_peak_lag": urmse_peak,
            "Bias_peak_lag": bias_peak,
            "n_overlap_peak_lag": int(n_overlap),
        })

    return results

# -------------------------------------------------------------
# Derived articulator: imll = mll - jaw_mid, where jaw_mid = (lj+rj)/2
# -------------------------------------------------------------
def add_imll_columns(df: pd.DataFrame, mll_stem="mll", lj_stem="lj", rj_stem="rj", out_stem="imll") -> pd.DataFrame:
    """
    Adds:
      jaw_mid_x/y/z
      imll_x/y/z  (mll - jaw_mid)
    Only if required source columns exist.
    Safe to call on any dataframe; no-op if columns are missing.
    """
    needed = []
    for s in (mll_stem, lj_stem, rj_stem):
        for ax in ("x", "y", "z"):
            needed.append(f"{s}_{ax}")

    if not all(c in df.columns for c in needed):
        return df

    for ax in ("x", "y", "z"):
        df[f"jaw_mid_{ax}"] = 0.5 * (df[f"{lj_stem}_{ax}"] + df[f"{rj_stem}_{ax}"])
        df[f"{out_stem}_{ax}"] = df[f"{mll_stem}_{ax}"] - df[f"jaw_mid_{ax}"]

    return df

def _align_by_lag(a, b, lag_samp):
    """
    Align two 1D arrays by integer lag (your peak_cc_lag sign convention).
    +lag means b lags behind a -> shift b earlier by lag.
    Crop to overlapping region and return arrays of equal length.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(len(a), len(b))
    if n <= 0:
        return np.array([], float), np.array([], float)
    a = a[:n]
    b = b[:n]

    lag = int(lag_samp)

    if lag > 0:
        # b lags behind -> drop first lag samples of b, last lag samples of a
        if lag >= n:
            return np.array([], float), np.array([], float)
        a_al = a[:-lag]
        b_al = b[lag:]
    elif lag < 0:
        k = -lag
        if k >= n:
            return np.array([], float), np.array([], float)
        a_al = a[k:]
        b_al = b[:-k]
    else:
        a_al, b_al = a, b

    m = min(len(a_al), len(b_al))
    if m <= 0:
        return np.array([], float), np.array([], float)
    return a_al[:m], b_al[:m]


def _rmse_metrics(a, b):
    """
    Compute RMSE, uRMSE (unbiased RMSE = std of error), Bias for aligned signals.
    Keeps the exact same uRMSE definition used elsewhere in this script.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return dict(rmse=np.nan, urmse=np.nan, bias=np.nan, n=0)

    a_ok = a[mask]
    b_ok = b[mask]

    # error sign consistent with your pipeline: cortex - test (b - a)
    err = b_ok - a_ok

    rmse_val = rmse(err)
    urmse_val = urmse(err)          # <-- THIS is the key correction
    bias_val = float(np.mean(err))

    return dict(rmse=rmse_val, urmse=urmse_val, bias=bias_val, n=int(len(a_ok)))

def metrics_3d(A, B, lo_pct=5.0, hi_pct=95.0, range_floor_mm=0.5, fs=None, max_lag_samples=None):
    """
    3D magnitude metrics.
    Range/correlation computed from scalar 3D excursion magnitudes (||X - mean(Cortex)||).

    Adds RMSE_peak_lag, uRMSE_peak_lag, Bias_peak_lag, n_overlap_peak_lag.
    """
    e = B - A
    mags = np.linalg.norm(e, axis=1)

    R = rmse(mags)
    U = urmse(mags)

        # Scalar 3D excursion traces (translation-invariant) for range/correlation.
    # We anchor both systems to the Cortex mean position so magnitudes are directly comparable.
    refB = np.nanmean(B, axis=0)
    sigB = np.linalg.norm(B - refB, axis=1)  # cortex excursion magnitude
    sigA = np.linalg.norm(A - refB, axis=1)  # test excursion magnitude in cortex reference
    rng_ctx = robust_range(sigB, lo_pct=lo_pct, hi_pct=hi_pct)
    rng_gfm = robust_range(sigA, lo_pct=lo_pct, hi_pct=hi_pct)

    valid, diff, ratio, log_ratio = range_agreement(
        rng_ctx, rng_gfm, floor_mm=range_floor_mm
    )

    rng = rng_ctx
    c3 = cc0(sigA, sigB)
    lag_samp, lag_ms, c3_peak = peak_cc_lag(sigA, sigB, fs=fs, max_lag_samples=max_lag_samples)

    # peak-lag alignment for 3D mags
    if np.isfinite(lag_samp):
        a_al, b_al = _align_by_lag(sigA, sigB, lag_samp)
        peak_metrics = _rmse_metrics(a_al, b_al)
        rmse_peak = peak_metrics["rmse"]
        urmse_peak = peak_metrics["urmse"]
        bias_peak = peak_metrics["bias"]
        n_overlap = peak_metrics["n"]
    else:
        rmse_peak = np.nan
        urmse_peak = np.nan
        bias_peak = np.nan
        n_overlap = 0

    return {
        "axis": "3D",
        "RMSE": R,
        "uRMSE": U,
        "Bias": float(np.nanmean(mags)),

        "Range": rng,
        "Range_ctx": rng_ctx,
        "Range_gfm": rng_gfm,
        "Range_valid": valid,
        "Range_diff": diff,
        "Range_ratio": ratio,
        "Range_log_ratio": log_ratio,
"CC0": c3,
        "CC_peak": c3_peak,
        "CC_peak_lag_samp": lag_samp,
        "CC_peak_lag_ms": lag_ms,

        # NEW peak-lag metrics
        "RMSE_peak_lag": rmse_peak,
        "uRMSE_peak_lag": urmse_peak,
        "Bias_peak_lag": bias_peak,
        "n_overlap_peak_lag": int(n_overlap),
    }



# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fs", type=float, default=30.0, help="Sampling rate (Hz) of the aligned time series (used to convert lag samples -> ms).")
    ap.add_argument("--max-lag-ms", type=float, default=500.0, help="Max lag search window (milliseconds) for peak cross-correlation.")
    args = ap.parse_args()

    TS = args.tsdir
    OUT = args.outdir
    fs = args.fs
    max_lag_samples = None
    if args.max_lag_ms is not None and np.isfinite(args.max_lag_ms) and args.max_lag_ms >= 0 and fs is not None and np.isfinite(fs) and fs > 0:
        max_lag_samples = int(round((args.max_lag_ms / 1000.0) * fs))
    os.makedirs(OUT, exist_ok=True)

    # Load core time-series
    ctx_pos = load_csv(TS, "ctx_position.csv")
    gfm_pos = load_csv(TS, "gfm_position.csv")

    ctx_disp = load_csv(TS, "ctx_displacement.csv")
    gfm_disp = load_csv(TS, "gfm_displacement.csv")

    ctx_vel = load_csv(TS, "ctx_velocity.csv")
    gfm_vel = load_csv(TS, "gfm_velocity.csv")

    # ----------------------------
    # PATCH: add imll columns
    # ----------------------------
    # We do this per-file so imll exists consistently for pos/disp/vel/acc.
    ctx_pos = add_imll_columns(ctx_pos)
    gfm_pos = add_imll_columns(gfm_pos)

    ctx_disp = add_imll_columns(ctx_disp)
    gfm_disp = add_imll_columns(gfm_disp)

    ctx_vel = add_imll_columns(ctx_vel)
    gfm_vel = add_imll_columns(gfm_vel)

    # Identify anatomical stems from displacement (* etc.)
    stems = sorted({
        s for s in (stem_of(c) for c in ctx_disp.columns)
        if s is not None and f"{s}_x" in gfm_disp.columns and f"{s}_y" in gfm_disp.columns and f"{s}_z" in gfm_disp.columns
    })

    if not stems:
        raise ValueError("No * stems with xyz triplets found in displacement CSVs.")

    rows = []

    # ---------------------------------------------------------
    # Per-stem metrics for position, disp, vel, acc
    # ---------------------------------------------------------
    for stem in stems:
        # note: first arg to axis_metrics/metrics_3d = test (gfm), second = cortex (ctx)

        # POSITION
        ctx_pos_block = xyz_block(ctx_pos, stem)
        test_pos = xyz_block(gfm_pos, stem)

        # DISPLACEMENT
        ctx_disp_block = xyz_block(ctx_disp, stem)
        test_disp = xyz_block(gfm_disp, stem)

        # VELOCITY
        ctx_vel_block = xyz_block(ctx_vel, stem)
        test_vel = xyz_block(gfm_vel, stem)

        signals = [
            ("pos",  test_pos,  ctx_pos_block),
            ("disp", test_disp, ctx_disp_block),
            ("vel",  test_vel,  ctx_vel_block),
                   ]

        for sig, A, B in signals:
            # per-axis
            for m in axis_metrics(A, B, fs=fs, max_lag_samples=max_lag_samples):
                rows.append({
                    "stem": stem,
                    "signal": sig,
                    "axis": m["axis"],
                    "RMSE": m["RMSE"],
                    "uRMSE": m["uRMSE"],
                    "Bias": m["Bias"],
                    "Range": m["Range"],
"CC0": m["CC0"],
                    "CC_peak": m.get("CC_peak", np.nan),
                    "CC_peak_lag_samp": m.get("CC_peak_lag_samp", np.nan),
                    "CC_peak_lag_ms": m.get("CC_peak_lag_ms", np.nan),
                    "Range_ctx": m["Range_ctx"],
                    "Range_gfm": m["Range_gfm"],
                    "Range_valid": m["Range_valid"],
                    "Range_diff": m["Range_diff"],
                    "Range_ratio": m["Range_ratio"],
                    "Range_log_ratio": m["Range_log_ratio"],
                    "RMSE_peak_lag": m.get("RMSE_peak_lag", np.nan),
                    "uRMSE_peak_lag": m.get("uRMSE_peak_lag", np.nan),
                    "Bias_peak_lag": m.get("Bias_peak_lag", np.nan),
                    "n_overlap_peak_lag": m.get("n_overlap_peak_lag", np.nan),
                })

            # 3D magnitude metrics
            m3 = metrics_3d(A, B, fs=fs, max_lag_samples=max_lag_samples)
            rows.append({
                "stem": stem,
                "signal": sig,
                "axis": "3D",
                "RMSE": m3["RMSE"],
                "uRMSE": m3["uRMSE"],
                "Bias": m3["Bias"],
                "Range": m3["Range"],
"CC0": m3["CC0"],
                "CC_peak": m3.get("CC_peak", np.nan),
                "CC_peak_lag_samp": m3.get("CC_peak_lag_samp", np.nan),
                "CC_peak_lag_ms": m3.get("CC_peak_lag_ms", np.nan),
                "Range_ctx": m3["Range_ctx"],
                "Range_gfm": m3["Range_gfm"],
                "Range_valid": m3["Range_valid"],
                "Range_diff": m3["Range_diff"],
                "Range_ratio": m3["Range_ratio"],
                "Range_log_ratio": m3["Range_log_ratio"],
                "RMSE_peak_lag": m3.get("RMSE_peak_lag", np.nan),
                "uRMSE_peak_lag": m3.get("uRMSE_peak_lag", np.nan),
                "Bias_peak_lag": m3.get("Bias_peak_lag", np.nan),
                "n_overlap_peak_lag": m3.get("n_overlap_peak_lag", np.nan),
            })

    # ---------------------------------------------------------
    # Jaw distance trajectories à la Bandini 2017
    # NJL / NJR / NJmid (NT → LJ / RJ / midpoint)
    # ---------------------------------------------------------
    required = ["nt", "lj", "rj"]
    if all(f"{stem}_x" in ctx_pos.columns for stem in required) and all(f"{stem}_x" in gfm_pos.columns for stem in required):

        def dist(a, b):
            return np.linalg.norm(a - b, axis=1)

        ntC = xyz_block(ctx_pos, "nt")
        ljC = xyz_block(ctx_pos, "lj")
        rjC = xyz_block(ctx_pos, "rj")

        ntG = xyz_block(gfm_pos, "nt")
        ljG = xyz_block(gfm_pos, "lj")
        rjG = xyz_block(gfm_pos, "rj")

        NJL_ctx = dist(ntC, ljC)
        NJL_gfm = dist(ntG, ljG)

        NJR_ctx = dist(ntC, rjC)
        NJR_gfm = dist(ntG, rjG)

        midC = 0.5 * (ljC + rjC)
        midG = 0.5 * (ljG + rjG)

        NJM_ctx = dist(ntC, midC)
        NJM_gfm = dist(ntG, midG)

        jaw_trajs = [
            ("jaw_L",   NJL_ctx, NJL_gfm),
            ("jaw_R",   NJR_ctx, NJR_gfm),
            ("jaw_mid", NJM_ctx, NJM_gfm),
        ]

        for name, ctx_d, test_d in jaw_trajs:
            # error sign: cortex - test
            e = ctx_d - test_d

            R = rmse(e)
            U = urmse(e)
            bias = float(np.nanmean(e))

            rng_ctx = robust_range(ctx_d)
            rng_gfm = robust_range(test_d)
            valid, diff, ratio, log_ratio = range_agreement(rng_ctx, rng_gfm, floor_mm=0.5)

            # legacy Range = cortex
            rng = rng_ctx
            c = cc0(test_d, ctx_d)  # correlation between the two distance traces
            lag_samp, lag_ms, c_peak = peak_cc_lag(test_d, ctx_d, fs=fs, max_lag_samples=max_lag_samples)

            rows.append({
                "stem": name,
                "signal": "jawtraj",
                "axis": "dist",
                "RMSE": R,
                "uRMSE": U,
                "Bias": bias,
                "Range": rng,            # keep consistent with other rows
"CC0": c,
                "CC_peak": c_peak,
                "CC_peak_lag_samp": lag_samp,
                "CC_peak_lag_ms": lag_ms,
                "Range_ctx": rng_ctx,
                "Range_gfm": rng_gfm,
                "Range_valid": valid,
                "Range_diff": diff,
                "Range_ratio": ratio,
                "Range_log_ratio": log_ratio,
            })

    # ---------------------------------------------------------
    # Save WIDE summary
    # ---------------------------------------------------------
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "error_summary.csv")
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
