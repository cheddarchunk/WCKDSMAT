# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 WCKDSMAT contributors
"""Agreement metrics between a test system (webcam, A) and the reference (motion capture, B).

Ported from the research scripts ``bandini_full_pipeline_STEP2.py`` and
``temporal_validation.py`` (Interspeech 2026, section 3.6). Error sign is
reference minus test (B - A), as before.

* per axis: RMSE, uRMSE (SD of the error), bias, 5-95 % ranges and their
  agreement, zero-lag and peak cross-correlation (+ lag), and RMSE/uRMSE/bias
  after removing that lag.
* 3D: the same on the per-frame 3D error distance ``||B - A||``; ranges and
  correlations use each stream's distance from the reference's mean position.
* spectral: Welch dominant frequency, spectral centroid, 0-5 Hz band power.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, welch


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def urmse(e: np.ndarray) -> float:
    m = float(np.mean(e))
    return float(np.sqrt(max(0.0, np.mean(e ** 2) - m * m)))


def cc0(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def peak_cc_lag(a, b, fs=None, max_lag_samples=None) -> tuple[float, float, float]:
    """(lag_samples, lag_ms, cc_peak); +lag means b lags behind a. NaNs count as 0 after centering."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    n = min(len(a), len(b))
    a, b = a[:n] - np.nanmean(a[:n]), b[:n] - np.nanmean(b[:n])
    sa, sb = np.nanstd(a), np.nanstd(b)
    if not (np.isfinite(sa) and np.isfinite(sb)) or sa == 0 or sb == 0:
        return np.nan, np.nan, np.nan
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0 or not np.isfinite(denom):
        return np.nan, np.nan, np.nan
    cc = np.correlate(a, b, mode="full") / denom
    lags = np.arange(-n + 1, n)
    if max_lag_samples is not None:
        keep = np.abs(lags) <= int(max(0, max_lag_samples))
        lags, cc = lags[keep], cc[keep]
    if not len(cc):
        return np.nan, np.nan, np.nan
    i = int(np.argmax(cc))
    lag = int(lags[i])
    lag_ms = float(lag / fs * 1000.0) if fs and np.isfinite(fs) and fs > 0 else np.nan
    return lag, lag_ms, float(cc[i])


def robust_range(x, lo=5.0, hi=95.0) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return float("nan")
    return float(np.percentile(x, hi) - np.percentile(x, lo))


def range_agreement(r_ref, r_test, floor_mm=0.5, eps=1e-12):
    if not (np.isfinite(r_ref) and np.isfinite(r_test)):
        return False, np.nan, np.nan, np.nan
    valid = bool(r_ref >= floor_mm)
    ratio = float(r_test / r_ref) if valid and r_ref > 0 else np.nan
    log_ratio = float(np.log10((r_test + eps) / (r_ref + eps))) if valid else np.nan
    return valid, float(r_test - r_ref), ratio, log_ratio


def align_by_lag(a, b, lag):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b, lag = a[:n], b[:n], int(lag)
    if abs(lag) >= n:
        return np.array([]), np.array([])
    if lag > 0:
        return a[:-lag], b[lag:]
    if lag < 0:
        return a[-lag:], b[:lag]
    return a, b


def _lagged_errors(a, b, lag):
    if not np.isfinite(lag):
        return dict(RMSE_peak_lag=np.nan, uRMSE_peak_lag=np.nan, Bias_peak_lag=np.nan, n_overlap_peak_lag=0)
    a2, b2 = align_by_lag(a, b, lag)
    ok = np.isfinite(a2) & np.isfinite(b2)
    if not ok.any():
        return dict(RMSE_peak_lag=np.nan, uRMSE_peak_lag=np.nan, Bias_peak_lag=np.nan, n_overlap_peak_lag=0)
    e = b2[ok] - a2[ok]
    return dict(RMSE_peak_lag=rmse(e), uRMSE_peak_lag=urmse(e), Bias_peak_lag=float(np.mean(e)),
                n_overlap_peak_lag=int(ok.sum()))


def _ranges(sig_ref, sig_test, lo, hi, floor):
    r_ref, r_test = robust_range(sig_ref, lo, hi), robust_range(sig_test, lo, hi)
    valid, diff, ratio, log_ratio = range_agreement(r_ref, r_test, floor)
    return dict(Range=r_ref, Range_ctx=r_ref, Range_gfm=r_test, Range_valid=valid, Range_diff=diff,
                Range_ratio=ratio, Range_log_ratio=log_ratio)


def axis_metrics(A, B, *, fs, max_lag_samples, lo=5.0, hi=95.0, floor=0.5) -> list[dict]:
    out = []
    for i, ax in enumerate("xyz"):
        a, b = A[:, i], B[:, i]
        e = b - a
        lag, lag_ms, cpk = peak_cc_lag(a, b, fs, max_lag_samples)
        out.append({"axis": ax, "RMSE": rmse(e), "uRMSE": urmse(e), "Bias": float(np.nanmean(e)),
                    **_ranges(b, a, lo, hi, floor), "CC0": cc0(a, b), "CC_peak": cpk,
                    "CC_peak_lag_samp": lag, "CC_peak_lag_ms": lag_ms, **_lagged_errors(a, b, lag)})
    return out


def metrics_3d(A, B, *, fs, max_lag_samples, lo=5.0, hi=95.0, floor=0.5) -> dict:
    mags = np.linalg.norm(B - A, axis=1)
    ref = np.nanmean(B, axis=0)
    sig_b, sig_a = np.linalg.norm(B - ref, axis=1), np.linalg.norm(A - ref, axis=1)
    lag, lag_ms, cpk = peak_cc_lag(sig_a, sig_b, fs, max_lag_samples)
    return {"axis": "3D", "RMSE": rmse(mags), "uRMSE": urmse(mags), "Bias": float(np.nanmean(mags)),
            **_ranges(sig_b, sig_a, lo, hi, floor), "CC0": cc0(sig_a, sig_b), "CC_peak": cpk,
            "CC_peak_lag_samp": lag, "CC_peak_lag_ms": lag_ms, **_lagged_errors(sig_a, sig_b, lag)}


# -- temporal_validation.py ----------------------------------------------------
def zscore_cross_correlation(x, y, fs, max_lag_ms=250.0) -> dict:
    """Length-normalized z-score cross-correlation; peak = largest |cc| within the lag window."""
    n = min(len(x), len(y))
    if n < 5:
        return {"cc_zero": np.nan, "cc_peak": np.nan, "lag_peak_ms": np.nan}
    xz = (x[:n] - np.mean(x[:n])) / (np.std(x[:n]) + 1e-12)
    yz = (y[:n] - np.mean(y[:n])) / (np.std(y[:n]) + 1e-12)
    cc = correlate(xz, yz, mode="full") / n
    lags = np.arange(-n + 1, n)
    win = np.abs(lags) <= int(round(max_lag_ms / 1000.0 * fs))
    if not win.any():
        return {"cc_zero": np.nan, "cc_peak": np.nan, "lag_peak_ms": np.nan}
    i = int(np.argmax(np.abs(cc[win])))
    return {"cc_zero": float(cc[lags == 0][0]), "cc_peak": float(cc[win][i]),
            "lag_peak_ms": float(lags[win][i] / fs * 1000.0)}


def psd_features(x, fs, band_hi=5.0) -> dict:
    x = np.asarray(x, float)
    x = x - np.mean(x)
    nan = {"dom_freq": np.nan, "centroid": np.nan, "band_power_0_5": np.nan, "total_power": np.nan}
    if len(x) < 16:
        return nan
    f, P = welch(x, fs=fs, nperseg=max(min(256, len(x) // 4), 8), scaling="density")
    if P.size == 0 or np.sum(P) <= 0:
        return nan
    return {"dom_freq": float(f[int(np.argmax(P))]), "centroid": float(np.sum(f * P / (np.sum(P) + 1e-12))),
            "band_power_0_5": float(np.sum(P[(f >= 0) & (f <= band_hi)])), "total_power": float(np.sum(P))}
