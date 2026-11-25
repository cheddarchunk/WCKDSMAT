#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-stream GFM Processor: LP → Head-Normalize → Mouth Residuals

This script applies the low-pass filtering, head-normalization (Kabsch), and
mouth residual computation pipeline to a single GFM landmark CSV.

Input:
    A single CSV containing landmark XYZ columns (e.g., ddk_gfm_sp_t_mm.csv)

Outputs:
    1. <stem>_lp_hn_om.csv  — full processed output
    2. <stem>_hn_lp.csv      — plain head-normalized coordinates only

Example:
    python ./pipeline/ves_gfm_lp_hn_om.py \
        --in ./data/tracking/gfm_sp_t/SUBJ001_task_gfm_sp_t.csv\
        --out ./data/tracking/lp_hn_om/SUBJ001_task_gfm_sp_t_hn_lp_om.csv \
        --anchors LCE,RCE,LTH,RTH,LBH,RBH,NT \
        --jaw-stems RJ,LJ \
        --lip-stems MUL,MLL,RCL,LCL \
        --lp 10 \

"""

import argparse, os, re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# ------------------ utility ------------------
def parse_list(s: str) -> List[str]:
    return [t.strip() for t in s.split(',') if t and t.strip()]

def _lower_map(columns) -> Dict[str,str]:
    return {str(c).lower(): c for c in columns}

def _find_triplet_cols(stem: str, columns) -> Tuple[str,str,str] | None:
    lower = _lower_map(columns)
    s = stem.lower()
    cx, cy, cz = lower.get(f"{s}_x"), lower.get(f"{s}_y"), lower.get(f"{s}_z")
    if cx and cy and cz:
        return cx, cy, cz
    return None

def has_triplet(df: pd.DataFrame, stem: str) -> bool:
    return _find_triplet_cols(stem, df.columns) is not None

def series_triplet_to_array(df: pd.DataFrame, stem: str) -> np.ndarray:
    cx, cy, cz = _find_triplet_cols(stem, df.columns)
    xs = pd.to_numeric(df[cx], errors='coerce')
    ys = pd.to_numeric(df[cy], errors='coerce')
    zs = pd.to_numeric(df[cz], errors='coerce')
    return np.column_stack([xs, ys, zs])

def list_all_triplet_stems(df: pd.DataFrame) -> List[str]:
    stems = {}
    for col in df.columns:
        m = re.match(r'^(.*)_(x|y|z)$', str(col), flags=re.IGNORECASE)
        if not m:
            continue
        base = m.group(1)
        info = stems.setdefault(base.lower(), {"base": base, "have": set()})
        info["have"].add(m.group(2).lower())
    return sorted([info["base"] for info in stems.values() if {"x","y","z"} <= info["have"]])

def find_time_column(df: pd.DataFrame) -> str:
    for name in ['timestamp','time_s','time','t','frame','frame_idx']:
        for c in df.columns:
            if c.lower() == name:
                return c
    raise ValueError("No valid time column found.")

# ------------------ geometry ------------------
def kabsch(P, Q):
    muP, muQ = P.mean(0), Q.mean(0)
    X, Y = P - muP, Q - muQ
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = muQ - R @ muP
    return R, t

def apply_rigid_point(R, t, p):
    return (R @ p) + t

# ------------------ filtering ------------------
def butter_lowpass_filtfilt(y, fs, fc, order=4):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(order, fc/nyq, btype='low')
    return filtfilt(b, a, y, axis=0, method='gust')

# ------------------ main stream ------------------
def process_stream(df, time_col, ref_frame, anchors, lp_hz, jaw_pair, lip_stems):
    t = pd.to_numeric(df[time_col], errors='coerce').to_numpy()
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    stems_all = list_all_triplet_stems(df)
    T = len(t)

    # LP
    Raw = {s: series_triplet_to_array(df, s) for s in stems_all}
    F = {s: np.column_stack([butter_lowpass_filtfilt(Raw[s][:,k], fs, lp_hz) for k in range(3)]) for s in stems_all}

    # HN
    Q = np.stack([F[a][ref_frame,:] for a in anchors], axis=0)
    hn = {s: np.full((T,3), np.nan) for s in stems_all}
    for i in range(T):
        P = np.stack([F[a][i,:] for a in anchors], axis=0)
        R, tr = kabsch(P, Q)
        for s in stems_all:
            hn[s][i,:] = apply_rigid_point(R, tr, F[s][i,:])

    # Jaw + residuals
    JL, RJ = jaw_pair
    jaw_cent = 0.5 * (hn[JL] + hn[RJ])
    mouth_resid = {s: (hn[s] - jaw_cent) if s.upper() == 'MLL' else hn[s] for s in lip_stems}

    # Outputs
    out = df.copy()
    for s in stems_all:
        out[f'hn_{s}_X'], out[f'hn_{s}_Y'], out[f'hn_{s}_Z'] = hn[s][:,0], hn[s][:,1], hn[s][:,2]
    out['jaw_cent_X'], out['jaw_cent_Y'], out['jaw_cent_Z'] = jaw_cent[:,0], jaw_cent[:,1], jaw_cent[:,2]
    for s in lip_stems:
        out[f'r_{s}_X'], out[f'r_{s}_Y'], out[f'r_{s}_Z'] = mouth_resid[s][:,0], mouth_resid[s][:,1], mouth_resid[s][:,2]

    hn_plain = pd.DataFrame({time_col: t})
    for s in stems_all:
        hn_plain[f'{s}_X'], hn_plain[f'{s}_Y'], hn_plain[f'{s}_Z'] = hn[s][:,0], hn[s][:,1], hn[s][:,2]
    return out, hn_plain, fs

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser(description='Single-stream GFM: LP → HN → Mouth residuals.')
    ap.add_argument('--in', dest='in_path', required=True, help='Input GFM CSV (mm)')
    ap.add_argument('--out', dest='out_path', required=True, help='Output CSV')
    ap.add_argument('--anchors', default='LCE,RCE,LTH,RTH,LBH,RBH,NT')
    ap.add_argument('--jaw-stems', default='RJ,LJ')
    ap.add_argument('--lip-stems', default='MUL,MLL,RCL,LCL')
    ap.add_argument('--lp', type=float, default=10.0)
    ap.add_argument('--ref-frame', type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    tcol = find_time_column(df)
    anchors = parse_list(args.anchors)
    jaw = parse_list(args.jaw_stems)
    lips = parse_list(args.lip_stems)

    out, hn_plain, fs = process_stream(df, tcol, args.ref_frame, anchors, args.lp, jaw, lips)
    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    out.to_csv(args.out_path, index=False)
    root, _ = os.path.splitext(args.out_path)
    hn_path = f'{root}_hn_lp.csv'
    hn_plain.to_csv(hn_path, index=False)
    print(f"[ok] wrote {args.out_path}  (fs≈{fs:.3f} Hz; LP={args.lp} Hz)")
    print(f"[ok] wrote {hn_path}  (HN+LP only)")

if __name__ == '__main__':
    main()
