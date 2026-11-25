"""Pair a webcam (GFM) stream with a motion-capture (CTX) stream for one segment.

Ported from ``replay_segmentation.py`` and ``spatial_alignment.py``:

1. cut: GFM rows with t in [t0, t1]; CTX rows with t + offset in [t0, t1];
   each rebased to start at 0 (CTX on its shifted timeline).
2. resample: CTX linearly interpolated onto the GFM frame times (values held
   at the edges, as ``np.interp`` does).
3. align: one rigid (Kabsch) transform CTX -> GFM fitted on the alignment
   landmarks, stacked over every frame where they are finite in both streams,
   then applied to every CTX point.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TIME_NAMES = ("time_s", "time", "timestamp", "t", "sec", "seconds")
POINT_RE = re.compile(r"^(?:hn_)?(?P<stem>[A-Za-z][A-Za-z0-9_]*?)(?:_hn)?_(?P<ax>[xyzXYZ])$")
SKIP_STEMS = {"com", "time", "frame", "segment", "source_row"}


def time_column(df: pd.DataFrame) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for n in TIME_NAMES:
        if n in lower:
            return lower[n]
    for c in df.columns:
        if "time" in str(c).lower() or str(c).lower().endswith("_s"):
            return c
    raise ValueError("no time column")


def read_points(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """{stem (lowercase): (T, 3)} from NAME_hn_X (package) or NAME_X / hn_NAME_x (legacy) columns.

    Un-normalized ``NAME_lp_*`` head trajectories are not validation points and are skipped.
    """
    found: dict[str, dict[str, str]] = {}
    for c in df.columns:
        m = POINT_RE.match(str(c))
        if not m:
            continue
        stem = m["stem"].lower()
        if stem in SKIP_STEMS or stem.endswith("_lp"):
            continue
        found.setdefault(stem, {})[m["ax"].lower()] = c
    return {s: df[[ax["x"], ax["y"], ax["z"]]].to_numpy(float)
            for s, ax in sorted(found.items()) if set(ax) == {"x", "y", "z"}}


def cut(df: pd.DataFrame, t0: float, t1: float, offset: float = 0.0) -> pd.DataFrame:
    tc = time_column(df)
    t = pd.to_numeric(df[tc], errors="coerce").to_numpy(float) + offset
    seg = df.loc[(t >= t0) & (t <= t1)].copy()
    if seg.empty:
        raise ValueError(f"no samples in [{t0}, {t1}] (offset {offset})")
    ts = pd.to_numeric(seg[tc], errors="coerce").to_numpy(float) + offset
    seg[tc] = ts - ts[0]
    return seg


def monotonic(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and average rows that share a timestamp (numeric columns only)."""
    tc = time_column(df)
    d = df.copy()
    d[tc] = d[tc].astype(float)
    d = d.sort_values(tc, kind="mergesort")
    num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    return d[num].groupby(tc, as_index=False).mean()


def straight_line_fraction(X: np.ndarray, tol: float = 1e-9, min_run: int = 3) -> float:
    """Share of reference frames that lie on an exact straight line (filled gaps), or missing.

    Measured motion is never exactly linear, even after low-pass filtering, so a zero
    second difference on all three axes marks interpolated (gap-filled) samples.
    """
    if len(X) < 3:
        return float("nan")
    d2 = np.abs(np.diff(X, 2, axis=0)).max(axis=1)
    flat = (d2 < tol) | ~np.isfinite(d2)
    # count only runs of >= min_run zero second differences (>= min_run + 2 collinear samples):
    # an isolated inflection point is not a filled gap
    straight = np.zeros(len(X), bool)
    edges = np.flatnonzero(np.diff(np.r_[0, flat.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= min_run:
            straight[s:e + 2] = True
    return float(np.mean(straight | ~np.isfinite(X).all(axis=1)))


def rigid_fit(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch: R, t minimizing ||R A + t - B|| over stacked points (N, 3)."""
    A0, B0 = A - A.mean(0), B - B.mean(0)
    H = A0.T @ B0
    if not np.isfinite(H).all():
        raise ValueError("non-finite values in the alignment landmarks")
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R, B.mean(0) - R @ A.mean(0)


def pair_segment(gfm: pd.DataFrame, ctx: pd.DataFrame, *, t0: float, t1: float, offset: float,
                 bases: list[str], min_valid_frames: int = 5) -> dict:
    """Cut, resample and align. Returns t, gfm/ctx point dicts on the GFM time base, and R, t."""
    g = monotonic(cut(gfm, t0, t1))
    c = monotonic(cut(ctx, t0, t1, offset))
    tg, tc = g[time_column(g)].to_numpy(float), c[time_column(c)].to_numpy(float)
    G, Craw = read_points(g), read_points(c)
    fill = {s: straight_line_fraction(a) for s, a in Craw.items()}
    C = {s: np.column_stack([np.interp(tg, tc, a[:, k]) for k in range(3)]) for s, a in Craw.items()}
    common = sorted(set(G) & set(C))
    bases = [b.lower() for b in bases]
    missing = [b for b in bases if b not in common]
    if missing:
        raise ValueError(f"alignment landmarks missing from one stream: {missing}")
    A = np.stack([C[b] for b in bases], axis=1)
    B = np.stack([G[b] for b in bases], axis=1)
    ok = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(B).all(axis=(1, 2))
    if ok.sum() < min_valid_frames:
        raise ValueError(f"only {int(ok.sum())} frames have every alignment landmark in both streams")
    R, tr = rigid_fit(A[ok].reshape(-1, 3), B[ok].reshape(-1, 3))
    return {"t": tg, "fs": 1.0 / float(np.median(np.diff(tg))) if len(tg) > 1 else float("nan"),
            "gfm": {s: G[s] for s in common}, "ctx": {s: C[s] @ R.T + tr for s in common},
            "R": R, "translation": tr, "n_fit_frames": int(ok.sum()), "ctx_fill": fill}
