"""Stage 3: head normalization (rigid Kabsch) and zero-phase low-pass filtering.

Reads a calibrated landmark CSV and writes, for every configured point:

* ``<NAME>_hn_<AXIS>``: head-normalized and low-passed. Derived points
  (e.g. ``IMLL``) are computed from these.
* ``<NAME>_lp_<AXIS>``: low-passed only (not normalized), for the points in
  ``preprocess.preserve_unnormalized`` (head-movement analyses).

The recording is split into segments wherever the time step jumps by more
than ``gap_split_factor`` x the median step. Gap filling and filtering never
cross a segment boundary. Short dropouts (up to ``max_interp_gap_frames``)
are filled linearly; longer ones stay NaN without spoiling the rest of the
channel. All settings come from :class:`wckdsmat.config.Config`.

Behavior merged from the earlier research scripts (see docs/PROVENANCE.md):
dropout handling and preserved head trajectories from the POLO batch
script, single-file and residual logic from the oral-motor script.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .config import AXES, Config, column

TIME_CANDIDATES = ("time_s", "timestamp", "time", "t")


@dataclass
class PreprocessReport:
    source: str = ""
    n_frames: int = 0
    fs_hz: float = float("nan")
    lowpass_hz_effective: float = float("nan")    # after any nyquist_cap clamp
    duplicate_timestamps_dropped: int = 0
    segments: list[tuple[int, int]] = field(default_factory=list)   # [start, end) output rows
    ref_frame_requested: int = 0
    ref_frame_used: int = 0
    anchors_used: list[str] = field(default_factory=list)   # tracked at the reference frame
    frames_without_head_pose: int = 0
    samples_interpolated: int = 0
    samples_too_short_to_filter: int = 0
    missing_landmarks: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------------
def find_time_column(df: pd.DataFrame) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for name in TIME_CANDIDATES:
        if name in lower:
            return lower[name]
    raise ValueError(f"no time column found; expected one of {TIME_CANDIDATES} (seconds)")


def read_point(df: pd.DataFrame, name: str) -> np.ndarray | None:
    """(T, 3) array for a point, accepting NAME_t_X or legacy NAME_x/NAME_X columns."""
    lower = {str(c).lower(): c for c in df.columns}
    for pattern in ("{n}_t_{a}", "{n}_{a}"):
        cols = [lower.get(pattern.format(n=name, a=a).lower()) for a in AXES]
        if all(cols):
            return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return None


def split_segments(t: np.ndarray, factor: float) -> list[tuple[int, int]]:
    dt = np.diff(t)
    cut = np.flatnonzero(dt > factor * np.median(dt)) + 1
    edges = [0, *cut.tolist(), len(t)]
    return list(zip(edges[:-1], edges[1:]))


# ---------------------------------------------------------------------------
# geometry and filtering
# ---------------------------------------------------------------------------
def kabsch_batch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame rigid fit of anchors P (T, K, 3) onto reference Q (K, 3).

    Returns R (T, 3, 3) and t (T, 3) such that ``R @ p + t`` maps frame points
    into the reference head pose. Frames with any non-finite anchor get NaN.
    """
    T = P.shape[0]
    R = np.full((T, 3, 3), np.nan)
    tr = np.full((T, 3), np.nan)
    ok = np.isfinite(P).all(axis=(1, 2))
    if not ok.any():
        return R, tr
    Pv = P[ok]
    mu_p = Pv.mean(axis=1)
    mu_q = Q.mean(axis=0)
    H = np.einsum("tki,kj->tij", Pv - mu_p[:, None, :], Q - mu_q)
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1))
    Ut = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ Ut))
    D = np.zeros((len(d), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    Rv = V @ D @ Ut
    R[ok] = Rv
    tr[ok] = mu_q - np.einsum("tij,tj->ti", Rv, mu_p)
    return R, tr


def pick_ref_frame(finite: np.ndarray, ref: int, max_search: int, min_anchors: int) -> int:
    """Frame within +/- max_search of ref with the most tracked anchors (>= min_anchors).

    Ties go to the frame nearest ref. ``finite`` is (T, K) booleans.
    """
    T = finite.shape[0]
    lo, hi = max(0, ref - max_search), min(T, ref + max_search + 1)
    counts = finite[lo:hi].sum(axis=1)
    frames = np.arange(lo, hi)
    ok = counts >= min_anchors
    if not ok.any():
        raise ValueError(f"no frame within +/-{max_search} frames of ref_frame {ref} "
                         f"has >= {min_anchors} tracked anchors")
    order = np.lexsort((np.abs(frames[ok] - ref), -counts[ok]))   # most anchors, then nearest
    return int(frames[ok][order[0]])


def apply_rigid(R: np.ndarray, tr: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.einsum("tij,tj->ti", R, X) + tr


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) indices of True runs."""
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return starts, ends


def fill_short_gaps(y: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    """Linearly fill interior NaN runs of length <= max_gap. Returns (y, n_filled)."""
    y = y.copy()
    bad = ~np.isfinite(y)
    if not bad.any() or max_gap <= 0 or (~bad).sum() < 2:
        return y, 0
    idx = np.arange(len(y))
    interp = np.interp(idx, idx[~bad], y[~bad])
    filled = 0
    for s, e in zip(*_runs(bad)):
        if s > 0 and e < len(y) and (e - s) <= max_gap:
            y[s:e] = interp[s:e]
            filled += e - s
    return y, filled


def lowpass_runs(y: np.ndarray, sos: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero-phase filter each contiguous finite run; runs too short to filter become NaN."""
    out = np.full_like(y, np.nan)
    too_short = 0
    for s, e in zip(*_runs(np.isfinite(y))):
        try:
            out[s:e] = sosfiltfilt(sos, y[s:e])
        except ValueError:          # shorter than the filter's padding length
            too_short += e - s
    return out, too_short


# ---------------------------------------------------------------------------
# main entry points
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, cfg: Config, source: str = "") -> tuple[pd.DataFrame, PreprocessReport]:
    p = cfg.preprocess
    rep = PreprocessReport(source=source)

    tcol = find_time_column(df)
    t_all = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    if not np.isfinite(t_all).all():
        raise ValueError(f"time column {tcol!r} has non-numeric or missing values")
    if (np.diff(t_all) < 0).any():
        raise ValueError(f"time column {tcol!r} is not sorted")
    keep = np.r_[True, np.diff(t_all) > 0]
    rep.duplicate_timestamps_dropped = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)
    rows = np.flatnonzero(keep)
    t = t_all[keep]
    T = len(t)
    if T < 3:
        raise ValueError("need at least 3 frames")
    rep.n_frames = T

    fs = 1.0 / float(np.median(np.diff(t)))
    rep.fs_hz = fs
    lp_hz = float(p.get("lowpass_hz", 10.0))
    cap = p.get("nyquist_cap")
    if lp_hz >= fs / 2 * (cap or 1.0):
        if cap is None:
            raise ValueError(f"lowpass_hz {lp_hz} must be below Nyquist ({fs / 2:.2f} Hz for fs = {fs:.2f} Hz); "
                             "lower it or set preprocess.nyquist_cap")
        lp_hz = float(cap) * fs / 2
    rep.lowpass_hz_effective = lp_hz
    sos = butter(int(p.get("filter_order", 4)), lp_hz, btype="low", fs=fs, output="sos")
    max_gap = int(p.get("max_interp_gap_frames", 5))
    segments = split_segments(t, float(p.get("gap_split_factor", 3.0)))
    rep.segments = segments

    anchors = cfg.resolve(p.get("anchors", "head_anchors"))
    preserve = cfg.resolve(p["preserve_unnormalized"]) if p.get("preserve_unnormalized") else ()

    raw: dict[str, np.ndarray] = {}
    for name in cfg.landmarks:
        arr = read_point(df, name)
        if arr is None:
            rep.missing_landmarks.append(name)
        else:
            raw[name] = arr
    missing_anchors = [a for a in anchors if a not in raw]
    if missing_anchors:
        raise ValueError(f"anchor landmarks missing from input: {missing_anchors}")

    def smooth(arr: np.ndarray) -> np.ndarray:
        out = np.full_like(arr, np.nan)
        for s, e in segments:
            for k in range(3):
                y, nf = fill_short_gaps(arr[s:e, k], max_gap)
                out[s:e, k], ns = lowpass_runs(y, sos)
                rep.samples_interpolated += nf
                rep.samples_too_short_to_filter += ns
        return out

    order = p.get("order", "hn_then_lp")
    base = raw if order == "hn_then_lp" else {n: smooth(a) for n, a in raw.items()}

    P = np.stack([base[a] for a in anchors], axis=1)            # (T, K, 3)
    ref_req = int(p.get("ref_frame", 0))
    if ref_req >= T:
        raise ValueError(f"ref_frame {ref_req} is beyond the last frame ({T - 1})")
    min_anchors = int(p.get("min_anchors", 3))
    finite = np.isfinite(P).all(axis=2)                          # (T, K)
    ref_used = pick_ref_frame(finite, ref_req, int(round(float(p.get("ref_search_s", 2.0)) * fs)), min_anchors)
    rep.ref_frame_requested, rep.ref_frame_used = ref_req, ref_used
    at_ref = finite[ref_used]
    rep.anchors_used = [a for a, ok in zip(anchors, at_ref) if ok]

    R = np.full((T, 3, 3), np.nan)
    tr = np.full((T, 3), np.nan)
    if p.get("anchor_mode", "fixed") == "fixed":
        # one subset (the anchors tracked at the reference) for every frame
        R, tr = kabsch_batch(P[:, at_ref], P[ref_used, at_ref])
    else:
        # adaptive: per frame, every anchor tracked both there and at the reference
        usable = finite & at_ref
        usable[usable.sum(axis=1) < min_anchors] = False
        for mask in np.unique(usable[usable.any(axis=1)], axis=0):
            frames = np.flatnonzero((usable == mask).all(axis=1))
            R[frames], tr[frames] = kabsch_batch(P[frames][:, mask], P[ref_used, mask])
    rep.frames_without_head_pose = int((~np.isfinite(tr).all(axis=1)).sum())

    hn = {n: apply_rigid(R, tr, a) for n, a in base.items()}
    if order == "hn_then_lp":
        hn = {n: smooth(a) for n, a in hn.items()}

    lp = {name: smooth(raw[name]) for name in preserve if name in raw}
    for d in cfg.derived:
        pool = hn if d.space == "hn" else lp
        if not all(ref in pool for ref in d.of):
            rep.missing_landmarks.append(d.name)
            continue
        parts = [pool[ref] for ref in d.of]
        pool[d.name] = np.mean(parts, axis=0) if d.op == "mean" else parts[0] - parts[1]

    seg_id = np.zeros(T, dtype=int)
    for i, (s, e) in enumerate(segments):
        seg_id[s:e] = i
    out: dict[str, np.ndarray] = {"time_s": t, "segment": seg_id, "source_row": rows}
    for name, arr in hn.items():
        for k, a in enumerate(AXES):
            out[column(name, a, "hn")] = arr[:, k]
    for name, arr in lp.items():
        for k, a in enumerate(AXES):
            out[column(name, a, "lp")] = arr[:, k]

    rep.settings = {
        "config": cfg.source,
        "preprocess": dict(p),
        "anchors": list(anchors),
        "landmark_indices": {n: cfg.landmarks[n] for n in raw},
        "derived": [asdict(d) for d in cfg.derived],
    }
    return pd.DataFrame(out), rep


def output_paths(in_path: Path, out_dir: Path) -> tuple[Path, Path]:
    stem = in_path.name[:-4] if in_path.name.lower().endswith(".csv") else in_path.stem
    return out_dir / f"{stem}_lp_hn.csv", out_dir / f"{stem}_lp_hn.meta.json"


def process_file(in_path: str | Path, out_dir: str | Path, cfg: Config) -> PreprocessReport:
    in_path, out_dir = Path(in_path), Path(out_dir)
    out_csv, out_meta = output_paths(in_path, out_dir)
    out, rep = preprocess(pd.read_csv(in_path), cfg, source=in_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    out_meta.write_text(json.dumps(asdict(rep), indent=2, default=str))
    return rep


def _worker(args):
    in_path, out_dir, cfg = args
    try:
        return in_path, process_file(in_path, out_dir, cfg), None
    except Exception as e:  # report per file, keep the batch going
        return in_path, None, f"{type(e).__name__}: {e}"


def process_many(paths, out_dir, cfg: Config, jobs: int = 1):
    """Yield (path, report | None, error | None) for each input file."""
    tasks = [(Path(p), Path(out_dir), cfg) for p in paths]
    if jobs <= 1:
        yield from map(_worker, tasks)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            yield from ex.map(_worker, tasks)
