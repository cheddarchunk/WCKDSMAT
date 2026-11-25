"""Data for the Speech Movement Data Viewer: recordings, audio and spectrograms (no web code).

Recordings can be any stage's output: preprocess (``NAME_hn_X``, ``NAME_lp_X``),
calibrate (``NAME_t_X``), extract (``NAME_px_X``) or legacy ``NAME_x`` files.
Times are shown on the recording's own time base (``time_s`` / ``timestamp``),
so every annotation is in the units the rest of the pipeline uses.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram

from ..preprocess import find_time_column

POINT_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9_]*?)(?:_(?P<space>hn|lp|t|px))?_(?P<axis>[XYZxyz])$")
SPACE_LABELS = {"hn": "head-normalized", "lp": "un-normalized (head movement)", "t": "calibrated",
                "px": "pixels", "": "as recorded"}
STAGE_SUFFIXES = ("_lp_hn", "_cal", "_gfm_sp_t", "_gfm_all_t", "_gfm")
BANDS = {"wide": 0.005, "narrow": 0.030}       # analysis window (s): ~300 Hz vs ~45 Hz bandwidth


@dataclass
class Recording:
    path: Path
    name: str                                   # recording key used in segment tables
    t: np.ndarray
    points: dict[tuple[str, str], np.ndarray]   # (NAME, space) -> (T, 3)
    fs: float
    frame_step: float
    extra_columns: list[str] = field(default_factory=list)

    @property
    def spaces(self) -> list[str]:
        return sorted({s for _, s in self.points}, key=lambda s: ["hn", "lp", "t", "px", ""].index(s))

    def names(self, space: str) -> list[str]:
        return [n for n, s in self.points if s == space]


def recording_name(path: str | Path) -> str:
    """Strip stage suffixes, so ``SUBJ_task_gfm_sp_t_cal_lp_hn.csv`` -> ``SUBJ_task``."""
    stem = Path(path).name
    stem = stem[:-4] if stem.lower().endswith(".csv") else stem
    changed = True
    while changed:
        changed = False
        for suf in STAGE_SUFFIXES:
            if stem.endswith(suf):
                stem, changed = stem[: -len(suf)], True
    return stem


def load_recording(path: str | Path) -> Recording:
    path = Path(path)
    df = pd.read_csv(path)
    tcol = find_time_column(df)
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    found: dict[tuple[str, str], dict[str, str]] = {}
    for c in df.columns:
        m = POINT_RE.match(str(c))
        if m and not str(c).lower().startswith("lmk_"):
            key = (m["name"].upper(), m["space"] or "")
            found.setdefault(key, {})[m["axis"].upper()] = c
    points = {k: df[[ax["X"], ax["Y"], ax["Z"]]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
              for k, ax in found.items() if set(ax) == {"X", "Y", "Z"}}
    if not points:
        raise ValueError(f"{path.name}: no NAME_X/Y/Z point columns found")
    dt = np.diff(t[np.isfinite(t)])
    step = float(np.median(dt[dt > 0])) if (dt > 0).any() else float("nan")
    used = {c for ax in found.values() for c in ax.values()} | {tcol}
    return Recording(path, recording_name(path), t, points, 1.0 / step if step else float("nan"), step,
                     [c for c in df.columns if c not in used])


def auto_audio_offset(rec: Recording) -> float:
    """Seconds to add to audio time to land on the recording's time base.

    ``wckdsmat extract`` (and the earlier scripts) stamp frame k (1-based) at k/fps,
    one frame after its presentation time, so audio from the same video is shifted
    by one frame. Any other time base gets no shift.
    """
    if np.isfinite(rec.frame_step) and len(rec.t) and abs(rec.t[0] - rec.frame_step) < 1e-6:
        return rec.frame_step
    return 0.0


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def extract_audio(video: str | Path, out_wav: str | Path, sr: int = 16000) -> Path:
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), str(out_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not extract audio from {Path(video).name}: {proc.stderr[-500:]}")
    return out_wav


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    return int(sr), x.astype(np.float32, copy=False)


def spectrogram_window(x: np.ndarray, sr: int, t0: float, t1: float, *, band: str = "wide",
                       max_freq: float = 8000.0, max_cols: int = 1200, max_rows: int = 320,
                       dynamic_range_db: float = 70.0, pre_emphasis: float = 0.97):
    """Spectrogram of audio time [t0, t1]: (times_s, freqs_hz, dB image rows=freq).

    ``band``: "wide" (5 ms window, ~300 Hz bandwidth: formants, bursts) or
    "narrow" (30 ms window, ~45 Hz bandwidth: harmonics). The hop adapts so the
    visible window never needs more than ``max_cols`` columns.
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {list(BANDS)}")
    win = BANDS[band]
    nper = max(16, int(round(win * sr)))
    t0, t1 = max(0.0, t0), min(len(x) / sr, t1)
    if t1 - t0 <= win:
        return np.array([]), np.array([]), np.zeros((0, 0))
    a, b = int(max(0, (t0 - win) * sr)), int(min(len(x), (t1 + win) * sr))
    seg = x[a:b].astype(np.float64)
    if pre_emphasis:
        seg = np.append(seg[0], seg[1:] - pre_emphasis * seg[:-1])
    hop = max(1, int(round(max(win / 8, (t1 - t0) / max_cols) * sr)))
    nfft = max(256, 1 << int(np.ceil(np.log2(nper * 2))))
    f, tt, S = spectrogram(seg, fs=sr, window="hann", nperseg=nper, noverlap=max(0, nper - hop),
                           nfft=nfft, detrend=False, scaling="spectrum", mode="magnitude")
    tt = tt + a / sr
    if hop > nper:                               # zoomed out: scipy steps one window at most, so pool columns
        k = int(np.ceil(hop / nper))
        n = (S.shape[1] // k) * k
        S = S[:, :n].reshape(S.shape[0], -1, k).mean(axis=2)
        tt = tt[:n].reshape(-1, k).mean(axis=1)
    keep_t = (tt >= t0) & (tt <= t1)
    keep_f = f <= max_freq
    S, f, tt = S[np.ix_(keep_f, keep_t)], f[keep_f], tt[keep_t]
    if len(f) > max_rows:                       # average neighbouring frequency bins
        k = int(np.ceil(len(f) / max_rows))
        n = (len(f) // k) * k
        S = S[:n].reshape(-1, k, S.shape[1]).mean(axis=1)
        f = f[:n].reshape(-1, k).mean(axis=1)
    db = 20 * np.log10(np.maximum(S, 1e-12))
    top = np.nanmax(db) if db.size else 0.0
    return tt, f, np.clip(db, top - dynamic_range_db, top)
