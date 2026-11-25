#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight single-stream segmentation viewer for tracking + optional video/audio.

This version keeps the smooth HTML/CSS overlay playhead and adds lightweight
annotation tags for event marking. Tags are dropped at the current playhead
position, rendered across the combined signal display, listed in a small table,
and can be exported for downstream batch segmentation.

python ./WCKDSMAT/wckdsmat_overlay_tags_speechdetail_batch_video.py \
  --batch-root ./data/lp_hn_om/speech \
  --case-index 0 \
  --video-root /path/to/videos \
  --outdir ./data/kinematics/af/070126 \
  --axis y

python ./WCKDSMAT/wckdsmat_batch_reviewer.py \
  --batch-root ./data/lp_hn_om/speech \
  --video-root /path/to/videos \
  --outdir ./data/kinematics/af/070126\
  --axis y \
  --port 8050
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import mimetypes
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dash
from dash import Input, Output, State, callback_context, dcc, html
from flask import Response, abort, request, send_file
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.io import wavfile
from scipy.signal import spectrogram
import librosa


MAX_TIME_BINS = 4000
MAX_FREQ_BINS = 1600
LOG_FREQ_MIN_HZ = 50.0
LOG_FREQ_MAX_HZ = 8000.0
LOG_BINS_PER_OCTAVE = 24
SPEECH_DETAIL_BINS_PER_OCTAVE = 36
TIME_CANDIDATES = ("time_s", "time", "timestamp", "t", "sec", "seconds")
TRACE_KEYS = ("IMLL", "MLL", "JAW_MID", "LJ", "RJ", "MUL", "RCL", "LCL", "MJ")
TRACE_HINTS = {
    "IMLL": ("imll",),
    "MLL": ("mll",),
    "JAW_MID": ("jaw_mid", "jawmid"),
    "LJ": ("lj",),
    "RJ": ("rj",),
    "MUL": ("mul",),
    "RCL": ("rcl",),
    "LCL": ("lcl",),
    "MJ": ("mj", "menton", "chin"),
}
TRACE_PANEL_FRACTION = 0.58
SPEC_PANEL_FRACTION = 0.42
PLAYHEAD_POLL_MS = 50
CLICK_POLL_MS = 40
F0_HOP_MS = 10.0
F0_FMIN = 60.0
F0_FMAX = 500.0
TAG_STYLE = {
    "burst": {"color": "#c62828", "dash": "solid"},
    "voicing_offset": {"color": "#1565c0", "dash": "dot"},
    "lip_closure": {"color": "#2e7d32", "dash": "dash"},
}


@dataclass(frozen=True)
class SpectrogramData:
    time_s: np.ndarray
    freq_hz: np.ndarray
    db: np.ndarray
    vmin: float
    vmax: float
    sr: int
    mode: str


@dataclass(frozen=True)
class F0Data:
    time_s: np.ndarray
    f0_hz: np.ndarray
    voiced_flag: np.ndarray


@dataclass(frozen=True)
class AppState:
    track_df: pd.DataFrame
    track_time_col: str
    track_t: np.ndarray
    trace_cols: Dict[str, str]
    video_path: Optional[Path]
    audio_wav_path: Optional[Path]
    out_dir: Path
    base_name: str
    track_input_path: Path
    spec_wide: Optional[SpectrogramData]
    spec_narrow: Optional[SpectrogramData]
    spec_speech: Optional[SpectrogramData]
    f0_data: Optional[F0Data]


def find_time_column(df: pd.DataFrame) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in TIME_CANDIDATES:
        if cand in lower:
            return lower[cand]
    for c in df.columns:
        cl = str(c).lower()
        if "time" in cl or cl.endswith("_s"):
            return c
    raise ValueError("Could not find a time column.")


def is_coord_col(colname: str, axis: str) -> bool:
    cl = str(colname).lower()
    axis = str(axis).lower()
    if axis == "3d":
        return cl.endswith("_3d") or cl.endswith("3d")
    if axis in {"x", "y", "z"}:
        return cl.endswith(f"_{axis}") or cl.endswith(axis)
    return False


def _trace_match_score(col: str, hint: str, axis: str) -> tuple[int, int, str]:
    """Prefer exact stem-axis matches over loose substring matches."""
    cl = str(col).lower()
    hint = str(hint).lower()
    axis = str(axis).lower()
    suffix = "_3d" if axis == "3d" else f"_{axis}"
    exact = f"{hint}{suffix}"
    if cl == exact:
        return (0, len(cl), cl)
    if cl.endswith(exact):
        return (1, len(cl), cl)
    if cl.startswith(f"{hint}_") and is_coord_col(cl, axis):
        return (2, len(cl), cl)
    return (3, len(cl), cl)


def find_trace_columns(df: pd.DataFrame, axis: str = "y") -> Dict[str, str]:
    cols_lower = [(c, str(c).lower()) for c in df.columns]
    out: Dict[str, str] = {}
    for key in TRACE_KEYS:
        hints = TRACE_HINTS[key]
        matches = []
        for c, cl in cols_lower:
            if is_coord_col(c, axis) and any(h in cl for h in hints):
                best_hint = next((h for h in hints if h in cl), hints[0])
                matches.append((_trace_match_score(str(c), best_hint, axis), c))
        if matches:
            matches.sort(key=lambda item: item[0])
            out[key] = matches[0][1]
    return out


def extract_audio_wav(video_path: Path, out_wav: Path, sr: int = 16000) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(out_wav)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to extract audio.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
    return out_wav


def load_audio_float32(wav_path: Path) -> Tuple[int, np.ndarray]:
    sr, x = wavfile.read(str(wav_path))
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / max(float(np.iinfo(x.dtype).max), 1.0)
    else:
        x = x.astype(np.float32, copy=False)
    return int(sr), x


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def make_log_freq_grid(freq_hz: np.ndarray, sr: int, bins_per_octave: int = LOG_BINS_PER_OCTAVE) -> np.ndarray:
    f_lo = max(LOG_FREQ_MIN_HZ, float(np.nanmin(freq_hz[freq_hz > 0])))
    f_hi = min(LOG_FREQ_MAX_HZ, 0.5 * float(sr))
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi <= f_lo:
        return np.asarray(freq_hz, dtype=np.float32)
    n_bins = int(np.ceil(float(bins_per_octave) * np.log2(f_hi / f_lo))) + 1
    n_bins = max(192, min(MAX_FREQ_BINS, n_bins))
    return np.geomspace(f_lo, f_hi, n_bins, dtype=np.float32)


def log_bin_spectrogram(freq_hz: np.ndarray, s_db: np.ndarray, log_freq_hz: np.ndarray) -> np.ndarray:
    """Map a linear-frequency spectrogram onto a log-spaced display grid without
    creating artificial low-frequency blur from dense interpolation.

    For each log bin, pool the contributing linear FFT bins with a max operator
    in dB space. That preserves ridge contrast better than linear interpolation
    when the plot is ultimately used as a display, not as an analysis surface.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float32)
    s_db = np.asarray(s_db, dtype=np.float32)
    log_freq_hz = np.asarray(log_freq_hz, dtype=np.float32)

    if len(log_freq_hz) < 2 or len(freq_hz) < 2:
        return s_db.astype(np.float32, copy=False)

    edges = np.empty(len(log_freq_hz) + 1, dtype=np.float32)
    edges[1:-1] = np.sqrt(log_freq_hz[:-1] * log_freq_hz[1:])
    ratio0 = log_freq_hz[1] / max(log_freq_hz[0], 1e-6)
    ratio1 = log_freq_hz[-1] / max(log_freq_hz[-2], 1e-6)
    edges[0] = log_freq_hz[0] / np.sqrt(max(ratio0, 1e-6))
    edges[-1] = log_freq_hz[-1] * np.sqrt(max(ratio1, 1e-6))
    edges[0] = max(edges[0], freq_hz[0])
    edges[-1] = min(edges[-1], freq_hz[-1])

    out = np.empty((len(log_freq_hz), s_db.shape[1]), dtype=np.float32)
    for i in range(len(log_freq_hz)):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (freq_hz >= lo) & (freq_hz < hi if i < len(log_freq_hz) - 1 else freq_hz <= hi)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            k = int(np.argmin(np.abs(freq_hz - log_freq_hz[i])))
            out[i, :] = s_db[k, :]
        elif idx.size == 1:
            out[i, :] = s_db[idx[0], :]
        else:
            out[i, :] = np.nanmax(s_db[idx, :], axis=0)
    return out


def compute_compact_spectrogram(x: np.ndarray, sr: int, mode: str) -> SpectrogramData:
    if mode == "wide":
        win_sec = 0.008
        nperseg = max(128, int(round(win_sec * sr)))
        nfft = max(1024, next_pow2(nperseg * 4))
        bins_per_octave = LOG_BINS_PER_OCTAVE
    elif mode == "speech":
        win_sec = 0.040
        nperseg = max(640, int(round(win_sec * sr)))
        nfft = max(8192, next_pow2(nperseg * 16))
        bins_per_octave = SPEECH_DETAIL_BINS_PER_OCTAVE
    else:
        win_sec = 0.025
        nperseg = max(400, int(round(win_sec * sr)))
        nfft = max(4096, next_pow2(nperseg * 8))
        bins_per_octave = LOG_BINS_PER_OCTAVE
    noverlap = int(round(nperseg * 0.90))

    f, t, sxx = spectrogram(
        x,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    s_db = 20.0 * np.log10(np.maximum(sxx, 1e-10)).astype(np.float32)

    t_step = max(1, int(np.ceil(len(t) / MAX_TIME_BINS)))
    t_ds = np.asarray(t[::t_step], dtype=np.float32)
    s_t = np.asarray(s_db[:, ::t_step], dtype=np.float32)

    keep = np.isfinite(f) & (f > 0.0)
    f_keep = np.asarray(f[keep], dtype=np.float32)
    s_keep = np.asarray(s_t[keep, :], dtype=np.float32)

    log_freq = make_log_freq_grid(f_keep, sr, bins_per_octave=bins_per_octave)
    s_log = log_bin_spectrogram(f_keep, s_keep, log_freq)

    vmin = float(np.nanpercentile(s_log, 1.0))
    vmax = float(np.nanpercentile(s_log, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(s_log))
        vmax = float(np.nanmax(s_log))

    return SpectrogramData(
        time_s=t_ds,
        freq_hz=log_freq.astype(np.float32, copy=False),
        db=s_log.astype(np.float32, copy=False),
        vmin=vmin,
        vmax=vmax,
        sr=sr,
        mode=mode,
    )


def compute_f0_track(x: np.ndarray, sr: int) -> Optional[F0Data]:
    try:
        hop_length = max(1, int(round(sr * (F0_HOP_MS / 1000.0))))
        frame_length = int(2 ** np.ceil(np.log2(max(1024, sr * 0.04))))
        f0, voiced_flag, _ = librosa.pyin(
            x.astype(np.float32, copy=False),
            sr=sr,
            fmin=librosa.note_to_hz('C2') if F0_FMIN <= 65.41 else F0_FMIN,
            fmax=F0_FMAX,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            fill_na=np.nan,
        )
        if f0 is None:
            return None
        t = librosa.times_like(f0, sr=sr, hop_length=hop_length).astype(np.float32)
        f0 = np.asarray(f0, dtype=np.float32)
        voiced_flag = np.asarray(voiced_flag, dtype=bool)
        return F0Data(time_s=t, f0_hz=f0, voiced_flag=voiced_flag)
    except Exception as e:
        print(f'...F0 extraction failed: {e}')
        return None


def prepare_log_spectrogram(spec: SpectrogramData) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(spec.freq_hz, dtype=np.float32), np.asarray(spec.db, dtype=np.float32)


def get_plotly_log_tickvals() -> tuple[list[float], list[str]]:
    vals = [50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 6000, 8000]
    return vals, [str(v) for v in vals]

def range_response(path: Path, mimetype: str):
    if not path.exists():
        abort(404)

    file_size = path.stat().st_size
    range_header = request.headers.get("Range")
    if not range_header:
        return send_file(str(path), mimetype=mimetype, conditional=True)

    try:
        units, rng = range_header.split("=")
        if units.strip() != "bytes":
            return send_file(str(path), mimetype=mimetype, conditional=True)
        start_s, end_s = (rng.split("-") + [""])[:2]
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
    except Exception:
        return send_file(str(path), mimetype=mimetype, conditional=True)

    length = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(length)

    rv = Response(data, 206, mimetype=mimetype, direct_passthrough=True)
    rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    rv.headers["Accept-Ranges"] = "bytes"
    rv.headers["Content-Length"] = str(length)
    return rv


def default_visible_range(track_t: np.ndarray) -> List[float]:
    return [float(np.nanmin(track_t)), float(np.nanmax(track_t))]


def get_trace_options(state: AppState) -> List[dict]:
    if state.trace_cols:
        return [{"label": key, "value": key} for key in TRACE_KEYS if key in state.trace_cols]
    numeric_cols = [c for c in state.track_df.columns if pd.api.types.is_numeric_dtype(state.track_df[c])]
    return [{"label": str(c), "value": str(c)} for c in numeric_cols]


def processed_trace_y(df: pd.DataFrame, col: str, display_flags: set[str]) -> np.ndarray:
    y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
    y_plot = y.copy()

    if "demean" in display_flags:
        mu = np.nanmean(y_plot)
        if np.isfinite(mu):
            y_plot = y_plot - mu

    if "zscore" in display_flags:
        mu = np.nanmean(y_plot)
        sd = np.nanstd(y_plot)
        if np.isfinite(mu) and np.isfinite(sd) and sd > 1e-12:
            y_plot = (y_plot - mu) / sd

    return y_plot


def nearest_track_index(track_t: np.ndarray, t_s: float) -> int:
    return int(np.searchsorted(track_t, np.float32(t_s), side="left").clip(0, len(track_t) - 1))


def make_tag_record(state: AppState, label: str, t_s: float, source: str = "playhead") -> dict:
    idx = nearest_track_index(state.track_t, float(t_s))
    track_time = float(state.track_t[idx])
    return {
        "tag_id": f"{label}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}",
        "label": str(label),
        "time_s": float(t_s),
        "track_frame_idx": idx,
        "track_time_s": track_time,
        "source": str(source),
        "created_iso": dt.datetime.now().isoformat(timespec="milliseconds"),
    }




def apply_shape_relayout_to_tags(state: AppState, tags: List[dict], relayout: Optional[dict]) -> Optional[List[dict]]:
    if not relayout:
        return None

    updated = [dict(tag) for tag in (tags or [])]
    changed = False

    if isinstance(relayout.get("shapes"), list):
        shapes = relayout.get("shapes") or []
        n = min(len(updated), len(shapes))
        for i in range(n):
            shp = shapes[i] or {}
            x0 = shp.get("x0", updated[i].get("time_s"))
            x1 = shp.get("x1", x0)
            try:
                t_s = float(x0 if x0 is not None else x1)
                if x1 is not None:
                    t_s = 0.5 * (t_s + float(x1))
            except Exception:
                continue
            updated[i] = make_tag_record(state, str(updated[i].get("label", "tag")), t_s, source="drag") | {"tag_id": updated[i].get("tag_id", "")}
            changed = True

    pattern = re.compile(r"^shapes\[(\d+)\]\.(x0|x1)$")
    pending = {}
    for key, value in relayout.items():
        m = pattern.match(str(key))
        if not m:
            continue
        i = int(m.group(1))
        axis_key = m.group(2)
        if 0 <= i < len(updated):
            pending.setdefault(i, {})[axis_key] = value

    for i, vals in pending.items():
        try:
            x0 = float(vals.get("x0", updated[i].get("time_s")))
            x1 = float(vals.get("x1", x0))
            t_s = 0.5 * (x0 + x1)
        except Exception:
            continue
        updated[i] = make_tag_record(state, str(updated[i].get("label", "tag")), t_s, source="drag") | {"tag_id": updated[i].get("tag_id", "")}
        changed = True

    return updated if changed else None


def click_time_from_clickdata(click_data: Optional[dict]) -> Optional[float]:
    try:
        points = (click_data or {}).get("points") or []
        if not points:
            return None
        x = points[0].get("x")
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:
        return None

def tags_to_shapes_and_annotations(tags: List[dict]) -> Tuple[List[dict], List[dict]]:
    shapes: List[dict] = []
    annotations: List[dict] = []
    for tag in tags:
        label = str(tag.get("label", "tag"))
        t_s = float(tag.get("time_s", np.nan))
        if not np.isfinite(t_s):
            continue
        style = TAG_STYLE.get(label, {"color": "#444", "dash": "dash"})
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=t_s,
                x1=t_s,
                y0=0.0,
                y1=1.0,
                line=dict(color=style["color"], width=2, dash=style["dash"]),
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=t_s,
                y=1.0,
                xref="x",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                text=label,
                showarrow=False,
                font=dict(size=11, color=style["color"]),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor=style["color"],
                borderwidth=1,
                borderpad=2,
            )
        )
    return shapes, annotations


def build_tag_table(tags: List[dict]) -> html.Div:
    if not tags:
        return html.Div("No tags yet.", style={"fontSize": "12px", "color": "#555"})

    header = html.Tr([
        html.Th("#"),
        html.Th("Label"),
        html.Th("Time (s)"),
        html.Th("Frame"),
        html.Th("Nearest frame time (s)"),
    ])

    rows = []
    for i, tag in enumerate(tags, start=1):
        rows.append(
            html.Tr([
                html.Td(str(i)),
                html.Td(str(tag.get("label", ""))),
                html.Td(f"{float(tag.get('time_s', np.nan)):.4f}"),
                html.Td(str(int(tag.get("track_frame_idx", -1)))),
                html.Td(f"{float(tag.get('track_time_s', np.nan)):.4f}"),
                html.Td(str(tag.get("source", ""))),
            ])
        )

    return html.Div(
        style={"maxHeight": "220px", "overflowY": "auto", "border": "1px solid #ddd"},
        children=[
            html.Table(
                style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"},
                children=[html.Thead(header), html.Tbody(rows)],
            )
        ],
    )


def build_combined_figure(
    state: AppState,
    selected: List[str],
    display_opts: List[str],
    spec: Optional[SpectrogramData],
    visible_range: Optional[List[float]],
    brightness_db: float,
    contrast_scale: float,
    tags: List[dict],
    show_f0: bool,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[TRACE_PANEL_FRACTION, SPEC_PANEL_FRACTION],
    )

    display_flags = set(display_opts or [])

    for key in selected or []:
        col = state.trace_cols.get(key, key)
        y_plot = processed_trace_y(state.track_df, col, display_flags)
        fig.add_trace(
            go.Scatter(
                x=state.track_t,
                y=y_plot,
                mode="lines",
                name=f"{key} ({col})" if key in state.trace_cols else str(col),
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )

    if spec is None:
        fig.add_annotation(
            text="No video/audio provided - spectrogram disabled.",
            x=0.5,
            y=0.18,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    else:
        freq_plot, s_db = prepare_log_spectrogram(spec)
        mean_db = float(np.nanmean(s_db))
        s_adj = (s_db - mean_db) * float(contrast_scale) + mean_db + float(brightness_db)

        vmin = spec.vmin
        vmax = spec.vmax
        rng = max(vmax - vmin, 1e-9)
        s_norm = np.clip((s_adj - vmin) / rng, 0.0, 1.0).astype(np.float32, copy=False)

        tick_vals = np.linspace(0.0, 1.0, 5)
        tick_text = [f"{(vmin + tv * rng):.1f} dB" for tv in tick_vals]

        fig.add_trace(
            go.Heatmap(
                x=spec.time_s,
                y=freq_plot,
                z=s_norm,
                colorscale="Greys",
                zmin=0.0,
                zmax=1.0,
                customdata=np.round(s_adj, 2),
                hovertemplate="Time: %{x:.3f}s<br>Freq: %{y:.0f} Hz<br>Power: %{customdata:.2f} dB<extra></extra>",
                hoverongaps=False,
                colorbar=dict(
                    title=dict(text="dB", side="right"),
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.16,
                    yanchor="top",
                    len=0.75,
                    thickness=14,
                    tickmode="array",
                    tickvals=tick_vals.tolist(),
                    ticktext=tick_text,
                ),
                showscale=True,
                name="Spectrogram",
            ),
            row=2,
            col=1,
        )

        if show_f0 and state.f0_data is not None:
            f0 = state.f0_data
            fig.add_trace(
                go.Scatter(
                    x=f0.time_s,
                    y=f0.f0_hz,
                    mode="lines",
                    name="F0",
                    line=dict(width=2, color="#ff8f00"),
                    hovertemplate="Time: %{x:.3f}s<br>F0: %{y:.1f} Hz<extra></extra>",
                    connectgaps=False,
                ),
                row=2,
                col=1,
            )

    shapes, annotations = tags_to_shapes_and_annotations(tags)

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=10, t=35, b=95),
        legend=dict(orientation="h", y=1.04, x=0),
        uirevision="combined-lock",
        hovermode="x unified",
        clickmode="event",
        shapes=shapes,
        annotations=list(fig.layout.annotations) + annotations,
    )

    fig.update_xaxes(title_text="Time (s)", domain=[0.0, 1.0], row=2, col=1)
    fig.update_yaxes(title_text="Value (processed for display)", row=1, col=1)
    log_tickvals, log_ticktext = get_plotly_log_tickvals()
    fig.update_yaxes(title_text="Frequency (Hz, log)", type="log", tickmode="array", tickvals=log_tickvals, ticktext=log_ticktext, row=2, col=1)

    if visible_range and len(visible_range) == 2:
        fig.update_xaxes(range=visible_range, row=1, col=1)
        fig.update_xaxes(range=visible_range, row=2, col=1)

    return fig



class ReviewerSession:
    """Mutable state wrapper for true batch review.

    The original app was written around one immutable AppState. This proxy keeps
    that API intact: callbacks can still read state.track_df/state.video_path/etc.,
    but those attributes resolve to the currently selected case.
    """
    def __init__(self, args: argparse.Namespace, initial_state: AppState, cases: Optional[List[Path]] = None, initial_index: int = 0):
        self.args = args
        self.cases = list(cases or [])
        self.current_index = int(initial_index)
        self.cache: Dict[int, AppState] = {self.current_index: initial_state}
        self.current = initial_state

    def __getattr__(self, name):
        return getattr(self.current, name)

    @property
    def is_batch(self) -> bool:
        return bool(self.cases)

    @property
    def n_cases(self) -> int:
        return len(self.cases) if self.cases else 1

    def case_relpath(self, index: Optional[int] = None) -> str:
        if not self.cases:
            return self.current.base_name
        idx = self.current_index if index is None else int(index)
        root = Path(self.args.batch_root).expanduser().resolve()
        case_dir = self.cases[idx]
        try:
            rel = case_dir.relative_to(root)
            if rel.name == "time_series":
                rel = rel.parent
            return str(rel)
        except Exception:
            return case_dir.name

    def case_options(self) -> List[dict]:
        if not self.cases:
            return [{"label": self.current.base_name, "value": 0}]
        opts = []
        for i, _case in enumerate(self.cases):
            label = f"[{i+1}/{len(self.cases)}] {self.case_relpath(i)}"
            opts.append({"label": label, "value": i})
        return opts

    def case_status(self) -> str:
        video = str(self.current.video_path) if self.current.video_path else "none"
        audio = str(self.current.audio_wav_path) if self.current.audio_wav_path else "none"
        return (
            f"Case {self.current_index + 1}/{self.n_cases}: {self.case_relpath()}\n"
            f"Tracking: {self.current.track_input_path}\n"
            f"Video: {video}\n"
            f"Audio cache: {audio}"
        )

    def set_index(self, index: int) -> AppState:
        idx = int(index)
        if idx < 0 or idx >= self.n_cases:
            raise IndexError(f"Case index {idx} out of range for {self.n_cases} cases")
        self.current_index = idx
        if idx not in self.cache:
            # Build a fresh AppState for this case, but use the same CLI settings.
            args_i = copy.copy(self.args)
            args_i.case = None
            args_i.case_index = idx
            print(f"\n...Loading batch reviewer case {idx + 1}/{self.n_cases}: {self.case_relpath(idx)}", flush=True)
            self.cache[idx] = build_state(args_i)
        self.current = self.cache[idx]
        return self.current


def resolve_initial_case_index(args: argparse.Namespace, cases: List[Path]) -> int:
    if not cases:
        return 0
    if args.case:
        root = Path(args.batch_root).expanduser().resolve()
        q = str(args.case).lower()
        matches = [i for i, c in enumerate(cases) if q in str(c.relative_to(root)).lower()]
        if not matches:
            preview = "\n".join(f"  [{i}] {c.relative_to(root)}" for i, c in enumerate(cases[:20]))
            raise ValueError(f"No case matched --case {args.case!r}. Available cases include:\n{preview}")
        if len(matches) > 1:
            preview = "\n".join(f"  [{i}] {cases[i].relative_to(root)}" for i in matches[:25])
            raise ValueError(f"--case {args.case!r} matched multiple cases; use --case-index. Matches:\n{preview}")
        return matches[0]
    return int(args.case_index or 0)

def create_app(state: ReviewerSession) -> dash.Dash:
    app = dash.Dash(__name__)
    app.title = "Zoom single-stream segmentation"
    server = app.server

    @server.route("/media/video")
    def media_video():
        if state.video_path is None:
            abort(404)
        mt, _ = mimetypes.guess_type(str(state.video_path))
        return range_response(state.video_path, mt or "video/mp4")

    @server.route("/media/audio")
    def media_audio():
        if state.audio_wav_path is None:
            abort(404)
        mt, _ = mimetypes.guess_type(str(state.audio_wav_path))
        return range_response(state.audio_wav_path, mt or "audio/wav")

    trace_options = get_trace_options(state)
    initial_visible_range = default_visible_range(state.track_t)

    app.layout = html.Div(
        style={"fontFamily": "Courier New, monospace", "padding": "10px"},
        children=[
            html.H3("Zoom batch segmentation reviewer - displacement + video/audio + tags"),
            html.Div(
                style={"border": "1px solid #ddd", "padding": "8px", "marginBottom": "10px", "backgroundColor": "#fafafa"},
                children=[
                    html.Div("Batch case navigation", style={"fontWeight": "bold", "marginBottom": "6px"}),
                    html.Div(
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                        children=[
                            html.Button("Previous case", id="prev-case-btn", n_clicks=0),
                            html.Button("Next case", id="next-case-btn", n_clicks=0),
                            dcc.Dropdown(
                                id="case-select",
                                options=state.case_options(),
                                value=state.current_index,
                                clearable=False,
                                style={"minWidth": "520px", "flex": "1"},
                            ),
                        ],
                    ),
                    html.Div(id="case-status", children=state.case_status(), style={"fontSize": "12px", "whiteSpace": "pre-line", "marginTop": "6px"}),
                    html.Div("Switching cases clears unsaved tags in the browser. Export segments/tags before moving on.", style={"fontSize": "12px", "color": "#8a4b00", "marginTop": "4px"}),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "18px", "alignItems": "flex-start"},
                children=[
                    html.Div(
                        style={"flex": "2"},
                        children=[
                            html.Label("Traces to display"),
                            dcc.Dropdown(
                                id="trace-select",
                                options=trace_options,
                                value=[trace_options[0]["value"]] if trace_options else [],
                                multi=True,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("Display"),
                            dcc.Checklist(
                                id="display-options",
                                options=[
                                    {"label": "Demean", "value": "demean"},
                                    {"label": "Normalize to z-score", "value": "zscore"},
                                ],
                                value=["demean"],
                                style={"display": "flex", "gap": "14px"},
                            ),
                            html.Br(),
                            html.Label("Spectrogram mode"),
                            dcc.RadioItems(
                                id="spec-mode",
                                options=[
                                    {"label": "Wideband (better time resolution)", "value": "wide"},
                                    {"label": "Narrowband (better frequency resolution)", "value": "narrow"},
                                    {"label": "Speech detail (formant-friendly)", "value": "speech"},
                                ],
                                value="speech",
                            ),
                            html.Label("Brightness (dB shift)"),
                            dcc.Slider(
                                id="spec-brightness",
                                min=-40,
                                max=40,
                                step=1,
                                value=0,
                                marks={-40: "-40", -20: "-20", 0: "0", 20: "20", 40: "40"},
                            ),
                            html.Br(),
                            html.Label("Contrast (scale)"),
                            dcc.Slider(
                                id="spec-contrast",
                                min=0.5,
                                max=2.5,
                                step=0.05,
                                value=1.0,
                                marks={0.5: "0.5", 1.0: "1.0", 2.0: "2.0"},
                            ),
                            html.Br(),
                            dcc.Checklist(
                                id="spec-overlays",
                                options=[
                                    {"label": "Overlay F0 track", "value": "show_f0"},
                                ],
                                value=["show_f0"] if state.f0_data is not None else [],
                                style={"display": "flex", "gap": "14px"},
                            ),
                            html.Br(),
                            html.Label("Media offset (seconds) - shifts video/audio relative to tracking"),
                            dcc.Slider(
                                id="media-offset",
                                min=-10.0,
                                max=10.0,
                                step=0.001,
                                value=0.0,
                                marks={-2.0: "-2.0", -0.5: "-0.5", 0.0: "0", 0.5: "0.5", 2.0: "2.0"},
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Segment name"),
                            dcc.Input(id="seg-name", type="text", value="seg1", style={"width": "100%"}),
                            html.Br(),
                            html.Button("Export segment CSV", id="export-btn", n_clicks=0, style={"marginTop": "8px"}),
                            html.Div(id="export-status", style={"marginTop": "8px", "whiteSpace": "pre-line"}),
                            html.Hr(),
                            html.Div(id="media-status", style={"fontSize": "12px", "whiteSpace": "pre-line"}),
                            html.Hr(),
                            html.Div("Tagging"),
                            html.Div(id="playhead-readout", style={"fontSize": "12px", "marginTop": "6px", "whiteSpace": "pre-line"}),
                            html.Label("Selected tag label", style={"marginTop": "8px", "display": "block"}),
                            dcc.RadioItems(
                                id="tag-label-select",
                                options=[
                                    {"label": "burst", "value": "burst"},
                                    {"label": "voicing offset", "value": "voicing_offset"},
                                    {"label": "lip closure", "value": "lip_closure"},
                                ],
                                value="burst",
                                labelStyle={"display": "block", "marginTop": "4px"},
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "8px"},
                                children=[
                                    html.Button("Drop selected at playhead", id="drop-selected-tag-btn", n_clicks=0),
                                    html.Button("Arm click-to-place", id="arm-click-tag-btn", n_clicks=0),
                                    html.Button("Delete last tag", id="delete-last-tag-btn", n_clicks=0),
                                    html.Button("Clear all tags", id="clear-tags-btn", n_clicks=0),
                                    html.Button("Export tags CSV", id="export-tags-btn", n_clicks=0),
                                ],
                            ),
                            html.Div(id="click-mode-status", style={"fontSize": "12px", "marginTop": "8px", "whiteSpace": "pre-line"}),
                            html.Div(id="tag-status", style={"fontSize": "12px", "marginTop": "8px", "whiteSpace": "pre-line"}),
                        ],
                    ),
                ],
            ),
            html.Hr(),
            html.Div(
                id="combined-plot-wrap",
                style={"position": "relative"},
                children=[
                    dcc.Graph(
                        id="combined-plot",
                        figure=go.Figure(),
                        style={"height": "860px"},
                        config={"editable": True, "edits": {"shapePosition": True}, "scrollZoom": True},
                    ),
                    html.Div(
                        id="playhead-overlay",
                        style={
                            "position": "absolute",
                            "display": "none",
                            "pointerEvents": "none",
                            "zIndex": "20",
                            "overflow": "hidden",
                        },
                        children=[
                            html.Div(
                                id="playhead-line",
                                style={
                                    "position": "absolute",
                                    "top": "0px",
                                    "bottom": "0px",
                                    "left": "0px",
                                    "width": "2px",
                                    "backgroundColor": "red",
                                    "transform": "translateX(0px)",
                                    "willChange": "transform",
                                },
                            )
                        ],
                    ),
                ],
            ),
            html.Hr(),
            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "flex-start"},
                children=[
                    html.Div(
                        style={"flex": "2"},
                        children=[
                            html.Div("Video (segment in view)"),
                            html.Video(
                                id="video-player",
                                src="",
                                controls=True,
                                style={"width": "100%", "maxHeight": "360px", "backgroundColor": "#111"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Div("Audio (segment in view)"),
                            html.Audio(id="audio-player", src="", controls=True, style={"width": "100%"}),
                            html.Div("Tip: if playback drifts vs traces, adjust 'Media offset'.", style={"marginTop": "8px", "fontSize": "12px"}),
                        ],
                    ),
                ],
            ),
            html.Hr(),
            html.Div("Tags"),
            html.Div(id="tag-table"),
            dcc.Store(id="visible-range", data=initial_visible_range),
            dcc.Store(id="current-case-index", data=state.current_index),
            dcc.Store(id="tag-store", data=[]),
            dcc.Store(id="tag-click-armed", data=False),
            dcc.Store(id="playhead-track-time", data=None),
            dcc.Store(id="graph-click-store", data=None),
            dcc.Interval(id="playhead-time-interval", interval=PLAYHEAD_POLL_MS, n_intervals=0),
            dcc.Interval(id="graph-click-interval", interval=CLICK_POLL_MS, n_intervals=0),
            html.Div(id="raf-bootstrap", style={"display": "none"}),
        ],
    )


    @app.callback(
        Output("case-select", "value"),
        Input("prev-case-btn", "n_clicks"),
        Input("next-case-btn", "n_clicks"),
        State("case-select", "value"),
        prevent_initial_call=True,
    )
    def navigate_cases(n_prev, n_next, selected_index):
        del n_prev, n_next
        idx = int(selected_index or 0)
        trigger = callback_context.triggered_id
        if trigger == "prev-case-btn":
            return max(0, idx - 1)
        if trigger == "next-case-btn":
            return min(state.n_cases - 1, idx + 1)
        return idx

    @app.callback(
        Output("current-case-index", "data"),
        Output("case-status", "children"),
        Output("trace-select", "options"),
        Output("trace-select", "value"),
        Output("spec-overlays", "value"),
        Output("tag-store", "data", allow_duplicate=True),
        Input("case-select", "value"),
        prevent_initial_call=True,
    )
    def load_selected_case(case_index):
        state.set_index(int(case_index or 0))
        opts = get_trace_options(state)
        default_value = [opts[0]["value"]] if opts else []
        spec_overlay_value = ["show_f0"] if state.f0_data is not None else []
        return state.current_index, state.case_status(), opts, default_value, spec_overlay_value, []

    app.clientside_callback(
        """
        function(n_intervals, media_offset) {
            try {
                var v = document.getElementById('video-player');
                if (!v) {
                    return window.dash_clientside.no_update;
                }
                var ct = Number(v.currentTime);
                if (!isFinite(ct)) {
                    return window.dash_clientside.no_update;
                }
                var off = (media_offset === undefined || media_offset === null) ? 0.0 : Number(media_offset);
                return ct - off;
            } catch (e) {
                return window.dash_clientside.no_update;
            }
        }
        """,
        Output("playhead-track-time", "data"),
        Input("playhead-time-interval", "n_intervals"),
        State("media-offset", "value"),
        prevent_initial_call=False,
    )


    app.clientside_callback(
        """
        function(n_intervals) {
            try {
                window.wckdsmatRAF = window.wckdsmatRAF || {};
                var sync = window.wckdsmatRAF;
                if (!sync.pendingGraphClick) {
                    return window.dash_clientside.no_update;
                }
                var payload = sync.pendingGraphClick;
                sync.pendingGraphClick = null;
                return payload;
            } catch (e) {
                return window.dash_clientside.no_update;
            }
        }
        """,
        Output("graph-click-store", "data"),
        Input("graph-click-interval", "n_intervals"),
        prevent_initial_call=False,
    )

    app.clientside_callback(
        """
        function(fig, media_offset) {
            try {
                if (!fig) {
                    return window.dash_clientside.no_update;
                }

                window.wckdsmatRAF = window.wckdsmatRAF || {};
                var sync = window.wckdsmatRAF;
                sync.wrapId = 'combined-plot-wrap';
                sync.graphId = 'combined-plot';
                sync.videoId = 'video-player';
                sync.overlayId = 'playhead-overlay';
                sync.lineId = 'playhead-line';
                sync.mediaOffset = (media_offset === undefined || media_offset === null) ? 0.0 : Number(media_offset);
                sync.minFrameMs = 8;
                sync.minDeltaPx = 0.25;
                sync.maxDriftS = 0.080;
                sync.running = true;
                sync.lastPx = null;
                sync.lastUpdateMs = sync.lastUpdateMs || 0;
                sync.anchorVideoTime = sync.anchorVideoTime || null;
                sync.anchorPerfMs = sync.anchorPerfMs || null;
                sync.playbackRate = sync.playbackRate || 1.0;
                sync.geometry = sync.geometry || null;

                function graphDiv() {
                    var wrap = document.getElementById(sync.graphId);
                    if (!wrap) return null;
                    return wrap.querySelector('.js-plotly-plot') || wrap;
                }

                function overlayEl() {
                    return document.getElementById(sync.overlayId);
                }

                function lineEl() {
                    return document.getElementById(sync.lineId);
                }

                function setAnchor(video, nowMs) {
                    if (!video) return;
                    var ct = Number(video.currentTime);
                    if (!isFinite(ct)) return;
                    sync.anchorVideoTime = ct;
                    sync.anchorPerfMs = (nowMs === undefined || nowMs === null) ? performance.now() : Number(nowMs);
                    sync.playbackRate = Number(video.playbackRate || 1.0);
                }

                function estimateVideoTime(video, nowMs) {
                    var actual = Number(video.currentTime);
                    if (!isFinite(actual)) return null;

                    if (video.paused || video.ended) {
                        setAnchor(video, nowMs);
                        return actual;
                    }

                    if (sync.anchorVideoTime === null || sync.anchorPerfMs === null) {
                        setAnchor(video, nowMs);
                        return actual;
                    }

                    var elapsed = Math.max(0.0, (Number(nowMs) - Number(sync.anchorPerfMs)) / 1000.0);
                    var predicted = Number(sync.anchorVideoTime) + elapsed * Number(sync.playbackRate || 1.0);

                    if (!isFinite(predicted)) {
                        setAnchor(video, nowMs);
                        return actual;
                    }

                    if (Math.abs(predicted - actual) > Number(sync.maxDriftS || 0.08)) {
                        setAnchor(video, nowMs);
                        return actual;
                    }

                    return predicted;
                }

                function computeGeometry() {
                    var gd = graphDiv();
                    var overlay = overlayEl();
                    if (!gd || !gd._fullLayout || !overlay) return null;

                    var fl = gd._fullLayout;
                    var xa = fl.xaxis;
                    var y1 = fl.yaxis;
                    var y2 = fl.yaxis2;
                    if (!xa || !y1 || !y2) return null;
                    if (![xa._offset, xa._length, y1._offset, y1._length, y2._offset, y2._length].every(function(v) { return typeof v === 'number' && isFinite(v); })) {
                        return null;
                    }

                    var top = Math.min(y1._offset, y2._offset);
                    var bottom = Math.max(y1._offset + y1._length, y2._offset + y2._length);
                    var height = Math.max(0, bottom - top);
                    var left = xa._offset;
                    var width = Math.max(0, xa._length);

                    overlay.style.display = (width > 0 && height > 0) ? 'block' : 'none';
                    overlay.style.left = left + 'px';
                    overlay.style.top = top + 'px';
                    overlay.style.width = width + 'px';
                    overlay.style.height = height + 'px';

                    sync.geometry = {
                        width: width,
                        xMin: Number(xa.range[0]),
                        xMax: Number(xa.range[1])
                    };
                    return sync.geometry;
                }

                function ensureGeometry() {
                    var g = sync.geometry;
                    if (!g || !isFinite(g.width) || g.width <= 0) {
                        g = computeGeometry();
                    }
                    return g;
                }

                function movePlayhead(force) {
                    var video = document.getElementById(sync.videoId);
                    var line = lineEl();
                    if (!video || !line) return;

                    var g = ensureGeometry();
                    if (!g || !isFinite(g.xMin) || !isFinite(g.xMax) || g.xMax <= g.xMin) return;

                    var now = performance.now();
                    var videoTime = estimateVideoTime(video, now);
                    if (!isFinite(videoTime)) return;
                    var x = videoTime - Number(sync.mediaOffset || 0.0);
                    var px = ((x - g.xMin) / (g.xMax - g.xMin)) * g.width;

                    if (!force) {
                        if ((now - sync.lastUpdateMs) < sync.minFrameMs) return;
                        if (sync.lastPx !== null && Math.abs(px - sync.lastPx) < sync.minDeltaPx) return;
                    }

                    sync.lastUpdateMs = now;
                    sync.lastPx = px;
                    line.style.transform = 'translateX(' + px.toFixed(3) + 'px)';
                }

                if (!sync.bound) {
                    sync.bound = true;

                    sync.loop = function() {
                        if (!sync.running) {
                            sync.rafId = null;
                            return;
                        }
                        var video = document.getElementById(sync.videoId);
                        if (video && !video.paused && !video.ended) {
                            movePlayhead(false);
                        }
                        sync.rafId = window.requestAnimationFrame(sync.loop);
                    };

                    sync.bindVideo = function() {
                        var video = document.getElementById(sync.videoId);
                        if (!video || sync.videoEl === video) return;
                        sync.videoEl = video;
                        ['play', 'pause', 'seeking', 'seeked', 'timeupdate', 'loadedmetadata', 'ratechange', 'waiting', 'canplay'].forEach(function(evt) {
                            video.addEventListener(evt, function() {
                                setAnchor(video);
                                movePlayhead(true);
                            }, {passive: true});
                        });

                        if (typeof video.requestVideoFrameCallback === 'function' && !sync.rvfcBound) {
                            sync.rvfcBound = true;
                            sync.onVideoFrame = function(now, metadata) {
                                if (!sync.videoEl) return;
                                sync.anchorVideoTime = Number(metadata && metadata.mediaTime);
                                sync.anchorPerfMs = Number(now);
                                sync.playbackRate = Number(sync.videoEl.playbackRate || 1.0);
                                movePlayhead(false);
                                if (sync.videoEl && typeof sync.videoEl.requestVideoFrameCallback === 'function') {
                                    sync.videoEl.requestVideoFrameCallback(sync.onVideoFrame);
                                }
                            };
                            video.requestVideoFrameCallback(sync.onVideoFrame);
                        }
                    };

                    sync.bindGraph = function() {
                        var gd = graphDiv();
                        if (!gd || sync.graphEl === gd) return;
                        sync.graphEl = gd;
                        sync.geometry = null;

                        gd.on('plotly_afterplot', function() {
                            sync.geometry = null;
                            computeGeometry();
                            movePlayhead(true);
                        });
                        gd.on('plotly_relayout', function() {
                            sync.geometry = null;
                            computeGeometry();
                            movePlayhead(true);
                        });
                        gd.on('plotly_redraw', function() {
                            sync.geometry = null;
                            computeGeometry();
                            movePlayhead(true);
                        });

                        if (!sync.graphClickBound) {
                            sync.graphClickBound = true;
                            gd.addEventListener('click', function(evt) {
                                var overlay = overlayEl();
                                var g = ensureGeometry();
                                if (!overlay || !g) return;
                                if (evt.target && (evt.target.closest('.modebar') || evt.target.closest('.draglayer') === null && evt.target.closest('.nsewdrag') === null && evt.target.closest('.plot') === null)) return;
                                var xPx = evt.clientX - overlay.getBoundingClientRect().left;
                                var yPx = evt.clientY - overlay.getBoundingClientRect().top;
                                if (!isFinite(xPx) || !isFinite(yPx)) return;
                                if (xPx < 0 || xPx > Number(g.width) || yPx < 0 || yPx > Number(overlay.getBoundingClientRect().height)) return;
                                var frac = xPx / Math.max(1.0, Number(g.width));
                                var x = Number(g.xMin) + frac * (Number(g.xMax) - Number(g.xMin));
                                if (!isFinite(x)) return;
                                sync.pendingGraphClick = {x: x, ts: Date.now()};
                            }, {passive: true, capture: true});
                        }
                    };

                    sync.onResize = function() {
                        sync.geometry = null;
                        computeGeometry();
                        movePlayhead(true);
                    };
                    window.addEventListener('resize', sync.onResize, {passive: true});
                }

                sync.bindVideo();
                sync.bindGraph();
                computeGeometry();
                var initialVideo = document.getElementById(sync.videoId);
                if (initialVideo) { setAnchor(initialVideo); }
                movePlayhead(true);
                if (!sync.rafId) {
                    sync.rafId = window.requestAnimationFrame(sync.loop);
                }
                return window.dash_clientside.no_update;
            } catch (e) {
                return window.dash_clientside.no_update;
            }
        }
        """,
        Output("raf-bootstrap", "children"),
        Input("combined-plot", "figure"),
        Input("media-offset", "value"),
        prevent_initial_call=False,
    )

    @app.callback(
        Output("combined-plot", "figure"),
        Input("trace-select", "value"),
        Input("display-options", "value"),
        Input("visible-range", "data"),
        Input("spec-mode", "value"),
        Input("spec-brightness", "value"),
        Input("spec-contrast", "value"),
        Input("spec-overlays", "value"),
        Input("tag-store", "data"),
        Input("current-case-index", "data"),
    )
    def update_combined_plot(selected, display_opts, visible_range, spec_mode, brightness_db, contrast_scale, spec_overlays, tags, case_index):
        state.set_index(int(case_index or 0))
        if spec_mode == "narrow":
            spec = state.spec_narrow
        elif spec_mode == "speech":
            spec = state.spec_speech
        else:
            spec = state.spec_wide
        return build_combined_figure(
            state=state,
            selected=selected or [],
            display_opts=display_opts or [],
            spec=spec,
            visible_range=visible_range,
            brightness_db=brightness_db or 0.0,
            contrast_scale=contrast_scale or 1.0,
            tags=tags or [],
            show_f0=("show_f0" in (spec_overlays or [])),
        )

    @app.callback(
        Output("visible-range", "data"),
        Input("combined-plot", "relayoutData"),
        Input("current-case-index", "data"),
        State("visible-range", "data"),
    )
    def capture_visible_range(relayout, case_index, current_range):
        state.set_index(int(case_index or 0))
        if callback_context.triggered_id == "current-case-index":
            return default_visible_range(state.track_t)
        if relayout is None:
            return current_range
        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")
        if x0 is not None and x1 is not None:
            return [float(x0), float(x1)]
        if relayout.get("xaxis.autorange"):
            return default_visible_range(state.track_t)
        return current_range

    @app.callback(
        Output("video-player", "src"),
        Output("audio-player", "src"),
        Output("media-status", "children"),
        Input("visible-range", "data"),
        Input("media-offset", "value"),
        Input("current-case-index", "data"),
    )
    def update_media_sources(visible_range, media_offset, case_index):
        state.set_index(int(case_index or 0))
        shift = float(media_offset or 0.0)
        t0, t1 = map(float, visible_range or default_visible_range(state.track_t))
        if t1 <= t0:
            return "", "", "Invalid visible range."
        mt0 = max(0.0, t0 + shift)
        mt1 = max(mt0, t1 + shift)
        if state.video_path is None:
            return "", "", "No video provided."
        v_src = f"/media/video#t={mt0:.3f},{mt1:.3f}"
        a_src = f"/media/audio#t={mt0:.3f},{mt1:.3f}" if state.audio_wav_path is not None else ""
        status = (
            f"Visible segment (tracking): {t0:.3f}-{t1:.3f} s (dur {t1 - t0:.3f}s)\n"
            f"Media segment (with offset {shift:+.3f}s): {mt0:.3f}-{mt1:.3f} s"
        )
        return v_src, a_src, status

    @app.callback(
        Output("playhead-readout", "children"),
        Input("playhead-track-time", "data"),
        Input("current-case-index", "data"),
    )
    def update_playhead_readout(playhead_t, case_index):
        state.set_index(int(case_index or 0))
        if playhead_t is None or not np.isfinite(playhead_t):
            return "Playhead: unavailable"
        idx = nearest_track_index(state.track_t, float(playhead_t))
        return (
            f"Playhead time: {float(playhead_t):.4f} s\n"
            f"Nearest tracking frame: {idx} @ {float(state.track_t[idx]):.4f} s"
        )

    @app.callback(
        Output("tag-click-armed", "data"),
        Output("click-mode-status", "children"),
        Input("arm-click-tag-btn", "n_clicks"),
        Input("graph-click-store", "data"),
        State("tag-click-armed", "data"),
        State("tag-label-select", "value"),
        prevent_initial_call=True,
    )
    def manage_click_arm(n_arm, graph_click, armed, selected_label):
        del n_arm
        trigger = callback_context.triggered_id
        if trigger == "arm-click-tag-btn":
            armed = not bool(armed)
        elif trigger == "graph-click-store" and bool(armed) and graph_click and graph_click.get("x") is not None:
            armed = False

        status = (
            f"Click-to-place armed for: {selected_label}. Click in either panel to drop a tag.\nExisting tags can be dragged horizontally to refine their timestamps."
            if armed else
            "Click-to-place idle. To add a tag by click, arm it first. Existing tags can be dragged horizontally."
        )
        return armed, status

    @app.callback(
        Output("tag-store", "data"),
        Output("tag-status", "children"),
        Input("drop-selected-tag-btn", "n_clicks"),
        Input("delete-last-tag-btn", "n_clicks"),
        Input("clear-tags-btn", "n_clicks"),
        Input("graph-click-store", "data"),
        Input("combined-plot", "relayoutData"),
        State("playhead-track-time", "data"),
        State("tag-store", "data"),
        State("tag-label-select", "value"),
        State("tag-click-armed", "data"),
        State("current-case-index", "data"),
        prevent_initial_call=True,
    )
    def manage_tags(n_drop, n_delete, n_clear, graph_click, relayout_data, playhead_t, tags, selected_label, click_armed, case_index):
        state.set_index(int(case_index or 0))
        del n_drop, n_delete, n_clear
        tags = list(tags or [])
        trigger = callback_context.triggered_id
        selected_label = str(selected_label or "burst")

        if trigger == "clear-tags-btn":
            return [], "Cleared all tags."

        if trigger == "delete-last-tag-btn":
            if not tags:
                return tags, "No tags to delete."
            removed = tags.pop()
            return tags, f"Deleted last tag: {removed.get('label', '')} at {float(removed.get('time_s', np.nan)):.4f} s"

        if trigger == "drop-selected-tag-btn":
            if playhead_t is None or not np.isfinite(playhead_t):
                return tags, "Playhead time is unavailable."
            tag = make_tag_record(state, selected_label, float(playhead_t), source="playhead")
            tags.append(tag)
            return tags, f"Added {selected_label} at {tag['time_s']:.4f} s (frame {tag['track_frame_idx']})."

        if trigger in {"combined-plot", "graph-click-store"}:
            updated = apply_shape_relayout_to_tags(state, tags, relayout_data)
            if updated is not None:
                return updated, "Adjusted tag position by dragging."

            click_t = None
            if graph_click and graph_click.get("x") is not None:
                try:
                    click_t = float(graph_click.get("x"))
                except Exception:
                    click_t = None
            if bool(click_armed) and click_t is not None and np.isfinite(click_t):
                tag = make_tag_record(state, selected_label, float(click_t), source="click")
                tags.append(tag)
                return tags, f"Added {selected_label} at {tag['time_s']:.4f} s by click (frame {tag['track_frame_idx']})."

        return dash.no_update, dash.no_update

    @app.callback(
        Output("tag-table", "children"),
        Input("tag-store", "data"),
    )
    def update_tag_table(tags):
        return build_tag_table(tags or [])

    @app.callback(
        Output("export-status", "children"),
        Input("export-btn", "n_clicks"),
        State("visible-range", "data"),
        State("seg-name", "value"),
        State("tag-store", "data"),
        State("current-case-index", "data"),
        prevent_initial_call=True,
    )
    def export_segment(n_clicks, visible_range, segname, tags, case_index):
        del n_clicks
        state.set_index(int(case_index or 0))
        if not segname or not segname.strip():
            return "Please enter a segment name."

        t0, t1 = map(float, visible_range or default_visible_range(state.track_t))
        if t1 <= t0:
            return "Invalid visible range (t1 <= t0). Zoom the region you want to segment."

        safe_seg = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in segname.strip())
        out_csv = state.out_dir / f"{state.base_name}_{safe_seg}_track.csv"
        log_path = state.out_dir / "segmentation_log.csv"
        tags_csv = state.out_dir / f"{state.base_name}_{safe_seg}_tags.csv"

        tser = pd.to_numeric(state.track_df[state.track_time_col], errors="coerce")
        seg = state.track_df.loc[(tser >= t0) & (tser <= t1)].copy()
        if seg.empty:
            return f"Empty segment for {t0:.3f}-{t1:.3f}s. Check your zoom window."

        t0_raw = float(pd.to_numeric(seg[state.track_time_col], errors="coerce").iloc[0])
        seg[state.track_time_col] = pd.to_numeric(seg[state.track_time_col], errors="coerce").astype(float) - t0_raw

        state.out_dir.mkdir(parents=True, exist_ok=True)
        seg.to_csv(out_csv, index=False)

        tag_lines = []
        tags_in_window = []
        for tag in list(tags or []):
            t_tag = float(tag.get("time_s", np.nan))
            if np.isfinite(t_tag) and t0 <= t_tag <= t1:
                rel_t = t_tag - t0
                rec = dict(tag)
                rec["segment_name"] = safe_seg
                rec["window_t0_s"] = t0
                rec["window_t1_s"] = t1
                rec["time_rel_segment_s"] = rel_t
                tags_in_window.append(rec)
                tag_lines.append(f"    - {rec['label']} @ {t_tag:.4f} s (segment-relative {rel_t:.4f} s)")

        if tags_in_window:
            pd.DataFrame(tags_in_window).to_csv(tags_csv, index=False)

        rec = pd.DataFrame([
            dict(
                timestamp_iso=dt.datetime.now().isoformat(timespec="seconds"),
                base_name=state.base_name,
                segment_name=safe_seg,
                track_input=str(state.track_input_path),
                track_output=str(out_csv),
                tags_output=str(tags_csv) if tags_in_window else "",
                window_t0_s=t0,
                window_t1_s=t1,
                window_duration_s=float(t1 - t0),
                n_rows=int(len(seg)),
                n_tags_window=int(len(tags_in_window)),
                time_col=str(state.track_time_col),
                trace_cols=json.dumps(state.trace_cols),
                video_input=str(state.video_path) if state.video_path else "",
                audio_wav=str(state.audio_wav_path) if state.audio_wav_path else "",
            )
        ])
        if log_path.exists():
            rec.to_csv(log_path, mode="a", header=False, index=False)
        else:
            rec.to_csv(log_path, mode="w", header=True, index=False)

        tag_block = "\n".join(tag_lines) if tag_lines else "    - none in current visible window"
        tag_file_line = f"\n  Tag file:\n    - {tags_csv}" if tags_in_window else ""

        return (
            f"Exported segment '{segname}' (time normalized to t=0)\n"
            f"  Rows: {len(seg)}\n"
            f"  Window: {t0:.3f}-{t1:.3f} s\n"
            f"  File:\n    - {out_csv}{tag_file_line}\n"
            f"  Tags in window:\n{tag_block}\n"
            f"  Log:\n    - {log_path}"
        )

    @app.callback(
        Output("tag-status", "children", allow_duplicate=True),
        Input("export-tags-btn", "n_clicks"),
        State("tag-store", "data"),
        State("seg-name", "value"),
        State("current-case-index", "data"),
        prevent_initial_call=True,
    )
    def export_tags(n_clicks, tags, segname, case_index):
        del n_clicks
        state.set_index(int(case_index or 0))
        tags = list(tags or [])
        if not tags:
            return "No tags to export."

        safe_seg = (segname or "seg1").strip() or "seg1"
        safe_seg = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in safe_seg)
        out_csv = state.out_dir / f"{state.base_name}_{safe_seg}_manual_tags.csv"
        state.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tags).to_csv(out_csv, index=False)
        return f"Exported {len(tags)} tags to\n{out_csv}"

    return app



# -----------------------------------------------------------------------------
# Batch-output loading helpers for batch_gfm_imll_timeseries*.py
# -----------------------------------------------------------------------------
def normalize_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names so displacement files can be loaded case-insensitively."""
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


def sanitize_name(value: str) -> str:
    return "_".join(part for part in re.split(r"[^A-Za-z0-9_-]+", str(value)) if part) or "case"


def find_batch_time_series_dirs(root: Path) -> List[Path]:
    """Find case directories created by batch_gfm_imll_timeseries*.py.

    Expected layout:
      <root>/<relative_input_stem>/time_series/gfm_displacement.csv
      <root>/<relative_input_stem>/time_series/imll_displacement.csv
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Batch output root not found: {root}")

    dirs = set()
    for path in root.rglob("*_displacement.csv"):
        if path.name.lower() not in {"gfm_displacement.csv", "imll_displacement.csv"}:
            continue
        if path.parent.name == "time_series":
            dirs.add(path.parent)
        else:
            # tolerate a flat legacy case folder
            dirs.add(path.parent)
    return sorted(dirs)


def resolve_batch_case(root: Path, case_query: Optional[str], case_index: int) -> Path:
    cases = find_batch_time_series_dirs(root)
    if not cases:
        raise FileNotFoundError(
            "No batch displacement outputs found. Expected time_series/gfm_displacement.csv "
            "and/or time_series/imll_displacement.csv under the batch root."
        )

    root = Path(root).expanduser().resolve()
    if case_query:
        q = str(case_query).lower()
        matches = [c for c in cases if q in str(c.relative_to(root)).lower()]
        if not matches:
            preview = "\n".join(f"  [{i}] {c.relative_to(root)}" for i, c in enumerate(cases[:20]))
            raise ValueError(f"No case matched --case {case_query!r}. Available cases include:\n{preview}")
        if len(matches) > 1:
            preview = "\n".join(f"  [{cases.index(c)}] {c.relative_to(root)}" for c in matches[:25])
            raise ValueError(f"--case {case_query!r} matched multiple cases; use --case-index. Matches:\n{preview}")
        return matches[0]

    if case_index < 0 or case_index >= len(cases):
        preview = "\n".join(f"  [{i}] {c.relative_to(root)}" for i, c in enumerate(cases[:30]))
        raise IndexError(f"--case-index {case_index} out of range for {len(cases)} cases. Available cases include:\n{preview}")
    return cases[case_index]


def load_batch_displacement_case(case_dir: Path, prefer: str = "combined") -> tuple[pd.DataFrame, Path, str]:
    """Load one batch-output case as a single tracking dataframe.

    In combined mode, this merges gfm_displacement.csv with imll_displacement.csv
    on time_s and keeps all unique columns. This gives the segmenter access to
    raw lip/jaw stems and derived imll/jaw_mid traces at the same time.
    """
    case_dir = Path(case_dir).expanduser().resolve()
    ts_dir = case_dir / "time_series" if (case_dir / "time_series").exists() else case_dir
    gfm_path = ts_dir / "gfm_displacement.csv"
    imll_path = ts_dir / "imll_displacement.csv"

    prefer = str(prefer or "combined").lower()
    paths = {"gfm": gfm_path, "imll": imll_path}

    if prefer in {"gfm", "imll"}:
        path = paths[prefer]
        if not path.exists():
            raise FileNotFoundError(f"Requested {prefer} displacement file does not exist: {path}")
        return normalize_columns_case_insensitive(pd.read_csv(path)), path, prefer

    frames = []
    source_paths = []
    for path in (gfm_path, imll_path):
        if path.exists():
            df = normalize_columns_case_insensitive(pd.read_csv(path))
            if "time_s" not in df.columns:
                raise ValueError(f"Missing time_s in {path}")
            frames.append(df)
            source_paths.append(path)

    if not frames:
        raise FileNotFoundError(f"No displacement files found in {ts_dir}")

    merged = frames[0].copy()
    for df in frames[1:]:
        add_cols = [c for c in df.columns if c == "time_s" or c not in merged.columns]
        merged = merged.merge(df[add_cols], on="time_s", how="outer", sort=True)

    merged = merged.sort_values("time_s").reset_index(drop=True)
    label = "+".join(p.stem for p in source_paths)
    return merged, source_paths[0], label



# -----------------------------------------------------------------------------
# Batch video discovery/matching helpers
# -----------------------------------------------------------------------------
DEFAULT_VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def normalize_match_key(value: str) -> str:
    """Normalize names for loose track-video matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def path_match_keys(base_name: str, track_path: Path) -> List[str]:
    """Build candidate keys from the selected batch case and source path."""
    keys: List[str] = []
    for raw in [base_name, track_path.stem, track_path.parent.name, track_path.parent.parent.name]:
        key = normalize_match_key(raw)
        if key and key not in keys and key not in {"timeseries", "gfmdisplacement", "imlldisplacement"}:
            keys.append(key)

    # Also include progressively larger path fragments above time_series. This helps when
    # the batch output preserves nested input folders such as subject/task/session.
    parts = [part for part in track_path.parts if part]
    if "time_series" in parts:
        i = parts.index("time_series")
        for width in range(1, min(5, i) + 1):
            frag = "_".join(parts[max(0, i - width):i])
            key = normalize_match_key(frag)
            if key and key not in keys:
                keys.append(key)
    return keys


def find_video_files(video_root: Path, exts: Sequence[str], pattern: str = "*") -> List[Path]:
    video_root = Path(video_root).expanduser().resolve()
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")
    extset = {e.lower() if str(e).startswith(".") else f".{str(e).lower()}" for e in exts}
    return sorted(
        p for p in video_root.rglob(pattern)
        if p.is_file() and p.suffix.lower() in extset
    )


def score_video_match(video_path: Path, keys: Sequence[str]) -> int:
    stem_key = normalize_match_key(video_path.stem)
    path_key = normalize_match_key(str(video_path))
    best = 0
    for key in keys:
        if not key:
            continue
        if stem_key == key:
            best = max(best, 10000 + len(key))
        elif key in stem_key:
            best = max(best, 5000 + len(key))
        elif stem_key in key and len(stem_key) >= 6:
            best = max(best, 3000 + len(stem_key))
        elif key in path_key:
            best = max(best, 1000 + len(key))

        # Token-overlap fallback for cases where names share subject/date/task tokens
        # but not a full contiguous substring.
        tokens = [t for t in re.split(r"[^a-z0-9]+", key) if len(t) >= 3]
        if tokens:
            overlap = sum(1 for t in tokens if t in stem_key)
            best = max(best, overlap * 100)
    return best


def resolve_video_path(args: argparse.Namespace, base_name: str, track_path: Path) -> Optional[Path]:
    """Resolve a single video for the selected case.

    Priority:
      1. --video explicit file path
      2. --video-root auto-match using the selected batch case / tracking path
      3. no video
    """
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return video_path

    if not args.video_root:
        return None

    candidates = find_video_files(
        Path(args.video_root),
        exts=args.video_exts or DEFAULT_VIDEO_EXTS,
        pattern=args.video_pattern or "*",
    )
    if not candidates:
        raise FileNotFoundError(
            f"No video files found under {Path(args.video_root).expanduser().resolve()} "
            f"with pattern {args.video_pattern!r} and extensions {args.video_exts}."
        )

    keys = path_match_keys(base_name, track_path)
    scored = sorted(
        [(score_video_match(path, keys), path) for path in candidates],
        key=lambda item: (-item[0], str(item[1])),
    )
    best_score, best_path = scored[0]

    if best_score <= 0:
        preview = "\n".join(f"  - {p}" for p in candidates[:20])
        raise ValueError(
            "Could not auto-match a video for this case.\n"
            f"Case keys: {keys}\n"
            f"Video candidates include:\n{preview}\n"
            "Use --video for a manual override, or rename/link videos so the subject/task/session tokens match."
        )

    if len(scored) > 1 and scored[1][0] == best_score and not args.allow_ambiguous_video_match:
        preview = "\n".join(f"  score={score:5d}  {path}" for score, path in scored[:10])
        raise ValueError(
            "Ambiguous video match for this case. Top matches tied.\n"
            f"Case keys: {keys}\n"
            f"Top candidates:\n{preview}\n"
            "Use --video to pick one manually, or pass --allow-ambiguous-video-match to use the first sorted match."
        )

    print(f"...Auto-matched video: {best_path}")
    print(f"    match keys: {keys}")
    return best_path

def resolve_track_dataframe(args: argparse.Namespace) -> tuple[pd.DataFrame, Path, str]:
    """Resolve either --track or --batch-root into a dataframe + display base name."""
    if args.batch_root:
        batch_root = Path(args.batch_root).expanduser().resolve()
        case_dir = resolve_batch_case(batch_root, args.case, int(args.case_index))
        track_df, source_path, source_label = load_batch_displacement_case(case_dir, prefer=args.batch_prefer)
        try:
            case_rel = case_dir.relative_to(batch_root)
            if case_rel.name == "time_series":
                case_rel = case_rel.parent
        except Exception:
            case_rel = Path(case_dir.name)
        base_name = sanitize_name(str(case_rel))
        print(f"...Selected batch case: {case_rel}")
        print(f"...Loaded displacement source: {source_label}")
        return track_df, source_path, base_name

    if not args.track:
        raise ValueError("Provide either --track <csv-or-case-dir> or --batch-root <batch_output_root>.")

    track_path = Path(args.track).expanduser().resolve()
    if not track_path.exists():
        raise FileNotFoundError(f"Tracking file or case directory not found: {track_path}")

    if track_path.is_dir():
        track_df, source_path, source_label = load_batch_displacement_case(track_path, prefer=args.batch_prefer)
        base_name = sanitize_name(track_path.name if track_path.name != "time_series" else track_path.parent.name)
        print(f"...Loaded displacement case directory: {track_path} ({source_label})")
        return track_df, source_path, base_name

    print(f"...Loading tracking: {track_path}")
    return pd.read_csv(track_path), track_path, track_path.stem

def build_state(args: argparse.Namespace) -> AppState:
    out_dir = Path(args.outdir).expanduser().resolve()

    track_df, track_path, base_name = resolve_track_dataframe(args)
    if bool(args.case_insensitive_columns):
        track_df = normalize_columns_case_insensitive(track_df)

    track_time_col = find_time_column(track_df)
    track_t = pd.to_numeric(track_df[track_time_col], errors="coerce").to_numpy(dtype=np.float32)
    trace_cols = find_trace_columns(track_df, axis=args.axis)

    print(f"    time column: {track_time_col}")
    if trace_cols:
        print("    detected traces:")
        for key, value in trace_cols.items():
            print(f"      {key}: {value}")
    else:
        print("    detected traces: (none) - dropdown will show numeric columns as fallback")

    video_path: Optional[Path] = None
    audio_wav_path: Optional[Path] = None
    spec_wide: Optional[SpectrogramData] = None
    spec_narrow: Optional[SpectrogramData] = None
    spec_speech: Optional[SpectrogramData] = None
    f0_data: Optional[F0Data] = None

    video_path = resolve_video_path(args, base_name=base_name, track_path=track_path)
    if video_path is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_wav_path = (out_dir / f"{base_name}_audio.wav").resolve()
        if not audio_wav_path.exists():
            print(f"...Extracting audio via ffmpeg -> {audio_wav_path}")
            extract_audio_wav(video_path, audio_wav_path, sr=int(args.audio_sr))
        else:
            print(f"...Audio already exists: {audio_wav_path}")

        print("...Computing compact spectrograms (pre-downsampled)")
        sr, x = load_audio_float32(audio_wav_path)
        spec_wide = compute_compact_spectrogram(x, sr, mode="wide")
        spec_narrow = compute_compact_spectrogram(x, sr, mode="narrow")
        spec_speech = compute_compact_spectrogram(x, sr, mode="speech")
        print("...Estimating F0 track")
        f0_data = compute_f0_track(x, sr)
        print(f"    wide spectrogram:   {len(spec_wide.freq_hz)} freqs x {len(spec_wide.time_s)} frames @ sr={spec_wide.sr}Hz")
        print(f"    narrow spectrogram: {len(spec_narrow.freq_hz)} freqs x {len(spec_narrow.time_s)} frames @ sr={spec_narrow.sr}Hz")
        print(f"    speech spectrogram: {len(spec_speech.freq_hz)} freqs x {len(spec_speech.time_s)} frames @ sr={spec_speech.sr}Hz")
        print(f"    wide display range:   {spec_wide.vmin:.1f} to {spec_wide.vmax:.1f} dB")
        print(f"    narrow display range: {spec_narrow.vmin:.1f} to {spec_narrow.vmax:.1f} dB")
        print(f"    speech display range: {spec_speech.vmin:.1f} to {spec_speech.vmax:.1f} dB")
        if f0_data is not None:
            voiced_n = int(np.sum(np.isfinite(f0_data.f0_hz)))
            print(f"    F0 track: {len(f0_data.time_s)} frames, {voiced_n} voiced")

    return AppState(
        track_df=track_df,
        track_time_col=track_time_col,
        track_t=track_t,
        trace_cols=trace_cols,
        video_path=video_path,
        audio_wav_path=audio_wav_path,
        out_dir=out_dir,
        base_name=base_name,
        track_input_path=track_path,
        spec_wide=spec_wide,
        spec_narrow=spec_narrow,
        spec_speech=spec_speech,
        f0_data=f0_data,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zoom single-stream segmentation viewer")
    parser.add_argument("--track", default=None, help="Tracking CSV file, or one batch-output case directory/time_series directory. Optional when --batch-root is used.")
    parser.add_argument("--video", default=None, help="Zoom video file (mp4/mov). Optional manual override.")
    parser.add_argument("--video-root", default=None, help="Root directory containing videos. When provided, the selected batch case is auto-matched to a video file.")
    parser.add_argument("--video-pattern", default="*", help="Recursive glob pattern used under --video-root. Default: *")
    parser.add_argument("--video-exts", nargs="+", default=list(DEFAULT_VIDEO_EXTS), help="Video extensions to consider under --video-root. Default: .mp4 .mov .m4v .avi .mkv")
    parser.add_argument("--allow-ambiguous-video-match", action="store_true", help="If multiple videos tie as the best match, use the first sorted candidate instead of failing.")
    parser.add_argument("--outdir", default="./segments", help="Output directory for segmented CSVs")
    parser.add_argument("--axis", default="y", choices=["x", "y", "z", "3d"], help="Coordinate axis/magnitude to auto-detect")
    parser.add_argument("--batch-root", default=None, help="Root output directory from batch_gfm_imll_timeseries*.py. Finds <case>/time_series/*_displacement.csv.")
    parser.add_argument("--case", default=None, help="Substring of a batch-output case path to load. Use with --batch-root.")
    parser.add_argument("--case-index", default=0, type=int, help="Case index to load from --batch-root when --case is omitted.")
    parser.add_argument("--batch-prefer", default="combined", choices=["combined", "gfm", "imll"], help="Which displacement file(s) to load from a batch-output case.")
    parser.add_argument("--case-insensitive-columns", action="store_true", default=True, help="Normalize loaded tracking columns to lowercase before trace detection.")
    parser.add_argument("--audio_sr", default=16000, type=int, help="Audio sample rate for extraction (Hz)")
    parser.add_argument("--port", default=8050, type=int, help="Dash server port")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug/reloader")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_root:
        batch_root = Path(args.batch_root).expanduser().resolve()
        cases = find_batch_time_series_dirs(batch_root)
        if not cases:
            raise FileNotFoundError(f"No batch cases found under {batch_root}")
        initial_index = resolve_initial_case_index(args, cases)
        if initial_index < 0 or initial_index >= len(cases):
            raise IndexError(f"--case-index {initial_index} out of range for {len(cases)} cases")
        args_i = copy.copy(args)
        args_i.case = None
        args_i.case_index = initial_index
        initial_state = build_state(args_i)
        state = ReviewerSession(args=args, initial_state=initial_state, cases=cases, initial_index=initial_index)
        print(f"...Batch reviewer ready: {len(cases)} cases", flush=True)
    else:
        initial_state = build_state(args)
        state = ReviewerSession(args=args, initial_state=initial_state, cases=[], initial_index=0)

    app = create_app(state)
    app.run(debug=bool(args.debug), port=int(args.port))


if __name__ == "__main__":
    main()
