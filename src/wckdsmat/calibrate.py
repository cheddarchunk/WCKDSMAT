# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 WCKDSMAT contributors
"""Stage 2: calibrate Face Mesh coordinates to millimetres (or face units).

Implements the method of Sanchez et al., Interspeech 2026, section 3.3:

1. Convert every axis to isotropic pixels (x*W, y*H, z*W). Input from
   ``wckdsmat.extract`` is already in pixels.
2. Per-frame scale from the outer-canthal distance:
   ``s(t) = eye_distance_mm / d_eye(t)``. ``scale_mode = "eye_2d"`` (the
   published method) uses the image-plane distance. ``"eye_3d"`` also uses
   the depth difference, so the scale does not change when the head turns.
3. Depth scale: ``z_scale = "global_median"`` (published, s_z = median s(t))
   or ``"per_frame"`` (s(t) on all three axes).
4. Optionally flip z so it increases toward the camera, then subtract one
   global z offset (the anchors' mean). A single offset keeps forward/back
   head motion for head-movement analyses; head normalization removes
   per-frame translation either way.

Without an eye distance the output is in face units (eye distance = 1).

Points are read by name (``NAME_x``) or, for all-landmark files, by
MediaPipe index (``lmk_<index>_x``) using the config's landmark map. Output
columns are ``NAME_t_X``. Settings come from ``Config.calibrate``.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import AXES, Config, column
from .preprocess import find_time_column

NORMALIZED_MAX = 1.5    # |x| above this cannot be a 0-1 coordinate


@dataclass
class CalibrateReport:
    source: str = ""
    units: str = "face"                       # "mm" or "face"
    scale_mode: str = "eye_2d"
    z_scale: str = "global_median"
    scale_center: str = "head_centroid"
    eye_distance_mm: float | None = None
    px_per_unit: tuple[float, float, float] = (1.0, 1.0, 1.0)
    median_scale_per_px: float = float("nan")
    scale_p5_p95: tuple[float, float] = (float("nan"), float("nan"))
    z_scale_per_px: float = float("nan")
    z_offset: float = 0.0
    frames_without_eye_distance: int = 0
    smoothing_window_frames: int = 0
    missing_landmarks: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def read_raw_point(df: pd.DataFrame, name: str, index: int) -> np.ndarray | None:
    """(T, 3) uncalibrated coordinates by name (NAME_x / NAME_px_X) or MediaPipe index (lmk_i_x)."""
    lower = {str(c).lower(): c for c in df.columns}
    for pattern in ("{n}_px_{a}", "{n}_{a}", "lmk_{i}_{a}"):
        cols = [lower.get(pattern.format(n=name, i=index, a=a).lower()) for a in AXES]
        if all(cols):
            if pattern.startswith("lmk") and "mirror_indices_swapped" in df.columns \
                    and bool(pd.to_numeric(df["mirror_indices_swapped"]).fillna(0).any()):
                raise ValueError("this all-landmarks file came from mirror_method = 'flip_coordinates', so its "
                                 "indices sit on the opposite side of the face; calibrate the named "
                                 "(*_gfm_sp_t.csv) file instead, or re-extract with mirror_method = 'flip_frame'")
            return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return None


def nan_median_filter(x: np.ndarray, win: int) -> np.ndarray:
    """Centered running median that ignores NaN; window forced odd."""
    if win <= 1 or x.size < 3:
        return x
    win = min(win | 1, x.size if x.size % 2 else x.size - 1)
    xp = np.pad(x, win // 2, mode="edge")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN windows stay NaN
        return np.nanmedian(sliding_window_view(xp, win), axis=-1)


def calibrate(df: pd.DataFrame, cfg: Config, *, eye_distance_mm: float | None = None,
              video_size: tuple[int, int] | None = None, source: str = "") -> tuple[pd.DataFrame, CalibrateReport]:
    c = cfg.calibrate
    rep = CalibrateReport(source=source)
    tcol = find_time_column(df)
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = 1.0 / float(np.median(dt)) if dt.size else float("nan")

    # 1. read points and convert to isotropic pixels
    raw: dict[str, np.ndarray] = {}
    for name, idx in cfg.landmarks.items():
        arr = read_raw_point(df, name, idx)
        if arr is None:
            rep.missing_landmarks.append(name)
        else:
            raw[name] = arr
    eye_a, eye_b = cfg.resolve(c.get("eye_pair", ["RCE", "LCE"]))
    for e in (eye_a, eye_b):
        if e not in raw:
            raise ValueError(f"eye landmark {e} missing from input")

    units = c.get("xy_units", "pixels")
    biggest = max(float(np.nanmax(np.abs(a[:, :2]))) for a in raw.values())
    if units == "normalized01":
        if biggest > NORMALIZED_MAX:
            raise ValueError(f"xy_units = 'normalized01' but coordinates reach {biggest:.0f}; "
                             "this input is already in pixels (wckdsmat extract writes pixels): set xy_units = 'pixels'")
        if not video_size:
            raise ValueError("xy_units = 'normalized01' needs the video size (W, H)")
        W, H = video_size
        ppu = (float(W), float(H), float(W))          # z is relative depth on the width basis
    elif units == "pixels":
        ppu = (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"unknown xy_units {units!r}")
    par = float(c.get("pixel_aspect", 1.0))
    ppu = (ppu[0] * par, ppu[1], ppu[2] * par)
    rep.px_per_unit = ppu
    px = {n: a * np.array(ppu) for n, a in raw.items()}

    # 2. per-frame scale from the eye distance
    mode = c.get("scale_mode", "eye_2d")
    d = px[eye_a] - px[eye_b]
    eye = np.hypot(d[:, 0], d[:, 1]) if mode == "eye_2d" else np.linalg.norm(d, axis=1)
    eye[~(eye > 0)] = np.nan
    win = int(round(float(c.get("scale_smooth_window_s") or 0) * fs)) if np.isfinite(fs) else 0
    if win > 1:
        eye = nan_median_filter(eye, win)
        rep.smoothing_window_frames = win | 1
    clip = c.get("scale_clip")
    if clip:
        med = np.nanmedian(eye)
        eye = np.clip(eye, clip[0] * med, clip[1] * med)
    rep.frames_without_eye_distance = int(np.isnan(eye).sum())

    mm = eye_distance_mm if eye_distance_mm is not None else c.get("eye_distance_mm")
    rep.units = "mm" if mm else "face"
    rep.eye_distance_mm = mm
    s = (mm or 1.0) / eye                                # mm (or face units) per pixel, per frame
    rep.median_scale_per_px = float(np.nanmedian(s))
    rep.scale_p5_p95 = (float(np.nanpercentile(s, 5)), float(np.nanpercentile(s, 95)))

    # 3. depth scale
    z_mode = c.get("z_scale", "global_median")
    sz = np.full_like(s, np.nanmedian(s)) if z_mode == "global_median" else s
    rep.scale_mode, rep.z_scale = mode, z_mode
    rep.z_scale_per_px = float(np.nanmedian(sz))
    S = np.column_stack([s, s, sz])                                   # per-frame scale (T, 3)

    # scale center. image_origin (published): p * s(t), so any change in s(t) also moves the
    # whole face by ds x (its distance from the image corner). head_centroid: the face shape is
    # scaled per frame about the anchors' centroid c(t), and c(t) itself is converted with the
    # median scale, so head translation carries no scale jitter. Head-normalized output is the
    # same either way (the two differ by a per-frame translation, which head normalization removes).
    center_mode = c.get("scale_center", "head_centroid")
    rep.scale_center = center_mode
    if center_mode == "head_centroid":
        anchors = [a for a in cfg.resolve(c.get("anchors", "head_anchors")) if a in px]
        if len(anchors) < 3:
            raise ValueError(f"scale_center = 'head_centroid' needs at least 3 anchors in the input, found {anchors}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ctr = np.nanmean(np.stack([px[a] for a in anchors]), axis=0)          # (T, 3) pixels
        S_ref = np.nanmedian(S, axis=0)
        scaled = {n: (a - ctr) * S + ctr * S_ref for n, a in px.items()}
    else:
        scaled = {n: a * S for n, a in px.items()}

    # 4. z direction and offset
    if c.get("z_direction", "decrease_toward_camera") == "decrease_toward_camera":
        for a in scaled.values():
            a[:, 2] *= -1               # MediaPipe z decreases toward the camera; flip so it increases
    zmode = c.get("z_zero_mode", "anchors_mean")
    if zmode != "none":
        pts = cfg.resolve(c.get("anchors", "head_anchors")) if zmode.startswith("anchors") else tuple(scaled)
        zs = np.concatenate([scaled[p][:, 2] for p in pts if p in scaled])
        stat = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}[zmode.rsplit("_", 1)[-1]]
        rep.z_offset = float(stat(zs))
        for a in scaled.values():
            a[:, 2] -= rep.z_offset

    out = {tcol: t}
    if "frame" in df.columns:
        out["frame"] = df["frame"].to_numpy()
    for n, a in scaled.items():
        for k, ax in enumerate(AXES):
            out[column(n, ax, "t")] = a[:, k]
    rep.settings = {"config": cfg.source, "calibrate": dict(c),
                    "landmark_indices": {n: cfg.landmarks[n] for n in raw}, "video_size": video_size}
    return pd.DataFrame(out), rep


def output_paths(in_path: Path, out_dir: Path) -> tuple[Path, Path]:
    stem = in_path.name[:-4] if in_path.name.lower().endswith(".csv") else in_path.stem
    return out_dir / f"{stem}_cal.csv", out_dir / f"{stem}_cal.meta.json"


def process_file(in_path, out_dir, cfg: Config, *, eye_distance_mm=None, video_size=None) -> CalibrateReport:
    in_path, out_dir = Path(in_path), Path(out_dir)
    out_csv, out_meta = output_paths(in_path, out_dir)
    out, rep = calibrate(pd.read_csv(in_path), cfg, eye_distance_mm=eye_distance_mm,
                         video_size=video_size, source=in_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    out_meta.write_text(json.dumps(asdict(rep), indent=2, default=str))
    return rep
