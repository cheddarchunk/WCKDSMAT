#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method G (dynXY_eyes3D_ratioZ) — aspect-ratio aware dynamic scaling to millimetres with 1D rw input (eye corner distance mm)

Goal: Convert GFM landmark coordinates to mm using a per-frame XY scale from the
eye baseline and a stable global Z scale.

XY (per frame):
  Let LCE=(xL,yL), RCE=(xR,yR). Convert units→pixels with video W×H (and PAR).
  eye_px(t) = sqrt( ((xL-xR)*px_per_unit_x)^2 + ((yL-yR)*px_per_unit_y)^2 )
  mm_per_px(t) = eye_mm / eye_px(t)
  s_x(t) = mm_per_px(t) * px_per_unit_x *compensating for yaw
  s_y(t) = mm_per_px(t) * px_per_unit_y *compensating for yaw
  X_mm(t) = s_x(t) * X_in(t),  Y_mm(t) = s_y(t) * Y_in(t)

Z (global):
  s_z = sqrt( median_t(s_x(t)) * median_t(s_y(t)) )
  Z_mm(t) = s_z * Z_in(t)
  (Optional) Flip Z direction and zero Z using head anchors; optional Z anchoring.

Key updates (aspect-ratio aware XY → pixels → mm):
- **XY scaling is explicitly aspect‑ratio aware.** We never assume square.
- You can supply **--video-size WxH** or a **--video-path** for auto‑detection
  (OpenCV/imageio). If both are provided, `--video-size` wins.
- New **custom px-per-unit** mode if your input XY isn't pixels or [0,1]:
  pass `--xy-units custom --px-per-unit-x ... --px-per-unit-y ...`.
- Optional **--pixel-aspect** (PAR), default 1.0; multiplies X scale (non‑square pixels for weird data).

gfm_cal_final_ves.py — Calibrate FaceMesh (GFM) coordinates into real-world millimetres

Implements **Method G: dynXY_eyes3D_ratioZ**, which applies aspect-ratio-aware scaling.
Per-frame XY scales are derived from the inter-eye baseline; Z uses a stable
global scale tied to the median XY. This produces per-frame real-world 3D
landmark coordinates in millimetres.

───────────────────────────────────────────────────────────────────────────────
USAGE (VEST NOVEMBER 2025 DEMO)
───────────────────────────────────────────────────────────────────────────────
python gfm_cal_final_ves.py \
  --in ./data/SUBJ001_task_gfm_sp_t.csv \
  --out ./data/SUBJ001_task_gfm_sp_t_mm.csv \
  --eye-mm 100.44 \
  --lce LCE --rce RCE \
  --xy-units normalized01 \
  --video-path ./data/SUBJ001_task.mov \
  --anchors LCE,RCE,LTH,RTH,LBH,RBH,NT \
  --z-zero-mode anchors_mean \
  --z-direction decrease_toward_camera \
  --scale-smooth-window-s 0.2 \
  --scale-clip 0.5 2.0 \
  --write-meta

───────────────────────────────────────────────────────────────────────────────
OPTIONS
───────────────────────────────────────────────────────────────────────────────
--in / --out             Input and output CSV paths.
--eye-mm                 True eye-corner distance in millimetres (LCE↔RCE).
--lce / --rce            Landmark stems for left/right eye corners.

XY UNIT SYSTEM
--xy-units {pixels,normalized01,custom}
--video-size WxH         Specify video resolution if XY are normalized.
--video-path PATH        Auto-detect WxH if video is available.
--px-per-unit-x/y        Custom pixels-per-unit for arbitrary coordinate scales.
--pixel-aspect PAR       Pixel aspect ratio (default 1.0).

Z CONTROL
--anchors LIST           Comma-separated anchor stems for Z zeroing.
--z-zero-mode MODE       One of: anchors_mean, anchors_max, anchors_min,
                         global_mean, global_max, global_min, tight_to_camera, none.
--z-direction DIR        increase_toward_camera (default) or decrease_toward_camera.
--z-anchor-stem STEM     Optional single landmark for absolute Z anchoring.
--z-anchor-target-mm MM  Target Z position (mm) for the anchor landmark.

FILTERING
--scale-smooth-window-s  Median-filter window (s) for per-frame eye scale.
--scale-clip LOW HIGH    Clamp per-frame XY scales relative to median.

OTHER
--time-col NAME          Time column name (default: time_s).
--write-meta             Write a *_calib_meta.json file with parameters and scales.

───────────────────────────────────────────────────────────────────────────────
OUTPUTS
───────────────────────────────────────────────────────────────────────────────
1. Calibrated CSV with all *_X/Y/Z columns converted to millimetres.
2. Optional *_calib_meta.json containing method name, video info, px/mm ratios,
   Z-zero mode, anchor shifts, and sample rate estimates.

Example quick-run (normalized input with 1920×1080 video):
    python gfm_cal_final_ves.py \
        --in demo_gfm.csv --out demo_gfm_cal.csv \
        --eye-mm 100.0 --lce LCE --rce RCE \
        --xy-units normalized01 --video-size 1920x1080 \
        --write-meta

"""

import argparse
import json
import math
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------- utils -----------------------------

def ensure_dir_for(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def parse_wxh(s: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", s or "")
    if not m:
        raise ValueError(f"Bad --video-size '{s}' (expected WxH, e.g., 1920x1080).")
    return int(m.group(1)), int(m.group(2))


def get_xyz_cols(df: pd.DataFrame, stem: str) -> Tuple[str, str, str]:
    """Case-robust resolver for STEM_(X|Y|Z)."""
    want = [f"{stem}_X", f"{stem}_Y", f"{stem}_Z"]
    upper = {c.upper(): c for c in df.columns}
    if all(w.upper() in upper for w in want):
        x, y, z = (upper[w.upper()] for w in want)
        return str(x), str(y), str(z)
    # try lowercase
    want = [f"{stem}_x", f"{stem}_y", f"{stem}_z"]
    if all(w in df.columns for w in want):
        return want[0], want[1], want[2]
    sample = [c for c in df.columns if c.upper().startswith(stem.upper())]
    raise KeyError(f"Could not find XYZ for stem '{stem}'. Sample columns: {sample[:8]}")


def compute_sampling_hz(df: pd.DataFrame, time_col: Optional[str]) -> Optional[float]:
    if time_col and time_col in df.columns:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            return float(1.0 / np.median(dt))
    return None


def median_filter(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win | 1)  # ensure odd
    if x.size < win:
        return x
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(x, win)
    med = np.median(sw, axis=-1)
    pad = (win - 1) // 2
    return np.concatenate([np.full(pad, med[0]), med, np.full(pad, med[-1])])

# ---------------- video probing (optional) ----------------

def probe_video_size(path: str) -> Tuple[Optional[int], Optional[int]]:
    """Try OpenCV, then imageio, to get (W,H). Returns (None,None) if probing fails."""
    # OpenCV
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    # imageio v3
    try:
        import imageio.v3 as iio  # type: ignore
        m = iio.immeta(path, exclude_applied=True)
        # common keys vary by container; try a few
        w = m.get("size", [None, None])[0]
        h = m.get("size", [None, None])[1]
        if (w is None or h is None) and "width" in m and "height" in m:
            w, h = m["width"], m["height"]
        if w and h:
            return int(w), int(h)
    except Exception:
        pass
    return None, None

# ------------------------- scaling helpers -----------------------

def compute_px_per_unit_xy(args) -> Tuple[float, float, Optional[int], Optional[int]]:
    """Return (px_per_unit_x, px_per_unit_y, vid_w, vid_h) based on XY units.
    - pixels: (1,1)
    - normalized01: (W,H) requires --video-size or --video-path
    - custom: use --px-per-unit-x/--px-per-unit-y
    """
    vid_w = vid_h = None
    if args.xy_units == "pixels":
        px_u_x = px_u_y = 1.0
    elif args.xy_units == "normalized01":
        if args.video_size:
            vid_w, vid_h = parse_wxh(args.video_size)
        elif args.video_path:
            vid_w, vid_h = probe_video_size(args.video_path)
            if not (vid_w and vid_h):
                raise ValueError("--xy-units normalized01 requires --video-size or a probeable --video-path")
        else:
            raise ValueError("--xy-units normalized01 requires --video-size WxH or --video-path")
        px_u_x, px_u_y = float(vid_w), float(vid_h)
    else:  # custom
        if args.px_per_unit_x is None or args.px_per_unit_y is None:
            raise ValueError("--xy-units custom requires --px-per-unit-x and --px-per-unit-y")
        px_u_x = float(args.px_per_unit_x)
        px_u_y = float(args.px_per_unit_y)
    # pixel aspect ratio (PAR): widen X if pixels are wide (PAR>1)
    par = float(args.pixel_aspect or 1.0)
    px_u_x *= par
    print(f"[xy] units={args.xy_units} px_per_unit=({px_u_x:.6g},{px_u_y:.6g})"
          f"{f' video={vid_w}x{vid_h}' if (vid_w and vid_h) else ''} PAR={par}")
    return px_u_x, px_u_y, vid_w, vid_h

# ------------------------------ main -----------------------------

def main():
    p = argparse.ArgumentParser(description="GFM calibration — Method G (dynXY_eyes3D_ratioZ), aspect-ratio aware")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)

    # XY units and video info
    p.add_argument("--xy-units", choices=["pixels", "normalized01", "custom"], default="pixels",
                   help="Coordinate system of input XY. 'custom' uses px-per-unit flags.")
    p.add_argument("--video-size", type=str, help="Video WxH if --xy-units normalized01 (e.g., 1920x1080)")
    p.add_argument("--video-path", type=str, help="Optional video path for auto W×H probing")
    p.add_argument("--px-per-unit-x", type=float, help="Custom: pixels per input X unit")
    p.add_argument("--px-per-unit-y", type=float, help="Custom: pixels per input Y unit")
    p.add_argument("--pixel-aspect", type=float, default=1.0, help="Pixel aspect ratio (PAR), default square")

    # Eye baseline (for per-frame XY scale)
    p.add_argument("--eye-mm", type=float, required=True, help="True LCE↔RCE distance in mm")
    p.add_argument("--lce", type=str, required=True)
    p.add_argument("--rce", type=str, required=True)

    # Z zero/direction + optional anchor
    p.add_argument("--anchors", type=str, default="LCE,RCE,LTH,RTH,NT",
                   help="Comma stems used for Z zero modes (default common head markers)")
    p.add_argument("--z-zero-mode",
                   choices=["anchors_mean", "global_mean", "anchors_max", "anchors_min",
                            "global_max", "global_min", "tight_to_camera", "none"],
                   default="anchors_mean")
    p.add_argument("--z-direction", choices=["decrease_toward_camera", "increase_toward_camera"],
                   default="increase_toward_camera")
    p.add_argument("--z-anchor-stem", type=str)
    p.add_argument("--z-anchor-target-mm", type=float)

    # Smoothing/clipping for the eye series
    p.add_argument("--scale-smooth-window-s", type=float, default=None,
                   help="Median filter window (seconds) for eye baseline (optional)")
    p.add_argument("--scale-clip", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   help="Clip per-frame eye baseline relative to its median (LOW HIGH)")

    # Meta/time
    p.add_argument("--time-col", type=str, default="time_s")
    p.add_argument("--write-meta", action="store_true")

    args = p.parse_args()

    # Load
    df = pd.read_csv(args.in_path)
    time_col = args.time_col if args.time_col in df.columns else None
    hz = compute_sampling_hz(df, time_col)

    # Resolve stems
    LCE_x, LCE_y, _ = get_xyz_cols(df, args.lce)
    RCE_x, RCE_y, _ = get_xyz_cols(df, args.rce)

    # Pixels-per-unit for XY (aspect-ratio aware)
    px_u_x, px_u_y, vid_w, vid_h = compute_px_per_unit_xy(args)

    # Per-frame 2D eye baseline in pixels
    dx_u = pd.to_numeric(df[LCE_x], errors="coerce").to_numpy() - pd.to_numeric(df[RCE_x], errors="coerce").to_numpy()
    dy_u = pd.to_numeric(df[LCE_y], errors="coerce").to_numpy() - pd.to_numeric(df[RCE_y], errors="coerce").to_numpy()
    eye_px = np.hypot(dx_u * px_u_x, dy_u * px_u_y)

    # Optional smooth/clip
    if args.scale_smooth_window_s and hz:
        win = max(1, int(round(args.scale_smooth_window_s * hz)))
        eye_px = median_filter(eye_px, win)
    if args.scale_clip:
        lo, hi = float(args.scale_clip[0]), float(args.scale_clip[1])
        med = float(np.median(eye_px))
        eye_px = np.clip(eye_px, med * lo, med * hi)

    # mm per pixel (per-frame), then mm per unit for X/Y
    mm_per_px = args.eye_mm / eye_px
    s_x_series = mm_per_px * px_u_x
    s_y_series = mm_per_px * px_u_y

    # Method G: per-frame XY, Z = sqrt(median(sx)*median(sy)) (global)
    sx_med = float(np.median(s_x_series))
    sy_med = float(np.median(s_y_series))
    s_z = math.sqrt(max(1e-12, sx_med * sy_med))
    print(f"[G] median(sx)={sx_med:.6g} mm/u_x  median(sy)={sy_med:.6g} mm/u_y  s_z={s_z:.6g} mm/u_z")

    # Apply scaling to all STEM_X/Y/Z
    out = df.copy()
    # locate all stems robustly
    stems = sorted({re.sub(r"_(?:X|Y|Z)$", "", c, flags=re.IGNORECASE) for c in df.columns if re.search(r"_(?:X|Y|Z)$", str(c), flags=re.IGNORECASE)})

    # broadcast per-frame series across columns
    for stem in stems:
        x, y, z = get_xyz_cols(df, stem)
        out[x] = pd.to_numeric(df[x], errors="coerce").to_numpy() * s_x_series
        out[y] = pd.to_numeric(df[y], errors="coerce").to_numpy() * s_y_series
        out[z] = pd.to_numeric(df[z], errors="coerce").to_numpy() * s_z

    # Z direction
    if args.z_direction == "decrease_toward_camera":
        for c in [c for c in out.columns if str(c).upper().endswith("_Z")]:
            out[c] = -pd.to_numeric(out[c], errors="coerce").to_numpy()

    # Z zeroing
    def collect_z(cols: List[str]) -> np.ndarray:
        return np.concatenate([pd.to_numeric(out[c], errors="coerce").to_numpy() for c in cols]) if cols else np.array([])

    anchor_stems = [s for s in (args.anchors.split(",") if args.anchors else []) if s]
    z0 = 0.0
    if args.z_zero_mode != "none":
        if "anchors" in args.z_zero_mode and anchor_stems:
            zcols = [get_xyz_cols(out, s)[2] for s in anchor_stems if all(k in out.columns for k in get_xyz_cols(out, s))]
        else:
            zcols = [c for c in out.columns if str(c).upper().endswith("_Z")]
        zvals = collect_z(zcols)
        if zvals.size:
            if args.z_zero_mode.endswith("mean"):
                z0 = float(np.mean(zvals))
            elif args.z_zero_mode.endswith("max"):
                z0 = float(np.max(zvals))
            elif args.z_zero_mode.endswith("min"):
                z0 = float(np.min(zvals))
            elif args.z_zero_mode == "tight_to_camera":
                z0 = float(np.min(zvals))
            for c in [c for c in out.columns if str(c).upper().endswith("_Z")]:
                out[c] = pd.to_numeric(out[c], errors="coerce").to_numpy() - z0

    # Optional Z anchor to target
    anchored = False
    z_anchor_shift = 0.0
    if args.z_anchor_stem and args.z_anchor_target_mm is not None:
        _, _, zc = get_xyz_cols(out, args.z_anchor_stem)
        z_med = float(np.median(pd.to_numeric(out[zc], errors="coerce").to_numpy()))
        z_anchor_shift = float(args.z_anchor_target_mm) - z_med
        for c in [c for c in out.columns if str(c).upper().endswith("_Z")]:
            out[c] = pd.to_numeric(out[c], errors="coerce").to_numpy() + z_anchor_shift
        anchored = True

    # Write outputs
    ensure_dir_for(args.out_path)
    out.to_csv(args.out_path, index=False)

    if args.write_meta:
        meta = {
            "input": args.in_path,
            "output": args.out_path,
            "xy_units": args.xy_units,
            "video_size": args.video_size,
            "video_path": args.video_path,
            "px_per_unit_x": float(px_u_x),
            "px_per_unit_y": float(px_u_y),
            "pixel_aspect": float(args.pixel_aspect or 1.0),
            "eye_mm": args.eye_mm,
            "lce": args.lce,
            "rce": args.rce,
            "method": "G_dynXY_eyes3D_ratioZ",
            "median_sx_mm_per_unit": sx_med,
            "median_sy_mm_per_unit": sy_med,
            "sz_mm_per_unit": s_z,
            "z_zero_mode": args.z_zero_mode,
            "z_zero_offset_applied": z0,
            "z_direction": args.z_direction,
            "anchored": anchored,
            "z_anchor_stem": args.z_anchor_stem,
            "z_anchor_target_mm": args.z_anchor_target_mm,
            "z_anchor_shift_mm": z_anchor_shift,
            "scale_smooth_window_s": args.scale_smooth_window_s,
            "scale_clip": args.scale_clip,
            "time_col": time_col,
            "sampling_hz_est": hz,
            "detected_video_w": vid_w,
            "detected_video_h": vid_h,
        }
        meta_path = os.path.splitext(args.out_path)[0] + "_calib_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[meta] {meta_path}")

    print(f"[ok] Wrote {args.out_path} | median(sx)={sx_med:.6g} mm/u_x  median(sy)={sy_med:.6g} mm/u_y  s_z={s_z:.6g} mm/u_z  z0={z0:.6g}")


if __name__ == "__main__":
    main()
