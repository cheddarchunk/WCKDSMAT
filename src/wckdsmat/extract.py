#!/usr/bin/env python3
"""
GFM VES Batch Extractor — MediaPipe Face Mesh landmark extraction

This script processes a folder of segmented face videos using Google Face Mesh,
outputting time-series CSVs for (a) all 468 landmarks and (b) a compact speech-
specific subset. Optional features include parallel processing, mirrored-label
swapping for selfie recordings, and skipping annotated MP4 outputs for faster
re-runs.

Input:
    Folder containing segmented face videos (e.g., .mp4, .mov, .avi, .mkv)

Outputs (default: ./complexity/vest/):
    ├── gfm_all_t/     → full 468-landmark CSV per video
    ├── gfm_sp_t/      → speech-subset CSV (mirrored labels if --mirror-mode)
    └── gfm_preview/   → optional annotated MP4 preview (unless --skip-mp4)

Example usage
-------------
# Typical run (3 workers, mirrored selfie recordings, CSV-only output)
python ./pipeline/gfm_vesbatch_lm_sp.py --skip-mp4 --workers 4 \
    --input /path/to/videos\
    --out ./data/tracking

# Full run including annotated MP4s mirror mode
python gfm_vesbatch_lm_sp.py --mirror-mode

# Single-threaded run for debugging
python gfm_vesbatch_lm_sp.py --workers 4 --input ./myclips --out./outdir

Flags
-----
--mirror-mode    : Swap right/left *labels* (not coordinates) in speech CSVs.
--skip-mp4       : Skip annotated MP4 writing (faster CSV-only mode).
--workers <int>  : Number of parallel workers (default = CPU_COUNT - 1).
--input <path>   : Input folder of video clips (default = captures/videos/gfm_segments).
--out <path>    : Output base folder (default = complexity/vest).

Notes
-----
- Mirror mode affects *only* the speech-subset CSV; the full 468-point CSV
  remains unchanged.
-Depth (Z) is converted to "pixel-equivalent" units by multiplying the MediaPipe z value by the video width in pixels. This keeps X/Y/Z in a
-consistent unit space (pixels) for downstream steps. Real-world scaling to mm should be applied later (e.g., via eye-distance calibration).
- Parallelization uses ProcessPoolExecutor with the "spawn" start method for
  macOS stability.
- Dependencies: mediapipe, opencv-python, numpy, pandas, tqdm, scipy
"""

import os
import sys
import cv2
import math
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import mediapipe as mp

# ----------------------------
# Configuration and constants *these can be changed here or with input arguments
# ----------------------------
DEFAULT_INPUT_FOLDER = "validation/data/flats/15fps"
DEFAULT_OUTPUT_ROOT = "validation/data/gfm_data/15pfs_021625"
FULLCSV_SUBDIR = "gfm_all_t"
SPCSV_SUBDIR = "gfm_sp_t"
MP4_SUBDIR = "gfm_vid_out"

# --- Speech subset mapping (preserve VES-selected labels) ---
# Convention R=Right, L=Left, H=Head, CL=Corner Lip, M=Middle, UL=Upper Lip, LL=Lower Lip, J=Jaw
# RTH=Right Top Head, RBH=Right Bottom Head, LTH=Left Top Head, LBH=Left Bottom Head, NT=Nose Tip
# RCL=Right Corner Lip, MUL=Middle Upper Lip, LCL=Left Corner Lip, MLL=Middle Lower Lip, RJ=Jaw Right, MJ= Middle Jaw, LJ=Jaw Left
speech_landmark_map = {
    108: 'RTH', 107: 'RBH', 336: 'LTH', 337: 'LBH', 1: 'NT',
    61: 'RCL', 0: 'MUL', 291: 'LCL', 17: 'MLL', 176: 'RJ', 199: 'MJ', 400: 'LJ', 33: 'LCE', 263: 'RCE'
}
# Extract just the indices from the above dictionary for use during landmark selection
speech_indices = list(speech_landmark_map.keys())

# Pairs to swap when --mirror-mode is enabled (speech CSV only)
MIRROR_PAIRS = [("RTH","LTH"),("RBH","LBH"),("RCL","LCL"),("RJ","LJ"),("RCE","LCE")]

mp_face_mesh = mp.solutions.face_mesh


@dataclass
class RunConfig:
    input_folder: str
    output_root: str
    mirror_mode: bool
    skip_mp4: bool


# ----------------------------
# Utility functions
# ----------------------------

def ensure_dirs(path_list: List[str]) -> None:
    for p in path_list:
        os.makedirs(p, exist_ok=True)


def lm_to_xyzn_array(face_landmarks, image_w: int, image_h: int) -> np.ndarray:
    """Convert MediaPipe landmarks to (468, 3) array in pixel-space.
    X,Y are normalized image coords -> pixels via width/height.
    Z is MediaPipe's relative depth -> pixel-equivalent via width.
    """
    lm = face_landmarks.landmark
    xyzn = np.zeros((468, 3), dtype=np.float32)
    for i in range(468):
        xyzn[i, 0] = lm[i].x * image_w
        xyzn[i, 1] = lm[i].y * image_h
        xyzn[i, 2] = lm[i].z * image_w  # pixel-equivalent depth (width basis) 012126
    return xyzn


def pick_speech_points(xyzn: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    out = {}
    for k, idx in SPEECH_IDX.items():
        out[k] = (float(xyzn[idx, 0]), float(xyzn[idx, 1]), float(xyzn[idx, 2]))
    return out


def apply_mirror_label_swap(row: Dict[str, float]) -> Dict[str, float]:
    """Swap R*/L* label values in-place for configured pairs (x,y,z each)."""
    for a, b in MIRROR_PAIRS:
        for comp in ("x", "y", "z"):
            akey, bkey = f"{a}_{comp}", f"{b}_{comp}"
            row[akey], row[bkey] = row[bkey], row[akey]
    return row


def draw_overlay(frame_bgr: np.ndarray, xyzn: np.ndarray) -> None:
    # Draw sparse points for speed: every 8th landmark
    for i in range(0, 468, 8):
        x, y = int(xyzn[i, 0]), int(xyzn[i, 1])
        cv2.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)


# ----------------------------
# Core processing
# ----------------------------
def apply_full_mirror(xyzn: np.ndarray, image_w: int) -> np.ndarray:
    """
    Horizontally mirror Face Mesh coordinates across the vertical image axis.
    Only flips X; Y and Z remain unchanged.
    """
    mirrored = xyzn.copy()
    mirrored[:, 0] = image_w - mirrored[:, 0]  # flip horizontally
    return mirrored

def process_video(video_path: str, cfg: RunConfig) -> None:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    print(f"processing {stem}: start (skip_mp4={cfg.skip_mp4}, mirror={cfg.mirror_mode})")

    # Prepare outputs
    out_all_dir = os.path.join(cfg.output_root, FULLCSV_SUBDIR)
    out_sp_dir = os.path.join(cfg.output_root, SPCSV_SUBDIR)
    out_mp4_dir = os.path.join(cfg.output_root, MP4_SUBDIR)
    # Only create MP4 dir if we're actually writing videos
    if cfg.skip_mp4:
        ensure_dirs([out_all_dir, out_sp_dir])
    else:
        ensure_dirs([out_all_dir, out_sp_dir, out_mp4_dir])

    out_all_csv = os.path.join(out_all_dir, f"{stem}_gfm_all_t.csv")
    out_sp_csv = os.path.join(out_sp_dir, f"{stem}_gfm_sp_t.csv")
    out_mp4 = os.path.join(out_mp4_dir, f"{stem}_gfm_dots.mp4")
    # If skipping MP4, proactively remove any stale file from prior runs
    if cfg.skip_mp4 and os.path.exists(out_mp4):
        try:
            os.remove(out_mp4)
        except Exception:
            pass
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Op! Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not cfg.skip_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_mp4, fourcc, fps, (width, height))

    # Initialize Face Mesh inside the process
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_idx = 0
        rows_all: List[List[float]] = []

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            t = frame_idx / fps

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            if not results.multi_face_landmarks:
                # Still write NaNs to keep time alignment if desired; here we skip rows with no face
                continue

            face_landmarks = results.multi_face_landmarks[0]
            xyzn = lm_to_xyzn_array(face_landmarks, width, height)
            if cfg.mirror_mode:
                xyzn = apply_full_mirror(xyzn, width)

            # --- Full landmarks row ---
            # Flatten as [t, x0,y0,z0, x1,y1,z1, ...]
            flat = [frame_idx, t]
            flat.extend(xyzn.reshape(-1).tolist())
            rows_all.append(flat)

            # --- MP4 overlay ---
            if writer is not None:
                draw_overlay(frame_bgr, xyzn)
                writer.write(frame_bgr)

    cap.release()
    if writer is not None:
        writer.release()
    elif cfg.skip_mp4:
        print(f"Writing CSV-only (skipped MP4): {stem}")

    # Write CSVs
    base_cols = ['frame', 'timestamp']
    lm_cols = [f'lmk_{i}_{axis}' for i in range(468) for axis in ('x','y','z')]
    df_all = pd.DataFrame(rows_all, columns=base_cols + lm_cols)
    df_all.to_csv(out_all_csv, index=False)

    # Build speech subset from df_all using VES-defined indices + labels
    sel = ['timestamp'] + [f'lmk_{i}_{axis}' for i in speech_indices for axis in ('x','y','z')]
    rename = {f'lmk_{i}_{axis}': f'{label}_{axis}' for i, label in speech_landmark_map.items() for axis in ('x','y','z')}
    if cfg.mirror_mode:
        mirror_swap = {
            'RTH':'LTH','LTH':'RTH',
            'RBH':'LBH','LBH':'RBH',
            'RCL':'LCL','LCL':'RCL',
            'RJ':'LJ','LJ':'RJ',
            'RCE':'LCE','LCE':'RCE'
        }
        for k, v in list(rename.items()):
            for r, l in mirror_swap.items():
                if f"{r}_" in v:
                    rename[k] = v.replace(f"{r}_", f"{l}_")
                elif f"{l}_" in v:
                    rename[k] = v.replace(f"{l}_", f"{r}_")

    df_sp = df_all[sel].rename(columns=rename)
    df_sp.to_csv(out_sp_csv, index=False)

    print(f"Done! : {stem}  →  ALL: {out_all_csv} | SP: {out_sp_csv} | MP4: {'skipped' if cfg.skip_mp4 else out_mp4}")


# ----------------------------
# CLI and parallel launcher
# ----------------------------

def discover_videos(input_folder: str) -> List[str]:
    if not os.path.isdir(input_folder):
        return []
    vids = []
    for f in os.listdir(input_folder):
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            vids.append(os.path.join(input_folder, f))
    vids.sort()
    return vids


def main():
    parser = argparse.ArgumentParser(description="GFM batch (parallel + mirror + skip-mp4)")
    parser.add_argument("--input", default=DEFAULT_INPUT_FOLDER, help="Input folder of video segments")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_ROOT, help="Output root folder")
    parser.add_argument("--skip-mp4", action="store_true", help="Skip writing annotated MP4s (CSV-only)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: CPU_COUNT-1)")
    parser.add_argument("--mirror-mode", action="store_true", help="Horizontally flip landmark coordinates (mirrored recordings)." "Applies to full dataset and speech CSV.")

    args = parser.parse_args()

    input_folder = args.input
    output_root = args.out

    vids = discover_videos(input_folder)
    if not vids:
        print(f"Op!! No videos found in: {input_folder}")
        sys.exit(0)

    cpu = os.cpu_count() or 2
    workers = args.workers if (args.workers and args.workers > 0) else max(1, cpu - 1)
    print(f"We Found {len(vids)} video(s) in {input_folder}")
    print(f"We're Using {workers} worker(s). Mirror-mode={args.mirror_mode}, Skip-MP4={args.skip_mp4}\n")

    cfg = RunConfig(
        input_folder=input_folder,
        output_root=output_root,
        mirror_mode=bool(args.mirror_mode),
        skip_mp4=bool(args.skip_mp4),
    )

    if workers == 1:
        for v in tqdm(vids, desc="Processing", unit="vid"):
            process_video(v, cfg)
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(process_video, v, cfg): v for v in vids}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing", unit="vid"):
                v = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"Yikes! Error while processing {os.path.basename(v)}: {e}")

    print("\nNice! All videos processed.")


if __name__ == "__main__":
    main()
