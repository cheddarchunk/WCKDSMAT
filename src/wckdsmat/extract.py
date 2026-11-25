# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 WCKDSMAT contributors
"""Stage 1: MediaPipe Face Mesh landmarks from video, in pixels.

For each video this writes:

* ``<stem>_gfm_all_t.csv``: every Face Mesh point as ``lmk_<i>_x/y/z`` (468, or
  478 with ``extract.include_iris``), so a new landmark set never needs
  re-extraction (``wckdsmat calibrate`` reads these by index).
* ``<stem>_gfm_sp_t.csv``: the config's named landmarks as ``NAME_px_X/Y/Z``.
* ``<stem>_gfm_dots.mp4`` (optional): an overlay video for QC.
* ``<stem>_gfm.meta.json``: video properties, detection counts and settings.

Coordinates: x = lm.x * W, y = lm.y * H, z = lm.z * W (MediaPipe's relative
depth on the width basis, decreasing toward the camera), i.e. isotropic
pixels (Interspeech 2026, section 3.3).

Frames without a detected face are written as NaN rows with ``face_detected
= 0``, so the time base stays uniform and later stages can fill short dropouts.
Mirrored (selfie) recordings, ``extract.mirror = true``:

* ``mirror_method = "flip_frame"`` (default): each frame is flipped *before*
  MediaPipe sees it, so every index is anatomically correct (index 33 is always
  the subject's right eye corner) in both output files.
* ``mirror_method = "flip_coordinates"`` (the published pipeline): detect on the
  mirrored frame, then x -> W - x. MediaPipe's indices then sit on the opposite
  side of the face, so the named file swaps each ``groups.mirror_pairs`` pair.
  The all-landmarks file keeps the raw indices and carries
  ``mirror_indices_swapped = 1``; calibrate refuses to read points by index from
  such a file. The two methods differ by ~1.5 mm SD per landmark on real
  recordings (MediaPipe is not exactly left-right symmetric).

MediaPipe is imported only when a video is processed; ``face_mesh_factory``
lets tests supply a stand-in.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import AXES, N_FACE_POINTS, N_POINTS_WITH_IRIS, Config, column

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


@dataclass
class ExtractReport:
    source: str = ""
    width: int = 0
    height: int = 0
    fps: float = float("nan")
    n_frames: int = 0
    frames_without_face: int = 0
    n_points: int = N_FACE_POINTS
    mirrored: bool = False
    mirror_method: str = ""
    timestamps: str = "nominal"
    mediapipe_version: str = ""
    outputs: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)


def landmarks_to_pixels(landmarks, width: int, height: int, n_points: int) -> np.ndarray:
    """(n_points, 3) isotropic pixels from a MediaPipe landmark list."""
    out = np.empty((n_points, 3))
    for i in range(n_points):
        lm = landmarks[i]
        out[i] = (lm.x * width, lm.y * height, lm.z * width)
    return out


def _default_face_mesh(ex: dict):
    import mediapipe as mp  # heavy; only needed for real videos
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=bool(ex.get("refine_landmarks", True)),
        min_detection_confidence=float(ex.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(ex.get("min_tracking_confidence", 0.5)),
    )


def _mediapipe_version() -> str:
    try:
        import mediapipe as mp
        return mp.__version__
    except Exception:  # noqa: BLE001
        return ""


def output_paths(video: Path, out_dir: Path) -> dict[str, Path]:
    s = video.stem
    return {"all": out_dir / f"{s}_gfm_all_t.csv", "named": out_dir / f"{s}_gfm_sp_t.csv",
            "video": out_dir / f"{s}_gfm_dots.mp4", "meta": out_dir / f"{s}_gfm.meta.json"}


def extract_video(video: str | Path, out_dir: str | Path, cfg: Config,
                  face_mesh_factory: Callable[[dict], object] | None = None) -> ExtractReport:
    import cv2

    video, out_dir = Path(video), Path(out_dir)
    ex = cfg.extract
    rep = ExtractReport(source=video.name, mirrored=bool(ex.get("mirror", False)),
                        timestamps=ex.get("timestamps", "nominal"))
    method = ex.get("mirror_method", "flip_frame")
    rep.mirror_method = method if rep.mirrored else ""
    flip_frame = rep.mirrored and method == "flip_frame"
    flip_coords = rep.mirrored and method == "flip_coordinates"
    n_points = N_POINTS_WITH_IRIS if ex.get("include_iris") else N_FACE_POINTS
    if n_points > N_FACE_POINTS and not ex.get("refine_landmarks", True):
        raise ValueError("include_iris needs refine_landmarks = true")
    rep.n_points = n_points
    paths = output_paths(video, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rep.width, rep.height, rep.fps = W, H, float(fps)

    writer = None
    if ex.get("write_overlay_video", True):
        writer = cv2.VideoWriter(str(paths["video"]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frames, times, found, coords = [], [], [], []
    mesh = (face_mesh_factory or _default_face_mesh)(ex)
    try:
        frame_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_idx += 1                                   # 1-based, as in earlier outputs
            t = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if rep.timestamps == "container"
                 else frame_idx / fps)
            if flip_frame:
                bgr = cv2.flip(bgr, 1)                       # un-mirror before detection
            result = mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            faces = getattr(result, "multi_face_landmarks", None)
            if faces:
                xyz = landmarks_to_pixels(faces[0].landmark, W, H, n_points)
                if flip_coords:
                    xyz[:, 0] = W - xyz[:, 0]                # published method: flip after detection
                found.append(1)
                if writer is not None:
                    for i in range(0, N_FACE_POINTS, 8):
                        cv2.circle(bgr, (int(xyz[i, 0]), int(xyz[i, 1])), 1, (0, 255, 0), -1)
            else:
                xyz = np.full((n_points, 3), np.nan)
                found.append(0)
            if writer is not None:
                writer.write(bgr)
            frames.append(frame_idx)
            times.append(t)
            coords.append(xyz)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if hasattr(mesh, "close"):
            mesh.close()

    C = np.stack(coords) if coords else np.empty((0, n_points, 3))   # (T, n, 3)
    rep.n_frames = len(frames)
    rep.frames_without_face = int(len(found) - sum(found))
    base = {"frame": frames, "timestamp": times, "face_detected": found}

    if ex.get("save_all_landmarks", True):
        all_cols = dict(base)
        if flip_coords:
            all_cols["mirror_indices_swapped"] = 1           # raw indices sit on the opposite side
        for i in range(n_points):
            for k, a in enumerate("xyz"):
                all_cols[f"lmk_{i}_{a}"] = C[:, i, k]
        pd.DataFrame(all_cols).to_csv(paths["all"], index=False)
        rep.outputs["all"] = paths["all"].name

    named = dict(base)
    partner = {}
    if flip_coords:
        for a, b in cfg.mirror_pairs:
            partner[a], partner[b] = b, a
    for name, idx in cfg.landmarks.items():
        idx = cfg.landmarks[partner[name]] if name in partner else idx
        if idx >= n_points:
            continue
        for k, a in enumerate(AXES):
            named[column(name, a, "px")] = C[:, idx, k]
    pd.DataFrame(named).to_csv(paths["named"], index=False)
    rep.outputs["named"] = paths["named"].name
    if writer is not None:
        rep.outputs["video"] = paths["video"].name

    rep.mediapipe_version = _mediapipe_version() if face_mesh_factory is None else "test-double"
    rep.settings = {"config": cfg.source, "extract": dict(ex), "landmark_indices": dict(cfg.landmarks)}
    paths["meta"].write_text(json.dumps(asdict(rep), indent=2, default=str))
    return rep


def find_videos(patterns) -> list[Path]:
    import glob
    found = set()
    for p in patterns:
        path = Path(p)
        if path.is_dir():
            found |= {f for f in path.iterdir() if f.suffix.lower() in VIDEO_SUFFIXES}
        else:
            found |= {Path(f) for f in glob.glob(p) if Path(f).suffix.lower() in VIDEO_SUFFIXES}
    return sorted(found)
