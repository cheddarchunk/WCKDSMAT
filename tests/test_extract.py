"""Extraction tests with a stand-in face mesh (no MediaPipe, no real faces)."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from wckdsmat.config import load_config  # noqa: E402
from wckdsmat.extract import extract_video, find_videos, landmarks_to_pixels  # noqa: E402

W, H, FPS, N = 64, 48, 30.0, 12
MISSING = {4, 5}                       # frames (0-based) where "no face" is detected


def make_video(path, marker_left=True):
    """Black frames with a white block on one side, so the fake mesh can tell if the frame was flipped."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N):
        f = np.zeros((H, W, 3), np.uint8)
        x0 = 2 if marker_left else W - 12
        f[20:30, x0:x0 + 10] = 255
        vw.write(f)
    vw.release()
    return path


class FakeMesh:
    """Returns point i at (x=marker_side, y=i/500, z=-i/1000) in normalized units; frame k encoded in y offset."""

    def __init__(self, n_points=478):
        self.k = -1
        self.n = n_points

    def process(self, rgb):
        self.k += 1
        if self.k in MISSING:
            return SimpleNamespace(multi_face_landmarks=None)
        left = rgb[:, : W // 2].mean() > rgb[:, W // 2:].mean()
        x = 0.25 if left else 0.75
        lms = [SimpleNamespace(x=x, y=i / 500 + self.k / 1000, z=-i / 1000) for i in range(self.n)]
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=lms)])


def run(tmp_path, overrides=None, marker_left=True):
    tmp_path.mkdir(parents=True, exist_ok=True)       # cv2.VideoWriter fails silently on a missing folder
    vid = make_video(tmp_path / "SUBJ001_task.mp4", marker_left)
    cfg = load_config(overrides={"extract": {"write_overlay_video": False, **(overrides or {})}})
    rep = extract_video(vid, tmp_path / "out", cfg, face_mesh_factory=lambda ex: FakeMesh())
    out = tmp_path / "out"
    return rep, pd.read_csv(out / "SUBJ001_task_gfm_all_t.csv"), pd.read_csv(out / "SUBJ001_task_gfm_sp_t.csv")


def test_landmarks_to_pixels_uses_width_for_z():
    lms = [SimpleNamespace(x=0.5, y=0.25, z=-0.1)]
    assert np.allclose(landmarks_to_pixels(lms, 1920, 1080, 1), [[960, 270, -192]])


def test_all_landmarks_file_in_pixels(tmp_path):
    rep, all_df, _ = run(tmp_path)
    assert rep.n_frames == N and rep.n_points == 468
    assert "lmk_467_x" in all_df and "lmk_468_x" not in all_df
    row = all_df.iloc[0]
    assert row["lmk_17_x"] == pytest.approx(0.25 * W)
    assert row["lmk_17_y"] == pytest.approx((17 / 500) * H)
    assert row["lmk_17_z"] == pytest.approx((-17 / 1000) * W)


def test_missing_face_frames_are_nan_rows_not_dropped(tmp_path):
    rep, all_df, sp = run(tmp_path)
    assert len(all_df) == N and rep.frames_without_face == len(MISSING)
    assert list(all_df["face_detected"]) == [0 if k in MISSING else 1 for k in range(N)]
    assert all_df.loc[list(MISSING), "lmk_0_x"].isna().all()
    assert np.allclose(np.diff(all_df["timestamp"]), 1 / FPS)            # uniform time base kept


def test_named_file_uses_config_indices_and_subject_labels(tmp_path):
    _, all_df, sp = run(tmp_path)
    cfg = load_config()
    for name in ("RCE", "LCE", "MLL"):
        idx = cfg.index_of(name)
        assert np.allclose(sp[f"{name}_px_Y"], all_df[f"lmk_{idx}_y"], equal_nan=True)
    assert cfg.index_of("RCE") == 33                                      # subject's right eye


def test_iris_points_when_enabled(tmp_path):
    rep, all_df, _ = run(tmp_path, {"include_iris": True})
    assert rep.n_points == 478 and "lmk_477_x" in all_df


def test_mirror_flips_frame_before_detection(tmp_path):
    _, plain, _ = run(tmp_path / "a", marker_left=True)
    _, mirrored, _ = run(tmp_path / "b", {"mirror": True}, marker_left=True)
    assert plain["lmk_0_x"].dropna().iloc[0] == pytest.approx(0.25 * W)  # marker seen on the left
    assert mirrored["lmk_0_x"].dropna().iloc[0] == pytest.approx(0.75 * W)  # flipped first: seen on the right


def test_meta_and_overlay(tmp_path):
    vid = make_video(tmp_path / "SUBJ001_task.mp4")
    rep = extract_video(vid, tmp_path / "out", load_config(), face_mesh_factory=lambda ex: FakeMesh())
    meta = json.loads((tmp_path / "out" / "SUBJ001_task_gfm.meta.json").read_text())
    assert meta["width"] == W and meta["frames_without_face"] == 2
    assert (tmp_path / "out" / "SUBJ001_task_gfm_dots.mp4").exists()


def test_extract_then_calibrate_reads_new_columns(tmp_path):
    from wckdsmat.calibrate import calibrate
    _, _, sp = run(tmp_path)
    cal, rep = calibrate(sp, load_config(overrides={"calibrate": {"scale_smooth_window_s": 0}}))
    # fake face has both eye corners at the same x -> eye distance from y only; just check it ran
    assert "MLL_t_X" in cal and rep.frames_without_eye_distance == len(MISSING)


def test_find_videos(tmp_path):
    make_video(tmp_path / "a.mp4")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in find_videos([str(tmp_path)])] == ["a.mp4"]
    assert [p.name for p in find_videos([str(tmp_path / "*.mp4")])] == ["a.mp4"]


def test_iris_requires_refine():
    from wckdsmat.config import ConfigError
    with pytest.raises(ConfigError, match="include_iris"):
        load_config(overrides={"extract": {"include_iris": True, "refine_landmarks": False}})


def test_flip_coordinates_reproduces_published_mirror_method(tmp_path):
    rep, all_df, sp = run(tmp_path, {"mirror": True, "mirror_method": "flip_coordinates"}, marker_left=True)
    assert rep.mirror_method == "flip_coordinates"
    assert all_df["lmk_0_x"].dropna().iloc[0] == pytest.approx(W - 0.25 * W)   # detected left, then flipped
    assert (all_df["mirror_indices_swapped"] == 1).all()
    cfg = load_config()
    # named file swaps mirror pairs: RCE reads LCE's index, midline points unchanged
    assert np.allclose(sp["RCE_px_Y"], all_df[f"lmk_{cfg.index_of('LCE')}_y"], equal_nan=True)
    assert np.allclose(sp["MLL_px_Y"], all_df[f"lmk_{cfg.index_of('MLL')}_y"], equal_nan=True)


def test_calibrate_refuses_index_reads_from_flip_coordinates_file(tmp_path):
    from wckdsmat.calibrate import calibrate
    _, all_df, _ = run(tmp_path, {"mirror": True, "mirror_method": "flip_coordinates"})
    with pytest.raises(ValueError, match="opposite side"):
        calibrate(all_df, load_config())


def test_interspeech_preset_uses_published_mirror():
    ex = load_config(preset="interspeech2026").extract
    assert ex["mirror"] is True and ex["mirror_method"] == "flip_coordinates"
