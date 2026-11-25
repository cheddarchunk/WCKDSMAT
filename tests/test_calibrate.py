import numpy as np
import pandas as pd
import pytest
from conftest import FACE, rot_y

from wckdsmat.calibrate import calibrate, process_file
from wckdsmat.config import load_config

EYE_MM = float(np.linalg.norm(np.subtract(FACE["LCE"], FACE["RCE"])))   # 90 mm in the synthetic face
PX_PER_MM = 3.0
NO_SMOOTH = {"scale_smooth_window_s": 0, "scale_clip": []}


def to_pixels(yaw_deg=None, fs=30.0, n=60, names=None):
    """Synthetic Face Mesh output in pixels: weak perspective, MediaPipe z (decreasing toward camera)."""
    yaw = np.zeros(n) if yaw_deg is None else np.asarray(yaw_deg, float)
    cols = {"timestamp": np.arange(n) / fs}
    for name, p in FACE.items():
        if names and name not in names:
            continue
        P = np.array([rot_y(np.deg2rad(a)) @ np.array(p, float) for a in yaw])
        cols[f"{name}_x"] = 960 + PX_PER_MM * P[:, 0]
        cols[f"{name}_y"] = 540 - PX_PER_MM * P[:, 1]
        cols[f"{name}_z"] = -PX_PER_MM * P[:, 2]
    return pd.DataFrame(cols)


def cfg(**cal):
    return load_config(overrides={"calibrate": {**NO_SMOOTH, **cal}})


def dist(out, a, b, axes="XYZ"):
    A = out[[f"{a}_t_{x}" for x in axes]].to_numpy()
    B = out[[f"{b}_t_{x}" for x in axes]].to_numpy()
    return np.linalg.norm(A - B, axis=1)


def test_frontal_face_in_mm():
    out, rep = calibrate(to_pixels(), cfg(), eye_distance_mm=EYE_MM)
    assert rep.units == "mm"
    assert np.allclose(dist(out, "LCE", "RCE"), EYE_MM)
    assert np.allclose(dist(out, "MLL", "MJ"), np.linalg.norm(np.subtract(FACE["MLL"], FACE["MJ"])))


def test_face_units_without_eye_distance():
    out, rep = calibrate(to_pixels(), cfg())
    assert rep.units == "face"
    assert np.allclose(dist(out, "LCE", "RCE"), 1.0)


def test_head_turn_inflates_eye_2d_but_not_eye_3d():
    yaw = np.linspace(0, 30, 60)
    true_lip_jaw = np.linalg.norm(np.subtract(FACE["MLL"], FACE["MJ"]))
    out2, _ = calibrate(to_pixels(yaw), cfg(scale_mode="eye_2d", z_scale="per_frame"), eye_distance_mm=EYE_MM)
    out3, _ = calibrate(to_pixels(yaw), cfg(scale_mode="eye_3d", z_scale="per_frame"), eye_distance_mm=EYE_MM)
    true_vertical = abs(FACE["MLL"][1] - FACE["MJ"][1])                    # yaw leaves vertical spans unchanged
    ratio2 = dist(out2, "MLL", "MJ", "Y")[-1] / true_vertical              # at 30 deg yaw
    assert ratio2 == pytest.approx(1 / np.cos(np.deg2rad(30)), rel=1e-6)   # published method: 15.5% too large
    assert np.allclose(dist(out3, "MLL", "MJ"), true_lip_jaw)              # 3D: exact at every angle


def test_z_direction_and_offset():
    out, rep = calibrate(to_pixels(), cfg(), eye_distance_mm=EYE_MM)
    assert (out["NT_t_Z"] > out["RCE_t_Z"]).all()                          # nose closer to camera = larger z
    anchors = load_config().resolve("head_anchors")
    assert np.isclose(np.mean([out[f"{a}_t_Z"].mean() for a in anchors]), 0.0)


def test_normalized_input_uses_width_for_z():
    px = to_pixels()
    W, H = 1920, 1080
    norm = px.copy()
    for c in norm.columns:
        if c.endswith("_x") or c.endswith("_z"):
            norm[c] = norm[c] / W
        elif c.endswith("_y"):
            norm[c] = norm[c] / H
    a, _ = calibrate(px, cfg(), eye_distance_mm=EYE_MM)
    b, _ = calibrate(norm, cfg(xy_units="normalized01"), eye_distance_mm=EYE_MM, video_size=(W, H))
    cols = [c for c in a.columns if c.endswith(("_X", "_Y", "_Z"))]
    assert np.allclose(a[cols], b[cols])


def test_pixels_labelled_normalized_is_refused():
    with pytest.raises(ValueError, match="already in pixels"):
        calibrate(to_pixels(), cfg(xy_units="normalized01"), video_size=(1920, 1080))


def test_reads_all_landmark_file_by_mediapipe_index():
    named = to_pixels()
    c = load_config()
    by_index = pd.DataFrame({"frame": np.arange(len(named)), "timestamp": named["timestamp"]})
    for name, idx in c.landmarks.items():
        for a in "xyz":
            by_index[f"lmk_{idx}_{a}"] = named[f"{name}_{a}"]
    a, _ = calibrate(named, cfg(), eye_distance_mm=EYE_MM)
    b, _ = calibrate(by_index, cfg(), eye_distance_mm=EYE_MM)
    assert np.allclose(a["MLL_t_Y"], b["MLL_t_Y"]) and "frame" in b


def test_dropout_frame_only_affects_that_frame():
    px = to_pixels()
    px.loc[10, ["LCE_x", "LCE_y", "LCE_z"]] = np.nan
    out, rep = calibrate(px, cfg(), eye_distance_mm=EYE_MM)
    assert rep.frames_without_eye_distance == 1
    assert out["MLL_t_Y"].isna().sum() == 1


def test_smoothing_runs_with_timestamp_column():
    out, rep = calibrate(to_pixels(), load_config(), eye_distance_mm=EYE_MM)   # preset: 0.2 s window
    assert rep.smoothing_window_frames == 7                                    # 0.2 s at 30 fps, odd


def test_process_file_writes_meta(tmp_path):
    src = tmp_path / "SUBJ001_task_gfm_sp_t.csv"
    to_pixels().to_csv(src, index=False)
    rep = process_file(src, tmp_path / "out", cfg(), eye_distance_mm=EYE_MM)
    assert (tmp_path / "out" / "SUBJ001_task_gfm_sp_t_cal.meta.json").exists() and rep.units == "mm"


def test_calibrate_then_preprocess_chain():
    from wckdsmat.preprocess import preprocess
    cal, _ = calibrate(to_pixels(), cfg(), eye_distance_mm=EYE_MM)
    out, rep = preprocess(cal, load_config())
    assert "MLL_hn_Y" in out and rep.missing_landmarks == []


# -- scale center ---------------------------------------------------------------
def jittery_recording(n=150, seed=1):
    """Head still in the image, but eye-corner tracking noise makes the per-frame scale jitter."""
    rng = np.random.default_rng(seed)
    px = to_pixels(n=n)
    for e in ("LCE", "RCE"):
        px[f"{e}_x"] += rng.normal(0, 2.0, n)          # +/- 2 px noise on a 270 px eye distance
    return px


def head_speed(cal, fs=30.0):
    c = np.mean([cal[[f"{p}_t_X", f"{p}_t_Y"]].to_numpy() for p in ("LBH", "LTH", "RBH", "RTH")], axis=0)
    return np.median(np.linalg.norm(np.diff(c, axis=0), axis=1)) * fs


def test_head_centroid_center_removes_scale_jitter_from_head_motion():
    px = jittery_recording()
    origin, _ = calibrate(px, cfg(scale_center="image_origin"), eye_distance_mm=EYE_MM)
    centre, rep = calibrate(px, cfg(scale_center="head_centroid"), eye_distance_mm=EYE_MM)
    assert rep.scale_center == "head_centroid"
    assert head_speed(origin) > 50                       # mm/s of fake motion from scale jitter alone
    assert head_speed(centre) < 0.05 * head_speed(origin)


def test_scale_center_does_not_change_head_normalized_articulators():
    from wckdsmat.preprocess import preprocess
    px = jittery_recording()
    outs = []
    for mode in ("image_origin", "head_centroid"):
        cal, _ = calibrate(px, cfg(scale_center=mode), eye_distance_mm=EYE_MM)
        pre, _ = preprocess(cal, load_config())
        outs.append(pre)
    for name in ("MLL", "IMLL", "RJ", "LCL"):
        cols = [f"{name}_hn_{a}" for a in "XYZ"]
        a, b = outs[0][cols].to_numpy(), outs[1][cols].to_numpy()
        assert np.allclose(a - a[0], b - b[0], atol=1e-8)   # same up to a constant offset


def test_shape_still_scaled_per_frame_with_head_centroid():
    yaw = np.linspace(0, 30, 60)
    out, _ = calibrate(to_pixels(yaw), cfg(scale_mode="eye_3d", z_scale="per_frame"), eye_distance_mm=EYE_MM)
    assert np.allclose(dist(out, "LCE", "RCE"), EYE_MM)
    assert np.allclose(dist(out, "MLL", "MJ"), np.linalg.norm(np.subtract(FACE["MLL"], FACE["MJ"])))


def test_interspeech_preset_keeps_published_scale_center():
    assert load_config(preset="interspeech2026").calibrate["scale_center"] == "image_origin"
