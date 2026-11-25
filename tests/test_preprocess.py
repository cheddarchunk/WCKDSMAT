import json

import numpy as np
import pytest
from conftest import make_recording

from wckdsmat.config import load_config, preset_text
from wckdsmat.preprocess import fill_short_gaps, kabsch_batch, pick_ref_frame, preprocess, process_file

EDGE = 15   # frames trimmed from each end where filter edge effects live


def hn(out, name):
    return out[[f"{name}_hn_X", f"{name}_hn_Y", f"{name}_hn_Z"]].to_numpy()


def interior(a):
    return a[EDGE:-EDGE]


# -- core geometry ------------------------------------------------------------
def test_kabsch_recovers_known_rotation():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(5, 3)) * 40
    from conftest import rot_x, rot_y
    R_true = rot_y(0.4) @ rot_x(-0.2)
    P = (Q @ R_true.T + [5, -3, 9])[None]          # observed = R Q + t
    R, t = kabsch_batch(P, Q)
    assert np.allclose(R[0] @ P[0].T + t[0][:, None], Q.T, atol=1e-9)
    assert np.isclose(np.linalg.det(R[0]), 1.0)     # proper rotation, no reflection


def test_head_normalization_removes_head_motion(cfg, recording):
    df, truth = recording
    out, rep = preprocess(df, cfg)
    for a in cfg.resolve("head_anchors"):               # anchors become stationary
        assert np.abs(interior(hn(out, a)) - interior(truth[a])).max() < 0.02
    err = np.abs(interior(hn(out, "MLL")) - interior(truth["MLL"])).max()
    assert err < 0.05, f"lower-lip motion not recovered (max err {err:.3f} mm)"
    assert rep.frames_without_head_pose == 0


def test_lp_then_hn_order_also_recovers_motion(recording):
    df, truth = recording
    out, _ = preprocess(df, load_config(overrides={"preprocess": {"order": "lp_then_hn"}}))
    assert np.abs(interior(hn(out, "MLL")) - interior(truth["MLL"])).max() < 0.1


def test_derived_points(cfg, recording):
    df, _ = recording
    out, _ = preprocess(df, cfg)
    jaw_mid = (hn(out, "LJ") + hn(out, "RJ")) / 2
    assert np.allclose(hn(out, "JAW_MID"), jaw_mid)
    assert np.allclose(hn(out, "IMLL"), hn(out, "MLL") - jaw_mid)


def test_preserved_head_motion_is_not_normalized(cfg, recording):
    df, truth = recording
    out, _ = preprocess(df, cfg)
    lp = out[["LBH_lp_X", "LBH_lp_Y", "LBH_lp_Z"]].to_numpy()
    raw = df[["LBH_x", "LBH_y", "LBH_z"]].to_numpy()
    assert np.abs(interior(lp) - interior(raw)).max() < 0.1       # follows the observed head
    assert np.ptp(interior(lp)[:, 0]) > 5                          # i.e. still moving


# -- references and anchors ---------------------------------------------------
def test_ref_frame_sets_the_head_pose(recording):
    df, _ = recording
    out, rep = preprocess(df, load_config(overrides={"preprocess": {"ref_frame": 15}}))
    assert rep.ref_frame_used == 15


def test_pick_ref_frame_prefers_most_anchors_then_nearest():
    finite = np.ones((20, 4), bool)
    finite[10, 3] = False                              # requested frame lacks one anchor
    assert pick_ref_frame(finite, 10, 3, 3) in (9, 11)
    assert pick_ref_frame(finite, 10, 0, 3) == 10        # no search: accept 3 anchors
    with pytest.raises(ValueError):
        pick_ref_frame(finite, 10, 0, 4)


def test_adaptive_mode_survives_a_lost_anchor(recording):
    df, truth = recording
    df = df.copy()
    df.loc[100:140, ["NT_x", "NT_y", "NT_z"]] = np.nan     # nose tip lost for 41 frames
    fixed, rep_f = preprocess(df, load_config())
    adapt, rep_a = preprocess(df, load_config(overrides={"preprocess": {"anchor_mode": "adaptive"}}))
    assert rep_f.frames_without_head_pose == 41
    assert rep_a.frames_without_head_pose == 0
    seg = slice(105, 136)
    assert np.abs(hn(adapt, "MLL")[seg] - truth["MLL"][seg]).max() < 0.1


def test_missing_anchor_column_is_an_error(cfg, recording):
    df, _ = recording
    with pytest.raises(ValueError, match="anchor landmarks missing"):
        preprocess(df.drop(columns=["NT_x", "NT_y", "NT_z"]), cfg)


def test_missing_non_anchor_is_reported_not_fatal(cfg, recording):
    df, _ = recording
    out, rep = preprocess(df.drop(columns=["MJ_x", "MJ_y", "MJ_z"]), cfg)
    assert "MJ" in rep.missing_landmarks and "MJ_hn_X" not in out


# -- dropouts and gaps --------------------------------------------------------
def test_short_dropout_filled_long_dropout_stays_nan_without_spreading(cfg, recording):
    df, truth = recording
    df = df.copy()
    df.loc[50:52, ["MLL_x", "MLL_y", "MLL_z"]] = np.nan     # 3 frames: filled
    df.loc[150:170, ["MLL_x", "MLL_y", "MLL_z"]] = np.nan   # 21 frames: stays NaN
    out, rep = preprocess(df, cfg)
    m = hn(out, "MLL")
    assert np.isfinite(m[50:53]).all()
    assert np.isnan(m[150:171]).all()
    assert np.isfinite(m[:150]).all() and np.isfinite(m[171:]).all()   # rest of channel intact
    assert rep.samples_interpolated == 9


def test_fill_short_gaps_leaves_edges_alone():
    y = np.array([np.nan, 1, np.nan, 3, np.nan])
    filled, n = fill_short_gaps(y, 5)
    assert n == 1 and filled[2] == 2 and np.isnan(filled[0]) and np.isnan(filled[-1])


def test_recording_gap_splits_segments_and_stops_smearing(cfg):
    fs = 30.0
    t = np.r_[np.arange(150), np.arange(150) + 450] / fs     # 10 s cut after frame 149
    df, truth = make_recording(t=t, step=(150, "MLL", 20.0))
    out, rep = preprocess(df, cfg)
    assert rep.segments == [(0, 150), (150, 300)]
    assert set(out["segment"]) == {0, 1}
    near_cut = slice(140, 160)
    err_split = np.abs(hn(out, "MLL")[near_cut] - truth["MLL"][near_cut]).max()

    unsplit, _ = preprocess(df, load_config(overrides={"preprocess": {"gap_split_factor": 1e9}}))
    err_unsplit = np.abs(hn(unsplit, "MLL")[near_cut] - truth["MLL"][near_cut]).max()
    assert err_split < 1.0 < err_unsplit, (err_split, err_unsplit)


def test_duplicate_timestamps_dropped(cfg, recording):
    df, _ = recording
    df = df.copy()
    df.loc[10, "timestamp"] = df.loc[9, "timestamp"]
    out, rep = preprocess(df, cfg)
    assert rep.duplicate_timestamps_dropped == 1 and len(out) == len(df) - 1
    assert 10 not in set(out["source_row"])


# -- filter limits ------------------------------------------------------------
def test_cutoff_clamped_at_low_frame_rate():
    df, _ = make_recording(fs=15.0)
    out, rep = preprocess(df, load_config())
    assert rep.lowpass_hz_effective == pytest.approx(0.95 * 7.5)


def test_cutoff_above_nyquist_without_cap_is_an_error(tmp_path):
    p = tmp_path / "c.toml"   # speech preset without nyquist_cap
    p.write_text('preset = "none"\n' + preset_text("speech").replace("nyquist_cap = 0.95", ""))
    df, _ = make_recording(fs=15.0)
    with pytest.raises(ValueError, match="below Nyquist"):
        preprocess(df, load_config(p))


# -- file I/O -----------------------------------------------------------------
def test_process_file_writes_csv_and_meta(cfg, recording, tmp_path):
    df, _ = recording
    src = tmp_path / "SUBJ001_task_gfm_sp_t.csv"
    df.to_csv(src, index=False)
    process_file(src, tmp_path / "out", cfg)
    meta = json.loads((tmp_path / "out" / "SUBJ001_task_gfm_sp_t_lp_hn.meta.json").read_text())
    assert meta["settings"]["landmark_indices"]["MLL"] == 17
    assert meta["ref_frame_used"] == 0
    assert (tmp_path / "out" / "SUBJ001_task_gfm_sp_t_lp_hn.csv").exists()


def test_cli_preprocess(recording, tmp_path, capsys):
    from wckdsmat.cli import main
    df, _ = recording
    df.to_csv(tmp_path / "SUBJ001_a.csv", index=False)
    df.drop(columns=["NT_x", "NT_y", "NT_z"]).to_csv(tmp_path / "SUBJ002_a.csv", index=False)
    code = main(["preprocess", str(tmp_path / "SUBJ00*_a.csv"), "--out-dir", str(tmp_path / "out")])
    text = capsys.readouterr().out
    assert code == 1                                    # one bad file -> nonzero exit, batch still ran
    assert "[ok] SUBJ001_a.csv" in text and "[fail] SUBJ002_a.csv" in text
    assert (tmp_path / "out" / "SUBJ001_a_lp_hn.csv").exists()


def test_head_centroid_from_unnormalized_points(cfg, recording):
    df, _ = recording
    out, _ = preprocess(df, cfg)
    pts = ["LBH", "LTH", "RBH", "RTH"]
    expect = np.mean([out[[f"{p}_lp_{a}" for a in "XYZ"]].to_numpy() for p in pts], axis=0)
    got = out[["HEAD_CENTROID_lp_X", "HEAD_CENTROID_lp_Y", "HEAD_CENTROID_lp_Z"]].to_numpy()
    assert np.allclose(got, expect) and np.ptp(got[:, 0]) > 5      # follows head motion
    assert "HEAD_CENTROID_hn_X" not in out
