import numpy as np
import pandas as pd
import pytest

from wckdsmat.config import ConfigError, load_config
from wckdsmat.kinematics import find_points, kinematics, process_file
from wckdsmat.segments import load_segments, rows_for_recording, value_for_recording


def recording(fs=30.0, n=300, t=None, piece=None):
    """Preprocess-shaped table: MLL moves vertically as a 2 Hz cosine (mm), head centroid drifts."""
    t = np.arange(n) / fs if t is None else t
    y = -5 * (1 - np.cos(2 * np.pi * 2 * t)) / 2
    df = pd.DataFrame({"time_s": t, "segment": np.zeros(len(t), int) if piece is None else piece})
    df["MLL_hn_X"], df["MLL_hn_Y"], df["MLL_hn_Z"] = 0.0, -38 + y, 20.0
    df["HEAD_CENTROID_lp_X"], df["HEAD_CENTROID_lp_Y"], df["HEAD_CENTROID_lp_Z"] = 3 * t, 50.0, 10.0
    return df, y


def table(*rows):
    return load_segments(pd.DataFrame(rows, columns=["recording", "segment_id", "task", "start_s", "end_s"]))


def test_find_points():
    df, _ = recording()
    assert find_points(df) == [("MLL", "hn"), ("HEAD_CENTROID", "lp")]


def test_displacement_relative_to_segment_onset():
    df, y = recording()
    out, rep = kinematics(df, load_config(), table(("R", "s1", "ddk", 1.0, 2.0), ("R", "s2", "srt", 5.0, 6.0)))
    d = out["displacement"]
    for sid in ("s1", "s2"):
        seg = d[d.segment_id == sid]
        assert seg["MLL_hn_Y"].iloc[0] == 0 and seg["t_rel_s"].iloc[0] == 0
    assert rep.n_segments == 2 and set(d.task) == {"ddk", "srt"}


def test_velocity_per_second_matches_analytic():
    df, _ = recording()
    out, _ = kinematics(df, load_config())
    t = out["velocity"]["time_s"].to_numpy()
    analytic = -5 * np.pi * 2 * np.sin(2 * np.pi * 2 * t)          # d/dt of -5(1-cos)/2
    assert np.abs(out["velocity"]["MLL_hn_Y"].to_numpy()[2:-2] - analytic[2:-2]).max() < 1.0   # mm/s
    assert np.allclose(out["velocity"]["HEAD_CENTROID_lp_X"], 3.0)                          # 3 mm/s drift


def test_frame_units_option():
    df, _ = recording()
    a, _ = kinematics(df, load_config())
    b, rep = kinematics(df, load_config(overrides={"kinematics": {"derivative": "frame"}}))
    assert rep.derivative_units == "per_frame"
    assert np.allclose(b["velocity"]["MLL_hn_Y"], a["velocity"]["MLL_hn_Y"] / 30.0)


def test_3d_magnitudes_never_on_position():
    df, _ = recording()
    out, _ = kinematics(df, load_config())
    assert "MLL_hn_3D" not in out["position"]
    assert {"MLL_hn_3D", "HEAD_CENTROID_lp_3D"} <= set(out["velocity"].columns)
    disp = out["displacement"]
    assert np.allclose(disp["MLL_hn_3D"], np.abs(disp["MLL_hn_Y"]))          # excursion
    with pytest.raises(ConfigError):
        load_config(overrides={"kinematics": {"magnitude_3d": ["position"]}})


def test_derivatives_do_not_cross_recording_gaps():
    fs = 30.0
    t = np.r_[np.arange(90), np.arange(90) + 400] / fs
    df, _ = recording(t=t, piece=np.r_[np.zeros(90, int), np.ones(90, int)])
    df.loc[90:, "MLL_hn_Y"] += 30.0                                  # jump across the cut
    out, rep = kinematics(df, load_config(), table(("R", "all", "", 0.0, 20.0)))
    assert rep.segments_crossing_gaps == ["all"]
    v = out["velocity"]["MLL_hn_Y"].to_numpy()
    assert np.nanmax(np.abs(v)) < 40                                  # no spike from the 30 mm jump


def test_no_table_uses_each_continuous_piece():
    t = np.r_[np.arange(90), np.arange(90) + 400] / 30.0
    df, _ = recording(t=t, piece=np.r_[np.zeros(90, int), np.ones(90, int)])
    out, rep = kinematics(df, load_config())
    assert rep.n_segments == 2 and set(out["displacement"].segment_id) == {"piece0", "piece1"}


def test_empty_segment_reported():
    df, _ = recording()
    _, rep = kinematics(df, load_config(), table(("R", "late", "", 50.0, 60.0)))
    assert rep.empty_segments == ["late"] and rep.n_segments == 0


def test_extra_segment_columns_carried_through():
    df, _ = recording()
    segs = table(("R", "s1", "ddk", 1.0, 2.0))
    segs["rater"] = "A"
    out, _ = kinematics(df, load_config(), segs)
    assert (out["velocity"]["rater"] == "A").all()


# -- segment tables -----------------------------------------------------------
def test_segment_table_validation():
    bad = pd.DataFrame({"recording": ["R", "R"], "segment_id": [1, 1], "start_s": [2.0, "x"], "end_s": [1.0, 3.0]})
    with pytest.raises(ValueError) as e:
        load_segments(bad)
    msg = str(e.value)
    assert "end_s must be after start_s" in msg and "not a number" in msg and "duplicate segment_id" in msg
    with pytest.raises(ValueError, match="missing columns"):
        load_segments(pd.DataFrame({"recording": ["R"]}))


def test_recording_match_longest_prefix():
    segs = table(("SUBJ001_task", "1", "", 0, 1), ("SUBJ001_task_b", "1", "", 0, 1))
    assert set(rows_for_recording(segs, "SUBJ001_task_b_gfm_lp_hn.csv").recording) == {"SUBJ001_task_b"}
    assert set(rows_for_recording(segs, "SUBJ001_task_gfm_lp_hn.csv").recording) == {"SUBJ001_task"}
    assert rows_for_recording(segs, "SUBJ002.csv").empty


def test_value_for_recording():
    eyes = pd.DataFrame({"recording": ["SUBJ001", "SUBJ002"], "eye_distance_mm": [96.0, 101.5]})
    assert value_for_recording(eyes, "SUBJ002_task_gfm_sp_t.csv", "eye_distance_mm") == 101.5
    assert value_for_recording(eyes, "SUBJ009.csv", "eye_distance_mm") is None


# -- files and CLI ------------------------------------------------------------
def test_process_file_and_cli(tmp_path, capsys):
    from wckdsmat.cli import main
    df, _ = recording()
    src = tmp_path / "SUBJ001_task_lp_hn.csv"
    df.to_csv(src, index=False)
    table(("SUBJ001_task", "s1", "ddk", 1.0, 2.0)).to_csv(tmp_path / "segs.csv", index=False)
    rep = process_file(src, tmp_path / "out", load_config(), load_segments(tmp_path / "segs.csv"))
    assert rep.n_segments == 1 and (tmp_path / "out" / "SUBJ001_task_lp_hn_velocity.csv").exists()
    code = main(["kinematics", str(src), "--out-dir", str(tmp_path / "out2"), "--segments", str(tmp_path / "segs.csv")])
    assert code == 0 and "1 segment(s)" in capsys.readouterr().out


def test_full_chain_calibrate_preprocess_kinematics(tmp_path):
    from conftest import FACE  # noqa: F401  (synthetic face layout)
    from test_calibrate import EYE_MM, to_pixels

    from wckdsmat.calibrate import calibrate
    from wckdsmat.preprocess import preprocess
    cal, _ = calibrate(to_pixels(n=150), load_config(), eye_distance_mm=EYE_MM)
    pre, _ = preprocess(cal, load_config())
    out, rep = kinematics(pre, load_config(), table(("R", "s1", "", 1.0, 3.0)))
    assert rep.n_segments == 1
    assert {"IMLL_hn_3D", "HEAD_CENTROID_lp_3D"} <= set(out["velocity"].columns)
    assert np.allclose(out["displacement"]["IMLL_hn_3D"], 0, atol=1e-6)     # rigid static face: no motion
