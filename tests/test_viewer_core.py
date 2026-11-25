"""Viewer logic without the web app: loading, audio, spectrograms, annotations, export."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from wckdsmat.config import load_config
from wckdsmat.viewer.annotations import Annotations, upsert_segment_table
from wckdsmat.viewer.data import (auto_audio_offset, load_recording, load_wav, recording_name,
                                  spectrogram_window)
from wckdsmat.viewer.export import segment_file


def preprocess_like(tmp_path, n=300, fs=30.0, start_at_one_frame=False):
    t = (np.arange(n) + (1 if start_at_one_frame else 0)) / fs
    df = pd.DataFrame({"time_s": t, "segment": 0, "source_row": np.arange(n)})
    for name in ("MLL", "LJ", "RJ", "LBH"):
        for k, a in enumerate("XYZ"):
            df[f"{name}_hn_{a}"] = np.sin(2 * np.pi * (1 + k) * t) + k
    for a in "XYZ":
        df[f"LBH_lp_{a}"] = t
    p = tmp_path / "SUBJ001_task_gfm_sp_t_cal_lp_hn.csv"
    df.to_csv(p, index=False)
    return p


# -- data ---------------------------------------------------------------------
def test_recording_name_strips_stage_suffixes():
    assert recording_name("SUBJ001_task_gfm_sp_t_cal_lp_hn.csv") == "SUBJ001_task"
    assert recording_name("SUBJ001_task.csv") == "SUBJ001_task"


def test_load_recording_finds_points_and_spaces(tmp_path):
    rec = load_recording(preprocess_like(tmp_path))
    assert rec.name == "SUBJ001_task" and rec.spaces == ["hn", "lp"]
    assert rec.names("hn") == ["MLL", "LJ", "RJ", "LBH"] and rec.points[("MLL", "hn")].shape == (300, 3)
    assert rec.fs == pytest.approx(30.0)


def test_legacy_and_extract_columns(tmp_path):
    df = pd.DataFrame({"timestamp": np.arange(5) / 30, "MLL_x": 1.0, "MLL_y": 2.0, "MLL_z": 3.0,
                       "RCE_px_X": 1.0, "RCE_px_Y": 1.0, "RCE_px_Z": 1.0, "lmk_0_x": 0, "lmk_0_y": 0, "lmk_0_z": 0})
    p = tmp_path / "a.csv"
    df.to_csv(p, index=False)
    rec = load_recording(p)
    assert set(rec.points) == {("MLL", ""), ("RCE", "px")}


def test_audio_offset_for_one_based_frame_stamps(tmp_path):
    assert auto_audio_offset(load_recording(preprocess_like(tmp_path))) == 0.0
    rec = load_recording(preprocess_like(tmp_path, start_at_one_frame=True))
    assert auto_audio_offset(rec) == pytest.approx(1 / 30)


def tone(sr=16000, dur=1.0, f0=200.0):
    t = np.arange(int(sr * dur)) / sr
    return sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 20)).astype(np.float32)


def test_narrowband_resolves_harmonics_wideband_does_not():
    sr, x = 16000, tone()
    for band, resolved in (("narrow", True), ("wide", False)):
        t, f, db = spectrogram_window(x, sr, 0.2, 0.8, band=band, pre_emphasis=0)
        spec = db.mean(axis=1)
        at_h = spec[np.argmin(np.abs(f - 1000))]          # 5th harmonic
        between = spec[np.argmin(np.abs(f - 1100))]       # halfway to the 6th
        assert bool(at_h - between > 10) == resolved, band


def test_spectrogram_window_limits_size_and_range():
    sr, x = 16000, tone(dur=20.0)
    t, f, db = spectrogram_window(x, sr, 0, 20, band="wide", max_cols=500, max_rows=100, max_freq=5000)
    assert len(t) <= 501 and len(f) <= 100 and f.max() <= 5000 and t.min() >= 0 and t.max() <= 20
    assert db.max() - db.min() <= 70.0 + 1e-9


def test_load_wav_int16(tmp_path):
    p = tmp_path / "a.wav"
    wavfile.write(p, 16000, (tone() * 10000).astype(np.int16))
    sr, x = load_wav(p)
    assert sr == 16000 and x.dtype == np.float32 and np.abs(x).max() < 1


# -- annotations --------------------------------------------------------------
def test_flags_sorted_and_intervals():
    a = Annotations("R")
    f2, f1, f3 = a.add_flag(2.0, "b"), a.add_flag(1.0, "a"), a.add_flag(3.0, "c")
    assert [f.t for f in a.flags] == [1.0, 2.0, 3.0]
    assert [(x.id, y.id) for x, y in a.intervals()] == [(f1, f2), (f2, f3)]


def test_label_segments_between_flags_and_table():
    a = Annotations("R")
    ids = [a.add_flag(t) for t in (1.0, 2.0, 3.0, 4.0)]
    a.set_segment(ids[0], ids[1], "pa1", "ddk")
    a.set_segment(ids[2], ids[3], "pa2", "ddk")
    tab = a.segment_table()
    assert list(tab.segment_id) == ["1", "2"] and list(tab.label) == ["pa1", "pa2"]
    assert list(tab.start_s) == [1.0, 3.0] and set(tab.recording) == {"R"}
    with pytest.raises(ValueError):
        a.set_segment(ids[0], ids[2], "x")                 # not neighbours


def test_moving_flag_moves_boundary_and_stale_segments_drop():
    a = Annotations("R")
    i1, i2, i3 = (a.add_flag(t) for t in (1.0, 2.0, 3.0))
    a.set_segment(i1, i2, "s1")
    a.move_flag(i2, 1.5)
    assert a.segment_table().end_s.iloc[0] == 1.5
    a.move_flag(i3, 1.2)                                   # now between i1 and i2: old pair no longer neighbours
    assert a.segment_table().empty
    a2 = Annotations("R")
    j1, j2 = a2.add_flag(1.0), a2.add_flag(2.0)
    a2.set_segment(j1, j2, "s")
    a2.delete_flag(j2)
    assert a2.segments == {}


def test_save_load_round_trip(tmp_path):
    a = Annotations("R")
    i1, i2 = a.add_flag(1.0, "on"), a.add_flag(2.0, "off")
    a.set_segment(i1, i2, "pa", "ddk")
    p = a.save(tmp_path / "R_annotations.json")
    b = Annotations.load(p, "R")
    assert b.segment_table().equals(a.segment_table()) and json.loads(p.read_text())["recording"] == "R"
    assert Annotations.load(tmp_path / "missing.json", "R").flags == []


def test_upsert_replaces_only_this_recording(tmp_path):
    p = tmp_path / "segments.csv"
    for rec, n in (("A", 2), ("B", 1), ("A", 3)):
        a = Annotations(rec)
        ids = [a.add_flag(t) for t in range(n + 1)]
        for x, y in zip(ids, ids[1:]):
            a.set_segment(x, y, "s")
        upsert_segment_table(p, a.segment_table(), rec)
    t = pd.read_csv(p)
    assert (t.recording == "A").sum() == 3 and (t.recording == "B").sum() == 1


# -- segment this file ---------------------------------------------------------
def test_segment_file_writes_cuts_and_kinematics(tmp_path):
    p = preprocess_like(tmp_path)
    a = Annotations("SUBJ001_task")
    ids = [a.add_flag(t) for t in (1.0, 2.0, 3.0, 4.0)]
    a.set_segment(ids[0], ids[1], "pa", "ddk")
    a.set_segment(ids[2], ids[3], "PA", "ddk")             # differs only in case: files must not collide
    res = segment_file(p, a.segment_table(), tmp_path / "out", load_config(), "SUBJ001_task")
    assert len(res["cut_files"]) == 2 and len(set(n.lower() for n in res["cut_files"])) == 2
    cut = pd.read_csv(tmp_path / "out" / "SUBJ001_task" / "segments" / res["cut_files"][0])
    assert cut.time_s.min() >= 1.0 and cut.time_s.max() <= 2.0 and cut.t_rel_s.iloc[0] == 0
    assert res["kinematics"]["n_segments"] == 2
    vel = pd.read_csv(tmp_path / "out" / "SUBJ001_task" / "kinematics" / "SUBJ001_task_velocity.csv")
    assert set(vel.segment_id.astype(str)) == {"1", "2"} and "MLL_hn_3D" in vel


def test_exported_table_feeds_kinematics_cli(tmp_path):
    from wckdsmat.cli import main
    p = preprocess_like(tmp_path)
    a = Annotations("SUBJ001_task")
    i1, i2 = a.add_flag(1.0), a.add_flag(2.5)
    a.set_segment(i1, i2, "pa", "ddk")
    a.segment_table().to_csv(tmp_path / "segs.csv", index=False)
    assert main(["kinematics", str(p), "--out-dir", str(tmp_path / "k"), "--segments", str(tmp_path / "segs.csv")]) == 0
