"""Validation tests on synthetic webcam / motion-capture pairs with known transforms and offsets."""
import numpy as np
import pandas as pd
import pytest
from conftest import FACE, rot_x, rot_y

from wckdsmat.config import load_config
from wckdsmat.validate import metrics as M
from wckdsmat.validate import validate_many, validate_segment
from wckdsmat.validate.pairing import cut, pair_segment, rigid_fit, straight_line_fraction

R_TRUE = rot_y(0.3) @ rot_x(-0.15)
T_TRUE = np.array([40.0, -12.0, 300.0])
OFFSET = 1.25          # CTX clock runs 1.25 s behind: ctx_t + OFFSET = webcam time


def motion(t):
    """Head-frame trajectories: MLL and jaw oscillate, everything else still."""
    out = {}
    for name, p in FACE.items():
        X = np.tile(np.array(p, float), (len(t), 1))
        if name in ("MLL", "RJ", "LJ", "MJ"):
            X[:, 1] -= (6 if name == "MLL" else 3) * (1 - np.cos(2 * np.pi * 3 * t)) / 2
        out[name] = X
    return out


def streams(gfm_noise=0.0, seed=0, fs_ctx=120.0, fs_gfm=30.0, dur=8.0, drop=None):
    rng = np.random.default_rng(seed)
    tg = np.arange(int(dur * fs_gfm)) / fs_gfm
    tc = np.arange(int(dur * fs_ctx)) / fs_ctx - OFFSET               # CTX on its own clock
    G, C = motion(tg), motion(tc + OFFSET)
    g = {"timestamp": tg}
    c = {"time_s": tc}
    for j, n in enumerate(FACE):
        # real markers are never perfectly still: a 0.02 mm, slow wiggle keeps traces non-linear
        wig = 0.02 * np.sin(2 * np.pi * (0.7 + 0.1 * j) * (tc + OFFSET))[:, None]
        wig_g = 0.02 * np.sin(2 * np.pi * (0.7 + 0.1 * j) * tg)[:, None]
        gp = G[n] + wig_g + rng.normal(0, gfm_noise, G[n].shape)
        cp = (C[n] + wig - T_TRUE) @ R_TRUE                             # CTX lives in another frame
        for k, a in enumerate("XYZ"):
            g[f"{n}_hn_{a}"] = gp[:, k]
            c[f"{n}_{a}"] = cp[:, k]
    if drop:
        for a in "XYZ":
            c[f"{drop}_{a}"] = np.full(len(tc), np.nan)
    return pd.DataFrame(g), pd.DataFrame(c)


CFG = load_config(overrides={"validate": {"lag_fs": "measured"}})


# -- metrics ------------------------------------------------------------------
def test_urmse_ignores_bias():
    e = np.array([1.0, 3.0, 1.0, 3.0])
    assert M.rmse(e) == pytest.approx(np.sqrt(5)) and M.urmse(e) == pytest.approx(1.0)


def test_peak_lag_sign_convention():
    t = np.arange(200) / 30.0
    a = np.sin(2 * np.pi * 1.5 * t)
    b = np.roll(a, 3)                                    # b lags a by 3 samples
    lag, lag_ms, cc = M.peak_cc_lag(a, b, fs=30.0, max_lag_samples=15)
    assert lag == -3 and lag_ms == pytest.approx(-100.0) and cc > 0.95


def test_rigid_fit_recovers_transform():
    A = np.random.default_rng(1).normal(size=(20, 3)) * 30
    R, t = rigid_fit(A, A @ R_TRUE.T + T_TRUE)
    assert np.allclose(R, R_TRUE) and np.allclose(t, T_TRUE)


def test_straight_line_fraction_flags_filled_gaps():
    t = np.linspace(0, 1, 100)
    real = np.column_stack([np.sin(7 * t), np.cos(5 * t), t ** 2])
    filled = real.copy()
    filled[30:70] = np.linspace(real[30], real[69], 40)
    assert straight_line_fraction(real) == 0.0
    assert straight_line_fraction(filled) == pytest.approx(0.40, abs=0.03)


# -- pairing ------------------------------------------------------------------
def test_cut_applies_offset_and_rebases():
    _, c = streams()
    seg = cut(c, 2.0, 3.0, OFFSET)
    assert seg["time_s"].iloc[0] == 0 and seg["time_s"].iloc[-1] == pytest.approx(1.0, abs=1 / 120)


def test_pairing_aligns_ctx_onto_webcam_space():
    g, c = streams()
    p = pair_segment(g, c, t0=1.0, t1=6.0, offset=OFFSET, bases=list(CFG.resolve("head_anchors")))
    assert np.abs(p["ctx"]["mll"] - p["gfm"]["mll"]).max() < 0.05      # mm, after resampling 120 -> 30 Hz
    assert np.allclose(p["R"], R_TRUE, atol=1e-6) and p["fs"] == pytest.approx(30.0)


def test_missing_alignment_marker_fails_clearly():
    g, c = streams(drop="RCE")
    with pytest.raises(ValueError, match="alignment landmark"):
        pair_segment(g, c, t0=1.0, t1=6.0, offset=OFFSET, bases=list(CFG.resolve("head_anchors")))


# -- segment and batch --------------------------------------------------------
def test_validate_segment_recovers_known_noise():
    g, c = streams(gfm_noise=0.5)
    r = validate_segment(g, c, CFG, t0=1.0, t1=7.0, offset=OFFSET)
    rows = pd.DataFrame(r.rows)
    x = rows[(rows.stem == "mll") & (rows.signal == "pos") & (rows.axis == "x")].iloc[0]
    assert x.uRMSE == pytest.approx(0.5, rel=0.15)                       # injected 0.5 mm noise
    assert {"imll", "jaw_mid"} <= set(rows.stem)                          # derived points rebuilt after alignment
    v = rows[(rows.stem == "mll") & (rows.signal == "vel") & (rows.axis == "y")].iloc[0]
    assert v.CC_peak > 0.9 and abs(v.CC_peak_lag_samp) <= 1
    assert (rows.ref_fill_frac == 0).all() and rows.ref_ok.all()


def test_validate_many_with_failures_and_summary(tmp_path):
    g1, c1 = streams(gfm_noise=0.3, seed=1)
    g2, c2 = streams(gfm_noise=0.3, seed=2, drop="LCE")
    for name, df in (("S1_ddk_gfm.csv", g1), ("S1_ddk_ctx.csv", c1), ("S2_ddk_gfm.csv", g2), ("S2_ddk_ctx.csv", c2)):
        df.to_csv(tmp_path / name, index=False)
    table = pd.DataFrame({"recording": ["S1_ddk", "S1_ddk", "S2_ddk"], "segment_id": ["a", "b", "a"],
                          "subject": ["S1", "S1", "S2"], "start_s": [1.0, 4.0, 1.0], "end_s": [3.0, 6.0, 3.0],
                          "ctx_offset_s": [OFFSET] * 3})
    rows, spec, fails = validate_many(table, tmp_path, tmp_path, CFG, log=lambda *a: None)
    assert len(fails) == 1 and "alignment landmark" in fails[0]["error"]
    assert rows[["recording", "segment_id"]].drop_duplicates().shape[0] == 2
    assert rows["uRMSE_norm_ROM95"].notna().any()
    from wckdsmat.validate import summarize
    head, per = summarize(rows, spec, CFG)
    pos = head[head.measure.str.startswith("3D position")].iloc[0]
    # 3D uRMSE = SD of the error distance; for 0.3 mm noise per axis that is 0.3 * sqrt(3 - 8/pi) = 0.205 mm
    assert pos.n_segments == 2 and pos["mean"] == pytest.approx(0.3 * np.sqrt(3 - 8 / np.pi), rel=0.2)


def test_max_reference_fill_excludes_filled_traces_from_summary():
    g, c = streams(gfm_noise=0.3)
    i = c.index[(c.time_s + OFFSET > 2.0) & (c.time_s + OFFSET < 4.0)]
    for a in "XYZ":
        c.loc[i, f"MLL_{a}"] = np.linspace(c.loc[i[0], f"MLL_{a}"], c.loc[i[-1], f"MLL_{a}"], len(i))
    strict = load_config(overrides={"validate": {"max_reference_fill": 0.5}})
    r = pd.DataFrame(validate_segment(g, c, strict, t0=1.5, t1=4.5, offset=OFFSET).rows)
    assert not r[r.stem == "mll"].ref_ok.any() and r[r.stem == "lcl"].ref_ok.all()


def test_cli_validate(tmp_path, capsys):
    from wckdsmat.cli import main
    g, c = streams(gfm_noise=0.3)
    g.to_csv(tmp_path / "S1_gfm.csv", index=False)
    c.to_csv(tmp_path / "S1_ctx.csv", index=False)
    pd.DataFrame({"recording": ["S1"], "segment_id": ["a"], "subject": ["S1"], "start_s": [1.0], "end_s": [5.0],
                  "ctx_offset_s": [OFFSET]}).to_csv(tmp_path / "segs.csv", index=False)
    code = main(["validate", "--segments", str(tmp_path / "segs.csv"), "--gfm-dir", str(tmp_path),
                 "--ctx-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    assert code == 0 and (tmp_path / "out" / "validation_summary.csv").exists()
    assert "3D position uRMSE" in capsys.readouterr().out
