# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 WCKDSMAT contributors
"""Validation run: per-segment agreement metrics and the paper-style summary.

Per segment (see :mod:`wckdsmat.validate.pairing` for cut / resample / align):

* traces on the GFM time base: position, displacement (minus the first frame),
  velocity (np.gradient on time, edge order 2); derived points from the config
  (e.g. ``IMLL``) are rebuilt from the aligned base points;
* metrics per point, signal (pos / disp / vel) and axis (x, y, z, 3D);
* spectral features on displacement and velocity.

Across segments: movement scales ROM95 / Speed95 per participant and point
(the largest per-segment 95th percentile of the reference 3D excursion /
speed) and normalized uRMSE. ``summarize`` reproduces the paper's headline
aggregation (mean per participant, then mean +/- SD across participants).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..segments import load_segments, rows_for_recording
from . import metrics as M
from .pairing import pair_segment

SIGNALS = ("pos", "disp", "vel")


def _derivative(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    tu, idx = np.unique(t, return_index=True)
    d = np.gradient(X[idx], tu, axis=0, edge_order=2 if len(tu) >= 3 else 1)
    return np.column_stack([np.interp(t, tu, d[:, k], left=np.nan, right=np.nan) for k in range(X.shape[1])])


def _with_derived(points: dict[str, np.ndarray], cfg: Config) -> dict[str, np.ndarray]:
    names = {d.name.lower() for d in cfg.derived}
    pts = {k: v for k, v in points.items() if k not in names}      # rebuild derived points after alignment
    for d in cfg.derived:
        if d.space != "hn":
            continue
        of = [r.lower() for r in d.of]
        if all(r in pts for r in of):
            parts = [pts[r] for r in of]
            pts[d.name.lower()] = np.mean(parts, axis=0) if d.op == "mean" else parts[0] - parts[1]
    return pts


def traces(points: dict[str, np.ndarray], t: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return {"pos": points,
            "disp": {s: p - p[:1] for s, p in points.items()},
            "vel": {s: _derivative(p, t) for s, p in points.items()}}


@dataclass
class SegmentResult:
    rows: list[dict]
    spectral: list[dict]
    traces: dict
    info: dict


def validate_segment(gfm: pd.DataFrame, ctx: pd.DataFrame, cfg: Config, *, t0: float, t1: float,
                     offset: float) -> SegmentResult:
    v = cfg.validate
    bases = list(cfg.resolve(v.get("bases", "head_anchors")))
    p = pair_segment(gfm, ctx, t0=t0, t1=t1, offset=offset, bases=bases,
                     min_valid_frames=int(v.get("min_valid_frames", 5)))
    lag_fs = p["fs"] if v.get("lag_fs", "measured") == "measured" else float(v["lag_fs"])
    max_lag = int(round(float(v.get("max_lag_ms", 500.0)) / 1000.0 * lag_fs))
    lo, hi = v.get("range_percentiles", [5.0, 95.0])
    floor = float(v.get("range_floor_mm", 0.5))
    G = traces(_with_derived(p["gfm"], cfg), p["t"])
    C = traces(_with_derived(p["ctx"], cfg), p["t"])

    fill = dict(p["ctx_fill"])
    for d in cfg.derived:                    # a derived point is as filled as its worst input
        of = [r.lower() for r in d.of]
        if all(r in fill for r in of):
            fill[d.name.lower()] = max(fill[r] for r in of)
    max_fill = float(v.get("max_reference_fill", 1.0))
    rows, spectral = [], []
    for stem in sorted(G["pos"]):
        f = fill.get(stem, float("nan"))
        q = {"ref_fill_frac": f, "ref_ok": bool(np.isfinite(f) and f <= max_fill)}
        for sig in SIGNALS:
            A, B = G[sig][stem], C[sig][stem]
            kw = dict(fs=lag_fs, max_lag_samples=max_lag, lo=lo, hi=hi, floor=floor)
            for m in [*M.axis_metrics(A, B, **kw), M.metrics_3d(A, B, **kw)]:
                rows.append({"stem": stem, "signal": sig, **m, **q})
        for sig in ("disp", "vel"):
            A, B = G[sig][stem], C[sig][stem]
            for ax, a, b in [*((x, A[:, i], B[:, i]) for i, x in enumerate("xyz")),
                             ("3d", np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1))]:
                ok = np.isfinite(a) & np.isfinite(b)
                a, b = a[ok], b[ok]
                cc = M.zscore_cross_correlation(b, a, p["fs"], float(v.get("spectral_max_lag_ms", 250.0)))
                pc, pg = M.psd_features(b, p["fs"]), M.psd_features(a, p["fs"])
                spectral.append({"stem": stem, "signal": sig, "axis": ax, **q, **cc,
                                 **{f"{k}_ctx": x for k, x in pc.items()}, **{f"{k}_gfm": x for k, x in pg.items()},
                                 "dom_freq_absdiff": abs(pc["dom_freq"] - pg["dom_freq"]),
                                 "centroid_absdiff": abs(pc["centroid"] - pg["centroid"])})
    info = {"n_frames": int(len(p["t"])), "fs": p["fs"], "lag_fs": lag_fs, "n_fit_frames": p["n_fit_frames"],
            "bases": bases, "R": p["R"].tolist(), "translation": p["translation"].tolist()}
    return SegmentResult(rows, spectral, {"t": p["t"], "gfm": G, "ctx": C}, info)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------
def _find(root: Path, recording: str, suffix_hint: str) -> Path:
    hits = sorted(p for p in root.rglob("*.csv") if p.name.startswith(recording))
    if len(hits) > 1:
        narrowed = [h for h in hits if suffix_hint in h.name]
        hits = narrowed or hits
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one file for {recording!r} under {root}, found {len(hits)}")
    return hits[0]


def validate_many(segments: pd.DataFrame, gfm_root: Path, ctx_root: Path, cfg: Config,
                  log=print) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """segments needs recording, segment_id, start_s, end_s, ctx_offset_s; optional task, subject, ctx_recording."""
    segs = load_segments(segments)
    if "ctx_offset_s" not in segs:
        raise ValueError("segment table needs a ctx_offset_s column (CTX time shift, seconds)")
    cache: dict[Path, pd.DataFrame] = {}
    rows, spec, scales, failures = [], [], [], []
    for _, s in segs.iterrows():
        meta = {"recording": s.recording, "segment_id": s.segment_id, "task": s.get("task", ""),
                "subject": s.get("subject", "")}
        try:
            gpath = _find(Path(gfm_root), str(s.recording), "gfm")
            cpath = _find(Path(ctx_root), str(s.get("ctx_recording") or s.recording), "ctx")
            for path in (gpath, cpath):
                if path not in cache:
                    cache[path] = pd.read_csv(path)
            r = validate_segment(cache[gpath], cache[cpath], cfg, t0=float(s.start_s), t1=float(s.end_s),
                                 offset=float(s.ctx_offset_s))
            rows += [{**meta, **x} for x in r.rows]
            spec += [{**meta, **x} for x in r.spectral]
            # movement scales from the reference 3D excursion / speed
            scales += [{**meta, "stem": st,
                        "ROM95_seg": float(np.nanpercentile(np.linalg.norm(r.traces["ctx"]["disp"][st], axis=1), 95)),
                        "Speed95_seg": float(np.nanpercentile(np.linalg.norm(r.traces["ctx"]["vel"][st], axis=1), 95))}
                       for st in r.traces["ctx"]["pos"]]
        except Exception as e:  # noqa: BLE001 - keep the batch going
            failures.append({**meta, "error": f"{type(e).__name__}: {e}"})
            log(f"  [fail] {s.recording} / {s.segment_id}: {e}")
    out = add_normalized(pd.DataFrame(rows), pd.DataFrame(scales), cfg)
    return out, pd.DataFrame(spec), failures


def add_normalized(rows: pd.DataFrame, scales: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if rows.empty or scales.empty:
        return rows
    v = cfg.validate
    key = "subject" if (scales["subject"].astype(str) != "").all() else "recording"
    sc = scales.groupby([key, "stem"], as_index=False).agg(ROM95_mm=("ROM95_seg", "max"),
                                                          Speed95_mm_s=("Speed95_seg", "max"))
    out = rows.merge(sc, on=[key, "stem"], how="left")
    is3d = out["axis"] == "3D"
    rom_ok = is3d & out["signal"].isin(["pos", "disp"]) & (out["ROM95_mm"] >= float(v.get("rom_floor_mm", 2.0)))
    sp_ok = is3d & (out["signal"] == "vel") & (out["Speed95_mm_s"] >= float(v.get("speed_floor_mm_s", 2.0)))
    out["uRMSE_norm_ROM95"] = np.where(rom_ok, out["uRMSE"] / out["ROM95_mm"], np.nan)
    out["uRMSE_norm_Speed95"] = np.where(sp_ok, out["uRMSE"] / out["Speed95_mm_s"], np.nan)
    return out


def summarize(rows: pd.DataFrame, spectral: pd.DataFrame | None, cfg: Config) -> pd.DataFrame:
    """Paper-style headline: per-participant means over the summary landmarks, then mean +/- SD."""
    v = cfg.validate
    stems = [s.lower() for s in cfg.resolve(v.get("summary_landmarks", "lips"))]
    by = "subject" if "subject" in rows and (rows["subject"].astype(str) != "").all() else "recording"
    r3 = rows[(rows["axis"] == "3D") & rows["stem"].isin(stems)]
    if "ref_ok" in r3:
        r3 = r3[r3["ref_ok"].astype(bool)]
        if spectral is not None and "ref_ok" in spectral:
            spectral = spectral[spectral["ref_ok"].astype(bool)]
    out = []

    def add(name, df, col, scale=1.0):
        per = df.groupby(by)[col].mean() * scale
        out.append({"measure": name, "mean": per.mean(), "sd": per.std(), "n_participants": len(per),
                    "n_segments": df[["recording", "segment_id"]].drop_duplicates().shape[0]})

    add("3D position uRMSE (mm)", r3[r3.signal == "pos"], "uRMSE")
    if "uRMSE_norm_ROM95" in r3:
        add("3D displacement uRMSE / ROM95 (%)", r3[r3.signal == "disp"], "uRMSE_norm_ROM95", 100)
        add("3D speed uRMSE / Speed95 (%)", r3[r3.signal == "vel"], "uRMSE_norm_Speed95", 100)
    add("3D velocity peak cross-correlation", r3[r3.signal == "vel"], "CC_peak")
    add("3D velocity peak lag (ms)", r3[r3.signal == "vel"], "CC_peak_lag_ms")
    if spectral is not None and len(spectral):
        sp = spectral[(spectral["axis"] == "3d") & spectral["stem"].isin(stems)]
        add("displacement spectral centroid difference (Hz)", sp[sp.signal == "disp"], "centroid_absdiff")
        add("speed spectral centroid difference (Hz)", sp[sp.signal == "vel"], "centroid_absdiff")
    per_lm = (r3[r3.signal == "pos"].groupby("stem")["uRMSE"].mean().rename("3D position uRMSE (mm)").reset_index())
    return pd.DataFrame(out), per_lm


def write_outputs(out_dir: Path, rows, spectral, failures, cfg: Config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "validation_metrics.csv", index=False)
    spectral.to_csv(out_dir / "validation_spectral.csv", index=False)
    headline, per_lm = summarize(rows, spectral, cfg)
    headline.to_csv(out_dir / "validation_summary.csv", index=False)
    per_lm.to_csv(out_dir / "validation_by_landmark.csv", index=False)
    (out_dir / "validation.meta.json").write_text(json.dumps(
        {"config": cfg.source, "validate": cfg.validate, "failures": failures}, indent=2, default=str))
    return headline, per_lm


__all__ = ["validate_segment", "validate_many", "summarize", "write_outputs", "rows_for_recording"]
