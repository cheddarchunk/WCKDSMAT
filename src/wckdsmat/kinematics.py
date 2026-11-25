"""Stage 4: per-segment kinematics from preprocessed trajectories.

For each task segment (see :mod:`wckdsmat.segments`), and for every point in
the preprocess output (``NAME_hn_*`` articulators and derived points,
``NAME_lp_*`` head-movement points):

* position: the input trajectory
* displacement: position minus the segment's reference (``displacement_ref``)
* velocity: first derivative (np.gradient), per second or per frame
* acceleration: derivative of velocity

Derivatives never cross a recording gap: inside a segment they are taken
separately on each continuous piece (the ``segment`` column written by
preprocess). ``<NAME>_<space>_3D`` magnitudes are added for the products in
``kinematics.magnitude_3d``. The config never allows position here (fix log).

Replaces the earlier ``kinematics_imll``, ``kinematics_grouped`` and ``head``
scripts (see docs/PROVENANCE.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AXES, Config
from .segments import load_segments, rows_for_recording

POINT_RE = re.compile(r"^(?P<name>[A-Z][A-Z0-9_]*)_(?P<space>hn|lp)_(?P<axis>[XYZ])$")
ID_COLUMNS = ("recording", "segment_id", "task", "time_s", "t_rel_s")


@dataclass
class KinematicsReport:
    source: str = ""
    n_segments: int = 0
    empty_segments: list[str] = field(default_factory=list)
    segments_crossing_gaps: list[str] = field(default_factory=list)
    derivative_units: str = "per_second"
    points: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def find_points(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(name, space) for every complete NAME_space_X/Y/Z triplet, in column order."""
    seen: dict[tuple[str, str], set[str]] = {}
    for c in df.columns:
        m = POINT_RE.match(str(c))
        if m:
            seen.setdefault((m["name"], m["space"]), set()).add(m["axis"])
    return [k for k, axes in seen.items() if axes == set(AXES)]


def _derivative(X: np.ndarray, t: np.ndarray, pieces: np.ndarray, per_second: bool) -> np.ndarray:
    out = np.full_like(X, np.nan)
    for p in np.unique(pieces):
        idx = np.flatnonzero(pieces == p)
        if len(idx) < 2:
            continue
        edge = 2 if len(idx) >= 3 else 1
        out[idx] = (np.gradient(X[idx], t[idx], axis=0, edge_order=edge) if per_second
                    else np.gradient(X[idx], axis=0, edge_order=edge))
    return out


def _reference(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.nanmean(X, axis=0)
    if mode == "first":
        return X[0]
    ok = np.flatnonzero(np.isfinite(X).all(axis=1))          # first_valid
    return X[ok[0]] if len(ok) else np.full(3, np.nan)


def kinematics(df: pd.DataFrame, cfg: Config, segments: pd.DataFrame | None = None,
               recording: str = "", source: str = "") -> tuple[dict[str, pd.DataFrame], KinematicsReport]:
    k = cfg.kinematics
    rep = KinematicsReport(source=source)
    products = list(k.get("products", ["position", "displacement", "velocity", "acceleration"]))
    mags = set(k.get("magnitude_3d", []))
    per_second = k.get("derivative", "time") == "time"
    rep.derivative_units = "per_second" if per_second else "per_frame"
    ref_mode = k.get("displacement_ref", "first_valid")

    if "time_s" not in df.columns:
        raise ValueError("input needs a time_s column (run wckdsmat preprocess first)")
    t = df["time_s"].to_numpy(float)
    pieces = df["segment"].to_numpy() if "segment" in df.columns else np.zeros(len(df), int)
    points = find_points(df)
    if not points:
        raise ValueError("no NAME_hn_X / NAME_lp_X point columns found")
    rep.points = [f"{n}_{s}" for n, s in points]

    if segments is None or segments.empty:      # no table: each continuous piece is one segment
        segs = [(f"piece{p}", "", t[pieces == p][0], t[pieces == p][-1]) for p in np.unique(pieces)]
        extra_cols: list[str] = []
        extra_vals = [{} for _ in segs]
    else:
        segs = list(segments[["segment_id", "task", "start_s", "end_s"]].itertuples(index=False, name=None))
        extra_cols = [c for c in segments.columns if c not in ("recording", "segment_id", "task", "start_s", "end_s")]
        extra_vals = [{c: row[c] for c in extra_cols} for _, row in segments.iterrows()]   # one dict per row, even with no extras

    tables: dict[str, list[pd.DataFrame]] = {p: [] for p in products}
    for (sid, task, start, end), extra in zip(segs, extra_vals):
        rows = np.flatnonzero((t >= start) & (t <= end))
        if not len(rows):
            rep.empty_segments.append(str(sid))
            continue
        rep.n_segments += 1
        if len(np.unique(pieces[rows])) > 1:
            rep.segments_crossing_gaps.append(str(sid))
        ts, ps = t[rows], pieces[rows]
        base = {"recording": recording, "segment_id": str(sid), "task": task,
                "time_s": ts, "t_rel_s": ts - ts[0], **extra}
        cols = {p: dict(base) for p in products}
        for name, space in points:
            X = df[[f"{name}_{space}_{a}" for a in AXES]].to_numpy(float)[rows]
            vals = {"position": X}
            if "displacement" in products:
                vals["displacement"] = X - _reference(X, ref_mode)
            if "velocity" in products or "acceleration" in products:
                vals["velocity"] = _derivative(X, ts, ps, per_second)
            if "acceleration" in products:
                vals["acceleration"] = _derivative(vals["velocity"], ts, ps, per_second)
            for p in products:
                for i, a in enumerate(AXES):
                    cols[p][f"{name}_{space}_{a}"] = vals[p][:, i]
                if p in mags:
                    cols[p][f"{name}_{space}_3D"] = np.linalg.norm(vals[p], axis=1)
        for p in products:
            tables[p].append(pd.DataFrame(cols[p]))

    out = {p: (pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=list(ID_COLUMNS)))
           for p, v in tables.items()}
    rep.settings = {"config": cfg.source, "kinematics": dict(k), "extra_segment_columns": extra_cols}
    return out, rep


def process_file(in_path, out_dir, cfg: Config, segments: pd.DataFrame | None = None) -> KinematicsReport:
    in_path, out_dir = Path(in_path), Path(out_dir)
    stem = in_path.name[:-4] if in_path.name.lower().endswith(".csv") else in_path.stem
    seg_rows = None
    recording = stem
    if segments is not None:
        seg_rows = rows_for_recording(segments, in_path.name)
        if seg_rows.empty:
            raise ValueError(f"no rows in the segment table match {in_path.name}")
        recording = str(seg_rows["recording"].iloc[0])
    tables, rep = kinematics(pd.read_csv(in_path), cfg, seg_rows, recording=recording, source=in_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for product, table in tables.items():
        table.to_csv(out_dir / f"{stem}_{product}.csv", index=False)
    (out_dir / f"{stem}_kinematics.meta.json").write_text(json.dumps(asdict(rep), indent=2, default=str))
    return rep


__all__ = ["kinematics", "process_file", "find_points", "load_segments"]
