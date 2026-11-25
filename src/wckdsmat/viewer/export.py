"""Segment one recording from the viewer, without waiting for a batch run."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from ..kinematics import find_points, kinematics
from ..preprocess import find_time_column
from .annotations import safe_name


def segment_file(recording_path: str | Path, table: pd.DataFrame, out_dir: str | Path, cfg: Config,
                 recording: str) -> dict:
    """Write one cut CSV per labeled segment and, for preprocess outputs, per-segment kinematics.

    Cut files keep every original column (original time base) plus ``t_rel_s``.
    File names use the numeric segment id, so labels that differ only in case cannot
    overwrite each other on case-insensitive file systems.
    """
    recording_path, out_dir = Path(recording_path), Path(out_dir) / recording
    if table.empty:
        raise ValueError("no labeled segments to write")
    df = pd.read_csv(recording_path)
    tcol = find_time_column(df)
    t = pd.to_numeric(df[tcol], errors="coerce")
    cut_dir = out_dir / "segments"
    cut_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for _, s in table.iterrows():
        rows = df[(t >= s.start_s) & (t <= s.end_s)].copy()
        rows.insert(1, "t_rel_s", t[rows.index] - t[rows.index].iloc[0] if len(rows) else [])
        name = f"{recording}_seg{int(s.segment_id):03d}_{safe_name(str(s.label or s.task))}.csv"
        rows.to_csv(cut_dir / name, index=False)
        written.append(name)
    table.to_csv(out_dir / f"{recording}_segments.csv", index=False)

    result = {"out_dir": str(out_dir), "cut_files": written, "kinematics": None}
    if find_points(df):                         # preprocess output: run kinematics for these segments
        tables, rep = kinematics(df, cfg, table, recording=recording, source=recording_path.name)
        kin_dir = out_dir / "kinematics"
        kin_dir.mkdir(exist_ok=True)
        for product, tab in tables.items():
            tab.to_csv(kin_dir / f"{recording}_{product}.csv", index=False)
        result["kinematics"] = {"dir": str(kin_dir), "n_segments": rep.n_segments,
                                "empty_segments": rep.empty_segments,
                                "segments_crossing_gaps": rep.segments_crossing_gaps}
    return result
