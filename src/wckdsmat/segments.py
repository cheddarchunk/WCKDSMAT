"""Task-segment tables and per-recording lookup tables.

A task segment is an analysis window (a DDK trial, a sentence, a BAA block).
It is distinct from the *recording* segments that ``preprocess`` creates at
gaps in the timestamps. A segment table is a CSV with one row per window:

    recording,segment_id,task,start_s,end_s
    SUBJ001_task,1,ddk_pa,12.40,17.85
    SUBJ001_task,2,srt,20.10,23.00

``recording`` matches an input file whose name starts with it; the longest
match wins, so ``SUBJ001_task`` matches ``SUBJ001_task_gfm_sp_t_cal_lp_hn.csv``.
Times are seconds on the webcam timeline (the ``time_s`` column). Extra
columns are carried through to the outputs. The QC viewer writes this format.

The same matching is used for per-recording values such as the eye distance
(``recording,eye_distance_mm``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SEGMENT_COLUMNS = ("recording", "segment_id", "start_s", "end_s")


def _read(table: str | Path | pd.DataFrame) -> pd.DataFrame:
    return table.copy() if isinstance(table, pd.DataFrame) else pd.read_csv(table)


def load_segments(table: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read and check a segment table; raises ValueError listing every problem."""
    df = _read(table)
    missing = [c for c in SEGMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"segment table is missing columns {missing}; required: {list(SEGMENT_COLUMNS)}")
    if "task" not in df.columns:
        df["task"] = ""
    df["recording"] = df["recording"].astype(str)
    df["segment_id"] = df["segment_id"].astype(str)
    df["task"] = df["task"].fillna("").astype(str)
    problems = []
    for col in ("start_s", "end_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for i in df.index[df[col].isna()]:
            problems.append(f"row {i + 2}: {col} is missing or not a number")
    for i in df.index[df["end_s"] <= df["start_s"]]:
        problems.append(f"row {i + 2}: end_s must be after start_s")
    dup = df.duplicated(["recording", "segment_id"], keep=False)
    for i in df.index[dup]:
        problems.append(f"row {i + 2}: duplicate segment_id {df.at[i, 'segment_id']!r} for {df.at[i, 'recording']!r}")
    if problems:
        raise ValueError("invalid segment table:\n  - " + "\n  - ".join(problems))
    return df


def rows_for_recording(table: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Rows whose ``recording`` is the longest prefix of ``file_name``."""
    name = Path(file_name).name
    keys = [k for k in table["recording"].astype(str).unique() if name.startswith(k)]
    if not keys:
        return table.iloc[0:0]
    best = max(keys, key=len)
    return table[table["recording"].astype(str) == best]


def value_for_recording(table: pd.DataFrame, file_name: str, column: str):
    """Single per-recording value (e.g. eye_distance_mm), or None if the recording is not listed."""
    rows = rows_for_recording(table, file_name)
    if rows.empty:
        return None
    vals = rows[column].dropna().unique()
    if len(vals) != 1:
        raise ValueError(f"{file_name}: expected one {column} value, found {list(vals)}")
    return vals[0].item() if hasattr(vals[0], "item") else vals[0]
