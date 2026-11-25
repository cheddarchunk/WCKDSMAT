# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 WCKDSMAT contributors
"""Event flags and labeled segments (no web code).

Flags are time points with an optional label. Every interval between two
neighbouring flags is a candidate segment; giving it a label makes it a
segment. Segment labels are keyed by the two flag ids, so dragging a flag
moves the segment boundary with it. Deleting a flag drops the segments that
used it.

``segment_table`` writes the package segment-table format
(``recording, segment_id, task, start_s, end_s`` + ``label``), which
``wckdsmat kinematics`` and ``wckdsmat validate`` read directly.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Flag:
    id: str
    t: float
    label: str = ""


@dataclass
class Annotations:
    recording: str
    flags: list[Flag] = field(default_factory=list)
    segments: dict[str, dict] = field(default_factory=dict)    # "startid|endid" -> {label, task}

    # -- flags ---------------------------------------------------------------
    def add_flag(self, t: float, label: str = "") -> str:
        fid = uuid.uuid4().hex[:8]
        self.flags.append(Flag(fid, float(t), label))
        self._sort()
        return fid

    def move_flag(self, fid: str, t: float) -> None:
        self._get(fid).t = float(t)
        self._sort()
        self._drop_stale()

    def relabel_flag(self, fid: str, label: str) -> None:
        self._get(fid).label = label

    def delete_flag(self, fid: str) -> None:
        self.flags = [f for f in self.flags if f.id != fid]
        self._drop_stale()

    # -- segments ------------------------------------------------------------
    def intervals(self) -> list[tuple[Flag, Flag]]:
        return list(zip(self.flags[:-1], self.flags[1:]))

    def set_segment(self, start_id: str, end_id: str, label: str = "", task: str = "") -> None:
        key = f"{start_id}|{end_id}"
        if key not in {f"{a.id}|{b.id}" for a, b in self.intervals()}:
            raise ValueError("segments run between neighbouring flags")
        if label.strip() or task.strip():
            self.segments[key] = {"label": label.strip(), "task": task.strip()}
        else:
            self.segments.pop(key, None)

    def interval_rows(self) -> list[dict]:
        """One row per interval between neighbouring flags (labeled or not), in time order."""
        rows = []
        for a, b in self.intervals():
            s = self.segments.get(f"{a.id}|{b.id}", {})
            rows.append({"key": f"{a.id}|{b.id}", "start_s": a.t, "end_s": b.t, "duration_s": b.t - a.t,
                         "start_flag": a.label, "end_flag": b.label,
                         "label": s.get("label", ""), "task": s.get("task", "")})
        return rows

    def segment_table(self) -> pd.DataFrame:
        rows = [r for r in self.interval_rows() if r["label"] or r["task"]]
        out = pd.DataFrame([{"recording": self.recording, "segment_id": str(i + 1), "task": r["task"],
                             "label": r["label"], "start_s": round(r["start_s"], 6), "end_s": round(r["end_s"], 6),
                             "start_flag": r["start_flag"], "end_flag": r["end_flag"]}
                            for i, r in enumerate(rows)])
        return out if len(out) else pd.DataFrame(columns=["recording", "segment_id", "task", "label",
                                                          "start_s", "end_s", "start_flag", "end_flag"])

    def event_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"recording": self.recording, "t_s": f.t, "label": f.label} for f in self.flags],
                            columns=["recording", "t_s", "label"])

    # -- persistence -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"recording": self.recording, "flags": [asdict(f) for f in self.flags], "segments": self.segments}

    @classmethod
    def from_dict(cls, d: dict) -> Annotations:
        a = cls(d["recording"], [Flag(**f) for f in d.get("flags", [])], dict(d.get("segments", {})))
        a._sort()
        a._drop_stale()
        return a

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(path)                       # atomic: never a half-written file
        return path

    @classmethod
    def load(cls, path: str | Path, recording: str) -> Annotations:
        path = Path(path)
        if not path.exists():
            return cls(recording)
        a = cls.from_dict(json.loads(path.read_text()))
        a.recording = recording
        return a

    # -- internals -----------------------------------------------------------
    def _get(self, fid: str) -> Flag:
        for f in self.flags:
            if f.id == fid:
                return f
        raise KeyError(fid)

    def _sort(self) -> None:
        self.flags.sort(key=lambda f: f.t)

    def _drop_stale(self) -> None:
        valid = {f"{a.id}|{b.id}" for a, b in self.intervals()}
        self.segments = {k: v for k, v in self.segments.items() if k in valid}


def upsert_segment_table(path: str | Path, table: pd.DataFrame, recording: str) -> Path:
    """Replace this recording's rows in a project-wide segment table (created if missing)."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, dtype={"segment_id": str})
        old = old[old["recording"].astype(str) != recording]
        table = pd.concat([old, table], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "seg"
