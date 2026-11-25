#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC viewer for displacement files produced by batch_gfm_imll_timeseries*.py.

This version is matched to the batch script output layout:

    <output_root>/<relative_input_stem>/time_series/
        gfm_position.csv
        gfm_displacement.csv
        gfm_velocity.csv
        imll_position.csv
        imll_displacement.csv
        imll_velocity.csv
        speech_landmarks_long.csv

The viewer treats each time_series/ directory as one case, but labels and saved
PNGs using the parent relative path so files are not all named "time_series".

Interactive controls:
    n / right / space : next case
    p / left          : previous case
    g                 : jump to case number
    s                 : save current QC PNG
    q / escape        : quit
    h                 : print controls

Examples:
    python WCKDSMAT/qc_lip_jaw_displacement_viewer_for_batch_outputs.py \
        --root ./data/lp_hn_om/speech

    python qc_lip_jaw_displacement_viewer_for_batch_outputs.py \
        --root /path/to/imll_outputs \
        --save-all-png /path/to/qc_pngs
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AXES = ("x", "y", "z")
DEFAULT_STEMS = ("imll", "jaw_mid", "mll", "lj", "rj")
DISPLACEMENT_FILENAMES = ("gfm_displacement.csv", "imll_displacement.csv")


@dataclass
class Case:
    """One batch output case."""

    output_root: Path
    case_root: Path        # <output_root>/<relative_input_stem>
    time_series_dir: Path  # <case_root>/time_series
    imll_file: Optional[Path]
    gfm_file: Optional[Path]

    @property
    def rel_case(self) -> Path:
        try:
            return self.case_root.relative_to(self.output_root)
        except ValueError:
            return Path(self.case_root.name)

    @property
    def label(self) -> str:
        return str(self.rel_case)

    @property
    def sort_key(self) -> str:
        return str(self.rel_case).lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Case-insensitive column handling with collision protection."""
    lower_cols = [str(c).strip().lower() for c in df.columns]
    duplicates = sorted({c for c in lower_cols if lower_cols.count(c) > 1})
    if duplicates:
        raise ValueError(
            "Case-insensitive column normalization produced duplicate columns: "
            + ", ".join(duplicates)
        )
    out = df.copy()
    out.columns = lower_cols
    return out


def _looks_like_time_series_dir(path: Path) -> bool:
    return path.name == "time_series" and any((path / name).exists() for name in DISPLACEMENT_FILENAMES)


def find_cases(root: Path) -> List[Case]:
    """
    Find cases in the batch output tree.

    Preferred structure:
        root/<relative_input_stem>/time_series/gfm_displacement.csv

    Also tolerates older/flat layouts where displacement files are directly
    inside a case directory.
    """
    root = Path(root).expanduser().resolve()
    candidates: set[Path] = set()

    # Batch-script layout: explicit time_series directories.
    for p in root.rglob("time_series"):
        if p.is_dir() and _looks_like_time_series_dir(p):
            candidates.add(p)

    # Tolerate flat layout: a directory directly containing displacement files.
    for name in DISPLACEMENT_FILENAMES:
        for f in root.rglob(name):
            candidates.add(f.parent)

    cases: List[Case] = []
    for tsdir in sorted(candidates):
        if tsdir.name == "time_series":
            case_root = tsdir.parent
        else:
            case_root = tsdir

        imll = tsdir / "imll_displacement.csv"
        gfm = tsdir / "gfm_displacement.csv"
        cases.append(
            Case(
                output_root=root,
                case_root=case_root,
                time_series_dir=tsdir,
                imll_file=imll if imll.exists() else None,
                gfm_file=gfm if gfm.exists() else None,
            )
        )

    # De-duplicate if both discovery passes found the same time_series directory.
    unique: Dict[Path, Case] = {c.time_series_dir: c for c in cases}
    return sorted(unique.values(), key=lambda c: c.sort_key)


def load_case(case: Case) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    if case.gfm_file is not None:
        data["gfm"] = normalize_columns(pd.read_csv(case.gfm_file))
    if case.imll_file is not None:
        data["imll"] = normalize_columns(pd.read_csv(case.imll_file))
    return data


def get_time(df: pd.DataFrame, fs: Optional[float] = None) -> np.ndarray:
    if "time_s" in df.columns:
        return pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
    if "time" in df.columns:
        return pd.to_numeric(df["time"], errors="coerce").to_numpy(float)
    if fs is not None and np.isfinite(fs) and fs > 0:
        return np.arange(len(df), dtype=float) / float(fs)
    return np.arange(len(df), dtype=float)


def has_xyz(df: pd.DataFrame, stem: str) -> bool:
    stem = stem.lower()
    return all(f"{stem}_{ax}" in df.columns for ax in AXES)


def xyz(df: pd.DataFrame, stem: str) -> np.ndarray:
    stem = stem.lower()
    return np.vstack([pd.to_numeric(df[f"{stem}_{ax}"], errors="coerce").to_numpy(float) for ax in AXES]).T


def ensure_3d(df: pd.DataFrame, stem: str) -> np.ndarray:
    stem = stem.lower()
    c = f"{stem}_3d"
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    return np.linalg.norm(xyz(df, stem), axis=1)


def robust_limits(values: Sequence[np.ndarray], pad: float = 0.08) -> Optional[Tuple[float, float]]:
    clean = []
    for v in values:
        arr = np.asarray(v, float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            clean.append(arr)
    if not clean:
        return None
    allv = np.concatenate(clean)
    lo = np.nanpercentile(allv, 1)
    hi = np.nanpercentile(allv, 99)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        delta = 1.0 if lo == 0 else abs(lo) * 0.1
        return lo - delta, hi + delta
    delta = (hi - lo) * pad
    return lo - delta, hi + delta


def summary_stats(df: pd.DataFrame, stems: Iterable[str]) -> List[str]:
    lines = []
    for stem in stems:
        stem = stem.lower()
        if not has_xyz(df, stem):
            lines.append(f"{stem}: missing")
            continue
        block = xyz(df, stem)
        mag = ensure_3d(df, stem)
        peak = np.nanmax(np.abs(block), axis=0)
        mag_peak = np.nanmax(np.abs(mag))
        nan_pct = 100.0 * np.isnan(block).mean()
        lines.append(
            f"{stem}: peak xyz=({peak[0]:.2f}, {peak[1]:.2f}, {peak[2]:.2f}) units | "
            f"peak 3d={mag_peak:.2f} units | NaN={nan_pct:.1f}%"
        )
    return lines


def safe_png_name(case: Case, idx: int) -> str:
    rel = str(case.rel_case).replace("\\", "/")
    rel = re.sub(r"[^A-Za-z0-9_.-]+", "__", rel).strip("_")
    if not rel:
        rel = case.case_root.name or "case"
    return f"{idx + 1:04d}_{rel}_lip_jaw_qc.png"


def choose_main_df(data: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    # gfm_displacement.csv contains source stems plus derived jaw_mid/imll in the
    # batch output. imll_displacement.csv is the fallback focused file.
    if "gfm" in data:
        return "gfm_displacement.csv", data["gfm"]
    if "imll" in data:
        return "imll_displacement.csv", data["imll"]
    raise FileNotFoundError("No readable displacement dataframe for case.")


def plot_case(
    case: Case,
    idx: int,
    n_cases: int,
    stems: Sequence[str],
    fs: Optional[float] = None,
    fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    data = load_case(case)
    source_name, main_df = choose_main_df(data)
    t = get_time(main_df, fs=fs)

    if fig is None:
        fig = plt.figure(figsize=(14, 9))
    fig.clear()

    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.0, 0.8])
    ax_imll = fig.add_subplot(gs[0, 0])
    ax_jaw = fig.add_subplot(gs[1, 0], sharex=ax_imll)
    ax_sources = fig.add_subplot(gs[2, 0], sharex=ax_imll)
    ax_text = fig.add_subplot(gs[3, 0])
    ax_text.axis("off")

    missing = [s.lower() for s in stems if not has_xyz(main_df, s)]

    # Panel 1: isolated middle lower lip.
    if has_xyz(main_df, "imll"):
        block = xyz(main_df, "imll")
        mag = ensure_3d(main_df, "imll")
        for j, ax_name in enumerate(AXES):
            ax_imll.plot(t, block[:, j], label=f"imll_{ax_name}", linewidth=1.3)
        ax_imll.plot(t, mag, label="imll_3d", linewidth=1.6, alpha=0.9)
        lim = robust_limits([block[:, 0], block[:, 1], block[:, 2], mag])
        if lim:
            ax_imll.set_ylim(*lim)
    else:
        ax_imll.text(0.5, 0.5, "imll missing", transform=ax_imll.transAxes, ha="center", va="center")
    ax_imll.set_ylabel("IMLL disp (units)")
    ax_imll.legend(loc="upper right", ncols=4, fontsize=8)
    ax_imll.grid(True, alpha=0.35)

    # Panel 2: jaw midpoint and jaw landmarks, y/z for sanity checks.
    jaw_values = []
    for stem in ("jaw_mid", "lj", "rj"):
        if has_xyz(main_df, stem):
            block = xyz(main_df, stem)
            ax_jaw.plot(t, block[:, 1], label=f"{stem}_y", linewidth=1.1)
            ax_jaw.plot(t, block[:, 2], label=f"{stem}_z", linewidth=1.1, alpha=0.8)
            jaw_values.extend([block[:, 1], block[:, 2]])
    if not jaw_values:
        ax_jaw.text(0.5, 0.5, "jaw_mid / lj / rj missing", transform=ax_jaw.transAxes, ha="center", va="center")
    else:
        lim = robust_limits(jaw_values)
        if lim:
            ax_jaw.set_ylim(*lim)
    ax_jaw.set_ylabel("Jaw disp (units)")
    ax_jaw.legend(loc="upper right", ncols=3, fontsize=8)
    ax_jaw.grid(True, alpha=0.35)

    # Panel 3: source lower lip vs jaw_mid comparison.
    source_values = []
    for stem in ("mll", "jaw_mid", "imll"):
        if has_xyz(main_df, stem):
            block = xyz(main_df, stem)
            ax_sources.plot(t, block[:, 1], label=f"{stem}_y", linewidth=1.2)
            source_values.append(block[:, 1])
    if not source_values:
        ax_sources.text(0.5, 0.5, "mll / jaw_mid / imll unavailable", transform=ax_sources.transAxes, ha="center", va="center")
    else:
        lim = robust_limits(source_values)
        if lim:
            ax_sources.set_ylim(*lim)
    ax_sources.set_ylabel("Y disp (units)")
    ax_sources.set_xlabel("Time (s)" if "time_s" in main_df.columns or "time" in main_df.columns else "Frame")
    ax_sources.legend(loc="upper right", ncols=3, fontsize=8)
    ax_sources.grid(True, alpha=0.35)

    fig.suptitle(f"[{idx + 1}/{n_cases}] {case.label}\n{case.time_series_dir}", fontsize=12, y=0.98)

    stats = []
    stats.append("Controls: n/right/space=next | p/left=previous | g=jump | s=save png | q/esc=quit | h=help")
    stats.append(f"Source: {source_name} | gfm={'yes' if case.gfm_file else 'no'} | imll={'yes' if case.imll_file else 'no'}")
    stats.append(f"Case root: {case.case_root}")
    if missing:
        stats.append("Missing requested stems: " + ", ".join(missing))
    stats.extend(summary_stats(main_df, stems))

    ax_text.text(
        0.01,
        0.98,
        "\n".join(stats),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=ax_text.transAxes,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


class Viewer:
    def __init__(self, cases: List[Case], stems: Sequence[str], fs: Optional[float], save_dir: Optional[Path]):
        self.cases = cases
        self.stems = [s.lower() for s in stems]
        self.fs = fs
        self.save_dir = save_dir
        self.idx = 0
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    def redraw(self) -> None:
        plot_case(self.cases[self.idx], self.idx, len(self.cases), self.stems, fs=self.fs, fig=self.fig)
        self.fig.canvas.draw_idle()

    def save_current(self) -> Path:
        if self.save_dir is None:
            self.save_dir = self.cases[self.idx].output_root / "qc_pngs"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        out = self.save_dir / safe_png_name(self.cases[self.idx], self.idx)
        self.fig.savefig(out, dpi=160)
        print(f"[saved] {out}")
        return out

    def jump(self) -> None:
        try:
            raw = input(f"Jump to case 1-{len(self.cases)}: ").strip()
            if not raw:
                return
            target = int(raw) - 1
            self.idx = max(0, min(len(self.cases) - 1, target))
            self.redraw()
        except Exception as e:
            print(f"[warn] jump failed: {e}")

    def on_key(self, event) -> None:
        key = event.key
        if key in ("n", "right", " ", "space"):
            self.idx = min(len(self.cases) - 1, self.idx + 1)
            self.redraw()
        elif key in ("p", "left"):
            self.idx = max(0, self.idx - 1)
            self.redraw()
        elif key in ("q", "escape"):
            plt.close(self.fig)
        elif key == "s":
            self.save_current()
        elif key == "g":
            self.jump()
        elif key == "h":
            print("Controls: n/right/space=next | p/left=previous | g=jump | s=save png | q/esc=quit | h=help")


def save_all_png(cases: List[Case], stems: Sequence[str], fs: Optional[float], outdir: Path) -> None:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    for i, case in enumerate(cases):
        try:
            plot_case(case, i, len(cases), stems, fs=fs, fig=fig)
            out = outdir / safe_png_name(case, i)
            fig.savefig(out, dpi=160)
            print(f"[saved] {out}")
        except Exception as e:
            print(f"[warn] failed {case.time_series_dir}: {e}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive QC viewer for batch_gfm_imll_timeseries displacement outputs.")
    ap.add_argument("--root", required=True, help="Output root from batch_gfm_imll_timeseries*.py.")
    ap.add_argument("--fs", type=float, default=None, help="Sampling rate if no time_s column exists.")
    ap.add_argument(
        "--stems",
        nargs="+",
        default=list(DEFAULT_STEMS),
        help="Stems to summarize. Default: imll jaw_mid mll lj rj",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based starting case index for interactive review.",
    )
    ap.add_argument(
        "--save-dir",
        default=None,
        help="Where 's' saves current QC PNG. Defaults to <root>/qc_pngs.",
    )
    ap.add_argument(
        "--save-all-png",
        default=None,
        help="Non-interactive mode: save one PNG per case into this directory and exit.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    cases = find_cases(root)
    if not cases:
        raise SystemExit(
            "No batch output cases found. Expected files like:\n"
            f"  {root}/<relative_input_stem>/time_series/gfm_displacement.csv\n"
            f"  {root}/<relative_input_stem>/time_series/imll_displacement.csv"
        )

    print(f"[info] found {len(cases)} cases under {root}")
    for i, case in enumerate(cases[:10], start=1):
        print(f"  {i:04d}: {case.label}")
    if len(cases) > 10:
        print(f"  ... {len(cases) - 10} more")

    stems = [s.lower() for s in args.stems]
    if args.save_all_png:
        save_all_png(cases, stems, args.fs, Path(args.save_all_png))
        return

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    viewer = Viewer(cases, stems, args.fs, save_dir)
    viewer.idx = max(0, min(len(cases) - 1, args.start - 1))
    viewer.redraw()
    plt.show()


if __name__ == "__main__":
    main()
