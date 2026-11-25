#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-B: End-to-end batch pipeline for per-task validation and Bandini-style metrics.

Stages:
  1. Spatial alignment for all segments (reps)
  2. Validation / time-series error metrics for all reps
  3. Bandini metrics for all reps
  4. Aggregation and summary

Parallelism (NEW):
  --jobs N runs per-rep subprocess work in parallel for stages 1/2/3.
  Stage 4 remains single-threaded.

Notes:
  - Per-rep prints removed to keep logs clean (as requested).
  - Fail-fast: first failing rep raises an error after all running jobs resolve.
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Iterable, Sequence
import re
import os
import pandas as pd


# ---------------------------------------------------------------------
# PATHS TO STAGE SCRIPTS (assume all live in same directory as this file)
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
SPATIAL_ALIGN = THIS_DIR / "spatial_alignment.py"
VALIDATION    = THIS_DIR / "traces.py"
BANDINI       = THIS_DIR / "bandini.py"


# ---------------------------------------------------------------------
# Utilities: segment / rep mapping
# ---------------------------------------------------------------------
SEGMENT_RE = re.compile(r"^(rep\d+|all\d*|messy)$", re.IGNORECASE)

def strip_suffix(stem: str, suffix: str) -> str:
    """Remove suffix only if it appears at the end of the stem."""
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def _parse_subtask_subject_segment_from_key(key: str) -> Tuple[str, str, str]:
    """
    Robust parsing for keys like:
      task_SUBJ001_000000_gfm_sp_t_mm_lp_hn_seg_1_9-10

    We interpret:
      subtask = first token
      subject = second token
      seg_label = tail pattern if present (e.g., btc_162_9-10) else last 3 tokens.
    """
    parts = key.split("_")
    if len(parts) < 2:
        return ("unknown", "unknown", key)

    subtask = parts[0]
    subject = parts[1]

    m = re.search(r"([A-Za-z]+_\d+_\d+-\d+)$", key)
    if m:
        seg_label = m.group(1)
    else:
        seg_label = "_".join(parts[-3:]) if len(parts) >= 3 else key

    return (subtask, subject, seg_label)


def find_pairs(segments_root: Path) -> List[Tuple[str, str, str, Path, Path]]:
    """
    Discover segment pairs (ctx, gfm) in the segments_root.

    Expected filenames:
      *_ctx.csv
      *_gfm.csv

    We pair by shared prefix (drop the trailing '_ctx' / '_gfm').

    Returns: List of (subtask, subject, seg_label, ctx_file, gfm_file)
    """
    segments_root = Path(segments_root)
    ctx_files = sorted(segments_root.glob("*_ctx.csv"))
    gfm_files = sorted(segments_root.glob("*_gfm.csv"))

    ctx_map = {strip_suffix(f.stem, "_ctx"): f for f in ctx_files}
    gfm_map = {strip_suffix(f.stem, "_gfm"): f for f in gfm_files}

    keys = sorted(set(ctx_map) & set(gfm_map))

    pairs: List[Tuple[str, str, str, Path, Path]] = []
    for key in keys:
        subtask, subject, seg_label = _parse_subtask_subject_segment_from_key(key)
        pairs.append((subtask, subject, seg_label, ctx_map[key], gfm_map[key]))

    print(f"[info] found {len(pairs)} segment pairs under: {segments_root}")
    return pairs


# ---------------------------------------------------------------------
# Parallel subprocess dispatcher
# ---------------------------------------------------------------------
def _run_cmd(cmd: Sequence[str]) -> None:
    # No per-rep prints; fail loudly if subprocess fails.
    subprocess.run(list(map(str, cmd)), check=True)


def run_many(cmds: Iterable[Sequence[str]], jobs: int) -> None:
    """
    Run subprocess commands either sequentially (jobs<=1) or in parallel threads (jobs>1).

    Threads are correct here because the heavy work is in subprocesses.
    """
    cmds = list(cmds)
    if not cmds:
        return

    jobs = max(1, int(jobs))

    if jobs <= 1 or len(cmds) == 1:
        for c in cmds:
            _run_cmd(c)
        return

    errors: List[BaseException] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        fut_map = {ex.submit(_run_cmd, c): c for c in cmds}
        for fut in as_completed(fut_map):
            try:
                fut.result()
            except BaseException as e:
                errors.append(e)

    if errors:
        raise errors[0]


# ---------------------------------------------------------------------
# Stage 1: Spatial alignment (parallel per rep)
# ---------------------------------------------------------------------
def run_spatial(pairs, aligned_root: Path, bases: List[str], fit: str, jobs: int):
    """
    Stage 1: Spatial alignment for all reps.

    For each segment pair, calls spatial_alignment.py to produce aligned 30 Hz files in:
      <aligned_root>/<subtask>/<subject>/<seg_label>/
    """
    aligned_root = Path(aligned_root)
    aligned_root.mkdir(parents=True, exist_ok=True)

    cmds = []
    for subtask, subject, seg_label, ctx_in, gfm_in in pairs:
        repdir = aligned_root / subtask / subject / seg_label
        repdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", str(SPATIAL_ALIGN),
            "--ctx", str(ctx_in),
            "--gfm", str(gfm_in),
            "--out", str(repdir),
            "--fit", fit,
            "--bases", *bases,
        ]
        cmds.append(cmd)

    print(f"[stage1] spatial align: {len(cmds)} reps | jobs={jobs}")
    run_many(cmds, jobs=jobs)
    print("[stage1] done")


# ---------------------------------------------------------------------
# Stage 2: Validation / TS generation (parallel per rep)
# ---------------------------------------------------------------------
def run_validation(pairs, aligned_root: Path, metrics_root: Path, jobs: int, hz: int):
    """
    Stage 2: For each segment pair, find aligned CTX + GFM 30 Hz streams and
    run validation_pipeline.py to compute per-segment metrics.

    Expects Stage 1 outputs in:
      <aligned_root>/<subtask>/<subject>/<seg_label>/
        *_ctx_final_tracking_<Hz>Hz.csv
        *_gfm_final_tracking_<Hz>Hz.csv

    Writes metrics to:
      <metrics_root>/<subtask>/<subject>/<seg_label>/
    """
    aligned_root = Path(aligned_root)
    metrics_root = Path(metrics_root)
    metrics_root.mkdir(parents=True, exist_ok=True)

    cmds = []
    skipped = 0

    for subtask, subject, seg_label, _ctx_in, _gfm_in in pairs:
        rep_aligned = aligned_root / subtask / subject / seg_label
        rep_metrics = metrics_root / subtask / subject / seg_label
        rep_metrics.mkdir(parents=True, exist_ok=True)

        ctx_files = sorted(rep_aligned.glob(f"*_ctx_final_tracking_{hz}Hz.csv"))
        gfm_files = sorted(rep_aligned.glob(f"*_gfm_final_tracking_{hz}Hz.csv"))

        if not ctx_files or not gfm_files:
            skipped += 1
            continue

        if len(ctx_files) != 1 or len(gfm_files) != 1:
            raise RuntimeError(f"Ambiguous tracking Hz selection in {rep_aligned}: ctx={len(ctx_files)} gfm={len(gfm_files)} (expected 1 each).")

        ctx_file = ctx_files[0]
        gfm_file = gfm_files[0]

        cmd = [
            "python", str(VALIDATION),
            "--ctx", str(ctx_file),
            "--gfm", str(gfm_file),
            "--out", str(rep_metrics),
        ]
        cmds.append(cmd)

    print(f"[stage2] validation: {len(cmds)} reps (skipped {skipped}) | jobs={jobs}")
    run_many(cmds, jobs=jobs)
    print("[stage2] done")


# ---------------------------------------------------------------------
# Stage 3: Bandini metrics (parallel per rep)
# ---------------------------------------------------------------------
def run_bandini(pairs, metrics_root: Path, jobs: int):
    """
    Stage 3: Run Bandini on each segment's time_series produced in Stage 2.

    Expects Stage 2 outputs in:
      metrics_root/<subtask>/<subject>/<seg_label>/time_series/

    Writes:
      metrics_root/<subtask>/<subject>/<seg_label>/bandini/
    """
    metrics_root = Path(metrics_root)

    cmds = []
    skipped = 0

    for subtask, subject, seg_label, _ctx_in, _gfm_in in pairs:
        tsdir = metrics_root / subtask / subject / seg_label / "time_series"
        outdir = metrics_root / subtask / subject / seg_label / "bandini"
        outdir.mkdir(parents=True, exist_ok=True)

        if not tsdir.exists():
            skipped += 1
            continue

        required = [
            "ctx_position.csv", "gfm_position.csv",
            "ctx_displacement.csv", "gfm_displacement.csv",
            "ctx_velocity.csv", "gfm_velocity.csv",
        ]
        missing = [f for f in required if not (tsdir / f).exists()]
        if missing:
            skipped += 1
            print(f"[stage3][skip] missing time_series files for {subtask}/{subject}/{seg_label}: {missing}")
            continue

        cmd = [
            "python", str(BANDINI),
            "--tsdir",  str(tsdir),
            "--outdir", str(outdir),
        ]
        cmds.append(cmd)

    print(f"[stage3] bandini: {len(cmds)} reps (skipped {skipped}) | jobs={jobs}")
    run_many(cmds, jobs=jobs)
    print("[stage3] done")


# ---------------------------------------------------------------------
# Stage 4: Aggregation and summary (single-threaded)
# ---------------------------------------------------------------------
def aggregate_bandini_split_by_signal(metrics_root: Path, summary_root: Path):
    """
    Aggregate Bandini error_summary.csv across the tree, keeping signal types separate.

    Reads:
      metrics_root/<subtask>/<subject>/<seg_label>/bandini/error_summary.csv

    Writes (for each signal):
      summary_root/bandini_<signal>_all_long.csv
      summary_root/bandini_<signal>_by_stem_axis.csv
      summary_root/bandini_<signal>_by_stem.csv
    """
    metrics_root = Path(metrics_root)
    summary_root = Path(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(metrics_root.rglob("bandini/error_summary.csv"))
    if not csv_paths:
        print(f"[stage4] warn: no bandini/error_summary.csv found under: {metrics_root}")
        return

    dfs = []
    for csv_path in csv_paths:
        rel = csv_path.relative_to(metrics_root)
        parts = rel.parts
        if len(parts) < 5:
            continue

        subtask, subject, seg_label = parts[0], parts[1], parts[2]
        df = pd.read_csv(csv_path)

        required = {"stem", "signal", "axis"}
        if not required.issubset(df.columns):
            continue

        df.insert(0, "subtask", subtask)
        df.insert(1, "subject", subject)
        df.insert(2, "seg_label", seg_label)
        dfs.append(df)

    if not dfs:
        print("[stage4] warn: no valid Bandini summaries loaded.")
        return

    full = pd.concat(dfs, ignore_index=True)

    metric_cols = [
        c for c in full.columns
        if c not in ("subtask", "subject", "seg_label", "stem", "signal", "axis")
        and pd.api.types.is_numeric_dtype(full[c])
    ]
    if not metric_cols:
        print("[stage4] warn: no numeric metric columns found to summarize.")
        return

    signals = sorted(full["signal"].astype(str).str.lower().unique())
    print(f"[stage4] aggregating signals={signals} | files={len(csv_paths)}")

    for sig in signals:
        sub = full[full["signal"].astype(str).str.lower() == sig].copy()
        if sub.empty:
            continue

        long_path = summary_root / f"bandini_{sig}_all_long.csv"
        sub.to_csv(long_path, index=False)

        g_cols = ["stem", "axis"]
        means = sub.groupby(g_cols)[metric_cols].mean().add_suffix("_mean")
        sds   = sub.groupby(g_cols)[metric_cols].std(ddof=1).add_suffix("_sd")
        by_stem_axis = pd.concat([means, sds], axis=1).reset_index()
        (summary_root / f"bandini_{sig}_by_stem_axis.csv").write_text(
            by_stem_axis.to_csv(index=False), encoding="utf-8"
        )

        means2 = sub.groupby(["stem"])[metric_cols].mean().add_suffix("_mean")
        sds2   = sub.groupby(["stem"])[metric_cols].std(ddof=1).add_suffix("_sd")
        by_stem = pd.concat([means2, sds2], axis=1).reset_index()
        (summary_root / f"bandini_{sig}_by_stem.csv").write_text(
            by_stem.to_csv(index=False), encoding="utf-8"
        )

    print(f"[stage4] done → {summary_root}")


# ---------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", required=True, help="Path to segments/<task> directory")
    parser.add_argument("--aligned",  required=True, help="Path to aligned/<task> directory")
    parser.add_argument("--metrics",  required=True, help="Path to metrics/<task> directory")
    parser.add_argument("--summary",  required=True, help="Path to summary/<task> directory")
    parser.add_argument("--bases",    nargs="+", required=True, help="Base head landmarks for rigid fit")
    parser.add_argument("--fit",      default="rigid", help="Fit type for spatial alignment")
    parser.add_argument("--hz", type=int, default=30, help="Sampling rate (Hz) to select from *_final_tracking_<Hz>Hz.csv. Default=30.")

    parser.add_argument("--jobs", type=int, default=3,
                        help="Parallel workers for per-rep subprocesses (stages 1/2/3). Default=1.")

    parser.add_argument(
        "--only-stage",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only a single internal stage: 1=spatial, 2=validation, 3=bandini, 4=aggregation"
    )

    args = parser.parse_args()

    segments_root = Path(args.segments)
    aligned_root  = Path(args.aligned)
    metrics_root  = Path(args.metrics)
    summary_root  = Path(args.summary)
    bases         = args.bases
    fit           = args.fit
    hz            = int(args.hz)
    jobs          = max(1, int(args.jobs))
    only_stage    = args.only_stage

    pairs = find_pairs(segments_root)

    # STAGE 1
    if only_stage is None or only_stage == 1:
        run_spatial(pairs, aligned_root, bases, fit, jobs=jobs)
        if only_stage == 1:
            return

    # STAGE 2
    if only_stage is None or only_stage == 2:
        run_validation(pairs, aligned_root, metrics_root, jobs=jobs, hz=hz)
        if only_stage == 2:
            return

    # STAGE 3
    if only_stage is None or only_stage == 3:
        run_bandini(pairs, metrics_root, jobs=jobs)
        if only_stage == 3:
            return

    # STAGE 4
    if only_stage is None or only_stage == 4:
        aggregate_bandini_split_by_signal(metrics_root, summary_root)
        if only_stage == 4:
            return


if __name__ == "__main__":
    main()
