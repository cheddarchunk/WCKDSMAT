"""Command-line entry point: ``wckdsmat <command>``.

Stage commands (extract, calibrate, ...) will be added here as each stage is
refactored to read a :class:`~wckdsmat.config.Config`.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from .config import ConfigError, available_presets, load_config, preset_text


def _config_init(args) -> int:
    out = Path(args.path)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    text = preset_text(args.preset)
    out.write_text(f'preset = "{args.preset}"\n\n' + text)
    print(f"wrote {out} from preset '{args.preset}'")
    return 0


def _config_check(args) -> int:
    try:
        cfg = load_config(args.path)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"OK: {cfg.source}")
    print(f"  {len(cfg.landmarks)} landmarks, {len(cfg.derived)} derived, groups: {', '.join(cfg.groups)}")
    return 0


def _config_show(args) -> int:
    try:
        cfg = load_config(args.path, preset=None if args.path else args.preset)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"# resolved from {cfg.source}")
    print("landmarks:")
    for name, idx in cfg.landmarks.items():
        print(f"  {name:<10} MediaPipe {idx}")
    for d in cfg.derived:
        print(f"  {d.name:<10} {d.op}({', '.join(d.of)})")
    print("groups:")
    for g, members in cfg.groups.items():
        print(f"  {g:<22} {', '.join(members)}")
    for sect in ("extract", "calibrate", "preprocess", "kinematics", "validate"):
        print(f"{sect}:")
        for k, v in getattr(cfg, sect).items():
            print(f"  {k:<26} {v}")
    return 0


def _preprocess(args) -> int:
    from .preprocess import process_many
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    paths = sorted({Path(p) for pattern in args.inputs for p in (glob.glob(pattern) or [pattern])
                    if Path(p).is_file()})
    if not paths:
        print("no input files matched", file=sys.stderr)
        return 1
    print(f"preprocess: {len(paths)} file(s) -> {args.out_dir}  (config: {cfg.source})")
    failed = 0
    for path, rep, err in process_many(paths, args.out_dir, cfg, jobs=args.jobs):
        if err:
            failed += 1
            print(f"  [fail] {path.name}: {err}")
        else:
            notes = []
            if len(rep.segments) > 1:
                notes.append(f"{len(rep.segments)} segments")
            if rep.frames_without_head_pose:
                notes.append(f"{rep.frames_without_head_pose} frames without head pose")
            if rep.ref_frame_used != rep.ref_frame_requested:
                notes.append(f"ref frame {rep.ref_frame_used}")
            if rep.lowpass_hz_effective < float(cfg.preprocess.get("lowpass_hz", 10.0)):
                notes.append(f"cutoff clamped to {rep.lowpass_hz_effective:.2f} Hz")
            print(f"  [ok] {path.name}  fs={rep.fs_hz:.2f} Hz" + (f"  ({'; '.join(notes)})" if notes else ""))
    print(f"done: {len(paths) - failed} ok, {failed} failed")
    return 1 if failed else 0


def _expand(patterns) -> list[Path]:
    return sorted({Path(p) for pattern in patterns for p in (glob.glob(pattern) or [pattern]) if Path(p).is_file()})


def _run_each(paths, label, fn) -> int:
    """Run fn(path) -> one-line note for each file; report failures and keep going."""
    if not paths:
        print("no input files matched", file=sys.stderr)
        return 1
    print(f"{label}: {len(paths)} file(s)")
    failed = 0
    for path in paths:
        try:
            note = fn(path)
            print(f"  [ok] {path.name}" + (f"  ({note})" if note else ""))
        except Exception as e:  # noqa: BLE001 - report and continue the batch
            failed += 1
            print(f"  [fail] {path.name}: {type(e).__name__}: {e}")
    print(f"done: {len(paths) - failed} ok, {failed} failed")
    return 1 if failed else 0


def _extract(args) -> int:
    from .extract import extract_video, find_videos
    try:
        overrides = {"extract": {"mirror": True}} if args.mirror else None
        if args.no_video:
            overrides = {"extract": {**(overrides or {}).get("extract", {}), "write_overlay_video": False}}
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1

    def one(path):
        rep = extract_video(path, args.out_dir, cfg)
        missing = f", {rep.frames_without_face} frames without a face" if rep.frames_without_face else ""
        return f"{rep.n_frames} frames at {rep.fps:.2f} fps, {rep.width}x{rep.height}{missing}"

    return _run_each(find_videos(args.inputs), f"extract -> {args.out_dir}", one)


def _calibrate(args) -> int:
    import pandas as pd

    from .calibrate import process_file
    from .segments import value_for_recording
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    eye_table = pd.read_csv(args.eye_table) if args.eye_table else None
    size = tuple(int(v) for v in args.video_size.lower().split("x")) if args.video_size else None

    def one(path):
        eye = args.eye_mm
        if eye_table is not None:
            eye = value_for_recording(eye_table, path.name, "eye_distance_mm")
            if eye is None and not args.allow_face_units:
                raise ValueError("not listed in --eye-table (use --allow-face-units to calibrate in face units)")
        rep = process_file(path, args.out_dir, cfg, eye_distance_mm=eye, video_size=size)
        lo, hi = rep.scale_p5_p95
        return f"{rep.units}, {rep.scale_mode}, scale p5-p95 = {lo / rep.median_scale_per_px:.2f}-{hi / rep.median_scale_per_px:.2f} x median"

    return _run_each(_expand(args.inputs), f"calibrate -> {args.out_dir}", one)


def _kinematics(args) -> int:
    from .kinematics import process_file
    from .segments import load_segments
    try:
        cfg = load_config(args.config)
        segs = load_segments(args.segments) if args.segments else None
    except (ConfigError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    def one(path):
        rep = process_file(path, args.out_dir, cfg, segs)
        notes = [f"{rep.n_segments} segment(s)"]
        if rep.empty_segments:
            notes.append(f"{len(rep.empty_segments)} empty")
        if rep.segments_crossing_gaps:
            notes.append(f"{len(rep.segments_crossing_gaps)} span a recording gap")
        return ", ".join(notes)

    return _run_each(_expand(args.inputs), f"kinematics -> {args.out_dir}", one)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wckdsmat")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="MediaPipe Face Mesh landmarks from videos (pixels)")
    p.add_argument("inputs", nargs="+", help="video files, glob patterns or folders")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", help="config TOML (default: speech preset)")
    p.add_argument("--mirror", action="store_true", help="recordings are mirrored (selfie view)")
    p.add_argument("--no-video", action="store_true", help="skip the overlay video")
    p.set_defaults(func=_extract)

    p = sub.add_parser("calibrate", help="convert extracted landmarks to mm (or face units)")
    p.add_argument("inputs", nargs="+", help="CSV files or glob patterns (*_gfm_sp_t.csv or *_gfm_all_t.csv)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", help="config TOML (default: speech preset)")
    eye = p.add_mutually_exclusive_group()
    eye.add_argument("--eye-mm", type=float, help="outer-canthal distance in mm, applied to every input")
    eye.add_argument("--eye-table", help="CSV with recording,eye_distance_mm (one row per participant/recording)")
    p.add_argument("--allow-face-units", action="store_true", help="with --eye-table: face units for unlisted files")
    p.add_argument("--video-size", help="WxH, only for xy_units = normalized01")
    p.set_defaults(func=_calibrate)

    p = sub.add_parser("kinematics", help="per-segment displacement, velocity, acceleration")
    p.add_argument("inputs", nargs="+", help="preprocess outputs (*_lp_hn.csv)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", help="config TOML (default: speech preset)")
    p.add_argument("--segments", help="segment table CSV (recording,segment_id,task,start_s,end_s); "
                                      "omit to treat each continuous piece as one segment")
    p.set_defaults(func=_kinematics)

    p = sub.add_parser("preprocess", help="head-normalize and low-pass calibrated landmark CSVs")
    p.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", help="config TOML (default: speech preset)")
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=_preprocess)

    cfg = sub.add_parser("config", help="create, check or show a pipeline config")
    csub = cfg.add_subparsers(dest="action", required=True)

    p = csub.add_parser("init", help="write a new config file from a preset")
    p.add_argument("path")
    p.add_argument("--preset", default="speech", choices=available_presets())
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_config_init)

    p = csub.add_parser("check", help="validate a config file and report every problem")
    p.add_argument("path")
    p.set_defaults(func=_config_check)

    p = csub.add_parser("show", help="print the fully resolved settings")
    p.add_argument("path", nargs="?")
    p.add_argument("--preset", default="speech", choices=available_presets())
    p.set_defaults(func=_config_show)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
