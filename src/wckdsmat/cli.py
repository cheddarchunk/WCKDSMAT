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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wckdsmat")
    sub = ap.add_subparsers(dest="command", required=True)

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
