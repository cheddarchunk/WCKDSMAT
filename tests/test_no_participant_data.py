"""Guardrail: fail if participant data or identifiers ever land in the repo."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {".mov", ".mp4", ".avi", ".mkv", ".wav", ".m4a", ".mp3", ".c3d"}
ID_PATTERNS = [
    re.compile(r"POLO[-_]?\d{3}"),       # POLO participant IDs
    re.compile(r"/Volumes/"),            # external study drives
    re.compile(r"/Users/[^/\s]+/"),      # personal absolute paths
]


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [ROOT / f for f in out]


def test_no_media_files():
    bad = [str(p.relative_to(ROOT)) for p in tracked_files() if p.suffix.lower() in MEDIA]
    assert not bad, f"media files must not be committed: {bad}"


def test_no_data_outside_fixtures():
    bad = [
        str(p.relative_to(ROOT)) for p in tracked_files()
        if p.suffix.lower() in {".csv", ".xlsx", ".npz"} and "tests/fixtures" not in p.as_posix()
    ]
    assert not bad, f"data files outside tests/fixtures: {bad}"


def test_no_identifiers_in_text():
    hits = []
    for p in tracked_files():
        if p.suffix not in {".py", ".md", ".toml", ".txt", ".cfg", ".yml", ".yaml"} or p == Path(__file__):
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if any(rx.search(line) for rx in ID_PATTERNS):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert not hits, f"identifiers or personal paths found: {hits}"
