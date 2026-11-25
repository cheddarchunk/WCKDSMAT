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
    re.compile(r"_[a-z]{2,4}_(0[1-9]|1[0-2])([0-2][0-9]|3[01])[0-9]{2}_"),  # recording names: task_subject_MMDDYY_
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


def test_every_source_file_has_license_header():
    missing = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*")
               if p.suffix in {".py", ".js", ".css", ".toml"}
               and "SPDX-License-Identifier: AGPL-3.0-or-later" not in p.read_text()[:300]]
    assert not missing, f"add the SPDX header to: {missing}"
