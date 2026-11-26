"""The documentation's walkthrough must actually work, and its links must resolve."""
import importlib.util
import re
from pathlib import Path

import pandas as pd

from wckdsmat.cli import main

ROOT = Path(__file__).resolve().parents[1]


def slug(heading: str) -> str:
    """GitHub's anchor for a markdown heading."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def anchors(md: Path) -> set[str]:
    return {slug(m.group(1)) for m in re.finditer(r"^#{1,6} (.+)$", md.read_text(), re.M)}


def test_markdown_links_resolve():
    for md in (ROOT / "README.md", ROOT / "HOWTOUSE.md"):
        text = md.read_text()
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith("http"):
                continue
            path, _, frag = target.partition("#")
            dest = (md.parent / path) if path else md
            assert dest.exists(), f"{md.name}: broken link {target}"
            if frag and dest.suffix == ".md":
                assert frag in anchors(dest), f"{md.name}: no heading for #{frag} in {dest.name}"


def test_demo_walkthrough(tmp_path, capsys):
    """HOWTOUSE section 2: make the demo, calibrate, preprocess, then kinematics and the viewer session."""
    spec = importlib.util.spec_from_file_location("make_demo", ROOT / "examples" / "make_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    demo.main(str(tmp_path))
    d = tmp_path
    assert main(["calibrate", str(d / "DEMO_ddk_gfm_sp_t.csv"), "--eye-mm", "95", "--out-dir", str(d)]) == 0
    assert main(["preprocess", str(d / "DEMO_ddk_gfm_sp_t_cal.csv"), "--out-dir", str(d)]) == 0
    rec = d / "DEMO_ddk_gfm_sp_t_cal_lp_hn.csv"
    pre = pd.read_csv(rec)
    jaw = pre["MJ_hn_Y"].max() - pre["MJ_hn_Y"].min()
    assert 7 < jaw < 12, jaw                                   # the demo jaw opens ~9 mm
    pd.DataFrame({"recording": ["DEMO_ddk"], "segment_id": ["1"], "task": ["ddk"],
                  "start_s": [3.0], "end_s": [6.0]}).to_csv(d / "segments.csv", index=False)
    assert main(["kinematics", str(rec), "--segments", str(d / "segments.csv"), "--out-dir", str(d / "k")]) == 0
    vel = pd.read_csv(d / "k" / "DEMO_ddk_gfm_sp_t_cal_lp_hn_velocity.csv")
    assert vel["MJ_hn_3D"].max() > 50                         # mm/s: 9 mm opening at 5 syllables/s
    try:
        from wckdsmat.viewer.app import Session
    except ImportError:
        return
    from wckdsmat.config import load_config
    s = Session(rec, load_config(), audio=d / "DEMO_ddk.wav", out_dir=d / "ann")
    assert s.rec.name == "DEMO_ddk" and s.sr == 16000 and abs(s.audio_offset - 1 / 30) < 1e-9


def test_version_is_consistent_everywhere(capsys):
    import pytest

    import wckdsmat
    v = wckdsmat.__version__
    cff = (ROOT / "CITATION.cff").read_text()
    assert f"\nversion: {v}\n" in cff, "CITATION.cff version differs from wckdsmat.__version__"
    assert f"## {v} " in (ROOT / "CHANGELOG.md").read_text(), "CHANGELOG.md has no section for this version"
    assert 'dynamic = ["version"]' in (ROOT / "pyproject.toml").read_text()
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"wckdsmat {v}"
