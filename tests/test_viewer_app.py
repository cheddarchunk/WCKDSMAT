"""Viewer app layer: figure, session, audio serving (synthetic data, no browser)."""
import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("dash")

from test_viewer_core import preprocess_like, tone  # noqa: E402

from wckdsmat.config import load_config  # noqa: E402
from wckdsmat.viewer.app import Session, _x_range, build_figure, create_app  # noqa: E402


@pytest.fixture
def session(tmp_path):
    rec = preprocess_like(tmp_path)
    wav = tmp_path / "a.wav"
    wavfile.write(wav, 16000, (tone(dur=10.0) * 8000).astype(np.int16))
    return Session(rec, load_config(), audio=wav, out_dir=tmp_path / "ann")


def test_figure_has_three_axes_per_landmark_and_spectrogram(session):
    fig = build_figure(session, ["MLL", "LJ"], "hn", True, "wide", 5000, None)
    kinds = [t.type for t in fig.data]
    assert kinds.count("scattergl") == 6 and kinds.count("heatmap") == 1
    assert {t.name for t in fig.data if t.type == "scattergl"} == {"MLL", "LJ"}
    assert sum(t.showlegend for t in fig.data if t.type == "scattergl") == 2   # one legend entry per landmark
    assert fig.layout.uirevision == "keep"                                    # zoom survives redraws


def test_flags_and_segments_drawn(session):
    a = session.ann
    i1, i2, i3 = a.add_flag(1.0, "on"), a.add_flag(2.0), a.add_flag(3.0)
    a.set_segment(i1, i2, "pa", "ddk")
    fig = build_figure(session, ["MLL"], "hn", False, "narrow", 8000, [0.5, 3.5])
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert [s.x0 for s in lines] == [1.0, 2.0, 3.0] and fig.layout.shapes[0].type == "line"   # flags first
    assert len(rects) == 1 and (rects[0].x0, rects[0].x1) == (1.0, 2.0)
    texts = {n.text for n in fig.layout.annotations}
    assert "on" in texts and "pa · ddk" in texts


def test_spectrogram_follows_view_and_audio_offset(session):
    session.audio_offset = 0.5
    fig = build_figure(session, ["MLL"], "hn", True, "wide", 5000, [2.0, 4.0])
    hm = next(t for t in fig.data if t.type == "heatmap")
    assert hm.x[0] >= 2.0 - 1e-6 and hm.x[-1] <= 4.0 + 1e-6


def test_x_range_parsing():
    assert _x_range({"xaxis.range[0]": 1, "xaxis.range[1]": 2}) == [1.0, 2.0]
    assert _x_range({"xaxis3.range[0]": 1, "xaxis3.range[1]": 2}) == [1.0, 2.0]
    assert _x_range({"xaxis.autorange": True}) == "auto"
    assert _x_range({"shapes[0].x0": 3}) is None and _x_range(None) is None


def test_session_resumes_saved_annotations(tmp_path, session):
    session.ann.add_flag(1.5, "x")
    session.save()
    again = Session(session.rec.path, load_config(), audio=session.wav, out_dir=session.out_dir)
    assert [f.t for f in again.ann.flags] == [1.5]


def test_audio_served_with_byte_ranges(session):
    client = create_app(session).server.test_client()
    full = client.get("/media/audio.wav")
    part = client.get("/media/audio.wav", headers={"Range": "bytes=100-199"})
    assert full.status_code == 200 and part.status_code == 206 and len(part.data) == 100


def test_no_audio_gives_movement_only(tmp_path):
    s = Session(preprocess_like(tmp_path), load_config(), out_dir=tmp_path / "ann")
    fig = build_figure(s, ["MLL"], "hn", True, "wide", 5000, None)
    assert [t.type for t in fig.data].count("heatmap") == 0 and s.sr == 0
