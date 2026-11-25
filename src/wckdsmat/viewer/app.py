"""Speech Movement Data Viewer (Dash app).

Movement (X, Y, Z of each selected landmark) and a wide- or narrowband
spectrogram share one time axis. Drop event flags, label the intervals
between neighbouring flags, save a segment table, or segment this file now.

Run with ``wckdsmat view RECORDING.csv --video VIDEO`` (or ``--audio WAV``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from flask import send_file
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from ..config import Config
from .annotations import Annotations, upsert_segment_table
from .data import SPACE_LABELS, auto_audio_offset, extract_audio, load_recording, load_wav, spectrogram_window
from .export import segment_file

ASSETS = Path(__file__).parent / "assets"
COLORS = qualitative.D3 + qualitative.Set2
FLAG_COLOR, SEG_FILL = "#c0392b", "rgba(41, 128, 185, 0.13)"


class Session:
    """Everything the app needs for one recording."""

    def __init__(self, recording, cfg: Config, *, audio=None, video=None, out_dir=None, segments_table=None,
                 audio_offset="auto"):
        self.rec = load_recording(recording)
        self.cfg = cfg
        self.out_dir = Path(out_dir or Path(recording).parent / "annotations")
        self.ann_path = self.out_dir / f"{self.rec.name}_annotations.json"
        self.segments_table = Path(segments_table) if segments_table else None
        self.ann = Annotations.load(self.ann_path, self.rec.name)
        self.wav = None
        if audio:
            self.wav = Path(audio)
        elif video:
            self.wav = extract_audio(video, self.out_dir / ".audio" / f"{self.rec.name}.wav")
        self.sr, self.x = load_wav(self.wav) if self.wav else (0, np.zeros(0, np.float32))
        self.audio_offset = auto_audio_offset(self.rec) if audio_offset == "auto" else float(audio_offset)

    def save(self):
        return self.ann.save(self.ann_path)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def build_figure(s: Session, names, space, center, band, max_freq, view):
    rec = s.rec
    has_audio = s.sr > 0
    rows = 4 if has_audio else 3
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.2, 0.2, 0.2, 0.4][:rows] if has_audio else [1 / 3] * 3)
    for i, name in enumerate(names or []):
        X = rec.points.get((name, space))
        if X is None:
            continue
        color = COLORS[i % len(COLORS)]
        for k, ax in enumerate("XYZ"):
            y = X[:, k] - np.nanmedian(X[:, k]) if center else X[:, k]
            fig.add_trace(go.Scattergl(x=rec.t, y=y, mode="lines", name=name, legendgroup=name,
                                       showlegend=(k == 0), line=dict(color=color, width=1.4),
                                       hovertemplate=f"{name} {ax}: %{{y:.2f}}<br>t = %{{x:.3f}} s<extra></extra>"),
                          row=k + 1, col=1)
    for k, ax in enumerate("XYZ"):
        fig.update_yaxes(title_text=f"{ax}{' (centered)' if center else ''}", row=k + 1, col=1)

    if has_audio:
        t0, t1 = view if view else (float(np.nanmin(rec.t)), float(np.nanmax(rec.t)))
        off = s.audio_offset
        tt, f, db = spectrogram_window(s.x, s.sr, t0 - off, t1 - off, band=band, max_freq=max_freq)
        if len(tt):
            fig.add_trace(go.Heatmap(x=tt + off, y=f, z=db, colorscale="Greys", showscale=False,
                                     hovertemplate="t = %{x:.3f} s<br>%{y:.0f} Hz<extra></extra>"), row=rows, col=1)
        fig.update_yaxes(title_text=f"Hz ({band}band)", range=[0, max_freq], row=rows, col=1)
    fig.update_xaxes(title_text="time (s)", row=rows, col=1)

    shapes, notes = [], []
    for f in s.ann.flags:                                     # flags first: shape index = flag order
        shapes.append(dict(type="line", xref="x", yref="paper", x0=f.t, x1=f.t, y0=0, y1=1, editable=True,
                           line=dict(color=FLAG_COLOR, width=1.5, dash="dot")))
        if f.label:
            notes.append(dict(x=f.t, y=1.0, xref="x", yref="paper", text=f.label, showarrow=False,
                              yanchor="bottom", font=dict(size=11, color=FLAG_COLOR)))
    for r in s.ann.interval_rows():
        if r["label"] or r["task"]:
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=r["start_s"], x1=r["end_s"], y0=0, y1=1,
                               fillcolor=SEG_FILL, line=dict(width=0), layer="below", editable=False))
            notes.append(dict(x=(r["start_s"] + r["end_s"]) / 2, y=0.995, xref="x", yref="paper",
                              text=" · ".join(v for v in (r["label"], r["task"]) if v), showarrow=False,
                              yanchor="top", font=dict(size=11, color="#1f4e79")))
    fig.update_layout(shapes=shapes, annotations=notes + list(fig.layout.annotations), uirevision="keep",
                      margin=dict(l=70, r=20, t=30, b=40), hovermode="x unified", dragmode="zoom",
                      legend=dict(orientation="h", y=1.04, x=0), plot_bgcolor="white",
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12))
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    return fig


def _x_range(relayout):
    if not relayout:
        return None
    for key in list(relayout):
        if key.startswith("xaxis") and key.endswith(".range[0]"):
            ax = key.split(".")[0]
            return [float(relayout[key]), float(relayout[f"{ax}.range[1]"])]
        if key.startswith("xaxis") and key.endswith(".autorange"):
            return "auto"
    return None


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
def create_app(s: Session) -> Dash:
    app = Dash(__name__, assets_folder=str(ASSETS), title="Speech Movement Data Viewer")
    server = app.server

    @server.route("/media/audio.wav")
    def _audio():
        return send_file(str(s.wav), mimetype="audio/wav", conditional=True)

    rec = s.rec
    space0 = rec.spaces[0]
    preferred = [n.upper() for n in s.cfg.resolve("speech_landmarks")] if "speech_landmarks" in s.cfg.groups else []
    names0 = [n for n in preferred if n in rec.names(space0)][:4] or rec.names(space0)[:3]

    def label(text, child):
        return html.Label([html.Span(text, className="lbl"), child], className="ctl")

    app.layout = html.Div(className="smv", children=[
        html.Header([html.H1("Speech Movement Data Viewer"),
                     html.Div(f"{rec.name} · {rec.path.name} · {rec.fs:.2f} Hz · {len(rec.t)} frames"
                              + (f" · audio {s.sr} Hz, offset {s.audio_offset * 1000:.1f} ms" if s.sr else " · no audio"),
                              className="sub")]),
        html.Div(className="row", children=[
            label("Landmarks", dcc.Dropdown(id="landmarks", options=rec.names(space0), value=names0, multi=True,
                                            style={"minWidth": "320px"})),
            label("Space", dcc.RadioItems(id="space", value=space0, inline=True,
                                          options=[{"label": SPACE_LABELS.get(sp, sp), "value": sp} for sp in rec.spaces])),
            dcc.Checklist(id="center", options=[{"label": "center traces", "value": "c"}], value=["c"], inline=True),
            label("Spectrogram", dcc.RadioItems(id="band", value="wide", inline=True, options=[
                {"label": "wideband (5 ms)", "value": "wide"}, {"label": "narrowband (30 ms)", "value": "narrow"}])),
            label("up to", dcc.Dropdown(id="maxfreq", value=5000, clearable=False, style={"width": "110px"},
                                        options=[{"label": f"{v // 1000} kHz", "value": v} for v in (4000, 5000, 8000)])),
        ]),
        html.Div(className="row transport", children=[
            html.Audio(id="audio", src="/media/audio.wav" if s.sr else None, controls=True, preload="auto"),
            html.Button("▶ Play view", id="play-view", className="btn"),
            label("Click on plot to", dcc.RadioItems(id="click-mode", value="seek", inline=True, options=[
                {"label": "seek", "value": "seek"}, {"label": "add flag", "value": "flag"}])),
            dcc.Input(id="flag-label", placeholder="flag label (optional)", debounce=True),
            html.Button("⚑ Flag at playhead  [F]", id="flag-btn", className="btn primary"),
            dcc.Input(id="default-task", placeholder="task for new segments, e.g. ddk", debounce=True),
        ]),
        html.Div(id="graph-wrap", children=[
            dcc.Graph(id="graph", style={"height": "780px"},
                      config={"scrollZoom": True, "displaylogo": False, "edits": {"shapePosition": True}}),
            html.Div(id="playhead"),
        ]),
        html.Div(className="tables", children=[
            html.Div([html.H3("Event flags"),
                      html.P("Edit a time or label in place; delete with ×. Drag a flag on the plot to move it.",
                             className="hint"),
                      dash_table.DataTable(id="flags", row_deletable=True, editable=True, page_size=12,
                                           columns=[{"name": "time (s)", "id": "t", "type": "numeric",
                                                     "format": {"specifier": ".3f"}},
                                                    {"name": "label", "id": "label"},
                                                    {"name": "id", "id": "id", "editable": False}],
                                           hidden_columns=["id"], css=[{"selector": ".show-hide", "rule": "display:none"}],
                                           style_cell={"fontFamily": "inherit", "padding": "4px 8px"})]),
            html.Div([html.H3("Segments between flags"),
                      html.P("Type a label (and task) to make an interval a segment; clear both to remove it.",
                             className="hint"),
                      dash_table.DataTable(id="segments", editable=True, page_size=12,
                                           columns=[{"name": "start (s)", "id": "start_s", "editable": False,
                                                     "type": "numeric", "format": {"specifier": ".3f"}},
                                                    {"name": "end (s)", "id": "end_s", "editable": False,
                                                     "type": "numeric", "format": {"specifier": ".3f"}},
                                                    {"name": "dur (s)", "id": "duration_s", "editable": False,
                                                     "type": "numeric", "format": {"specifier": ".3f"}},
                                                    {"name": "label", "id": "label"}, {"name": "task", "id": "task"},
                                                    {"name": "key", "id": "key", "editable": False}],
                                           hidden_columns=["key"], css=[{"selector": ".show-hide", "rule": "display:none"}],
                                           style_cell={"fontFamily": "inherit", "padding": "4px 8px"},
                                           style_data_conditional=[{"if": {"filter_query": "{label} != ''"},
                                                                    "backgroundColor": "#eaf2fb"}])]),
        ]),
        html.Div(className="row actions", children=[
            html.Button("Save segment table", id="save-table", className="btn primary"),
            html.Button("Segment this file now", id="segment-now", className="btn"),
            html.Span(f"annotations autosave to {s.ann_path}", className="hint"),
        ]),
        html.Div(id="status", className="status"),
        dcc.Store(id="view"), dcc.Store(id="playhead-t"), dcc.Store(id="ann-version", data=0),
        html.Div(id="audio-offset", children=str(s.audio_offset), style={"display": "none"}),
        html.Div(id="seek-sink", style={"display": "none"}), html.Div(id="play-sink", style={"display": "none"}),
    ])

    # space change -> landmark options
    @app.callback(Output("landmarks", "options"), Output("landmarks", "value"), Input("space", "value"),
                  State("landmarks", "value"))
    def _space(space, current):
        opts = rec.names(space)
        keep = [n for n in (current or []) if n in opts]
        return opts, keep or opts[:3]

    @app.callback(Output("view", "data"), Input("graph", "relayoutData"), State("view", "data"),
                  prevent_initial_call=True)
    def _view(relayout, current):
        r = _x_range(relayout)
        if r is None:
            return no_update
        return None if r == "auto" else r

    @app.callback(Output("graph", "figure"),
                  Input("landmarks", "value"), Input("space", "value"), Input("center", "value"),
                  Input("band", "value"), Input("maxfreq", "value"), Input("view", "data"),
                  Input("ann-version", "data"))
    def _figure(names, space, center, band, maxfreq, view, _v):
        return build_figure(s, names, space, bool(center), band, maxfreq, view)

    # every annotation change goes through here; the JSON file is rewritten each time
    @app.callback(Output("ann-version", "data"), Output("flags", "data"), Output("segments", "data"),
                  Output("status", "children", allow_duplicate=True),
                  Input("playhead-t", "data"), Input("graph", "clickData"), Input("graph", "relayoutData"),
                  Input("flags", "data_timestamp"), Input("segments", "data_timestamp"),
                  State("flags", "data"), State("segments", "data"), State("click-mode", "value"),
                  State("flag-label", "value"), State("default-task", "value"), State("ann-version", "data"),
                  prevent_initial_call="initial_duplicate")
    def _annotate(ph, click, relayout, _ft, _st, flag_rows, seg_rows, mode, flag_label, task, version):
        trig = ctx.triggered_id
        a = s.ann
        msg = no_update
        if trig == "playhead-t" and ph:
            a.add_flag(ph["t"], flag_label or "")
            msg = f"flag at {ph['t']:.3f} s"
        elif trig == "graph" and ctx.triggered[0]["prop_id"].endswith("clickData"):
            if mode != "flag" or not click:
                return no_update, no_update, no_update, no_update
            t = float(click["points"][0]["x"])
            a.add_flag(t, flag_label or "")
            msg = f"flag at {t:.3f} s"
        elif trig == "graph":                                  # a dragged flag
            moved = {}
            for key, val in (relayout or {}).items():
                if key.startswith("shapes[") and key.endswith("].x0"):
                    moved[int(key[7:key.index("]")])] = float(val)
            if not moved:
                return no_update, no_update, no_update, no_update
            ids = [f.id for f in a.flags]
            for i, t in moved.items():
                if i < len(ids):
                    a.move_flag(ids[i], t)
            msg = "flag moved"
        elif trig == "flags":
            rows = {r["id"]: r for r in (flag_rows or [])}
            for f in list(a.flags):
                if f.id not in rows:
                    a.delete_flag(f.id)
                else:
                    r = rows[f.id]
                    a.relabel_flag(f.id, str(r.get("label") or ""))
                    try:
                        if float(r["t"]) != f.t:
                            a.move_flag(f.id, float(r["t"]))
                    except (TypeError, ValueError):
                        msg = "time must be a number"
        elif trig == "segments":
            new_task = task or ""
            for r in seg_rows or []:
                start, end = r["key"].split("|")
                lab, tsk = str(r.get("label") or ""), str(r.get("task") or "")
                if lab and not tsk and new_task:
                    tsk = new_task
                a.set_segment(start, end, lab, tsk)
        if trig is not None:
            s.save()
        flags = [{"id": f.id, "t": round(f.t, 4), "label": f.label} for f in a.flags]
        segs = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in a.interval_rows()]
        return (version or 0) + 1, flags, segs, msg

    @app.callback(Output("status", "children"), Input("save-table", "n_clicks"), Input("segment-now", "n_clicks"),
                  prevent_initial_call=True)
    def _export(_a, _b):
        table = s.ann.segment_table()
        if table.empty:
            return "No labeled segments yet: label at least one interval between flags."
        s.out_dir.mkdir(parents=True, exist_ok=True)
        if ctx.triggered_id == "save-table":
            path = s.out_dir / f"{rec.name}_segments.csv"
            table.to_csv(path, index=False)
            s.ann.event_table().to_csv(s.out_dir / f"{rec.name}_events.csv", index=False)
            extra = ""
            if s.segments_table:
                upsert_segment_table(s.segments_table, table, rec.name)
                extra = f" and updated {s.segments_table}"
            return f"Saved {len(table)} segment(s) to {path}{extra}."
        res = segment_file(rec.path, table, s.out_dir, s.cfg, rec.name)
        kin = res["kinematics"]
        tail = (f"; kinematics for {kin['n_segments']} segment(s) in {kin['dir']}" if kin
                else "; no head-normalized points in this file, so kinematics was skipped")
        return f"Wrote {len(res['cut_files'])} segment file(s) to {res['out_dir']}/segments{tail}."

    # browser-side: seek, flag-at-playhead time, play the visible window
    app.clientside_callback(
        """function(click, mode){
            if(!click || mode !== 'seek') return window.dash_clientside.no_update;
            const a = document.getElementById('audio'); if(!a) return '';
            const off = parseFloat(document.getElementById('audio-offset').textContent) || 0;
            a.currentTime = Math.max(0, click.points[0].x - off); return '';
        }""", Output("seek-sink", "children"), Input("graph", "clickData"), State("click-mode", "value"))
    app.clientside_callback(
        """function(n){
            if(!n) return window.dash_clientside.no_update;
            const a = document.getElementById('audio');
            const off = parseFloat(document.getElementById('audio-offset').textContent) || 0;
            return {t: (a ? a.currentTime : 0) + off, n: n};
        }""", Output("playhead-t", "data"), Input("flag-btn", "n_clicks"))
    app.clientside_callback(
        """function(n, view){
            if(!n) return window.dash_clientside.no_update;
            const a = document.getElementById('audio'); if(!a) return '';
            const off = parseFloat(document.getElementById('audio-offset').textContent) || 0;
            const gd = document.querySelector('#graph .js-plotly-plot');
            const r = view || (gd && gd._fullLayout ? gd._fullLayout.xaxis.range : null);
            if(!r) return '';
            a.currentTime = Math.max(0, r[0] - off); window.smvStopAt = r[1] - off; a.play(); return '';
        }""", Output("play-sink", "children"), Input("play-view", "n_clicks"), State("view", "data"))
    return app
