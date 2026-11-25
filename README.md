# WCKDSMAT

Webcam-based facial kinematics for speech and oral-motor research.

> **Status: pre-release (0.1.0.dev0).** The stage scripts were consolidated from research code and are being refactored into a library with a CLI. The interfaces will change.

## Pipeline

| Stage | Module | Input → Output |
|---|---|---|
| 1. Landmark extraction | `wckdsmat extract` | video → MediaPipe Face Mesh landmarks (pixels); needs `pip install wckdsmat[video]` |
| 2. Calibration (Method G) | `wckdsmat calibrate` | pixels → mm (or face units) |
| 3. Preprocessing | `wckdsmat preprocess` | head normalization, then low-pass; derived points |
| 4. Task segmentation | segment table CSV | analysis windows (`recording,segment_id,task,start_s,end_s`) |
| 5. Kinematics | `wckdsmat kinematics` | per-segment displacement, velocity, acceleration |
| 6. Validation | `wckdsmat validate` | agreement with motion capture (CTX): uRMSE, normalized error, cross-correlation, spectra |

## Install (development)

```bash
pip install -e ".[all,dev]"
```

## Speech Movement Data Viewer

```bash
pip install -e ".[viewer]"        # needs ffmpeg on PATH to read audio from video
wckdsmat view RECORDING_lp_hn.csv --video RECORDING.mov --segments-table segments.csv
```

- **Movement:** X, Y and Z of the selected landmarks in stacked panels, in any space (head-normalized, un-normalized head movement, calibrated, pixels), optionally centered.
- **Acoustics:** a wideband (5 ms, about 300 Hz) or narrowband (30 ms, about 45 Hz) spectrogram on the same time axis, recomputed for the visible window.
- **Playback:** a smooth playhead across all panels. Click to seek, *Play view* plays the visible window, **Space** plays or pauses, and **←/→** step 0.1 s. Audio is aligned one frame earlier automatically for `wckdsmat extract` time stamps (`--audio-offset`).
- **Event flags:** **F** or *Flag at playhead*, or click in "add flag" mode. Drag flags on the plot, and edit or delete them in the table.
- **Segments:** label (and task) any interval between neighboring flags in the segments table.
- **Saving:** annotations autosave and reload next time. *Save segment table* writes `RECORDING_segments.csv` and `RECORDING_events.csv`, and can update a project-wide table. These are the format `wckdsmat kinematics` and `wckdsmat validate` read.
- **Segment this file now:** writes one file per segment plus per-segment kinematics, without waiting for a batch run.

## Configuration

One TOML file controls the whole pipeline: which MediaPipe Face Mesh points to track and what to call them, named groups (head anchors, lips, jaw), derived points (e.g. `IMLL = MLL - mean(LJ, RJ)`), and every stage's settings. The `speech` preset holds the defaults for speech and oral-motor work. Override only what you need:

```bash
wckdsmat config init my_study.toml      # start from the speech preset
wckdsmat config check my_study.toml     # report every problem at once
wckdsmat config show my_study.toml      # print the resolved settings
```

Presets: `speech` (default) and `interspeech2026`, which reproduces the published validation (Sanchez et al., Interspeech 2026: 3D position uRMSE 2.03 ± 0.59 mm over 116 segments).

## Signal conventions

See [docs/fix_log.md](docs/fix_log.md). In short: 3D traces are excursion, error or speed magnitudes, never `||position||`.

## Citation

If you use WCKDSMAT, please cite:

> Sanchez, V. E., Roesler, O., Neumann, M., Pautler, D., Richburg, B., Chenausky, K. V., Yunusova, Y., Ramanarayanan, V., & Green, J. (2026). From Lab to Laptop: Validating 3D Speech Kinematics with MediaPipe Face Mesh. *Proceedings of Interspeech 2026*.

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

WCKDSMAT is free software under the [GNU Affero General Public License v3.0 or later](LICENSE) (`AGPL-3.0-or-later`). You may use, study, modify and share it. If you distribute a modified version, **or run one as a network service** (for example a hosted analysis platform or the viewer), you must make its source available under the same license, so improvements stay open.
