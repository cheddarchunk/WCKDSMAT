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
| QC viewer | `viewer/` (in progress) | interactive review and segmentation |

## Install (development)

```bash
pip install -e ".[all,dev]"
```

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

TODO
