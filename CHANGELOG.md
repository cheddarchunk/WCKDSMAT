# Changelog

All notable changes to WCKDSMAT. Versions follow [semantic versioning](https://semver.org):

- **1.0.x** fixes bugs;
- **1.x.0** adds features without breaking existing commands, config keys or files;
- **2.0.0** would be the first release allowed to break them.

## Unreleased

*(Add changes here as you make them; move them under a version number when you release.)*

## 1.0.0 (2026-11-26)

First public release, consolidated from the research pipeline behind Sanchez et al., Interspeech 2026.

### Pipeline
- `wckdsmat extract`: MediaPipe Face Mesh tracking (468 points, optional iris) in pixels, with a named-landmark file and an all-points file. Frames without a detected face are kept as empty rows. Mirrored recordings are handled with `--mirror`, keeping anatomically correct left/right labels.
- `wckdsmat calibrate`: the published eye-distance calibration (Method G) to millimetres or face units, with a per-recording eye-distance table. Options: a 3D eye distance (`eye_3d`), unaffected by head turns, and scaling about the head centroid (the default), which keeps scale jitter out of head-movement trajectories.
- `wckdsmat preprocess`: rigid head normalization, then zero-phase low-pass filtering. Recordings are split at timestamp breaks, short dropouts are filled, and un-normalized head trajectories are kept for head-movement work.
- Task segment tables (`recording, segment_id, task, start_s, end_s`).
- `wckdsmat kinematics`: per-segment position, displacement (from segment onset), velocity and acceleration, with 3D magnitudes. Derivatives never cross a recording break.
- `wckdsmat validate`: agreement with optical motion capture: uRMSE, RMSE, bias, normalized error, cross-correlation and lag, and spectral centroid. Every row reports how much of the reference is gap-filled.

### Speech Movement Data Viewer
- `wckdsmat view`: X/Y/Z movement and a wideband or narrowband spectrogram on one timeline, with a synchronized playhead.
- Event flags, labeled segments between flags, autosave and resume.
- Segment-table export, and segmenting a single file from the viewer.

### Configuration
- One TOML file sets the landmarks (named MediaPipe indices), groups, derived points and every stage's settings. Configs are validated and every problem is reported at once.
- Presets: `speech` (default) and `interspeech2026`, which reproduces the published 2.03 ± 0.59 mm over 116 segments.

### Documentation
- README, the step-by-step [HOWTOUSE.md](HOWTOUSE.md), a synthetic demo recording (`examples/make_demo.py`), and [docs/PROVENANCE.md](docs/PROVENANCE.md), which records how each stage was checked against the research code.
