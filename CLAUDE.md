# WCKDSMAT — instructions for coding agents

## Hard rules
- **No participant data in this repo, ever.** No video, audio, or real tracking CSVs, and no participant IDs or session dates in code, comments, docstrings or commit messages. Use `SUBJ001_task`-style placeholders. Only synthetic data under `tests/fixtures/` may be committed.
- **Edit files in place. Never create `_v2`, `_FINAL`, `_GOOD`, `_new` or dated copies.** Git holds the history. Work on a branch, commit when a change works, and let the commit message say what changed.
- No hardcoded absolute paths. Every input and output path comes from CLI arguments or a config file.

## Configuration
- Every setting a scientist might change (landmark names and MediaPipe indices, groups, filter cutoffs, reference frame, derivative units, ...) lives in the config (`src/wckdsmat/config.py`, presets in `src/wckdsmat/presets/`). **Never hardcode a landmark index, landmark name or tunable in a stage**: read it from `Config`.
- A new setting needs: a preset default, a schema entry plus a validation rule in `config.py`, and a test in `tests/test_config.py`.
- Columns follow `<NAME>_<space>_<AXIS>`; build them with `config.column()`.
- Left/right names are from the subject's point of view.

## Signal conventions (see docs/fix_log.md, non-negotiable)
- Never use `||position||` as a "3D trace" (it depends on the origin).
- Valid 3D scalars: error magnitude `||B - A||`, excursion `||X - ref||` (ref = first frame unless stated), speed `||dX/dt||`.
- "Displacement" means position minus a reference (the first frame by default), not a copy of position.

## Pipeline stages (src/wckdsmat/)
Order is fixed (Interspeech 2026, sections 3.2-3.6):
1. `extract` — MediaPipe Face Mesh landmarks from video, in pixels → `*_gfm_sp_t.csv`, `*_gfm_all_t.csv`
2. `calibrate` — Method G → `NAME_t_X` in mm or face units → `*_cal.csv`
3. `preprocess` — head-normalize, then low-pass the full-length trajectory; derived points → `NAME_hn_X`, `NAME_lp_X` → `*_lp_hn.csv`
4. task segmentation — segment table (`wckdsmat.segments`); distinct from recording gaps
5. `kinematics` — per-segment displacement (from segment onset), velocity, acceleration; derivatives never cross a recording gap
6. `validate` — motion capture (CTX) vs webcam: cut with CTX offset → resample CTX to webcam frames → one rigid fit on the head anchors → traces → metrics → ROM95/Speed95 normalization → summary. Must keep reproducing 2.030 ± 0.589 mm with `interspeech2026` (see docs/PROVENANCE.md).

`viewer/` is the QC and segmentation reviewer. It is **not yet part of the package**: it moves into `src/` only once it is functional and finished.

## Definition of done for any change
- `pytest` passes, and runs in seconds on the synthetic fixture.
- Any new behavior has a test. Signal-convention changes need a regression test.
- Study-specific analysis (POLO oral-motor, the validation-study tables) does not belong here.
