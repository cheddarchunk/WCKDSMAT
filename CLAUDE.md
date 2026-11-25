# WCKDSMAT — instructions for coding agents

## Hard rules
- **No participant data in this repo, ever.** No video, audio, or real tracking CSVs, and no participant IDs or session dates in code, comments, docstrings or commit messages. Use `SUBJ001_task`-style placeholders. Only synthetic data under `tests/fixtures/` may be committed.
- **Edit files in place. Never create `_v2`, `_FINAL`, `_GOOD`, `_new` or dated copies.** Git holds the history. Work on a branch, commit when a change works, and let the commit message say what changed.
- No hardcoded absolute paths. Every input and output path comes from CLI arguments or a config file.

## Signal conventions (see docs/fix_log.md, non-negotiable)
- Never use `||position||` as a "3D trace" (it depends on the origin).
- Valid 3D scalars: error magnitude `||B - A||`, excursion `||X - ref||` (ref = first frame unless stated), speed `||dX/dt||`.
- "Displacement" means position minus a reference (the first frame by default), not a copy of position.

## Pipeline stages (src/wckdsmat/)
1. `extract` — MediaPipe Face Mesh landmarks from video → `*_gfm_sp.csv`
2. `calibrate` — Method G (dynXY_eyes3D_ratioZ) → mm space `*_t.csv`
3. `preprocess` — low-pass + head normalization + mouth residuals → `*_lp_hn*.csv`
4. `kinematics_imll`, `kinematics_grouped`, `head` — articulator and head time series
5. `validate/` — CTX (motion capture) vs GFM agreement: traces → Bandini metrics → stage B summaries

`viewer/` is the QC and segmentation reviewer. It is **not yet part of the package**: it moves into `src/` only once it is functional and finished.

## Definition of done for any change
- `pytest` passes, and runs in seconds on the synthetic fixture.
- Any new behavior has a test. Signal-convention changes need a regression test.
- Study-specific analysis (POLO oral-motor, the validation-study tables) does not belong here.
