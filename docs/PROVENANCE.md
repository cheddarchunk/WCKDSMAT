# Provenance

Each file was copied from the original research folder `ValidationPipeline_VES/WCKDSMAT/` on 2026-11-25. The only changes: study paths, participant IDs and initials in the usage examples were replaced with placeholders, and `validate/stage_b.py` now calls `traces.py` / `bandini.py` (the renamed STEP1/STEP2 scripts). The md5 is the hash of the original source file.

| Package file | Copied from | Source md5 | Source modified |
|---|---|---|---|
| `src/wckdsmat/extract.py` | **rewritten** (2026-11-25) from `pipeline/gfm_vesbatch_lm_sp.py` | `0e3d3cc54d` | 2026-03-28 |
| `src/wckdsmat/calibrate.py` | **rewritten** (2026-11-25), Method G from `pipeline/gfm_cal_final_ves.py` (same code as `POLO_R/polor_gfm_cal_final_ves.py`) | `6bcee2df8b` | 2025-12-19 |
| `src/wckdsmat/preprocess.py` | **rewritten** (2026-11-25) from `POLO_R/polor_batch_lp_hn_ves_preserve_head.py` + `pipeline/validation/ds_batch_lp_hn_ves_021726.py`; initial copy was `pipeline/ves_gfm_lp_hn_om.py` | `624bcfcddb` | 2026-04-30 |
| ~~`src/wckdsmat/kinematics_imll.py`~~ (removed; replaced by `kinematics.py`) | `WCKDSMAT/batch_gfm_imll_timeseries_case_insensitive.py` | `c0014e9fd1` | 2026-06-30 |
| ~~`src/wckdsmat/kinematics_grouped.py`~~ (removed; replaced by `kinematics.py`) | `POLO_R/batch_gfm_kinematic_timeseries_grouped_v5_3d.py` | `9c51c88a90` | 2026-09-07 |
| ~~`src/wckdsmat/head.py`~~ (removed; `HEAD_CENTROID` derived point + `kinematics.py`) | `POLO_R/batch_head_centroid_from_lp.py` | `3fad569135` | 2026-05-20 |
| `src/wckdsmat/validate/traces.py` | `pipeline/validation/validation_pipeline_STEP1.py` | `b59800091e` | 2026-02-18 |
| `src/wckdsmat/validate/bandini.py` | `pipeline/validation/bandini_full_pipeline_STEP2.py` | `1e2d34bb38` | 2026-02-18 |
| `src/wckdsmat/validate/stage_b.py` | `pipeline/validation/batch_stage_b_pipeline_STEP3.py` | `a6ba09f538` | 2026-02-18 |
| `viewer/reviewer.py` | `WCKDSMAT/wckdsmat_batch_reviewer.py` | `28d1b49cd4` | 2026-07-01 |
| `viewer/qc_displacement.py` | `WCKDSMAT/qc_lip_jaw_displacement_viewer_for_batch_outputs.py` | `a1f36a83b2` | 2026-07-01 |
| `src/wckdsmat/validate/spatial_alignment.py` | `pipeline/validation/spatial_alignment.py` | `2c748f09a1` | 2026-02-17 |
| `docs/fix_log.md` | `pipeline/validation/pipeline_fix_log.md` | `0ee191f924` | 2026-02-18 |

## Open decisions (alternatives that were *not* copied)

| Stage | Chosen | Alternative to reconcile | Difference |
|---|---|---|---|
| calibrate | `pipeline/gfm_cal_final_ves.py` | `POLO_R/polor_gfm_cal_final_ves.py` | 8 lines |
| preprocess | `pipeline/ves_gfm_lp_hn_om.py` (single file) | `POLO_R/polor_batch_lp_hn_ves_preserve_head.py`, `polor_batch_lp_hn_ves.py`, `pipeline/pp_gfm_batch_lp_hn.py` | batch variants; preserve_head differs by 81 lines |
| kinematics | both IMLL and grouped_v5_3d copied | grouped v1–v4 | possibly overlapping functionality |
| validate/traces | `validation_pipeline_STEP1.py` (fix log Step 1) | `validation_pipeline.py`, `_GOOD30HZ`, `_15hz`, `_128`, `_129`, `_og` | STEP1 is the corrected version per the fix log |
| validate (Bland–Altman) | not copied yet | `pipeline/residual_error_ba.py`, `validation/ba_plots.py` | — |
| calibrate batch wrapper | not copied yet | `POLO_R/gfm_cal_batch_ves.py`, `batch_calibrate_faceunits.py` | — |
| viewer | `wckdsmat_batch_reviewer.py` (newest, 2026-07-01) | ~20 `wckdsmat_*`/`segtool_*`/`ves_seg_*` variants, incl. a Qt version | needs a feature inventory before it counts as finished |

## Known issues found during reconciliation (2026-11-25)

- **Eye labels swapped in `extract.py`.** `speech_landmark_map` names index 33 `LCE` and 263 `RCE`, but 33 is on the same side as 61 (`RCL`), 108 (`RTH`) and 176 (`RJ`). The speech preset uses subject-perspective names (`RCE = 33`). Outputs written by the old map have the eye columns swapped.
- **`||position||` written by default.** `kinematics_grouped.py` (`position/*_3D`) and `head.py` (`head_centroid_lp_3D`) do this, in violation of docs/fix_log.md. The config rejects `position` in `magnitude_3d`.
- **Velocity units differ.** `kinematics_grouped` defaults to per-frame and `kinematics_imll` to per-second.
- **Recording gaps are ignored.** Filtering and derivatives run across cuts in segmented recordings (e.g. jumps of up to ~16 s in 25 fps files).
- **Iris points are dropped.** `extract.py` runs `refine_landmarks=True` (478 points) but writes only 0-467.
- **Undefined keys in head pose.** `extract.py` `P2D_KEYS` uses `LMC`/`RMC`, which the landmark map doesn't define.

## Preprocess rewrite and regression check (2026-11-25)

`preprocess.py` is now config-driven and combines the following:
- **From the POLO batch script:** head-normalize then low-pass, preserved un-normalized head trajectories (`*_lp_*`), and interpolation of dropouts.
- **From the validation dual-stream script** (same `kabsch`, filter and head-normalize code as POLO): fixed or adaptive anchor subsets, `min_anchors`, a reference-frame search window, and the 0.95 x Nyquist cutoff cap.
- **New:** splitting at recording gaps, a maximum length for gap filling, derived points from the config, `sosfiltfilt`, and a JSON report per file.

Regression check against the POLO script's real output (2 POLO faceunits files, 25 fps, 54 columns, settings matched to the old behavior: ref 15, no search, no gap split, unlimited fill):
- Median |new - old| was about 3e-15; beyond 30 frames from the file edges the maximum was 1.6e-8.
- Up to 0.03 face units within 30 frames of the file edges, from `sosfiltfilt` vs `filtfilt` edge padding.
- Not covered: neither file had tracking dropouts.

With the new defaults, one file split into 12 segments, several only 4-6 frames long. Segments shorter than the filter's padding stay NaN and are counted in `samples_too_short_to_filter`.

## Order of operations (confirmed against Interspeech 2026, sections 3.2-3.6)

extract → calibrate → preprocess (head-normalize, then low-pass the full-length trajectory) → *(validation: align CTX)* → task segmentation → kinematics (displacement from segment onset, then derivatives) → metrics.

`interspeech2026` preset: reproduces the paper's settings, with no scale smoothing or clipping, as the February 2026 calibration metadata records.

## Calibration rewrite and findings (2026-11-25)

- **Regression check:** on 10 POLO faceunits files with matched settings, the maximum |new - old| was 3.6e-15.
- **Head-pose columns removed from extraction.** The solvePnP model faced +z, so every frame came out about 180° rotated (median |yaw| 134-164°). Nothing downstream read these columns.
- **Scale smoothing never ran.** `--time-col` defaulted to `time_s`, but inputs use `timestamp`, so `--scale-smooth-window-s` was silently skipped. This affects all 243 calibration metadata files found. The new code finds the time column itself.
- **xy_units mismatch.** 50 January 2026 validation calibrations (`data/calibration/flats_gfm`, `data/calibration/012126/normalized`) ran `normalized01` on pixel input, which scales Y by H/W (0.56) and Z by about 0.75. The February runs (`15fps_021626`) and `012126/pixels` used `pixels` correctly. The new code refuses this mismatch.
- **Z scale for normalized01.** The old `sqrt(median sx * median sy)` used a sqrt(W*H) basis, while MediaPipe z uses the width. The new code converts to pixels first (x*W, y*H, z*W), as in the paper.
- **Z offset.** The code subtracts one global anchor-mean offset; the paper's eq. 4 describes a per-frame mean. Head-normalized output is identical either way, and the global offset keeps head z motion.
- **eye_2d vs eye_3d.** `scale_mode = "eye_3d"` is new. The published `eye_2d` inflates all scales by 1/cos(yaw) when the head turns (tested: 15.5% at 30°).
- **Scale jitter enters the un-normalized head trajectories.** Positions are scaled about the image origin, so changes in s(t) move every point by roughly ds x (distance from the image corner). Head normalization cancels this for articulators, but the preserved `*_lp` head points keep it. On 9 POLO files, median head-centroid speed was 3.1-4.5x what a constant scale gives; 0.2 s smoothing reduces this to 1.8-2.4x. Head-movement results from the old pipeline (preserve_head + head centroid) are affected.
- **Fixed: `scale_center = "head_centroid"`** (the speech default). The face shape is scaled per frame about the anchors' centroid, and the centroid's own motion is converted with the median scale. On the same 9 files, head speed went from 3.0-4.3x to 0.96-1.02x of the constant-scale value. `IMLL_hn` displacement was identical (about 1e-14), and `image_origin` still matches the old outputs (about 1e-15). `interspeech2026` keeps `image_origin`, as published.
- **POLO head-movement analyses need re-running** with the new calibration.

## Extraction rewrite (2026-11-25)

- **Config-driven.** Named landmarks and their indices come from the config (subject-perspective labels), and iris points are available with `include_iris`. Frames with no detected face become NaN rows with `face_detected = 0` instead of being dropped. `timestamps = "nominal"` (frame/fps, as before) or `"container"` (for variable-frame-rate video). Requires MediaPipe 0.10.21, the validated version, which pins numpy < 2.
- **Your recordings.** 8 videos checked from `flats/webcam_video`, `flats/15fps` and `POLO_R/videos/adult`: all have a constant frame rate (container = nominal to 0.0 ms), and the old outputs dropped no frames.
- **The old validation outputs used `--mirror-mode`**, which detects on the mirrored frame and then flips x. MediaPipe's indices then sit on the opposite side of the face. The old speech file swapped the pair names, but the eye pair was already mislabeled, so in that file the subject's right eye was called `LCE`. The old all-landmarks file kept the opposite-side indices.
- **Two mirror methods.** `flip_frame` (new default) flips before detection, so indices are anatomically correct in both files. `flip_coordinates` is the published method; it relabels the named file through `mirror_pairs`, and flags the all-landmarks file so calibrate refuses to read lateral points from it by index. On a real 30 fps validation video the two methods recovered all 14 known left-right pairs but differ by 1.1-1.7 mm SD per landmark, because MediaPipe is not exactly symmetric. Which is closer to motion capture is an open validation question.
- **Regression check** (`interspeech2026` settings vs old outputs, 2 videos at 30 fps + 1 at 15 fps from `flats`): all-landmarks and named files match to 3e-5 px, with the eye labels corrected.
