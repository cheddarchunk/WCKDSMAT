# Provenance

Each file was copied from the original research folder `ValidationPipeline_VES/WCKDSMAT/` on 2026-11-25. The only changes: study paths, participant IDs and initials in the usage examples were replaced with placeholders, and `validate/stage_b.py` now calls `traces.py` / `bandini.py` (the renamed STEP1/STEP2 scripts). The md5 is the hash of the original source file.

| Package file | Copied from | Source md5 | Source modified |
|---|---|---|---|
| `src/wckdsmat/extract.py` | `pipeline/gfm_vesbatch_lm_sp.py` | `0e3d3cc54d` | 2026-03-28 |
| `src/wckdsmat/calibrate.py` | `pipeline/gfm_cal_final_ves.py` | `6bcee2df8b` | 2025-12-19 |
| `src/wckdsmat/preprocess.py` | **rewritten** (2026-11-25) from `POLO_R/polor_batch_lp_hn_ves_preserve_head.py` + `pipeline/validation/ds_batch_lp_hn_ves_021726.py`; initial copy was `pipeline/ves_gfm_lp_hn_om.py` | `624bcfcddb` | 2026-04-30 |
| `src/wckdsmat/kinematics_imll.py` | `WCKDSMAT/batch_gfm_imll_timeseries_case_insensitive.py` | `c0014e9fd1` | 2026-06-30 |
| `src/wckdsmat/kinematics_grouped.py` | `POLO_R/batch_gfm_kinematic_timeseries_grouped_v5_3d.py` | `9c51c88a90` | 2026-09-07 |
| `src/wckdsmat/head.py` | `POLO_R/batch_head_centroid_from_lp.py` | `3fad569135` | 2026-05-20 |
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
