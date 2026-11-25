# ValidationPipeline_VES — Fix Log

This document tracks pipeline correctness fixes (trace construction, 3D semantics, normalization), implemented systematically and verified with targeted regression checks.

## Conventions (non-negotiable)

- **Never** use `||position||` as a “3D trace” (origin-dependent).
- Valid scalar 3D traces:
  - **3D error magnitude**: `||B - A||`
  - **3D excursion magnitude**: `||X - ref||` (ref = first frame or mean)
  - **3D speed**: `||dX/dt||`
- “Displacement” means **position zeroed to a reference** (first frame by default), not a copy of position.

---

## Step 1 — Stage 2 trace construction: `validation_pipeline.py`

**Status:** ✅ Implemented

**Primary issue(s) fixed**
1. `_3d` was defined as `||position||` (distance-from-origin), contaminating any downstream 3D “range/correlation” interpretation.
2. `ctx_displacement.csv` / `gfm_displacement.csv` were being written incorrectly (position re-dump instead of displacement).
3. `vel_3d` was being computed as `d/dt(||position||)` (radial velocity), not speed.

**Change summary**
- **Position `_3d`** is now **3D excursion magnitude** relative to the first frame:
  - `pos_3d(t) = ||pos_xyz(t) - pos_xyz(t0)||`
- **Displacement** is now correctly constructed:
  - `disp_xyz(t) = pos_xyz(t) - pos_xyz(t0)`
  - `disp_3d(t) = ||disp_xyz(t)||`
- **Velocity `_3d`** is now **speed**:
  - `vel_3d(t) = ||vel_xyz(t)||`

**Patched output**
- `validation_pipeline_STEP1.py` (drop-in replacement for `validation_pipeline.py`)

**Regression checks**
- Displacement first frame ≈ 0 for every landmark axis:
  - `*_x[0]≈0`, `*_y[0]≈0`, `*_z[0]≈0`, `*_3d[0]≈0`
- Velocity 3D equals vector speed:
  - `vel_3d ≈ sqrt(vx^2 + vy^2 + vz^2)` (numerical match)
- No file labeled “displacement” is identical to “position” (column-wise).

**Notes**
- This step is upstream-critical: all later metrics (Bandini-style) assume these TS files are semantically correct.

---
