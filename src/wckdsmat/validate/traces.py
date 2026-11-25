#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")   # Prevents macOS GUI crash during parallel overlays

"""
validation_pipeline.py
Per-repetition validation logic for CTX vs GFM.

Features:
  • Full time series for pos / disp / velocity
  • safe_derivative() for irregular time axes
  • Fast, parallel overlay generation
  • Stable styling (Dark Vanilla aesthetic)
  • EXACT expected output layout retained for bandini jaw scripts !
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor


def safe_derivative(arr, time_s, order=2):
    t = np.asarray(time_s, float)
    arr = np.asarray(arr, float)

    t_unique, idx = np.unique(t, return_index=True)
    arr_unique = arr[idx]

    if len(t_unique) < len(t):
        print(f"[warn] safe_derivative: removed {len(t) - len(t_unique)} duplicated timestamps")

    if not np.all(np.diff(t_unique) > 0):
        raise ValueError("time_s is not strictly increasing — cannot compute derivative safely.")

    deriv_clean = np.gradient(arr_unique, t_unique, axis=0, edge_order=order)

    deriv_interp = np.zeros_like(arr)

    if arr.ndim == 1:
        deriv_interp = np.interp(t, t_unique, deriv_clean, left=np.nan, right=np.nan)
    else:
        for d in range(arr.shape[1]):
            deriv_interp[:, d] = np.interp(t, t_unique, deriv_clean[:, d],
                                           left=np.nan, right=np.nan)

    return deriv_interp


BG_COLOR   = "#FBE3E7"
GRID_COLOR = "#8DA08A"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "grid.linewidth":   0.8,
    "savefig.facecolor": BG_COLOR,
})

sns.set_style("whitegrid", {
    "axes.facecolor": BG_COLOR,
    "grid.color": GRID_COLOR,
})


def apply_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    return ax


def _save_overlay(args):
    df, t, col_ctx, col_gfm, out = args

    y_ctx = df[col_ctx].to_numpy()
    y_gfm = df[col_gfm].to_numpy()

    y_ctx0 = y_ctx - y_ctx[0]
    y_gfm0 = y_gfm - y_gfm[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    apply_ax(ax)

    ax.plot(t, y_ctx0, color="#3A4A7F", lw=2.0, label="CTX")
    ax.plot(t, y_gfm0, color="#D597EC", lw=2.0, alpha=0.8, label="GFM")

    ax.set_title(col_ctx.replace("_ctx", ""), loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def generate_overlays_parallel(df, time_s, ctx_cols, gfm_cols, out_prefix):
    tasks = []
    for c_ctx, c_gfm in zip(ctx_cols, gfm_cols):
        fname = f"{out_prefix}_{c_ctx.replace('_ctx','')}_overlay.png"
        tasks.append((df, time_s, c_ctx, c_gfm, fname))

    with ThreadPoolExecutor(max_workers=12) as ex:
        ex.map(_save_overlay, tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", required=True)
    parser.add_argument("--gfm", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    CTX = Path(args.ctx)
    GFM = Path(args.gfm)
    OUT = Path(args.out)

    (OUT / "time_series").mkdir(parents=True, exist_ok=True)
    (OUT / "disp").mkdir(exist_ok=True)
    (OUT / "inflation").mkdir(exist_ok=True)

    df_ctx = pd.read_csv(CTX)
    df_gfm = pd.read_csv(GFM)

    time_s = df_ctx["time_s"].to_numpy()

    def detect_xyz_roots(cols, exclude_roots=None):
        exclude_roots = set(exclude_roots or [])
        colset = set(cols)

        roots = set()
        for c in colset:
            if not c.endswith("_x"):
                continue
            root = c[:-2]
            if root in exclude_roots:
                continue
            if f"{root}_y" in colset and f"{root}_z" in colset:
                roots.add(root)
        return roots

    EXCLUDE = {"COM", "com"}

    ctx_roots = detect_xyz_roots(df_ctx.columns, exclude_roots=EXCLUDE)
    gfm_roots = detect_xyz_roots(df_gfm.columns, exclude_roots=EXCLUDE)

    roots = sorted(ctx_roots & gfm_roots)

    # --- Build per-landmark position arrays: shape (T, N, 3)
    pos_ctx = np.stack([
        np.stack([
            df_ctx[f"{r}_x"].to_numpy(),
            df_ctx[f"{r}_y"].to_numpy(),
            df_ctx[f"{r}_z"].to_numpy(),
        ], axis=1)
        for r in roots
    ], axis=1)

    pos_gfm = np.stack([
        np.stack([
            df_gfm[f"{r}_x"].to_numpy(),
            df_gfm[f"{r}_y"].to_numpy(),
            df_gfm[f"{r}_z"].to_numpy(),
        ], axis=1)
        for r in roots
    ], axis=1)

    # NOTE on "_3d" traces:
    #  - For POSITION files, we store 3D EXCURSION magnitude relative to the first frame
    #    (translation-invariant, not distance-from-origin).
    #  - For DISPLACEMENT files, we store axis-wise displacement (zeroed at first frame)
    #    plus the same 3D excursion magnitude.
    #  - For VELOCITY files, we store axis-wise velocity plus 3D SPEED magnitude (||v||),
    #    not d/dt(||x||).

    # 3D excursion magnitude from first frame (per landmark), mm
    exc_ctx = pos_ctx - pos_ctx[:1, :, :]   # (T,N,3)
    exc_gfm = pos_gfm - pos_gfm[:1, :, :]
    mag_exc_ctx = np.linalg.norm(exc_ctx, axis=2)  # (T,N)
    mag_exc_gfm = np.linalg.norm(exc_gfm, axis=2)  # (T,N)

    # -----------------------------
    # Write POSITION time series
    # -----------------------------
    pos_ctx_df = pd.DataFrame({"time_s": time_s})
    pos_gfm_df = pd.DataFrame({"time_s": time_s})

    for i, r in enumerate(roots):
        pos_ctx_df[f"{r}_x"] = pos_ctx[:, i, 0]
        pos_ctx_df[f"{r}_y"] = pos_ctx[:, i, 1]
        pos_ctx_df[f"{r}_z"] = pos_ctx[:, i, 2]
        pos_ctx_df[f"{r}_3d"] = mag_exc_ctx[:, i]   # excursion magnitude

        pos_gfm_df[f"{r}_x"] = pos_gfm[:, i, 0]
        pos_gfm_df[f"{r}_y"] = pos_gfm[:, i, 1]
        pos_gfm_df[f"{r}_z"] = pos_gfm[:, i, 2]
        pos_gfm_df[f"{r}_3d"] = mag_exc_gfm[:, i]   # excursion magnitude

    pos_ctx_df.to_csv(OUT / "time_series" / "ctx_position.csv", index=False)
    pos_gfm_df.to_csv(OUT / "time_series" / "gfm_position.csv", index=False)

    # -----------------------------
    # Write DISPLACEMENT time series (zeroed at first frame)
    # -----------------------------
    ctx_disp_out = OUT / "time_series" / "ctx_displacement.csv"
    gfm_disp_out = OUT / "time_series" / "gfm_displacement.csv"

    ctx_rows = {"time_s": time_s}
    gfm_rows = {"time_s": time_s}

    for i, r in enumerate(roots):
        ctx_rows[f"{r}_x"] = exc_ctx[:, i, 0]
        ctx_rows[f"{r}_y"] = exc_ctx[:, i, 1]
        ctx_rows[f"{r}_z"] = exc_ctx[:, i, 2]
        ctx_rows[f"{r}_3d"] = mag_exc_ctx[:, i]

        gfm_rows[f"{r}_x"] = exc_gfm[:, i, 0]
        gfm_rows[f"{r}_y"] = exc_gfm[:, i, 1]
        gfm_rows[f"{r}_z"] = exc_gfm[:, i, 2]
        gfm_rows[f"{r}_3d"] = mag_exc_gfm[:, i]

    pd.DataFrame(ctx_rows).to_csv(ctx_disp_out, index=False)
    pd.DataFrame(gfm_rows).to_csv(gfm_disp_out, index=False)

    # -----------------------------
    # Velocity (derivative of position; invariant to constant offset)
    # -----------------------------
    vel_ctx = safe_derivative(pos_ctx.reshape(len(time_s), -1), time_s)
    vel_gfm = safe_derivative(pos_gfm.reshape(len(time_s), -1), time_s)

    vel_ctx = vel_ctx.reshape(pos_ctx.shape)
    vel_gfm = vel_gfm.reshape(pos_gfm.shape)

    # 3D SPEED magnitude (per landmark), mm/s
    vel_speed_ctx = np.linalg.norm(vel_ctx, axis=2)  # (T,N)
    vel_speed_gfm = np.linalg.norm(vel_gfm, axis=2)

    vel_ctx_df = pd.DataFrame({"time_s": time_s})
    vel_gfm_df = pd.DataFrame({"time_s": time_s})

    for i, r in enumerate(roots):
        vel_ctx_df[f"{r}_x"] = vel_ctx[:, i, 0]
        vel_ctx_df[f"{r}_y"] = vel_ctx[:, i, 1]
        vel_ctx_df[f"{r}_z"] = vel_ctx[:, i, 2]
        vel_ctx_df[f"{r}_3d"] = vel_speed_ctx[:, i]

        vel_gfm_df[f"{r}_x"] = vel_gfm[:, i, 0]
        vel_gfm_df[f"{r}_y"] = vel_gfm[:, i, 1]
        vel_gfm_df[f"{r}_z"] = vel_gfm[:, i, 2]
        vel_gfm_df[f"{r}_3d"] = vel_speed_gfm[:, i]

    vel_ctx_df.to_csv(OUT / "time_series" / "ctx_velocity.csv", index=False)
    vel_gfm_df.to_csv(OUT / "time_series" / "gfm_velocity.csv", index=False)

    disp = np.linalg.norm(pos_ctx - pos_gfm, axis=2)
    disp_df = pd.DataFrame({"time_s": time_s})
    for i, r in enumerate(roots):
        disp_df[r] = disp[:, i]

    disp_df.to_csv(OUT / "disp" / "errors_timeseries.csv", index=False)

    merged = df_ctx.merge(df_gfm, on="time_s", suffixes=("_ctx", "_gfm"))
    ctx_cols = [c for c in merged.columns if c.endswith("_ctx")]
    gfm_cols = [c for c in merged.columns if c.endswith("_gfm")]

    generate_overlays_parallel(
        merged,
        merged["time_s"].to_numpy(),
        ctx_cols,
        gfm_cols,
        str(OUT / "overlay")
    )


if __name__ == "__main__":
    main()
