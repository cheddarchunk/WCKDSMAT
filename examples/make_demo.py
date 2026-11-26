# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 WCKDSMAT contributors
"""Make a synthetic demo recording: no real person, no real data.

Writes, into the given folder:

* ``DEMO_ddk_gfm_sp_t.csv``: landmark tracks in pixels, as ``wckdsmat extract``
  would write them, for a face saying "pa-ta-ka" five times a second with a
  little head movement;
* ``DEMO_ddk.wav``: matching audio (a buzzy vowel source shaped by formants,
  with a burst at the start of each syllable).

Then try the whole pipeline::

    python examples/make_demo.py demo
    wckdsmat calibrate demo/DEMO_ddk_gfm_sp_t.csv --eye-mm 95 --out-dir demo
    wckdsmat preprocess demo/DEMO_ddk_gfm_sp_t_cal.csv --out-dir demo
    wckdsmat view demo/DEMO_ddk_gfm_sp_t_cal_lp_hn.csv --audio demo/DEMO_ddk.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import lfilter

FPS, SR, DUR = 30.0, 16000, 12.0
SYL_RATE = 5.0                       # syllables per second during the task
TASK = (2.0, 10.0)                   # task runs 2-10 s; rest before and after
PX_PER_MM = 2.8

# face in mm, head frame (x: subject's left +, y: up +, z: toward camera +); MediaPipe-style names
FACE = {
    "NT": (0, 0, 30), "RCE": (-47, 35, 0), "LCE": (47, 35, 0),
    "RTH": (-20, 75, 5), "LTH": (20, 75, 5), "RBH": (-15, 50, 10), "LBH": (15, 50, 10),
    "RCL": (-25, -30, 15), "LCL": (25, -30, 15), "MUL": (0, -22, 22), "MLL": (0, -38, 20),
    "RJ": (-35, -60, 0), "MJ": (0, -75, 10), "LJ": (35, -60, 0),
}


def syllables(t):
    """Opening (0-1) per sample and the syllable onsets: jaw and lips open on each vowel."""
    on = (t >= TASK[0]) & (t < TASK[1])
    phase = ((t - TASK[0]) * SYL_RATE) % 1.0
    open_ = np.where(on, np.sin(np.pi * np.clip(phase / 0.8, 0, 1)) ** 2, 0.0)
    onsets = np.arange(TASK[0], TASK[1], 1 / SYL_RATE)
    return open_, onsets


def tracks(seed=0):
    rng = np.random.default_rng(seed)
    n = int(DUR * FPS)
    t = np.arange(1, n + 1) / FPS                       # extract stamps frame k at k/fps
    open_, _ = syllables(t)
    kind =np.where(open_ > 0, (np.floor((t - TASK[0]) * SYL_RATE) % 3), -1)   # 0 pa, 1 ta, 2 ka
    cols = {"frame": np.arange(1, n + 1), "timestamp": t, "face_detected": 1}
    yaw = np.deg2rad(4) * np.sin(2 * np.pi * 0.15 * t)
    drift = np.column_stack([6 * np.sin(2 * np.pi * 0.1 * t), 3 * np.sin(2 * np.pi * 0.13 * t), 0 * t])
    for name, p in FACE.items():
        X = np.tile(np.array(p, float), (n, 1))
        if name in ("MJ", "RJ", "LJ"):
            X[:, 1] -= 9 * open_
        if name == "MLL":
            X[:, 1] -= 9 * open_ + 3 * open_ * (kind == 0)    # extra lower-lip movement on "pa"
        if name == "MUL":
            X[:, 1] += 1.5 * open_ * (kind == 0)
        if name in ("RCL", "LCL"):
            X[:, 0] *= 1 - 0.06 * open_
        c, s = np.cos(yaw), np.sin(yaw)
        Xr = np.column_stack([c * X[:, 0] + s * X[:, 2], X[:, 1], -s * X[:, 0] + c * X[:, 2]]) + drift
        Xr += rng.normal(0, 0.15, Xr.shape)                   # tracking noise, mm
        px = np.column_stack([960 + PX_PER_MM * Xr[:, 0], 540 - PX_PER_MM * Xr[:, 1], -PX_PER_MM * Xr[:, 2]])
        for i, a in enumerate("XYZ"):
            cols[f"{name}_px_{a}"] = px[:, i]
    return pd.DataFrame(cols)


def audio(seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(DUR * SR)) / SR
    open_, onsets = syllables(t)
    f0 = 120 + 15 * np.sin(2 * np.pi * 0.3 * t)
    pulses = (np.diff(np.floor(np.cumsum(f0) / SR), prepend=0) > 0).astype(float)   # glottal pulse train
    src = lfilter([1], [1, -0.97], pulses)
    out = np.zeros_like(t)
    for f, bw in ((700, 110), (1250, 140), (2600, 200)):      # an /a/-like vowel
        r = np.exp(-np.pi * bw / SR)
        out += lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r], src)
    voice = out * open_
    burst = np.zeros_like(t)
    for o in onsets:
        i = int(o * SR)
        burst[i:i + int(0.015 * SR)] = rng.normal(0, 1, int(0.015 * SR))
    y = voice / (np.abs(voice).max() + 1e-9) + 0.35 * burst + 0.003 * rng.normal(size=len(t))
    return (y / np.abs(y).max() * 0.8 * 32767).astype(np.int16)


def main(out="demo"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tracks().to_csv(out / "DEMO_ddk_gfm_sp_t.csv", index=False)
    wavfile.write(out / "DEMO_ddk.wav", SR, audio())
    print(f"wrote {out / 'DEMO_ddk_gfm_sp_t.csv'} and {out / 'DEMO_ddk.wav'}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
