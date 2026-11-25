"""Synthetic recordings with known ground truth (no participant data)."""
import numpy as np
import pandas as pd
import pytest

from wckdsmat.config import load_config

# Face-shaped layout in mm, head frame (x: subject's left+, y: up+, z: toward camera+)
FACE = {
    "NT": (0, 0, 30), "RCE": (-45, 35, 0), "LCE": (45, 35, 0),
    "RTH": (-20, 75, 5), "LTH": (20, 75, 5), "RBH": (-15, 50, 10), "LBH": (15, 50, 10),
    "RCL": (-25, -30, 15), "LCL": (25, -30, 15), "MUL": (0, -22, 22), "MLL": (0, -38, 20),
    "RJ": (-35, -60, 0), "MJ": (0, -75, 10), "LJ": (35, -60, 0),
}
MOVING = {"MLL": (0, 5.0, 4.0), "RJ": (0, 3.0, 4.0), "LJ": (0, 3.0, 4.0), "MJ": (0, 3.0, 4.0)}  # (axis, amp mm, Hz)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def make_recording(fs=30.0, seconds=10.0, t=None, head_motion=True, step=None):
    """Return (observed_df, truth) where truth[name] is the (T,3) head-frame trajectory.

    Head pose is identity at frame 0, so head-normalized output should equal truth.
    ``step`` = (frame, name, mm) adds a sudden offset (to test filtering across gaps).
    """
    if t is None:
        t = np.arange(int(fs * seconds)) / fs
    T = len(t)
    truth = {}
    for name, p in FACE.items():
        arr = np.tile(np.array(p, float), (T, 1))
        if name in MOVING:
            ax, amp, hz = MOVING[name]
            arr[:, 1] -= amp * (1 - np.cos(2 * np.pi * hz * t)) / 2   # opens downward from rest
        truth[name] = arr
    if step:
        f, name, mm = step
        truth[name][f:, 1] += mm
    cols = {"timestamp": t}
    for name, arr in truth.items():
        obs = arr.copy()
        if head_motion:
            for i, ti in enumerate(t - t[0]):
                R = rot_y(np.deg2rad(12) * np.sin(2 * np.pi * 0.3 * ti)) @ rot_x(np.deg2rad(6) * np.sin(2 * np.pi * 0.2 * ti))
                obs[i] = R @ arr[i] + np.array([8, -4, 15]) * np.sin(2 * np.pi * 0.25 * ti)
        for k, a in enumerate("xyz"):
            cols[f"{name}_{a}"] = obs[:, k]          # legacy calibrated naming
    return pd.DataFrame(cols), truth


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def recording():
    return make_recording()
