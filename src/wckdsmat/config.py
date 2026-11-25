"""Single source of truth for pipeline settings.

A config is a TOML file layered over a preset (``speech`` by default)::

    preset = "speech"            # or "none" to define everything yourself

    [landmarks]
    UPPER_LID_R = 159            # add a MediaPipe point

    [preprocess]
    lowpass_hz = 8.0             # override one setting

Tables merge key by key; lists and scalars replace. Every stage reads its
settings from :class:`Config`, so a script, the CLI and a GUI all share the
same file.

Column contract: ``<NAME>_<space>_<AXIS>``, e.g. ``MLL_hn_X``, where space is
one of ``SPACES``. Use :func:`column` rather than building names by hand.
"""
from __future__ import annotations

import copy
import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

N_FACE_POINTS = 468          # MediaPipe Face Mesh without iris
N_POINTS_WITH_IRIS = 478     # with refine_landmarks / iris
SPACES = ("px", "t", "hn", "lp")   # pixel, calibrated, head-normalized, low-passed only
AXES = ("X", "Y", "Z")
KIN_PRODUCTS = ("position", "displacement", "velocity", "acceleration")
NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

_SCHEMA: dict[str, set[str]] = {
    "meta": {"name", "description", "side_convention"},
    "extract": {"refine_landmarks", "include_iris", "min_detection_confidence",
                "min_tracking_confidence", "mirror", "mirror_method", "save_all_landmarks",
                "write_overlay_video", "timestamps"},
    "calibrate": {"eye_pair", "eye_distance_mm", "xy_units", "pixel_aspect", "scale_mode", "z_scale", "scale_center",
                  "scale_smooth_window_s", "scale_clip", "anchors", "z_zero_mode", "z_direction"},
    "preprocess": {"order", "anchors", "anchor_mode", "min_anchors", "ref_frame", "ref_search_s",
                   "lowpass_hz", "nyquist_cap", "filter_order", "preserve_unnormalized",
                   "gap_split_factor", "max_interp_gap_frames"},
    "kinematics": {"derivative", "displacement_ref", "products", "magnitude_3d"},
    "validate": {"reference_fs", "max_lag_ms", "fit", "bases", "min_valid_frames", "lag_fs",
                 "range_percentiles", "range_floor_mm", "rom_floor_mm", "speed_floor_mm_s",
                 "spectral_max_lag_ms", "summary_landmarks", "max_reference_fill"},
}
_TOP_LEVEL = {"preset", "landmarks", "groups", "derived", *_SCHEMA}


class ConfigError(ValueError):
    """Raised with every problem found, one per line."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("invalid config:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class Derived:
    name: str
    op: str                  # "mean" | "difference"
    of: tuple[str, ...]
    space: str = "hn"        # "hn": from head-normalized points | "lp": from un-normalized (preserved) points


@dataclass(frozen=True)
class Config:
    meta: dict[str, Any]
    landmarks: dict[str, int]
    groups: dict[str, tuple[str, ...]]
    mirror_pairs: tuple[tuple[str, str], ...]
    derived: tuple[Derived, ...]
    extract: dict[str, Any]
    calibrate: dict[str, Any]
    preprocess: dict[str, Any]
    kinematics: dict[str, Any]
    validate: dict[str, Any]
    source: str = field(default="", compare=False)

    # -- convenience -------------------------------------------------------
    def resolve(self, ref: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Turn a group name or a list of names into a tuple of landmark names."""
        if isinstance(ref, str):
            return self.groups[ref] if ref in self.groups else (ref,)
        return tuple(ref)

    def index_of(self, name: str) -> int:
        return self.landmarks[name]

    @property
    def names(self) -> tuple[str, ...]:
        """All point names: landmarks, then derived points."""
        return (*self.landmarks, *(d.name for d in self.derived))


def column(name: str, axis: str, space: str | None = None) -> str:
    """Canonical column name, e.g. column("MLL", "x", "hn") -> "MLL_hn_X"."""
    axis = axis.upper()
    if axis not in (*AXES, "3D"):
        raise ValueError(f"axis must be one of {AXES} or 3D, got {axis!r}")
    if space is None:
        return f"{name}_{axis}"
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")
    return f"{name}_{space}_{axis}"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def available_presets() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files("wckdsmat.presets").iterdir()
                  if p.name.endswith(".toml"))


def preset_text(name: str) -> str:
    if name not in available_presets():
        raise ConfigError([f"unknown preset {name!r}; available: {available_presets()}"])
    return resources.files("wckdsmat.presets").joinpath(f"{name}.toml").read_text()


def _preset_dict(name: str, _seen: tuple[str, ...] = ()) -> dict:
    """A preset's settings, layered over the preset it extends (``preset = "..."``)."""
    if name == "none":
        return {}
    if name in _seen:
        raise ConfigError([f"preset cycle: {' -> '.join((*_seen, name))}"])
    raw = tomllib.loads(preset_text(name))
    parent = raw.pop("preset", None)
    return _merge(_preset_dict(parent, (*_seen, name)), raw) if parent else raw


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, *, preset: str | None = None,
                overrides: dict | None = None) -> Config:
    """Load a config file layered over its preset, apply overrides, and validate.

    ``overrides`` is a nested dict with the same shape as the TOML (what a GUI
    or a notebook would pass).
    """
    user: dict = {}
    if path is not None:
        with open(path, "rb") as fh:
            user = tomllib.load(fh)
    base_name = preset or user.get("preset", "speech")
    base = _preset_dict(base_name)
    raw = _merge(base, user)
    if overrides:
        raw = _merge(raw, overrides)
    raw.pop("preset", None)
    return _build(raw, source=str(path) if path else f"preset:{base_name}")


def _build(raw: dict, source: str) -> Config:
    problems: list[str] = []
    for key in raw:
        if key not in _TOP_LEVEL:
            problems.append(f"unknown top-level key [{key}]")
    for sect, allowed in _SCHEMA.items():
        for key in raw.get(sect, {}):
            if key not in allowed:
                problems.append(f"[{sect}] unknown key {key!r} (allowed: {sorted(allowed)})")

    extract = dict(raw.get("extract", {}))
    iris = bool(extract.get("include_iris", False))
    max_idx = (N_POINTS_WITH_IRIS if iris else N_FACE_POINTS) - 1

    # landmarks
    landmarks: dict[str, int] = {}
    seen_idx: dict[int, str] = {}
    for name, idx in raw.get("landmarks", {}).items():
        if not NAME_RE.match(name):
            problems.append(f"landmark name {name!r} must be UPPER_CASE (letters, digits, _)")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx <= max_idx:
            hint = " (iris points need extract.include_iris = true)" if isinstance(idx, int) and N_FACE_POINTS <= idx < N_POINTS_WITH_IRIS else ""
            problems.append(f"landmark {name} = {idx!r}: index must be an integer 0-{max_idx}{hint}")
            continue
        if idx in seen_idx:
            problems.append(f"landmarks {seen_idx[idx]} and {name} both use index {idx}")
        seen_idx[idx] = name
        landmarks[name] = idx
    if not landmarks:
        problems.append("no landmarks defined")

    # derived points (may reference landmarks or earlier derived points)
    derived: list[Derived] = []
    known = set(landmarks)
    for i, d in enumerate(raw.get("derived", [])):
        name, op, of = d.get("name"), d.get("op"), tuple(d.get("of", ()))
        space = d.get("space", "hn")
        label = name or f"derived[{i}]"
        if space not in ("hn", "lp"):
            problems.append(f"{label}: space must be 'hn' or 'lp', got {space!r}")
        for k in d:
            if k not in ("name", "op", "of", "space"):
                problems.append(f"{label}: unknown key {k!r}")
        if not name or not NAME_RE.match(name):
            problems.append(f"{label}: name must be UPPER_CASE")
        elif name in known:
            problems.append(f"{label}: name already used")
        if op not in ("mean", "difference"):
            problems.append(f"{label}: op must be 'mean' or 'difference', got {op!r}")
        if op == "difference" and len(of) != 2:
            problems.append(f"{label}: difference needs exactly 2 points, got {len(of)}")
        if op == "mean" and len(of) < 2:
            problems.append(f"{label}: mean needs at least 2 points")
        for ref in of:
            if ref not in known:
                problems.append(f"{label}: unknown point {ref!r}")
        if name:
            known.add(name)
            derived.append(Derived(name, op, of, space))

    # groups
    groups_raw = dict(raw.get("groups", {}))
    pairs_raw = groups_raw.pop("mirror_pairs", [])
    groups: dict[str, tuple[str, ...]] = {}
    for g, members in groups_raw.items():
        missing = [m for m in members if m not in landmarks]
        if missing:
            problems.append(f"group {g!r} references unknown landmarks {missing}")
        groups[g] = tuple(members)

    mirror_pairs: list[tuple[str, str]] = []
    in_pair: set[str] = set()
    for pair in pairs_raw:
        if len(pair) != 2 or pair[0] == pair[1]:
            problems.append(f"mirror pair {pair} must be two different landmarks")
            continue
        for m in pair:
            if m not in landmarks:
                problems.append(f"mirror pair {pair}: unknown landmark {m!r}")
            if m in in_pair:
                problems.append(f"landmark {m} appears in more than one mirror pair")
            in_pair.add(m)
        mirror_pairs.append((pair[0], pair[1]))

    def resolve(ref, where, minimum=1):
        names = groups.get(ref, (ref,)) if isinstance(ref, str) else tuple(ref)
        bad = [n for n in names if n not in landmarks]
        if bad:
            problems.append(f"{where}: unknown landmarks/group {bad}")
        elif len(names) < minimum:
            problems.append(f"{where}: needs at least {minimum} landmarks, got {len(names)}")

    def choice(sect, key, options):
        v = raw.get(sect, {}).get(key)
        if v is not None and v not in options:
            problems.append(f"[{sect}] {key} = {v!r}: must be one of {list(options)}")

    def positive(sect, key, *, integer=False, allow_zero=False):
        v = raw.get(sect, {}).get(key)
        if v is None:
            return
        ok_type = isinstance(v, int) if integer else isinstance(v, (int, float))
        if isinstance(v, bool) or not ok_type or v < 0 or (v == 0 and not allow_zero):
            problems.append(f"[{sect}] {key} = {v!r}: must be a {'non-negative' if allow_zero else 'positive'} {'integer' if integer else 'number'}")

    # extract
    for k in ("min_detection_confidence", "min_tracking_confidence"):
        v = extract.get(k)
        if v is not None and not (isinstance(v, (int, float)) and 0 < v <= 1):
            problems.append(f"[extract] {k} = {v!r}: must be in (0, 1]")
    choice("extract", "timestamps", ("nominal", "container"))
    choice("extract", "mirror_method", ("flip_frame", "flip_coordinates"))
    if extract.get("mirror") and extract.get("mirror_method") == "flip_coordinates" and not mirror_pairs:
        problems.append("[extract] mirror_method = 'flip_coordinates' needs groups.mirror_pairs")
    if extract.get("include_iris") and extract.get("refine_landmarks") is False:
        problems.append("[extract] include_iris = true needs refine_landmarks = true")

    # calibrate
    cal = raw.get("calibrate", {})
    if "eye_pair" in cal:
        resolve(cal["eye_pair"], "[calibrate] eye_pair")
        if len(cal["eye_pair"]) != 2:
            problems.append("[calibrate] eye_pair must list exactly 2 landmarks")
    if "anchors" in cal:
        resolve(cal["anchors"], "[calibrate] anchors", minimum=3)
    positive("calibrate", "eye_distance_mm")
    choice("calibrate", "xy_units", ("pixels", "normalized01"))
    choice("calibrate", "scale_mode", ("eye_2d", "eye_3d"))
    choice("calibrate", "z_scale", ("global_median", "per_frame"))
    choice("calibrate", "scale_center", ("head_centroid", "image_origin"))
    choice("calibrate", "z_zero_mode", ("anchors_mean", "anchors_max", "anchors_min",
                                         "global_mean", "global_max", "global_min", "none"))
    positive("calibrate", "pixel_aspect")
    positive("calibrate", "scale_smooth_window_s", allow_zero=True)
    sc = cal.get("scale_clip")
    if sc not in (None, []) and not (isinstance(sc, list) and len(sc) == 2
                               and all(isinstance(v, (int, float)) for v in sc) and 0 < sc[0] < 1 < sc[1]):
        problems.append(f"[calibrate] scale_clip = {sc!r}: must be [low, high] with 0 < low < 1 < high, or [] for off")
    choice("calibrate", "z_direction", ("decrease_toward_camera", "increase_toward_camera"))

    # preprocess
    pre = raw.get("preprocess", {})
    choice("preprocess", "order", ("hn_then_lp", "lp_then_hn"))
    if "anchors" in pre:
        resolve(pre["anchors"], "[preprocess] anchors", minimum=3)
    if "preserve_unnormalized" in pre:
        resolve(pre["preserve_unnormalized"], "[preprocess] preserve_unnormalized")
    if "preserve_unnormalized" in pre and not problems:
        kept = set(groups.get(pre["preserve_unnormalized"], (pre["preserve_unnormalized"],))
                   if isinstance(pre["preserve_unnormalized"], str) else pre["preserve_unnormalized"])
        lp_names = {d.name for d in derived if d.space == "lp"}
        for d in derived:
            if d.space == "lp":
                bad = [r for r in d.of if r not in kept and r not in lp_names]
                if bad:
                    problems.append(f"{d.name}: space 'lp' needs {bad} in preprocess.preserve_unnormalized")
    positive("preprocess", "ref_frame", integer=True, allow_zero=True)
    positive("preprocess", "lowpass_hz")
    positive("preprocess", "filter_order", integer=True)
    positive("preprocess", "max_interp_gap_frames", integer=True, allow_zero=True)
    choice("preprocess", "anchor_mode", ("fixed", "adaptive"))
    positive("preprocess", "ref_search_s", allow_zero=True)
    ma = pre.get("min_anchors")
    if ma is not None and not (isinstance(ma, int) and not isinstance(ma, bool) and ma >= 3):
        problems.append(f"[preprocess] min_anchors = {ma!r}: must be an integer >= 3 (rigid fit)")
    cap = pre.get("nyquist_cap")
    if cap is not None and not (isinstance(cap, (int, float)) and 0 < cap < 1):
        problems.append(f"[preprocess] nyquist_cap = {cap!r}: must be between 0 and 1 (omit it to raise an error instead)")
    gsf = pre.get("gap_split_factor")
    if gsf is not None and not (isinstance(gsf, (int, float)) and gsf > 1):
        problems.append(f"[preprocess] gap_split_factor = {gsf!r}: must be > 1")

    # kinematics
    kin = raw.get("kinematics", {})
    choice("kinematics", "derivative", ("time", "frame"))
    choice("kinematics", "displacement_ref", ("first_valid", "first", "mean"))
    products = kin.get("products", list(KIN_PRODUCTS))
    for p in products:
        if p not in KIN_PRODUCTS:
            problems.append(f"[kinematics] unknown product {p!r}; options: {list(KIN_PRODUCTS)}")
    for p in kin.get("magnitude_3d", []):
        if p == "position":
            problems.append("[kinematics] magnitude_3d may not include 'position': ||position|| "
                            "depends on the origin; use 'displacement' (excursion) instead")
        elif p not in products:
            problems.append(f"[kinematics] magnitude_3d {p!r} is not in products")

    # validate
    positive("validate", "reference_fs")
    positive("validate", "max_lag_ms", allow_zero=True)
    choice("validate", "fit", ("rigid",))   # only rigid is implemented (validate/pairing.py)
    val = raw.get("validate", {})
    if "bases" in val:
        resolve(val["bases"], "[validate] bases", minimum=3)
    if "summary_landmarks" in val:
        resolve(val["summary_landmarks"], "[validate] summary_landmarks")
    lf = val.get("lag_fs")
    if lf is not None and lf != "measured" and not (isinstance(lf, (int, float)) and lf > 0):
        problems.append(f"[validate] lag_fs = {lf!r}: must be 'measured' or a positive number")
    for k in ("range_floor_mm", "rom_floor_mm", "speed_floor_mm_s", "spectral_max_lag_ms"):
        positive("validate", k, allow_zero=True)
    positive("validate", "min_valid_frames", integer=True)
    mrf = val.get("max_reference_fill")
    if mrf is not None and not (isinstance(mrf, (int, float)) and 0 <= mrf <= 1):
        problems.append(f"[validate] max_reference_fill = {mrf!r}: must be between 0 and 1")

    if problems:
        raise ConfigError(problems)

    return Config(
        meta=dict(raw.get("meta", {})),
        landmarks=landmarks,
        groups=groups,
        mirror_pairs=tuple(mirror_pairs),
        derived=tuple(derived),
        extract=extract,
        calibrate=dict(cal),
        preprocess=dict(pre),
        kinematics=dict(kin),
        validate=dict(raw.get("validate", {})),
        source=source,
    )
