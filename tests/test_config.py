import pytest

from wckdsmat.cli import main
from wckdsmat.config import ConfigError, column, load_config


def write(tmp_path, text, name="cfg.toml"):
    p = tmp_path / name
    p.write_text(text)
    return p


# -- speech preset ----------------------------------------------------------
def test_speech_preset_loads_and_resolves_groups():
    cfg = load_config()
    assert cfg.index_of("MLL") == 17
    assert cfg.resolve("head_anchors")[0] == "RCE"
    assert len(cfg.resolve(cfg.preprocess["anchors"])) >= 3
    assert [d.name for d in cfg.derived] == ["JAW_MID", "IMLL"]
    assert "position" not in cfg.kinematics["magnitude_3d"]


def test_speech_eye_labels_follow_subject_convention():
    """Index 33 sits on the same side as 61 (RCL), 108 (RTH), 176 (RJ)."""
    cfg = load_config()
    same_side = {cfg.index_of(n) for n in ("RCE", "RCL", "RTH", "RJ")}
    assert same_side == {33, 61, 108, 176}


# -- layering ---------------------------------------------------------------
def test_override_one_setting_keeps_rest_of_preset(tmp_path):
    cfg = load_config(write(tmp_path, "[preprocess]\nlowpass_hz = 8.0\n"))
    assert cfg.preprocess["lowpass_hz"] == 8.0
    assert cfg.preprocess["order"] == "hn_then_lp"
    assert "MLL" in cfg.landmarks


def test_add_landmark_and_group(tmp_path):
    cfg = load_config(write(tmp_path, """
[landmarks]
R_UPPER_LID = 159
R_LOWER_LID = 145
[groups]
eyelids = ["R_UPPER_LID", "R_LOWER_LID"]
"""))
    assert cfg.resolve("eyelids") == ("R_UPPER_LID", "R_LOWER_LID")
    assert "MLL" in cfg.landmarks


def test_custom_landmark_set_without_speech_preset(tmp_path):
    cfg = load_config(write(tmp_path, """
preset = "none"
[landmarks]
NT = 1
RCE = 33
LCE = 263
R_BROW = 105
[groups]
head = ["NT", "RCE", "LCE"]
[preprocess]
anchors = "head"
[kinematics]
products = ["displacement", "velocity"]
magnitude_3d = ["velocity"]
"""))
    assert set(cfg.landmarks) == {"NT", "RCE", "LCE", "R_BROW"}
    assert cfg.derived == ()


def test_dict_overrides_for_gui_or_notebook():
    cfg = load_config(overrides={"preprocess": {"ref_frame": 15}})
    assert cfg.preprocess["ref_frame"] == 15


# -- validation -------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("[kinematics]\nmagnitude_3d = ['position']\n", "may not include 'position'"),
    ("[landmarks]\nFOO = 999\n", "index must be an integer"),
    ("[landmarks]\nFOO = 470\n", "iris points need"),
    ("[landmarks]\nDUP = 17\n", "both use index 17"),
    ("[landmarks]\nlower = 5\n", "must be UPPER_CASE"),
    ("[groups]\nbad = ['NOPE']\n", "unknown landmarks"),
    ("[preprocess]\nanchors = ['NT', 'RCE']\n", "at least 3"),
    ("[preprocess]\nlowpas_hz = 10\n", "unknown key 'lowpas_hz'"),
    ("[preprocess]\norder = 'sideways'\n", "must be one of"),
    ("[preprocess]\ngap_split_factor = 1\n", "must be > 1"),
    ("[[derived]]\nname = 'X1'\nop = 'difference'\nof = ['MLL']\n", "exactly 2"),
    ("[calibrate]\neye_distance_mm = -3\n", "positive number"),
])
def test_invalid_configs_are_rejected(tmp_path, text, expected):
    with pytest.raises(ConfigError) as err:
        load_config(write(tmp_path, text))
    assert expected in str(err.value)


def test_all_problems_reported_at_once(tmp_path):
    with pytest.raises(ConfigError) as err:
        load_config(write(tmp_path, "[preprocess]\nlowpass_hz = -1\nfilter_order = 0\n"))
    assert len(err.value.problems) == 2


def test_iris_points_allowed_when_enabled(tmp_path):
    cfg = load_config(write(tmp_path, "[extract]\ninclude_iris = true\n[landmarks]\nR_IRIS = 468\n"))
    assert cfg.index_of("R_IRIS") == 468


# -- columns + CLI ----------------------------------------------------------
def test_column_contract():
    assert column("MLL", "x", "hn") == "MLL_hn_X"
    assert column("MLL", "3d") == "MLL_3D"
    with pytest.raises(ValueError):
        column("MLL", "x", "mm")


def test_cli_init_then_check(tmp_path, capsys):
    p = tmp_path / "study.toml"
    assert main(["config", "init", str(p)]) == 0
    assert main(["config", "check", str(p)]) == 0
    assert main(["config", "init", str(p)]) == 1          # refuses to overwrite
    assert "OK" in capsys.readouterr().out
