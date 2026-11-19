"""demo.yaml mistakes must stop the conductor at startup, naming the bad key."""
from pathlib import Path

import pytest
import yaml

from arcdemo.conductor.config import DemoConfig

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_are_valid():
    DemoConfig(ROOT / "config/demo.yaml")
    DemoConfig(ROOT / "config/demo-mock.yaml")
    DemoConfig(ROOT / "config/demo-full.yaml")
    DemoConfig(ROOT / "config/demo-local.yaml")


def test_typo_in_settings_is_reported(tmp_path):
    raw = yaml.safe_load((ROOT / "config/demo.yaml").read_text(encoding="utf-8"))
    raw["settings"]["t2i"]["iso_time"]["dgx"].update(model="sd35-largee", step=40)
    raw["settings"]["t2i"]["iso_time"]["dgx"].pop("steps")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        DemoConfig(p)
    msg = str(e.value)
    assert "model 'sd35-largee' is not in models.yaml" in msg
    assert "unknown keys ['step']" in msg
    assert msg.count("sd35-largee") == 1          # reported once, not once per segment
