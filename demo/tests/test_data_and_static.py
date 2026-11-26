"""Declared inputs and UI files: every config and data file parses and has the expected shape, every
path a config names exists, every file a page loads is there, and the mock backends' outputs are sane."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave

import pytest
import yaml

from arcdemo.conductor.config import DemoConfig
from arcdemo.conductor.moderation import check_prompt
from arcdemo.layout_prompt import ROOMS, check, load_catalog, parse_line
from arcdemo.schemas import JobRequest
from arcdemo.worker.backends.base import _BACKENDS
from arcdemo.worker.backends.layout import mock_layout
from arcdemo.worker.backends.llm import LlmMock
from arcdemo.worker.backends.mock import _placeholder_image
from arcdemo.worker.backends.music_stub import write_tone
from tests.helpers import CONFIG, ROOT

STATIC = ROOT / "arcdemo" / "conductor" / "static"
DEMO_CONFIGS = sorted(p.name for p in CONFIG.glob("demo*.yaml"))
WORKER_CONFIGS = sorted(p.name for p in CONFIG.glob("worker-*.yaml"))
REQUIRED = {"t2i": {"id", "label", "seed"}, "i23d": {"id", "label", "input", "seed"},
            "music": {"id", "label", "prompt", "seed"}, "llm": {"id", "label", "prompt", "seed"},
            "layout": {"id", "label", "room_type", "floor_plan", "seed"}}


# ------------------------------------------------------------------ config/
@pytest.mark.parametrize("name", DEMO_CONFIGS)
def test_every_demo_config_loads(name):
    cfg = DemoConfig(CONFIG / name)
    assert cfg.machines and cfg.segments


@pytest.mark.parametrize("name", WORKER_CONFIGS)
def test_every_worker_config_has_the_required_keys(name):
    raw = yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))
    assert {"machine", "devices"} <= set(raw), name
    models = yaml.safe_load((CONFIG / "models.yaml").read_text())["models"]
    assert set(raw.get("model_overrides") or {}) <= set(models), name
    for task, keys in (raw.get("preload") or {}).items():
        assert all(models[k]["task"] == task for k in keys), (name, task)


def test_models_yaml_entries_are_complete():
    models = yaml.safe_load((CONFIG / "models.yaml").read_text())["models"]
    for key, spec in models.items():
        assert spec["task"] in REQUIRED and spec["backend"] in _BACKENDS, key
        assert spec.get("repo"), key


def test_blocklist_is_one_term_per_line():
    terms = [ln for ln in (CONFIG / "blocklist.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert all(len(t.split()) <= 3 for t in terms)


@pytest.mark.xfail(reason="Task 3 (music) is the reserve task: inputs and settings, but no segment", strict=False)
def test_every_task_with_inputs_has_a_segment():
    cfg = DemoConfig(CONFIG / "demo.yaml")
    assert set(cfg.inputs) <= {s["task"] for s in cfg.segments}


# ------------------------------------------------------------------ data/
@pytest.mark.parametrize("task", list(REQUIRED))
def test_stage_inputs_have_the_required_fields_and_files(task):
    rel = yaml.safe_load((CONFIG / "demo.yaml").read_text())["inputs"][task]
    items = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), f"duplicate ids in {rel}"
    for item in items:
        if item["id"] == "live":
            continue
        assert REQUIRED[task] <= set(item), (rel, item["id"])
        for key in ("input", "audio"):
            if item.get(key):
                assert (ROOT / item[key]).exists(), (rel, item["id"], item[key])


@pytest.mark.parametrize("name", ["live_fallback_prompts.json", "rehearsal_prompts.json"])
def test_t2i_prompt_lists(name):
    doc = json.loads((ROOT / "data/t2i" / name).read_text(encoding="utf-8"))
    assert doc["prompts"] and all(isinstance(p, (str, dict)) for p in doc["prompts"])


def test_spare_prompts_pass_moderation():
    doc = json.loads((ROOT / "data/t2i/live_fallback_prompts.json").read_text(encoding="utf-8"))
    texts = [p if isinstance(p, str) else p.get("prompt", "") for p in doc["prompts"]]
    assert [t for t in texts if not check_prompt(t)["ok"]] == []


def test_layout_exemplars_are_valid_rooms():
    for e in json.loads((ROOT / "data/layout/exemplars.json").read_text()):
        assert e["room_type"] in ROOMS and len(e["v3"]) == len(e["layout"]) == e["max_objects"]
        assert all(parse_line(ln, e) is not None for ln in e["v3"]), e["id"]


def test_stage_objects_are_not_scored():
    # Scoring against the GSO scans was dropped on purpose: the scans are no longer in the repo.
    for item in json.loads((ROOT / "data/i23d/stage_objects.json").read_text()):
        assert "truth" not in item, item["id"]


# ------------------------------------------------------------------ static UI
def page_refs(html: str) -> list[str]:
    refs = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    for m in re.finditer(r'<script type="importmap">(.*?)</script>', html, re.S):
        refs += [v for v in json.loads(m.group(1))["imports"].values() if v.startswith("/static/")]
    return refs


@pytest.mark.parametrize("page", ["stage.html", "operator.html"])
def test_every_file_a_page_loads_exists(page):
    refs = page_refs((STATIC / page).read_text(encoding="utf-8"))
    assert refs
    for ref in refs:
        assert (STATIC / ref.removeprefix("/static/")).exists(), ref


def test_import_map_comes_before_every_module_script():
    html = (STATIC / "stage.html").read_text(encoding="utf-8")
    assert html.index('type="importmap"') < html.index('type="module"')


def test_layout3d_imports_resolve_to_vendored_files():
    js = (STATIC / "layout3d.js").read_text(encoding="utf-8")
    for spec in re.findall(r'from\s+"([^"]+)"', js):
        if spec == "three":
            assert (STATIC / "vendor/three/three.module.min.js").is_file()
        elif spec.startswith("three/addons/"):
            assert (STATIC / "vendor/three/addons" / spec.removeprefix("three/addons/")).is_file(), spec


@pytest.mark.parametrize("name", ["common.js", "stage.js", "operator.js", "layout.js", "layout3d.js"])
def test_javascript_parses(name):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    src = STATIC / name
    if name == "layout3d.js":     # an ES module: node only parses it as one from stdin with --input-type
        r = subprocess.run([node, "--input-type=module", "--check"], input=src.read_text(encoding="utf-8"),
                           capture_output=True, text=True)
    else:
        r = subprocess.run([node, "--check", str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_furniture_catalogue_models_exist():
    cat = load_catalog(check_models=True)
    assert cat["furniture"] and (ROOT / cat["assets_dir"]).is_dir()


# ------------------------------------------------------------------ mock backends' outputs
def test_mock_placeholder_image_is_seeded():
    req = JobRequest(task="t2i", model="sd15", input_id="i", seed=3)
    a, b = _placeholder_image(req, 0, "dgx", "sd15"), _placeholder_image(req, 0, "dgx", "sd15")
    assert a.size == (512, 512) and a.tobytes() == b.tobytes()
    assert a.tobytes() != _placeholder_image(req, 1, "dgx", "sd15").tobytes()


def test_music_stub_writes_a_stereo_wav(tmp_path):
    write_tone(tmp_path / "t.wav", seconds=1, seed=2)
    with wave.open(str(tmp_path / "t.wav")) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (2, 44100, 44100)


def test_llm_mock_streams_max_tokens():
    b = LlmMock("k", {}, {"mock_speed": 0.0}, ["cpu"])
    out = list(b._stream(JobRequest(task="llm", model="k", input_id="i", max_tokens=12), 0))
    assert len(out) == 12 and out[0][0].startswith("[MOCK]")


def test_layout_mock_rooms_are_clean():
    spec = {"room_type": "bedroom", "floor_plan": [[0, 0], [4, 0], [4, 3.5], [0, 3.5]], "door": [3.5, 0],
            "max_objects": 6}
    placed = []
    for line in mock_layout(spec, 0):
        o = parse_line(line.strip(), spec)
        assert o is not None and not check(o, placed, spec)
        placed.append(o)
    assert placed


# ------------------------------------------------------------------ documentation links
DOCS = sorted(p for p in ROOT.rglob("*.md")
              if not any(part.startswith((".venv", "third_party", "hf-cache", ".pytest")) for part in p.parts))


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_markdown_links_resolve(doc):
    """[text](path) links to files in the repo must exist (anchors and web links are not checked)."""
    text = doc.read_text(encoding="utf-8")
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#")[0].replace("%20", " ")
        if path and not (doc.parent / path).exists():
            broken.append(target)
    assert not broken, f"{doc.relative_to(ROOT)}: broken links {broken}"
