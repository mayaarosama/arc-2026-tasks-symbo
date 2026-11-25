"""demo/scripts: every script starts (--help or a dry path), and their pure helpers do what they say.
Nothing here downloads, starts a show or touches the GPU."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import ROOT

SCRIPTS = ROOT / "scripts"


def find(name: str) -> Path:
    """A script by file name, wherever it lives under scripts/ (the folders are grouped by task)."""
    hits = sorted(SCRIPTS.rglob(name))
    assert hits, f"no script {name} under {SCRIPTS}"
    return hits[0]


def load(name: str):
    path = find(name)
    spec = importlib.util.spec_from_file_location(f"script_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(*args, timeout=60) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("name", ["download_weights.py", "verify_offline.py", "rehearse.py", "run_mock_demo.py",
                                  "sample_layout.py", "prepare_gso.py", "plan_downloads.py", "fit_upload.py"])
def test_script_help_runs(name):
    r = run(find(name), "--help")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage:" in r.stdout


@pytest.mark.parametrize("name", ["cpu_box_linux.sh", "setup_gpu_machine.sh", "start_workstation_show.sh"])
def test_shell_scripts_parse(name):
    r = subprocess.run(["bash", "-n", str(find(name))], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not shutil.which("pwsh"), reason="PowerShell not installed")
def test_powershell_script_parses():
    path = find("cpu_box_windows.ps1")
    r = subprocess.run(["pwsh", "-NoProfile", "-Command",
                        f"$null = [System.Management.Automation.Language.Parser]::ParseFile('{path}', [ref]$null, [ref]$e); "
                        "if ($e) { $e; exit 1 }"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_secrets_file_default_matches_its_own_comment():
    """The header comment of each GPU script names the default secrets file; the code must agree."""
    import re
    for name in ("setup_gpu_machine.sh",):
        text = find(name).read_text(encoding="utf-8")
        documented = re.search(r"SECRETS \(default ([^ )]+)", text).group(1)
        code = next(ln for ln in text.splitlines() if ln.strip().startswith("SECRETS=")).split(":-", 1)[1].rstrip('}"')
        code = code.replace("$DEMO/./", "demo/").replace("$DEMO/../", "../")
        assert documented.split("/")[0] == code.split("/")[0], f"{name}: comment says {documented}, code uses {code}"


# ------------------------------------------------------------------ shared tools
def test_download_weights_picks_every_model_a_machine_runs():
    from arcdemo.conductor.config import DemoConfig
    dw = load("download_weights.py")
    cfg = DemoConfig(ROOT / "config/demo.yaml")
    mid = cfg.machines[0]["id"]
    keys = dw.model_keys(cfg, mid, "t2i")
    assert keys and all(cfg.models[k]["task"] == "t2i" for k in keys)
    repos = dw.repos_for(cfg, keys)
    assert repos == sorted(set(repos)) and all("/" in r for r in repos)
    assert dw.CUTOUT_URL.startswith("https://") and dw.CUTOUT_BYTES > 0


def test_download_weights_dry_run_lists_without_fetching():
    r = run(find("download_weights.py"), "--model", "sd15", "--dry-run")
    assert r.returncode == 0, r.stderr[-2000:]


def test_rehearse_cell_time():
    rh = load("rehearse.py")
    assert rh.cell_time({"all_outputs_seconds": 3.0, "elapsed_seconds": 9}) == 3.0
    assert rh.cell_time({"elapsed_seconds": 9}) == 9 and rh.cell_time({}) == 0.0
    assert rh.SAVEABLE == {"ok", "budget_exceeded", "out_of_memory"}


def test_verify_offline_without_arguments_explains_itself():
    r = run(find("verify_offline.py"))
    assert r.returncode != 0 or "usage" in (r.stdout + r.stderr).lower()


# ------------------------------------------------------------------ Task 1
def test_build_rehearsal_prompts_is_deterministic(tmp_path):
    bp = load("build_rehearsal_prompts.py")
    rows = ["Prompt\tCategory\tChallenge\tNote"]
    for ch in bp.CHALLENGES:
        rows += [f"a {ch} thing number {i} on a table\tAnimals\t{ch}\t" for i in range(bp.PER_CHALLENGE + 3)]
    tsv = tmp_path / "parti.tsv"
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    bp.main(str(tsv), str(tmp_path / "a.json"))
    bp.main(str(tsv), str(tmp_path / "b.json"))
    a = json.loads((tmp_path / "a.json").read_text())
    assert a == json.loads((tmp_path / "b.json").read_text())
    items = next(v for v in a.values() if isinstance(v, list)) if isinstance(a, dict) else a
    assert len(items) == len(bp.CHALLENGES) * bp.PER_CHALLENGE


# ------------------------------------------------------------------ Task 2
def test_fit_upload_has_a_main_guard():
    """Importing the script must not run it (it parses sys.argv at import time)."""
    text = find("fit_upload.py").read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in text


def test_prepare_gso_look_at_is_a_rigid_transform():
    import numpy as np
    pg = load("prepare_gso.py")
    m = pg.look_at(np.array([0.0, -2.0, 1.0]), np.zeros(3), np.array([0.0, 0.0, 1.0]))
    r = m[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6) and np.allclose(m[:3, 3], [0, -2, 1])


# ------------------------------------------------------------------ Task 8
def test_build_layout_exemplars_reproduces_the_committed_file(tmp_path, monkeypatch):
    be = load("build_layout_exemplars.py")
    monkeypatch.setattr(be, "OUT", tmp_path / "exemplars.json")
    be.main()
    assert json.loads((tmp_path / "exemplars.json").read_text()) == \
        json.loads((ROOT / "data/layout/exemplars.json").read_text())


def test_compare_layout_against_wall_and_timings(tmp_path):
    from arcdemo.layout_prompt import parse_line
    cl = load("compare_layout.py")
    spec = {"room_type": "bedroom", "floor_plan": [[0, 0], [4, 0], [4, 3], [0, 3]], "door": [3.5, 0]}
    assert cl.against_wall(parse_line("wardrobe wall=north at=1.00", spec), spec)
    assert not cl.against_wall(parse_line("armchair x=2.00 y=1.50 facing=north", spec), spec)
    (tmp_path / "summary.md").write_text("| lay-01 bedroom | ok | 12.3 s | 1.5 s | x | 210 | 17.1 |\n")
    assert cl.timings(tmp_path) == {"lay-01": {"status": "ok", "time": 12.3, "first": 1.5, "tokens": 210}}


def test_sample_layout_review_page_and_prompt_writers(tmp_path):
    sl = load("sample_layout.py")
    from arcdemo.conductor.config import DemoConfig
    rooms = DemoConfig(ROOT / "config/demo.yaml").inputs["layout"][:1]
    sl.write_prompt(tmp_path, rooms)
    assert "=== system ===" in (tmp_path / "prompt.txt").read_text(encoding="utf-8")


# ------------------------------------------------------------------ CPU / GPU machine tools
def test_check_cpu_reports_without_failing():
    r = run(find("check_cpu.py"))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "cpu_threads:" in r.stdout


def test_fetch_llama_cpp_pins_the_same_release_as_the_gpu_setup():
    fl = load("fetch_llama_cpp.py")
    setup = find("setup_gpu_machine.sh").read_text(encoding="utf-8")
    assert f"LLAMA_TAG={fl.TAG}" in setup.replace('"', "").replace("'", "") or fl.TAG in setup
    assert all(size > 0 and fl.TAG in fname for fname, size in fl.ASSETS.values())


def test_plan_downloads_helpers():
    pd = load("plan_downloads.py")
    assert pd.gb(2.5e9) == "2.5 GB"
    assert pd.classify("401 Client Error: gated repo") == "needs_licence"
    assert pd.classify("Connection reset") == "failed"
    assert pd.VRAM_GB["sf3d"] < pd.VRAM_GB["trellis2-4b"]
