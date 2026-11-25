"""Conductor configuration: run of show, machines, settings matrix, inputs.

Validated on load (`_validate`), so a typo in demo.yaml stops the conductor at startup with
a message naming the bad key, instead of failing one cell in front of the audience.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from ..layout_prompt import DEFAULT_CATALOG, load_catalog
from ..schemas import JobRequest

ROOT = Path(__file__).resolve().parents[2]      # demo/


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = data.pop("extends", None)
    if base:
        parent = _load_yaml(path.parent / base)
        parent.update(data)                     # shallow: a child key replaces the parent's
        return parent
    return data


class DemoConfig:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.raw = _load_yaml(self.path)
        self.models = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))["models"]
        self.machines: list[dict] = self.raw["machines"]
        self.segments: list[dict] = self.raw["segments"]
        self.session_dir = (ROOT / self.raw.get("session_dir", "session")).resolve()
        self.replay_dir = (ROOT / self.raw.get("replay_dir", "replay")).resolve()
        self.grace = float(self.raw.get("timeout_grace_seconds", 15))
        self.unreachable_after = float(self.raw.get("unreachable_after_seconds", 6))
        self.token_env = self.raw.get("token_env", "ARC_DEMO_TOKEN")
        self.inputs: dict[str, list[dict]] = {}
        for task, rel in (self.raw.get("inputs") or {}).items():
            self.inputs[task] = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        # model_choices: {task: {model_key: {settings that model needs}}} — the models the operator
        # may pick instead of the planned one (operator page, "AI model"). Not downloaded by default.
        self.choices: dict[str, dict[str, dict]] = {
            task: {k: dict(v or {}) for k, v in (models or {}).items()}
            for task, models in (self.raw.get("model_choices") or {}).items()}
        # The Task 8 furniture catalogue: what the layout model may place, and the model files the stage
        # draws them with (served from its assets_dir). Every model file must exist.
        self.layout_catalog: dict = {}
        if "layout" in self.inputs:
            self.layout_catalog = load_catalog(self.raw.get("layout_catalog", DEFAULT_CATALOG), check_models=True)
        self._validate()

    def _validate(self) -> None:
        job_fields = set(JobRequest.model_fields)
        errors = []
        for seg in self.segments:
            task, protocol = seg["task"], seg["protocol"]
            ids = {i["id"] for i in self.inputs.get(task, [])}
            errors += [f"segment {seg['id']}: unknown {task} input '{i}'" for i in seg["inputs"] if i not in ids]
            for m in self.machines:
                where = f"settings {task}/{protocol}/{m['id']}"
                if not self.worker_url(m["id"], task):
                    errors.append(f"machine {m['id']}: no worker URL for task {task}")
                try:
                    s = self.settings(task, protocol, m["id"])
                except KeyError as e:
                    errors.append(e.args[0])
                    continue
                if s["model"] not in self.models:
                    errors.append(f"{where}: model '{s['model']}' is not in models.yaml")
                bad = set(s) - job_fields - {"mode", "prepare"}
                if bad:
                    errors.append(f"{where}: unknown keys {sorted(bad)}")
        for task, models in self.choices.items():
            for key, extra in models.items():
                where = f"model_choices {task}/{key}"
                if key not in self.models:
                    errors.append(f"{where}: not in models.yaml")
                elif self.models[key].get("task") != task:
                    errors.append(f"{where}: models.yaml says it is a {self.models[key].get('task')} model")
                bad = set(extra) - job_fields - {"mode", "prepare"} | ({"model"} & set(extra))
                if bad:
                    errors.append(f"{where}: unknown keys {sorted(bad)}")
        rooms = self.layout_catalog.get("rooms", {})
        for item in self.inputs.get("layout", []):
            if item.get("room_type") not in rooms:
                errors.append(f"layout input {item.get('id')}: room_type '{item.get('room_type')}' is not in "
                              f"the furniture catalogue (known: {', '.join(rooms)})")
        if errors:   # dict.fromkeys: segments sharing a task/protocol would repeat the same error
            raise ValueError(f"{self.path.name} is invalid:\n  " + "\n  ".join(dict.fromkeys(errors)))

    def machine(self, mid: str) -> dict:
        for m in self.machines:
            if m["id"] == mid:
                return m
        raise KeyError(f"no machine '{mid}'")

    def segment(self, sid: str) -> dict:
        for s in self.segments:
            if s["id"] == sid:
                return s
        raise KeyError(f"no segment '{sid}'")

    def input(self, task: str, input_id: str) -> dict:
        for item in self.inputs.get(task, []):
            if item["id"] == input_id:
                return dict(item)
        raise KeyError(f"no {task} input '{input_id}'")

    def add_input(self, task: str, item: dict) -> None:
        """A runtime input (an operator upload), offered like the configured ones; replaces one with the same id."""
        items = self.inputs.setdefault(task, [])
        items[:] = [x for x in items if x["id"] != item["id"]] + [item]

    def settings(self, task: str, protocol: str, mid: str, model: Optional[str] = None) -> dict[str, Any]:
        """The planned settings for one cell. `model` (the operator's pick) replaces the planned
        model and applies that model's own settings from model_choices (resolution, steps...);
        the protocol's keys (streams, temperature, budget behaviour) stay as planned, and so do
        the keys the block lists under keep_for_any_model."""
        block = (self.raw.get("settings") or {}).get(task, {}).get(protocol)
        if block is None:
            raise KeyError(f"no settings for {task}/{protocol}")
        merged = {**block.get("all", {}), **block.get(mid, {})}
        if "model" not in merged:
            raise KeyError(f"settings {task}/{protocol}/{mid} name no model")
        if model and model != merged["model"]:
            kept = {k: merged[k] for k in block.get("keep_for_any_model", []) if k in merged}
            merged = {**merged, **self.choices.get(task, {}).get(model, {}), "model": model, **kept}
        return merged

    def model_options(self, task: str) -> list[str]:
        """Every model the operator may pick for a task: the planned ones, then model_choices."""
        planned = [s.get("model") for block in ((self.raw.get("settings") or {}).get(task) or {}).values()
                   for s in block.values() if isinstance(s, dict) and s.get("model")]
        return list(dict.fromkeys(planned + list(self.choices.get(task, {}))))

    def model_label(self, key: str) -> str:
        return self.models.get(key, {}).get("label", key)

    def worker_url(self, mid: str, task: str) -> Optional[str]:
        return (self.machine(mid).get("workers") or {}).get(task)

    def public(self) -> dict:
        """What the operator page needs to draw its controls."""
        tasks = list(dict.fromkeys(s["task"] for s in self.segments))
        return {"machines": self.machines, "segments": self.segments, "inputs": self.inputs,
                "model_labels": {k: m.get("label", k) for k, m in self.models.items()},
                "model_options": {t: self.model_options(t) for t in tasks}}
