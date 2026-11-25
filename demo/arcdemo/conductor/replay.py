"""Replay table: one prerecorded output + its real rehearsal timing per demo cell.

Keyed on task / protocol / machine / input_id, with a hash of the settings stored alongside
so a record made under different settings is flagged as stale rather than silently reused.
Layout:  replay/<task>/<protocol>/<machine>/<input_id>/record.json  (+ output files)
         replay/<task>/<protocol>/<machine>/<input_id>/full/record.json   run to completion, no
         time limit (scripts/common/rehearse.py --full): what a slow machine shows when its time is up
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional


def settings_hash(settings: dict) -> str:
    clean = {k: v for k, v in settings.items() if k not in ("mode", "prepare")}
    return hashlib.sha1(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:10]


class ReplayStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def cell_dir(self, task: str, protocol: str, machine: str, input_id: str, variant: str = "") -> Path:
        d = self.root / task / protocol / machine / input_id
        return d / variant if variant else d

    def get(self, task: str, protocol: str, machine: str, input_id: str,
            settings: Optional[dict] = None, variant: str = "") -> Optional[dict]:
        d = self.cell_dir(task, protocol, machine, input_id, variant)
        f = d / "record.json"
        if not f.exists():
            return None
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec["_dir"] = str(d)
        rec["stale"] = bool(settings is not None and rec.get("settings_hash") != settings_hash(settings))
        rec["variant"] = variant
        return rec

    def get_full(self, task: str, protocol: str, machine: str, input_id: str,
                 settings: Optional[dict] = None) -> Optional[dict]:
        """A recording that ran to completion: the dedicated full run, else the normal record
        if that one happens to be complete (e.g. iso-quality, which has no time limit)."""
        rec = self.get(task, protocol, machine, input_id, settings, variant="full")
        if rec:
            return rec
        rec = self.get(task, protocol, machine, input_id, settings)
        if rec and rec["result"].get("status") == "ok" and rec["files"]:
            return rec
        return None

    def save(self, task: str, protocol: str, machine: str, input_id: str, settings: dict,
             result: dict, files: list[Path], extra: Optional[dict] = None, variant: str = "") -> Path:
        d = self.cell_dir(task, protocol, machine, input_id, variant)
        if d.exists():
            if variant:
                shutil.rmtree(d)
            else:   # keep the full/ sub-record when the normal record is re-rehearsed
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
        d.mkdir(parents=True, exist_ok=True)
        names = []
        for f in files:
            shutil.copy2(f, d / f.name)
            names.append(f.name)
        rec = {"task": task, "protocol": protocol, "machine": machine, "input_id": input_id,
               "settings": settings, "settings_hash": settings_hash(settings),
               "files": names, "result": result, **(extra or {})}
        (d / "record.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return d
