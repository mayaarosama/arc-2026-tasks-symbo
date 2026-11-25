"""Task 2 — image to 3D: GSO inputs scored against their scans, and operator uploads.

Uploads: an operator's picture is checked, cut out and normalised once, here (arcdemo/imageprep.py),
so every machine gets the same input. Kept in session_dir/uploads/<id>.png (+ the original) and listed
in uploads.json, so a conductor restart still offers them.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from ...imageprep import INPUT_SPEC, MAX_BYTES, Rejected, cutout_model_path, prepare
from ...scoring import score_files
from ..config import ROOT
from .base import TaskHandler

log = logging.getLogger("arcdemo.conductor")
MEDIA_SESSION = "/media"                 # where the web app serves session_dir (app.py mounts it)
UPLOAD_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
UPLOAD_LABEL_CHARS = 40
UPLOAD_DIGEST_CHARS = 10


class UploadBody(BaseModel):
    name: str = "picture"
    data_url: str           # data:image/...;base64,... (the browser's FileReader), up to ~14 MB


def uploads_dir(cond) -> Path:
    return cond.cfg.session_dir / "uploads"


def load_uploads(cond) -> list[dict]:
    index = uploads_dir(cond) / "uploads.json"
    try:
        items = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    except (OSError, ValueError):
        log.warning("ignoring unreadable %s", index)
        return []
    return [x for x in items if (cond.cfg.session_dir / x.get("upload", "-")).is_file()]


def add_upload(cond, name: str, data: bytes) -> dict:
    """Check, cut out and normalise an operator's picture, and offer it as an image-to-3D input.
    Raises imageprep.Rejected with the reason. -> the input item + warnings"""
    prep = prepare(data)
    digest = hashlib.sha1(data).hexdigest()[:UPLOAD_DIGEST_CHARS]
    folder = uploads_dir(cond)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{digest}.png").write_bytes(prep.png)
    suffix = Path(name).suffix.lower()
    ext = suffix if suffix in UPLOAD_EXTENSIONS else ".bin"
    (folder / f"{digest}-original{ext}").write_bytes(data)
    label = re.sub(r"[^\w .()-]+", " ", Path(name).stem).strip()[:UPLOAD_LABEL_CHARS] or "Uploaded picture"
    item = {"id": f"up-{digest}", "label": label, "tier": "upload",
            "exposes": "An operator's own picture: live only (no recording, no reference mesh)",
            "upload": f"uploads/{digest}.png", "input_url": f"{MEDIA_SESSION}/uploads/{digest}.png",
            "original_url": f"{MEDIA_SESSION}/uploads/{digest}-original{ext}",
            "seed": int(digest[:6], 16), "cutout": prep.cutout, "original_size": list(prep.original_size)}
    cond.cfg.add_input("i23d", item)
    uploads = [x for x in load_uploads(cond) if x["id"] != item["id"]] + [item]
    (folder / "uploads.json").write_text(json.dumps(uploads, indent=1), encoding="utf-8")
    log.info("upload %s (%s, %s x %s, cut-out %s)", item["id"], label, *prep.original_size, prep.cutout)
    return {**item, "warnings": prep.warnings}


class ImageTo3D(TaskHandler):
    task = "i23d"

    def on_start(self, cond) -> None:
        for item in load_uploads(cond):
            cond.cfg.add_input("i23d", item)

    def caption(self, item: dict, caption: str) -> str:
        if item.get("tier") == "upload":          # the segment's caption may promise a score
            return (caption + " This picture was brought today: there is no scanned original, "
                    "so no error is measured.").strip()
        return caption

    def job_fields(self, cond, item: dict, settings: dict) -> dict[str, Any]:
        if item.get("upload"):     # an operator upload: the workers do not have it, so send the picture
            return {"image_b64": base64.b64encode((cond.cfg.session_dir / item["upload"]).read_bytes()).decode()}
        return {"image_path": item["input"]}

    def budget_note(self, res: dict, budget: float) -> str:
        if res.get("completed", 0):              # 3D budgets are soft: the mesh lands after the bell
            return f"Finished after the bell, at {res['all_outputs_seconds']:.1f} s"
        return super().budget_note(res, budget)

    async def after_result(self, cond, rnd: dict, mid: str) -> None:
        """Chamfer / F-score of the mesh against the scanned original (none for uploads)."""
        truth = rnd["input"].get("truth")
        cell = rnd["cells"][mid]
        if not truth or not (ROOT / truth).exists() or not cell["outputs"]:
            return
        cell["score"] = {"pending": True}
        try:
            sc = await asyncio.to_thread(score_files, cell["outputs"][0]["path"], str(ROOT / truth))
            cell["score"] = sc.as_dict()
        except Exception as e:  # noqa: BLE001
            cell["score"] = {"error": type(e).__name__}
            log.exception("scoring %s failed", mid)

    def routes(self, app, cfg, get_cond) -> None:
        @app.get("/api/i23d/spec")
        def upload_spec() -> dict:
            """What an uploaded picture must be like (shown next to the upload box)."""
            return {"spec": [{"what": a, "rule": b} for a, b in INPUT_SPEC],
                    "cutout_installed": cutout_model_path() is not None}

        @app.post("/api/i23d/upload")
        async def upload(body: UploadBody) -> dict:
            """An operator's picture for image-to-3D. 422 with the reason when it cannot be used."""
            head, _, b64 = body.data_url.partition(",")
            if not head.startswith("data:") or ";base64" not in head:
                raise HTTPException(400, "expected a data URL (data:image/...;base64,...)")
            if len(b64) > MAX_BYTES * 4 // 3 + 16:        # base64 is 4/3 the size, plus padding
                raise HTTPException(422, f"The file is larger than {MAX_BYTES // 2**20} MB.")
            try:
                data = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(400, "the picture is not valid base64")
            try:   # the cut-out takes a second or two: keep the conductor's loop (health, rounds) running
                return await asyncio.to_thread(add_upload, get_cond(), body.name, data)
            except Rejected as e:
                raise HTTPException(422, str(e))
