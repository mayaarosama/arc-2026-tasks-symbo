# scripts/i23d/: Task 2 — image to 3D

Scripts run as `python -m scripts.i23d.<name>` from `demo/`; shell scripts by path. The full list of scripts, and what each folder is for, is in [../README.md](../README.md).

| File | Runs where | What |
| --- | --- | --- |
| `prepare_gso.py` | once, anywhere (needs `requirements/gso.txt`) | Downloads the Google Scanned Objects models and renders the stage inputs → `data/i23d/` |
| `fit_upload.py` | anywhere | Rescales a photo to meet the upload rules (upright, 512 px, PNG) |
