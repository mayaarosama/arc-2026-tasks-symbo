"""ARC symposium hardware demo: workers, conductor, scoring."""
import os as _os
from pathlib import Path as _Path

# Model weights live in the project (demo/hf-cache/hub), not the user's C: profile.
# Set before any huggingface_hub import; an HF_HUB_CACHE set in the shell still wins.
_os.environ.setdefault("HF_HUB_CACHE", str(_Path(__file__).resolve().parents[1] / "hf-cache" / "hub"))
