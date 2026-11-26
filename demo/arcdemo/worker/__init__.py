"""The worker: one process per task venv, serving one or more models over HTTP."""
import os as _os

from ..defaults import DEFAULTS as _DEFAULTS

# Before torch is imported (by worker.app and the backends): importing it with diffusers otherwise
# starts torch's compile-worker pool, 16 idle processes, though the demo never compiles anything.
# A TORCHINDUCTOR_COMPILE_THREADS set in the shell still wins.
_os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(_DEFAULTS.worker.torch_compile_threads))
