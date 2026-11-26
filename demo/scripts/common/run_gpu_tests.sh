#!/usr/bin/env bash
# Run the real-model tests (pytest -m gpu, tests/test_gpu_real.py) in every task venv on this machine.
#
#     scripts/common/run_gpu_tests.sh            # t2i, i23d, music, llm venvs in turn
#     scripts/common/run_gpu_tests.sh t2i llm    # only these
#
# The task venvs do not ship pytest; it is installed once into demo/.pytest-overlay (gitignored) and
# put on PYTHONPATH, so the venvs themselves are never changed. Each task's test skips itself in the
# other venvs. Exit code: 0 when every venv passed (or skipped), 1 otherwise.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1          # demo/
OVERLAY="$PWD/.pytest-overlay"
if [ ! -d "$OVERLAY/pytest" ]; then
    .venv/bin/python -m pip install --quiet --no-deps --target "$OVERLAY" pytest pluggy iniconfig pygments || exit 1
fi
status=0
for task in "${@:-t2i i23d music llm}"; do
    for t in $task; do
        py=".venv-$t/bin/python"
        if [ ! -x "$py" ]; then echo "== $t: no $py, skipped"; continue; fi
        echo "== $t ($py)"
        PYTHONPATH="$OVERLAY" "$py" -m pytest -m gpu -q -p no:cacheprovider tests/test_gpu_real.py || status=1
    done
done
exit $status
