"""House rules checked on every run.

1. Imports are at the top of each file: no import inside a function, and no statement before the
   last top-level import (a guarded `try: import x / except ImportError` block counts as an import).
   The deferred imports below are the documented exceptions, each with its reason in the code.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.helpers import ROOT

SOURCES = sorted(p for folder in ("arcdemo", "scripts", "tests") for p in (ROOT / folder).rglob("*.py"))

# file -> modules it may import inside a function, and why
DEFERRED = {
    "arcdemo/worker/backends/trellis.py": {"trellis2", "trellis", "o_voxel"},   # env vars set before import
    "arcdemo/worker/backends/sf3d.py": {"sf3d"},                                # third_party clone path fallback
    "scripts/common/download_weights.py": {"huggingface_hub"},                  # after arcdemo sets HF_HUB_CACHE
    "scripts/gpu/plan_downloads.py": {"huggingface_hub"},                       # after arcdemo sets HF_HUB_CACHE
}


def _root(node: ast.AST) -> str:
    if isinstance(node, ast.ImportFrom):
        return "." if node.level else node.module.split(".")[0]
    return node.names[0].name.split(".")[0]


def _is_import_block(node: ast.AST) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    return isinstance(node, ast.Try) and all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in node.body)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_imports_are_at_the_top(path):
    rel = path.relative_to(ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    inside = [f"line {sub.lineno}: {_root(sub)}" for fn in ast.walk(tree)
              if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
              for sub in ast.walk(fn) if isinstance(sub, (ast.Import, ast.ImportFrom))
              and _root(sub) not in DEFERRED.get(rel, set())]
    assert not inside, f"{rel}: imports inside functions: {inside}"
    body = tree.body
    last = max((i for i, n in enumerate(body) if _is_import_block(n)), default=-1)
    early = [f"line {n.lineno}" for n in body[:last]
             if not _is_import_block(n) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert not early, f"{rel}: code before the last import: {early}"
