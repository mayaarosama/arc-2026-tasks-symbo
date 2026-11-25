"""Stop / bell against a stalled llama-server: the job must end at once, not after the 120 s read
timeout. A fake server stands in for llama-server (no binary, no model); the real LlamaServer and
LayoutServer code talks to it over HTTP."""
import json
import threading
import time

import httpx
import pytest

from arcdemo.schemas import JobRequest, JobResult
from arcdemo.worker.backends.base import JobContext
from arcdemo.worker.backends.layout import LayoutServer
from arcdemo.worker.backends.llm import LlamaServer
from arcdemo.worker.metrics import Clock

def backend(cls, url: str, tmp_path):
    b = cls("qwen35-4b", {"task": "llm", "file": "*.gguf"}, {"results_dir": str(tmp_path)}, ["cpu"])
    b.http = httpx.Client(base_url=url, trust_env=False, timeout=httpx.Timeout(120, connect=5))
    return b


def context(req: JobRequest, tmp_path) -> JobContext:
    res = JobResult(run_id=req.run_id, machine="test", task=req.task, model=req.model,
                    input_id=req.input_id, protocol=req.protocol)
    clock = Clock()
    clock.start()
    return JobContext(req, res, clock, tmp_path, ["cpu"], threading.Event())


def run_job(b, req, ctx, stop_after=None, limit=10.0):
    """generate() on a thread, as the worker does; optionally press Stop. -> (seconds, error)."""
    out = {}

    def go():
        try:
            b.generate(req, ctx)
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    t0 = time.perf_counter()
    t = threading.Thread(target=go, daemon=True)
    t.start()
    if stop_after is not None:
        time.sleep(stop_after)
        ctx.cancel.set()
    t.join(limit)
    assert not t.is_alive(), f"generate() still running after {limit} s: the stalled stream was not closed"
    return time.perf_counter() - t0, out.get("error")


LLM = dict(task="llm", model="qwen35-4b", input_id="llm-01", prompt="hi", max_tokens=64)


def test_stop_closes_stalled_llm_streams_at_once(fake_server, tmp_path):
    b = backend(LlamaServer, fake_server("tokens_then_stall"), tmp_path)
    req = JobRequest(**LLM, streams=4)
    ctx = context(req, tmp_path)
    seconds, err = run_job(b, req, ctx, stop_after=0.5)
    assert err is None, f"a stream closed by Stop must not count as a failure: {err!r}"
    assert seconds < 2.5
    assert ctx.result.tokens == 12                 # 3 tokens per stream before the stall still count
    assert not b._live                             # every stream let go


def test_bell_closes_stalled_llm_streams(fake_server, tmp_path):
    b = backend(LlamaServer, fake_server("tokens_then_stall"), tmp_path)
    req = JobRequest(**LLM, streams=2, budget_seconds=1.0, repeat_until_budget=True)
    ctx = context(req, tmp_path)
    seconds, err = run_job(b, req, ctx)            # nobody presses Stop: the budget ends it
    assert err is None, repr(err)
    assert 1.0 <= seconds < 3.0
    assert ctx.result.tokens == 6


def test_stop_while_server_has_sent_nothing_yet(fake_server, tmp_path):
    """Stop while llama-server is still reading the prompt and has not answered at all."""
    b = backend(LlamaServer, fake_server("stall_before_headers"), tmp_path)
    req = JobRequest(**LLM, streams=2)
    ctx = context(req, tmp_path)
    seconds, err = run_job(b, req, ctx, stop_after=0.5, limit=4.0)
    assert err is None, repr(err)
    assert seconds < 2.5
    assert not b._live


def test_server_that_answers_after_stop_is_hung_up_on(tmp_path):
    """The abandoned request's response arrives late: it must be closed, not left streaming."""
    import arcdemo.worker.backends.llm as llm

    gate, closed = threading.Event(), threading.Event()

    class LateResp:
        def close(self):
            closed.set()

    class SlowClient:
        def build_request(self, *a, **k):
            return None

        def send(self, req, stream):
            gate.wait(5)                           # the server answers only after the Stop
            return LateResp()

    b = llm.LlamaServer("qwen35-4b", {"task": "llm"}, {"results_dir": str(tmp_path)}, ["cpu"])
    b.http = SlowClient()
    out = {}

    def call():
        try:
            out["value"] = b._open({})
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=call, daemon=True)
    t.start()
    time.sleep(0.2)
    b._abort_streams()
    t.join(2)
    assert not t.is_alive()
    assert isinstance(out.get("error"), llm.StreamAborted), out
    gate.set()
    assert closed.wait(2)


def test_a_real_stream_error_still_fails_the_job(fake_server, tmp_path):
    b = backend(LlamaServer, fake_server("error"), tmp_path)
    req = JobRequest(**LLM, streams=2)
    seconds, err = run_job(b, req, context(req, tmp_path))
    assert isinstance(err, httpx.HTTPStatusError) and err.response.status_code == 500


def test_stop_closes_stalled_layout_streams(fake_server, tmp_path):
    b = backend(LayoutServer, fake_server("tokens_then_stall"), tmp_path)
    req = JobRequest(task="layout", model="qwen35-4b-layout", input_id="layout-01", streams=3, max_tokens=256,
                     room_type="bedroom", floor_plan=[[0, 0], [4, 0], [4, 3], [0, 3]], door=[3.5, 0],
                     max_objects=6)
    ctx = context(req, tmp_path)
    seconds, err = run_job(b, req, ctx, stop_after=0.5)
    assert err is None, repr(err)
    assert seconds < 2.5
    assert ctx.result.tokens == 9
