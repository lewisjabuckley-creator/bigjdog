"""Capability checks with a real instruction-tuned model (set JARVIS_TEST_MODEL, e.g. llama3.1:8b).

These run the same end-to-end checks as ``jarvis doctor --live``. Infrastructure failures fail the test;
a model that doesn't follow an instruction is reported as a warning, because that is a property of the
model, not of JARVIS.
"""

from __future__ import annotations

import os

import pytest

from jarvis.diagnostics import live_checks
from tests.integration.test_live_conversation import live_config

MODEL = os.environ.get("JARVIS_TEST_MODEL")


@pytest.mark.skipif(not MODEL, reason="set JARVIS_TEST_MODEL to an installed instruction model")
async def test_real_model_end_to_end(tmp_path, ollama_url):
    results = await live_checks(live_config(tmp_path, ollama_url), model=MODEL, level="full",
                                workdir=tmp_path / "live", report=lambda r: print(r.line()))
    failed = [r.line() for r in results if r.status == "FAIL"]
    assert not failed, "\n".join(failed)
    warned = [r.line() for r in results if r.status == "WARN"]
    if warned:
        print("model did not follow some instructions:\n" + "\n".join(warned))
