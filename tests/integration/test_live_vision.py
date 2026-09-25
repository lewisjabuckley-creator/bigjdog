"""Phase 4 against a real Ollama server: vision capability detection, routing, images passed through the real image
encoder, failures surfaced, fallback between vision models, and vision work kept to one at a time.

Uses "vision puppets" (see puppet.py): a puppet language model plus a tiny real CLIP projector, so Ollama reports the
vision capability and runs every image through llama.cpp's image encoder, while the reply stays scripted and every
assertion is exact. No model download is needed.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

import httpx
import pytest

from jarvis.config import config_from_dict
from jarvis.models.base import Capability, ChatMessage
from jarvis.perception.images import solid_png
from jarvis.perception.inputs import Attachment
from jarvis.runtime import Runtime

WORK = Path(tempfile.mkdtemp(prefix="jarvis-live-vision-"))


@pytest.fixture(scope="module")
def vision_puppets(ollama_url):
    pytest.importorskip("gguf", reason="pip install gguf numpy to build puppet models")
    from tests.integration.puppet import Script, create_puppet, create_vision_puppet
    names = {
        "a": create_vision_puppet(ollama_url, "jarvis-puppet-p4-vision-a:latest",
                                  Script("VISION-A: a red bar runs along the bottom.", "VISION-A"), WORK),
        "b": create_vision_puppet(ollama_url, "jarvis-puppet-p4-vision-b:latest",
                                  Script("VISION-B: a red bar runs along the bottom.", "VISION-B"), WORK),
        "text": create_puppet(ollama_url, "jarvis-puppet-p4-text:latest",
                              Script("TEXT-ANSWER: plain text.", "TEXT-ANSWER: plain text."), WORK),
    }
    yield names
    with httpx.Client(base_url=ollama_url, timeout=30, trust_env=False) as client:
        for name in names.values():
            client.request("DELETE", "/api/delete", json={"model": name})


def _runtime(tmp_path: Path, ollama_url: str, profiles: dict[str, list[str]]) -> Runtime:
    config = {"general": {"data_dir": str(tmp_path / "data")},
              "models": {"ollama": {"base_url": ollama_url, "num_ctx": 2048}, "profiles": profiles},
              "permissions": {"allowed_roots": [str(tmp_path), str(WORK)]}, "monitoring": {"enabled": False},
              "tasks": {"scheduler_interval_s": 0.05}, "scheduler": {"tick_s": 0.1}, "runtime": {"heartbeat_s": 0.2},
              "perception": {"ocr": "off"}}
    return Runtime(config_from_dict(config), mode="embedded")


def _picture() -> bytes:
    return solid_png(64, 48, (255, 255, 255), [(0, 40, 64, 8, (220, 0, 0))])


async def test_vision_models_are_detected_and_images_go_only_to_them(tmp_path, ollama_url, vision_puppets):
    rt = _runtime(tmp_path, ollama_url, {"vision": [vision_puppets["a"]], "conversation": [vision_puppets["text"]]})
    await rt.start()
    try:
        router = rt.svc.router
        await router.refresh()
        a, text = router.find(vision_puppets["a"]), router.find(vision_puppets["text"])
        assert a is not None and a.has(Capability.VISION)                  # detected from Ollama's /api/show
        assert text is not None and not text.has(Capability.VISION)
        assert "Vision model: AVAILABLE" in await rt.svc.perception.capabilities_text()

        o = rt.orchestrator()
        plain = await o.handle("What's my CPU usage?")
        assert plain.intent.value == "system_query" and plain.model is None        # text never touches vision

        seen = await o.handle("What's in this image?", attachments=[Attachment(name="bar.png", data=_picture())])
        assert seen.intent.value == "perceive" and seen.kind == "answer", seen.text
        assert "VISION-A" in seen.text and seen.model == vision_puppets["a"]
        assert any(p.kind.value == "vision_model" for p in seen.provenance)

        # the image really reaches the model: the same request with the image is longer in the model's own count
        provider = router.providers["ollama"]
        b64 = base64.b64encode(_picture()).decode()
        with_image = await provider.chat(vision_puppets["a"], [ChatMessage("user", "What is this?", images=[b64])])
        without = await provider.chat(vision_puppets["a"], [ChatMessage("user", "What is this?")])
        assert with_image.prompt_tokens and without.prompt_tokens and with_image.prompt_tokens > without.prompt_tokens
    finally:
        await rt.stop()


async def test_a_broken_image_is_refused_and_a_server_failure_is_reported_not_hidden(tmp_path, ollama_url,
                                                                                      vision_puppets):
    rt = _runtime(tmp_path, ollama_url, {"vision": [vision_puppets["a"], vision_puppets["b"]]})
    await rt.start()
    try:
        o = rt.orchestrator()
        refused = await o.handle("What's this?", attachments=[Attachment(name="x.png", data=b"\x89PNG\r\n\x1a\nno")])
        assert refused.kind == "error" and "damaged" in refused.text             # caught before any model call
        # a PNG whose header is fine but whose pixels can't be decoded: only the server can tell
        good = _picture()
        idat = good.index(b"IDAT")
        broken = good[:idat + 4] + b"\x00" * 12 + good[idat + 16:]
        seen = await o.handle("What's this?", attachments=[Attachment(name="odd.png", data=broken)])
        assert seen.kind == "error" and "couldn't analyse" in seen.text, seen.text
        assert "VISION-" not in seen.text                                        # nothing pretended
        failure = rt.svc.router.last_failure or {}
        assert "Failed to load image" in failure.get("error", "")                # the server's own refusal, logged
        assert rt.svc.router.find(failure["model"]).has(Capability.VISION)      # and only vision models were tried
    finally:
        await rt.stop()


async def test_a_failed_vision_model_falls_back_to_another_vision_model(tmp_path, ollama_url, vision_puppets):
    from tests.integration.puppet import Script, create_vision_puppet
    doomed = create_vision_puppet(ollama_url, "jarvis-puppet-p4-vision-doomed:latest",
                                  Script("DOOMED", "DOOMED"), WORK)
    rt = _runtime(tmp_path, ollama_url, {"vision": [doomed, vision_puppets["b"]],
                                         "conversation": [vision_puppets["text"]]})
    await rt.start()
    try:
        await rt.svc.router.refresh()
        assert rt.svc.router.find(doomed) is not None
        with httpx.Client(base_url=ollama_url, timeout=30, trust_env=False) as client:
            client.request("DELETE", "/api/delete", json={"model": doomed})     # gone after JARVIS listed it
        seen = await rt.orchestrator().handle("Describe it.", attachments=[Attachment(name="bar.png", data=_picture())])
        assert "VISION-B" in seen.text and seen.model == vision_puppets["b"], seen.text
        assert "instead" in seen.footnote                                       # the fallback is said, not hidden
        assert rt.svc.router.find(vision_puppets["text"]) is not None and seen.model != vision_puppets["text"]
    finally:
        await rt.stop()


async def test_vision_work_runs_one_at_a_time(tmp_path, ollama_url, vision_puppets):
    rt = _runtime(tmp_path, ollama_url, {"vision": [vision_puppets["a"]]})
    await rt.start()
    try:
        p = rt.svc.perception
        obs, _ = p.ingest([Attachment(name=f"{i}.png", data=solid_png(40 + i, 30, (i * 40, 0, 0))) for i in range(3)])
        peak = 0

        async def watch() -> None:
            nonlocal peak
            while True:
                peak = max(peak, p.vision.active)
                await asyncio.sleep(0.005)
        watcher = asyncio.create_task(watch())
        results = await asyncio.gather(*(p.understand([o], "What is it?") for o in obs))
        watcher.cancel()
        assert all("VISION-A" in r.text for r in results) and peak == 1         # perception.max_concurrent_vision
    finally:
        await rt.stop()
