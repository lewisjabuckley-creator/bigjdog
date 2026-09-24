"""Phase 2: the local API — authentication, the endpoints interfaces use, streaming, presence, and turns that
are never handled twice."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

from jarvis.service.api import ApiServer
from jarvis.tasks.models import TaskStatus
from tests.helpers import make_runtime, wait_until

S = TaskStatus
TOKEN = "test-token-123"


@asynccontextmanager
async def api(tmp_path, **overrides: Any) -> AsyncIterator[tuple[httpx.AsyncClient, Any, Any, ApiServer]]:
    rt, sim = make_runtime(str(tmp_path), mode="daemon", **overrides)
    await rt.start()
    stopped = asyncio.Event()
    server = ApiServer(rt, TOKEN, on_stop=stopped.set, sim=sim)
    port = await server.start()
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {TOKEN}"},
                               trust_env=False, timeout=20)
    try:
        yield client, rt, sim, server
    finally:
        await client.aclose()
        await server.stop()
        await rt.stop()


async def _ndjson(client: httpx.AsyncClient, method: str, path: str, **kw: Any) -> list[dict[str, Any]]:
    items = []
    async with client.stream(method, path, **kw) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.strip():
                items.append(json.loads(line))
    return items


async def test_every_request_needs_the_token_and_browsers_are_refused(tmp_path):
    async with api(tmp_path) as (client, rt, sim, server):
        base = str(client.base_url)
        async with httpx.AsyncClient(base_url=base, trust_env=False) as anonymous:
            assert (await anonymous.get("/v1/ping")).json()["ok"] is True          # liveness only
            assert (await anonymous.get("/v1/status")).status_code == 401
            wrong = await anonymous.get("/v1/tasks", headers={"Authorization": "Bearer nope"})
            assert wrong.status_code == 401
        browser = await client.get("/v1/status", headers={"Origin": "http://evil.example"})
        assert browser.status_code == 403
        assert (await client.get("/v1/nope")).status_code == 404
        assert (await client.delete("/v1/tasks")).status_code == 405
        assert server.host == "127.0.0.1"


async def test_status_health_and_state(tmp_path):
    async with api(tmp_path) as (client, rt, sim, server):
        await wait_until(lambda: "scheduler" in rt.loop_ticks())
        status = (await client.get("/v1/status")).json()
        assert status["runtime"]["mode"] == "daemon" and "SYSTEM" in status["text"]
        health = (await client.get("/v1/health")).json()
        assert health["overall"] == "healthy" and "ollama" in health["components"]
        state = (await client.get("/v1/state")).json()
        assert state["resources"]["memory_percent"] == 40.0 and state["presence"]["away"] is True


async def test_tasks_create_get_pause_resume_cancel(tmp_path):
    async with api(tmp_path) as (client, rt, sim, server):
        rt.svc.permissions.grant("*", 4, tools=["shell_execute"])
        body = {"objective": "wait a bit", "steps": [{"tool": "shell_execute", "args": {"command": "sleep 5"}}],
                "idempotency_key": "client-42"}
        first = await client.post("/v1/tasks", json=body)
        again = await client.post("/v1/tasks", json=body)                   # a retried request
        assert first.status_code == 201 and again.status_code == 200
        task_id = first.json()["task"]["id"]
        assert again.json()["task"]["id"] == task_id and again.json()["created"] is False
        assert first.json()["task"]["origin"] == "api" and first.json()["task"]["permissions"]["interactive"]
        await rt.svc.pool.wait_for(task_id, [S.RUNNING])
        assert (await client.post(f"/v1/tasks/{task_id}/pause")).json()["ok"]
        await rt.svc.pool.wait_for(task_id, [S.PAUSED])
        assert (await client.get(f"/v1/tasks/{task_id}")).json()["task"]["status"] == "paused"
        assert (await client.post(f"/v1/tasks/{task_id}/resume")).json()["ok"]
        assert (await client.post(f"/v1/tasks/{task_id}/cancel")).json()["ok"]
        done = await rt.svc.pool.wait_for(task_id)
        assert done.status == S.CANCELLED
        listing = (await client.get("/v1/tasks", params={"status": "all"})).json()["tasks"]
        assert [t["id"] for t in listing] == [task_id]
        bad = await client.post("/v1/tasks", json={"objective": "x", "steps": [{"tool": "no_such_tool"}]})
        assert bad.status_code == 400 and "unknown tool" in bad.json()["error"]
        assert (await client.get("/v1/tasks/task-missing")).status_code == 404


async def test_events_and_notifications_with_acknowledgement(tmp_path):
    async with api(tmp_path) as (client, rt, sim, server):
        n = rt.svc.notifications.notify(3, "Build failed", "2 tests failed", source="test")
        await rt.svc.bus.drain()
        events = (await client.get("/v1/events", params={"types": "NOTIFICATION_CREATED"})).json()["events"]
        assert events and events[0]["payload"]["id"] == n.id
        queued = (await client.get("/v1/notifications", params={"state": "queued"})).json()["notifications"]
        assert [q["id"] for q in queued] == [n.id] and queued[0]["text"] == "Build failed. 2 tests failed"
        assert (await client.post(f"/v1/notifications/{n.id}/ack")).json()["acknowledged"] == 1
        assert (await client.post(f"/v1/notifications/{n.id}/ack")).status_code == 404
        await rt.svc.bus.drain()
        acked = (await client.get("/v1/events", params={"types": "NOTIFICATION_ACKNOWLEDGED"})).json()["events"]
        assert acked and acked[0]["payload"]["ids"] == [n.id]


async def test_conversation_turn_is_never_handled_twice(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("print(1)\n")
    async with api(tmp_path) as (client, rt, sim, server):
        sim.provider.when("Analyze the software project", "REPORT: tiny.")
        body = {"text": "Analyze this project.", "request_id": "req-abc", "cwd": str(project)}
        first = (await client.post("/v1/conversation", json=body)).json()
        again = (await client.post("/v1/conversation", json=body)).json()     # the client retried
        assert first["replayed"] is False and again["replayed"] is True
        assert first["response"]["task_id"] == again["response"]["task_id"]
        analyses = [t for t in rt.svc.tasks.list_tasks(limit=50) if t.title.startswith("Analyze")]
        assert len(analyses) == 1 and analyses[0].origin == "conversation:default"
        done = await rt.svc.pool.wait_for(analyses[0].id)
        assert done.result == "REPORT: tiny."
        stored = (await client.get("/v1/conversation/requests/req-abc")).json()
        assert stored["status"] == "done" and stored["response"]["task_id"] == done.id


async def test_streamed_turn_finishes_even_if_the_interface_disconnects(tmp_path):
    async with api(tmp_path) as (client, rt, sim, server):
        sim.provider.when("tell me about disks", "Disks store data. Some are fast, some are slow, all fail eventually.")
        items = await _ndjson(client, "POST", "/v1/conversation",
                              json={"text": "tell me about disks", "stream": True, "request_id": "r1"})
        tokens = "".join(i["text"] for i in items if i["type"] == "token")
        final = items[-1]
        assert final["type"] == "response" and final["response"]["streamed"] and tokens == final["response"]["text"]
        # now drop the connection as soon as the answer starts
        async with client.stream("POST", "/v1/conversation",
                                 json={"text": "tell me about disks", "stream": True, "request_id": "r2"}) as resp:
            async for _ in resp.aiter_lines():
                break
        await wait_until(lambda: rt.svc.db.query_one("SELECT status FROM requests WHERE id='r2'")["status"] == "done")
        replay = (await client.post("/v1/conversation", json={"text": "ignored", "request_id": "r2"})).json()
        assert replay["replayed"] and replay["response"]["text"].startswith("Disks store data")


async def test_attach_stream_and_detach_track_presence(tmp_path):
    async with api(tmp_path) as (client, rt, sim, server):
        svc = rt.svc
        rt.svc.notifications.notify(3, "Queued while away", source="test")
        att = (await client.post("/v1/sessions/attach", json={"kind": "cli"})).json()
        assert att["first"] and att["returning"]["notifications"] == ["Queued while away"]
        assert not svc.notifications.pending()                          # shown on return
        cid = att["client_id"]
        received: list[dict[str, Any]] = []

        async def listen() -> None:
            async with client.stream("GET", "/v1/stream", params={"client_id": cid}) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        received.append(json.loads(line))
                        if received[-1]["type"] == "notification":
                            return

        listener = asyncio.create_task(listen())
        await wait_until(lambda: received and received[0]["type"] == "hello")
        live = svc.notifications.notify(3, "Delivered live", source="test")
        await asyncio.wait_for(listener, 5)
        assert received[-1]["text"] == "Delivered live" and live.state == "delivered"
        await wait_until(lambda: svc.presence.away)           # the stream closed: the user is away again
        detached = [e for e in svc.bus.recent if e.type == "UI_DETACHED"]
        assert detached
        again = (await client.post("/v1/sessions/attach", json={"kind": "cli"})).json()
        assert again["returning"] is not None
        assert (await client.post("/v1/sessions/detach", json={"client_id": again["client_id"]})).json()["detached"]


async def test_away_briefing_and_schedules_endpoints(tmp_path):
    async with api(tmp_path) as (client, rt, sim, server):
        away = (await client.get("/v1/away")).json()
        assert away["text"].startswith("While you were away") and "report" in away
        assert (await client.get("/v1/briefing")).status_code == 404
        made = (await client.post("/v1/briefing")).json()
        assert made["text"].startswith("It's") and (await client.get("/v1/briefing")).json()["id"] == made["id"]
        created = await client.post("/v1/schedules", json={
            "name": "nightly", "schedule": {"type": "daily", "at": "02:00"},
            "action": {"type": "task", "objective": "nightly check", "steps": [{"tool": "time_now"}]}})
        assert created.status_code == 201
        sid = created.json()["schedule"]["id"]
        assert created.json()["schedule"]["describe"] == "every day at 02:00"
        assert (await client.post(f"/v1/schedules/{sid}/disable")).json()["schedule"]["enabled"] is False
        assert (await client.post(f"/v1/schedules/{sid}/enable")).json()["schedule"]["enabled"] is True
        bad = await client.post("/v1/schedules", json={"name": "x", "schedule": {"type": "daily", "at": "99:00"},
                                                       "action": {"type": "task", "objective": "x"}})
        assert bad.status_code == 400
        assert (await client.delete(f"/v1/schedules/{sid}")).json()["deleted"]
        assert (await client.get("/v1/schedules")).json()["schedules"] == []


async def test_stop_request_reaches_the_daemon(tmp_path):
    async with api(tmp_path) as (client, rt, sim, server):
        assert (await client.post("/v1/runtime/stop")).status_code == 202
        await wait_until(lambda: server.on_stop is not None and rt.started)
        await asyncio.sleep(0.2)
