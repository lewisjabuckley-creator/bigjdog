"""Local API: how interfaces talk to the persistent runtime.

HTTP/1.1 with JSON on the loopback interface only, authenticated with a bearer
token that the runtime writes to a file only the user can read. Built on the
standard library's asyncio streams (no web framework). Streaming responses
(conversation tokens, the live notification stream) use chunked transfer of
newline-delimited JSON.

Every action requested through the API goes through the same pipeline as a
typed command: request → authorization (the token) → permission (the
permission manager, per tool call) → execution (the tool registry) →
verification → audit. The API adds no privileges of its own.

Browsers are refused (any request with an ``Origin`` header), so a web page
cannot drive the runtime even though it listens on localhost.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from urllib.parse import parse_qs, unquote, urlsplit

from jarvis import __version__
from jarvis.core import reports
from jarvis.core.awareness import away_report, latest_briefing, prepare_briefing
from jarvis.core.types import Priority, Severity
from jarvis.log import get_logger
from jarvis.notifications.manager import Notification
from jarvis.service.conversations import Conversations
from jarvis.service.status import health_report, state_snapshot
from jarvis.tasks.models import OPEN, TaskStatus

if TYPE_CHECKING:
    from jarvis.runtime import Runtime

log = get_logger("api")

MAX_BODY = 1_000_000
_REASONS = {200: "OK", 201: "Created", 202: "Accepted", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
            404: "Not Found", 405: "Method Not Allowed", 409: "Conflict", 413: "Payload Too Large",
            500: "Internal Server Error", 503: "Service Unavailable"}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: bytes
    groups: tuple[str, ...] = ()
    writer: asyncio.StreamWriter | None = field(default=None, repr=False)
    reader: asyncio.StreamReader | None = field(default=None, repr=False)

    def json(self) -> dict[str, Any]:
        if not self.body:
            return {}
        try:
            data = json.loads(self.body)
        except ValueError:
            raise ApiError(400, "body is not valid JSON") from None
        if not isinstance(data, dict):
            raise ApiError(400, "body must be a JSON object")
        return data

    def int(self, name: str, default: int, maximum: int = 1000) -> int:
        try:
            return max(1, min(int(self.query.get(name, default)), maximum))
        except ValueError:
            raise ApiError(400, f"{name} must be an integer") from None


Handler = Callable[[Request], Awaitable[Any]]


class Streamer:
    """Chunked NDJSON writer. ``send`` raises ConnectionError once the client has gone."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.started = False

    async def start(self) -> None:
        self.writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n"
                          b"Cache-Control: no-store\r\nConnection: close\r\n\r\n")
        await self.writer.drain()
        self.started = True

    async def send(self, obj: dict[str, Any]) -> None:
        data = (json.dumps(obj, default=str) + "\n").encode()
        self.writer.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        await self.writer.drain()

    async def end(self) -> None:
        self.writer.write(b"0\r\n\r\n")
        await self.writer.drain()


class ApiServer:
    def __init__(self, runtime: "Runtime", token: str, *, host: str = "127.0.0.1", port: int = 0,
                 on_stop: Callable[[], None] | None = None, sim: Any = None) -> None:
        self.runtime = runtime
        self.token = token
        self.host = host
        self.port = port
        self.on_stop = on_stop
        self.sim = sim
        self.conversations = Conversations(runtime)
        self._server: asyncio.base_events.Server | None = None
        self._connections: set[asyncio.Task[Any]] = set()
        self.routes: list[tuple[str, re.Pattern[str], Handler, bool]] = []
        self._register_routes()

    # -- lifecycle ----------------------------------------------------------------------------
    async def start(self) -> int:
        self._server = await asyncio.start_server(self._accept, self.host, self.port, limit=MAX_BODY + 65536)
        self.port = self._server.sockets[0].getsockname()[1]
        log.info("api_listening", host=self.host, port=self.port)
        return self.port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for task in list(self._connections):
            task.cancel()
        if self._connections:
            await asyncio.gather(*self._connections, return_exceptions=True)
        # turns still running finish before the runtime stops
        pending = list(self.conversations._inflight.values())
        if pending:
            await asyncio.wait(pending, timeout=30)

    # -- HTTP -------------------------------------------------------------------------------------
    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._serve(reader, writer))
        self._connections.add(task)
        task.add_done_callback(self._connections.discard)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            try:
                request = await asyncio.wait_for(self._read(reader), timeout=15)
            except ApiError as exc:
                await self._send_json(writer, exc.status, {"error": exc.message})
                return
            except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, ValueError):
                return
            request.writer = writer
            request.reader = reader
            status, payload = await self._dispatch(request)
            if payload is not _STREAMED:
                await self._send_json(writer, status, payload)
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as exc:     # the API must never take the runtime down
            log.error("api_error", error=repr(exc))
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _read(self, reader: asyncio.StreamReader) -> Request:
        line = (await reader.readline()).decode("latin-1").strip()
        if not line:
            raise asyncio.IncompleteReadError(b"", None)
        try:
            method, target, _version = line.split(" ", 2)
        except ValueError:
            raise ApiError(400, "malformed request line") from None
        headers: dict[str, str] = {}
        while True:
            raw = (await reader.readline()).decode("latin-1")
            if raw in ("\r\n", "\n", ""):
                break
            name, _, value = raw.partition(":")
            headers[name.strip().lower()] = value.strip()
            if len(headers) > 100:
                raise ApiError(400, "too many headers")
        length = int(headers.get("content-length") or 0)
        if length > MAX_BODY:
            raise ApiError(413, "request body too large")
        body = await reader.readexactly(length) if length else b""
        parts = urlsplit(target)
        query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
        return Request(method.upper(), unquote(parts.path), query, headers, body)

    async def _send_json(self, writer: asyncio.StreamWriter, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode()
        head = (f"HTTP/1.1 {status} {_REASONS.get(status, 'OK')}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n")
        writer.write(head.encode() + body)
        await writer.drain()

    async def _dispatch(self, request: Request) -> tuple[int, Any]:
        if "origin" in request.headers:
            return 403, {"error": "browser requests are not accepted"}
        allowed_methods = []
        for method, pattern, handler, public in self.routes:
            match = pattern.fullmatch(request.path)
            if not match:
                continue
            if method != request.method:
                allowed_methods.append(method)
                continue
            if not public and not self._authorized(request):
                return 401, {"error": "missing or invalid token"}
            if self.runtime.svc is None or not self.runtime.started:
                return 503, {"error": "the runtime is not running"}
            request.groups = match.groups()
            try:
                result = await handler(request)
            except ApiError as exc:
                return exc.status, {"error": exc.message}
            except (KeyError, ValueError, TypeError) as exc:
                return 400, {"error": f"invalid request: {exc}"}
            if isinstance(result, tuple):
                return result
            return 200, result
        if allowed_methods:
            return 405, {"error": f"use {', '.join(allowed_methods)}"}
        return 404, {"error": f"no such endpoint: {request.method} {request.path}"}

    def _authorized(self, request: Request) -> bool:
        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(supplied.strip().encode(), self.token.encode())

    # -- routes -------------------------------------------------------------------------------------
    def route(self, method: str, path: str, handler: Handler, *, public: bool = False) -> None:
        pattern = re.compile(re.sub(r"\{(\w+)\}", r"([^/]+)", path))
        self.routes.append((method, pattern, handler, public))

    def _register_routes(self) -> None:
        r = self.route
        r("GET", "/v1/ping", self.ping, public=True)
        r("GET", "/v1/status", self.status)
        r("GET", "/v1/health", self.health)
        r("GET", "/v1/state", self.state)
        r("GET", "/v1/doctor", self.doctor)
        r("GET", "/v1/tasks", self.list_tasks)
        r("POST", "/v1/tasks", self.create_task)
        r("GET", "/v1/tasks/{id}", self.get_task)
        r("POST", "/v1/tasks/{id}/pause", self.pause_task)
        r("POST", "/v1/tasks/{id}/resume", self.resume_task)
        r("POST", "/v1/tasks/{id}/cancel", self.cancel_task)
        r("GET", "/v1/events", self.events)
        r("GET", "/v1/notifications", self.notifications)
        r("POST", "/v1/notifications/ack", self.ack)
        r("POST", "/v1/notifications/{id}/ack", self.ack_one)
        r("POST", "/v1/conversation", self.converse)
        r("GET", "/v1/conversation/requests/{id}", self.get_request)
        r("POST", "/v1/sessions/attach", self.attach)
        r("POST", "/v1/sessions/detach", self.detach)
        r("GET", "/v1/stream", self.stream)
        r("GET", "/v1/away", self.away)
        r("GET", "/v1/briefing", self.briefing)
        r("POST", "/v1/briefing", self.make_briefing)
        r("GET", "/v1/schedules", self.schedules)
        r("POST", "/v1/schedules", self.create_schedule)
        r("POST", "/v1/schedules/{id}/enable", self.enable_schedule)
        r("POST", "/v1/schedules/{id}/disable", self.disable_schedule)
        r("DELETE", "/v1/schedules/{id}", self.delete_schedule)
        r("GET", "/v1/approvals", self.approvals)
        r("GET", "/v1/grants", self.grants)
        r("GET", "/v1/models", self.models)
        r("POST", "/v1/runtime/stop", self.stop_runtime)
        r("POST", "/v1/sim", self.simulate)

    @property
    def svc(self) -> Any:
        return self.runtime.svc

    # -- runtime ----------------------------------------------------------------------------------
    async def ping(self, req: Request) -> dict[str, Any]:
        return {"ok": True, "service": "jarvis", "version": __version__}

    async def status(self, req: Request) -> dict[str, Any]:
        svc = self.svc
        state = state_snapshot(self.runtime)
        readiness = svc.extra.get("model_readiness")
        return {"runtime": state["runtime"], "health": state["health"], "tasks": state["tasks"],
                "workers": state["workers"], "presence": state["presence"], "mode": state["mode"],
                "model": readiness.summary() if readiness is not None and readiness.can_converse else None,
                "api": {"host": self.host, "port": self.port}, "text": reports.status_report(svc)}

    async def health(self, req: Request) -> dict[str, Any]:
        return await health_report(self.runtime, probe=req.query.get("probe", "1") != "0")

    async def state(self, req: Request) -> dict[str, Any]:
        return state_snapshot(self.runtime)

    async def doctor(self, req: Request) -> dict[str, Any]:
        svc = self.svc
        readiness = svc.extra.get("model_readiness")
        return {"config": svc.config.source, "data_dir": str(svc.config.data_path),
                "database": {"ok": svc.db.healthy(), "schema": svc.db.schema_version()},
                "providers": await svc.router.provider_health(), "models": [m.name for m in svc.router.inventory],
                "tools": len(svc.registry.list()),
                "resources": svc.state.values("resources."),
                "open_tasks": len(svc.tasks.open_tasks()), "pending_approvals": len(svc.approvals.pending()),
                "recovered": [r.summary for r in svc.extra.get("recovery_reports") or []],
                "subsystems": {c.name: {"status": c.status.label, "detail": c.detail}
                               for c in svc.health.components.values()},
                "readiness": None if readiness is None else {
                    "summary": readiness.summary(), "embedding_model": readiness.embedding_model,
                    "issues": readiness.issues, "tips": readiness.tips},
                "health": await health_report(self.runtime)}

    async def stop_runtime(self, req: Request) -> tuple[int, dict[str, Any]]:
        if self.on_stop is None:
            raise ApiError(409, "this runtime is not managed by a daemon")
        asyncio.get_running_loop().call_later(0.1, self.on_stop)
        return 202, {"stopping": True}

    async def simulate(self, req: Request) -> dict[str, Any]:
        if self.sim is None:
            raise ApiError(409, "not running in simulation mode")
        return {"text": self.sim.control(str(req.json().get("command", "")))}

    # -- tasks --------------------------------------------------------------------------------------
    def _task(self, task_id: str) -> Any:
        task = self.svc.tasks.get_task(task_id)
        if task is None:
            raise ApiError(404, f"no task {task_id}")
        return task

    async def list_tasks(self, req: Request) -> dict[str, Any]:
        which = req.query.get("status", "open")
        statuses = None if which == "all" else OPEN if which == "open" else \
            [TaskStatus(s) for s in which.split(",")]
        tasks = self.svc.tasks.list_tasks(statuses, order="recent", limit=req.int("limit", 50, 500))
        return {"tasks": [t.to_api() for t in tasks]}

    async def get_task(self, req: Request) -> dict[str, Any]:
        return {"task": self._task(req.groups[0]).to_api()}

    async def create_task(self, req: Request) -> tuple[int, dict[str, Any]]:
        """A task requested by the user through an interface: the user's (interactive) authority, the normal
        tool pipeline. Steps are validated against the tool registry; without steps JARVIS plans them."""
        from jarvis.planner.planner import Planner
        body = req.json()
        objective = str(body["objective"]).strip()
        if not objective:
            raise ApiError(400, "objective is required")
        steps = None
        if body.get("steps"):
            steps, rejected = Planner(self.svc.registry).validate(list(body["steps"]), 40)
            if rejected:
                raise ApiError(400, "invalid steps: " + "; ".join(rejected[:3]))
        existing = self.svc.tasks.find_by_key(body["idempotency_key"]) if body.get("idempotency_key") else None
        svc = self.svc
        task = svc.tasks.create_task(objective, title=str(body.get("title") or ""), steps=steps,
                                     priority=Priority[str(body.get("priority", "P2"))], cwd=body.get("cwd"),
                                     created_by=f"user:{svc.user}", owner=svc.user,
                                     authority={"interactive": True}, request=objective, origin="api",
                                     idempotency_key=body.get("idempotency_key"))
        return (200 if existing else 201), {"task": task.to_api(), "created": existing is None}

    async def pause_task(self, req: Request) -> dict[str, Any]:
        result = self.svc.tasks.pause_task(self._task(req.groups[0]).id, by=f"user:{self.svc.user}")
        return {"ok": result.ok, "message": result.message}

    async def resume_task(self, req: Request) -> dict[str, Any]:
        result = self.svc.tasks.resume_task(self._task(req.groups[0]).id, by=f"user:{self.svc.user}")
        return {"ok": result.ok, "message": result.message}

    async def cancel_task(self, req: Request) -> dict[str, Any]:
        result = self.svc.tasks.cancel_task(self._task(req.groups[0]).id, by=f"user:{self.svc.user}")
        return {"ok": result.ok, "message": result.message}

    # -- events and notifications ----------------------------------------------------------------------
    async def events(self, req: Request) -> dict[str, Any]:
        types = req.query.get("types")
        severity = req.query.get("min_severity")
        events = self.svc.events.query(
            since=float(req.query["since"]) if "since" in req.query else None,
            types=types.split(",") if types else None, task_id=req.query.get("task_id"),
            min_severity=Severity[severity.upper()] if severity else None, limit=req.int("limit", 50, 1000))
        return {"events": [e.to_dict() for e in events], "text": _events_text(events)}

    async def notifications(self, req: Request) -> dict[str, Any]:
        states = req.query.get("state")
        return {"notifications": self.svc.notifications.list(states=states.split(",") if states else None,
                                                               limit=req.int("limit", 50, 500))}

    async def ack(self, req: Request) -> dict[str, Any]:
        ids = req.json().get("ids")
        if ids:
            count = sum(self.svc.notifications.acknowledge(str(i)) for i in ids)
        else:
            count = self.svc.notifications.acknowledge(None)
        return {"acknowledged": count}

    async def ack_one(self, req: Request) -> dict[str, Any]:
        count = self.svc.notifications.acknowledge(req.groups[0])
        if not count:
            raise ApiError(404, "no such unacknowledged notification")
        return {"acknowledged": count}

    # -- conversation, presence ----------------------------------------------------------------------
    async def converse(self, req: Request) -> Any:
        body = req.json()
        text = str(body["text"])
        session = str(body.get("session") or "default")
        if body.get("client_id") and self.svc.presence is not None:
            self.svc.presence.touch(str(body["client_id"]))
        if not body.get("stream"):
            rid, response, replayed = await self.conversations.handle(
                text, session=session, request_id=body.get("request_id"), cwd=body.get("cwd"))
            return {"request_id": rid, "response": response, "replayed": replayed}
        assert req.writer is not None
        out = Streamer(req.writer)
        await out.start()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        alive = {"ok": True}

        async def pump() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if alive["ok"]:
                    try:
                        await out.send(item)
                    except (ConnectionError, RuntimeError):
                        alive["ok"] = False      # keep draining; the turn itself carries on

        writer_task = asyncio.create_task(pump())
        rid, response, replayed = await self.conversations.handle(
            text, session=session, request_id=body.get("request_id"), cwd=body.get("cwd"),
            on_token=lambda piece: queue.put_nowait({"type": "token", "text": piece}))
        queue.put_nowait({"type": "response", "request_id": rid, "response": response, "replayed": replayed})
        queue.put_nowait(None)
        await writer_task
        if alive["ok"]:
            try:
                await out.end()
            except ConnectionError:
                pass
        return 200, _STREAMED

    async def get_request(self, req: Request) -> dict[str, Any]:
        row = self.svc.db.query_one("SELECT * FROM requests WHERE id=?", (req.groups[0],))
        if row is None:
            raise ApiError(404, "no such request")
        return {"request_id": row["id"], "status": row["status"],
                "response": json.loads(row["response"]) if row["response"] else None}

    async def attach(self, req: Request) -> dict[str, Any]:
        """An interface opened. On the first one after an absence, report what is waiting."""
        svc = self.svc
        body = req.json()
        presence = svc.presence
        ret = presence.attach(str(body.get("kind") or "cli"), str(body.get("session") or "default"),
                              client_id=body.get("client_id"))
        returning = None
        if ret.first and ret.away_since is not None:
            queued = [n for n in svc.notifications.pending()]
            report = away_report(svc, since=ret.away_since, until=svc.clock.now())
            svc.notifications.mark_delivered(queued)       # shown by the interface now
            returning = {"away_since": ret.away_since, "notifications": [n.text() for n in queued],
                         "finished": [{"id": t["id"], "title": t["title"], "status": t["status"]}
                                      for t in report.finished],
                         "waiting": [{"id": t["id"], "title": t["title"], "status": t["status"],
                                      "reason": t["status_reason"]} for t in report.open
                                     if t["status"] in ("waiting", "blocked", "paused")],
                         "unclean_stops": len(report.unclean), "kept_running": report.kept_running}
        readiness = svc.extra.get("model_readiness")
        recovered = svc.extra.get("recovery_reports") or []
        return {"client_id": ret.client_id, "first": ret.first, "returning": returning,
                "runtime": {"pid": state_snapshot(self.runtime)["runtime"]["pid"], "run": self.runtime.run_id,
                            "started_at": self.runtime.started_at, "simulated": self.runtime.simulated,
                            "version": __version__},
                "model": readiness.summary() if readiness is not None and readiness.can_converse else None,
                "model_issues": list(readiness.issues[:2]) if readiness is not None else [],
                "recovered": [r.summary for r in recovered if not r.resumed][:3]}

    async def detach(self, req: Request) -> dict[str, Any]:
        client = str(req.json().get("client_id", ""))
        return {"detached": self.svc.presence.detach(client) if client else False}

    async def stream(self, req: Request) -> Any:
        """Live notifications for an attached interface; the connection itself is its presence."""
        svc = self.svc
        client_id = req.query.get("client_id") or ""
        presence = svc.presence
        if not client_id or client_id not in presence.clients:
            # e.g. the runtime restarted and the interface reconnected with its old id
            client_id = presence.attach(req.query.get("kind", "cli"), req.query.get("session", "default"),
                                        client_id=client_id or None).client_id
        assert req.writer is not None
        out = Streamer(req.writer)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)

        def sink(n: Notification) -> None:
            try:
                queue.put_nowait({"type": "notification", "id": n.id, "text": n.text(),
                                  "priority": n.priority.name.lower(), "task_id": n.task_id})
            except asyncio.QueueFull:
                pass

        svc.notifications.sinks.append(sink)
        # the client closing its end (window closed, process killed) shows up as EOF on the socket
        closed = asyncio.create_task(req.reader.read(1)) if req.reader is not None else None
        try:
            await out.start()
            await out.send({"type": "hello", "client_id": client_id})
            while True:
                getter = asyncio.create_task(queue.get())
                waiting = {getter} | ({closed} if closed is not None else set())
                done, _ = await asyncio.wait(waiting, timeout=10, return_when=asyncio.FIRST_COMPLETED)
                if closed is not None and closed in done:
                    getter.cancel()
                    break
                if getter in done:
                    item = getter.result()
                else:
                    getter.cancel()
                    item = {"type": "ping", "ts": svc.clock.now()}
                await out.send(item)
                presence.touch(client_id)
        except (ConnectionError, asyncio.CancelledError, RuntimeError):
            pass
        finally:
            if closed is not None and not closed.done():
                closed.cancel()
            if sink in svc.notifications.sinks:
                svc.notifications.sinks.remove(sink)
            presence.detach(client_id, reason="stream closed")
        return 200, _STREAMED

    async def away(self, req: Request) -> dict[str, Any]:
        report = away_report(self.svc)
        return {"text": report.text(), "report": report.to_dict()}

    async def briefing(self, req: Request) -> dict[str, Any]:
        latest = latest_briefing(self.svc)
        if latest is None:
            raise ApiError(404, "no briefing has been prepared yet")
        return latest

    async def make_briefing(self, req: Request) -> dict[str, Any]:
        data = prepare_briefing(self.svc, kind="on_demand")
        return latest_briefing(self.svc) or {"data": data}

    # -- schedules ---------------------------------------------------------------------------------------
    async def schedules(self, req: Request) -> dict[str, Any]:
        return {"schedules": [a.to_api() for a in self.svc.automations.list()]}

    async def create_schedule(self, req: Request) -> tuple[int, dict[str, Any]]:
        body = req.json()
        try:
            auto = self.svc.automations.schedule(str(body["name"]), dict(body["schedule"]), dict(body["action"]),
                                                 owner=self.svc.user, catch_up=str(body.get("catch_up", "once")))
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None
        return 201, {"schedule": auto.to_api()}

    async def enable_schedule(self, req: Request) -> dict[str, Any]:
        if not self.svc.automations.set_enabled(req.groups[0], True):
            raise ApiError(404, "no such schedule")
        return {"schedule": self.svc.automations.get(req.groups[0]).to_api()}

    async def disable_schedule(self, req: Request) -> dict[str, Any]:
        if not self.svc.automations.set_enabled(req.groups[0], False):
            raise ApiError(404, "no such schedule")
        return {"schedule": self.svc.automations.get(req.groups[0]).to_api()}

    async def delete_schedule(self, req: Request) -> dict[str, Any]:
        if not self.svc.automations.delete(req.groups[0]):
            raise ApiError(404, "no such schedule")
        return {"deleted": True}

    # -- authority, models -----------------------------------------------------------------------------------
    async def approvals(self, req: Request) -> dict[str, Any]:
        pending = self.svc.approvals.pending()
        return {"approvals": [{"id": a.id, "summary": a.summary, "task_id": a.task_id, "tool": a.tool,
                               "created_at": a.created_at} for a in pending]}

    async def grants(self, req: Request) -> dict[str, Any]:
        return {"grants": [{"id": g.id, "subject": g.subject, "describe": g.describe()}
                           for g in self.svc.permissions.list_grants()]}

    async def models(self, req: Request) -> dict[str, Any]:
        return {"status": self.svc.router.status(), "text": reports.models_report(self.svc)}


_STREAMED = object()


def _events_text(events: list[Any]) -> str:
    from jarvis.clock import format_time
    return "\n".join(f"{format_time(e.ts)}  {e.severity.name:<8} {e.type:<28} {e.source:<16} {str(e.payload)[:90]}"
                     for e in reversed(events)) or "No events."
