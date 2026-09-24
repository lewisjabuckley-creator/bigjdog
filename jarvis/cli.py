"""Command-line interface: one interface into JARVIS, not the architecture itself.

The CLI is a client of the persistent runtime (``jarvis runtime ...``). Closing
it never stops tasks, monitoring, schedules or notifications.

    jarvis                     interactive session (starts the runtime in the background if needed)
    jarvis ask "..."           one request, then exit
    jarvis status              compact system status
    jarvis tasks [--all]       task list;  jarvis task <id> [pause|resume|cancel]
    jarvis away                what happened while you were away
    jarvis notifications       recent notifications (--ack to acknowledge them)
    jarvis briefing [--now]    the latest morning briefing (or prepare one now)
    jarvis schedule ...        list, add, enable, disable or remove schedules
    jarvis doctor [--live]     self-diagnostics
    jarvis models | events | approvals | grants
    jarvis runtime start|stop|restart|status|health|logs|run|install-service
    jarvis --embedded ...      run the runtime inside this process instead (it stops when you exit)
    jarvis --simulate ...      run against simulated hardware, network and model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from jarvis import __version__
from jarvis.clock import format_datetime, format_time
from jarvis.config import ConfigError, JarvisConfig, find_config_file, load_config
from jarvis.core import reports
from jarvis.core.types import NotificationPriority
from jarvis.notifications.manager import Notification
from jarvis.runtime import Runtime
from jarvis.simulation.environment import SimulatedEnvironment
from jarvis.tasks.models import OPEN

_COLORS = {NotificationPriority.CRITICAL: "\033[1;31m", NotificationPriority.URGENT: "\033[33m",
           NotificationPriority.IMPORTANT: "\033[36m"}
_RESET = "\033[0m"


def _color(text: str, prio: NotificationPriority) -> str:
    # classic Windows consoles print ANSI codes literally; Windows Terminal (WT_SESSION) handles them
    if not sys.stdout.isatty() or (sys.platform == "win32" and not os.environ.get("WT_SESSION")):
        return text
    return f"{_COLORS.get(prio, '')}{text}{_RESET}"


def build_runtime(args: argparse.Namespace, *, passive: bool = False) -> tuple[Runtime, SimulatedEnvironment | None]:
    cfg: JarvisConfig = load_config(args.config)
    if args.data_dir:
        cfg.general.data_dir = args.data_dir
    sim = None
    if args.simulate:
        sim = SimulatedEnvironment()
        if not args.data_dir:
            cfg.general.data_dir = os.path.join(tempfile.gettempdir(), "jarvis-sim")
        return Runtime(cfg, providers=[sim.provider], metrics=sim.metrics, network_probe=sim.probe,
                       simulated=True, log_to_stderr=args.verbose, passive=passive), sim
    return Runtime(cfg, log_to_stderr=args.verbose, passive=passive), None


async def _with_runtime(args: argparse.Namespace, fn: Any, *, monitoring: bool = False) -> int:
    """One command against the data directory while no background runtime is running. Only ``ask --embedded``
    runs a full runtime; everything else opens it read-only (no workers, scheduler or recovery)."""
    from jarvis.platforms import LockHeld
    runtime, sim = build_runtime(args, passive=getattr(args, "command", None) != "ask")
    try:
        report = await runtime.start(monitoring=monitoring)
    except LockHeld as exc:
        print(f"{exc}. Use the running runtime (`jarvis runtime status`) or stop it first.", file=sys.stderr)
        return 3
    try:
        return await fn(runtime, report, sim)
    finally:
        await runtime.stop()


# -- interactive session ------------------------------------------------------------------------------

async def interactive_embedded(args: argparse.Namespace) -> int:
    """The runtime inside this process (``--embedded``): everything stops when the session ends."""
    runtime, sim = build_runtime(args)
    report = await runtime.start()
    svc = runtime.svc
    assert svc is not None
    orch = runtime.orchestrator(args.session)
    attached = svc.presence.attach("cli-embedded", args.session) if svc.presence else None
    prompt = "you › "

    def sink(n: Notification) -> None:
        print(f"\r{_color('● ' + n.text(), n.priority)}\n{prompt}", end="", flush=True)

    svc.notifications.sinks.append(sink)
    label = " (simulation)" if sim else ""
    print(f"JARVIS {__version__}{label} (embedded) — {report.greeting()}  Type 'help', or /quit to exit.")

    async def idle_drain() -> None:
        while True:
            await asyncio.sleep(5)
            svc.notifications.drain_if_idle(30)

    drainer = asyncio.create_task(idle_drain())
    try:
        while True:
            svc.notifications.set_activity("idle")
            try:
                line = await asyncio.to_thread(input, prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = _clean_input(line)
            if not line:
                continue
            if line.startswith("/"):
                if line in ("/quit", "/exit", "/q"):
                    break
                print(await slash(line, runtime, sim))
                continue
            svc.notifications.set_activity("typing")
            streaming = {"started": False}

            def on_token(text: str) -> None:
                if not streaming["started"]:
                    streaming["started"] = True
                    print("\r" + " " * 14 + "\rjarvis › ", end="", flush=True)
                print(text, end="", flush=True)

            print("jarvis › …", end="", flush=True)
            try:
                response = await orch.handle(line, on_token=on_token, cwd=os.getcwd())
            except Exception as exc:  # never let the interface die on one bad turn
                print(f"\rjarvis › internal error: {exc}")
                continue
            if response.streamed:
                print()
                extra = response.render(include_text=False)
                if extra:
                    print(extra)
            elif streaming["started"]:        # some text streamed, then e.g. an approval question
                print(f"\njarvis › {response.render()}")
            else:
                print("\r" + " " * 14 + f"\rjarvis › {response.render()}")
    finally:
        drainer.cancel()
        svc.notifications.sinks.remove(sink)
        if attached is not None and svc.presence is not None:
            svc.presence.detach(attached.client_id)
        print("Shutting down — checkpointing work in progress.")
        await runtime.stop()
    return 0


async def slash(line: str, runtime: Runtime, sim: SimulatedEnvironment | None) -> str:
    svc = runtime.svc
    assert svc is not None
    cmd, _, rest = line[1:].partition(" ")
    if cmd == "tasks":
        return format_tasks(svc, all_tasks=rest == "all")
    if cmd == "events":
        return format_events(svc, int(rest or 15))
    if cmd == "approvals":
        pending = svc.approvals.pending()
        return "\n".join(f"{a.id}  {a.summary}  (task {a.task_id})" for a in pending) or "No pending approvals."
    if cmd == "grants":
        grants = svc.permissions.list_grants()
        return "\n".join(f"{g.id}  {g.subject}  {g.describe()}" for g in grants) or "No active grants."
    if cmd == "revoke" and rest:
        return "revoked" if svc.permissions.revoke(rest.strip()) else "no such active grant"
    if cmd == "health":
        return "\n".join(f"{c.name}: {c.status.label} {c.detail}" for c in svc.health.components.values())
    if cmd == "status":
        return reports.status_report(svc)
    if cmd == "debug":
        return debug_dump(svc)
    if cmd == "sim":
        return sim.control(rest) if sim else "not running in simulation mode (start with --simulate)"
    return "commands: /tasks [all], /events [n], /approvals, /grants, /revoke <id>, /health, /status, /debug, " \
           "/sim ..., /quit"


def format_tasks(svc: Any, all_tasks: bool = False) -> str:
    tasks = svc.tasks.list_tasks(None if all_tasks else OPEN, order="recent", limit=30)
    if not tasks:
        return "No tasks." if all_tasks else "No open tasks."
    lines = []
    for t in tasks:
        progress = f" {int(t.compute_progress() * 100)}%" if t.plan else ""
        reason = f" — {t.status_reason}" if t.status_reason else ""
        lines.append(f"{t.id}  [{t.kind.value}/{t.priority.name}] {t.status.value}{progress}  {t.title}{reason}")
    return "\n".join(lines)


def format_events(svc: Any, limit: int = 15) -> str:
    events = svc.events.query(limit=limit)
    return "\n".join(f"{format_time(e.ts)}  {e.severity.name:<8} {e.type:<28} {e.source:<16} "
                     f"{str(e.payload)[:90]}" for e in reversed(events)) or "No events."


def debug_dump(svc: Any) -> str:
    """Operational metadata for developers — never chain-of-thought (spec §92)."""
    lines = [f"events published: {svc.bus.published}, handler errors: {svc.bus.handler_errors}",
             f"model calls: {svc.router.calls}, tokens: {svc.router.tokens}",
             f"providers: {svc.router.provider_status}", f"pins: {svc.router.pins}",
             f"running workers: {list(svc.pool.running)}", f"resource locks: {svc.resources.locks}",
             f"tool stats: {svc.registry.stats}", f"mode: {svc.modes.current.value}, quiet={svc.modes.quiet}, "
             f"private={svc.modes.private}"]
    return "\n".join(lines)


# -- one-shot commands --------------------------------------------------------------------------------

async def cmd_ask(args: argparse.Namespace) -> int:
    async def run(runtime: Runtime, report: Any, sim: Any) -> int:
        response = await runtime.orchestrator(args.session).handle(" ".join(args.text), cwd=os.getcwd())
        print(response.render())
        return 0 if response.kind != "error" else 1
    return await _with_runtime(args, run)


async def cmd_status(args: argparse.Namespace) -> int:
    async def run(runtime: Runtime, report: Any, sim: Any) -> int:
        print(reports.status_report(runtime.svc))
        return 0
    return await _with_runtime(args, run)


async def cmd_tasks(args: argparse.Namespace) -> int:
    async def run(runtime: Runtime, report: Any, sim: Any) -> int:
        print(format_tasks(runtime.svc, all_tasks=args.all))
        return 0
    return await _with_runtime(args, run)


async def cmd_models(args: argparse.Namespace) -> int:
    async def run(runtime: Runtime, report: Any, sim: Any) -> int:
        print(reports.models_report(runtime.svc))
        return 0
    return await _with_runtime(args, run)


async def cmd_events(args: argparse.Namespace) -> int:
    async def run(runtime: Runtime, report: Any, sim: Any) -> int:
        print(format_events(runtime.svc, args.limit))
        return 0
    return await _with_runtime(args, run)


async def cmd_approvals(args: argparse.Namespace) -> int:
    async def run(runtime: Runtime, report: Any, sim: Any) -> int:
        print(await slash("/approvals", runtime, sim))
        return 0
    return await _with_runtime(args, run)


async def cmd_grants(args: argparse.Namespace) -> int:
    async def run(runtime: Runtime, report: Any, sim: Any) -> int:
        print(await slash("/grants", runtime, sim))
        return 0
    return await _with_runtime(args, run)


async def cmd_doctor(args: argparse.Namespace) -> int:
    async def run(runtime: Runtime, report: Any, sim: Any) -> int:
        svc = runtime.svc
        assert svc is not None
        cfg = svc.config
        print(f"JARVIS {__version__}{' (simulation)' if sim else ''}")
        print(f"config: {cfg.source}")
        print(f"data dir: {cfg.data_path}")
        print(f"database: {'ok' if svc.db.healthy() else 'FAILED'} (schema v{svc.db.schema_version()})")
        health = await svc.router.provider_health()
        for name, h in health.items():
            status = "ok" if h["available"] else "unreachable"
            print(f"model provider {name}: {status} {h.get('version') or ''} {'' if h['available'] else h['detail']}")
        print(f"models: {', '.join(report.models) or 'none'}")
        print(f"tools: {len(svc.registry.list())} registered")
        sample = svc.state.values("resources.")
        print("resources: " + ", ".join(f"{k.split('.', 1)[1]}={v}" for k, v in list(sample.items())[:6]))
        print(f"open tasks: {len(svc.tasks.open_tasks())}, pending approvals: {len(svc.approvals.pending())}")
        for r in report.recovered:
            print(f"recovered: {r.summary}")
        for issue in report.issues:
            print(f"issue: {issue}")
        for comp in svc.health.components.values():
            print(f"subsystem {comp.name}: {comp.status.label} {comp.detail}")
        readiness = report.readiness
        if readiness is not None:
            print(f"conversation model: {readiness.summary()}")
            print(f"embedding model: {readiness.embedding_model or 'none'}")
            for issue in readiness.issues:
                print(f"model issue: {issue}")
            for tip in readiness.tips:
                print(f"tip: {tip}")
        return 0
    code = await _with_runtime(args, run)
    if getattr(args, "live", False):
        code = max(code, await live_doctor(args))
    return code


async def live_doctor(args: argparse.Namespace) -> int:
    """End-to-end checks against the real model runtime, in a throwaway data directory."""
    from jarvis.diagnostics import live_checks
    if args.simulate:
        print("\nlive checks need a real model runtime; run without --simulate")
        return 1
    cfg = load_config(args.config)
    print("\nLive checks (temporary data directory; your real memory and tasks are not touched).")
    print("On a computer without a graphics card each step can take a minute.\n")
    results = await live_checks(cfg, model=args.model, report=lambda r: print(r.line(), flush=True))
    failed = [r for r in results if r.status == "FAIL"]
    warned = [r for r in results if r.status == "WARN"]
    if failed:
        print(f"\n{len(failed)} check(s) failed — see above.")
        return 1
    if warned:
        print(f"\nWorking, but the model didn't follow {len(warned)} instruction(s). Larger or tool-tuned models "
              "(e.g. qwen2.5:7b, llama3.1:8b) do better.")
    else:
        print("\nAll live checks passed: you can have a real conversation with JARVIS.")
    return 0




# == client mode: the CLI talks to the persistent runtime ==============================================

def resolve_config(args: argparse.Namespace) -> tuple[JarvisConfig, str | None]:
    """The configuration and data directory the runtime for these arguments uses (same rules as embedded)."""
    cfg: JarvisConfig = load_config(args.config)
    if args.data_dir:
        cfg.general.data_dir = args.data_dir
    elif args.simulate:
        cfg.general.data_dir = os.path.join(tempfile.gettempdir(), "jarvis-sim")
    config_file = args.config or (str(find_config_file()) if find_config_file() else None)
    return cfg, config_file


def _client(args: argparse.Namespace, *, auto_start: bool | None = None) -> Any:
    from jarvis.service.client import connect
    cfg, config_file = resolve_config(args)
    start = cfg.runtime.auto_start if auto_start is None else auto_start
    return connect(data_dir=cfg.data_path, config=config_file, simulate=args.simulate, auto_start=start,
                   announce=lambda text: print(text, file=sys.stderr, flush=True))


def _running_client(args: argparse.Namespace) -> Any | None:
    """A client if the runtime is already running, else None (never starts it)."""
    from jarvis.service.client import Client, read_info
    cfg, _ = resolve_config(args)
    info = read_info(cfg.data_path)
    if info is None:
        return None
    client = Client(info)
    if client.ping():
        return client
    client.close()
    return None


def format_task_rows(tasks: list[dict[str, Any]], *, empty: str = "No open tasks.") -> str:
    if not tasks:
        return empty
    lines = []
    for t in tasks:
        progress = f" {int(t['progress'] * 100)}%" if t.get("progress") else ""
        reason = f" — {t['status_reason']}" if t.get("status_reason") else ""
        lines.append(f"{t['id']}  [{t['kind']}/{t['priority']}] {t['status']}{progress}  {t['title']}{reason}")
    return "\n".join(lines)


def format_task_detail(t: dict[str, Any]) -> str:
    lines = [f"{t['title']}  ({t['id']})", f"status: {t['status']}" + (f" — {t['status_reason']}"
                                                                        if t.get("status_reason") else ""),
             f"goal: {t['goal']}"]
    if t.get("request") and t["request"] != t["goal"]:
        lines.append(f"request: {t['request']}")
    if t.get("outcome"):
        lines.append(f"outcome: {t['outcome']}")
    lines.append(f"origin: {t.get('origin') or t.get('created_by')}  priority: {t['priority']}  "
                 f"retries: {t.get('retry_count', 0)}")
    for label, key in (("done", "completed_steps"), ("failed", "failed_steps"), ("pending", "pending_steps")):
        steps = t.get(key) or []
        if steps:
            lines.append(f"{label}: " + "; ".join(s["description"] + (" (outcome unknown)" if s.get("outcome_unknown")
                                                                     else "") for s in steps))
    if t.get("recovery"):
        lines.append(f"recovery: {t['recovery']}")
    if t.get("artifacts"):
        lines.append("artifacts: " + ", ".join(a.get("path") or a.get("type", "?") for a in t["artifacts"]))
    if t.get("error"):
        lines.append(f"last error: {t['error']}")
    for key, label in (("created_at", "created"), ("started_at", "started"), ("finished_at", "finished")):
        if t.get(key):
            lines.append(f"{label}: {format_datetime(t[key])}")
    if t.get("result"):
        lines += ["result:", t["result"]]
    return "\n".join(lines)


def format_health(health: dict[str, Any]) -> str:
    marks = {"healthy": "OK  ", "disabled": "--  ", "unknown": "??  ", "degraded": "WARN", "warning": "WARN",
             "critical": "FAIL", "offline": "FAIL"}
    lines = [f"overall: {health['overall']}"]
    for name, comp in health["components"].items():
        lines.append(f"[{marks.get(comp['status'], comp['status'])}] {name:<12} {comp['status']:<9} {comp.get('detail', '')}")
    return "\n".join(lines)


def _clean_input(line: str) -> str:
    """Drop control characters (Ctrl+A arrives as \\x01 and shows as ^A) and surrounding space."""
    return "".join(ch for ch in line if ch.isprintable() or ch == "\t").strip()


def _print_response(response: dict[str, Any], streamed_any: bool) -> None:
    if streamed_any and response.get("streamed"):
        print()
        if response.get("extra"):
            print(response["extra"])
    elif streamed_any:          # some text streamed, then e.g. an approval question
        print(f"\njarvis › {response['rendered']}")
    else:                       # nothing streamed here (including a replayed answer after a reconnect)
        print("\r" + " " * 14 + f"\rjarvis › {response['rendered']}")


class _NotificationStream(threading.Thread):
    """Prints notifications pushed by the runtime while the prompt waits for input."""

    def __init__(self, client: Any, client_id: str, session: str, prompt: str) -> None:
        super().__init__(daemon=True, name="jarvis-notifications")
        self.client = client
        self.client_id = client_id
        self.session = session
        self.prompt = prompt
        self.stopped = threading.Event()
        self.lost = False

    def run(self) -> None:
        failures = 0
        while not self.stopped.is_set():
            try:
                for item in self.client.stream("GET", "/v1/stream", params={"client_id": self.client_id,
                                                                            "session": self.session}):
                    failures = 0
                    if self.lost:
                        self.lost = False
                    if item.get("type") == "notification" and not self.stopped.is_set():
                        prio = NotificationPriority[item["priority"].upper()]
                        print(f"\r{_color('● ' + item['text'], prio)}\n{self.prompt}", end="", flush=True)
                    if self.stopped.is_set():
                        return
            except Exception:
                failures += 1
                if failures == 3 and not self.stopped.is_set():
                    self.lost = True
                    print(f"\r● Lost contact with the JARVIS runtime; I'll reconnect when you send a message.\n"
                          f"{self.prompt}", end="", flush=True)
            if self.stopped.wait(min(2.0 * failures, 10.0) if failures else 0.5):
                return

    def stop(self) -> None:
        self.stopped.set()


def _attach(client: Any, session: str, client_id: str | None = None) -> dict[str, Any]:
    return client.post("/v1/sessions/attach", {"kind": "cli", "session": session, "client_id": client_id})


def _returning_lines(att: dict[str, Any], skip: set[str] | None = None) -> list[str]:
    ret = att.get("returning")
    if not ret:
        return []
    skip = skip or set()
    lines = []
    if ret.get("unclean_stops"):
        lines.append("While you were away JARVIS stopped unexpectedly and recovered.")
    finished = ret.get("finished") or []
    if finished:
        lines.append("While you were away: " + "; ".join(f"{t['title']} {t['status']}" for t in finished[:5]) + ".")
    for text in ret.get("notifications") or []:
        if not any(t["title"] in text for t in finished):
            lines.append(f"● {text}")
    waiting = [t for t in ret.get("waiting") or [] if t["id"] not in skip]     # recovery notes are shown above
    if waiting:
        lines.append("Waiting on you: " + "; ".join(f"{t['title']} ({t['reason'] or t['status']})"
                                                   for t in waiting[:3]) + ".")
    if lines:
        lines.append("Ask \"what happened while I was away?\" for the details.")
    return lines


def interactive_client(args: argparse.Namespace) -> int:
    import httpx

    from jarvis.service.client import ApiFailure, RuntimeNotRunning
    prompt = "you › "
    try:
        client = _client(args, auto_start=True)
    except RuntimeNotRunning as exc:
        print(f"Couldn't start the JARVIS runtime: {exc}\nTry `jarvis --embedded` to run it inside this window.")
        return 1
    att = _attach(client, args.session)
    client_id = att["client_id"]
    label = " (simulation)" if att["runtime"].get("simulated") else ""
    model = f"Talking through {att['model']}." if att.get("model") else \
        "No language model is available, so I'm running on deterministic capabilities only."
    print(f"JARVIS {__version__}{label} — connected to the runtime (pid {att['runtime']['pid']}). {model} "
          + " ".join(att.get("model_issues") or []))
    recovered = att.get("recovered") or []
    for item in recovered:
        summary = item["summary"] if isinstance(item, dict) else str(item)
        print(f"● {summary}" + ("" if "'continue'" in summary else " Say 'continue' to resume."))
    for line in _returning_lines(att, skip={item["id"] for item in recovered if isinstance(item, dict)}):
        print(line)
    print("Type 'help', or /quit to leave (JARVIS keeps running in the background).")
    stream = _NotificationStream(client, client_id, args.session, prompt)
    stream.start()
    try:
        while True:
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = _clean_input(line)
            if not line:
                continue
            if line.startswith("/"):
                if line in ("/quit", "/exit", "/q"):
                    break
                try:
                    print(client_slash(line, client))
                except Exception as exc:
                    print(f"error: {exc}")
                continue
            request_id = uuid.uuid4().hex
            body = {"text": line, "session": args.session, "request_id": request_id, "cwd": os.getcwd(),
                    "client_id": client_id, "stream": True}
            for attempt in (1, 2):
                streaming = {"started": False}
                response = None
                print("jarvis › …", end="", flush=True)
                try:
                    for item in client.stream("POST", "/v1/conversation", body):
                        if item["type"] == "token":
                            if not streaming["started"]:
                                streaming["started"] = True
                                print("\r" + " " * 14 + "\rjarvis › ", end="", flush=True)
                            print(item["text"], end="", flush=True)
                        elif item["type"] == "response":
                            response = item["response"]
                except ApiFailure as exc:              # the runtime answered with an error
                    print(f"\rjarvis › {exc.message}")
                    break
                except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
                    if attempt == 2:
                        print(f"\rjarvis › I couldn't reach the runtime ({exc}).")
                        break
                    # the runtime may have restarted: reconnect and resend the same request id, which returns
                    # the original answer if it was already handled instead of doing the work twice
                    try:
                        client = _client(args, auto_start=True)
                        att = _attach(client, args.session, client_id)
                        stream.client = client
                    except Exception as reconnect:
                        print(f"\rjarvis › I couldn't reach the runtime ({reconnect}).")
                        break
                    continue
                if response is not None:
                    _print_response(response, streaming["started"])
                break
    finally:
        stream.stop()
        try:
            client.post("/v1/sessions/detach", {"client_id": client_id}, timeout=3)
        except Exception:
            pass
        client.close()
        print("JARVIS keeps running in the background (`jarvis runtime stop` stops it).")
    return 0


def client_slash(line: str, client: Any) -> str:
    cmd, _, rest = line[1:].partition(" ")
    if cmd == "tasks":
        data = client.get("/v1/tasks", status="all" if rest == "all" else "open", limit=30)
        return format_task_rows(data["tasks"], empty="No tasks." if rest == "all" else "No open tasks.")
    if cmd == "task" and rest:
        return format_task_detail(client.get(f"/v1/tasks/{rest.strip()}")["task"])
    if cmd == "events":
        return client.get("/v1/events", limit=int(rest or 15))["text"]
    if cmd == "approvals":
        pending = client.get("/v1/approvals")["approvals"]
        return "\n".join(f"{a['id']}  {a['summary']}  (task {a['task_id']})" for a in pending) or "No pending approvals."
    if cmd == "grants":
        grants = client.get("/v1/grants")["grants"]
        return "\n".join(f"{g['id']}  {g['subject']}  {g['describe']}" for g in grants) or "No active grants."
    if cmd == "health":
        return format_health(client.get("/v1/health"))
    if cmd == "status":
        return client.get("/v1/status")["text"]
    if cmd == "away":
        return client.get("/v1/away")["text"]
    if cmd in ("state", "debug"):
        return json.dumps(client.get("/v1/state"), indent=1, default=str)
    if cmd == "sim":
        return client.post("/v1/sim", {"command": rest})["text"]
    return "commands: /tasks [all], /task <id>, /events [n], /approvals, /grants, /health, /status, /away, /state, " \
           "/sim ..., /quit"


# -- one-shot commands in client mode ------------------------------------------------------------------

def _via_runtime_or_embedded(args: argparse.Namespace, api: Any, embedded: Any) -> int:
    """Use the running runtime if there is one; otherwise run the command in-process (as before)."""
    client = None if args.embedded else _running_client(args)
    if client is None:
        return asyncio.run(embedded(args))
    with client:
        return api(client)


def client_ask(args: argparse.Namespace) -> int:
    from jarvis.service.client import RuntimeNotRunning
    try:
        client = _client(args)
    except RuntimeNotRunning as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with client:
        data = client.converse(" ".join(args.text), request_id=uuid.uuid4().hex, session=args.session,
                               cwd=os.getcwd())
    response = data["response"]
    print(response["rendered"])
    return 0 if response.get("kind") != "error" else 1


def cmd_task(args: argparse.Namespace) -> int:
    def api(client: Any) -> int:
        if args.action:
            result = client.post(f"/v1/tasks/{args.id}/{args.action}")
            print(result["message"])
            return 0 if result["ok"] else 1
        print(format_task_detail(client.get(f"/v1/tasks/{args.id}")["task"]))
        return 0

    async def embedded(a: argparse.Namespace) -> int:
        async def run(runtime: Runtime, report: Any, sim: Any) -> int:
            svc = runtime.svc
            task = svc.tasks.get_task(a.id)
            if task is None:
                print(f"no task {a.id}")
                return 1
            if a.action:
                result = getattr(svc.tasks, f"{a.action}_task")(task.id, by=f"user:{svc.user}")
                print(result.message)
                return 0 if result.ok else 1
            print(format_task_detail(task.to_api()))
            return 0
        return await _with_runtime(a, run)
    return _via_runtime_or_embedded(args, api, embedded)


def cmd_away(args: argparse.Namespace) -> int:
    def api(client: Any) -> int:
        print(client.get("/v1/away")["text"])
        return 0

    async def embedded(a: argparse.Namespace) -> int:
        from jarvis.core.awareness import away_report

        async def run(runtime: Runtime, report: Any, sim: Any) -> int:
            print(away_report(runtime.svc).text())
            return 0
        return await _with_runtime(a, run)
    return _via_runtime_or_embedded(args, api, embedded)


def cmd_notifications(args: argparse.Namespace) -> int:
    def api(client: Any) -> int:
        if args.ack:
            print(f"acknowledged {client.post('/v1/notifications/ack')['acknowledged']}")
            return 0
        items = client.get("/v1/notifications", limit=args.limit,
                           state=None if args.all else "queued,delivered")["notifications"]
        print("\n".join(f"{format_datetime(n['ts'])}  {n['priority']:<13} {n['state']:<12} {n['text']}"
                        for n in reversed(items)) or "No notifications.")
        return 0

    async def embedded(a: argparse.Namespace) -> int:
        async def run(runtime: Runtime, report: Any, sim: Any) -> int:
            svc = runtime.svc
            if a.ack:
                print(f"acknowledged {svc.notifications.acknowledge(None)}")
                return 0
            items = svc.notifications.list(states=None if a.all else ["queued", "delivered"], limit=a.limit)
            print("\n".join(f"{format_datetime(n['ts'])}  {n['priority']:<13} {n['state']:<12} {n['text']}"
                            for n in reversed(items)) or "No notifications.")
            return 0
        return await _with_runtime(a, run)
    return _via_runtime_or_embedded(args, api, embedded)


def cmd_briefing(args: argparse.Namespace) -> int:
    from jarvis.service.client import ApiFailure, RuntimeNotRunning
    try:
        client = _client(args)
    except RuntimeNotRunning as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with client:
        try:
            data = client.post("/v1/briefing") if args.now else client.get("/v1/briefing")
        except ApiFailure as exc:
            print(exc.message)
            return 1
    print(f"{format_datetime(data['ts'])} ({data['kind']}): {data['text']}")
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    from jarvis.service.client import ApiFailure, RuntimeNotRunning
    try:
        client = _client(args)
    except RuntimeNotRunning as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with client:
        try:
            if args.op in (None, "list"):
                items = client.get("/v1/schedules")["schedules"]
                if not items:
                    print("No schedules or automations.")
                for a in items:
                    when = a.get("describe") or f"when {a.get('when')}"
                    nxt = f", next {format_datetime(a['next_run'])}" if a.get("next_run") and a["enabled"] else ""
                    last = f", last {a['last_status']}" if a.get("last_status") else ""
                    print(f"{a['id']}  {'on ' if a['enabled'] else 'off'}  {a['name']} — {when}{nxt}{last}")
                return 0
            if args.op == "add":
                when = _schedule_from_args(args)
                action = {"type": "briefing"} if args.briefing else \
                    {"type": "notify", "title": args.notify} if args.notify else \
                    {"type": "task", "objective": args.task,
                     "steps": [{"tool": "shell_execute", "args": {"command": args.shell_command},
                                "description": args.shell_command}] if args.shell_command else [],
                     "priority": "P3"}
                if action["type"] == "task" and not args.task:
                    print("say what to do: --task \"objective\" [--command \"...\"], --notify \"text\" or --briefing")
                    return 2
                created = client.post("/v1/schedules", {"name": args.name, "schedule": when, "action": action})
                s = created["schedule"]
                print(f"scheduled {s['name']} ({s['describe']}); next run {format_datetime(s['next_run'])}. "
                      "It runs with automation authority: anything consequential needs a grant.")
                return 0
            if args.op in ("enable", "disable"):
                client.post(f"/v1/schedules/{args.name}/{args.op}")
                print(f"{args.op}d")
                return 0
            if args.op == "remove":
                client.delete(f"/v1/schedules/{args.name}")
                print("removed")
                return 0
        except ApiFailure as exc:
            print(f"error: {exc.message}")
            return 1
    return 2


def _schedule_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.at_time:
        from datetime import datetime
        try:
            ts = datetime.fromisoformat(args.at_time).timestamp()
        except ValueError:
            raise SystemExit("--at takes an ISO date and time, e.g. 2026-10-01T09:00")
        return {"type": "once", "at": ts}
    if args.every:
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        raw = args.every.strip().lower()
        try:
            seconds = float(raw[:-1]) * units[raw[-1]] if raw[-1] in units else float(raw)
        except (ValueError, IndexError):
            raise SystemExit("--every takes a duration such as 30m, 2h or 1d")
        return {"type": "interval", "every_s": seconds}
    if args.weekly:
        return {"type": "weekly", "days": [d.strip() for d in args.weekly.split(",")], "at": args.daily or "09:00"}
    if args.daily:
        return {"type": "daily", "at": args.daily}
    raise SystemExit("give a time: --daily HH:MM, --weekly mon,thu [--daily HH:MM], --every 2h or --at ISO-TIME")


def cmd_status_any(args: argparse.Namespace) -> int:
    def api(client: Any) -> int:
        print(client.get("/v1/status")["text"])
        return 0
    return _via_runtime_or_embedded(args, api, cmd_status)


def cmd_tasks_any(args: argparse.Namespace) -> int:
    def api(client: Any) -> int:
        data = client.get("/v1/tasks", status="all" if args.all else "open", limit=30)
        print(format_task_rows(data["tasks"], empty="No tasks." if args.all else "No open tasks."))
        return 0
    return _via_runtime_or_embedded(args, api, cmd_tasks)


def cmd_models_any(args: argparse.Namespace) -> int:
    return _via_runtime_or_embedded(args, lambda c: print(c.get("/v1/models")["text"]) or 0, cmd_models)


def cmd_events_any(args: argparse.Namespace) -> int:
    return _via_runtime_or_embedded(args, lambda c: print(c.get("/v1/events", limit=args.limit)["text"]) or 0,
                                    cmd_events)


def cmd_approvals_any(args: argparse.Namespace) -> int:
    return _via_runtime_or_embedded(args, lambda c: print(client_slash("/approvals", c)) or 0, cmd_approvals)


def cmd_grants_any(args: argparse.Namespace) -> int:
    return _via_runtime_or_embedded(args, lambda c: print(client_slash("/grants", c)) or 0, cmd_grants)


def cmd_doctor_any(args: argparse.Namespace) -> int:
    def api(client: Any) -> int:
        d = client.get("/v1/doctor")
        print(f"JARVIS {__version__} (running runtime, pid {client.info.pid})")
        print(f"config: {d['config']}\ndata dir: {d['data_dir']}")
        print(f"database: {'ok' if d['database']['ok'] else 'FAILED'} (schema v{d['database']['schema']})")
        for name, h in d["providers"].items():
            print(f"model provider {name}: {'ok' if h['available'] else 'unreachable'} {h.get('version') or ''} "
                  f"{'' if h['available'] else h['detail']}")
        print(f"models: {', '.join(d['models']) or 'none'}\ntools: {d['tools']} registered")
        print(f"open tasks: {d['open_tasks']}, pending approvals: {d['pending_approvals']}")
        for line in d["recovered"]:
            print(f"recovered: {line}")
        r = d.get("readiness")
        if r:
            print(f"conversation model: {r['summary']}\nembedding model: {r['embedding_model'] or 'none'}")
            for issue in r["issues"]:
                print(f"model issue: {issue}")
        print(format_health(d["health"]))
        return 0
    code = _via_runtime_or_embedded(args, api, _doctor_embedded)
    if getattr(args, "live", False):
        code = max(code, asyncio.run(live_doctor(args)))
    return code


async def _doctor_embedded(args: argparse.Namespace) -> int:
    live = args.live
    args.live = False      # the live checks run once, after either path
    try:
        return await cmd_doctor(args)
    finally:
        args.live = live


# -- runtime management -----------------------------------------------------------------------------

def _answering_info(data_dir: Path) -> Any:
    """The runtime record, only if that runtime actually answers (a stale record after a crash, or a reused
    pid, does not)."""
    from jarvis.service.client import Client, read_info
    info = read_info(data_dir)
    if info is None:
        return None
    with Client(info, timeout=5) as client:
        return info if client.ping() else None


def _is_jarvis_runtime(pid: int) -> bool:
    import psutil
    try:
        cmdline = " ".join(psutil.Process(pid).cmdline())
    except (psutil.Error, OSError):
        return False
    return "jarvis" in cmdline and "runtime" in cmdline


def cmd_runtime(args: argparse.Namespace) -> int:
    from jarvis.platforms import current as current_platform
    from jarvis.service.client import Client, RuntimeNotRunning, launch, read_info, runtime_log
    cfg, config_file = resolve_config(args)
    data_dir = cfg.data_path
    platform = current_platform()
    op = args.op

    if op == "run" or (op == "start" and args.foreground):
        from jarvis.service.daemon import run_daemon
        return run_daemon(cfg, simulate=args.simulate, log_to_stderr=args.verbose)

    def stop() -> bool:
        info = read_info(data_dir)
        if info is None:
            print("The JARVIS runtime is not running.")
            return True
        try:
            with Client(info, timeout=10) as client:
                client.post("/v1/runtime/stop")
        except Exception as exc:
            if not _is_jarvis_runtime(info.pid):      # a stale record whose pid now belongs to something else
                print("The JARVIS runtime is not running (its last record is stale).")
                return True
            print(f"The runtime didn't answer ({exc}); stopping process {info.pid}.")
            platform.terminate(info.pid)
        if platform.wait_gone(info.pid, 45):
            print(f"Stopped the JARVIS runtime (pid {info.pid}); running tasks were checkpointed.")
            return True
        print(f"The runtime (pid {info.pid}) did not stop in time; forcing it. Interrupted steps will be "
              "reviewed when it next starts.")
        return platform.terminate(info.pid, timeout=5)

    def start() -> int:
        info = _answering_info(data_dir)
        if info is not None:
            print(f"The JARVIS runtime is already running (pid {info.pid}, 127.0.0.1:{info.port}).")
            return 0
        try:
            info = launch(config=config_file, data_dir=data_dir, simulate=args.simulate)
        except RuntimeNotRunning as exc:
            print(f"error: {exc}")
            return 1
        print(f"Started the JARVIS runtime (pid {info.pid}, 127.0.0.1:{info.port}). Logs: {runtime_log(data_dir)}")
        return 0

    if op == "start":
        return start()
    if op == "stop":
        return 0 if stop() else 1
    if op == "restart":
        if not stop():
            return 1
        return start()
    if op == "status":
        info = _answering_info(data_dir)
        if info is None:
            print("The JARVIS runtime is not running." + _last_run_note(data_dir))
            return 3
        with Client(info, timeout=10) as client:
            st = client.get("/v1/status")
        rt = st["runtime"]
        presence = st.get("presence") or {}
        attached = len(presence.get("clients") or [])
        print(f"running — pid {rt['pid']}, up {_dur(rt['uptime_s'])}, api 127.0.0.1:{info.port}, "
              f"{'simulation, ' if rt.get('simulated') else ''}run {rt['run']}")
        print(f"health: {st['health']['overall']}" + (f" ({', '.join(st['health']['unhealthy'])})"
                                                      if st['health']['unhealthy'] else ""))
        open_counts = st["tasks"]["open"]
        print("tasks: " + (", ".join(f"{n} {s}" for s, n in open_counts.items()) or "none open") +
              f"; workers {st['workers']['running']}/{st['workers']['max_concurrent']}")
        print(f"model: {st.get('model') or 'none available'}")
        print(f"interfaces attached: {attached}" + ("" if attached else
                                                   f" (away since {format_datetime(presence['away_since'])})"
                                                   if presence.get("away_since") else ""))
        me = rt.get("self") or {}
        if me:
            print(f"JARVIS itself: {me.get('memory_mb', '?')} MB, CPU {me.get('cpu_percent', '?')}%")
        return 0
    if op == "health":
        info = _answering_info(data_dir)
        if info is None:
            print("The JARVIS runtime is not running.")
            return 3
        with Client(info, timeout=30) as client:
            health = client.get("/v1/health")
        print(format_health(health))
        return 0 if health["overall"] in ("healthy", "degraded") else 1
    if op == "logs":
        return _show_logs(data_dir, args.lines, args.follow, args.structured)
    if op == "install-service":
        from jarvis.service.client import daemon_argv, source_root
        definition = platform.service_definition(daemon_argv(config=config_file, data_dir=str(data_dir),
                                                             simulate=False), data_dir,
                                                 env={"PYTHONPATH": source_root(), "PYTHONUNBUFFERED": "1"})
        if not args.write:
            print(f"# {definition.path}\n{definition.content}")
            print(f"# write it with: jarvis runtime install-service --write\n# then: {definition.enable_hint}")
        else:
            definition.path.parent.mkdir(parents=True, exist_ok=True)
            definition.path.write_text(definition.content)
            print(f"Wrote {definition.path}. To activate: {definition.enable_hint}")
        if not definition.tested:
            print(f"Note: the {platform.name} service definition has not been tested on {platform.name} yet.")
        return 0
    print("usage: jarvis runtime start|stop|restart|status|health|logs|run|install-service")
    return 2


def _dur(seconds: float) -> str:
    from jarvis.clock import format_duration
    return format_duration(seconds)


def _last_run_note(data_dir: Path) -> str:
    """Read the last run record without starting anything (read-only)."""
    import sqlite3
    db = data_dir / "jarvis.db"
    if not db.exists():
        return ""
    try:
        conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=2)
        row = conn.execute("SELECT started_at, heartbeat_at, stopped_at, clean FROM runtime_runs "
                           "ORDER BY started_at DESC LIMIT 1").fetchone()
        conn.close()
    except sqlite3.Error:
        return ""
    if row is None:
        return ""
    started, heartbeat, stopped, clean = row
    if clean:
        return f" It last stopped cleanly at {format_datetime(stopped)}."
    return f" Its last run stopped unexpectedly (last seen {format_datetime(heartbeat or started)})."


def _show_logs(data_dir: Path, lines: int, follow: bool, structured: bool) -> int:
    from jarvis.service.client import runtime_log
    log_dir = data_dir / "logs"
    if structured:
        files = sorted(log_dir.glob("jarvis-*.jsonl"))
        path = files[-1] if files else None
    else:
        path = runtime_log(data_dir)
    if path is None or not path.exists():
        print(f"No logs yet in {log_dir}.")
        return 1
    with open(path, "rb") as fh:
        tail = fh.read().decode(errors="replace").splitlines()[-lines:]
        print("\n".join(tail))
        if not follow:
            return 0
        try:
            while True:
                chunk = fh.read()
                if chunk:
                    print(chunk.decode(errors="replace"), end="", flush=True)
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="JARVIS — local-first AI operating environment")
    parser.add_argument("--config", help="path to jarvis.toml")
    parser.add_argument("--data-dir", help="override the data directory")
    parser.add_argument("--simulate", action="store_true", help="run against simulated hardware, network and model")
    parser.add_argument("--embedded", action="store_true",
                        help="run the runtime inside this process instead of the background runtime")
    parser.add_argument("--session", default="default", help="conversation session name")
    parser.add_argument("--verbose", action="store_true", help="structured logs to stderr")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    sub = parser.add_subparsers(dest="command")
    ask = sub.add_parser("ask", help="handle one request and exit")
    ask.add_argument("text", nargs="+")
    sub.add_parser("status", help="compact system status")
    tasks = sub.add_parser("tasks", help="list tasks")
    tasks.add_argument("--all", action="store_true")
    task = sub.add_parser("task", help="show one task (with its result), or pause/resume/cancel it")
    task.add_argument("id")
    task.add_argument("action", nargs="?", choices=["pause", "resume", "cancel"])
    sub.add_parser("away", help="what happened while you were away")
    notes = sub.add_parser("notifications", help="recent notifications")
    notes.add_argument("--ack", action="store_true", help="acknowledge all")
    notes.add_argument("--all", action="store_true", help="include acknowledged and logged ones")
    notes.add_argument("--limit", type=int, default=20)
    brief = sub.add_parser("briefing", help="the latest morning briefing")
    brief.add_argument("--now", action="store_true", help="prepare one now")
    sched = sub.add_parser("schedule", help="scheduled work: list | add | enable | disable | remove")
    sched.add_argument("op", nargs="?", choices=["list", "add", "enable", "disable", "remove"])
    sched.add_argument("name", nargs="?", help="name (add) or id (enable/disable/remove)")
    sched.add_argument("--daily", metavar="HH:MM")
    sched.add_argument("--weekly", metavar="DAYS", help="e.g. mon,thu (time from --daily, default 09:00)")
    sched.add_argument("--every", metavar="DURATION", help="e.g. 30m, 2h, 1d")
    sched.add_argument("--at", dest="at_time", metavar="ISO-TIME", help="once, e.g. 2026-10-01T09:00")
    sched.add_argument("--task", metavar="OBJECTIVE")
    # dest must not be "command": that is the subcommand's own dest, and a clash silently opened a chat instead
    sched.add_argument("--command", dest="shell_command",
                       help="shell command for the task (runs with automation authority)")
    sched.add_argument("--notify", metavar="TEXT")
    sched.add_argument("--briefing", action="store_true", help="prepare the morning briefing")
    doctor = sub.add_parser("doctor", help="self-diagnostics")
    doctor.add_argument("--live", action="store_true",
                        help="also run end-to-end checks against the real model runtime (Ollama)")
    doctor.add_argument("--model", help="model to use for --live checks (default: JARVIS's choice)")
    sub.add_parser("models", help="installed models")
    events = sub.add_parser("events", help="recent events")
    events.add_argument("--limit", type=int, default=20)
    sub.add_parser("approvals", help="pending approvals")
    sub.add_parser("grants", help="active delegated authority")
    rt = sub.add_parser("runtime", help="manage the persistent runtime")
    rt.add_argument("op", choices=["start", "stop", "restart", "status", "health", "logs", "run", "install-service"])
    rt.add_argument("--foreground", action="store_true", help="start: run in this terminal")
    rt.add_argument("-n", "--lines", type=int, default=50, help="logs: how many lines")
    rt.add_argument("-f", "--follow", action="store_true", help="logs: keep following")
    rt.add_argument("--structured", action="store_true", help="logs: the structured JSON log instead")
    rt.add_argument("--write", action="store_true", help="install-service: write the file")
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            return asyncio.run(interactive_embedded(args)) if args.embedded else interactive_client(args)
        if args.command == "ask":
            return asyncio.run(cmd_ask(args)) if args.embedded else client_ask(args)
        handlers = {"status": cmd_status_any, "tasks": cmd_tasks_any, "task": cmd_task, "away": cmd_away,
                    "notifications": cmd_notifications, "briefing": cmd_briefing, "schedule": cmd_schedule,
                    "doctor": cmd_doctor_any, "models": cmd_models_any, "events": cmd_events_any,
                    "approvals": cmd_approvals_any, "grants": cmd_grants_any, "runtime": cmd_runtime}
        return handlers[args.command](args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
