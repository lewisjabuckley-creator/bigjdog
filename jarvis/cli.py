"""Command-line interface: one interface into JARVIS, not the architecture itself.

    jarvis                 interactive session
    jarvis ask "..."       one request, then exit
    jarvis status          compact system status
    jarvis tasks           task list
    jarvis doctor          self-diagnostics
    jarvis models          installed models
    jarvis events          recent events
    jarvis approvals       pending approvals
    jarvis grants          active delegated authority
    jarvis --simulate ...  run against simulated hardware, network and model
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from typing import Any

from jarvis import __version__
from jarvis.clock import format_datetime, format_time
from jarvis.config import ConfigError, JarvisConfig, load_config
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
    if not sys.stdout.isatty():
        return text
    return f"{_COLORS.get(prio, '')}{text}{_RESET}"


def build_runtime(args: argparse.Namespace) -> tuple[Runtime, SimulatedEnvironment | None]:
    cfg: JarvisConfig = load_config(args.config)
    if args.data_dir:
        cfg.general.data_dir = args.data_dir
    sim = None
    if args.simulate:
        sim = SimulatedEnvironment()
        if not args.data_dir:
            cfg.general.data_dir = os.path.join(tempfile.gettempdir(), "jarvis-sim")
        return Runtime(cfg, providers=[sim.provider], metrics=sim.metrics, network_probe=sim.probe,
                       simulated=True, log_to_stderr=args.verbose), sim
    return Runtime(cfg, log_to_stderr=args.verbose), None


async def _with_runtime(args: argparse.Namespace, fn: Any, *, monitoring: bool = False) -> int:
    runtime, sim = build_runtime(args)
    report = await runtime.start(monitoring=monitoring)
    try:
        return await fn(runtime, report, sim)
    finally:
        await runtime.stop()


# -- interactive session ------------------------------------------------------------------------------

async def interactive(args: argparse.Namespace) -> int:
    runtime, sim = build_runtime(args)
    report = await runtime.start()
    svc = runtime.svc
    assert svc is not None
    orch = runtime.orchestrator()
    prompt = "you › "

    def sink(n: Notification) -> None:
        print(f"\r{_color('● ' + n.text(), n.priority)}\n{prompt}", end="", flush=True)

    svc.notifications.sinks.append(sink)
    label = " (simulation)" if sim else ""
    print(f"JARVIS {__version__}{label} — {report.greeting()}  Type 'help', or /quit to exit.")

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
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if line in ("/quit", "/exit", "/q"):
                    break
                print(await slash(line, runtime, sim))
                continue
            svc.notifications.set_activity("typing")
            try:
                response = await orch.handle(line)
            except Exception as exc:  # never let the interface die on one bad turn
                print(f"jarvis › internal error: {exc}")
                continue
            print(f"jarvis › {response.render()}")
    finally:
        drainer.cancel()
        svc.notifications.sinks.remove(sink)
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
        response = await runtime.orchestrator().handle(" ".join(args.text))
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
        return 0
    return await _with_runtime(args, run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="JARVIS — local-first AI operating environment")
    parser.add_argument("--config", help="path to jarvis.toml")
    parser.add_argument("--data-dir", help="override the data directory")
    parser.add_argument("--simulate", action="store_true", help="run against simulated hardware, network and model")
    parser.add_argument("--verbose", action="store_true", help="structured logs to stderr")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    sub = parser.add_subparsers(dest="command")
    ask = sub.add_parser("ask", help="handle one request and exit")
    ask.add_argument("text", nargs="+")
    sub.add_parser("status", help="compact system status")
    tasks = sub.add_parser("tasks", help="list tasks")
    tasks.add_argument("--all", action="store_true")
    sub.add_parser("doctor", help="self-diagnostics")
    sub.add_parser("models", help="installed models")
    events = sub.add_parser("events", help="recent events")
    events.add_argument("--limit", type=int, default=20)
    sub.add_parser("approvals", help="pending approvals")
    sub.add_parser("grants", help="active delegated authority")
    args = parser.parse_args(argv)
    handlers = {None: interactive, "ask": cmd_ask, "status": cmd_status, "tasks": cmd_tasks, "doctor": cmd_doctor,
                "models": cmd_models, "events": cmd_events, "approvals": cmd_approvals, "grants": cmd_grants}
    try:
        return asyncio.run(handlers[args.command](args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
