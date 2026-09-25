"""The persistent runtime process: one Runtime, its local API, and a clean shutdown path.

``jarvis runtime run`` executes this in the foreground (what a systemd unit or
launchd agent runs); ``jarvis runtime start`` launches it detached. The
process writes ``runtime.json`` (pid, port) and ``api.token`` (mode 0600) into
the data directory so interfaces can find and authenticate to it, and removes
both on a clean stop. SIGTERM, SIGINT and ``POST /v1/runtime/stop`` all stop
it the same way: running tasks are checkpointed first.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import jarvis
from jarvis import __version__
from jarvis.config import JarvisConfig
from jarvis.log import get_logger
from jarvis.platforms import LockHeld
from jarvis.runtime import Runtime
from jarvis.service.api import ApiServer

log = get_logger("daemon")

INFO_FILE = "runtime.json"
TOKEN_FILE = "api.token"


def source_root() -> str:
    """The folder containing this ``jarvis`` package (a source checkout or site-packages)."""
    return str(Path(jarvis.__file__).resolve().parent.parent)


def info_path(data_dir: Path) -> Path:
    return Path(data_dir) / INFO_FILE


def token_path(data_dir: Path) -> Path:
    return Path(data_dir) / TOKEN_FILE


def _write_private(path: Path, text: str) -> None:
    """Write a file only the current user can read, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Daemon:
    def __init__(self, config: JarvisConfig, *, simulate: bool = False, runtime_kwargs: dict[str, Any] | None = None,
                 log_to_stderr: bool = False) -> None:
        self.config = config
        self.sim = None
        kwargs = dict(runtime_kwargs or {})
        if simulate:
            from jarvis.simulation.environment import SimulatedEnvironment
            self.sim = SimulatedEnvironment()
            kwargs.setdefault("providers", [self.sim.provider])
            kwargs.setdefault("metrics", self.sim.metrics)
            kwargs.setdefault("network_probe", self.sim.probe)
            kwargs["simulated"] = True
        self.runtime = Runtime(config, mode="daemon", log_to_stderr=log_to_stderr, **kwargs)
        self.server: ApiServer | None = None
        self._stop = asyncio.Event()
        self.data_dir = config.data_path

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self, *, ready: asyncio.Event | None = None) -> int:
        try:
            report = await self.runtime.start()
        except LockHeld as exc:
            print(f"jarvis runtime: {exc}", file=sys.stderr, flush=True)
            return 3
        token = secrets.token_urlsafe(32)
        _write_private(token_path(self.data_dir), token)
        rcfg = self.config.runtime
        self.server = ApiServer(self.runtime, token, host=rcfg.api_host, port=rcfg.api_port, on_stop=self.request_stop,
                                sim=self.sim)
        try:
            port = await self.server.start()
        except OSError as exc:
            print(f"jarvis runtime: cannot listen on {rcfg.api_host}:{rcfg.api_port} ({exc})", file=sys.stderr,
                  flush=True)
            await self.runtime.stop()
            return 4
        info = {"pid": os.getpid(), "host": rcfg.api_host, "port": port, "version": __version__,
                "run": self.runtime.run_id, "started_at": time.time(), "data_dir": str(self.data_dir),
                "token_file": str(token_path(self.data_dir)), "simulated": self.runtime.simulated,
                "python": sys.executable, "source": source_root()}
        _write_private(info_path(self.data_dir), json.dumps(info, indent=1))
        self._install_signals()
        log.info("daemon_ready", port=port, pid=os.getpid(), recovered=len(report.recovered))
        print(f"JARVIS runtime {__version__} ready on {rcfg.api_host}:{port} (pid {os.getpid()}). "
              f"{report.greeting()}", flush=True)
        if ready is not None:
            ready.set()
        watchdog = asyncio.create_task(self._watch_for_stalls(), name="stall-watchdog")
        try:
            await self._stop.wait()
        finally:
            log.info("daemon_stopping")
            watchdog.cancel()
            try:
                await watchdog
            except (asyncio.CancelledError, Exception):
                pass
            await self.server.stop()
            await self.runtime.stop()
            for path in (info_path(self.data_dir), token_path(self.data_dir)):
                try:
                    if path == info_path(self.data_dir) and json.loads(path.read_text()).get("pid") != os.getpid():
                        continue
                    path.unlink()
                except (OSError, ValueError):
                    pass
            print("JARVIS runtime stopped.", flush=True)
        return 0

    async def _watch_for_stalls(self, *, beat_s: float = 5.0, dump_after_s: float = 120.0) -> None:
        """If the event loop ever stops turning (the runtime hangs and stops answering), write every thread's
        stack to logs/stall-traces.log so the cause can be found afterwards. A timer is re-armed on every beat;
        it only fires when the loop has been blocked for ``dump_after_s``. Shorter hiccups are logged too."""
        import faulthandler
        path = Path(self.data_dir) / "logs" / "stall-traces.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(path, "a", encoding="utf-8")
        except OSError:
            return
        loop = asyncio.get_running_loop()
        try:
            while True:
                faulthandler.dump_traceback_later(dump_after_s, repeat=False, file=fh, exit=False)
                before = loop.time()
                await asyncio.sleep(beat_s)
                lag = loop.time() - before - beat_s
                if lag > min(2.0, dump_after_s / 2):
                    log.warning("event_loop_lag", seconds=round(lag, 1))
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} JARVIS {__version__} pid {os.getpid()}: the "
                             f"event loop was blocked for {lag:.1f}s"
                             f"{' (thread stacks above)' if lag >= dump_after_s else ''}\n")
                    fh.flush()
        finally:
            faulthandler.cancel_dump_traceback_later()
            fh.close()

    def _install_signals(self) -> None:
        loop = asyncio.get_running_loop()
        for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError, ValueError):
                # Windows: no add_signal_handler; fall back to signal.signal for the console signals
                try:
                    signal.signal(sig, lambda *_: loop.call_soon_threadsafe(self.request_stop))
                except (ValueError, OSError):
                    pass


def run_daemon(config: JarvisConfig, *, simulate: bool = False, log_to_stderr: bool = False) -> int:
    return asyncio.run(Daemon(config, simulate=simulate, log_to_stderr=log_to_stderr).run())
