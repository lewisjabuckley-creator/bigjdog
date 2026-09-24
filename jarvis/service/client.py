"""Client side of the local API: find the runtime, start it if needed, talk to it.

Interfaces (the CLI today, a HUD later) use this and never touch the runtime's
database or subsystems directly.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx

import jarvis
from jarvis.platforms import current as current_platform
from jarvis.service.daemon import info_path, token_path


class RuntimeNotRunning(RuntimeError):
    pass


class ApiFailure(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{message} (HTTP {status})")
        self.status = status
        self.message = message


@dataclass
class RuntimeInfo:
    pid: int
    host: str
    port: int
    token: str
    data_dir: str
    version: str = ""
    run: str | None = None
    started_at: float = 0.0
    simulated: bool = False

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


def read_info(data_dir: Path) -> RuntimeInfo | None:
    """The running runtime for a data directory, or None (not running, or a stale record from a crash)."""
    try:
        info = json.loads(info_path(data_dir).read_text())
        token = token_path(data_dir).read_text().strip()
    except (OSError, ValueError):
        return None
    if not current_platform().is_alive(int(info.get("pid") or 0)):
        return None
    return RuntimeInfo(int(info["pid"]), info.get("host", "127.0.0.1"), int(info["port"]), token, str(data_dir),
                       info.get("version", ""), info.get("run"), float(info.get("started_at") or 0),
                       bool(info.get("simulated")))


class Client:
    def __init__(self, info: RuntimeInfo, *, timeout: float = 60.0) -> None:
        self.info = info
        self.http = httpx.Client(base_url=info.base_url, headers={"Authorization": f"Bearer {info.token}"},
                                 timeout=httpx.Timeout(timeout, connect=5.0), trust_env=False)

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _check(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            data = {"error": response.text[:200]}
        if response.status_code >= 400:
            raise ApiFailure(response.status_code, str(data.get("error") or data))
        return data

    def ping(self) -> bool:
        try:
            return bool(self.http.get("/v1/ping", timeout=3.0).json().get("ok"))
        except (httpx.HTTPError, ValueError):
            return False

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        return self._check(self.http.get(path, params={k: v for k, v in params.items() if v is not None}))

    def post(self, path: str, body: dict[str, Any] | None = None, *, timeout: float | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"json": body or {}}
        if timeout is not None:
            kwargs["timeout"] = httpx.Timeout(timeout, connect=5.0)
        return self._check(self.http.post(path, **kwargs))

    def delete(self, path: str) -> dict[str, Any]:
        return self._check(self.http.delete(path))

    def stream(self, method: str, path: str, body: dict[str, Any] | None = None,
               params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Newline-delimited JSON items from a streaming endpoint (no read timeout: answers can be slow)."""
        with self.http.stream(method, path, json=body, params=params,
                              timeout=httpx.Timeout(None, connect=5.0)) as response:
            if response.status_code != 200:
                response.read()
                self._check(response)
            for line in response.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def converse(self, text: str, *, request_id: str, session: str = "default", cwd: str | None = None,
                 client_id: str | None = None) -> dict[str, Any]:
        return self.post("/v1/conversation", {"text": text, "request_id": request_id, "session": session,
                                              "cwd": cwd, "client_id": client_id}, timeout=None)


def daemon_argv(*, config: str | None, data_dir: str, simulate: bool) -> list[str]:
    platform = current_platform()
    argv = [platform.python(), "-m", "jarvis"]
    if config:
        argv += ["--config", str(Path(config).expanduser().resolve())]
    argv += ["--data-dir", str(Path(data_dir).expanduser().resolve())]
    if simulate:
        argv.append("--simulate")
    return argv + ["runtime", "run"]


def source_root() -> str:
    """The folder containing this ``jarvis`` package (a source checkout or site-packages)."""
    return str(Path(jarvis.__file__).resolve().parent.parent)


def daemon_env() -> dict[str, str]:
    """The runtime must import this same JARVIS even when it isn't installed (run from a source folder)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([source_root()] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                                          if p])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def runtime_log(data_dir: Path) -> Path:
    return Path(data_dir) / "logs" / "runtime.out"


def launch(*, config: str | None, data_dir: Path, simulate: bool = False, timeout: float = 30.0) -> RuntimeInfo:
    """Start the runtime in the background and wait until its API answers."""
    platform = current_platform()
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_log(data_dir)
    offset = log_path.stat().st_size if log_path.exists() else 0
    pid = platform.spawn_detached(daemon_argv(config=config, data_dir=str(data_dir), simulate=simulate), log_path,
                                  env=daemon_env(), cwd=str(data_dir))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = read_info(data_dir)
        if info is not None and info.pid == pid:
            with Client(info) as client:
                if client.ping():
                    return info
        if not platform.is_alive(pid):
            tail = ""
            try:
                with open(log_path, "rb") as fh:
                    fh.seek(offset)
                    tail = fh.read().decode(errors="replace").strip()[-800:]
            except OSError:
                pass
            raise RuntimeNotRunning(f"the JARVIS runtime exited during startup. {tail}".strip())
        time.sleep(0.2)
    raise RuntimeNotRunning(f"the JARVIS runtime did not become ready within {timeout:.0f}s; see {log_path}")


def connect(*, data_dir: Path, config: str | None = None, simulate: bool = False, auto_start: bool = True,
            announce: Any = None) -> Client:
    info = read_info(data_dir)
    if info is not None:
        client = Client(info)
        if client.ping():
            return client
        client.close()
    if not auto_start:
        raise RuntimeNotRunning("the JARVIS runtime is not running (start it with: jarvis runtime start)")
    if announce:
        announce("Starting the JARVIS runtime in the background…")
    return Client(launch(config=config, data_dir=data_dir, simulate=simulate))


def is_windows() -> bool:
    return sys.platform == "win32"
