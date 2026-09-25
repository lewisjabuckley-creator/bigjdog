"""Metric sources: where live resource numbers come from.

The monitors depend on the :class:`MetricsSource` protocol, not on psutil, so the
simulation environment can drive every resource scenario deterministically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any, Protocol

import psutil

from jarvis.platforms import hidden_window_kwargs


class MetricsSource(Protocol):
    def sample(self) -> dict[str, Any]: ...


class PsutilMetrics:
    def __init__(self, disk_path: str | None = None, gpu_cache_s: float = 10.0) -> None:
        self.disk_path = disk_path or os.path.expanduser("~")
        self._gpu_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._gpu_cache_s = gpu_cache_s
        self._nvidia_smi = shutil.which("nvidia-smi")
        psutil.cpu_percent(interval=None)  # prime the counter
        self._primed_at = time.monotonic()

    def _cpu_percent(self) -> float:
        # A reading taken just after priming covers only milliseconds and is meaningless (often 0 or 100%);
        # measure over a short window instead.
        if time.monotonic() - self._primed_at < 0.5:
            value = psutil.cpu_percent(interval=0.5)
        else:
            value = psutil.cpu_percent(interval=None)
        self._primed_at = -1e9
        return value

    def sample(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        du = psutil.disk_usage(self.disk_path)
        data: dict[str, Any] = {
            "cpu_percent": self._cpu_percent(),
            "cpu_count": psutil.cpu_count(),
            "memory_percent": vm.percent,
            "memory_used_gb": round(vm.used / 2**30, 2),
            "memory_total_gb": round(vm.total / 2**30, 2),
            "swap_percent": psutil.swap_memory().percent,
            "disk_percent": du.percent,
            "disk_free_gb": round(du.free / 2**30, 2),
            "disk_path": self.disk_path,
            "process_count": len(psutil.pids()),
            "uptime_s": round(time.time() - psutil.boot_time()),
        }
        try:
            data["load_avg_1m"] = os.getloadavg()[0]
        except (OSError, AttributeError):
            pass
        battery = _safe(psutil.sensors_battery) if hasattr(psutil, "sensors_battery") else None
        if battery is not None:
            data["battery_percent"] = battery.percent
            data["battery_plugged"] = battery.power_plugged
        temps = _safe(psutil.sensors_temperatures) if hasattr(psutil, "sensors_temperatures") else None
        if temps:
            readings = [t.current for entries in temps.values() for t in entries if t.current]
            if readings:
                data["cpu_temp_c"] = max(readings)
        gpus = self._gpus()
        if gpus:
            data["gpus"] = gpus
            data["gpu_percent"] = max(g["utilization"] for g in gpus)
            data["vram_used_gb"] = round(sum(g["memory_used_mb"] for g in gpus) / 1024, 2)
            data["vram_total_gb"] = round(sum(g["memory_total_mb"] for g in gpus) / 1024, 2)
            data["gpu_temp_c"] = max(g["temperature_c"] for g in gpus)
        return data

    def _gpus(self) -> list[dict[str, Any]]:
        if not self._nvidia_smi:
            return []
        now = time.monotonic()
        if self._gpu_cache and now - self._gpu_cache[0] < self._gpu_cache_s:
            return self._gpu_cache[1]
        gpus: list[dict[str, Any]] = []
        try:
            out = subprocess.run(
                [self._nvidia_smi, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True,
                **hidden_window_kwargs()).stdout
            for line in out.strip().splitlines():
                name, util, used, total, temp = [x.strip() for x in line.split(",")]
                gpus.append({"name": name, "utilization": float(util), "memory_used_mb": float(used),
                             "memory_total_mb": float(total), "temperature_c": float(temp)})
        except (subprocess.SubprocessError, OSError, ValueError):
            gpus = []
        self._gpu_cache = (now, gpus)
        return gpus


class StaticMetrics:
    """Settable metrics for tests and simulation."""

    def __init__(self, **values: Any) -> None:
        self.values: dict[str, Any] = {"cpu_percent": 10.0, "memory_percent": 40.0, "disk_percent": 50.0,
                                       "memory_used_gb": 6.4, "memory_total_gb": 16.0, "disk_free_gb": 200.0}
        self.values.update(values)

    def set(self, **values: Any) -> None:
        self.values.update(values)

    def sample(self) -> dict[str, Any]:
        return dict(self.values)


def _safe(fn: Any) -> Any:
    try:
        return fn()
    except Exception:
        return None
