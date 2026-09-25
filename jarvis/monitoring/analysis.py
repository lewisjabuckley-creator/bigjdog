"""Threshold, trend and prediction analysis over metric samples (spec §79-81).

Deterministic and cheap: a sustained breach is distinguished from a transient
spike, trends come from least-squares fits over a sliding window, and
predictions are labelled as estimates with their fit quality.
"""

from __future__ import annotations

import operator
from collections import deque
from dataclasses import dataclass
from typing import Any

from jarvis.config import Threshold

# threshold name -> (metric key, comparison, human label)
THRESHOLD_RULES: dict[str, tuple[str, str, str]] = {
    "cpu_percent": ("cpu_percent", ">", "CPU usage"),
    "memory_percent": ("memory_percent", ">", "Memory usage"),
    "disk_percent": ("disk_percent", ">", "Disk usage"),
    "disk_critical_percent": ("disk_percent", ">", "Disk usage"),
    "gpu_temp_c": ("gpu_temp_c", ">", "GPU temperature"),
    "cpu_temp_c": ("cpu_temp_c", ">", "CPU temperature"),
    "battery_low_percent": ("battery_percent", "<", "Battery"),
}
_OPS = {">": operator.gt, "<": operator.lt}


@dataclass
class ThresholdEvent:
    name: str
    metric: str
    exceeded: bool          # True = crossed into breach, False = cleared
    value: float
    threshold: float
    severity: str
    sustained_s: float
    message: str

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, "metric": self.metric, "value": self.value, "threshold": self.threshold,
                "severity": self.severity, "sustained_s": self.sustained_s, "message": self.message}


class ThresholdDetector:
    def __init__(self, thresholds: dict[str, Threshold]) -> None:
        self.thresholds = thresholds
        self._breach_since: dict[str, float] = {}
        self._active: set[str] = set()

    def active(self) -> set[str]:
        return set(self._active)

    def evaluate(self, sample: dict[str, Any], now: float) -> list[ThresholdEvent]:
        events = []
        for name, th in self.thresholds.items():
            rule = THRESHOLD_RULES.get(name)
            if rule is None:
                continue
            metric, op, label = rule
            value = sample.get(metric)
            if not isinstance(value, (int, float)):
                continue
            if metric == "battery_percent" and sample.get("battery_plugged"):
                value = 100.0   # plugged in: not a low-battery situation
            breached = _OPS[op](value, th.value)
            unit = "°C" if metric.endswith("_c") else "%"
            if breached:
                since = self._breach_since.setdefault(name, now)
                sustained = now - since
                if name not in self._active and sustained >= th.sustain_s:
                    self._active.add(name)
                    for_text = f" for {int(sustained // 60)} minutes" if sustained >= 120 else ""
                    events.append(ThresholdEvent(name, metric, True, value, th.value, th.severity, sustained,
                                                 f"{label} is {value:.0f}{unit}{for_text} "
                                                 f"(threshold {th.value:.0f}{unit})"))
                continue
            self._breach_since.pop(name, None)
            if name in self._active:
                clear = th.clear_below if th.clear_below is not None else (
                    th.value * 0.95 if op == ">" else th.value * 1.05)
                if (op == ">" and value <= clear) or (op == "<" and value >= clear):
                    self._active.discard(name)
                    events.append(ThresholdEvent(name, metric, False, value, th.value, th.severity, 0.0,
                                                 f"{label} is back to {value:.0f}{unit}"))
        return events


@dataclass
class Trend:
    metric: str
    slope_per_hour: float
    r2: float
    span_s: float
    latest: float

    @property
    def direction(self) -> str:
        return "increasing" if self.slope_per_hour > 0 else "decreasing"


class TrendTracker:
    def __init__(self, window: int = 360) -> None:
        self.window = window
        self.series: dict[str, deque[tuple[float, float]]] = {}

    def add(self, metric: str, ts: float, value: float) -> None:
        self.series.setdefault(metric, deque(maxlen=self.window)).append((ts, float(value)))

    def add_sample(self, sample: dict[str, Any], ts: float) -> None:
        for key in ("cpu_percent", "memory_percent", "disk_percent", "disk_free_gb", "gpu_temp_c", "cpu_temp_c",
                    "battery_percent", "vram_used_gb"):
            value = sample.get(key)
            if isinstance(value, (int, float)):
                self.add(key, ts, value)

    def trend(self, metric: str, min_points: int = 10, min_span_s: float = 300.0) -> Trend | None:
        points = list(self.series.get(metric, ()))
        if len(points) < min_points:
            return None
        span = points[-1][0] - points[0][0]
        if span < min_span_s:
            return None
        n = len(points)
        t0 = points[0][0]
        xs = [p[0] - t0 for p in points]
        ys = [p[1] for p in points]
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0:
            return None
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return Trend(metric, slope * 3600, r2, span, ys[-1])

    def steady_trend(self, metric: str, min_r2: float = 0.8, min_abs_per_hour: float = 0.5) -> Trend | None:
        t = self.trend(metric)
        if t and t.r2 >= min_r2 and abs(t.slope_per_hour) >= min_abs_per_hour:
            return t
        return None

    def time_to_threshold(self, metric: str, threshold: float, min_r2: float = 0.8) -> float | None:
        """Estimated seconds until ``metric`` reaches ``threshold`` at the current rate (None if not converging)."""
        t = self.trend(metric)
        if t is None or t.r2 < min_r2 or t.slope_per_hour == 0:
            return None
        remaining = threshold - t.latest
        if remaining == 0:
            return 0.0
        if (remaining > 0) != (t.slope_per_hour > 0):
            return None
        return remaining / t.slope_per_hour * 3600
