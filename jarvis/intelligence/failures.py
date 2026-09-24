"""Failure categories and recovery strategies (Phase 3 §33).

A failed node is classified before anything is retried, because the right response depends on why it failed:
a timeout may pass on a second try, a permission refusal never will, a missing tool needs a different tool, and
a step whose outcome is unknown must not be repeated blindly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from jarvis.tasks.models import Step, StepStatus, Task, TaskStatus


class FailureCategory(StrEnum):
    TRANSIENT = "transient"        # timeouts, a service briefly unreachable
    PERMISSION = "permission"      # not authorized, or refused by the safety policy
    RESOURCE = "resource"          # out of memory or disk, resource pressure
    TOOL = "tool"                  # a tool missing, unsupported, or broken on this machine
    DATA = "data"                  # missing file, bad input, unreadable output
    PLANNING = "planning"          # the plan itself was wrong (invalid arguments, unknown tool)
    DEPENDENCY = "dependency"      # something it depended on failed
    VERIFICATION = "verification"  # it ran, but independent verification says it didn't work
    UNKNOWN = "unknown"            # includes "the outcome is unknown" after a crash


class Strategy(StrEnum):
    RETRY = "retry"                # try the same node again after a delay
    ALTERNATIVE = "alternative"    # swap in the node's next alternative recipe
    WAIT = "wait"                  # wait for the resource, then retry
    REPLAN = "replan"              # rebuild the remaining plan around the failure
    ASK = "ask"                    # stop and ask the user (never retried automatically)
    SKIP = "skip"                  # optional work: continue without it
    FAIL = "fail"                  # give up on the node


STRATEGIES: dict[FailureCategory, list[Strategy]] = {
    FailureCategory.TRANSIENT: [Strategy.RETRY, Strategy.ALTERNATIVE, Strategy.REPLAN],
    FailureCategory.PERMISSION: [Strategy.ASK],
    FailureCategory.RESOURCE: [Strategy.WAIT, Strategy.ALTERNATIVE],
    FailureCategory.TOOL: [Strategy.ALTERNATIVE, Strategy.REPLAN],
    FailureCategory.DATA: [Strategy.ALTERNATIVE, Strategy.REPLAN],
    FailureCategory.PLANNING: [Strategy.REPLAN],
    FailureCategory.DEPENDENCY: [Strategy.SKIP],
    FailureCategory.VERIFICATION: [Strategy.REPLAN],
    FailureCategory.UNKNOWN: [Strategy.ASK],
}


@dataclass
class Failure:
    category: FailureCategory
    detail: str
    step: str = ""
    signature: str = ""            # identical failures share a signature (loop protection)

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category.value, "detail": self.detail, "step": self.step,
                "signature": self.signature}


_TRANSIENT = re.compile(r"timed? ?out|timeout|temporar|connection (reset|refused|aborted)|try again|unavailable|"
                        r"503|502|429|rate limit|busy|econnreset|network is unreachable", re.I)
_RESOURCE = re.compile(r"out of memory|memoryerror|no space left|disk (is )?full|cannot allocate|resource "
                       r"pressure|too many open files|enospc|enomem", re.I)
_TOOL = re.compile(r"command not found|is not recognized as an internal or external command|not supported on this "
                   r"platform|no such tool|unknown tool|executable file not found|no module named|not installed|"
                   r"cannot find the path|failed: .*(error|exception)", re.I)
_DATA = re.compile(r"no such file|does not exist|not found|not a directory|is a directory|invalid|unreadable|"
                   r"decode|parse|malformed|empty report", re.I)
_PERMISSION = re.compile(r"not authori[sz]ed|permission denied|access (is )?denied|requires .* authori[sz]ation|"
                         r"refused by the safety policy|blocked by safety|operation not permitted|outside the allowed",
                         re.I)


def classify(task: Task | None, *, verification_failed: bool = False, dependency_failed: bool = False) -> Failure:
    """Classify why a node's task failed, from the task record (never from the model's opinion)."""
    if dependency_failed:
        return Failure(FailureCategory.DEPENDENCY, "something it depended on failed")
    if verification_failed:
        return Failure(FailureCategory.VERIFICATION, "independent verification says it didn't work")
    if task is None:
        return Failure(FailureCategory.UNKNOWN, "its task record is missing")
    step = _failed_step(task)
    text = " ".join(filter(None, [task.status_reason, task.error or "", step.error if step else "",
                                  str((step.result or {}).get("summary", "")) if step else ""]))
    where = step.description if step else ""
    signature = f"{step.tool if step else ''}:{_normalise(text)[:80]}"
    if step is not None and step.outcome_unknown:
        return Failure(FailureCategory.UNKNOWN, f"'{where}' was interrupted and its outcome is unknown", where,
                       signature)
    if task.status == TaskStatus.BLOCKED and "internal error" in text:
        return Failure(FailureCategory.UNKNOWN, text[:300], where, signature)
    if _PERMISSION.search(text) or (task.status == TaskStatus.BLOCKED and "authori" in text):
        return Failure(FailureCategory.PERMISSION, text[:300], where, signature)
    if task.checkpoint.get("waiting_for") == "model" or "model_unavailable" in text:
        return Failure(FailureCategory.RESOURCE, "no language model is available", where, signature)
    if _RESOURCE.search(text):
        return Failure(FailureCategory.RESOURCE, text[:300], where, signature)
    if step is not None and step.result and step.result.get("status") in ("invalid", "unknown_tool"):
        return Failure(FailureCategory.PLANNING, text[:300], where, signature)
    if "invalid arguments" in text or "unknown tool" in text or "no tool named" in text:
        return Failure(FailureCategory.PLANNING, text[:300], where, signature)
    if _TOOL.search(text):
        return Failure(FailureCategory.TOOL, text[:300], where, signature)
    if _TRANSIENT.search(text):
        return Failure(FailureCategory.TRANSIENT, text[:300], where, signature)
    if _DATA.search(text):
        return Failure(FailureCategory.DATA, text[:300], where, signature)
    if step is not None and step.tool == "shell_execute":
        code = (step.result or {}).get("exit_code")
        if code in (127, 9009):          # command not found (POSIX shells / cmd.exe)
            return Failure(FailureCategory.TOOL, text[:300] or f"exit {code}", where, signature)
    return Failure(FailureCategory.UNKNOWN, text[:300] or "it failed without a reason", where, signature)


def _failed_step(task: Task) -> Step | None:
    for step in task.plan:
        if step.outcome_unknown and not step.finished:
            return step
    failed = [s for s in task.plan if s.status == StepStatus.FAILED]
    if failed:
        return failed[-1]
    return next((s for s in task.plan if s.error), None) or task.current_step


def _normalise(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()
