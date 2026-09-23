"""Failure recovery decisions (spec §38-39).

Deterministic policy for what to do after a step fails. Every decision comes
with an operational reason so JARVIS can later explain "why did you retry?".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from jarvis.core.types import OperationalReason
from jarvis.tasks.models import Step, Task
from jarvis.tools.registry import Execution, ExecStatus


class RecoveryAction(StrEnum):
    RETRY = "retry"
    REPLAN = "replan"
    CONTINUE = "continue"
    BLOCK = "block"
    FAIL = "fail"


@dataclass
class RecoveryDecision:
    action: RecoveryAction
    reason: OperationalReason
    delay_s: float = 0.0


class RecoveryPolicy:
    def __init__(self, *, planner_available: bool = True) -> None:
        self.planner_available = planner_available

    def decide(self, task: Task, step: Step, execution: Execution, *, idempotent: bool,
               can_replan: bool | None = None) -> RecoveryDecision:
        can_replan = self.planner_available if can_replan is None else can_replan
        policy = task.policy
        err = (execution.result.error if execution.result else None) or execution.message or "unknown error"
        verification_failed = bool(execution.verification and execution.verification.performed
                                   and execution.verification.passed is False and execution.result
                                   and execution.result.ok)

        def reason(condition: str, rule: str, action: str, expected: str = "") -> OperationalReason:
            return OperationalReason(condition, rule, action, expected, {"step": step.description})

        if execution.status == ExecStatus.DENIED:
            return RecoveryDecision(RecoveryAction.BLOCK, reason(
                f"'{step.description}' was not authorized ({execution.message})",
                "capability is not permission", "blocked the task",
                "It can resume once authority is granted."))
        if verification_failed:
            if can_replan and task.usage.get("replans", 0) < policy.max_replans:
                return RecoveryDecision(RecoveryAction.REPLAN, reason(
                    f"the tool reported success but verification failed: {execution.verification.detail}",
                    "subagent and tool output is evidence, not truth", "rejected the result and replanned"))
            return RecoveryDecision(RecoveryAction.FAIL, reason(
                f"verification failed: {execution.verification.detail}", "never report unverified success",
                "marked the step failed"))
        # Only crashes/timeouts of idempotent steps are retried: re-running a non-idempotent action
        # could apply its effect twice, and re-running a clean non-zero exit rarely changes the answer.
        if execution.status == ExecStatus.FAILED and idempotent and step.attempts <= policy.max_retries:
            delay = policy.retry_backoff_s * (2 ** max(0, step.attempts - 1))
            return RecoveryDecision(RecoveryAction.RETRY, reason(
                f"'{step.description}' failed ({err}) on attempt {step.attempts}",
                f"idempotent steps may be retried up to {policy.max_retries} times", "retried the step",
                f"Next attempt in {delay:g}s."), delay)
        if step.allow_failure or policy.on_step_failure == "continue":
            return RecoveryDecision(RecoveryAction.CONTINUE, reason(
                f"'{step.description}' failed ({err})", "the step's failure is non-fatal for this task",
                "continued with the remaining steps"))
        if execution.status in (ExecStatus.INVALID, ExecStatus.UNKNOWN_TOOL) or \
                (policy.on_step_failure == "replan" and can_replan):
            if can_replan and task.usage.get("replans", 0) < policy.max_replans:
                return RecoveryDecision(RecoveryAction.REPLAN, reason(
                    f"'{step.description}' failed ({err})", "adapt the plan when a step fails",
                    "asked the planner for an alternative"))
        return RecoveryDecision(RecoveryAction.FAIL, reason(
            f"'{step.description}' failed ({err})", "no safe retry or alternative is available",
            "stopped the task and preserved its checkpoint",
            "Completed work is kept; the task can be retried."))
