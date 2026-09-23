"""Task-level verification (spec §37, §147-148).

A command returning 0 is not success; success is the task's declared
condition, checked against reality. The verifier inspects actual state (files,
exit codes, parsed test results, live processes) and returns an explicit
outcome — COMPLETE, PARTIAL, FAILED, BLOCKED or UNKNOWN — never flattening
partial success into success.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from jarvis.core.types import Outcome
from jarvis.tasks.models import StepStatus, Task
from jarvis.verification.test_results import parse_test_output


@dataclass
class VerificationReport:
    outcome: Outcome
    summary: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    performed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome.value, "summary": self.summary, "checks": self.checks,
                "performed": self.performed}


CommandRunner = Callable[[str, str | None], Awaitable[tuple[int | None, str]]]


class Verifier:
    def __init__(self, run_command: CommandRunner | None = None) -> None:
        self.run_command = run_command

    async def verify_task(self, task: Task) -> VerificationReport:
        steps_report = self._steps(task)
        cond = task.success_condition or {"type": "all_steps_ok"}
        ctype = cond.get("type", "all_steps_ok")
        if task.dry_run:
            return VerificationReport(Outcome.UNKNOWN, "dry run — nothing was executed, so nothing to verify",
                                      steps_report.checks, performed=False)
        if ctype == "all_steps_ok":
            return steps_report
        if steps_report.outcome == Outcome.FAILED:
            return steps_report
        if ctype == "tests":
            return self._tests(task, cond, steps_report)
        if ctype == "file_exists":
            path = os.path.expanduser(cond["path"])
            exists = os.path.exists(path)
            check = {"check": "file_exists", "path": path, "passed": exists}
            return VerificationReport(Outcome.COMPLETE if exists else Outcome.FAILED,
                                      f"{path} {'exists' if exists else 'does not exist'}",
                                      steps_report.checks + [check])
        if ctype == "command":
            if self.run_command is None:
                return VerificationReport(Outcome.UNKNOWN, "no command runner available for verification",
                                          steps_report.checks, performed=False)
            code, output = await self.run_command(cond["command"], task.cwd)
            expected = cond.get("expect_exit", 0)
            passed = code == expected
            check = {"check": "command", "command": cond["command"], "exit_code": code, "passed": passed}
            return VerificationReport(Outcome.COMPLETE if passed else Outcome.FAILED,
                                      f"verification command exited {code} (expected {expected})",
                                      steps_report.checks + [check])
        return VerificationReport(Outcome.UNKNOWN, f"unknown success condition {ctype!r}", steps_report.checks,
                                  performed=False)

    def _steps(self, task: Task) -> VerificationReport:
        checks = []
        failed = skipped = 0
        unverified = []
        for step in task.plan:
            verification = step.verification or {}
            checks.append({"step": step.description, "status": step.status.value,
                           "verified": verification.get("passed") if verification.get("performed") else None})
            if step.status == StepStatus.FAILED and not step.allow_failure:
                failed += 1
            elif step.status == StepStatus.SKIPPED:
                skipped += 1
            elif step.status != StepStatus.DONE:
                failed += 1
            if step.status == StepStatus.DONE and not verification.get("performed"):
                unverified.append(step.description)
        if failed:
            return VerificationReport(Outcome.FAILED, f"{failed} step(s) did not succeed", checks)
        if skipped:
            return VerificationReport(Outcome.PARTIAL, f"{skipped} step(s) were skipped", checks)
        summary = "all steps completed"
        summary += f"; {len(unverified)} without independent verification" if unverified else " and verified"
        return VerificationReport(Outcome.COMPLETE, summary, checks)

    def _tests(self, task: Task, cond: dict[str, Any], steps_report: VerificationReport) -> VerificationReport:
        step_id = cond.get("step")
        step = task.step(step_id) if step_id else next((s for s in reversed(task.plan) if s.result), None)
        if step is None or not step.result:
            return VerificationReport(Outcome.UNKNOWN, "no test output was captured", steps_report.checks)
        data = step.result.get("data") or {}
        output = f"{data.get('stdout', '')}\n{data.get('stderr', '')}" if isinstance(data, dict) else str(data)
        results = parse_test_output(output, step.result.get("exit_code"))
        task.outputs["tests"] = results.to_dict()
        check = {"check": "tests", **results.to_dict()}
        if not results.parsed:
            code = step.result.get("exit_code")
            outcome = Outcome.UNKNOWN if code is None else (Outcome.COMPLETE if code == 0 else Outcome.FAILED)
            return VerificationReport(outcome, f"test runner exited {code}; output not recognised",
                                      steps_report.checks + [check])
        # The objective was to *run* the tests: a run with failures completed and produced findings.
        require_pass = cond.get("require_pass", False)
        if require_pass and not results.ok:
            return VerificationReport(Outcome.FAILED, results.summary(), steps_report.checks + [check])
        return VerificationReport(Outcome.COMPLETE, results.summary(), steps_report.checks + [check])
