"""Planner: turns an objective into validated tool steps (spec §16-17).

Plans come from deterministic templates when one matches, otherwise from a
language model constrained to the tool registry. Model output is treated as a
proposal: unknown tools and invalid arguments are rejected before anything is
stored, and plan length is bounded by the task budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from jarvis.log import get_logger
from jarvis.models.base import ChatMessage, ModelError, Purpose
from jarvis.models.router import ModelRouter, TaskProfile
from jarvis.tasks.models import Step, StepStatus, Task
from jarvis.tools.registry import Execution, ToolRegistry
from jarvis.tools.schema import SchemaError, validate_args

log = get_logger("planner")

_PLAN_INSTRUCTIONS = """You are the planning component of JARVIS, a local AI operating environment.
Produce a short, safe plan of tool calls that achieves the objective. Rules:
- Use only the tools listed. Arguments must match each tool's parameters.
- Inspect before acting. Prefer read-only steps first; include verification steps.
- Do not include destructive actions unless the objective explicitly requires them.
- At most {max_steps} steps.
Respond with JSON only: {{"steps": [{{"description": "...", "tool": "<tool name>", "args": {{...}}}}], "notes": "..."}}"""


@dataclass
class PlanResult:
    steps: list[Step]
    source: str                 # template | model | none
    notes: str = ""
    rejected: list[str] = field(default_factory=list)


class Planner:
    def __init__(self, registry: ToolRegistry, router: ModelRouter | None = None) -> None:
        self.registry = registry
        self.router = router

    @property
    def available(self) -> bool:
        return self.router is not None and self.router.available()

    def _tool_catalog(self) -> str:
        lines = []
        for tool in self.registry.list():
            spec = tool.spec
            params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in spec.parameters.get("properties", {}).items())
            lines.append(f"- {spec.name}({params}) [level {int(spec.level)}]: {spec.description}")
        return "\n".join(lines)

    def validate(self, raw_steps: list[Any], max_steps: int) -> tuple[list[Step], list[str]]:
        steps, rejected = [], []
        for raw in raw_steps[:max_steps]:
            if not isinstance(raw, dict):
                rejected.append(f"not an object: {raw!r}")
                continue
            tool_name = raw.get("tool")
            tool = self.registry.get(tool_name) if tool_name else None
            if tool is None:
                rejected.append(f"unknown tool {tool_name!r}")
                continue
            try:
                args = validate_args(tool.spec.parameters, raw.get("args") or {})
            except SchemaError as exc:
                rejected.append(f"{tool_name}: {exc}")
                continue
            steps.append(Step(str(raw.get("description") or tool.preview(args))[:200], tool_name, args))
        return steps, rejected

    async def plan(self, task: Task, context: str = "") -> PlanResult:
        if task.plan:
            return PlanResult(task.plan, "existing")
        if not self.available:
            return PlanResult([], "none", "no language model is available for planning")
        max_steps = int(task.budget.get("max_steps", 20))
        prompt = (f"Objective: {task.objective}\nWorking directory: {task.cwd or 'unknown'}\n"
                  f"{context}\nAvailable tools:\n{self._tool_catalog()}")
        return await self._ask(task, prompt, max_steps)

    async def replan(self, task: Task, failed: Step, execution: Execution | None, context: str = "") -> PlanResult:
        if not self.available:
            return PlanResult([], "none", "no language model is available for replanning")
        done = [f"- {s.description}: {s.result.get('summary') if s.result else 'done'}" for s in task.completed_steps()]
        remaining = [f"- {s.description}" for s in task.plan if not s.finished and s.id != failed.id]
        failure = execution.describe() if execution else failed.error
        prompt = (f"Objective: {task.objective}\nWorking directory: {task.cwd or 'unknown'}\n{context}\n"
                  f"Completed steps:\n{chr(10).join(done) or '- none'}\n"
                  f"Failed step: {failed.description} ({failed.tool} {json.dumps(failed.args)[:300]})\n"
                  f"Failure: {failure}\nRemaining planned steps:\n{chr(10).join(remaining) or '- none'}\n"
                  "Produce replacement steps for the failed step and everything after it.\n"
                  f"Available tools:\n{self._tool_catalog()}")
        remaining_budget = max(1, int(task.budget.get("max_steps", 20)) - len(task.completed_steps()))
        return await self._ask(task, prompt, remaining_budget)

    async def _ask(self, task: Task, prompt: str, max_steps: int) -> PlanResult:
        assert self.router is not None
        messages = [ChatMessage("system", _PLAN_INSTRUCTIONS.format(max_steps=max_steps)),
                    ChatMessage("user", prompt)]
        try:
            routed = await self.router.chat(TaskProfile(purpose=Purpose.PLANNING, complexity="medium",
                                                        interactive=False), messages, format="json")
        except ModelError as exc:
            return PlanResult([], "none", f"planning model unavailable: {exc}")
        task.usage["model_calls"] = task.usage.get("model_calls", 0) + 1
        try:
            data = json.loads(_extract_json(routed.response.content))
        except ValueError:
            return PlanResult([], "none", "the planning model returned an unreadable plan")
        raw_steps = data.get("steps") if isinstance(data, dict) else data
        if not isinstance(raw_steps, list):
            return PlanResult([], "none", "the planning model returned no steps")
        steps, rejected = self.validate(raw_steps, max_steps)
        if rejected:
            log.warning("plan_steps_rejected", task_id=task.id, rejected=rejected)
        return PlanResult(steps, "model", str(data.get("notes", "")) if isinstance(data, dict) else "", rejected)


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start:end + 1] if end > start else text


def reset_pending(steps: list[Step]) -> list[Step]:
    for s in steps:
        s.status = StepStatus.PENDING
    return steps
