"""Model routing policy for planned work (Phase 3 §11-12).

The router already knows about purposes, capabilities, pins, health and fallback; this adds the planning
layer's view: how complex and important the work is, whether someone is waiting for it, how much context it
needs, whether it must stay private, and how much memory is free. Under memory or GPU pressure the choice moves
to a smaller or already-loaded model; with room to spare, complex background work may use a larger one. The
router's fallback behaviour is unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

from jarvis.models.base import Purpose
from jarvis.models.router import TaskProfile


class RoutingPolicy:
    def __init__(self, *, resources: Any = None, local_only: Callable[[], bool] | None = None) -> None:
        self.resources = resources
        self.local_only = local_only or (lambda: False)

    def _pressure(self) -> tuple[bool, str]:
        if self.resources is None:
            return False, ""
        constrained, why = self.resources.pressure()
        if constrained:
            return True, why
        gpu, why_gpu = self.resources.gpu_pressure()
        return gpu, why_gpu

    def profile(self, purpose: Purpose, *, complexity: str = "medium", interactive: bool = False,
                context_tokens: int = 0, importance: str = "normal", private: bool = False,
                needs_tools: bool = False) -> tuple[TaskProfile, str]:
        """A task profile and the reason for it (recorded with the decision that used it)."""
        constrained, why = self._pressure()
        profile = TaskProfile(purpose=purpose, complexity=complexity, needs_tools=needs_tools, interactive=interactive,
                              min_context=int(context_tokens * 1.3) if context_tokens else 0,
                              local_only=private or self.local_only())
        reasons = []
        if interactive:
            profile.priority = 0
            if importance != "high" and complexity == "high":
                profile.complexity = "medium"     # someone is waiting: favour a faster model
                reasons.append("someone is waiting, so a faster model")
        else:
            profile.priority = 1 if importance == "high" else 2
        if constrained:
            profile.complexity = "low"
            profile.prefer_loaded = True
            profile.max_params_b = 8.0
            reasons.append(f"resources are short ({why}), so a smaller or already-loaded model")
        elif importance == "high" and not interactive:
            profile.complexity = "high"
            reasons.append("important background work, so the most capable model")
        if profile.local_only:
            reasons.append("kept local for privacy")
        return profile, "; ".join(reasons) or f"{complexity}-complexity {purpose.value}"

    def adjust(self, profile: TaskProfile) -> TaskProfile:
        """Router hook: background requests move to a smaller or already-loaded model under memory or GPU
        pressure. Interactive requests are left alone (the user asked for that answer now)."""
        if profile.interactive or profile.max_params_b is not None:
            return profile
        constrained, _ = self._pressure()
        if not constrained:
            return profile
        from dataclasses import replace
        return replace(profile, complexity="low", prefer_loaded=True, max_params_b=8.0)

    def agent_profile(self, spec: Any, base: TaskProfile) -> TaskProfile:
        """Hook for the agent runner: agents are background work and yield to the user and to pressure."""
        profile, _ = self.profile(spec.purpose, complexity=getattr(spec, "complexity", "medium"),
                                  needs_tools=base.needs_tools)
        return profile
