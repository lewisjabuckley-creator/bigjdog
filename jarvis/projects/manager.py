"""Projects and project isolation (spec §40-41, §87).

A project bundles a root directory with a policy: which directories, tools,
network access and models it may use, and whether it is sensitive. Opening a
project loads its context (repository facts, recent work, memories, decisions)
and makes its policy the active execution scope.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import Confidence, Provenance, ProvenanceKind, new_id
from jarvis.database.db import Database, dumps, loads
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.planner.templates import probe_project
from jarvis.state.engine import StateEngine
from jarvis.world.model import CONTAINS, OWNS, WorldModel


@dataclass
class ProjectPolicy:
    allowed_dirs: list[str] = field(default_factory=list)       # empty = the project root
    allowed_tools: list[str] | None = None                      # None = every registered tool
    network: bool = True
    local_models_only: bool = False
    sensitive: bool = False
    retention_days: int | None = None
    allowed_agents: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectPolicy":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Project:
    id: str
    name: str
    root: str | None
    description: str = ""
    policy: ProjectPolicy = field(default_factory=ProjectPolicy)
    created_at: float = 0.0
    updated_at: float = 0.0
    last_opened: float | None = None

    @property
    def dirs(self) -> list[str]:
        return self.policy.allowed_dirs or ([self.root] if self.root else [])


def git_branch(root: str | None) -> str | None:
    if not root:
        return None
    head = Path(root) / ".git" / "HEAD"
    try:
        text = head.read_text().strip()
    except OSError:
        return None
    return text.removeprefix("ref: refs/heads/") if text.startswith("ref:") else text[:12]


class ProjectManager:
    def __init__(self, db: Database, state: StateEngine, *, world: WorldModel | None = None,
                 bus: EventBus | None = None, clock: Clock | None = None, owner: str = "owner") -> None:
        self.db = db
        self.state = state
        self.world = world
        self.bus = bus
        self.clock = clock or SystemClock()
        self.owner = owner

    def create(self, name: str, root: str | None = None, *, description: str = "",
               policy: ProjectPolicy | None = None) -> Project:
        now = self.clock.now()
        root = os.path.realpath(os.path.expanduser(root)) if root else None
        project = Project(new_id("proj"), name, root, description, policy or ProjectPolicy(), now, now)
        self.db.execute("INSERT INTO projects(id, name, root, description, policy, created_at, updated_at) "
                        "VALUES(?,?,?,?,?,?,?)", (project.id, name, root, description,
                                                  dumps(project.policy.to_dict()), now, now))
        if self.world:
            self.world.upsert_entity("project", name, {"root": root, "sensitive": project.policy.sensitive},
                                     id=f"project:{project.id}", source="projects")
            self.world.relate(f"user:{self.owner}", OWNS, f"project:{project.id}")
            if root:
                self.world.upsert_entity("folder", root, id=f"folder:{root}")
                self.world.relate(f"project:{project.id}", CONTAINS, f"folder:{root}")
        return project

    def update_policy(self, project_id: str, policy: ProjectPolicy) -> None:
        self.db.execute("UPDATE projects SET policy=?, updated_at=? WHERE id=?",
                        (dumps(policy.to_dict()), self.clock.now(), project_id))

    def get(self, project_id: str) -> Project | None:
        row = self.db.query_one("SELECT * FROM projects WHERE id=?", (project_id,))
        return _row(row) if row else None

    def get_by_name(self, name: str) -> Project | None:
        row = self.db.query_one("SELECT * FROM projects WHERE name=? COLLATE NOCASE", (name,))
        return _row(row) if row else None

    def list(self) -> list[Project]:
        return [_row(r) for r in self.db.query("SELECT * FROM projects ORDER BY IFNULL(last_opened, 0) DESC, name")]

    def find(self, text: str) -> list[Project]:
        needle = text.lower().strip().removeprefix("the ").removesuffix(" project").strip()
        projects = self.list()
        exact = [p for p in projects if p.name.lower() == needle]
        if exact:
            return exact
        return [p for p in projects if needle and (needle in p.name.lower() or p.name.lower() in needle
                                                   or (p.root and needle in os.path.basename(p.root).lower()))]

    def for_path(self, path: str) -> Project | None:
        real = os.path.realpath(os.path.expanduser(path))
        best = None
        for p in self.list():
            if p.root and (real == p.root or real.startswith(p.root.rstrip(os.sep) + os.sep)):
                if best is None or len(p.root) > len(best.root or ""):
                    best = p
        return best

    # -- active project ------------------------------------------------------------------
    def active(self) -> Project | None:
        active = self.state.value("project.active")
        return self.get(active["id"]) if isinstance(active, dict) and active.get("id") else None

    def open(self, project: Project) -> dict[str, Any]:
        now = self.clock.now()
        self.db.execute("UPDATE projects SET last_opened=? WHERE id=?", (now, project.id))
        previous = self.active()
        self.state.set("project.active", {"id": project.id, "name": project.name, "root": project.root},
                       confidence=Confidence.KNOWN, provenance=Provenance(ProvenanceKind.USER_STATEMENT, "open project"))
        context = self.context(project)
        if self.bus:
            self.bus.emit(Event(EventType.PROJECT_CHANGED, "projects",
                                {"project_id": project.id, "name": project.name,
                                 "previous": previous.name if previous else None},
                                entity_id=f"project:{project.id}"))
        return context

    def close(self) -> None:
        self.state.delete("project.active")

    def context(self, project: Project) -> dict[str, Any]:
        ctx: dict[str, Any] = {"name": project.name, "root": project.root, "description": project.description,
                               "sensitive": project.policy.sensitive}
        if project.root and os.path.isdir(project.root):
            probe = probe_project(project.root)
            ctx.update({"languages": probe.languages, "test_command": probe.test_command,
                        "build_command": probe.build_command, "branch": git_branch(project.root)})
        return ctx

    def discover(self, root: str, name: str | None = None) -> Project:
        """Register a directory as a project if it is not already known."""
        existing = self.for_path(root)
        real = os.path.realpath(os.path.expanduser(root))
        if existing and existing.root == real:
            return existing
        return self.create(name or os.path.basename(real.rstrip(os.sep)) or real, real)


def _row(r: Any) -> Project:
    return Project(r["id"], r["name"], r["root"], r["description"] or "", ProjectPolicy.from_dict(loads(r["policy"], {})),
                   r["created_at"], r["updated_at"], r["last_opened"])
