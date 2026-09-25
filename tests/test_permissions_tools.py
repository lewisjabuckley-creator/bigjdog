import asyncio
import os
from typing import Any

import pytest

from jarvis.audit.log import AuditLog
from jarvis.config import PermissionsConfig
from jarvis.core.types import RiskLevel
from jarvis.permissions.hierarchy import Directive, InstructionSource, may_override, resolve
from jarvis.permissions.manager import ApprovalManager, PermissionManager
from jarvis.permissions.model import AccessRequest, Actor, PermissionLevel
from jarvis.tools.base import ExecutionPolicy, Tool, ToolContext, ToolResult, ToolSpec, Verification
from jarvis.tools.builtin import register_builtin_tools
from jarvis.tools.builtin.shell import classify_command
from jarvis.tools.registry import ExecStatus, ToolRegistry

L = PermissionLevel


@pytest.fixture
def perms(db, clock, tmp_path):
    cfg = PermissionsConfig(allowed_roots=[str(tmp_path)], denied_paths=[str(tmp_path / "secrets")])
    return PermissionManager(db, cfg, clock=clock)


@pytest.fixture
def registry(db, clock, perms):
    reg = ToolRegistry(perms, AuditLog(db, clock))
    register_builtin_tools(reg)
    return reg


def ctx_for(tmp_path, actor=None, **kw):
    return ToolContext(actor=actor or Actor.user(), cwd=str(tmp_path), data_dir=str(tmp_path / ".jarvis"), **kw)


# -- permission model -----------------------------------------------------------------

def test_interactive_baseline_and_consequential_needs_approval(perms):
    user = Actor.user()
    assert perms.check(AccessRequest(user, "file_write", L.EXECUTE_REVERSIBLE)).allowed
    decision = perms.check(AccessRequest(user, "file_delete", L.EXECUTE_CONSEQUENTIAL))
    assert not decision.allowed and decision.needs_approval


def test_automation_is_denied_not_asked(perms):
    bot = Actor("automation", "nightly", interactive=False)
    decision = perms.check(AccessRequest(bot, "file_write", L.EXECUTE_REVERSIBLE))
    assert not decision.allowed and not decision.needs_approval


def test_grants_are_scoped_time_limited_and_revocable(perms, clock, tmp_path):
    bot = Actor("task", "t1")
    g = perms.grant("task:t1", L.EXECUTE_CONSEQUENTIAL, tools=["process_*"], ttl_s=60)
    assert perms.check(AccessRequest(bot, "process_stop", L.EXECUTE_CONSEQUENTIAL)).allowed
    assert not perms.check(AccessRequest(bot, "file_delete", L.EXECUTE_CONSEQUENTIAL)).allowed
    clock.advance(61)
    assert not perms.check(AccessRequest(bot, "process_stop", L.EXECUTE_CONSEQUENTIAL)).allowed
    g2 = perms.grant("task:t1", L.EXECUTE_CONSEQUENTIAL, paths=[str(tmp_path / "logs")])
    assert perms.check(AccessRequest(bot, "file_delete", L.EXECUTE_CONSEQUENTIAL,
                                     paths=[str(tmp_path / "logs" / "a.log")])).allowed
    assert not perms.check(AccessRequest(bot, "file_delete", L.EXECUTE_CONSEQUENTIAL,
                                         paths=[str(tmp_path / "src" / "a.py")])).allowed
    assert perms.revoke(g2.id)
    assert not perms.check(AccessRequest(bot, "file_delete", L.EXECUTE_CONSEQUENTIAL,
                                         paths=[str(tmp_path / "logs" / "a.log")])).allowed
    assert g.id not in {x.id for x in perms.list_grants()}


def test_approval_creates_single_use_grant(db, perms):
    approvals = ApprovalManager(db, perms)
    req = approvals.request(tool="file_delete", args={"path": "x"}, summary="delete x", risk=RiskLevel.MEDIUM,
                            level=L.EXECUTE_CONSEQUENTIAL, requested_by="user:owner", task_id="t9", step_id="s1")
    again = approvals.request(tool="file_delete", args={"path": "x"}, summary="delete x", risk=RiskLevel.MEDIUM,
                              level=L.EXECUTE_CONSEQUENTIAL, requested_by="user:owner", task_id="t9", step_id="s1")
    assert again.id == req.id
    assert approvals.approve(req.id) is not None
    actor = Actor("task", "t9", interactive=True)
    request = AccessRequest(actor, "file_delete", L.EXECUTE_CONSEQUENTIAL, task_id="t9")
    assert perms.check(request).allowed
    assert not perms.check(request).allowed  # used up
    assert approvals.pending() == []


def test_instruction_hierarchy():
    user_stop = Directive(InstructionSource.USER, "pause", issued_at=100)
    automation_resume = Directive(InstructionSource.AUTOMATION, "resume", issued_at=200)
    assert not may_override(automation_resume, user_stop)       # stale autonomy never beats the user
    assert may_override(Directive(InstructionSource.USER, "resume", 150), user_stop)
    assert may_override(Directive(InstructionSource.SAFETY, "halt", 50), user_stop)
    assert resolve([user_stop, automation_resume]).action == "pause"


# -- shell classification ---------------------------------------------------------------

@pytest.mark.parametrize("command,level,blocked,network", [
    ("ls -la", L.OBSERVE, False, False),
    ("git status && git diff", L.OBSERVE, False, False),
    ("cat foo | grep bar | wc -l", L.OBSERVE, False, False),
    ("ls > /dev/null 2>&1", L.OBSERVE, False, False),
    ("pytest -q", L.EXECUTE_REVERSIBLE, False, False),
    ("echo hi > out.txt", L.EXECUTE_REVERSIBLE, False, False),
    ("rm build/app.o", L.EXECUTE_CONSEQUENTIAL, False, False),
    ("git push origin main", L.EXECUTE_CONSEQUENTIAL, False, True),
    ("curl https://example.com", L.EXECUTE_CONSEQUENTIAL, False, True),
    ("find . -name '*.pyc' -delete", L.EXECUTE_CONSEQUENTIAL, False, False),
    ("frobnicate --all", L.EXECUTE_CONSEQUENTIAL, False, False),
    ("echo $(whoami)", L.EXECUTE_CONSEQUENTIAL, False, False),
    ("rm -rf /", L.AUTONOMOUS, True, False),
    ("sudo rm -rf ~", L.AUTONOMOUS, True, False),
    ("dd if=/dev/zero of=/dev/sda", L.AUTONOMOUS, True, False),
])
def test_shell_classification(command, level, blocked, network):
    a = classify_command(command)
    assert (a.level, a.blocked, a.requires_network) == (level, blocked, network), a.reason


# -- registry enforcement ----------------------------------------------------------------

async def test_read_and_verified_write_with_rollback(registry, tmp_path, db):
    ctx = ctx_for(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("original")
    ex = await registry.execute("file_write", {"path": str(target), "content": "updated"}, ctx)
    assert ex.status == ExecStatus.EXECUTED and ex.ok and ex.verified
    assert target.read_text() == "updated"
    rb = await registry.rollback(ex.audit_id, ctx)
    assert rb.ok and target.read_text() == "original"
    read = await registry.execute("file_read", {"path": "notes.txt"}, ctx)
    assert read.result.data["content"] == "original"
    actions = [e.action for e in AuditLog(db).query()]
    assert actions[:3] == ["tool_execute", "rollback", "tool_execute"]


async def test_consequential_needs_approval_and_denied_outside_scope(registry, tmp_path):
    ctx = ctx_for(tmp_path)
    (tmp_path / "junk.tmp").write_text("x")
    ex = await registry.execute("file_delete", {"path": "junk.tmp"}, ctx)
    assert ex.status == ExecStatus.NEEDS_APPROVAL
    assert (tmp_path / "junk.tmp").exists()
    (tmp_path / "secrets").mkdir()
    ex = await registry.execute("file_read", {"path": str(tmp_path / "secrets" / "k")}, ctx)
    assert ex.status == ExecStatus.DENIED and ex.decision.basis == "scope"
    ex = await registry.execute("file_read", {"path": "/etc/hostname"}, ctx)
    assert ex.status == ExecStatus.DENIED


async def test_symlink_escape_is_blocked(registry, tmp_path):
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "key").write_text("s3cret")
    os.symlink(tmp_path / "secrets" / "key", tmp_path / "innocent.txt")
    ex = await registry.execute("file_read", {"path": "innocent.txt"}, ctx_for(tmp_path))
    assert ex.status == ExecStatus.DENIED


async def test_catastrophic_command_blocked_even_with_grant(registry, perms, tmp_path):
    perms.grant("user:owner", L.AUTONOMOUS, tools=["*"])
    ex = await registry.execute("shell_execute", {"command": "rm -rf /"}, ctx_for(tmp_path))
    assert ex.status == ExecStatus.DENIED and ex.decision.basis == "safety"


async def test_network_blocked_by_mode_policy(db, clock, perms, tmp_path):
    reg = ToolRegistry(perms, AuditLog(db, clock),
                       policy_provider=lambda: ExecutionPolicy(network_allowed=False,
                                                               network_block_reason="private mode is active"))
    register_builtin_tools(reg)
    perms.grant("user:owner", L.EXECUTE_CONSEQUENTIAL)
    ex = await reg.execute("shell_execute", {"command": "curl https://example.com"}, ctx_for(tmp_path))
    assert ex.status == ExecStatus.DENIED and "private mode" in ex.message


async def test_project_isolation(db, clock, perms, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (tmp_path / "other.txt").write_text("x")
    reg = ToolRegistry(perms, AuditLog(db, clock), policy_provider=lambda: ExecutionPolicy(
        allowed_tools=["file_*"], allowed_dirs=[str(proj)], project_id="p1"))
    register_builtin_tools(reg)
    ex = await reg.execute("file_read", {"path": str(tmp_path / "other.txt")}, ctx_for(proj))
    assert ex.status == ExecStatus.DENIED and "project" in ex.message
    ex = await reg.execute("shell_execute", {"command": "ls"}, ctx_for(proj))
    assert ex.status == ExecStatus.DENIED and "not allowed" in ex.message


async def test_dry_run_has_no_side_effects(registry, tmp_path):
    ctx = ctx_for(tmp_path, dry_run=True)
    ex = await registry.execute("file_write", {"path": "new.txt", "content": "x"}, ctx)
    assert ex.status == ExecStatus.DRY_RUN
    assert not (tmp_path / "new.txt").exists()


async def test_shell_captures_exit_status_and_timeout(registry, perms, tmp_path):
    ctx = ctx_for(tmp_path)
    ok = await registry.execute("shell_execute", {"command": "echo hello"}, ctx)
    assert ok.ok and ok.result.data["stdout"].strip() == "hello"
    fail = await registry.execute("shell_execute", {"command": "ls /definitely/missing"}, ctx)
    assert fail.status == ExecStatus.EXECUTED and not fail.ok and fail.result.exit_code != 0
    assert fail.verification.passed  # the failure itself was reported honestly
    perms.grant("user:owner", L.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    slow = await registry.execute("shell_execute", {"command": "sleep 5", "timeout_s": 0.3}, ctx)
    assert slow.status == ExecStatus.FAILED and slow.result.error == "timeout"


async def test_cancellation_stops_running_tool(registry, perms, tmp_path):
    perms.grant("user:owner", L.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    ctx = ctx_for(tmp_path)
    task = asyncio.create_task(registry.execute("shell_execute", {"command": "sleep 10"}, ctx))
    await asyncio.sleep(0.2)
    ctx.cancel.set()
    ex = await asyncio.wait_for(task, 3)
    assert ex.status == ExecStatus.CANCELLED


class LyingWriteTool(Tool):
    """Claims success without writing anything: verification must catch it (spec §52, scenario 14)."""

    spec = ToolSpec("lying_write", "claims to write", {"type": "object", "properties": {"path": {"type": "string"}},
                                                       "required": ["path"]},
                    level=L.EXECUTE_REVERSIBLE, path_params=("path",))

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(True, "file written successfully")

    async def verify(self, args, result, ctx) -> Verification:
        exists = os.path.exists(os.path.join(ctx.cwd, args["path"]))
        return Verification(True, exists, "exists_check", "present" if exists else "file is not on disk")


async def test_verification_catches_lying_tool(registry, tmp_path):
    registry.register(LyingWriteTool())
    ex = await registry.execute("lying_write", {"path": "claimed.txt"}, ctx_for(tmp_path))
    assert ex.status == ExecStatus.EXECUTED
    assert not ex.ok and not ex.verified
    assert "verification failed" in ex.describe()


async def test_invalid_and_unknown(registry, tmp_path):
    ctx = ctx_for(tmp_path)
    assert (await registry.execute("nope", {}, ctx)).status == ExecStatus.UNKNOWN_TOOL
    assert (await registry.execute("file_read", {"bogus": 1}, ctx)).status == ExecStatus.INVALID
    assert (await registry.execute("file_read", {}, ctx)).status == ExecStatus.INVALID
