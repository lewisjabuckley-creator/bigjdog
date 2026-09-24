"""The command line reaches the right handler for every subcommand (a clash between an option and the subcommand
dest once turned `jarvis schedule add ... --command ...` into an interactive session)."""

from __future__ import annotations

import pytest

import jarvis.cli as cli

HANDLERS = ["interactive_client", "client_ask", "cmd_status_any", "cmd_tasks_any", "cmd_task", "cmd_away",
            "cmd_notifications", "cmd_briefing", "cmd_schedule", "cmd_doctor_any", "cmd_models_any", "cmd_events_any",
            "cmd_approvals_any", "cmd_grants_any", "cmd_runtime"]


@pytest.mark.parametrize("argv, handler, check", [
    ([], "interactive_client", {}),
    (["ask", "what", "are", "you", "doing?"], "client_ask", {"text": ["what", "are", "you", "doing?"]}),
    (["status"], "cmd_status_any", {}),
    (["tasks", "--all"], "cmd_tasks_any", {"all": True}),
    (["task", "task-1", "cancel"], "cmd_task", {"id": "task-1", "action": "cancel"}),
    (["away"], "cmd_away", {}),
    (["notifications", "--ack"], "cmd_notifications", {"ack": True}),
    (["briefing", "--now"], "cmd_briefing", {"now": True}),
    (["schedule", "add", "hello", "--every", "1m", "--notify", "Hello"], "cmd_schedule",
     {"op": "add", "name": "hello", "every": "1m", "notify": "Hello"}),
    (["schedule", "add", "tests", "--daily", "02:00", "--task", "run tests", "--command", "python -m pytest"],
     "cmd_schedule", {"daily": "02:00", "task": "run tests", "shell_command": "python -m pytest"}),
    (["schedule", "list"], "cmd_schedule", {"op": "list"}),
    (["doctor", "--live"], "cmd_doctor_any", {"live": True}),
    (["models"], "cmd_models_any", {}),
    (["events", "--limit", "5"], "cmd_events_any", {"limit": 5}),
    (["approvals"], "cmd_approvals_any", {}),
    (["grants"], "cmd_grants_any", {}),
    (["runtime", "status"], "cmd_runtime", {"op": "status"}),
    (["--simulate", "runtime", "logs", "-n", "10", "-f"], "cmd_runtime", {"op": "logs", "lines": 10, "follow": True}),
])
def test_every_subcommand_reaches_its_handler(monkeypatch, argv, handler, check):
    called = []
    for name in HANDLERS:
        monkeypatch.setattr(cli, name, lambda args, _name=name: called.append((_name, args)) or 0)
    assert cli.main(argv) == 0
    assert [c[0] for c in called] == [handler]
    args = vars(called[0][1])
    for key, value in check.items():
        assert args[key] == value, key
