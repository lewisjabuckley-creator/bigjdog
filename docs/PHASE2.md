# Phase 2: persistent, always-on runtime

This file records what the repository looked like before Phase 2, the plan that followed from it, and what
was built. The plan extends existing components; it does not add a second runtime, API layer, event bus or
model client.

## What the inspection found (commit 4342594)

| Question | Answer |
|---|---|
| Entry point | `python -m jarvis` → `jarvis/cli.py:main` |
| What keeps JARVIS running | The interactive CLI process. `cli.interactive()` calls `Runtime.start()` and `Runtime.stop()`; closing the terminal stops workers, monitors and automations. |
| One-shot commands | `jarvis ask/status/tasks/...` each start a throwaway `Runtime` on the same database. Two runtimes on one database both schedule queued tasks (duplicate execution), and the second one's `recover_interrupted` pauses the first one's running tasks. |
| Tied to the UI process | Runtime lifecycle, the notification sink (`print`), the idle-drain loop, the `Orchestrator` (conversation state), simulation controls. |
| Memory-only state | Orchestrator history, pending question and focus; notification queue and dedupe index; router pins and per-model health; outage history; resource-throttled set; automation rate limits; recovery reports. |
| Scheduler | `AutomationEngine.tick()` ran only as an extra loop of `MonitoringService`, so it stopped when monitoring was disabled. Schedules: `every_s` and `daily_at` only; no catch-up or missed-run record; a crash between firing and saving `next_run` could fire twice. |
| Tasks | Durable (SQLite), state machine, checkpoints, `recover_interrupted`, pause/resume/cancel with the instruction hierarchy. Missing: original request, origin, artifacts, result, retry count, idempotency. A worker crash marked the task `FAILED` even when a step's outcome was unknown. |
| Events | `EventBus` plus `EventStore` with retention; no `SYSTEM_STOPPED`, recovery, notification or overall-health events. |
| Health | `HealthRegistry` per component; overall health was computed but `HEALTH_CHANGED` never emitted; no runtime, scheduler or model-layer entries. |
| Models | `ModelRouter` with fallbacks and provider health; no record of the active request or last failure; a task planned while no model was reachable failed instead of waiting. |
| Resources | `ResourceManager` thresholds hard-coded; JARVIS's own CPU and memory not measured. |
| API / IPC | None. |
| Already partly there | Durable tasks with recovery, persisted events, edge-triggered monitors, the automation engine, briefing text, "where were we" / "what changed" reports, the interrupt policy. |

## Plan (in build order)

1. **Single runtime per data directory.** An OS file lock in `Runtime.start()`, and a `runtime_runs` table
   that records each run (pid, heartbeat, clean stop) so an unclean stop is detected on the next start.
2. **Schema v2.** Task idempotency keys; conversation request records (so a reconnecting client never
   repeats a turn); schedule state columns; `runtime_runs`; stored briefings.
3. **Task record.** `request`, `origin`, `artifacts`, `result`, `retry_count` on `Task`, plus an API view with
   the current, completed and failed steps and the last error.
4. **Restart recovery.** Validate each interrupted task (age, working folder, dependencies, the automation that
   created it), decide resume, pause or block, and audit and publish the decision. A step that was running
   when the process died has an unknown outcome and is never re-run automatically unless it is safe to repeat.
   A worker crash leaves the task blocked with an unknown outcome instead of claiming failure.
5. **Durable scheduler.** Extend `AutomationEngine` with once, daily, weekly and interval schedules, a catch-up
   window for runs missed while JARVIS was down, and a compare-and-set on `next_run` in the same transaction
   as task creation (no duplicate runs). Its own loop in `Runtime`, independent of monitoring and the UI.
   Scheduled work keeps automation authority.
6. **Notifications.** Restore the queue on start; queue instead of "delivering" while no interface is
   attached; `NOTIFICATION_CREATED` and `NOTIFICATION_ACKNOWLEDGED` events.
7. **Model layer.** Active request, last failure and fallback tracked in `ModelRouter`; tasks wait for a model
   instead of failing, and resume when a provider recovers.
8. **Health and live state.** One runtime health report (runtime, database, task engine, workers, scheduler,
   event bus, monitoring, model layer, Ollama) with `HEALTH_CHANGED` on transitions; one live-state snapshot
   including JARVIS's own CPU and memory.
9. **Resource policy.** Configurable thresholds and an explicit policy for low-priority work under pressure
   (pause with checkpoint, slow, wait, continue).
10. **Local API.** Loopback HTTP with a bearer token, stdlib `asyncio` server, `httpx` client (already a
    dependency). The `Runtime` is the only owner of state.
11. **Daemon and CLI.** `jarvis runtime start|stop|restart|status|health|logs|run|install-service`; the
    interactive CLI becomes a client that starts the runtime if needed; one-shot commands use the API.
12. **Away tracking.** Presence from attached interfaces; "What happened while I was away?" answered only from
    tasks, events, notifications and state.
13. **Morning briefing data.** A structured briefing that the scheduler can produce and store; no HUD.
14. **Platform adapters.** `jarvis/platforms/{linux,macos,windows}.py` for detached start, stop, file locks and
    service definitions.
15. **Tests and docs.** In-process API tests, process-level daemon tests including `kill -9` recovery,
    scheduler, notification and security tests, and a live Ollama acceptance test with puppet models.

## What was built

All fifteen plan items, on the existing components:

| Plan item | Where |
|---|---|
| Single runtime, run records | `Runtime.start()` lock (`platforms/`), `runtime_runs`, `SYSTEM_RECOVERED`/`SYSTEM_STOPPED` |
| Schema v2 | `database/schema.py` migration 2 (in-place upgrade of v1 databases) |
| Task record | `Task.request/origin/artifacts/result/retry_count/idempotency_key`, `Task.to_api()`, `TaskManager.create_task(idempotency_key=...)` |
| Restart recovery | `TaskManager.recover_interrupted` (validation, decisions, `TASK_RECOVERED`, audit), `Step.outcome_unknown`, `WorkerPool._crashed` |
| Durable scheduler | `AutomationEngine` (once/daily/weekly/interval, slots, catch-up, missed), the runtime's scheduler loop |
| Notifications | `NotificationManager.restore`, presence-aware `decide`, `NOTIFICATION_CREATED/ACKNOWLEDGED`, `list`, `mark_delivered` |
| Model layer | `ModelRouter.status()` (active requests, last success, failure, fallback), `MODEL_REQUEST_FAILED`, `model_report` tool, tasks waiting for a model |
| Health and live state | `service/status.py`, `HealthRegistry` `HEALTH_CHANGED` on transitions, runtime self-metrics |
| Resource policy | `[resources]` config, `ResourceManager.low_priority_policy`, pause at a step boundary |
| Local API | `service/api.py`, `service/conversations.py` |
| Daemon and CLI | `service/daemon.py`, `service/client.py`, `cli.py` (`runtime ...`, client mode, `--embedded`, read-only one-shot commands) |
| Away tracking | `core/presence.py`, `core/awareness.py`, intent `AWAY` |
| Briefing data | `core/awareness.py` (`briefing_data`, `prepare_briefing`), `briefing` action, `[briefing]` config |
| Platform adapters | `jarvis/platforms/` (named `platforms`: a package called `platform` shadows Python's own module when the working directory is inside `jarvis/`) |
| Tests and docs | 48 new offline tests, 3 new live tests; RUNTIME.md, DEVELOPMENT.md, README, ARCHITECTURE, DECISIONS (D20-D24), ROADMAP |

Also added: "Analyze this project." as a deterministic intent (`ANALYZE_PROJECT`). It creates a durable task that
measures the project with the read-only `project_scan` tool and has the model write the analysis from those facts
(`model_report`), so the acceptance scenario rests on evidence, not on the model's guesses.

Two gaps found and fixed on the way:

- A grant to `automation:<id>` never reached the tasks that automation creates (they act as `task:<id>`).
  `Actor.delegated_by` now carries the automation's subject, so its grants apply to its own work and nothing else.
- One-shot CLI commands started a full throwaway runtime, which could start queued tasks and interrupt them a
  moment later. They now open the database read-only when no runtime is running.

## Deliberately not changed

- The Ollama adapter, the model abstraction, the scripted/simulated providers and the mock-model tests (Phase 1)
  are unchanged apart from request tracking in the router.
- The event bus, task manager, executor, worker pool, tool registry and permission model are extended, not
  replaced. There is still one `Runtime` class.
- The default resource policy stays `pause` (the tested Phase 1 behaviour). `wait` is available for work that
  should not be cut short.
- Planning a task from scratch without any model still fails honestly (an existing test asserts it). Only steps
  inside a plan that need a model wait for one.
- Not built, as instructed: HUD, voice, vision, email, phone, robotics, smart home, multi-user, 3D, project-graph
  analysis.
