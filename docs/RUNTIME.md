# The persistent runtime

JARVIS runs as a background process, the **runtime**, that owns all state and does all the work: conversation
turns, tasks, monitoring, the scheduler, notifications and the model layer. Interfaces such as the `jarvis` CLI are
clients of it. Closing a window never stops tasks, monitoring, schedules, events, notifications or durable state.

```text
  jarvis (CLI) ─┐                        ┌── Runtime (jarvis/runtime.py) ────────────────────────┐
  jarvis (CLI) ─┼── HTTP on 127.0.0.1 ───┤  tasks · workers · scheduler · events · notifications │
  future HUD  ──┘   bearer token          │  state · memory · permissions · models (Ollama)      │
                    (jarvis/service/api)  └── SQLite: ~/.jarvis/jarvis.db ───────────────────────┘
```

There is one `Runtime` class. The background process (`jarvis/service/daemon.py`) is that same runtime plus the
local API. `jarvis --embedded` runs the same runtime inside the CLI process, as JARVIS did before Phase 2; it stops
when you exit.

## Commands

| Command | What it does |
|---|---|
| `jarvis` | Interactive session. Starts the runtime in the background if it isn't running (`runtime.auto_start`). |
| `jarvis runtime start` | Start the runtime in the background. `--foreground` runs it in this terminal. |
| `jarvis runtime stop` | Stop it cleanly: running tasks are checkpointed first. Waits up to 45 s, then forces it. |
| `jarvis runtime restart` | Stop, then start. |
| `jarvis runtime kill` | Stop it instantly with no checkpoint, as a crash would. For trying out crash recovery. |
| `jarvis runtime status` | Running or not, pid, uptime, health, open tasks, model, attached interfaces, JARVIS's own memory and CPU. When it isn't running, how the last run ended (clean or unexpected). |
| `jarvis runtime health` | The full health check (see below). Exit code 1 if anything is critical or offline. |
| `jarvis runtime logs [-n 50] [-f] [--structured]` | The runtime's console log, or the structured JSON log. Works when the runtime is down. |
| `jarvis runtime run` | Run in the foreground; what a service manager (systemd, launchd) runs. |
| `jarvis runtime install-service [--write]` | Print (or write) a start-at-login definition for this OS. |
| `jarvis away` | What happened while you were away. |
| `jarvis task <id> [pause\|resume\|cancel]` | One task with its result, or control it. |
| `jarvis notifications [--ack] [--all]` | Recent notifications; `--ack` acknowledges them. |
| `jarvis briefing [--now]` | The latest morning briefing, or prepare one now. |
| `jarvis schedule list \| add \| enable \| disable \| remove` | Scheduled work (see below). |

Windows: use `py -m jarvis` wherever this page says `jarvis`, unless you installed the package with pip.

`jarvis status`, `tasks`, `events`, `models`, `approvals`, `grants`, `away`, `notifications`, `task` and `doctor`
use the running runtime. If none is running they open the database **read-only** (no workers, no scheduler, no
recovery, no run record), so a quick look never starts, interrupts or re-labels work. `jarvis ask` and the
interactive session start the runtime when needed.

## Files in the data directory (`~/.jarvis` by default)

| File | Purpose |
|---|---|
| `jarvis.db` | Everything durable: tasks, events, notifications, schedules, memory, audit, conversation, run records. |
| `runtime.lock` | OS file lock held by the running runtime. One runtime per data directory; the OS releases it if the process dies, so a crash never leaves a stale lock. |
| `runtime.pid` | The pid of the lock holder (informational). |
| `runtime.json` | pid, host and port of the running runtime's API. Removed on a clean stop. |
| `api.token` | The API bearer token, regenerated at every start; readable only by you (mode 0600 on Linux and macOS). |
| `logs/runtime.out` | The background process's console output. |
| `logs/jarvis-YYYYMMDD.jsonl` | Structured operational log (one JSON object per line). |

## What happens when...

**A notification is raised while a window is open.** Urgent and critical ones (a failure, an approval
request, a security or critical resource problem) appear at your prompt straight away as a `●` line. Important
ones (a task finished, a scheduled reminder, the briefing) appear under JARVIS's next answer, or at the prompt
once you've been idle for about 30 seconds. Informational ones are only recorded. Focus, quiet and presentation
modes hold back more. Repeats are merged ("failed 3 times in 10 minutes"), and at most 5 interruptions are
shown per 10 minutes. `jarvis notifications` lists them with their state: `delivered` (shown), `queued`
(waiting for you), `logged` (recorded only) or `acknowledged`. There are no desktop pop-ups yet.

**You close the interface.** Nothing stops. The runtime notes that no interface is attached. From then on,
notifications worth your attention are queued for your return instead of being "delivered" to nobody. When you
open `jarvis` again it tells you what finished and what is waiting for you. Ask "What happened while I was away?"
for the full account.

**A conversation turn is interrupted.** A turn runs inside the runtime, not the interface, so it finishes and its
answer is stored. Every turn carries a request id chosen by the interface. If the interface resends the same id (it
does so automatically after a dropped connection), the runtime returns the stored answer instead of handling the
message again, so nothing a message started is duplicated.

**The runtime is stopped cleanly** (`jarvis runtime stop`, SIGTERM, Ctrl+C in the foreground). Running tasks are
checkpointed and paused, the run is recorded as clean, and `runtime.json` and `api.token` are removed.

**The runtime is killed** (crash, `kill -9`, power loss). On the next start, JARVIS sees that the previous run has
no clean-stop record. It publishes `SYSTEM_RECOVERED`, records it in the audit log, and queues a notification.
Then each interrupted task is recovered in these steps:

1. Its checkpoint is restored; completed steps are never re-run.
2. It is validated: does the working folder still exist, are its dependencies intact, is the automation that
   created it still enabled, was it interrupted more than `runtime.recovery_max_age_s` ago?
3. The step that was running when the process died has an **unknown outcome**: it may or may not have taken
   effect. It is repeated automatically only if that is safe (an idempotent tool, or a call that only observes,
   such as `git status`). Otherwise the step is marked `outcome_unknown` and the task waits for you. JARVIS says
   so: "I can't tell whether it finished, and repeating it could apply it twice. Say 'continue' to resume."
4. The decision (resumed, paused or blocked) and the reason are written to the task, the audit log
   (`recovery_decision`) and the event log (`TASK_RECOVERED`).

**The executor itself fails mid-step** (a bug, not a tool failure). The task is blocked with outcome `unknown`, not
reported as failed or done, and the step is marked `outcome_unknown`.

**Ollama goes away.** The model layer reports degraded, and everything that doesn't need a model keeps working. A
task step that needs the model (for example writing an analysis) makes the task wait ("waiting for a language
model") instead of failing. It resumes by itself when the provider comes back. Planning a task from scratch without
any model still fails honestly, as it did before.

**Resources are under pressure.** Low-priority work (P3 and below) follows `resources.low_priority_policy`.
`pause` checkpoints it now, and the interrupted step starts again later. `wait` lets the step in flight finish
before pausing. `slow` keeps one low-priority task running. `continue` ignores pressure. Paused work keeps its
checkpoint and resumes by itself when pressure clears. JARVIS's own memory and CPU are measured every heartbeat,
and crossing `resources.self_memory_warn_mb` raises a warning.

## Health

`jarvis runtime health` (or `GET /v1/health`) checks the runtime, database, task engine, workers, scheduler, event
bus, monitoring, model layer and Ollama. Each part is derived from something observable: a database query, a loop's
last tick, a provider probe. A signal that can't be read is reported as `unknown`, never assumed healthy.
Monitoring and Ollama show `disabled` when they are turned off or not configured. Optional parts can only
*degrade* the overall state; the core parts can make it critical. Overall health changes publish one
`HEALTH_CHANGED` event per transition, never one per sample.

## Scheduler

Schedules are durable (SQLite) and run on the runtime's own loop, whether or not an interface is attached or
monitoring is enabled.

```bash
jarvis schedule add "nightly tests" --daily 02:00 --task "run the test suite" --command "python -m pytest -q"
jarvis schedule add standup --weekly mon,thu --daily 09:00 --notify "Stand-up in 5 minutes"
jarvis schedule add sync --every 2h --task "sync notes" --command "git -C ~/notes pull"
jarvis schedule add reminder --at 2026-10-01T09:00 --notify "Renew the domain"
jarvis schedule add morning --daily 07:30 --briefing
jarvis schedule list
```

Types: `once`, `daily`, `weekly` and `interval`. Each is a small JSON object, for example
`{"type": "weekly", "days": ["mon", "thu"], "at": "08:00"}`, so a natural-language front end can produce it later.

- **One run per slot.** Each due time has an idempotency key. A crash between starting a run and recording it
  cannot run that slot twice.
- **Catch-up.** A run missed while JARVIS was not running is made up once if it is within
  `scheduler.catch_up_window_s` (6 hours by default). Otherwise it is recorded as missed, with a
  `SCHEDULE_MISSED` event and a notification. Nothing is silently dropped. `catch_up: "skip"` never makes up
  runs.
- **Authority.** Scheduled work runs with automation authority (level 1), never the user's. A scheduled job gains
  nothing by being scheduled: anything above level 1 needs an explicit grant to that automation, and each tool call
  goes through the same registry pipeline as a typed command. A grant to one automation covers only the tasks that
  automation creates.

The morning briefing can also be switched on in configuration (`[briefing] enabled = true`), which keeps a system
schedule in line with `briefing.time` and `briefing.days`. The briefing is stored as structured data (tasks,
health, warnings, resources, model, project, schedule, queued notifications) plus text, and announced with
`BRIEFING_READY`.

## The local API

Loopback only (`runtime.api_host` must be `127.0.0.1`, `localhost` or `::1`). Every endpoint except `/v1/ping`
needs `Authorization: Bearer <contents of api.token>`. Requests carrying an `Origin` header (browsers) are refused,
so a web page can't drive JARVIS even though it listens on localhost. The API adds no privileges: an action it
starts goes through request → authorization (the token) → permission (per tool call) → execution (the tool
registry) → verification → audit, like a typed command.

| Endpoint | |
|---|---|
| `GET /v1/ping` | Liveness (no token). |
| `GET /v1/status` · `/v1/health` · `/v1/state` · `/v1/doctor` | Runtime status, the health check, the live-state snapshot (resources, battery, GPU, network, Ollama, active model and requests, tasks, workers, alerts, presence, schedule), diagnostics. |
| `GET /v1/tasks?status=open\|all\|<status>` · `GET /v1/tasks/{id}` | Tasks as records (request, goal, status, current/completed/failed steps, checkpoint, retries, dependencies, permissions, artifacts, result, error, timestamps). |
| `POST /v1/tasks` | `{"objective", "steps"?, "priority"?, "cwd"?, "idempotency_key"?}`; the same key returns the same task. |
| `POST /v1/tasks/{id}/pause\|resume\|cancel` | Control, with your authority. |
| `GET /v1/events?since=&types=&task_id=&min_severity=&limit=` | The event log. |
| `GET /v1/notifications?state=` · `POST /v1/notifications/ack` (`{"ids"?}`) · `POST /v1/notifications/{id}/ack` | Notifications and acknowledgement. |
| `POST /v1/conversation` | `{"text", "request_id", "session"?, "cwd"?, "stream"?}`; with `stream` the answer comes as NDJSON (`token` items, then one `response`). |
| `GET /v1/conversation/requests/{id}` | A turn's stored answer. |
| `POST /v1/sessions/attach` · `/v1/sessions/detach` · `GET /v1/stream?client_id=` | Presence, and the live notification stream (NDJSON). |
| `GET /v1/away` · `GET/POST /v1/briefing` | "What happened while I was away?" and the briefing. |
| `GET/POST /v1/schedules` · `POST /v1/schedules/{id}/enable\|disable` · `DELETE /v1/schedules/{id}` | The scheduler. |
| `GET /v1/approvals` · `/v1/grants` · `/v1/models` | Pending approvals, grants, model-layer status. |
| `POST /v1/runtime/stop` | Clean stop. |

## Events

Phase 2 uses the existing event bus and event types (`UPPER_SNAKE` names). Important ones are persisted:

| Requested | Event type |
|---|---|
| runtime.started / stopped / recovered | `SYSTEM_STARTED`, `SYSTEM_STOPPING`, `SYSTEM_STOPPED`, `SYSTEM_RECOVERED` |
| task.* | `TASK_CREATED`, `TASK_STARTED`, `TASK_WAITING`, `TASK_BLOCKED`, `TASK_PAUSED`, `TASK_RESUMED`, `TASK_COMPLETED`, `TASK_FAILED`, `TASK_CANCELLED`, `TASK_INTERRUPTED`, `TASK_RECOVERED` |
| model.* | `MODEL_LOADED`, `MODEL_UNLOADED`, `MODEL_UNAVAILABLE`, `MODEL_RECOVERED`, `MODEL_FALLBACK`, `MODEL_REQUEST_FAILED` |
| system.* | `HEALTH_CHANGED`, `SUBSYSTEM_DEGRADED`, `SUBSYSTEM_RECOVERED`, `RESOURCE_THRESHOLD_EXCEEDED`/`CLEARED`, `NETWORK_CHANGED` |
| notification.created / acknowledged | `NOTIFICATION_CREATED`, `NOTIFICATION_ACKNOWLEDGED` |
| interfaces, scheduler | `UI_ATTACHED`, `UI_DETACHED`, `AUTOMATION_TRIGGERED`, `SCHEDULE_MISSED`, `BRIEFING_READY` |

Events are operational: what happened, to what, why. They never contain the model's reasoning.

## Platform support

| OS | Status |
|---|---|
| Linux | Tested: background start (new session), `flock` lock, SIGTERM clean stop, `kill -9` recovery. The systemd user unit from `install-service` is generated but has not been exercised under systemd. |
| macOS | Not tested. Same process model as Linux; `install-service` writes a launchd agent. |
| Windows | Not fully tested on Windows. Written against the documented Win32 behaviour. The runtime runs under `python.exe` with a hidden console of its own (`CREATE_NO_WINDOW`), and every child process JARVIS starts (nvidia-smi, shell commands) is also started without a window, so no console windows flash up. It uses an `msvcrt` byte-range lock, and `install-service` writes a Startup-folder launcher. Clean stop goes through the API; the fallback is a hard stop, after which restart recovery treats in-flight steps as unknown. `api.token` relies on your user profile's permissions (no `chmod`). |

Adapters: `jarvis/platforms/linux.py`, `macos.py` and `windows.py`. The package is named `platforms` because a
package called `platform` would shadow Python's own `platform` module whenever the working directory is inside
`jarvis/`.

## Configuration

```toml
[runtime]
api_host = "127.0.0.1"        # loopback only
api_port = 0                  # 0 = any free port (written to runtime.json)
heartbeat_s = 10.0
presence_timeout_s = 30.0     # an interface silent this long counts as closed
auto_start = true             # the CLI starts the runtime when needed
recovery_max_age_s = 86400    # older interrupted work waits for you instead of resuming

[scheduler]
tick_s = 5.0
catch_up_window_s = 21600

[resources]
memory_critical = 92.0
cpu_critical = 97.0
vram_critical = 90.0
low_priority_policy = "pause" # pause | wait | slow | continue
self_memory_warn_mb = 1500

[briefing]
enabled = false
time = "07:30"
days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| "another JARVIS runtime is already running" | One is running for this data directory, maybe an `--embedded` session. `jarvis runtime status`. |
| "The JARVIS runtime (process N) is running but not answering" | It hung. `py -m jarvis` offers to restart it (tasks are recovered); or run `py -m jarvis runtime restart`. Since 0.3.0 the runtime writes the stacks of all its threads to `logs/stall-traces.log` in the data directory (`C:\Users\<you>\.jarvis\logs` on Windows) whenever it is blocked for two minutes; that file shows where it hung. |
| "the JARVIS runtime running now is version X…" | You started a new copy of JARVIS (a new version, or another folder) while the old runtime kept running. `py -m jarvis` offers to restart it on the new code; other commands print a note. `runtime.json` records each runtime's version and source folder. |
| The interface says it lost contact | The runtime stopped or restarted. Send a message: the CLI reconnects, starting the runtime if needed, and resends the same request id, so nothing is done twice. |
| `jarvis runtime start` says it exited during startup | The message includes the end of `logs/runtime.out`; the full log is there. |
| Windows: a console window flashes up every few seconds | Fixed after 0.2.0: an older runtime ran without any console, so each GPU check (`nvidia-smi`) opened a window. Update, then run `py -m jarvis runtime restart` so the old runtime is replaced. |
| I typed `py -m jarvis ...` into the chat | JARVIS explains that it's a terminal command and doesn't run it. It never runs its own control commands, because running `runtime stop` from inside would stop it mid-command. Type `/quit`, then run the command in Command Prompt. |
| Leftover tasks I don't want | In the chat: "cancel everything" (or "cancel the <name> task"); or `jarvis task <id> cancel`. |
| A task says "outcome unknown" | JARVIS was stopped while that step ran. Check whether its effect happened, then `continue` (run it again) or cancel the task. |
