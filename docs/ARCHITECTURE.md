# Architecture

JARVIS is built as a **deterministic core with an intelligent layer**. Critical state (tasks, permissions,
approvals, live resource state, memory, audit) lives in structured storage and deterministic services. Language
models interpret and reason over that state; they never hold it. If the model disappears, JARVIS keeps
monitoring, running tasks, enforcing permissions, answering status questions and recording events.

```text
                      USER
                        │  (CLI today; voice / HUD clients later) — clients only, see "Process model"
                        ▼
                 Local API (loopback HTTP, token)  ── jarvis/service/
                        │
                ┌───────────────┐
                │ Orchestrator  │  intent → context → state → authority → decide → verify → respond
                └──────┬────────┘
         ┌─────────────┼──────────────┬──────────────┬─────────────┐
         ▼             ▼              ▼              ▼             ▼
   Intent engine   References    Reports (live)   Context       Model router ── Ollama / OpenAI-compatible
         │                                            │
         ▼                                            ▼
   ┌──────────────── Task manager (durable, state machine, checkpoints) ─────────────────┐
   │   Worker pool ── Executor: OBSERVE → PLAN → ACT → VERIFY → REPLAN                   │
   │        │             │            │                                                 │
   │   Resources      Planner      Tool registry ── permissions · scope · audit · verify │
   └──────────────────────────────────────────────────────────────────────────────────────┘
         │                       │
         ▼                       ▼
   Monitors ── events ──► Event bus ──► notifications · emergency · automations · health
   (system, network,         │
    models, self,            ▼
    watchers)          Event store (SQLite)
         │
         ▼
   Live state (facts with provenance, TTL) · World model (entities, relations) · Memory · Decisions
```

## Process model

JARVIS runs as one persistent background process per data directory, the **runtime**, and interfaces are its
clients (docs/RUNTIME.md). `jarvis/runtime.py` holds the single `Runtime` class. `jarvis/service/daemon.py`
hosts it with the local API (`service/api.py`, a stdlib-asyncio HTTP server on 127.0.0.1 with a bearer token).
The CLI uses `service/client.py`, which finds the runtime through `runtime.json` in the data directory and starts
it if needed. `jarvis --embedded` runs the same `Runtime` in-process.

- **Single instance.** `Runtime.start()` takes an OS file lock (`platforms/`). A second runtime on the same
  database, which would schedule the same queued tasks twice, refuses to start.
- **Run records.** `runtime_runs` stores each run's pid, heartbeat and clean-stop flag. A run without a clean stop
  means the process died: the next start publishes `SYSTEM_RECOVERED` and treats in-flight steps as having
  unknown outcomes.
- **Own loops.** Besides the worker pool, the runtime runs a scheduler loop, a heartbeat (run record, core
  health checks, JARVIS's own CPU and memory, presence expiry, resuming tasks that wait for a model) and
  maintenance (retention). None of them depends on monitoring being enabled or on an interface being attached.
- **Presence** (`core/presence.py`): attached interfaces, and when the user left (persisted). While nobody is
  attached, the notification policy queues news instead of "delivering" it to nobody.
- **Awareness** (`core/awareness.py`): "What happened while I was away?" and the morning briefing, assembled only
  from tasks, events, notifications and run records.
- **Conversation turns** (`service/conversations.py`) run inside the runtime with a client-chosen request id, so
  a closed window never cancels a turn and a retried request is never handled twice. Session history is restored
  from the conversation log.
- **Health and live state** (`service/status.py`): one health check across runtime, database, task engine,
  workers, scheduler, event bus, monitoring, model layer and Ollama, and one live-state snapshot.

## Runtime loops

The specification describes four loops (§181-184). Each maps to concrete code:

| Loop | Where | How |
|---|---|---|
| JARVIS loop (observe → understand → update world model → prioritise → plan → act → verify → update state → notify → wait) | `monitoring/service.py`, `tasks/workers.py`, `notifications/manager.py` | Monitors observe and write to `StateEngine` and `WorldModel`. Events flow through `EventBus`. The worker pool prioritises and admits work. Executors act and verify. The notification policy decides whether to speak. |
| Conversation loop | `core/orchestrator.py` | `Orchestrator.handle`: parse intent, resolve references, check live state, then answer, ask, plan, execute, delegate or monitor. Actions become tasks, so their outcomes are verified and durable. |
| Background loop | `tasks/executor.py` | Per step: authorise via the registry, execute, verify, checkpoint. On failure, `RecoveryPolicy` chooses retry, continue, replan, block or fail. |
| Emergency loop | `core/emergency.py` | A critical event enters emergency mode: snapshot state, preserve evidence, deprioritise P3+ work (via the resource manager), notify, keep monitoring, and exit only when the triggering condition clears. |

## Subsystems

### Events (`events/`)
`Event` is structured: type, source, severity, payload, entity and task ids, timestamp. `EventBus` delivers by
exact type, prefix (`TASK_*`) or wildcard, isolates subscriber failures and applies a per-handler timeout.
Important events are persisted in `EventStore`, with retention by severity. High-frequency events
(`SYSTEM_METRICS`, `TASK_PROGRESS`) are live-only by default.

### Live state and health (`state/`)
`StateEngine` stores `Fact`s: a value plus confidence (`observed`, `inferred`, `estimated`...), provenance
(`system state`, `tool output`, `memory`...), observation time and TTL, so JARVIS can tell when something has gone
stale and can answer "where did you get that?". `HealthRegistry` derives overall health from component reports
and records outages with durations (for "why didn't you answer?"). Heartbeats detect silent failures.

### World model (`world/`)
Entities (user, machine, project, folder, task, model, service, device...) and typed relations (`depends_on`,
`contains`, `runs_on`, `loaded_into`, `owns`...). It supports fuzzy name resolution for conversational
references and cycle-safe `chain()` traversal, which produces causal explanations such as
`api → postgres → connection-pool (used 100, max 100)`.

### Permissions (`permissions/`)
- `PermissionLevel` 0-5 (observe, recommend, prepare, reversible, consequential, autonomous).
- Baseline: actions serving an explicit user request are auto-approved up to `interactive_level` (≤ 3, enforced
  at config load). Automations and agents get `automation_level` (default 1).
- `Grant`s delegate authority by subject (`user:`, `task:`, `agent:`, `automation:`), tool globs, path prefixes,
  project, expiry and use count. They are inspectable and revocable.
- `ApprovalManager` turns "needs approval" into a persisted request. Approval creates a single-use, task-scoped
  grant, and the task resumes from its checkpoint.
- `PathPolicy` resolves symlinks, then applies allowed roots (including registered project folders), protected
  paths and the active project's directories.
- `hierarchy.py` encodes the instruction precedence of §26: safety > current user instruction > delegated
  authority > task policy > automation > default. The task manager uses it, so an automation cannot resume a
  task you paused.

### Tools (`tools/`)
`ToolSpec` declares parameters, permission level, risk, side effects, reversibility, idempotency, network
needs, timeout and verification method. `ToolRegistry.execute` is the only way anything causes an effect:

1. validate arguments (JSON-schema subset)
2. assess risk for these arguments (`Tool.assess`, e.g. shell command classification)
3. safety block, mode/network policy, project tool allowlist, path scope
4. permission decision (baseline → grants → needs approval / denied)
5. dry run returns a preview
6. run with timeout and cancellation
7. independent verification
8. audit record and event

Builtins: file read/list/search/write (backup-based rollback)/delete (to trash), shell (risk-classified,
process-group kill on timeout or cancel), process list/inspect/stop, system info, time. Internal tools let the
model start background tasks, search and write memory, and read live state. `device_command` and
`delegate_to_agent` are also registered.

### Tasks (`tasks/`)
The state machine is explicit (`models.TRANSITIONS`); invalid transitions raise.

```text
QUEUED → PLANNING → RUNNING → VERIFYING → COMPLETED
            │          │ ↘ WAITING (approval) ─┐
            │          │ ↘ BLOCKED (authority, dependency) ─┐
            ▼          ▼                        ▼            ▼
          FAILED    PAUSED ──resume──► QUEUED ◄──────────────┘
          CANCELLED / ABANDONED (terminal)          FAILED ──retry──► QUEUED
```

- `TaskManager` persists tasks (queryable columns plus a JSON blob), records history for every transition,
  emits events, and implements pause, resume, cancel, retry, reprioritise and plan edits. Control-plane fields
  have a single owner: a worker's stale in-memory copy cannot overwrite a pause or priority change, and plan
  edits to running tasks are queued on the `TaskController` and applied at the next step boundary.
- `TaskExecutor` runs plans and monitor tasks. Consequential steps put the task in `WAITING` (freeing the worker)
  until you approve.
- `WorkerPool` admits `QUEUED` work by priority, dependencies (a failed dependency blocks dependants), the
  concurrency limit, resource locks and pressure. It throttles P3+ under pressure, resumes it when pressure
  clears, and warns about deadlines at risk.
- `recover_interrupted()` runs at startup. Each interrupted task is validated (working folder, dependencies, the
  automation that created it, age). The step that was in flight when the process stopped has an **unknown
  outcome** and is repeated automatically only if it is safe to repeat (an idempotent tool, or a call whose
  assessed level is observe). Otherwise it is marked `outcome_unknown` and the task is paused for the user. The
  decision (resumed, paused or blocked) is written to the task, the audit log and a `TASK_RECOVERED` event.
  Monitors resume. A summary explains it ("interrupted during 'validation'; completed: schema migration; I
  haven't resumed it: ... can't tell whether it finished").
- A crash of the executor itself leaves the task `BLOCKED` with outcome `unknown`, never `FAILED` or `COMPLETED`.
- Task records carry the original request, origin (`conversation:<session>`, `api`, `schedule:<id>`...),
  artifacts, a final result, a retry count and an optional idempotency key (unique), and `Task.to_api()` exposes
  current, completed and failed steps, checkpoint, permissions and the last error.
- A step that needs a language model when none is reachable (`model_report`) puts the task in `WAITING`
  (`checkpoint.waiting_for = "model"`). It resumes when a provider recovers.
- Under resource pressure, P3+ work follows `resources.low_priority_policy`: pause, wait (finish the current
  step first), slow (one keeps running) or continue.

### Planning and verification (`planner/`, `verification/`)
Deterministic templates handle common intents (run tests, build) by probing the project for Python, Node, Rust,
Go and Make. Otherwise the planner asks a planning model for JSON steps and validates every step against the
registry (unknown tools and invalid arguments are rejected). Replanning after a failure is bounded by the task
policy. The `Verifier` checks the task's success condition (`all_steps_ok`, `tests`, `file_exists`, a read-only
`command`) and returns COMPLETE, PARTIAL, FAILED, BLOCKED or UNKNOWN.

### Models (`models/`)
`ModelProvider` is the only interface above the adapters. `ModelRouter` keeps an inventory with capabilities,
context length, size and loaded state. It selects by purpose (conversation, reasoning, planning, coding,
vision, summarization, classification, background, embedding), complexity, tools, vision, context and privacy,
then applies user pins and per-purpose preferences, and prefers loaded, small models for simple interactive
work. It tracks health per model and provider, falls back, re-probes providers when none is usable, and emits
`MODEL_UNAVAILABLE`, `MODEL_RECOVERED` and `MODEL_FALLBACK`.

**Talking to Ollama.** `OllamaProvider` requests a context window (`num_ctx`, default 8192) because Ollama's
default is small and it silently drops the start of an over-long prompt, which would lose the system prompt. It
merges configured sampling options, sends tool-call ids back with tool results, strips inline `<think>` traces,
and never routes a local Ollama through a system HTTP proxy. `models/readiness.py` turns the inventory into
plain guidance ("Ollama is running but no chat model is installed: `ollama pull llama3.1:8b`") for the startup
greeting and `jarvis doctor`. `jarvis/diagnostics.py` (`jarvis doctor --live`) runs the whole chain against the
user's own Ollama in a throwaway data directory.

### Memory (`memory/`)
`MemoryStore`: kinds (episodic, semantic, procedural, project, task, preference), FTS5 retrieval blended with
query-term coverage (BM25 alone is unstable on small corpora), optional embeddings (cosine), and a boost for the
active project, recency and importance. Session-scoped suppression backs "don't remember this", and forgetting
always needs an explicit scope. `DecisionLog` stores decision, context, alternatives, reason, involvement and
outcome, searchable by FTS.

### Monitoring (`monitoring/`)
`MetricsSource` (psutil, or `StaticMetrics` in simulation) feeds `SystemMonitor`. It writes state facts with a
TTL, updates the machine entity, and runs `ThresholdDetector` (sustain windows, hysteresis) and `TrendTracker`
(least-squares slope, R², time-to-threshold predictions labelled as estimates). `NetworkMonitor` classifies
online, high latency, limited, offline and unstable. `ModelMonitor` refreshes the inventory into state and the
world model. `SelfMonitor` checks the database, event bus and heartbeats. `MonitorChecker` powers monitor tasks
for task, process, path, command and metric targets, edge-triggered so it doesn't repeat itself.

### Notifications and modes (`notifications/`, `core/modes.py`)
Each mode is a policy: interrupt thresholds when idle or busy, whether queued items are delivered when you go
idle, humour, verbosity, network and cloud permission, and background level. Private mode, sensitive projects
and an offline network are overlays combined into one `EffectivePolicy`. Event rules turn events into useful
notifications ("Deployment failed. health check returned 503").

### Conversation (`core/`)
- `intent.py`: an ordered deterministic grammar covering the natural phrases in the spec, with dry-run
  prefixes, shell shortcuts and parameters. Anything unrecognised is `CHAT`.
- `references.py`: resolves "it", "that", "the build" and "the other one" against the conversation focus, then
  live tasks and projects. It returns candidates when a choice matters.
- `reports.py`: deterministic answers from live data (activity, status block, are we good, task status with
  estimates, re-entry, briefing, what changed, why, what did you do, provenance, diagnosis with observed facts,
  likely causes with confidence, and recommended actions).
- `context.py`: layered prompt (rules and style, live state labelled as observed, relevant memory with dates,
  decisions, preferences, recent turns) within a character budget.
- `orchestrator.py`: dispatch plus the model tool loop (bounded rounds). Long-running tool calls become tasks,
  and consequential ones become approval requests. The model sees only relevant tools: `device_command` only
  when devices exist, `delegate_to_agent` only for complex requests. Arguments it invents are dropped before
  validation. When the grammar matches but its target means nothing to the task system ("how's the weather?",
  "open the calculator"), the request goes to the model instead; if the model is unavailable or fails, the
  deterministic answer stands. Answers stream to the interface token by token (`handle(text, on_token=...)`),
  with reasoning traces and filler openers filtered out on the way.

### Everything else
- `automation/`: the durable scheduler (once, daily, weekly and interval schedules, with one run per slot through
  idempotency keys, catch-up within a window, and missed runs recorded and reported), `WHEN/IF/DO` rules and the
  `task`, `notify` and `briefing` actions. They run with automation authority. A grant to `automation:<id>`
  covers only the tasks that automation created (`Actor.delegated_by`). They cannot trigger on their own work and
  are rate-limited.
- `platforms/`: per-OS process start/stop, file locks and start-at-login definitions (Linux tested; macOS and
  Windows written but untested).
- `agents/`: bounded specialist agents with tool allowlists, a permission ceiling, budgets and a structured
  output contract. No recursive delegation, capped concurrency, and claims treated as evidence.
- `devices/`: the STATE/COMMAND/RESULT/TELEMETRY contract. Commands are verified from telemetry, never assumed.
- `security/`: `secret://` references resolved only inside tools; redaction of keys and known token formats in
  logs, audit records and events.
- `runtime.py`: startup (instance lock, config, DB, events, state restore, run record and unclean-stop detection,
  task recovery, model detection, workers, scheduler, heartbeat, monitors, subsystem verification, a concise
  report) and shutdown (checkpoint and mark interrupted work, stop loops, monitors and workers, record a clean
  stop, close resources, release the lock). A *passive* start (read-only CLI commands while no runtime is
  running) skips workers, scheduler, recovery and the run record.

## Data (SQLite, `database/schema.py`)

`events`, `state`, `entities`, `relations`, `tasks`, `approvals`, `grants`, `audit`, `memories` (+ `memories_fts`,
`embeddings`), `decisions` (+ `decisions_fts`), `projects`, `notifications`, `automations`, `conversation`,
`users`; since schema v2 also `runtime_runs`, `requests` (conversation turn idempotency), `briefings`, a unique
`tasks.idempotency_key` and schedule state columns on `automations`. WAL mode, one connection guarded by a
re-entrant lock, versioned migrations (an existing v1 database is upgraded in place).

## Extending JARVIS

- **A tool:** subclass `Tool`, declare a `ToolSpec` (level, risk, reversibility, `path_params`), implement
  `run`, and implement `verify` if anything can be checked. Register it in `register_builtin_tools` or at
  runtime. Consider `assess` if risk depends on the arguments.
- **A model runtime:** implement `ModelProvider` and add it in `Runtime._default_providers`.
- **A device:** implement `Device` (`state`, `telemetry`, `command` returning the expected state) and
  `DeviceRegistry.register` it.
- **An agent:** `AgentRegistry.register(AgentSpec(...))` with a tool allowlist and permission ceiling.
- **An automation:** `AutomationEngine.create(name, "rule" | "schedule", spec)`. New action types plug into
  `AutomationEngine.handlers`.
- **An intent:** add a `rule(...)` in `core/intent.py` and a handler in `Orchestrator.handlers`.

Before adding a capability, run through the feature design test in spec §194: input, context, state,
authority, plan, tools, resources, execution, verification, failure, recovery, memory, notification, UI and
audit.

## Testing layers

1. **Unit and subsystem tests** (`tests/test_*.py`): in-process, deterministic. Phase 2 adds
   `test_runtime_lifecycle.py` (lock, run records, recovery decisions, unknown outcomes, health, presence,
   notifications, model waiting, resource policies, briefing), `test_scheduler.py` and `test_api.py`.
2. **Process-level tests** (`tests/test_daemon_process.py`, POSIX): the real background runtime through the CLI:
   start, status, single instance, clean stop, closing the interface mid-task, `kill -9` mid-step, SIGTERM.
3. **Acceptance scenarios** (`tests/test_scenarios.py`): the full runtime with a simulated model, metrics and
   network, exercising the twenty scenarios in spec §200.
4. **Live integration** (`tests/integration/`, opt-in with `JARVIS_OLLAMA_TESTS=1`): a real Ollama server.
   `test_live_runtime.py` covers the persistent runtime against it: health and model state, a task waiting
   through an Ollama outage, and the Phase 2 acceptance scenario through real processes.
   *Puppet* models (`tests/integration/puppet.py`) are tiny GGUF files with hand-set weights: one-hot
   embeddings, zeroed attention and feed-forward, and a lookup-table output layer. They make the real server emit
   scripted replies, such as a specific `<tool_call>`, so the assertions are exact while the server does real
   template rendering, llama.cpp inference, tool-call parsing and streaming. `JARVIS_TEST_MODEL` adds
   capability checks with a real instruction model.
