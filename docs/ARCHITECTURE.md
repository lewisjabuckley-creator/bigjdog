# Architecture

JARVIS is built as a **deterministic core with an intelligent layer**. Critical state (tasks, permissions,
approvals, live resource state, memory, audit) lives in structured storage and deterministic services. Language
models interpret and reason over that state; they never hold it. If the model disappears, JARVIS keeps
monitoring, running tasks, enforcing permissions, answering status questions and recording events.

```text
                      USER
                        │  (CLI today; voice / HUD clients later)
                        ▼
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
- `recover_interrupted()` runs at startup. Monitors resume automatically. Other tasks resume automatically only
  if their policy allows it and the interrupted step is idempotent. Everything else is paused with a summary
  ("interrupted during 'validation'; completed: schema migration").

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
  and consequential ones become approval requests.

### Everything else
- `automation/`: schedules (`every_s`, `daily_at`) and `WHEN/IF/DO` rules. They run with automation authority,
  cannot trigger on their own work, and are rate-limited.
- `agents/`: bounded specialist agents with tool allowlists, a permission ceiling, budgets and a structured
  output contract. No recursive delegation, capped concurrency, and claims treated as evidence.
- `devices/`: the STATE/COMMAND/RESULT/TELEMETRY contract. Commands are verified from telemetry, never assumed.
- `security/`: `secret://` references resolved only inside tools; redaction of keys and known token formats in
  logs, audit records and events.
- `runtime.py`: startup (config, DB, events, state restore, task recovery, model detection, workers, monitors,
  subsystem verification, a concise report) and shutdown (checkpoint and mark interrupted work, stop monitors
  and workers, close resources).

## Data (SQLite, `database/schema.py`)

`events`, `state`, `entities`, `relations`, `tasks`, `approvals`, `grants`, `audit`, `memories` (+ `memories_fts`,
`embeddings`), `decisions` (+ `decisions_fts`), `projects`, `notifications`, `automations`, `conversation`,
`users`. WAL mode, one connection guarded by a re-entrant lock, versioned migrations.

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
