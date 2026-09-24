# JARVIS

A local-first, privacy-first personal AI operating environment.

JARVIS is not a chatbot with a persona. It is an orchestration layer that combines live system state, a world
model, memory, durable background tasks, tools, monitoring, permissions and model routing, with language models
as one component among many. You give it intent; it plans, executes through permission-checked tools, verifies
the result against reality, remembers what matters, keeps watching, and can always explain what it is doing
and why.

> **Status: v0.2, a persistent runtime.** The deterministic core, task engine, permission system, memory,
> monitoring and conversational layer are implemented and tested. JARVIS talks through a real Ollama model, and
> since Phase 2 it runs as a background runtime that interfaces connect to: closing the window doesn't stop
> tasks, monitoring, schedules or notifications, and "What happened while I was away?" answers from the record.
> The suite has 257 offline tests (including the twenty "final test" scenarios from the specification and
> process-level tests of the background runtime) plus 21 opt-in live tests against a real Ollama server. The
> runtime is tested on Linux; the macOS and Windows adapters are written but untested. Voice, vision, a
> graphical HUD, communications and physical device integrations are **not implemented yet**; see
> [docs/ROADMAP.md](docs/ROADMAP.md).

## What it does today

- **Keeps running when you close the window.** One background runtime per data directory owns all state and
  work; the CLI is a client of its local, token-authenticated API. Conversation turns, tasks, monitors, schedules
  and notifications carry on without an interface. When you come back, it tells you what finished and what needs
  you. "What happened while I was away?" gives the actual results, from tasks, events and notifications. See
  [docs/RUNTIME.md](docs/RUNTIME.md).
- **Recovers honestly after a crash.** An unclean stop is detected on the next start. Interrupted tasks are
  validated, and a step that was running when the process died is treated as having an *unknown* outcome: it is
  repeated automatically only if that is safe, otherwise JARVIS asks. Every decision is audited.
- **Schedules work durably.** Once, daily, weekly and interval schedules with one run per slot, catch-up for runs
  missed while JARVIS was down (or an honest "missed" report), and automation authority only: nothing gains
  privileges by being scheduled.
- **Understands control and status language without a model.** "What are you doing?", "stop", "continue",
  "keep an eye on it", "where were we?", "why did you do that?", "what changed?", "focus mode" and "use the
  local model" are handled by a deterministic intent engine, from live state. They keep working when no model
  is available.
- **Runs durable background work.** Tasks are persisted in SQLite with an explicit state machine, are
  checkpointed after every step, survive restarts (without silently re-running destructive steps), can be
  paused, resumed, cancelled, reprioritised and edited mid-flight, and respect dependencies, concurrency limits
  and resource pressure.
- **Keeps capability and authority separate.** Every effect goes through one tool registry that evaluates risk
  for the specific arguments (so `ls` is observation and `rm -rf /` is blocked outright), enforces filesystem
  scope and project isolation, checks permission levels and scoped, time-limited, revocable grants, and pauses
  for your approval before anything consequential. Everything is written to an audit log.
- **Verifies instead of trusting.** File writes are read back, test runs are parsed (pytest, jest, cargo, go,
  unittest), device commands are checked against telemetry, agent claims are treated as evidence, and "the
  command completed" is kept distinct from "the goal was achieved".
- **Watches the environment.** System, network, model and self-health monitors write facts (with provenance and
  staleness) into live state and raise events. Sustained thresholds, trends and labelled predictions come from
  deterministic analysis, not the model. Critical conditions trigger an emergency mode that preserves evidence
  and exits when the condition clears.
- **Knows when not to interrupt.** Notifications have priorities, expiry, dedupe with escalation ("failed 3 times
  in 10 minutes") and rate limits. What reaches you depends on mode (focus, presentation, quiet, emergency) and on
  whether you are busy.
- **Remembers deliberately.** Episodic, semantic, procedural, project and preference memory with FTS5 retrieval
  and optional embeddings, a decision history ("why did we choose PostgreSQL?"), and "remember that",
  "forget that" and "don't remember this".
- **Routes models by requirement.** An Ollama adapter (discovery, capabilities, tool calling, images, streaming,
  embeddings, load and unload) and an optional OpenAI-compatible adapter sit behind a router that selects by
  purpose, complexity, tools, vision, context, privacy and health. It falls back across models and providers and
  tells you when a fallback changed behaviour.

## Quick start

```bash
pip install -e ".[dev]"          # Python 3.11+; runtime dependencies: httpx, psutil
```

**With a real model** (recommended). Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull llama3.1:8b          # or llama3.2:3b on a computer without a graphics card
jarvis                            # interactive session; starts the background runtime (Windows: py -m jarvis)
jarvis doctor --live              # verify the whole chain against your Ollama
```

Step-by-step setup, model choice and troubleshooting: [docs/OLLAMA.md](docs/OLLAMA.md).

**Without any model**, with simulated metrics, network and model (its own data directory and runtime):

```bash
jarvis --simulate
```

**The runtime.** `jarvis` starts it when needed and leaves it running when you exit.

```bash
jarvis runtime start | stop | restart | status | health | logs [-f]
jarvis runtime install-service    # start at login (systemd / launchd / Windows Startup folder)
jarvis --embedded                 # run it inside this window instead (stops when you exit)
```

Other commands: `jarvis ask "..."`, `jarvis status`, `jarvis tasks`, `jarvis task <id>`, `jarvis away`,
`jarvis notifications`, `jarvis briefing`, `jarvis schedule ...`, `jarvis doctor`. Details:
[docs/RUNTIME.md](docs/RUNTIME.md).

Example session (simulation mode, in this repository):

```text
you › open the project ~/code/bigjdog
jarvis › Opened bigjdog (branch main, python, tests: `python3 -m pytest -q`).
you › run the tests
jarvis › Running the bigjdog test suite (`python3 -m pytest -q`). I'll let you know how it goes.
you › keep an eye on it
jarvis › I'll keep an eye on bigjdog tests and tell you if anything needs attention.
you › $ rm -r build
jarvis › `rm -r build` needs your approval (shell_execute requires execute consequential authorization). Proceed?
you › proceed
jarvis › Proceeding: $ rm -r build (in /home/me/code/bigjdog).
you › what are you doing?
jarvis › I'm working on bigjdog tests. I'm monitoring bigjdog tests. Nothing is blocked.
```

Closing the window and coming back later:

```text
you › Analyze this project.
jarvis › Analyzing bigjdog in the background (task task-1c30…). It keeps running if you close this window; I'll
         tell you when the analysis is ready.
(window closed … reopened)
JARVIS 0.2.0 — connected to the runtime (pid 3792). Talking through llama3.1:8b (local, tools enabled).
While you were away: Analyze bigjdog completed.
you › What happened while I was away?
jarvis › While you were away (15:10–15:41, 31 minutes):
         JARVIS kept running the whole time.
         Finished:
         • Analyze bigjdog — completed at 15:12.
             <the analysis the model wrote from the measured project facts>
```

In the interactive session, `/tasks`, `/task <id>`, `/events`, `/approvals`, `/grants`, `/health`, `/status`,
`/away`, `/state` and (in simulation) `/sim cpu 95`, `/sim model off` and `/sim network off` expose operational
detail. Say `help` for the language JARVIS understands without a model.

## Configuration

Everything important is configurable: models and per-purpose preferences, permission levels, allowed and
protected paths, thresholds, notification limits, memory, privacy and style. See
[config/jarvis.example.toml](config/jarvis.example.toml). Secrets never go in configuration: adapters reference
environment variable names, and tools resolve `secret://NAME` references only at execution time.

## Safety model in one paragraph

Levels 0-3 (observe, recommend, prepare, reversible execution) are auto-approved only for actions that directly
serve an explicit request from you. Level 4 (consequential) always needs your approval or a scoped grant. Level 5
(autonomous) exists only as explicit, bounded delegation. Automations and agents start at level 1. A short list
of catastrophic commands is blocked even with a grant. Protected paths (`~/.ssh`, cloud credentials and so on)
and active project boundaries are enforced after symlink resolution. Private mode and sensitive projects keep
everything local. An explicit current instruction from you always outranks a stale automation.

## Layout

```text
jarvis/
  core/          intent engine, references, orchestrator, reports, context, modes, emergency, personality
  tasks/         task model and state machine, manager, executor, worker pool, recovery, resources
  tools/         tool contract, registry (the single gate for effects), builtin and internal tools
  permissions/   levels, grants, approvals, path scope, instruction hierarchy
  models/        provider abstraction, Ollama, OpenAI-compatible, router, scripted provider
  memory/        memory store (FTS5 + embeddings), decision history
  monitoring/    metric sources, system/network/model/self monitors, thresholds and trends, watchers
  events/        event types, async bus, persistent store with retention
  state/         live state with provenance and staleness, health registry
  world/         entities and relations
  planner/       deterministic templates, validated LLM planning
  verification/  task success conditions, test-output parsing
  notifications/ interrupt policy, dedupe, delivery
  projects/      projects and isolation policy
  automation/    schedules and WHEN/IF/DO rules
  agents/        bounded, supervised specialist agents
  devices/       STATE/COMMAND/RESULT/TELEMETRY hardware abstraction
  audit/         action log
  security/      secret references and redaction
  simulation/    simulated model, metrics and network
  service/       background runtime process, local API, conversation turns, health and live state, client
  platforms/     per-OS process start/stop, locks, start-at-login (linux, macos, windows)
  runtime.py     wiring, startup and shutdown sequences (the one Runtime)
  cli.py         command-line interface (a client of the runtime)
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/RUNTIME.md](docs/RUNTIME.md) ·
[docs/DECISIONS.md](docs/DECISIONS.md) · [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## Tests

```bash
python -m pytest            # about 30 seconds; no network, no model, no GPU required
```

`tests/test_runtime_lifecycle.py`, `tests/test_scheduler.py` and `tests/test_api.py` cover the persistent runtime
in-process. `tests/test_daemon_process.py` starts real background runtimes through the CLI (start, status, single
instance, clean stop, closing the interface mid-task, `kill -9` mid-step, SIGTERM). Full testing guide:
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

`tests/test_scenarios.py` holds the acceptance suite. It drives the real runtime (SQLite, event bus, worker pool,
subprocesses, filesystem) with a simulated model, metrics and network through the twenty scenarios in the
specification: long-running work alongside conversation, unprompted failure detection, evidence-based diagnosis,
resource adaptation, dependency loss, plan changes, restart recovery, "what are you doing?", "why did you do
that?", stop and continue, delegated monitoring, authorization, lying tools, model outage, live-state answers,
memory, decision history, sensitive projects and attention.

`tests/integration/` runs against a real Ollama server and is opt-in:

```bash
pip install -e ".[integration]"
JARVIS_OLLAMA_TESTS=1 python -m pytest tests/integration                       # any running Ollama
JARVIS_OLLAMA_TESTS=1 JARVIS_TEST_MODEL=llama3.1:8b python -m pytest tests/integration   # plus a real model
```

The plumbing tests build tiny deterministic "puppet" models on the fly, so they need no model download. See
[docs/OLLAMA.md](docs/OLLAMA.md#for-developers-live-integration-tests). `tests/integration/test_live_runtime.py`
runs the Phase 2 acceptance scenario against the real server through real processes.
