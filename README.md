# JARVIS

A local-first, privacy-first personal AI operating environment.

JARVIS is not a chatbot with a persona. It is an orchestration layer that combines live system state, a world
model, memory, durable background tasks, tools, monitoring, permissions and model routing, with language models
as one component among many. You give it intent; it plans, executes through permission-checked tools, verifies
the result against reality, remembers what matters, keeps watching, and can always explain what it is doing
and why.

> **Status: v0.1, foundation.** The deterministic core, task engine, permission system, memory, monitoring and
> conversational layer are implemented and tested: 195 tests, including end-to-end tests for all twenty
> "final test" scenarios in the specification. Voice, vision, a graphical HUD, communications and physical
> device integrations are **not implemented yet**; their interfaces and the plan for them are in
> [docs/ROADMAP.md](docs/ROADMAP.md).

## What it does today

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

# with a local Ollama (https://ollama.com): ollama serve && ollama pull llama3.1:8b
jarvis                            # interactive session
jarvis doctor                     # self-diagnostics
jarvis status                     # compact system status

# without any model or real monitoring: simulated metrics, network and model
jarvis --simulate
```

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

In the interactive session, `/tasks`, `/events`, `/approvals`, `/grants`, `/revoke <id>`, `/health`, `/debug`
and (in simulation) `/sim cpu 95`, `/sim model off` and `/sim network off` expose operational detail. Say `help`
for the language JARVIS understands without a model.

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
  runtime.py     wiring, startup and shutdown sequences
  cli.py         command-line interface
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/DECISIONS.md](docs/DECISIONS.md) ·
[docs/ROADMAP.md](docs/ROADMAP.md)

## Tests

```bash
python -m pytest            # about 10 seconds; no network, no model, no GPU required
```

`tests/test_scenarios.py` holds the acceptance suite. It drives the real runtime (SQLite, event bus, worker pool,
subprocesses, filesystem) with a simulated model, metrics and network through the twenty scenarios in the
specification: long-running work alongside conversation, unprompted failure detection, evidence-based diagnosis,
resource adaptation, dependency loss, plan changes, restart recovery, "what are you doing?", "why did you do
that?", stop and continue, delegated monitoring, authorization, lying tools, model outage, live-state answers,
memory, decision history, sensitive projects and attention.
