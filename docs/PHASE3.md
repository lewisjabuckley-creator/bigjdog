# Phase 3: intelligence and autonomy

This file records the architecture map made before Phase 3, which capabilities already existed, the design that
followed, and what was built. The design adds one layer, `jarvis/intelligence/`, that coordinates the existing
systems. It does not replace them.

## Architecture map before Phase 3 (commit 7986be1)

```text
 interfaces (CLI) ──► local API ──► Runtime ─┬─ Orchestrator: intent grammar → handlers; CHAT → model tool loop
                                             ├─ Planner: templates + validated LLM steps (a linear list of Steps)
                                             ├─ TaskManager / WorkerPool / TaskExecutor
                                             │    durable tasks, state machine, checkpoints, dependencies between
                                             │    tasks, concurrency limit, resource admission and throttling,
                                             │    RecoveryPolicy (retry / replan / continue / block / fail)
                                             ├─ ToolRegistry: the only gate for effects (validate → risk → scope
                                             │    → permission → dry run → run → verify → audit → event)
                                             ├─ Permissions: levels, grants, approvals (single-use task grants)
                                             ├─ Agents: AgentSpec / AgentRunner / delegate_to_agent (bounded, no
                                             │    recursion, permission ceiling, output contract)
                                             ├─ ModelRouter: purpose, complexity, tools, context, pins, health,
                                             │    fallback, active-request tracking; Ollama provider
                                             ├─ Memory (FTS5 + embeddings), DecisionLog, projects, world model
                                             ├─ Verifier (success conditions), per-tool verify()
                                             ├─ EventBus + EventStore, AuditLog, notifications (attention policy),
                                             │    presence, awareness ("while you were away"), scheduler, health
                                             └─ ResourceManager (pressure, GPU pressure, low-priority policy)
```

| Phase 3 capability | Already there | Missing |
|---|---|---|
| Goal understanding / clarification | Intent grammar; reference resolution asks minimum questions | Structured goals, constraints, deadlines, success criteria, ambiguity classes |
| Plans | Linear `Task.plan` of `Step`s from templates or a model | A plan above tasks: DAG, parallel branches, conditions, loops, approval gates, lifecycle |
| Execution | Durable tasks, workers, concurrency, resources, dependencies between tasks | Turning a plan into tasks and following them |
| Replanning | Per-task replan after a failed step (model) | Plan-level triggers (failure, assumption, resources, user correction) and limits |
| Checkpointing | Tasks checkpoint after every step; restart recovery | Plan state (goal, graph, assumptions, decisions) persisted and recovered |
| Agents | Four agents, bounded runner, delegation tool | Contracts (input, output, timeout, fallback), more roles, a coordinator, conflict handling |
| Model routing | Purpose, complexity, pins, fallback | Priority, latency and context needs, resource-aware choice, an inference concurrency limit |
| Memory in planning | Memory in the chat context; DecisionLog | "We already tried that", recording plan decisions with evidence |
| Verification | Task success conditions, per-tool verify | Plan-level result quality (verified, partially verified, unverified, failed, conflicting), before/after measurements |
| Explanations | `why` from audit reasons; `what did you do` | Plan-aware "what are you doing" and plan history |
| Dry run | Registry dry run with preview; `dry run:` prefix | Simulation vs dry run vs prediction |
| Autonomy | Modes; permission baselines | Autonomy levels, event-driven reactions with loop protection |

## The design

One package, `jarvis/intelligence/`, coordinates the existing systems instead of replacing them. Its tasks are
the existing durable tasks, its approvals are the existing approvals, its agents run through the existing agent
runner, and every action still goes through the tool registry and the permission manager: request →
authorization → permission → execution → verification → audit, for plans exactly as for anything else.

```text
 "My computer is slow — find out why and fix it"
        │
        ▼  INTENT      goals.GoalParser ─► Goal: objective, outcome, constraints, deadline, priority, mode,
        │                                   complexity, ambiguity, success criteria, sub-goals
        │   simple ───────────────────────► existing direct paths ("check my CPU temperature": one tool call)
        │   ambiguous and consequential ──► one short question
        ▼  PLAN        builder.PlanBuilder ── playbooks (deterministic) │ compound mapping │ model-proposed graph
        │                    memory.PlanMemory: what was tried before │ routing │ project │ resources
        ▼              plans.Plan: a DAG of nodes (gather · analyze · decide · gate · action · agent · task ·
        │                          verify · report) with conditions, loops, estimates, assumptions
        ▼  EXECUTE     engine.PlanEngine ── ready nodes become ordinary TASKS (TaskManager, WorkerPool, TaskExecutor)
        │                 independent nodes in parallel · approval gates = the existing "proceed?" flow
        ▼  OBSERVE     task events → node states → facts (what each step produced)
        ▼  VERIFY      VERIFY nodes run as the verifier identity, re-measure and compare (never the executor's word)
        ▼  ADAPT       failures.classify → strategy (retry · alternative tool · wait · replan · ask)
        │              replanning.Replanner (verification failed · assumption broke · correction · resources)
        │              guard.LoopGuard limits every retry, replan, model call and added node
        ▼  COMPLETE    quality (verified · partially verified · unverified · failed · conflicting) → report →
                       decision outcomes and memory → notification
```

Complexity decides how much machinery a request gets:

| Complexity | Example | What happens |
|---|---|---|
| simple | "Check my CPU temperature" | One read-only tool call, answered directly. No plan, no task. |
| moderate | "Why is my laptop slow?", "back up X to Y" | A short plan (or the existing quick diagnosis for a plain question). |
| complex | "My computer is slow — find out why and fix it" | A structured plan: measure, analyze, decide, ask, act, verify, report. |
| very complex | research over many files, compound requests with a deadline | A structured plan with agents or milestones. |

## Module map

| Module | Responsibility |
|---|---|
| `goals.py` | `Goal`, `GoalParser` (deterministic): kind, execution mode, constraints, deadline, priority, compound clauses and conditions, ambiguity class, complexity, success and failure criteria. |
| `plans.py` | `Plan`, `PlanNode`, `Assumption`, node and plan lifecycles, conditions, loops, cycle detection, parallel waves, serialisation. |
| `store.py` | Goals, plans and plan revisions in SQLite (schema v3). |
| `playbooks.py` | Deterministic recipes: performance, disk cleanup, backup, research; compound requests mapped onto the intent grammar and templates. |
| `builder.py` | Picks the source, validates (structure, tools, arguments, constraints, safety, verifier-only-observes), inserts approval gates, decides on previews; validates model-proposed graphs. |
| `engine.py` | Runs plans on the task system; syncs node state from tasks; decisions; gates and fingerprint-bound grants; resources; assumptions; recovery; completion; events and audit. |
| `failures.py` | Failure categories and the strategies for each. |
| `guard.py` | Loop protection: attempts, replans, identical failures, plan size, model calls, loop iterations, resource waits, plan age, cycles. |
| `replanning.py` | Revising the remaining plan: next fix after a failed verification, broken assumptions, user corrections, resource shortage, model-assisted repair. |
| `analysis.py` | Deterministic analyzers (performance, disk, research) and comparators for independent verification. |
| `quality.py` | Node and plan quality from independent checks. |
| `tools.py` | `plan_gate`, `plan_analyze`, `plan_verify`, `disk_usage`, `file_copy`, `model_status`, `model_unload`. |
| `agents.py` | `AgentCoordinator`: when to use agents, packed context, research groups with deterministic fallbacks, collecting sourced claims. |
| `context.py` | `ContextPacker`: what a model needs, in priority order, within its context window. |
| `memory.py` | `PlanMemory`: earlier outcomes of the same actions ("we already tried that"), decision records with evidence and outcomes, episodic and procedural notes. |
| `routing.py` | `RoutingPolicy`: model choice by complexity, importance, latency, context, privacy and resource pressure. |
| `autonomy.py` | Autonomy levels. They change how much JARVIS does on its own initiative, never what is permitted. |
| `simulation.py` | Simulations and predictions, labelled as estimates with their basis. |
| `reactions.py` | Event-driven autonomy, proactive suggestions, the attention model. |
| `explain.py` | Reports, "what are you doing?", "why did you do that?", previews and history, from the plan record only. |
| `service.py` | `IntelligenceService`: wiring, and the one entry point for the conversation, API and CLI. |
| `core/plan_dialogue.py` | The conversation side of planning, kept out of the orchestrator. |
| `models/scheduler.py` | `InferenceScheduler`: concurrent model requests capped, queued by priority. |

## Goals

A request becomes a `Goal` before anything is planned:

| Field | From "urgent: my PC is slow, speed it up, but don't close Chrome by 5pm" |
|---|---|
| kind | `performance` (a playbook exists) |
| objective / outcome | "My PC is slow, speed it up" / "the computer is responsive again, with the cause identified" |
| constraints | `protect_process: Chrome` — every action is checked against it |
| deadline, priority | 17:00 today, `high` |
| mode | `execute` (others: `dry_run`, `simulate`, `predict`, `advise`) |
| complexity | `complex` |
| success criteria | the bottleneck is identified with evidence; after the fix, an independent measurement shows it eased |
| permissions | `changes` (so the plan will ask before changing anything) |

Compound requests become a dependency graph: "run the tests, and if they pass build it, then tell me" is three
parts, the second conditional on the first succeeding, the report after both. Parts joined by "and" run in
parallel; "then" makes them sequential.

**Ambiguity** is classified by what a wrong guess would cost. Harmless: proceed. Recoverable: proceed on a stated
assumption ("I'll look at the whole home folder and only recommend what to remove"). Consequential ("delete the
old files") and dangerous ("wipe everything"): one short question, and the plan waits for the answer.

## Plans

```text
 CREATED → VALIDATING → READY ──(preview confirmed / started)──► RUNNING ⇄ WAITING (approval, model, resources)
                                                                   │  ⇄ BLOCKED (needs you) ⇄ PAUSED (you, a restart)
                                                                   │  → REPLANNING → RUNNING
                                                                   └→ VERIFYING → COMPLETED | FAILED
                                                         (CANCELLED from any open state; FAILED → REPLANNING = "try again")
```

Node kinds: **gather** (observe), **analyze** (deterministic analysis of the evidence), **decide** (choose what to
change; recorded with its evidence; expands into action nodes at run time), **gate** (an approval point),
**action** (changes something), **agent** (a specialist), **task** (an objective the existing planner turns into
steps), **verify** (independent re-measurement), **report**. Nodes carry dependencies, optional conditions
(`{"node": "tests", "outcome": "success"}`, `{"fact": "decide.actions", "op": "nonempty"}`), bounded loops
(`{"until": ..., "max": 3, "restart_from": ...}`), alternative recipes, estimates, resources and rollback notes.

Everything is persisted after every change: the goal, the graph, which task carries which node, the facts each
node produced, assumptions, decisions, failures, approvals, corrections, milestones, limits used. A replan first
stores the previous graph in `plan_revisions`.

## The planner

1. **Playbooks** (no model needed). *Performance*: measure CPU, memory and the busiest processes (sampled over a
   window, since a single reading of a process's CPU is meaningless) and loaded models → analyze → decide → ask →
   act → verify → report. *Disk cleanup*: measure, find large old files in Downloads and temp only, decide, ask,
   move to JARVIS's trash (recoverable), verify. *Backup*: list, copy (identical files skipped, so repeating it is
   safe; partial copies never look complete), verify every file's size at the destination. *Research*: search,
   choose sources, read them in parallel (or hand groups to research agents when there are many and a model is
   available), extract claims with file and line, cross-check across sources, optionally have the model write a
   summary from the claims, verify every quote against its file, report.
2. **Compound requests**: each part is mapped onto what JARVIS already knows how to do (the intent grammar and
   the run-tests and build templates, the playbooks, a direct measurement), so the planner doesn't duplicate
   existing systems.
3. **Model-proposed graphs** for anything else, treated as untrusted proposals: unknown and internal tools and
   invalid arguments are rejected; a step the safety policy refuses is dropped; each node's kind follows from
   what its tool can do, never from the model's label; consequential steps always get a gate.
4. **A single objective** handed to the existing task planner, when the model can't produce a usable graph.

The builder is **memory-aware** (an action that didn't help last time is demoted and the plan says so),
**project-aware** (the project's root, test and build commands, its sensitivity and tool policy), and
**resource-aware** (estimates on nodes; heavy work waits under pressure).

## Approvals and permissions

The permission system stays authoritative; the planner never bypasses it.

- A **gate** is an ordinary consequential tool call (`plan_gate`). Before asking, the engine dry-runs every gated
  action through the registry: anything the safety policy or scope refuses is dropped with the reason; the rest
  become the *expected changes* shown in the question — PLAN → DRY RUN → EXPECTED CHANGES → APPROVAL → EXECUTE.
- When you approve, each approved action gets a **single-use grant scoped to its own task and tool**, bound to a
  fingerprint of the exact action you saw. If the action changed in between, it isn't covered and asks again.
- A standing grant you already gave (for example for shell commands) makes a gate unnecessary; the executor still
  checks it when the step runs.
- A **plan nobody asked for in the moment** (an event reaction, a schedule) runs with automation authority and
  can't be approved by anyone right then, so its decisions become recommendations; nothing gains authority by
  being automatic.
- **Autonomy never changes permissions.** No code path in the permission manager reads it.
- **Agents can't grant permissions**: the permission manager refuses any grant created by an agent identity.

## Verification and quality

The executor never certifies its own work. VERIFY nodes run as a separate identity (`agent:verifier`, automation
authority, observe-only tools) and compare fresh measurements with the ones taken before:

| Check | Verified when | Conflicting when |
|---|---|---|
| performance | the target processes are gone *and* the bottleneck eased | an action reported success but the process is still running |
| disk | every removed file is gone (it's in JARVIS's trash) | — |
| backup | every source file is at the destination with the same size | — |
| research | every claim is found word for word in the file it cites | sources disagree (reported, with both sides) |

Plan quality is one of **verified, partially verified, unverified, failed, conflicting** (or none, for a plan that
only observed). A plan is called done only if its latest independent check says the goal is met: a change that
was made but didn't resolve the problem ends as *not achieved*, and its report says so.

## Failures, replanning and loop protection

| Category | Examples | Strategy |
|---|---|---|
| transient | timeout, connection refused | retry after a delay, then another way, then replan |
| permission | not authorized; refused by the safety policy | ask (never retried; a safety refusal is final) |
| resource | out of memory, no model available | wait for it, then another way |
| tool | a tool crashed, command not found | another tool for the same information (e.g. `process_list` → `ps`/`tasklist`) |
| data | file missing, unreadable output | another way, then replan |
| planning | invalid arguments, unknown tool | replan |
| dependency | something it needed failed | skip what depended on it |
| verification | it ran, but the check says it didn't work | replan: the next most likely fix |
| unknown | includes "the outcome is unknown" after a crash | ask — never repeated blindly |

Replanning pauses the plan, keeps what is done and learned, works out which part is affected and rebuilds only
that part. Triggers: a failed verification, a broken assumption, a correction from you, a failed step with no
alternative (a model may propose a replacement, validated like any proposal), resource shortage.

Limits (all in `[intelligence]`): node attempts, plan revisions, identical failures (the same failure again stops
retrying), plan size, model calls per plan, loop iterations, how long work waits for resources, how long a plan
may stay open, and circular dependencies (rejected before a plan runs and after every revision). Reaching one
stops the plan with an explanation and a `LOOP_LIMIT_REACHED` event.

**Assumptions** are checked before the nodes they affect run and again after a plan resumes. "PID 4242 is still
running" failing means the stop is skipped ("it already exited"); "the source folder exists" failing blocks the
step with an explanation until it holds again.

## Agents

| Agent | Purpose | Tools | Ceiling | On failure |
|---|---|---|---|---|
| research | read sources, report findings with exact quotes | file_list, file_read, file_search, system_info, search_memory | observe | deterministic reading instead |
| testing | run and analyse tests | file_list, file_read, file_search, shell_execute | reversible | — |
| documentation | draft documentation | file_list, file_read, file_search, file_write | reversible | — |
| system | explain system behaviour | system_info, process_list, process_inspect, file_read | observe | — |
| analyst | interpret evidence already gathered | none | observe | skip |

Each has a declared contract (inputs, outputs, timeout, budgets, model preference). Agents are used only when that
helps: enough independent work, a model available, no resource pressure; otherwise the plan does the work
itself. An agent runs as an ordinary task through `delegate_to_agent`: its own identity, allowlist, ceiling,
budgets and audit trail; it can't delegate or grant itself anything. A failing agent affects only its node, which
falls back to the deterministic recipe. Its findings are evidence: they count only with a source, and every
quote is checked against that source.

## Model routing and resources

- `RoutingPolicy` picks a profile from complexity, importance, whether someone is waiting, context size and
  privacy. Under memory or GPU pressure, background requests move to a smaller or already-loaded model (a router
  hook, so it applies to agents, planning and summaries alike); what you asked for in the moment isn't degraded.
  The router's fallback behaviour is unchanged.
- `InferenceScheduler` caps concurrent model requests (`max_concurrent_inference`, one under pressure) and serves
  the queue by priority, so the conversation never waits behind background work.
- Heavy nodes of background plans wait while resources are short and start by themselves when they recover;
  light measurements are never deferred, so JARVIS can measure the problem it is under.
- Emergency and critical plans pause background and low-priority plans at their next step boundary and release
  them when done.

## Recovery after a restart

Task recovery runs first (Phase 2): each interrupted task is resumed if its in-flight step is safe to repeat,
otherwise paused with an unknown outcome. The plan engine then brings every open plan up to date with its tasks.
A plan whose step has an unknown outcome is paused with the reason ("I can't tell whether '...' finished, so I
haven't repeated it") until you say "continue". Completed nodes are never re-run.

## In the conversation

| You say | What happens |
|---|---|
| "Check my CPU temperature" | One measurement, answered directly; honest when the sensor isn't available. |
| "My computer is slow — find out why and fix it" | A plan. Within seconds, the evidence and one question: "I'd like to stop X (PID n) (can't be undone). Proceed?" |
| "yes" / "no" (also "sure, proceed", "ok go ahead", "yes please") | Approves or declines that change; after "yes", the verified result. A "yes" only starts a fix when the previous reply offered one. |
| "What are you doing?" | Plan status from the record: step n of m, what it's working on or waiting for. |
| "Why did you do that?" | The decision, the evidence, what was ruled out, who approved it and when, what verification found. |
| "What should I do about ...?" | Advice: it looks, recommends, changes nothing. Then "fix it" does the first recommendation, asking first. |
| "What would happen if I stopped Chrome?" | A simulation from current measurements, labelled as an estimate. |
| "How long will the tests take?" | A prediction from earlier runs, or "I have nothing to estimate from". |
| "dry run: free up disk space" | Observes for real, lists the changes it would make and which would need approval, changes nothing. The plan's title starts with "Dry run:". |
| "what are these files?" / "what did you find?" / "tell me more" | The full findings of the plan in focus, from the record (every candidate with its size and age), never a model's guess. After an investigation it offers the fix: "Say 'go ahead' and I'll move them to JARVIS's trash, asking you first." |
| "cancel all deletes", "cancel the free up disk space process", `cancel task "free up disk space"`, "pause free up disk space", "cancel that" | Stops the open plan it names, by title or by what it does ("deletes" means a disk clean-up; "all" means every match). Control phrases never start a plan and never go to the model. If nothing matches, the reply says so, changes nothing and lists what is open. A plan that already ended is named as such. |
| The same request twice | "I'm already on that", with the open plan's status or its pending question; no second plan. |
| "leave Chrome alone" / "no, back it up to E:\ instead" / "skip the summary" | A correction: the plan is revised in flight; completed work is kept. |
| "stop the backup" / "continue the backup" | Pauses and resumes the plan, also across restarts. |
| "show me the plan", "plan history", "what did you change in the plan?" | Preview, history and revisions. |
| "set autonomy to low / normal / high" | How much JARVIS does on its own initiative (never what's permitted). |

Honesty guard: if the model says it did something ("I've deleted…", "…has been cancelled") but no tool that changes
anything succeeded in that turn, the reply carries "Nothing was actually changed: I didn't run any action for that."
The model's context includes the open plans and the most recent finished plan's findings, so questions about them are
answered from the record. A file that disappears before approval is dropped from the question (and skipped if it
vanishes after approval); pausing a plan withdraws its approval question, which is asked again on resume.

## Event-driven autonomy and proactive help

Reactions to notable events (CPU or memory above threshold for a while, a disk filling up) depend on autonomy:
a suggestion with its reason (normal), an observe-only investigation whose findings are offered (high,
emergency), or nothing (low). Loop protection: a cooldown per kind, an hourly cap, never in response to its own
plans, never while a similar plan is open. The attention model holds suggestions back in focus or presentation
mode and backs off from kinds you keep ignoring.

## Observability

- **Events**: `GOAL_CREATED`, `GOAL_CLARIFICATION_NEEDED`, `PLAN_CREATED`, `PLAN_VALIDATED`, `PLAN_STARTED`,
  `PLAN_NODE_STARTED`, `PLAN_NODE_FINISHED`, `PLAN_WAITING`, `PLAN_BLOCKED`, `PLAN_PAUSED`, `PLAN_RESUMED`,
  `PLAN_REPLANNED`, `PLAN_COMPLETED`, `PLAN_FAILED`, `PLAN_CANCELLED`, `PLAN_RECOVERED`, `ASSUMPTION_INVALIDATED`,
  `DECISION_RECORDED`, `VERIFICATION_PERFORMED`, `AGENT_ASSIGNED`, `AGENT_FINISHED`, `LOOP_LIMIT_REACHED`,
  `CORRECTION_APPLIED`, `AUTONOMY_CHANGED`, `PROACTIVE_SUGGESTION`, and `DEADLINE_AT_RISK` for plans.
- **Audit** (`system:planner`): `plan_created`, `plan_decision`, `plan_verification`, `plan_retry`,
  `plan_alternative`, `plan_replanned`, `plan_deferred`, `assumption_invalidated`, `loop_limit`; plus every tool
  call of every node, agent and verification under its own identity.
- **Intelligence state** (`GET /v1/intelligence`): open and recent plans, what waits for you, running agents and
  their contracts, the inference queue, active model requests, autonomy, limits, resource pressure.
- **API**: `GET /v1/plans`, `GET /v1/plans/{id}` (with preview, status text, "why" and revisions),
  `POST /v1/plans/{id}/pause|resume|cancel|confirm`, `GET|POST /v1/goals`, `POST /v1/intelligence/autonomy`.
- **CLI**: `jarvis plans [--all]`, `jarvis plan <id> [pause|resume|cancel|confirm]`.

## Configuration (`[intelligence]` in jarvis.toml)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | true | Plan complex requests (false: everything goes through the Phase 2 paths). |
| `autonomy` | normal | low, normal or high. |
| `preview` | consequential | always, consequential or never: show a plan and wait for "go". |
| `max_parallel_nodes` | 3 | Steps of one plan running at once. |
| `max_nodes` | 40 | Steps in one plan, including ones added by revisions. |
| `max_node_attempts` | 3 | Times one step may be started. |
| `max_replans` | 3 | Revisions of one plan. |
| `max_identical_failures` | 2 | The same failure again stops retrying. |
| `retry_backoff_s` | 2.0 | First retry delay (doubles). |
| `max_model_calls` | 30 | Model calls per plan (planning and repair). |
| `max_plan_hours` | 12 | A plan still open after this waits for you. |
| `resource_wait_max_s` | 1800 | How long a step waits for resources before asking. |
| `max_concurrent_inference` | 2 | Model requests at once; the rest queue by priority. |
| `agents_max_concurrent` | 2 | Agents at once. |
| `reactions`, `proactive` | true | React to events; offer suggestions. |
| `reaction_cooldown_s`, `max_reactions_per_hour` | 1800, 4 | Loop protection for reactions. |
| `process_sample_s` | 1.5 | How long per-process CPU use is measured. |

## Extension points

- **A playbook**: a function `(goal, ctx, prefix="") -> Blueprint` in `playbooks.PLAYBOOKS`, plus a pattern in
  `goals._KINDS`. Gather → analyze → decide → verify → report keeps it adaptive and verifiable.
- **An analyzer or a comparator**: a pure function in `analysis.ANALYZERS` / `analysis.COMPARATORS`; use it through
  `plan_analyze` / `plan_verify` steps.
- **A failure strategy**: `failures.STRATEGIES`, and `PlanEngine._apply`.
- **An agent**: `AgentSpec` with its contract, registered in `AgentRegistry`; the coordinator decides when to use it.
- **A reaction**: a rule in `reactions._rule` returning a `Reaction`.
- **An assumption type**: `PlanEngine._evaluate_assumption`, and how to adapt in `Replanner._assumption`.

## Testing

| Suite | What it covers |
|---|---|
| `tests/test_intelligence_units.py` | Goal understanding, plan graphs and lifecycle, persistence, failure classification, loop limits, analyzers and comparators, context packing, the inference queue, routing under pressure, agent contracts, agents can't grant. |
| `tests/test_planning.py` | The engine on a real runtime with real processes and files: approve → act → verify; declining; a fix that verification rejects leads to the next fix; honest failure; a broken tool replaced by another; identical failures stop; cycles rejected; parallel branches within the limit; conditional steps; broken assumptions; resource waits; emergency pre-emption; restart (automatic resume and unknown outcome); research agents, their isolation and claim verification; malicious file content; dry runs; previews; corrections; recommendations-only plans; reactions and cooldowns; fingerprint-bound approvals; memory of what didn't help; simulation and prediction. |
| `tests/test_phase3_scenarios.py` | The ten definition-of-done scenarios through the conversation, and the plan and goal API. |
| `tests/integration/test_live_planning.py` | Against a real Ollama server with puppet models: a model-proposed plan whose destructive step never runs; research agents and a model-written summary with claims verified; advice that never offers the model a tool that could change anything. |

The ten scenarios: (1) a simple question answered directly without planning; (2) a complex slowness investigation
and fix; (3) recovery from a broken tool; (4) a correction in flight; (5) background work yielding to resource
pressure, and smaller models under pressure; (6) nothing called done until verified; (7) a restart in the middle,
then "continue the backup"; (8) a multi-agent research task; (9) "why did you do that?" from the record;
(10) "what should I do?" advises without acting, then "fix it".

## Limits and what was deliberately not built

- Not built (as required): voice, camera or screen vision, HUD, 3D, smart home, robotics, phone, fabrication.
- Research reads local files and JARVIS's memory; there is no web search tool.
- Performance fixes are limited to stopping a process (with approval) or unloading an idle model; JARVIS doesn't
  change system settings, drivers or startup programs.
- Disk cleanup only proposes large, old files in Downloads and the temp folder, and moves them to JARVIS's trash
  (the space returns when the trash is emptied).
- Conflict detection between sources is a heuristic (the same statement with different numbers); it flags, it
  doesn't decide who is right.
- Goal understanding is English and pattern-based; anything it doesn't recognise goes to the conversation model
  as before.

## Trying it

The same way as Phase 2 (`py -m jarvis` on Windows). Things to type:

1. `Check my CPU temperature` — an instant answer (on Windows it will honestly say the sensor isn't available).
2. `My computer is slow, find out why and fix it` — it measures, then asks before changing anything. Say `no` the
   first time if you'd rather not stop anything; `What should I do about my computer being slow?` gives advice
   without acting.
3. `Why did you do that?` and `What are you doing?` at any point.
4. `Back up C:\Users\<you>\Documents\SomeFolder to D:\Backup` (any two folders), then `stop the backup` while it runs,
   close JARVIS, open it again, and say `Continue the backup`.
5. `Research what my notes say about <topic> in C:\Users\<you>\Documents\Notes`.
6. `dry run: free up disk space` — see what it would remove, with nothing removed.
7. `jarvis plans --all` in a terminal lists every plan; `jarvis plan <id>` shows one, with its steps, result and why.
