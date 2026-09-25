# Architectural decisions

Short records of the decisions that shape the codebase: what was chosen, why, and what was considered instead.

### D1. Python 3.11 with asyncio
**Why:** the strongest ecosystem for local AI (Ollama clients, embeddings, speech and vision later), system
telemetry (psutil) and automation. asyncio covers what JARVIS is made of: many concurrent, mostly-waiting
activities such as monitors, tasks, model calls and subprocesses.
**Alternatives:** TypeScript/Node (good for a future UI, weaker for local ML and system integration), Rust (a
possible later home for hot paths, too slow to iterate for v0.1).

### D2. Two runtime dependencies: httpx and psutil
**Why:** httpx gives async HTTP with streaming and a mock transport, so the Ollama and OpenAI-compatible
adapters are fully tested without a server. psutil is the cross-platform standard for CPU, memory, disk,
battery, temperature and processes. Everything else is standard library: `sqlite3`, `tomllib`, `asyncio`,
`dataclasses`. **Rejected:** pydantic (dataclasses plus a small validator suffice), ORMs, vector databases and
agent frameworks (they would own the architecture).

### D3. SQLite as the single structured store, FTS5 for text, vectors as an optional side table
**Why:** local, zero-ops, transactional, durable across crashes. WAL mode, one connection, a re-entrant lock:
at personal scale, operations take well under a millisecond. Frequently queried attributes are columns;
evolving detail is JSON. **Rejected:** a vector database as the main store (the spec is explicit that
structured state belongs in structured storage). Brute-force cosine over stored vectors is fine up to tens of
thousands of memories; `sqlite-vec` is the planned upgrade.

### D4. Deterministic core, intelligent layer
**Why:** spec §135-139. Status, control, monitoring, thresholds, trends, permissions, recovery and task
templates never need a model. The model interprets, plans open-ended work and converses. This is what makes
scenario 15 (model offline, everything else still works) hold, and it keeps the system fast and cheap.

### D5. A deterministic intent grammar before the model
**Why:** control language has to be instant and must work without a model, and "stop" must never be
misinterpreted. The grammar covers the phrases in the spec; everything else goes to the model with tools.
**Trade-off:** phrasing outside the grammar goes to the model, which can still call the internal tools. A model
intent classifier for the ambiguous middle is on the roadmap.

### D6. One gate for effects: the tool registry
**Why:** "capability ≠ authority" is only enforceable if there is exactly one path to side effects. Models,
agents, automations, monitors and the orchestrator all call `ToolRegistry.execute`, so validation, risk
assessment, scope, permissions, dry run, timeout, cancellation, verification and audit cannot be skipped.

### D7. Permission baseline of 3 for explicit requests; 4+ only by approval or grant
**Why:** spec §144-145. "Debug this file" should not ask permission to read the file, but deleting,
publishing, stopping processes, installing and running unrecognised commands should always be authorised. Config
validation rejects `interactive_level > 3`. Automations and agents start at 1. **Approvals** are single-use,
task-scoped grants, so the authority granted is exactly the action approved.

### D8. Risk is assessed per invocation
**Why:** `shell_execute` is not one risk level: `git status` is observation, `pytest` is reversible local
execution, `rm` or `git push` is consequential, and `rm -rf /` is blocked even with a grant. Command
substitution cannot be analysed, so it is treated as consequential.

### D9. "Stop" pauses; "cancel" cancels
**Why:** scenarios 10 and 11 require "stop" followed by "continue" to resume from a checkpoint. "Cancel",
"abort", "kill" and "forget it" are terminal. Both take effect promptly: in-flight tools are cancelled and
subprocess groups are killed.

### D10. Tasks own durable state; a worker's copy never overwrites control
**Why:** a race showed up during testing: a worker checkpointing its in-memory task copy erased a
concurrent "skip the deploy step" edit. Control-plane fields (priority, control directive, deadline) are now
preserved on save unless changed deliberately, and plan edits to running tasks are queued on the controller and
applied at the next step boundary.

### D11. Monitors are tasks
**Why:** delegated responsibilities ("keep an eye on it") need the same properties as work: persistence,
recovery after restart, pause and cancel, stop conditions, expiry, audit and visibility in "what are you
doing?". They are exempt from the concurrency limit because they mostly sleep.

### D12. No model in watchers
**Why:** spec §135-136. Watchers and thresholds are deterministic and edge-triggered. The model is used only
where interpretation adds value, such as the optional interpretation of diagnostic evidence, clearly labelled
as inferred.

### D13. In-process event bus
**Why:** single-user, single-machine v0.1. Subscribers are decoupled by event type, failures are isolated and
important events are persisted. **Planned:** when UI and voice clients run as separate processes, a local API
will stream the same events.

### D14. Single-process runtime for now
**Why:** asyncio workers and subprocess-based tools keep conversation responsive during long work, and
durability comes from SQLite rather than process separation. **Trade-off:** tasks run only while the runtime is
running; they resume on the next start. A background daemon with a local API is the first roadmap item.

### D15. Strict configuration
**Why:** a silently ignored typo in `[permisions]` is a security bug. Unknown keys and wrong types fail at
startup with a clear message. Secrets are referenced by environment variable name only.

### D16. Honesty is architectural
**Why:** spec §36, §122 and §173-175. Tool results carry provenance, verification is recorded separately from
execution, partial outcomes stay partial, simulated components label themselves, and "What are you?" lists what
is *not* implemented.

### D17. Live integration tests use deterministic "puppet" models
**Why:** tests against a real LLM are slow, need multi-gigabyte downloads and are non-deterministic, so they
can't assert exact behaviour. A puppet is a tiny GGUF whose hand-set weights turn it into a lookup table: it emits
scripted replies (including tool calls in the format its template declares) through the real Ollama server. This
verifies the real contract (template rendering with tools, llama.cpp inference, Ollama's tool-call parsing,
streaming, lifecycle, error classes) exactly, with no download. Model *quality* is checked separately with
`JARVIS_TEST_MODEL` and `jarvis doctor --live`, which report a model that doesn't follow instructions as WARN,
not FAIL.

### D18. The grammar goes first; the model catches its misfires
**Why:** the deterministic grammar must stay instant and model-independent (D5), but patterns like "how's the X?"
or "open the X" also match ordinary conversation. When the matched target resolves to nothing JARVIS tracks and a
model is available, the request goes to the model; otherwise the deterministic answer stands. Control words
("stop", "continue", "proceed") never fall through while there is something they can act on.

### D19. Small-model hygiene is JARVIS's job
**Why:** local 3-8B models are the target, and they are less precise than cloud models. JARVIS therefore offers
only relevant tools, drops arguments the tool doesn't declare, requests an adequate context window, strips
reasoning traces, keeps each history message bounded, and feeds honest tool failures back so the model can
recover. The model is never given authority: every call still goes through the registry, permissions and
verification.

### D20. One runtime, many interfaces (Phase 2)
**Why:** work must continue when a window closes, and two runtimes on one database would schedule the same queued
task twice. The existing `Runtime` became the only owner of state, running as a background process with a local
API; interfaces are clients. There is still one `Runtime` class, reused in-process by `--embedded` and the tests.
An OS file lock (released by the OS if the process dies) enforces one runtime per data directory.

### D21. The local API is loopback HTTP on the standard library
**Why:** HTTP with JSON is easy to reach from any future interface (HUD, voice, scripts) and easy to debug, and
the stdlib `asyncio` server avoids a web-framework dependency for about twenty endpoints. It binds to loopback
only, authenticates with a per-start bearer token in a user-only file, and refuses browser requests (any
`Origin` header). It confers no authority: tool calls still go through the registry and permission manager.

### D22. Unknown is an outcome
**Why:** after a crash, JARVIS cannot know whether the step that was running took effect. Repeating a
non-idempotent step could apply it twice; marking it done or failed would be a guess. Such steps are marked
`outcome_unknown` and the task waits for the user unless the step is safe to repeat (idempotent, or assessed as
observation only). A crash of the executor is treated the same way.

### D23. Idempotency keys instead of distributed transactions
**Why:** "exactly once" for scheduled runs and conversation turns comes from keys, not from wrapping the action
and its bookkeeping in one transaction (which would hold SQLite's lock across awaits). A schedule slot's key
(`schedule:<id>:<slot>`) is unique in the tasks table, and a turn's request id is recorded before the turn runs,
so a crash between doing and recording returns the original result on retry.

### D24. Presence decides delivery; records decide the "away" answer
**Why:** a notification shown to nobody has not been delivered. The runtime tracks attached interfaces (an open
stream is presence) and queues news while nobody is there, persisting the queue across restarts. "What happened
while I was away?" is assembled from tasks, events, notifications and run records (including gaps when JARVIS
itself was not running), never from the model's recollection, so it cannot invent a result.
