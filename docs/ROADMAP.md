# Roadmap

Status of each build phase from the specification (§165), and what comes next. "Done" means implemented and
covered by tests. "Partial" means the architecture and a working subset exist.

| Phase | Status | Notes |
|---|---|---|
| 1. Foundation (Ollama adapter, model abstraction, conversation, config, SQLite, logging) | **Done** | Real conversations through Ollama: streaming, tool use, approvals, memory in context. Verified against a real Ollama server (built from source) with deterministic puppet models. Not yet exercised here with a real instruction-tuned model (downloads blocked in the build environment); `jarvis doctor --live` and `JARVIS_TEST_MODEL` cover that on a user's machine. |
| 2. Tools (filesystem, shell, process, system info, projects) | **Done** | Risk-classified shell, reversible writes, trash-based delete, project isolation. |
| 3. State (state engine, world model, event bus, monitoring) | **Done** | System, network, model and self monitors; thresholds, trends, predictions. |
| 4. Tasks (manager, workers, checkpoints, cancellation, scheduling) | **Done** | State machine, recovery, dependencies, throttling, deadlines, schedules and rules. |
| 5. Memory (episodic, semantic, project, retrieval, decisions) | **Done** | FTS5 plus optional embeddings; decision history. |
| 6. Autonomy (planner, verification, recovery, agents, proactive notifications) | **Done (v1)** | Templates plus validated LLM planning, bounded replanning, supervised agents, interrupt policy, emergency mode. |
| 7. Interface (dashboard, HUD, voice, visualisation, screen awareness) | **Partial** | CLI with status block, activity, events, debug view, attachments (drag in, `/attach`, `/paste`) and opt-in screen awareness. No GUI or voice yet. |
| 8. Advanced integration (communications, external APIs, smart devices, robotics, telemetry) | **Interfaces only** | Device contract with telemetry verification and a simulated device; call/message event types and notification rules. No real integrations. |
| Repository Phase 4: multimodal intelligence and perception | **Done (Linux-tested)** | Images, screenshots, documents and files as observations with metadata, provenance, sensitivity and retention. Vision routed only to vision models, local by default, with fallback, caching and a concurrency limit. OCR with measured confidence. Document structure, targeted retrieval, requirements, references and diffs, PDFs included. Measured image comparison. Opt-in, visible, stoppable screen awareness with change events and notifications. Conversational references and clarification. Problems seen in images become verified Phase 3 plans. Content from images and documents can't gain authority. Capability and device discovery. Voice and camera as interfaces only. See [PHASE4.md](PHASE4.md). |
| Repository Phase 3: intelligence and autonomy | **Done (Linux-tested)** | Goals as structured data with ambiguity classes; plans as DAGs on the task system (parallel branches, conditions, bounded loops, approval gates, checkpoints); playbooks, compound requests and validated model-proposed plans; independent verification with quality states; failure categories, dynamic replanning and loop protection; assumptions; agent contracts, coordination and isolation; resource- and priority-aware model routing with an inference queue; memory-aware planning and decision memory; advice, dry runs, simulation and prediction; corrections in flight; autonomy levels; event-driven investigations and proactive suggestions. See [PHASE3.md](PHASE3.md). |
| Repository Phase 2: persistent, always-on runtime | **Done (verified on Windows by the owner)** | Background runtime with a local authenticated API; the CLI is a client; work continues when the interface closes; restart recovery with validation and unknown outcomes; durable scheduler; presence-aware notifications; "what happened while I was away?"; briefing data; unified health; live state including JARVIS's own cost; resource policy; model waiting. macOS and Windows adapters are written but untested. See [RUNTIME.md](RUNTIME.md) and [PHASE2.md](PHASE2.md). |

## Next, in order

1. **Verify the runtime on Windows and macOS**, and exercise the generated systemd, launchd and Startup-folder
   definitions. Add approval decisions to the API (today they go through the conversation: "proceed" / "no").
2. **Web dashboard / HUD.** A local page on top of the API: activity centre (§47), task graph and timeline,
   resource charts from `TrendTracker`, model status, approvals queue, event stream, and dependency-chain
   visualisations from `WorldModel.chain`. Adaptive panels per mode (§126-127).
3. **Coding agent loop.** "Fix whatever is obvious": use the testing agent's failure classification to drive
   minimal, reviewable patches through `file_write` (backed up), re-run the tests, and escalate anything
   architectural. Add git-aware rollback (stash or branch per change set) and a change preview (§112).
4. **Codebase understanding.** Index modules, imports, tests and configuration into the world model (§108) so
   diagnoses and plans can reason about structure.
5. **Event-driven filesystem watching.** Use inotify, FSEvents or ReadDirectoryChangesW (via `watchdog`) instead
   of polling for path monitors.
6. **Voice.** Push-to-talk first: whisper.cpp for recognition, Piper for synthesis, barge-in (stop speaking when
   the user speaks, §70), and the existing interrupt policy deciding what gets spoken. `VoiceInput` (Phase 4) is
   the entry point: a transcript becomes the same normalized input as typing.
7. **Verify perception on Windows**: the screen capture, window information and clipboard scripts. Then UI
   actions (the `UIActionTool` interface exists but isn't registered) behind approval, verified by looking again.
8. **Communications.** Email and calendar as tools, keeping read, draft and send as separate permission levels
   (§24). Call and message screening builds on the existing event types and dedupe ("they're calling again").
9. **Security hardening.** OS keyring backend for `SecretStore`, optional encryption at rest for memory and
   conversation, local authentication for sensitive operations, and multi-user identities and permissions
   (§85-86).
10. **Retrieval upgrades.** `sqlite-vec` for embeddings at scale; a model-assisted intent classifier for requests
    between the deterministic grammar and open chat.
11. **Physical devices.** Real `Device` adapters (Home Assistant, MQTT, serial or microcontrollers,
    OctoPrint) behind the telemetry-first contract.
12. **CI.** Run the test suite on every push, plus the live Ollama tests with puppet models (they need no model
    download); add type checking.

## Known limitations

- Phase 4 is tested on Linux with a simulated screen; the Windows and macOS capture paths haven't been run on
  those systems yet. Without Pillow, comparing two large screenshots decodes them in pure Python (up to a
  couple of seconds each, off the event loop); while watching, screenshots over 2.4 megapixels aren't compared
  pixel by pixel. Rules that recognise errors, dialogs and builds on screen are English-only. JARVIS explains
  code errors it sees but doesn't edit code.
- Phase 3 research reads local files and memory only (no web search tool); performance fixes are limited to
  stopping a process or unloading an idle model, always with approval; conflict detection between sources is a
  heuristic that flags rather than decides.

- Tasks execute only while the runtime process is running. It now runs in the background and survives closing
  the interface, but it does not start at login unless you install the service definition
  (`jarvis runtime install-service`). Interrupted work is recovered on the next start.
- After a crash, a command that was running as a child process may still be running as an orphan; recovery marks
  its step's outcome as unknown but does not yet look for or stop the orphan.
- Planning a task from scratch while no model is available still fails (as before); only steps that need a model
  inside an existing plan wait for one.
- Path monitors poll (every 5 s) and are capped at 5,000 files.
- GPU telemetry covers NVIDIA via `nvidia-smi` only.
- The network probe is a TCP connect to a configurable host, which may not reflect captive portals.
- The deterministic grammar is English-only.
