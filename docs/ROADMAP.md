# Roadmap

Status of each build phase from the specification (§165), and what comes next. "Done" means implemented and
covered by tests. "Partial" means the architecture and a working subset exist.

| Phase | Status | Notes |
|---|---|---|
| 1. Foundation (Ollama adapter, model abstraction, conversation, config, SQLite, logging) | **Done** | Ollama adapter tested against a mock server. The OpenAI-compatible adapter is optional. |
| 2. Tools (filesystem, shell, process, system info, projects) | **Done** | Risk-classified shell, reversible writes, trash-based delete, project isolation. |
| 3. State (state engine, world model, event bus, monitoring) | **Done** | System, network, model and self monitors; thresholds, trends, predictions. |
| 4. Tasks (manager, workers, checkpoints, cancellation, scheduling) | **Done** | State machine, recovery, dependencies, throttling, deadlines, schedules and rules. |
| 5. Memory (episodic, semantic, project, retrieval, decisions) | **Done** | FTS5 plus optional embeddings; decision history. |
| 6. Autonomy (planner, verification, recovery, agents, proactive notifications) | **Done (v1)** | Templates plus validated LLM planning, bounded replanning, supervised agents, interrupt policy, emergency mode. |
| 7. Interface (dashboard, HUD, voice, visualisation, screen awareness) | **Partial** | CLI with status block, activity, events, debug view. No GUI, voice or screen awareness yet. |
| 8. Advanced integration (communications, external APIs, smart devices, robotics, telemetry) | **Interfaces only** | Device contract with telemetry verification and a simulated device; call/message event types and notification rules. No real integrations. |

## Next, in order

1. **Background daemon plus local API.** Run the runtime as a user service (systemd, launchd) and expose a local,
   authenticated API (Unix socket or localhost HTTP with a token) for state, tasks, events (streaming),
   approvals and conversation. The CLI becomes a client. Tasks then keep running when the terminal closes.
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
   the user speaks, §70), and the existing interrupt policy deciding what gets spoken.
7. **Vision and screen awareness.** Screenshots and images to vision-capable models via the router
   (`needs_vision`), with explicit, visible opt-in for screen monitoring (§68).
8. **Communications.** Email and calendar as tools, keeping read, draft and send as separate permission levels
   (§24). Call and message screening builds on the existing event types and dedupe ("they're calling again").
9. **Security hardening.** OS keyring backend for `SecretStore`, optional encryption at rest for memory and
   conversation, local authentication for sensitive operations, and multi-user identities and permissions
   (§85-86).
10. **Retrieval upgrades.** `sqlite-vec` for embeddings at scale; a model-assisted intent classifier for requests
    between the deterministic grammar and open chat.
11. **Physical devices.** Real `Device` adapters (Home Assistant, MQTT, serial or microcontrollers,
    OctoPrint) behind the telemetry-first contract.
12. **CI.** Run the test suite on every push; add type checking.

## Known limitations

- Tasks execute only while the runtime is running (see item 1). They are recovered on the next start.
- Path monitors poll (every 5 s) and are capped at 5,000 files.
- GPU telemetry covers NVIDIA via `nvidia-smi` only.
- The network probe is a TCP connect to a configurable host, which may not reflect captive portals.
- The deterministic grammar is English-only.
