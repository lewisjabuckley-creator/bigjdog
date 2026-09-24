# Development and testing

## Setup

```bash
git clone <this repository> && cd bigjdog
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                           # Python 3.11+; runtime dependencies: httpx, psutil
pip install -e ".[integration]"                   # optional: live Ollama tests (gguf, numpy)
```

Without installing, run everything as `python -m jarvis` from the repository root (Windows: `py -m jarvis`). The
background runtime is started with the same interpreter and the repository on its `PYTHONPATH`, so this works
from a source checkout too.

## Running JARVIS while developing

```bash
python -m jarvis --simulate                   # simulated model, metrics and network; data in <tmp>/jarvis-sim
python -m jarvis --simulate runtime status    # the simulated runtime keeps running after you exit
python -m jarvis --simulate runtime stop
python -m jarvis --embedded --simulate        # the runtime inside this process (stops when you exit)
python -m jarvis --data-dir /tmp/j1 ...       # an isolated data directory (its own runtime and lock)
python -m jarvis runtime run --verbose        # foreground runtime with structured logs on stderr
python -m jarvis runtime logs -f --structured # follow the JSON log of a background runtime
```

Each data directory has its own runtime, so a simulated runtime never touches your real `~/.jarvis`.

## Tests

| Layer | Command | Needs | Time |
|---|---|---|---|
| Everything offline | `python -m pytest` | nothing (no network, model or GPU) | ~100 s |
| Unit and subsystem | `python -m pytest tests --ignore=tests/test_daemon_process.py --ignore=tests/test_scenarios.py` | nothing | ~10 s |
| Acceptance scenarios (spec §200) | `python -m pytest tests/test_scenarios.py` | nothing | ~5 s |
| Persistent runtime (in-process) | `python -m pytest tests/test_runtime_lifecycle.py tests/test_scheduler.py tests/test_api.py` | nothing | ~10 s |
| Planning and autonomy (Phase 3) | `python -m pytest tests/test_intelligence_units.py tests/test_planning.py tests/test_phase3_scenarios.py` | nothing (starts short-lived busy processes) | ~70 s |
| Live planning (Phase 3) | `JARVIS_OLLAMA_TESTS=1 python -m pytest tests/integration/test_live_planning.py` | a running Ollama, `gguf` and `numpy` | ~5 s |
| Background runtime processes | `python -m pytest tests/test_daemon_process.py` | POSIX (skipped on Windows) | ~12 s |
| Live Ollama plumbing | `JARVIS_OLLAMA_TESTS=1 python -m pytest tests/integration` | a running Ollama, `gguf` and `numpy` | ~25 s |
| Ollama smoke tests for the runtime | `JARVIS_OLLAMA_TESTS=1 python -m pytest tests/integration/test_live_runtime.py` | same | ~6 s |
| Real-model capability checks | `JARVIS_OLLAMA_TESTS=1 JARVIS_TEST_MODEL=llama3.1:8b python -m pytest tests/integration` | an installed instruction model | minutes on CPU |

Use `python -m pytest`, not a bare `pytest` executable that may belong to another Python. On Windows (cmd):
`set JARVIS_OLLAMA_TESTS=1 && py -m pytest tests/integration`.

The offline suite never depends on Ollama. The live tests are opt-in, and they build tiny deterministic *puppet*
models (`tests/integration/puppet.py`) through the Ollama API, so they need no model download and every assertion
is exact. `tests/integration/test_live_runtime.py` includes the Phase 2 acceptance scenario: talk through the real
model, "Analyze this project.", close the interface, the task completes in the background runtime, reopen, "What
happened while I was away?" returns the actual result. It also includes a task that waits through an Ollama
outage, made by a TCP relay the test takes down and brings back.

### Writing tests

- Async test functions run on a fresh event loop (`tests/conftest.py`, no plugin): 30 s timeout, 900 s for tests
  marked `ollama`.
- `tests/helpers.py`: `build_engine()` (task engine on an in-memory database), `make_runtime()` (a full `Runtime`
  with simulated model, metrics and network; `mode="daemon"` for presence-aware behaviour) and `wait_until()`;
  for Phase 3, `spawn_cpu_hog()` (a busy process that is not a child of the test, since JARVIS protects its own
  process tree), `kill_pid()` and `wait_plan()`.
- Process-level tests start `python -m jarvis --config ... --data-dir ... --simulate runtime ...` with the
  repository on `PYTHONPATH`, and always kill any runtime they leave behind.
- Prefer asserting on records (task fields, audit entries, events) over wording, except where the wording is the
  contract (for example "outcome unknown").

## Where things live

See [ARCHITECTURE.md](ARCHITECTURE.md) (subsystems and the process model), [RUNTIME.md](RUNTIME.md) (operating the
runtime, API, recovery, scheduler), [DECISIONS.md](DECISIONS.md) (why), [PHASE2.md](PHASE2.md) (what Phase 2
changed and why it was built that way) and [PHASE3.md](PHASE3.md) (the planning and autonomy layer).
