# Talking to JARVIS with a real model (Ollama)

JARVIS uses [Ollama](https://ollama.com) to run language models on your own computer. Nothing you say leaves
the machine.

## 1. Install Ollama and a model

1. Download and install Ollama from https://ollama.com/download. On Windows and macOS it runs in the background
   after installation; look for the llama icon in the system tray or menu bar.
2. Open a terminal (Windows: press Start, type `cmd`, press Enter) and download a model:

   | Your computer | Command | Download |
   |---|---|---|
   | NVIDIA/AMD graphics card with 8 GB+ memory, or Apple Silicon with 16 GB+ | `ollama pull llama3.1:8b` | about 4.9 GB |
   | Better at using tools, same class | `ollama pull qwen2.5:7b` | about 4.7 GB |
   | No graphics card, or 8 GB of RAM | `ollama pull llama3.2:3b` | about 2.0 GB |

   Optional, for memory recall by meaning rather than keywords: `ollama pull nomic-embed-text` (about 270 MB).
3. Check that it works: `ollama run llama3.1:8b "say hello"` (use the model you pulled).

Pick a model that supports **tool calling** (llama3.1, llama3.2, qwen2.5, qwen3 and mistral-nemo do). Without
tool calling JARVIS can talk, but it can't act on requests like "create a file" or "check the disk".

## 2. Start JARVIS

From the JARVIS folder:

```text
py -m jarvis          (Windows)
python3 -m jarvis     (macOS / Linux)
```

The first line tells you what it found, for example:

```text
JARVIS 0.1.0 — Ready. Talking through llama3.1:8b (local, tools enabled).
```

If Ollama isn't running, or no model is installed, JARVIS tells you exactly what to do. Everything that doesn't
need a model still works in the meantime. JARVIS reconnects on its own once Ollama is up.

## 3. Check the whole chain

```text
py -m jarvis doctor --live
```

This runs a real conversation against your Ollama in a temporary data folder (your real memory and tasks are not
touched) and reports each link:

```text
[PASS] model runtime reachable — ollama 0.9.0
[PASS] chat model installed — llama3.1:8b
[PASS] model answers — llama3.1:8b replied 'ready'
[PASS] streaming — 12 chunk(s), final=yes
[PASS] request reaches tools — ran system_info; replied 'Memory usage is at 41%...'
[PASS] file write verified — file created and read back
[PASS] memory reaches the model — replied 'aubergine'
[PASS] approval gate — deletion paused for approval; declined; file kept
[PASS] background task — task ... completed
```

`WARN` means the plumbing works but the model didn't do what it was asked. That is common with very small models.
Try a larger or more tool-capable one (`--model qwen2.5:7b`). `FAIL` means something is broken; the line says
what.

## Things to try

Anything conversational now goes to the model, which can use JARVIS's tools:

- "How much disk space do I have left?"
- "What's using the most memory right now?"
- "Create a file called shopping.txt in my Documents folder with milk, eggs and bread."
- "Delete shopping.txt." (JARVIS asks for approval first; answer `proceed` or `no`)
- "Run `ping -n 20 127.0.0.1` in the background and tell me when it's done."
- "Remember that my dentist appointment is on Friday at 3pm", then later "When is my dentist appointment?"

The deterministic commands from the README (`status`, `what are you doing?`, `stop`, `continue`, `focus mode`...)
keep working exactly as before and answer instantly.

## Settings

In `jarvis.toml` (see `config/jarvis.example.toml`):

```toml
[models.ollama]
base_url = "http://127.0.0.1:11434"   # another machine on your network works too
num_ctx = 8192                         # context window requested from Ollama
options = { temperature = 0.3 }        # any Ollama sampling option

[models.profiles]
conversation = ["qwen2.5:7b", "llama3.1:8b"]   # preferred models, in order
```

In a conversation: "use the qwen2.5:7b model", "use the local model", "which models?", "unload the vision model".

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama isn't running at http://127.0.0.1:11434" | Start the Ollama app, or run `ollama serve` in a terminal. |
| "no chat model is installed" | `ollama pull llama3.1:8b` (or `llama3.2:3b`). |
| "... can't call tools" | Pull a tool-capable model (`qwen2.5:7b`), or pin one: "use the qwen2.5:7b model". |
| Answers are very slow | Normal on CPU: use a 3B model, close other heavy apps, or set `num_ctx = 4096`. |
| The model ignores tools and just talks | Try `qwen2.5:7b`; small models follow tool instructions less reliably. `jarvis doctor --live` shows which steps work. |
| Where do commands and relative paths run? | In the open project's folder; with no project open, in the folder you started JARVIS from if it's allowed, otherwise your home folder. |
| JARVIS says a path is "outside the allowed roots" | Open the folder as a project first (`open the project D:\my\folder`), or add it to `permissions.allowed_roots`. |

## For developers: live integration tests

```bash
pip install -e ".[integration]"
JARVIS_OLLAMA_TESTS=1 python -m pytest tests/integration                    # plumbing, needs any Ollama
JARVIS_OLLAMA_TESTS=1 JARVIS_TEST_MODEL=llama3.1:8b python -m pytest tests/integration   # + a real model
```

The plumbing tests register tiny deterministic "puppet" models, built on the fly with hand-set weights so each
produces one scripted output such as a specific tool call, and delete them afterwards. That exercises the real
server end to end (template rendering, llama.cpp inference, Ollama's tool-call parsing, streaming, lifecycle)
with exact assertions and no model download. `JARVIS_TEST_MODEL` adds the same checks as `doctor --live`, run
with a real instruction model.
