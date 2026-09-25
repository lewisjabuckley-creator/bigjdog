# Phase 4: multimodal intelligence and perception

This file records what existed before Phase 4, the design, what was built and what was deliberately left out.
Phase 4 adds one layer, `jarvis/perception/`, that turns images, screenshots, documents and (when the user
allows it) the screen into *observations* the existing systems can reason about. It does not replace any of
them: vision goes through the model router, actions through the planner and the tool registry, findings into
memory, alerts through the notification policy.

Typing stays the main way to use JARVIS. Nothing needs a microphone or a camera, and neither is ever switched on.

## Before Phase 4 (commit a277038)

| Phase 4 capability | Already there | Missing |
|---|---|---|
| Vision models | The Ollama adapter passed images; `Capability.VISION` was detected from `/api/show`; the router had `needs_vision` | Anything that produced an image to send; routing by sensitivity; a concurrency limit for vision work |
| Inputs | Typed text only | Images, screenshots, documents and files as first-class inputs with metadata, provenance and sensitivity |
| Documents | `file_read` (whole file, truncated) and `file_search` | Structure, targeted retrieval, requirements, references, comparison, PDFs |
| Screen | Nothing | Capture, window information, change detection, events, and an explicit on/off switch |
| Planning from what was seen | Phase 3 plans from typed goals | A problem seen in an image turning into a goal, a plan and a verified fix |
| Security | Permission levels, approvals, instruction hierarchy | Treating content from images and documents as untrusted data, and stopping it from gaining authority |

## The design

```text
  typed text ─────────────────────────────────────────────────────────► orchestrator (unchanged path)
  image / screenshot / document / file ─► PerceptionStore ─► Observation (id, kind, origin, session, project,
     (drag in, /attach, /paste, API)        content-addressed     sensitivity, provenance, derived results)
  screen (only if the user turned it on) ─► ScreenAwareness ─┘
                                                 │
            ┌────────────────────────────────────┼─────────────────────────────────────────┐
            ▼                                    ▼                                         ▼
     VisionService (router,             OCRService (Tesseract with measured        documents (structure,
     needs_vision, local unless          confidence; a vision model as an           retrieval, requirements,
     allowed, 1 at a time, cached)       unmeasured fallback)                       references, diffs, PDF)
            └──────────────── PerceptionService.understand / compare / document ────────────┘
                                                 │
     PerceptionDialogue (references: "this", "the second one", "the screenshot from earlier"; clarification)
                                                 │
       answer with provenance ─── memory (findings, not images) ─── "fix it" ─► goal ─► Phase 3 plan
```

### Inputs and observations (`perception/inputs.py`, `perception/store.py`)
Every non-text input becomes an `Observation`: kind (image, screenshot, document, file, and the future audio and
camera frame kinds), origin (the user, the clipboard, a screen capture, a tool, the API), session, project,
timestamp, sensitivity, labels and a handle people can use ("screenshot 2", "document 1"). The bytes are stored
once per content hash in the data directory; derived results (OCR, vision answers, document structure) are cached
by hash, operation and parameters, so asking the same thing twice costs nothing.

Inputs are validated before anything else sees them: size limit, empty files, image headers parsed and checked
for truncation without any library, unsupported binary refused, protected paths (`~/.ssh` and so on) refused.
Stored copies expire (7 days; screen captures after 1 day); what was learned from them stays. "Forget that
image" deletes the file, its cached results and the memories made from it.

`VoiceInput` and `CameraInput` are interfaces only. They report "not available" or "not built yet", and
`transcribe`/`capture` raise `NotImplementedError`. When they are built they produce the same normalized input as
typing, and nothing else has to change.

### Vision (`perception/vision.py`, `perception/images.py`)
Images go only to models that report the vision capability, chosen by the existing router with
`TaskProfile(purpose=VISION, needs_vision=True)`. A reply from a model without vision is rejected as a
safeguard. If no vision model is installed, JARVIS says so and names one to install; it never pretends to have
looked. If a vision model fails, the router falls back to another vision model and the answer says so. Vision
stays on local models unless `perception.allow_cloud_vision` *and* `privacy.allow_cloud` are both on, the mode
isn't offline or private, and the image isn't marked private (screen captures always are). One analysis runs at
a time (`max_concurrent_vision`); results are cached, audited (`vision_analysis`) and emitted as
`PERCEPTION_STARTED/COMPLETED/FAILED` events.

Large images are shrunk before sending when Pillow is installed. Without Pillow, images up to the pixel limit are
sent whole and larger ones are refused with a hint. PNG decoding for comparisons is pure Python (all colour types
and filters), so comparing two screenshots works without any library.

**Comparing images** ("what changed?") has three parts that are kept apart:
1. **Measured**: the share of the picture whose pixels changed and where ("25% of the picture changed (bottom
   left, bottom centre, bottom right)"). It compares every pixel of small images and an even sample of large
   ones, and says which it did.
2. **Text**: text that appeared or disappeared, read by OCR.
3. **The vision model's reading**, labelled with the model's name and "it can be wrong".

So a real difference is never confused with what the model thinks it sees.

### OCR (`perception/ocr.py`)
Tesseract, if installed, reads the text with **measured** confidence, line positions and `[?]` marks on words it
was unsure of. It is found on the PATH or in `C:\Program Files\Tesseract-OCR`. Without Tesseract, a vision model
can transcribe the text; that reading is labelled as unmeasured. Every reading says where it came from, which
engine read it and how sure it was.

### Documents (`perception/documents.py`)
Text, Markdown, code, JSON, CSV, logs, configuration and PDF (pypdf, then `pdftotext`, then a small built-in
reader for simple PDFs). Each kind gets a structure: headings, code definitions, JSON keys, CSV columns, log error
lines, config sections, PDF pages. Questions are answered from the relevant parts only, scored and fitted to
`document_max_chars`, and each part is cited with its lines or page. Deterministic extractors find requirements
("must", "must not", "should", "should not", numbered and `REQ-` items, with lines), references (files, URLs,
issue and section references) and differences between two versions. Code files are read the same way. Project
structure stays with the Phase 3 project probe and analysis; there is no second, competing index.

### Screen awareness (`perception/screen.py`)
Screen awareness is **off** until the user turns it on, and only the user can turn it on. An agent, an automation
or a model can switch it off but never on; a refused attempt is audited. There are three modes:
- **off**: nothing is captured.
- **on_request** ("turn on screen awareness"): JARVIS looks only when asked ("what's on my screen?").
- **watching** ("watch my screen"): a cheap window check every `screen_interval_s`, and a full capture plus OCR
  only on a change or every `screen_capture_every` checks.

The mode is remembered across restarts, and whenever it isn't off, every interface says so when it opens. It is
also shown by "what can you see?", `status` and `/screen`, and stopped with "stop watching my screen" or
`/screen off`. Every capture is audited with its reason. A capture showing a
password, key or card number is deleted at once and only a scrubbed reading is kept.

Rules turn what is on screen into a `ScreenState` (active application, window, detected state, relevant text,
elements) and into events, raised only once per change: `SCREEN_ERROR_DETECTED`, `DIALOG_APPEARED`,
`APPLICATION_CHANGED`, `APPLICATION_CLOSED`, `BUILD_FAILED`, `BUILD_COMPLETED`, `IMPORTANT_UI_CHANGE` and
`SCREEN_STATE_CHANGED`. Important ones reach the user through the existing notification priorities and dedupe.

The capture backends are Windows (PowerShell and .NET, window titles via ctypes), macOS (`screencapture`) and
Linux (grim, gnome-screenshot, spectacle, scrot or ImageMagick), plus a simulated screen for tests.

### Interface elements and actions (`perception/ui.py`)
A vision model can list on-screen elements (buttons, fields, tabs, dialogs) as structured data, and
`find(elements, "the Save button")` locates one. `UIActionTool` defines click, type, scroll, select, open and
close as a consequential-level tool that verifies by looking again. It is **not registered**: JARVIS does not
operate the mouse or keyboard in Phase 4.

### Conversation (`core/perception_dialogue.py`, `perception/references.py`)
Attachments arrive by dropping a file into the window (its path is recognised), `/attach <file>`, `/paste` (the
clipboard image, e.g. after Win+Shift+S) or the API. References are resolved from the record, not guessed:
- "this" or "that" means what was just attached;
- "the second screenshot" and "screenshot 2" are picked by number;
- "compare these" compares the last two;
- "the previous one" means the one before;
- "the screenshot from earlier" and "the diagram from yesterday" are found by time and topic, across sessions.

When a reference is ambiguous JARVIS asks which one. When nothing was shared it says so, rather than inventing an
image. Typed requests that merely contain a path ("back up C:\Docs to E:") keep going to the Phase 3 planner;
perception only takes over when the message is about an input.

### From seeing to doing (`intelligence/visual.py`)
A recognised problem in an image becomes a Phase 3 goal *only when the user asks* ("fix it", or "look at this
error, figure out what's causing it, fix it and verify the fix"):

| Seen | Plan |
|---|---|
| disk full | free up disk space (the Phase 3 playbook) |
| computer not responding / slow | find what is using it (Phase 3) |
| `ModuleNotFoundError` | check the import → `pip install` (asks first) → import again, checked by the verifier |
| an error in code | locate it in the project → run the tests → explain the cause (read-only) |
| anything else | guidance only, never an improvised plan |

The plan starts with a `perceive` step that re-reads the recorded analysis; no second model call is made. When
screen awareness is on, the plan ends with an optional "look at the screen again" step, which is reported but
doesn't decide the result, since a dialog may simply still be open. Everything else is an ordinary plan step:
approval gate, execution, independent verification, report.

### Memory, provenance and observability
Findings become episodic memories (`subject=observation:<id>`), so "what was wrong with that printer
screenshot?" can be answered later. The images themselves are never stored in memory. Answers carry provenance:
the input, OCR (engine, confidence) and the vision model (name, "may be wrong"). Input, perception, OCR and
screen events go to the event bus; captures and vision calls go to the audit log.

## Security: content is data, never instructions

- Text from images, documents, files and the screen is wrapped in delimited `<<<EXTERNAL DATA …>>>` blocks
  under a standing rule that it is data, and a delimiter inside the content can't close the block early.
- Instruction-like text ("ignore previous instructions", "SYSTEM:", "you are now…") is spotted and the answer
  says it was treated as part of the image or document.
- Once a model has read external content (the perception context is in the conversation, or it called
  `image_analyze`, `image_read_text`, `document_read`, `screen_look`, `file_read` or `file_search`), anything
  above observing needs the user's explicit approval. Standing grants don't count; the approval request says it
  was suggested after reading the content. Tasks and plans created in that state carry the same restriction,
  and plans built from perceived content gate every action above observing.
- Document summaries are made with no tools offered to the model. A model's interpretation never grants a
  permission, and screen awareness can't be enabled by anything but the user.

## Capability discovery

"What can you see?" (also `/perception` and `GET /v1/perception`) reports the real state of each capability:
- text input and text reasoning;
- the vision model and OCR engine (or what to install);
- documents (and which PDF reader);
- image shrinking;
- screen capture (disabled, on request, or watching);
- camera and microphone ("not connected" or "not built yet");
- displays;
- offline mode.

Devices are discovered with one read-only operating-system query, cached, and never opened.

## Settings (`[perception]` in `jarvis.toml`)

```toml
[perception]
ocr = "auto"                 # auto | tesseract | vision | off
allow_cloud_vision = false   # also needs [privacy] allow_cloud = true; never for private inputs
screen_mode = "off"          # the starting value; the user switches it in conversation
screen_interval_s = 30
screen_capture_every = 4
retention_days = 7           # stored copies of inputs (what was learned is kept)
screen_retention_days = 1
max_input_mb = 40
max_concurrent_vision = 1
document_max_chars = 12000
```

## Using it (Windows)

1. Optional extras: `py -m pip install pillow pypdf` (image shrinking and better PDF reading). For reading text
   in images, install Tesseract (the UB Mannheim installer); it's found automatically.
2. A vision model: `ollama pull llama3.2-vision` (about 8 GB; best with a graphics card), or
   `ollama pull moondream` (about 1.7 GB; runs on most computers, less accurate), or `ollama pull llava`.
3. In JARVIS:
   - "what can you see?"
   - drag a screenshot into the window and type "what's wrong with this?", or press Win+Shift+S, type `/paste`,
     then ask;
   - "what does the screenshot from earlier say?";
   - "find the important requirements in C:\path\spec.pdf";
   - attach two images and ask "what changed?";
   - "turn on screen awareness", then "what's on my screen?";
   - "watch my screen" and "stop watching my screen";
   - `/inputs` lists what you've shared; "forget that image" deletes one.

## Definition of done → tests

`tests/test_phase4_scenarios.py` runs the ten scenarios through the conversation:

| # | Scenario | Checked |
|---|---|---|
| 1 | Text unchanged | Status, chat and planning requests never touch vision or store anything |
| 2 | Image: "What's wrong with this?" | One call to a vision model with the image and the OCR reading; provenance |
| 3 | Screenshot: "How do I fix this?" | Classified as a screenshot; problem recognised; fix offered, not done; "the screenshot from earlier" |
| 4 | Two images: "What changed?" | Measured share and regions, text changes, the model's reading labelled |
| 5 | Document: requirements | must / must not / should with line numbers, without a vision model |
| 6 | Look, find the cause, fix, verify | Screenshot → plan with a perceive step → approval → files moved → verified independently → screen checked |
| 7 | Screen awareness | Nothing captured while off; watching raises `BUILD_FAILED`; stop really stops |
| 8 | No microphone | Everything works typed; a detected microphone is listed and never used |
| 9 | Offline | A cloud vision model is never used; a local one is |
| 10 | Security | Instructions in an image or a document do nothing without the user's approval |

`tests/test_perception.py` covers the parts:
- images, ingestion, routing, fallback, caching, offline and concurrency;
- OCR and documents;
- screen rules and privacy;
- references, follow-ups, injection and memory;
- retention, the API and the CLI.

`tests/integration/test_live_vision.py` runs against a real Ollama server with *vision puppets*: a puppet
language model plus a tiny real CLIP projector. Ollama reports the vision capability and runs each image through
llama.cpp's image encoder, while replies stay scripted. It checks that:
- vision is detected and routed;
- the image really reaches the model (the model's own token count grows);
- a broken image is refused by the server and the failure is reported;
- JARVIS falls back between vision models;
- only one analysis runs at a time.

## Not built in Phase 4

- Voice (speech recognition and synthesis) and camera input: interfaces only.
- The HUD.
- Operating the mouse and keyboard: the interface exists; the tool is not registered.
- Editing code to fix errors: the visual debugging plan explains the cause and the fix, and the user applies it.
- Tested on Linux with a simulated Windows screen. The Windows capture and clipboard scripts use built-in
  PowerShell and .NET, but haven't been run on Windows yet.
