"""The perception layer as the rest of JARVIS sees it (Phase 4).

``PerceptionService`` takes inputs in (``ingest``), answers questions about them (``understand`` for images and
screenshots, ``document`` for files, ``compare`` for two inputs), reads text (``read_text``), reports capabilities, and
owns screen awareness and device discovery. Every answer is a :class:`Perceived`: the text, what it is based on
(vision, OCR text only, the document itself), provenance for each source, notes that must be shown (images shrunk, a
fallback model, text in the image that looked like instructions, OCR confidence), and whether it failed. A failure is
said plainly — an image JARVIS couldn't see is never described.

Combining sources is the point: the user's words are the question; OCR text, the active project, and the current
system state are context; the vision model looks at the pixels. Only the user's words are instructions.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.types import Provenance, ProvenanceKind
from jarvis.log import get_logger
from jarvis.memory.store import MemoryKind
from jarvis.models.base import ChatMessage, ModelError, NoModelAvailable, Purpose
from jarvis.models.router import TaskProfile
from jarvis.perception import documents as docs
from jarvis.perception import safety
from jarvis.perception.capabilities import render, report
from jarvis.perception.devices import DeviceDiscovery, DeviceReport
from jarvis.perception.inputs import (Attachment, CameraInput, InputKind, Observation, Sensitivity, Status,
                                      VoiceInput)
from jarvis.perception.ocr import OCRResult, OCRService, TesseractEngine, describe as describe_ocr
from jarvis.perception.references import Resolution, resolve
from jarvis.perception.screen import ScreenAwareness, default_backend
from jarvis.perception.store import InputRejected, PerceptionStore
from jarvis.perception.ui import describe_elements, parse_elements
from jarvis.perception.vision import VisionService, VisionUnavailable

log = get_logger("perception")


@dataclass
class Perceived:
    text: str
    observations: list[Observation] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    basis: str = "none"                  # vision | ocr | document | measured | none
    model: str | None = None
    ocr: OCRResult | None = None
    finding: str = ""                    # one line: what was found (for follow-ups and memory)
    problem: dict[str, Any] | None = None
    failed: bool = False
    data: dict[str, Any] = field(default_factory=dict)


# -- recognising problems in what was seen (deterministic, for "fix it") ----------------------------------------------

_PROBLEMS: list[tuple[str, re.Pattern[str]]] = [
    ("disk_full", re.compile(r"(not enough (free )?(disk )?space|disk (is )?(almost )?full|low (on )?disk space|"
                             r"no space left on device|insufficient (disk )?space|ENOSPC|out of disk space|"
                             r"there is not enough space)", re.I)),
    ("missing_module", re.compile(r"(ModuleNotFoundError: No module named ['\"]?(?P<m1>[\w.\-]+)|"
                                  r"No module named ['\"]?(?P<m2>[\w.\-]+)|Cannot find module ['\"](?P<m3>[^'\"]+)|"
                                  r"ImportError: No module named (?P<m4>[\w.\-]+))", re.I)),
    ("performance", re.compile(r"(\bnot responding\b|out of memory|\bhigh cpu\b|running (very )?slow|MemoryError|"
                               r"OutOfMemory|your computer is low on memory)", re.I)),
    ("missing_path", re.compile(r"(FileNotFoundError|No such file or directory|cannot find the (file|path) specified|"
                                r"system cannot find the (file|path)|ENOENT|path not found)", re.I)),
    ("permission", re.compile(r"(PermissionError|Access is denied|permission denied|EACCES|requires administrator|"
                              r"Operation not permitted|run as administrator)", re.I)),
    ("network", re.compile(r"(could not resolve host|name or service not known|connection refused|ECONNREFUSED|"
                           r"ETIMEDOUT|network is unreachable|DNS_PROBE|ERR_CONNECTION|no internet|"
                           r"failed to establish a new connection)", re.I)),
    ("code_error", re.compile(r"(Traceback \(most recent call last\)|File \"[^\"]+\", line \d+|\b\w+Error: |"
                              r"\bException: |error\[E\d+\]|\bFAILED\b|AssertionError|SyntaxError)", re.I)),
]
_FILE_LINE = re.compile(r"File \"(?P<p1>[^\"]+)\", line (?P<l1>\d+)|(?P<p2>[\w./\\-]+\.(py|js|ts|go|rs|java|c|cpp|cs|rb))"
                        r":(?P<l2>\d+)")


def classify_problem(text: str) -> dict[str, Any] | None:
    """The kind of problem a screenshot/document shows, with the evidence line (None when nothing is recognised)."""
    for kind, pattern in _PROBLEMS:
        m = pattern.search(text or "")
        if not m:
            continue
        line = next((l.strip() for l in text.splitlines() if pattern.search(l)), m.group(0))[:240]
        out: dict[str, Any] = {"kind": kind, "evidence": line}
        if kind == "missing_module":
            module = next((m.group(g) for g in ("m1", "m2", "m3", "m4") if m.group(g)), "")
            out["module"] = module.strip("'\".")
        refs = []
        for f in _FILE_LINE.finditer(text):
            path, line_no = (f.group("p1"), f.group("l1")) if f.group("p1") else (f.group("p2"), f.group("l2"))
            refs.append({"path": path, "line": int(line_no)})
        if refs:
            out["locations"] = refs[:5]
        return out
    return None


_LABEL_WORDS = ("error", "warning", "exception", "traceback", "diagram", "chart", "graph", "terminal", "browser",
                "settings", "dialog", "login", "build", "test", "failed", "crash", "code", "circuit", "flowchart",
                "architecture", "schematic", "table", "invoice", "receipt", "map", "photo", "screenshot")


def labels_for(*texts: str) -> list[str]:
    hay = " ".join(t or "" for t in texts).lower()
    return [w for w in _LABEL_WORDS if re.search(rf"\b{w}\b", hay)][:8]


def _first_sentence(text: str, limit: int = 200) -> str:
    text = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text)[:limit]


class PerceptionService:
    def __init__(self, svc: Any, *, screen_backend: Any = None, devices: DeviceDiscovery | None = None,
                 ocr_engines: list[Any] | None = None) -> None:
        self.svc = svc
        cfg = svc.config.perception
        self.config = cfg
        self.store = PerceptionStore(svc.db, svc.config.data_path / "perception", clock=svc.clock, bus=svc.bus,
                                     max_bytes=int(cfg.max_input_mb * 1_000_000),
                                     retention_s=cfg.retention_days * 86400,
                                     screen_retention_s=cfg.screen_retention_days * 86400,
                                     path_check=self._path_check)
        self.vision = VisionService(svc.router, self.store, config=cfg,
                                    local_only=lambda: svc.modes.effective().local_only,
                                    allow_cloud=lambda: svc.config.privacy.allow_cloud and not svc.modes.private,
                                    clock=svc.clock, audit=svc.audit, bus=svc.bus)
        self.ocr = OCRService(self.store, mode=cfg.ocr, tesseract=TesseractEngine(cfg.tesseract_path),
                              transcriber=self.vision.transcribe, vision_available=lambda: self.vision.status()[0],
                              clock=svc.clock, engines=ocr_engines)
        self.screen = ScreenAwareness(screen_backend or default_backend(svc.simulated), self.store, ocr=self.ocr,
                                      vision=self.vision, state=svc.state, bus=svc.bus, audit=svc.audit,
                                      world=svc.world, config=cfg, clock=svc.clock)
        self.devices = devices or DeviceDiscovery(ttl_s=cfg.device_discovery_ttl_s,
                                                  simulated=DeviceReport(method="simulated") if svc.simulated else None)

    def _path_check(self, path: str) -> tuple[bool, str]:
        """A file the user names may be read even outside the usual roots, but never inside a protected path."""
        policy = self.svc.permissions.paths
        for denied in policy.denied_paths:
            if path == denied or path.startswith(denied.rstrip("/\\") + ("\\" if "\\" in denied else "/")):
                return False, f"it's inside a protected location ({denied}), which JARVIS never reads"
        return True, ""

    # -- inputs -------------------------------------------------------------------------------------------------
    def ingest(self, attachments: list[Attachment], *, session_id: str | None = None,
               project_id: str | None = None) -> tuple[list[Observation], list[str]]:
        taken, problems = [], []
        for a in attachments:
            try:
                taken.append(self.store.ingest(a, session_id=session_id, project_id=project_id))
            except InputRejected as exc:
                problems.append(str(exc))
        return taken, problems

    def recent(self, session_id: str | None, limit: int = 20) -> list[Observation]:
        return self.store.recent(session_id, limit=limit)

    def resolve(self, text: str, session_id: str | None, attached: list[Observation] | None = None) -> Resolution:
        return resolve(text, self.recent(session_id), attached=attached, now=self.svc.clock.now(), store=self.store)

    def voice(self) -> VoiceInput:
        report_ = self.devices.cached()
        return VoiceInput(bool(report_ and report_.of("microphone")))

    def camera(self) -> CameraInput:
        report_ = self.devices.cached()
        return CameraInput(len(report_.of("camera")) if report_ else 0)

    # -- images and screenshots -------------------------------------------------------------------------------------
    async def read_text(self, obs: Observation) -> Perceived:
        """"What does it say?": OCR with source, confidence and time."""
        try:
            result = await self.ocr.read(obs)
        except (RuntimeError, InputRejected) as exc:
            return Perceived(f"I couldn't read the text in the {obs.handle}: {exc}.", [obs], failed=True)
        self._remember_ocr(obs, result)
        when = time.strftime("%H:%M on %d %b", time.localtime(result.created_at or self.svc.clock.now()))
        notes = []
        warning = safety.injection_note(safety.instruction_like(result.text), obs.kind.label)
        if warning:
            notes.append(warning)
        return Perceived(describe_ocr(result, f"{obs.handle} ({obs.name})", when), [obs],
                         [obs.provenance(), Provenance(ProvenanceKind.OCR, result.engine, _conf(result))], notes,
                         "ocr", None, result, finding=_first_sentence(result.text))

    async def _ocr_quietly(self, obs: Observation) -> OCRResult | None:
        """OCR from a real engine only (a vision model's transcription would cost a second model call)."""
        engines = [e for e in self.ocr.extra_engines if e.available()[0]]
        if not engines and not (self.ocr.mode in ("auto", "tesseract") and self.ocr.tesseract.available()[0]):
            cached = self.store.cached(obs.sha256, "ocr")
            return OCRResult.from_dict(cached) if cached else None
        try:
            result = await self.ocr.read(obs)
        except Exception:
            return None
        self._remember_ocr(obs, result)
        return result

    def _remember_ocr(self, obs: Observation, result: OCRResult) -> None:
        obs.derived["ocr"] = {**result.to_dict(), "text": safety.scrub(result.text)[:4000], "lines": [
            l.to_dict() for l in result.lines[:80]]}
        if safety.contains_secret(result.text):
            obs.sensitivity = Sensitivity.SECRET
        self.store.update(obs)

    async def understand(self, observations: list[Observation], question: str, *, context: str = "",
                         interactive: bool = True) -> Perceived:
        """Answer the user's question about one or more images, combining OCR, context and a vision model."""
        visual = [o for o in observations if o.kind.visual]
        if not visual:
            return Perceived("There's no image to look at.", observations, failed=True)
        ocr_results = {o.id: await self._ocr_quietly(o) for o in visual}
        extracted = {f"text read from {o.handle} by OCR ({r.engine})": r.marked_text()
                     for o in visual if (r := ocr_results.get(o.id)) and not r.empty}
        injected = [s for r in ocr_results.values() if r for s in safety.instruction_like(r.text)]
        notes: list[str] = []
        provenance = [o.provenance() for o in visual]
        provenance += [Provenance(ProvenanceKind.OCR, r.engine, _conf(r)) for r in ocr_results.values() if r]
        try:
            result = await self.vision.analyze(visual, question, context=context, extracted=extracted,
                                               interactive=interactive)
        except VisionUnavailable as exc:
            return await self._without_vision(visual, question, str(exc), ocr_results, context, provenance)
        injected += safety.instruction_like(result.text)
        notes += result.notes
        warning = safety.injection_note(injected, visual[0].kind.label)
        if warning:
            notes.append(warning)
        text = result.text or "The vision model returned nothing for that image."
        ocr_text = "\n".join(r.text for r in ocr_results.values() if r)
        problem = classify_problem("\n".join([ocr_text, text]))
        finding = _first_sentence(text)
        for o in visual:
            o.derived["finding"] = finding
            if not o.derived.get("description") and re.search(r"\b(describe|what is|what's in|what does)\b",
                                                              question, re.I):
                o.derived["description"] = _first_sentence(text, 300)
            if problem:
                o.derived["problem"] = problem
            if warning:
                o.derived["injection_warning"] = warning
            o.labels = list(dict.fromkeys(o.labels + labels_for(text, ocr_text, o.name)))
            o.status = Status.PROCESSED
            if safety.contains_secret(text):
                o.sensitivity = Sensitivity.SECRET
            self.store.update(o)
        await self._remember(visual, finding, problem)
        return Perceived(text, visual, provenance + result.provenance[len(visual):], notes, "vision", result.model,
                         next((r for r in ocr_results.values() if r), None), finding, problem,
                         data={"duration_s": result.duration_s, "cached": result.cached})

    async def _without_vision(self, visual: list[Observation], question: str, why: str,
                              ocr_results: dict[str, OCRResult | None], context: str,
                              provenance: list[Provenance]) -> Perceived:
        """No vision model: say so, and help from the image's text if an OCR engine could read it."""
        texts = {o.id: r for o in visual if (r := ocr_results.get(o.id)) and not r.empty}
        if not texts:
            ocr_ok, ocr_why = self.ocr.status()
            extra = "" if ocr_ok else f" I also can't read text in images ({ocr_why})."
            which = " or ".join(o.handle for o in visual[:3])
            return Perceived(f"I can't look at images right now: {why}.{extra} I haven't analysed {which}.", visual,
                             provenance[:len(visual)], failed=True, basis="none")
        notes = [f"I couldn't look at the image itself ({why}); this is based only on the text I could read from it."]
        answer = ""
        model = None
        joined = "\n\n".join(safety.frame(f"text read from {o.handle} by OCR", texts[o.id].marked_text())
                             for o in visual if o.id in texts)
        try:
            routed = await self.svc.router.chat(
                TaskProfile(purpose=Purpose.REASONING, complexity="medium", local_only=self.vision.local_only(visual)),
                [ChatMessage("system", "You are JARVIS. You only have the text read from an image by OCR, not the "
                                       "image itself; say so if the text isn't enough. " + safety.STANDING_RULE),
                 ChatMessage("user", (f"Context: {context}\n\n" if context else "") + joined +
                             f"\n\nThe user's request: {question}")])
            answer, model = routed.response.content.strip(), routed.response.model
        except (NoModelAvailable, ModelError):
            pass
        ocr_text = "\n".join(r.text for r in texts.values())
        if not answer:
            first = next(iter(texts.values()))
            answer = "The text in it reads:\n" + first.marked_text()[:1500]
        warning = safety.injection_note(safety.instruction_like(ocr_text), visual[0].kind.label)
        if warning:
            notes.append(warning)
        problem = classify_problem(ocr_text)
        for o in visual:
            if problem:
                o.derived["problem"] = problem
            o.labels = list(dict.fromkeys(o.labels + labels_for(ocr_text, o.name)))
            self.store.update(o)
        prov = provenance + ([Provenance(ProvenanceKind.INFERENCE, model, "reasoning over OCR text")] if model else [])
        return Perceived(answer, visual, prov, notes, "ocr", model, next(iter(texts.values())),
                         _first_sentence(answer), problem)

    async def compare(self, a: Observation, b: Observation, question: str = "What changed?") -> Perceived:
        if a.kind in (InputKind.DOCUMENT, InputKind.FILE) and b.kind in (InputKind.DOCUMENT, InputKind.FILE):
            return await self.compare_documents(a, b)
        can_ocr = bool([e for e in self.ocr.extra_engines if e.available()[0]]) or \
            (self.ocr.mode in ("auto", "tesseract") and self.ocr.tesseract.available()[0])
        out = await self.vision.compare(a, b, question, ocr=self.ocr.read if can_ocr else None)
        lines: list[str] = []
        pixels = out.get("pixels", {})
        lines.append(f"Measured: {pixels.get('description', 'not compared')}.")
        text = out.get("text")
        if text and text.get("available"):
            if text["same"]:
                lines.append("Text: the readable text is the same in both.")
            else:
                if text["added"]:
                    lines.append("Text that appeared: " + "; ".join(f"\"{t[:80]}\"" for t in text["added"][:5]) + ".")
                if text["removed"]:
                    lines.append("Text that disappeared: " + "; ".join(f"\"{t[:80]}\"" for t in text["removed"][:5])
                                 + ".")
        model = out.get("model", {})
        notes = []
        basis = "measured"
        used_model = None
        if model.get("text"):
            lines.append(f"The vision model's reading ({model['model']}; it can be wrong): {model['text']}")
            notes += model.get("notes", [])
            basis, used_model = "vision", model["model"]
        elif model.get("unavailable"):
            notes.append(f"No vision model looked at them ({model['unavailable']}), so the comparison above is only "
                         "what could be measured.")
        if pixels.get("identical"):
            lines = ["They are the same image: every pixel is identical."]
        prov = [a.provenance(), b.provenance()]
        if used_model:
            prov.append(Provenance(ProvenanceKind.VISION, used_model, "a vision model's reading; may be wrong"))
        return Perceived("\n".join(lines), [a, b], prov, notes, basis, used_model,
                         finding=_first_sentence(model.get("text") or lines[0]), data=out)

    async def ui(self, obs: Observation, question: str = "") -> Perceived:
        try:
            result = await self.vision.ui_elements(obs, context=question)
        except VisionUnavailable as exc:
            return Perceived(f"I can't identify on-screen elements without a vision model ({exc}).", [obs],
                             failed=True)
        app, window, elements = parse_elements(result.data)
        if not elements:
            return Perceived(result.text or "I couldn't make out distinct interface elements.", [obs],
                             result.provenance, result.notes, "vision", result.model)
        obs.derived["ui"] = {"application": app, "window": window, "elements": [e.to_dict() for e in elements]}
        self.store.update(obs)
        return Perceived(describe_elements(app, window, elements), [obs], result.provenance, result.notes, "vision",
                         result.model, finding=f"{window or app}: {len(elements)} elements",
                         data={"elements": [e.to_dict() for e in elements]})

    async def diagram(self, obs: Observation, question: str = "") -> Perceived:
        try:
            result = await self.vision.diagram(obs, question)
        except VisionUnavailable as exc:
            return Perceived(f"I can't read diagrams without a vision model ({exc}).", [obs], failed=True)
        data = result.data if isinstance(result.data, dict) else {}
        connections = [c for c in data.get("connections") or [] if isinstance(c, dict) and c.get("from") and c.get("to")]
        if not connections and not data.get("components"):
            return Perceived(result.text, [obs], result.provenance, result.notes, "vision", result.model)
        lines = [f"{(data.get('kind') or 'Diagram').capitalize()}: " +
                 ", ".join(str(c) for c in (data.get("components") or [])[:20])]
        for c in connections[:30]:
            lines.append(f"{c['from']} → {c['to']}" + (f" ({c['label']})" if c.get("label") else ""))
        if data.get("notes"):
            lines.append(str(data["notes"]))
        obs.derived["diagram"] = {"kind": data.get("kind"), "components": data.get("components"),
                                  "connections": connections[:40]}
        obs.labels = list(dict.fromkeys(obs.labels + ["diagram"]))
        self.store.update(obs)
        return Perceived("\n".join(lines), [obs], result.provenance, result.notes + [
            "Read by a vision model: check connections that matter before relying on them."], "vision", result.model,
            finding=lines[0][:200], data={"diagram": obs.derived["diagram"]})

    # -- documents -------------------------------------------------------------------------------------------------
    def extract(self, obs: Observation) -> docs.DocumentExtract:
        doc = docs.read_document(self.store.data(obs), obs.name, obs.mime)
        obs.derived["document"] = doc.summary_dict()
        obs.pages = doc.pages
        if safety.contains_secret(doc.text):
            obs.sensitivity = Sensitivity.SECRET
        obs.status = Status.PROCESSED
        obs.labels = list(dict.fromkeys(obs.labels + [doc.kind] + labels_for(doc.title)))
        self.store.update(obs)
        return doc

    async def document(self, obs: Observation, question: str, *, mode: str = "answer",
                       topic: str = "") -> Perceived:
        """Summaries, requirements, references and answers from a document, with line/page provenance."""
        try:
            doc = self.extract(obs)
        except InputRejected as exc:
            return Perceived(str(exc), [obs], failed=True)
        prov = [obs.provenance()]
        notes = [n[:1].upper() + n[1:] + "." for n in doc.notes]
        if obs.sensitivity == Sensitivity.SECRET:
            notes.append("It contains what look like credentials, so it stays on this computer and I won't remember "
                         "anything from it.")
        if not doc.readable:
            return Perceived(f"I couldn't get readable text out of {obs.name}. " + " ".join(notes), [obs], prov,
                             failed=True, basis="document")
        injected = safety.instruction_like(doc.text[:200000])
        warning = safety.injection_note(injected, "document")
        if warning:
            notes.append(warning)
        if mode == "requirements":
            found = docs.requirements(doc)
            if not found:
                return Perceived(f"I didn't find requirement statements (shall / must / should / required, or "
                                 f"numbered REQ items) in {obs.name}.", [obs], prov, notes, "document")
            by = {"must": [], "must not": [], "should": [], "should not": [], "listed": []}
            for r in found:
                by[r["strength"]].append(r)
            lines = [f"Requirements in {obs.name} ({len(found)} found):"]
            for strength in ("must", "must not", "should", "should not", "listed"):
                for r in by[strength]:
                    lines.append(f"- [{strength}] {r['text']} ({r['where']})" + (f" {r['id']}" if r["id"] and
                                                                                   r["id"] not in r["text"] else ""))
            finding = f"{len(found)} requirement(s) in {obs.name}"
            obs.derived["summary"] = finding
            self.store.update(obs)
            return Perceived("\n".join(lines), [obs], prov, notes, "document", finding=finding,
                             data={"requirements": found})
        if mode == "references":
            hits = docs.references(doc, topic or question)
            if not hits:
                return Perceived(f"{obs.name} doesn't mention \"{topic or question}\".", [obs], prov, notes,
                                 "document")
            lines = [f"\"{topic}\" in {obs.name} ({len(hits)} place(s)):"] + [f"- {h['where']}: {h['text']}"
                                                                              for h in hits[:20]]
            return Perceived("\n".join(lines), [obs], prov, notes, "document", data={"references": hits})
        if mode == "outline":
            return Perceived(docs.outline_summary(doc), [obs], prov, notes, "document")
        chunks = docs.relevant(doc, question if mode == "answer" else "",
                               max_chars=self.config.document_max_chars)
        answer, model = await self._summarise(obs, doc, chunks, question, mode)
        if not answer:
            answer = docs.outline_summary(doc)
            notes.append("No language model was available, so this is a factual outline rather than a summary.")
        finding = _first_sentence(answer)
        obs.derived["summary"] = finding
        self.store.update(obs)
        await self._remember([obs], finding, None)
        if model:
            prov.append(Provenance(ProvenanceKind.INFERENCE, model, "summary of the parts shown, with their lines"))
        return Perceived(answer, [obs], prov, notes, "document", model, finding=finding,
                         data={"parts": [{"where": c.where(doc), "title": c.title} for c in chunks]})

    async def _summarise(self, obs: Observation, doc: docs.DocumentExtract, chunks: list[docs.Chunk],
                         question: str, mode: str) -> tuple[str, str | None]:
        if not chunks:
            return "", None
        parts = "\n\n".join(safety.frame(f"{obs.name}, {c.where(doc)}" + (f" — {c.title}" if c.title else ""), c.text)
                            for c in chunks)
        ask = question if mode == "answer" else "Summarise this document: what it is, and its important points."
        try:
            routed = await self.svc.router.chat(
                TaskProfile(purpose=Purpose.SUMMARIZATION, complexity="medium",
                            local_only=self.vision.local_only([obs]) or obs.sensitivity != Sensitivity.NORMAL),
                [ChatMessage("system", "You are JARVIS. Answer only from the document parts provided. After each "
                                       "point, cite where it came from in brackets, like [lines 12-30] or [page 3]. "
                                       "If the parts don't contain the answer, say so. " + safety.STANDING_RULE),
                 ChatMessage("user", f"Document: {obs.name} ({doc.kind}, {len(doc.lines)} lines"
                                     + (f", {doc.pages} pages" if doc.pages else "") + ")\n\n" + parts +
                             f"\n\nThe user's request: {ask}")])
        except (NoModelAvailable, ModelError):
            return "", None
        return routed.response.content.strip(), routed.response.model

    async def compare_documents(self, a: Observation, b: Observation) -> Perceived:
        try:
            da, db = self.extract(a), self.extract(b)
        except InputRejected as exc:
            return Perceived(str(exc), [a, b], failed=True)
        diff = docs.compare(da, db)
        if diff["identical"]:
            return Perceived(f"{a.name} and {b.name} have the same content.", [a, b], [a.provenance(), b.provenance()],
                             basis="document", data=diff)
        lines = [f"{a.name} → {b.name}: {diff['added_lines']} line(s) added, {diff['removed_lines']} removed "
                 f"({diff['similarity']:.0%} similar)."]
        if diff["sections_added"]:
            lines.append("New sections: " + ", ".join(diff["sections_added"]) + ".")
        if diff["sections_removed"]:
            lines.append("Removed sections: " + ", ".join(diff["sections_removed"]) + ".")
        if diff["changed_sections"]:
            lines.append("Changed sections: " + ", ".join(diff["changed_sections"]) + ".")
        for c in diff["changes"][:8]:
            where = f"new lines {c['new_lines'][0]}-{c['new_lines'][1]}" if c["new_lines"] else \
                f"old lines {c['old_lines'][0]}-{c['old_lines'][1]}"
            if c["type"] == "changed":
                lines.append(f"- changed ({where}): \"{c['old'][:100]}\" → \"{c['new'][:100]}\"")
            elif c["type"] == "added":
                lines.append(f"- added ({where}): \"{c['new'][:140]}\"")
            else:
                lines.append(f"- removed ({where}): \"{c['old'][:140]}\"")
        return Perceived("\n".join(lines), [a, b], [a.provenance(), b.provenance()], basis="document",
                         finding=lines[0], data=diff)

    # -- memory ----------------------------------------------------------------------------------------------------
    async def _remember(self, observations: list[Observation], finding: str, problem: dict[str, Any] | None) -> None:
        """Keep what was learned (never the image): a line per important observation, via the memory system."""
        if not self.config.remember_findings or not finding or self.svc.memory.suppressed:
            return
        for o in observations:
            if o.sensitivity != Sensitivity.NORMAL:
                continue
            when = time.strftime("%d %b %H:%M", time.localtime(o.created_at))
            text = safety.scrub(f"{o.handle} ({o.name}, {when}): {finding}")
            try:
                await self.svc.memory.remember(text, kind=MemoryKind.EPISODIC, subject=f"observation:{o.id}",
                                               project_id=o.project_id, provenance=o.provenance(),
                                               importance=0.6 if problem else 0.4,
                                               expires_at=o.created_at + 90 * 86400)
            except Exception as exc:          # memory is best-effort; perception must not fail because of it
                log.warning("perception_memory_failed", error=repr(exc))

    # -- capabilities, context, maintenance ----------------------------------------------------------------------------
    async def capabilities(self, *, refresh: bool = False) -> list[Any]:
        return await report(self, refresh_devices=refresh)

    async def capabilities_text(self) -> str:
        return render(await self.capabilities())

    def context_block(self, session_id: str | None, *, within_s: float = 3600.0, limit: int = 3) -> str:
        """Recent observations for the chat model, framed as data (what they showed, not the pixels)."""
        now = self.svc.clock.now()
        lines = []
        for o in self.recent(session_id, limit=limit):
            if now - o.created_at > within_s or o.sensitivity == Sensitivity.SECRET:
                continue
            summary = o.summary()
            lines.append(f"- {o.handle} ({o.name}, {o.kind.label}): {summary or 'not analysed yet'}")
        if not lines:
            return ""
        return safety.frame("inputs the user shared recently (what JARVIS found in them)", "\n".join(lines))

    def forget(self, obs_id: str) -> Observation | None:
        obs = self.store.forget(obs_id)
        if obs is not None:
            for row in self.svc.db.query("SELECT id FROM memories WHERE subject=?", (f"observation:{obs_id}",)):
                self.svc.memory.forget(row["id"])      # what was remembered from it goes too
        return obs

    def maintenance(self) -> int:
        return self.store.purge_expired()


def _conf(result: OCRResult) -> str:
    if result.measured and result.mean_confidence is not None:
        return f"{result.mean_confidence:.0%} average confidence"
    return "confidence not measured"
