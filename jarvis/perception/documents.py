"""Reading documents and files without loading them whole into a model (Phase 4 §15-16).

A document is parsed deterministically into structure: sections (Markdown/reStructuredText headings, numbered
headings, PDF pages), code definitions (Python via ``ast``, other languages by pattern), JSON keys, CSV columns, log
errors, configuration sections. Questions are answered from *targeted* parts: the sections that match the question
(scored by term overlap) up to a character budget, each with its line or page range, so provenance survives into the
answer ("report.md, lines 40-58"). Requirements, references to a topic and differences between two documents are
extracted without a model at all; a model is only used to summarise the selected parts.

Everything read here is external data: callers frame it before a model sees it.
"""

from __future__ import annotations

import ast
import csv
import difflib
import io
import json
import os
import re
import shutil
import subprocess
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from jarvis.platforms import hidden_window_kwargs
from jarvis.security.redaction import is_sensitive_key

_CODE_EXT = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
             ".java": "java", ".c": "c", ".h": "c", ".cpp": "c++", ".hpp": "c++", ".cs": "c#", ".go": "go",
             ".rs": "rust", ".rb": "ruby", ".php": "php", ".sh": "shell", ".ps1": "powershell", ".bat": "batch",
             ".sql": "sql", ".kt": "kotlin", ".swift": "swift", ".lua": "lua"}
_CONFIG_EXT = {".ini", ".cfg", ".conf", ".toml", ".yaml", ".yml", ".properties", ".env", ".gradle"}


@dataclass
class Section:
    title: str
    start: int                    # first line (1-based)
    end: int                      # last line
    level: int = 1
    page: int | None = None

    def where(self) -> str:
        if self.page:
            return f"page {self.page}"
        return f"line {self.start}" if self.start == self.end else f"lines {self.start}-{self.end}"

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "start": self.start, "end": self.end, "level": self.level, "page": self.page}


@dataclass
class DocumentExtract:
    name: str
    kind: str                     # markdown | code | json | csv | log | config | pdf | html | text
    text: str
    lines: list[str]
    sections: list[Section] = field(default_factory=list)
    structure: dict[str, Any] = field(default_factory=dict)
    language: str | None = None
    pages: int | None = None
    page_of_line: list[int] = field(default_factory=list)   # PDFs: page number per line
    title: str = ""
    notes: list[str] = field(default_factory=list)          # extraction caveats (basic PDF reader, truncation...)
    readable: bool = True

    @property
    def size_chars(self) -> int:
        return len(self.text)

    def where(self, line: int) -> str:
        if self.page_of_line and 0 < line <= len(self.page_of_line):
            return f"page {self.page_of_line[line - 1]}"
        return f"line {line}"

    def outline(self, limit: int = 12) -> list[str]:
        return [("  " * (s.level - 1)) + f"{s.title} ({s.where()})" for s in self.sections[:limit]]

    def summary_dict(self) -> dict[str, Any]:
        """What is kept on the observation (never the whole text)."""
        return {"kind": self.kind, "language": self.language, "title": self.title, "lines": len(self.lines),
                "chars": self.size_chars, "pages": self.pages, "sections": [s.to_dict() for s in self.sections[:60]],
                "structure": self.structure, "notes": self.notes, "readable": self.readable}


# -- reading ------------------------------------------------------------------------------------------------------

def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def read_document(data: bytes, name: str, mime: str = "") -> DocumentExtract:
    ext = os.path.splitext(name.lower())[1]
    if data.startswith(b"%PDF") or ext == ".pdf" or mime == "application/pdf":
        return _pdf(data, name)
    text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if ext in (".md", ".markdown"):
        doc = DocumentExtract(name, "markdown", text, lines, _markdown_sections(lines))
    elif ext == ".rst":
        doc = DocumentExtract(name, "text", text, lines, _rst_sections(lines))
    elif ext in _CODE_EXT:
        doc = _code(name, text, lines, _CODE_EXT[ext])
    elif ext == ".json" or mime == "application/json":
        doc = _json(name, text, lines)
    elif ext in (".csv", ".tsv"):
        doc = _csv(name, text, lines, "\t" if ext == ".tsv" else None)
    elif ext == ".log" or _looks_like_log(lines):
        doc = _log(name, text, lines)
    elif ext in _CONFIG_EXT or name.lower().startswith(".env"):
        doc = _config(name, text, lines)
    elif ext in (".html", ".htm", ".xml"):
        doc = _markup(name, text, lines)
    else:
        doc = DocumentExtract(name, "text", text, lines, _numbered_sections(lines) or _markdown_sections(lines))
    if not doc.title:
        doc.title = next((s.title for s in doc.sections if s.level == 1), "") or \
            next((l.strip()[:80] for l in lines if l.strip()), name)
    return doc


def _markdown_sections(lines: list[str]) -> list[Section]:
    sections: list[Section] = []
    in_code = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code = not in_code
        m = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if m and not in_code:
            if sections:
                sections[-1].end = i - 1
            sections.append(Section(m.group(2).strip(), i, len(lines), len(m.group(1))))
    return sections


def _rst_sections(lines: list[str]) -> list[Section]:
    sections: list[Section] = []
    levels: list[str] = []
    for i in range(1, len(lines)):
        under = lines[i].strip()
        if under and len(set(under)) == 1 and under[0] in "=-~^*#+" and len(under) >= len(lines[i - 1].strip()) > 0:
            char = under[0]
            if char not in levels:
                levels.append(char)
            if sections:
                sections[-1].end = i - 1
            sections.append(Section(lines[i - 1].strip(), i, len(lines), levels.index(char) + 1))
    return sections


_NUMBERED = re.compile(r"^\s*((\d+(\.\d+){0,3})\.?|[A-Z]\.|[IVX]+\.)\s+([A-Z][^.!?]{2,80})\s*$")


def _numbered_sections(lines: list[str]) -> list[Section]:
    """Headings in plain text: "1. Introduction", "2.3 Security", "REQUIREMENTS" (short, alone on a line)."""
    sections: list[Section] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        m = _NUMBERED.match(line)
        heading = m.group(4).strip() if m else None
        level = (m.group(2) or "").count(".") + 1 if m and m.group(2) else 1
        if not heading and 3 <= len(stripped) <= 60 and stripped.isupper() and re.search(r"[A-Z]{3}", stripped):
            heading, level = stripped.title(), 1
        if heading and (i == 1 or not lines[i - 2].strip() or m):
            if sections:
                sections[-1].end = i - 1
            sections.append(Section(heading, i, len(lines), level))
    return sections if len(sections) >= 2 else []


def _code(name: str, text: str, lines: list[str], language: str) -> DocumentExtract:
    sections: list[Section] = []
    structure: dict[str, Any] = {"language": language}
    if language == "python":
        try:
            tree = ast.parse(text)
            imports = sorted({(n.module or "") if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                              for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                              for a in (n.names if isinstance(n, ast.Import) else [n])} - {""})
            structure["imports"] = imports[:40]
            structure["docstring"] = (ast.get_docstring(tree) or "")[:300]
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "def"
                    sections.append(Section(f"{kind} {node.name}", node.lineno, getattr(node, "end_lineno", node.lineno),
                                            1))
                    if isinstance(node, ast.ClassDef):
                        for sub in node.body:
                            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                sections.append(Section(f"def {node.name}.{sub.name}", sub.lineno,
                                                        getattr(sub, "end_lineno", sub.lineno), 2))
        except SyntaxError as exc:
            structure["syntax_error"] = f"line {exc.lineno}: {exc.msg}"
    if not sections:
        pattern = re.compile(r"^\s*(export\s+)?(async\s+)?(public\s+|private\s+|protected\s+|static\s+)*"
                             r"(function\s+(\w+)|class\s+(\w+)|def\s+(\w+)|fn\s+(\w+)|func\s+(\w+)|"
                             r"interface\s+(\w+)|struct\s+(\w+)|(\w+)\s*=\s*(async\s*)?\([^)]*\)\s*=>)")
        for i, line in enumerate(lines, 1):
            m = pattern.match(line)
            if m:
                ident = next((g for g in m.groups()[4:12] if g), None)
                if ident:
                    sections.append(Section(line.strip()[:80], i, i, 1))
    structure["definitions"] = len(sections)
    return DocumentExtract(name, "code", text, lines, sections, structure, language)


def _json(name: str, text: str, lines: list[str]) -> DocumentExtract:
    structure: dict[str, Any] = {}
    try:
        data = json.loads(text)
        structure = {"type": type(data).__name__, "outline": _json_outline(data)}
    except ValueError as exc:
        structure = {"error": f"not valid JSON ({exc})"}
    return DocumentExtract(name, "json", text, lines, [], structure)


def _json_outline(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return type(value).__name__
    if isinstance(value, dict):
        return {k: ("[hidden]" if is_sensitive_key(str(k)) else _json_outline(v, depth + 1))
                for k, v in list(value.items())[:30]}
    if isinstance(value, list):
        return [f"{len(value)} item(s)"] + ([_json_outline(value[0], depth + 1)] if value else [])
    return type(value).__name__


def _csv(name: str, text: str, lines: list[str], delimiter: str | None) -> DocumentExtract:
    try:
        dialect = csv.Sniffer().sniff(text[:4096]) if delimiter is None else None
    except csv.Error:
        dialect = None
    reader = csv.reader(io.StringIO(text), delimiter=delimiter or (dialect.delimiter if dialect else ","))
    rows = [r for r in reader if r]
    header = rows[0] if rows else []
    body = rows[1:]
    columns = []
    for c, title in enumerate(header[:40]):
        values = [r[c] for r in body if c < len(r) and r[c].strip()]
        numbers = []
        for v in values:
            try:
                numbers.append(float(v.replace(",", "")))
            except ValueError:
                pass
        col: dict[str, Any] = {"name": title, "filled": len(values)}
        if numbers and len(numbers) >= len(values) * 0.8:
            col.update({"numeric": True, "min": min(numbers), "max": max(numbers),
                        "mean": round(sum(numbers) / len(numbers), 4)})
        else:
            col["examples"] = list(dict.fromkeys(values))[:3]
        columns.append(col)
    return DocumentExtract(name, "csv", text, lines, [], {"rows": len(body), "columns": columns})


_LOG_LEVEL = re.compile(r"\b(ERROR|FATAL|CRITICAL|WARN(ING)?|EXCEPTION|Traceback|panic:|Unhandled|FAILED)\b")


def _looks_like_log(lines: list[str]) -> bool:
    sample = [l for l in lines[:200] if l.strip()]
    if len(sample) < 5:
        return False
    stamped = sum(1 for l in sample if re.match(r"^\s*\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|^\s*\d{2}:\d{2}:\d{2}", l))
    return stamped >= len(sample) * 0.5


def _log(name: str, text: str, lines: list[str]) -> DocumentExtract:
    levels: Counter[str] = Counter()
    problems: list[dict[str, Any]] = []
    for i, line in enumerate(lines, 1):
        m = _LOG_LEVEL.search(line)
        if m:
            level = m.group(1).upper().replace("WARNING", "WARN")
            levels[level] += 1
            if level != "WARN" and len(problems) < 40:
                problems.append({"line": i, "text": line.strip()[:240]})
    stamps = [m.group(0) for l in lines if (m := re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", l))]
    structure = {"levels": dict(levels), "problems": problems, "first": stamps[0] if stamps else None,
                 "last": stamps[-1] if stamps else None}
    sections = [Section(p["text"][:80], p["line"], p["line"], 1) for p in problems[:30]]
    return DocumentExtract(name, "log", text, lines, sections, structure)


def _config(name: str, text: str, lines: list[str]) -> DocumentExtract:
    sections: list[Section] = []
    keys: list[str] = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        m = re.match(r"^\[+([^\]]+)\]+$", s)
        if m:
            if sections:
                sections[-1].end = i - 1
            sections.append(Section(m.group(1), i, len(lines), 1))
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*[:=]", s)
        if m and not s.startswith("#"):
            keys.append(m.group(1))
    structure = {"keys": keys[:60], "sensitive_keys": [k for k in keys if is_sensitive_key(k)][:20]}
    return DocumentExtract(name, "config", text, lines, sections, structure)


def _markup(name: str, text: str, lines: list[str]) -> DocumentExtract:
    sections = []
    for i, line in enumerate(lines, 1):
        m = re.search(r"<h([1-6])[^>]*>(.*?)</h\1>", line, re.I)
        if m:
            sections.append(Section(re.sub(r"<[^>]+>", "", m.group(2)).strip()[:80], i, i, int(m.group(1))))
    title = re.search(r"<title>(.*?)</title>", text, re.I | re.S)
    return DocumentExtract(name, "html", text, lines, sections, {}, title=(title.group(1).strip()[:80] if title else ""))


# -- PDF ------------------------------------------------------------------------------------------------------------

def pdf_backend() -> str:
    try:
        import importlib.util
        if importlib.util.find_spec("pypdf") is not None:
            return "pypdf"
    except Exception:
        pass
    if shutil.which("pdftotext"):
        return "pdftotext"
    return "basic"


def _pdf(data: bytes, name: str) -> DocumentExtract:
    backend = pdf_backend()
    pages: list[str] = []
    notes: list[str] = []
    if backend == "pypdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                return DocumentExtract(name, "pdf", "", [], notes=["the PDF is password-protected"], readable=False)
            pages = [(p.extract_text() or "") for p in reader.pages]
        except Exception as exc:
            notes.append(f"pypdf couldn't read it ({exc}); used the basic reader")
            backend = "basic"
    if backend == "pdftotext":
        try:
            proc = subprocess.run(["pdftotext", "-layout", "-", "-"], input=data, capture_output=True, timeout=60,
                                  **hidden_window_kwargs())
            pages = proc.stdout.decode("utf-8", errors="replace").split("\f")
        except Exception as exc:
            notes.append(f"pdftotext failed ({exc}); used the basic reader")
            backend = "basic"
    if backend == "basic":
        pages, count = _basic_pdf_text(data)
        notes.append("read with JARVIS's basic PDF reader: complex layouts or fonts can come out incomplete "
                     "(py -m pip install pypdf reads PDFs properly)")
        if not pages:
            pages = [""] * max(count, 1)
    lines: list[str] = []
    page_of_line: list[int] = []
    for number, page in enumerate(pages, 1):
        for line in page.replace("\r", "").split("\n"):
            lines.append(line)
            page_of_line.append(number)
    text = "\n".join(lines)
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    readable = bool(text.strip()) and printable / max(1, len(text)) > 0.9
    if not readable:
        notes.append("no readable text came out: it may be scanned pages (images), which need OCR")
    sections = [Section(f"Page {n}", page_of_line.index(n) + 1,
                        len(page_of_line) - page_of_line[::-1].index(n), 1, page=n)
                for n in range(1, len(pages) + 1) if n in page_of_line]
    headed = _numbered_sections(lines)
    for s in headed:
        s.page = page_of_line[s.start - 1] if s.start - 1 < len(page_of_line) else None
    doc = DocumentExtract(name, "pdf", text, lines, headed or sections, {"backend": backend}, pages=len(pages),
                          page_of_line=page_of_line, notes=notes, readable=readable)
    return doc


def _basic_pdf_text(data: bytes) -> tuple[list[str], int]:
    """Text from simple PDFs: Flate streams, text operators (Tj, TJ, ', "), one entry per page-ish content stream."""
    count = len(re.findall(rb"/Type\s*/Page[^s]", data))
    pages: list[str] = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = m.group(1)
        header = data[max(0, m.start() - 300):m.start()]
        if b"/FlateDecode" in header:
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                continue
        if b"BT" not in raw or (b"/Subtype" in header and b"/Image" in header):
            continue
        texts: list[str] = []
        for block in re.findall(rb"BT(.*?)ET", raw, re.S):
            line: list[str] = []
            for token in re.finditer(rb"\((?:\\.|[^\\)])*\)|\[(.*?)\]\s*TJ|(T\*|Td|TD|Tm|')", block, re.S):
                t = token.group(0)
                if t.startswith(b"("):
                    line.append(_pdf_string(t[1:-1]))
                elif t.startswith(b"["):
                    line.append("".join(_pdf_string(s[1:-1]) for s in re.findall(rb"\((?:\\.|[^\\)])*\)", t)))
                elif line:
                    texts.append("".join(line))
                    line = []
            if line:
                texts.append("".join(line))
        page = "\n".join(t for t in texts if t.strip())
        if page.strip():
            pages.append(page)
    return pages, count


def _pdf_string(raw: bytes) -> str:
    out = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == 0x5C and i + 1 < len(raw):
            n = raw[i + 1]
            esc = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12, ord("("): 40, ord(")"): 41,
                   0x5C: 0x5C}
            if n in esc:
                out.append(esc[n])
                i += 2
                continue
            m = re.match(rb"[0-7]{1,3}", raw[i + 1:i + 4])
            if m:
                out.append(int(m.group(0), 8) & 0xFF)
                i += 1 + len(m.group(0))
                continue
            i += 1
            continue
        out.append(c)
        i += 1
    return out.decode("latin-1")


# -- asking questions of a document -------------------------------------------------------------------------------

_STOP = set("the a an and or of to in on for with about what which who is are was were be it this that these those "
            "please find show tell me my our from by at as how why when where do does document file text".split())


def terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9_\-]+", text.lower()) if w not in _STOP and len(w) > 2]


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    title: str = ""
    score: float = 0.0

    def where(self, doc: DocumentExtract) -> str:
        if doc.page_of_line:
            a, b = doc.where(self.start), doc.where(self.end)
            return a if a == b else f"{a}-{b.split()[-1]}"
        return f"line {self.start}" if self.start == self.end else f"lines {self.start}-{self.end}"


def chunks(doc: DocumentExtract, *, size: int = 40) -> list[Chunk]:
    """Sections, or fixed windows of lines when the document has no structure."""
    out: list[Chunk] = []
    if doc.sections and doc.kind not in ("log",):
        for s in doc.sections:
            end = min(s.end, s.start + 120)
            out.append(Chunk("\n".join(doc.lines[s.start - 1:end]), s.start, end, s.title))
    if not out:
        for start in range(0, len(doc.lines), size):
            out.append(Chunk("\n".join(doc.lines[start:start + size]), start + 1, min(len(doc.lines), start + size)))
    return out


def relevant(doc: DocumentExtract, question: str, *, max_chars: int = 12000) -> list[Chunk]:
    """The parts of a document worth showing a model for this question, best first, within a character budget."""
    wanted = terms(question)
    pieces = chunks(doc)
    if not wanted:
        return _budget(pieces, max_chars)
    df = Counter(t for c in pieces for t in set(terms(c.text + " " + c.title)))
    n = len(pieces)
    for c in pieces:
        words = Counter(terms(c.text + " " + c.title * 3))
        c.score = sum(words[t] ** 0.5 * (1 + (n / (1 + df[t])) ** 0.5) for t in wanted if words[t])
    ranked = sorted([c for c in pieces if c.score > 0], key=lambda c: -c.score) or pieces[:3]
    return _budget(ranked, max_chars)


def _budget(pieces: list[Chunk], max_chars: int) -> list[Chunk]:
    out, used = [], 0
    for c in pieces:
        if used >= max_chars:
            break
        text = c.text[: max(0, max_chars - used)]
        out.append(Chunk(text, c.start, c.end, c.title, c.score))
        used += len(text)
    return out


_REQUIREMENT = re.compile(r"\b(must\s+not|shall\s+not|may\s+not|should\s+not|shall|must|is\s+required\s+to|"
                          r"are\s+required\s+to|required|should|needs?\s+to|has\s+to|have\s+to|will\s+provide)\b",
                          re.I)
_REQ_ID = re.compile(r"\b(REQ|FR|NFR|SR|UR|R)[-_ ]?\d+(\.\d+)*\b")


def requirements(doc: DocumentExtract, *, limit: int = 60) -> list[dict[str, Any]]:
    """Statements that impose requirements, with where they are and how strong they are (deterministic)."""
    out: list[dict[str, Any]] = []
    in_code = False
    for i, line in enumerate(doc.lines, 1):
        text = line.strip()
        if text.startswith("```"):
            in_code = not in_code
        if in_code or len(text) < 12:
            continue
        rid = _REQ_ID.search(text)
        m = _REQUIREMENT.search(text)
        checkbox = re.match(r"^[-*]\s+\[[ xX]\]\s+", text)
        if not (m or rid or checkbox):
            continue
        word = " ".join((m.group(1).lower() if m else "").split())
        if word in ("must not", "shall not", "may not"):
            strength = "must not"
        elif word == "should not":
            strength = "should not"
        elif any(w in word for w in ("shall", "must", "required", "has to", "have to", "need")):
            strength = "must"
        elif "should" in word:
            strength = "should"
        else:
            strength = "listed"
        out.append({"text": re.sub(r"^[-*#>\s]+", "", text)[:300], "line": i, "where": doc.where(i),
                    "strength": strength, "id": rid.group(0) if rid else None})
        if len(out) >= limit:
            break
    return out


def references(doc: DocumentExtract, topic: str, *, limit: int = 40) -> list[dict[str, Any]]:
    """Every line mentioning the topic (all its words, any order), with a line of context either side."""
    wanted = terms(topic) or [topic.lower().strip()]
    stems = [w[:6] if len(w) > 6 else w for w in wanted]
    out = []
    for i, line in enumerate(doc.lines, 1):
        low = line.lower()
        if all(s in low for s in stems):
            before = doc.lines[i - 2].strip() if i > 1 else ""
            after = doc.lines[i].strip() if i < len(doc.lines) else ""
            out.append({"line": i, "where": doc.where(i), "text": line.strip()[:300], "before": before[:160],
                        "after": after[:160]})
            if len(out) >= limit:
                break
    return out


def compare(a: DocumentExtract, b: DocumentExtract, *, limit: int = 40) -> dict[str, Any]:
    """What changed between two documents: added/removed lines (with line numbers) and which sections changed."""
    matcher = difflib.SequenceMatcher(a=a.lines, b=b.lines, autojunk=False)
    added, removed, changes = 0, 0, []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed += i2 - i1
        added += j2 - j1
        if len(changes) < limit:
            changes.append({"type": {"replace": "changed", "delete": "removed", "insert": "added"}[tag],
                            "old_lines": [i1 + 1, i2] if i2 > i1 else None, "new_lines": [j1 + 1, j2] if j2 > j1 else None,
                            "old": "\n".join(a.lines[i1:i2])[:400], "new": "\n".join(b.lines[j1:j2])[:400],
                            "section": _section_at(b, j1 + 1) or _section_at(a, i1 + 1)})
    titles_a = {s.title for s in a.sections}
    titles_b = {s.title for s in b.sections}
    return {"added_lines": added, "removed_lines": removed, "similarity": round(matcher.ratio(), 3),
            "sections_added": sorted(titles_b - titles_a)[:20], "sections_removed": sorted(titles_a - titles_b)[:20],
            "changed_sections": list(dict.fromkeys(c["section"] for c in changes if c["section"]))[:20],
            "changes": changes, "identical": added == 0 and removed == 0}


def _section_at(doc: DocumentExtract, line: int) -> str | None:
    best = None
    for s in doc.sections:
        if s.start <= line <= s.end:
            best = s.title
    return best


def outline_summary(doc: DocumentExtract, *, max_sections: int = 10) -> str:
    """A factual, model-free overview: what it is, its size and structure, and the start of each section."""
    head = f"{doc.name}: {doc.kind}" + (f" ({doc.language})" if doc.language and doc.kind == "code" else "") + \
        f", {len(doc.lines):,} lines" + (f", {doc.pages} pages" if doc.pages else "")
    parts = [head + "."]
    if doc.kind == "csv":
        cols = doc.structure.get("columns", [])
        parts.append(f"{doc.structure.get('rows', 0):,} rows; columns: " +
                     ", ".join(c["name"] + (f" ({c['min']:g}–{c['max']:g})" if c.get("numeric") else "")
                               for c in cols[:12]) + ".")
    elif doc.kind == "json":
        outline = doc.structure.get("outline")
        if isinstance(outline, dict):
            parts.append("Top-level keys: " + ", ".join(list(outline)[:15]) + ".")
        elif doc.structure.get("error"):
            parts.append(doc.structure["error"].capitalize() + ".")
    elif doc.kind == "log":
        levels = doc.structure.get("levels", {})
        if levels:
            parts.append("Messages by level: " + ", ".join(f"{k} {v}" for k, v in levels.items()) + ".")
        for p in doc.structure.get("problems", [])[:5]:
            parts.append(f"line {p['line']}: {p['text'][:160]}")
    elif doc.kind == "code":
        if doc.structure.get("docstring"):
            parts.append(doc.structure["docstring"].split("\n")[0][:200])
        if doc.sections:
            parts.append("Defines: " + ", ".join(s.title for s in doc.sections if s.level == 1)[:400] + ".")
        if doc.structure.get("syntax_error"):
            parts.append(f"Syntax error at {doc.structure['syntax_error']}.")
    else:
        for s in doc.sections[:max_sections]:
            body = " ".join(l.strip() for l in doc.lines[s.start:min(s.end, s.start + 6)] if l.strip())
            first = re.split(r"(?<=[.!?])\s", body)[0][:200] if body else ""
            parts.append(f"{s.title} ({s.where()})" + (f": {first}" if first else ""))
        if not doc.sections:
            body = " ".join(l.strip() for l in doc.lines[:12] if l.strip())
            parts.append(body[:400])
    for note in doc.notes:
        parts.append(f"(Note: {note}.)")
    return "\n".join(parts)
