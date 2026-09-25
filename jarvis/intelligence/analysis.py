"""Deterministic analysis and verification (Phase 3 §14-19, §35-36).

Analyzers turn gathered evidence into findings and candidate actions; comparators check a result against
fresh, independent measurements. Both are plain functions of their inputs: no model, no guessing, and every
finding says whether it was observed or inferred. A candidate action records why it was proposed, what it is
expected to change, how confident the analysis is, and whether it is ruled out (protected process, a user
constraint, or an approach that already failed before).
"""

from __future__ import annotations

import csv
import io
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jarvis.intelligence.goals import Constraint
from jarvis.intelligence.plans import Quality

CPU_HIGH = 85.0
MEMORY_HIGH = 85.0
DISK_HIGH = 90.0
PROCESS_CPU_HOG = 50.0          # percent of one core
PROCESS_MEMORY_HOG = 15.0       # percent of RAM

# never proposed for stopping, whatever they use: the OS, the desktop, security software, JARVIS and the
# model runtime (idle models are unloaded instead)
PROTECTED_PROCESSES = {
    "system", "system idle process", "idle", "registry", "memory compression", "secure system", "smss.exe",
    "csrss.exe", "wininit.exe", "winlogon.exe", "services.exe", "lsass.exe", "svchost.exe", "dwm.exe",
    "explorer.exe", "fontdrvhost.exe", "msmpeng.exe", "searchindexer.exe", "sihost.exe", "ctfmon.exe",
    "systemd", "init", "kthreadd", "launchd", "kernel_task", "windowserver", "loginwindow", "xorg", "xwayland",
    "gnome-shell", "kwin_x11", "kwin_wayland", "plasmashell", "sshd", "dbus-daemon", "pipewire", "pulseaudio",
    "ollama", "ollama.exe", "ollama_llama_server", "ollama_llama_server.exe", "ollama app.exe",
}


def jarvis_pids() -> set[int]:
    """This process and the processes it belongs to (the terminal or service that started it)."""
    pids = {os.getpid()}
    try:
        import psutil
        me = psutil.Process()
        pids.update(p.pid for p in me.parents()[:3])
        pids.update(c.pid for c in me.children(recursive=True))
    except Exception:
        pass
    return pids


# -- process tables -----------------------------------------------------------------------------------------

def processes(data: Any) -> list[dict[str, Any]]:
    """Process rows from process_list output, or parsed from a fallback command (`ps`, `tasklist`)."""
    if isinstance(data, dict) and isinstance(data.get("processes"), list):
        return [p for p in data["processes"] if isinstance(p, dict)]
    stdout = data.get("stdout") if isinstance(data, dict) else data if isinstance(data, str) else ""
    return parse_process_table(stdout or "")


def parse_process_table(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return rows
    if lines[0].startswith('"'):                    # tasklist /fo csv /nh: "name","pid","session","#","12,345 K"
        for rec in csv.reader(io.StringIO("\n".join(lines))):
            if len(rec) >= 5 and rec[1].isdigit():
                mem_kb = float(re.sub(r"[^\d]", "", rec[4]) or 0)
                rows.append({"pid": int(rec[1]), "name": rec[0], "cpu_percent": None, "memory_kb": mem_kb})
        return rows
    header = lines[0].lower().split()
    if "pid" in header:                             # ps -eo pid,pcpu,pmem,comm
        idx = {name: i for i, name in enumerate(header)}
        for ln in lines[1:]:
            parts = ln.split(None, len(header) - 1)
            if len(parts) < len(header):
                continue
            try:
                row = {"pid": int(parts[idx["pid"]]), "name": os.path.basename(parts[-1].strip())}
                if "%cpu" in idx:
                    row["cpu_percent"] = float(parts[idx["%cpu"]])
                if "%mem" in idx:
                    row["memory_percent"] = float(parts[idx["%mem"]])
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _history_note(key: str, history: list[dict[str, Any]]) -> tuple[float, str]:
    """Memory-aware planning: how earlier attempts of the same action went (multiplier, note)."""
    for entry in history:
        if entry.get("key") != key:
            continue
        outcome = entry.get("outcome", "")
        when = entry.get("when", "earlier")
        if outcome in ("failed", "no_effect", "reverted"):
            return 0.3, f"we already tried this {when} and it didn't help"
        if outcome in ("helped", "success", "verified"):
            return 1.15, f"this helped {when}"
    return 1.0, ""


def _constraint_block(tool: str, args: dict[str, Any], meta: dict[str, Any], level: int,
                      constraints: list[dict[str, Any]]) -> str | None:
    for raw in constraints:
        why = Constraint.from_dict(raw).forbids(tool, args, meta, level=level)
        if why:
            return why
    return None


# -- performance ------------------------------------------------------------------------------------------------

def analyze_performance(evidence: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    system = evidence.get("system") or {}
    by_cpu = processes(evidence.get("by_cpu"))
    by_memory = processes(evidence.get("by_memory")) or by_cpu
    models = evidence.get("models") or {}
    constraints = context.get("constraints") or []
    history = context.get("history") or []
    mine = set(context.get("jarvis_pids") or []) | jarvis_pids()
    ncpu = int(_num(system.get("cpu_count")) or os.cpu_count() or 1)
    cpu, mem = _num(system.get("cpu_percent")), _num(system.get("memory_percent"))
    swap, disk = _num(system.get("swap_percent")), _num(system.get("disk_percent"))
    findings: list[dict[str, Any]] = []
    bottlenecks: list[str] = []
    notes: list[str] = []

    def find(text: str, how: str = "observed", **ev: Any) -> None:
        findings.append({"text": text, "confidence": how, "evidence": ev})

    if cpu is None and mem is None:
        find("no system measurements were available", "unknown")
    if cpu is not None:
        if cpu >= CPU_HIGH:
            bottlenecks.append("cpu")
            find(f"CPU usage is {cpu:.0f}%", cpu=cpu)
    if mem is not None and mem >= MEMORY_HIGH:
        bottlenecks.append("memory")
        find(f"memory usage is {mem:.0f}%", memory=mem)
    if swap is not None and swap >= 50 and (mem or 0) >= 70:
        bottlenecks.append("swap")
        find(f"swap is {swap:.0f}% used, so memory is being paged to disk", swap=swap)
    if disk is not None and disk >= 95:
        bottlenecks.append("disk")
        find(f"the disk is {disk:.0f}% full", disk=disk)
    vram_used, vram_total = _num(system.get("vram_used_gb")), _num(system.get("vram_total_gb"))
    if vram_used is not None and vram_total:
        if vram_used / vram_total * 100 >= 90:
            bottlenecks.append("gpu_memory")
            find(f"GPU memory is {vram_used / vram_total * 100:.0f}% allocated", vram_used_gb=vram_used)
    if not any(p.get("cpu_percent") is not None for p in by_cpu) and by_cpu:
        notes.append("per-process CPU use wasn't available from the fallback process list")

    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()

    def consider(p: dict[str, Any], kind: str) -> None:
        pid = p.get("pid")
        name = str(p.get("name") or "?")
        if not isinstance(pid, int) or pid in seen:
            return
        seen.add(pid)
        pcpu = _num(p.get("cpu_percent")) or 0.0
        pmem = _num(p.get("memory_percent")) or 0.0
        share = pcpu / ncpu
        key = f"process_stop:{name.lower()}"
        meta = {"target": name, "pid": pid, "cpu": round(pcpu, 1), "memory": round(pmem, 1)}
        args = {"pid": pid}
        blocked = None
        if pid in mine or pid in (0, 1, 4):
            blocked = "it is JARVIS itself or a core system process"
        elif name.lower() in PROTECTED_PROCESSES:
            blocked = f"{name} is part of the operating system, the desktop or the model runtime"
        else:
            blocked = _constraint_block("process_stop", args, meta, 4, constraints)
        if kind == "cpu":
            base = 0.45 + min(0.45, share / 100 * 1.5) + (0.1 if "cpu" in bottlenecks else -0.15)
            reason = f"{name} (PID {pid}) was using {pcpu:.0f}% of a CPU core ({share:.0f}% of the whole machine)"
            expected = f"CPU use should drop by about {share:.0f} points"
        else:
            base = 0.4 + min(0.45, pmem / 100 * 1.5) + (0.1 if "memory" in bottlenecks else -0.15)
            reason = f"{name} (PID {pid}) was using {pmem:.0f}% of memory"
            expected = f"memory use should drop by about {pmem:.0f} points"
        factor, note = _history_note(key, history)
        confidence = round(max(0.05, min(0.95, base * factor)), 2)
        candidates.append({"key": key, "kind": kind, "tool": "process_stop", "args": args,
                           "title": f"stop {name} (PID {pid})", "meta": meta, "reason": reason, "expected": expected,
                           "confidence": confidence, "level": 4, "reversible": False,
                           "rollback": f"start {name} again yourself if you need it", "blocked": blocked,
                           "history": note})
        find(reason, "observed", pid=pid, cpu=pcpu, memory=pmem)

    for p in by_cpu[:6]:
        if (_num(p.get("cpu_percent")) or 0) >= PROCESS_CPU_HOG or ((_num(p.get("cpu_percent")) or 0) / ncpu >= 20):
            consider(p, "cpu")
    for p in by_memory[:6]:
        if (_num(p.get("memory_percent")) or 0) >= PROCESS_MEMORY_HOG:
            consider(p, "memory")

    pressure = bool({"memory", "swap", "gpu_memory"} & set(bottlenecks))
    for name in models.get("loaded") or []:
        key = f"model_unload:{name}"
        factor, note = _history_note(key, history)
        in_use = name in (models.get("in_use") or [])
        base = (0.75 if pressure else 0.3) * factor
        candidates.append({
            "key": key, "kind": "memory", "tool": "model_unload", "args": {"model": name},
            "title": f"unload the idle model {name}", "meta": {"target": name},
            "reason": f"the language model {name} is loaded and holds memory" + (" while memory is short" if pressure
                                                                                   else ""),
            "expected": "its memory is released; it loads again automatically the next time it is needed",
            "confidence": round(min(0.95, base), 2), "level": 3, "reversible": True,
            "rollback": "it reloads automatically when next needed",
            "blocked": "a request is using it right now" if in_use else None, "history": note})

    for c in candidates:        # a candidate that doesn't address an actual bottleneck is only a suggestion
        if c["kind"] not in bottlenecks and not (c["kind"] == "memory" and pressure):
            c["confidence"] = round(min(c["confidence"], 0.45), 2)
    candidates.sort(key=lambda c: (c["blocked"] is not None, -c["confidence"]))
    usable = [c for c in candidates if not c["blocked"]]
    if bottlenecks and usable:
        top = usable[0]
        cause = f"{top['reason']}" + (f", while {_join([_LABEL.get(b, b) for b in bottlenecks])} "
                                      f"{'is' if len(bottlenecks) == 1 else 'are'} under pressure" if bottlenecks else "")
        confidence = top["confidence"]
    elif bottlenecks:
        cause = f"{_join([_LABEL.get(b, b) for b in bottlenecks])} {'is' if len(bottlenecks) == 1 else 'are'} under " \
                "pressure, but nothing I could safely change stands out"
        confidence = 0.4
    else:
        measured = ", ".join(f for f in (f"CPU {cpu:.0f}%" if cpu is not None else "",
                                         f"memory {mem:.0f}%" if mem is not None else "") if f)
        cause = f"nothing is overloaded right now ({measured}); the slowness may come and go"
        confidence = 0.3
    recommendations = []
    if not bottlenecks:
        recommendations.append("keep an eye on it: say 'watch my CPU' and I'll tell you when it spikes")
    if "disk" in bottlenecks:
        recommendations.append("free some disk space ('free up disk space')")
    for c in usable[:3]:
        recommendations.append(c["title"] + f" ({c['expected']})")
    for c in [c for c in candidates if c["blocked"]][:2]:
        notes.append(f"not suggesting {c['title']}: {c['blocked']}")
    return {"bottlenecks": bottlenecks, "findings": findings, "candidates": candidates, "cause": cause,
            "confidence": confidence, "recommendations": recommendations, "notes": notes,
            "measured": {"cpu_percent": cpu, "memory_percent": mem, "swap_percent": swap, "disk_percent": disk}}


_LABEL = {"cpu": "the CPU", "memory": "memory", "swap": "swap", "disk": "the disk", "gpu_memory": "GPU memory"}


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# -- disk ------------------------------------------------------------------------------------------------------------

def analyze_disk(evidence: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    usage = evidence.get("usage") or {}
    extra = [u for u in (evidence.get("extra") or []) if isinstance(u, dict)]
    constraints = context.get("constraints") or []
    history = context.get("history") or []
    findings: list[dict[str, Any]] = []
    percent = _num(usage.get("percent"))
    free = _num(usage.get("free"))
    if percent is not None:
        findings.append({"text": f"the disk holding {usage.get('path')} is {percent:.0f}% full "
                                 f"({_size(free)} free)", "confidence": "observed", "evidence": {"percent": percent}})
    children = sorted([c for u in [usage] + extra for c in u.get("children") or []],
                      key=lambda c: -(c.get("size") or 0))
    for c in children[:5]:
        findings.append({"text": f"{c['path']} uses {_size(c.get('size'))}", "confidence": "observed",
                         "evidence": {"size": c.get("size")}})
    candidates = []
    files = sorted([f for u in [usage] + extra for f in u.get("largest_files") or []], key=lambda f: -(f.get("size") or 0))
    for f in files[:8]:
        path, size = f["path"], f.get("size") or 0
        key = f"file_delete:{path}"
        args = {"path": path}
        blocked = _constraint_block("file_delete", args, {"target": path}, 4, constraints)
        factor, note = _history_note(key, history)
        candidates.append({"key": key, "kind": "disk", "tool": "file_delete", "args": args,
                           "title": f"move {os.path.basename(path)} ({_size(size)}) to JARVIS's trash",
                           "meta": {"target": path, "size": size}, "reason": f"{path} is {_size(size)} and hasn't "
                           f"been changed in {int(f.get('age_days') or 0)} days",
                           "expected": f"frees about {_size(size)} once the trash is emptied", "confidence":
                           round(min(0.9, 0.6 * factor), 2), "level": 4, "reversible": True,
                           "rollback": "restore it from JARVIS's trash", "blocked": blocked, "history": note,
                           "size": size})
    usable = [c for c in candidates if not c["blocked"]]
    total = sum(c["size"] for c in usable)
    recommendations = [c["title"] for c in usable[:5]]
    if percent is not None and percent < DISK_HIGH and not usable:
        cause = f"the disk is {percent:.0f}% full, which is not a problem yet"
    elif usable:
        cause = f"{len(usable)} large, old file(s) could free about {_size(total)}"
    else:
        cause = "nothing safe to remove stands out"
    notes = [f"not suggesting {c['title']}: {c['blocked']}" for c in candidates if c["blocked"]][:3]
    return {"findings": findings, "candidates": candidates, "cause": cause, "recommendations": recommendations,
            "confidence": 0.7 if usable else 0.4, "notes": notes, "bottlenecks": ["disk"] if (percent or 0) >= DISK_HIGH
            else [], "measured": {"disk_percent": percent, "free": free}}


# -- research ---------------------------------------------------------------------------------------------------

_STOP = set("the a an and or of to in on for with about what my our is are was were be it this that these those "
            "notes note docs documents files say says from into by at as how why when where which who do does "
            "research find out summarise summarize compile report everything all".split())


def keywords(topic: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9_-]+", topic.lower()) if w not in _STOP]
    return list(dict.fromkeys(words))[:6]


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9-]+", text.lower()) if w not in _STOP and len(w) > 2}


def analyze_research(evidence: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Claims about the topic, each traced to the file and line it came from, cross-checked across sources."""
    topic = str(context.get("topic") or "")
    keys = context.get("keywords") or keywords(topic)
    documents = [d for d in (evidence.get("documents") or []) if isinstance(d, dict) and d.get("content")]
    claims: list[dict[str, Any]] = []
    for doc in documents:
        path = str(doc.get("path"))
        for lineno, line in enumerate(str(doc["content"]).splitlines(), 1):
            for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
                sentence = sentence.strip(" -*#>\t")
                if len(sentence) < 12:
                    continue
                hits = [k for k in keys if k in sentence.lower()]
                if not hits:
                    continue
                claims.append({"text": sentence[:300], "quote": sentence[:300], "source": path, "line": lineno,
                               "keywords": hits, "score": len(hits)})
    # agent findings are evidence too, but only count once a source is attached (checked by citation verification)
    for finding in evidence.get("agent_claims") or []:
        if isinstance(finding, dict) and finding.get("source") and finding.get("quote"):
            claims.append({"text": str(finding.get("text") or finding["quote"])[:300], "quote": str(finding["quote"]),
                           "source": str(finding["source"]), "line": finding.get("line"), "keywords": [],
                           "score": 1, "by": finding.get("by", "agent")})
    claims.sort(key=lambda c: -c["score"])
    claims = claims[:40]
    # cross-check: corroboration across files, and numeric disagreements about the same thing
    for c in claims:
        words = _content_words(c["text"])
        c["_words"] = words
    for c in claims:
        support = {c["source"]}
        for other in claims:
            if other is c or other["source"] == c["source"]:
                continue
            union = c["_words"] | other["_words"]
            if union and len(c["_words"] & other["_words"]) / len(union) >= 0.5:
                support.add(other["source"])
        c["sources"] = sorted(support)
        c["support"] = len(support)
    conflicts = []
    for i, a in enumerate(claims):
        for b in claims[i + 1:]:
            if a["source"] == b["source"]:
                continue
            na, nb = re.findall(r"\b\d+(?:\.\d+)?\b", a["text"]), re.findall(r"\b\d+(?:\.\d+)?\b", b["text"])
            if not na or not nb or set(na) == set(nb):
                continue
            wa = {w for w in a["_words"] if not w.isdigit()}
            wb = {w for w in b["_words"] if not w.isdigit()}
            union = wa | wb
            if union and len(wa & wb) / len(union) >= 0.5:
                conflicts.append({"a": {k: a[k] for k in ("text", "source", "line")},
                                  "b": {k: b[k] for k in ("text", "source", "line")}})
                a["conflict"] = b["conflict"] = True
    for c in claims:
        c.pop("_words", None)
        c["confidence"] = "conflicting" if c.get("conflict") else "corroborated" if c["support"] > 1 else "single source"
    sources = sorted({c["source"] for c in claims})
    findings = [{"text": f"{len(claims)} statement(s) about {topic or 'the topic'} in {len(sources)} source(s)",
                 "confidence": "observed", "evidence": {"sources": sources}}]
    if conflicts:
        findings.append({"text": f"{len(conflicts)} disagreement(s) between sources", "confidence": "observed",
                         "evidence": {"conflicts": len(conflicts)}})
    if not claims:
        cause = f"I found nothing about {topic or 'that'} in the sources I searched"
    else:
        cause = f"{len(claims)} statement(s) from {len(sources)} source(s)"
    return {"claims": claims, "conflicts": conflicts, "sources": sources, "findings": findings, "cause": cause,
            "candidates": [], "recommendations": [], "confidence": 0.8 if claims and not conflicts else 0.4,
            "notes": [], "bottlenecks": []}


ANALYZERS = {"performance": analyze_performance, "disk": analyze_disk, "research": analyze_research}


# -- verification ------------------------------------------------------------------------------------------------

def verify_performance(args: dict[str, Any]) -> dict[str, Any]:
    """Compare fresh measurements with the ones taken before the change. The target processes must be gone and
    the bottleneck must have eased; a claimed success contradicted by the measurements is CONFLICTING."""
    before = args.get("before") or {}
    after = args.get("after") or {}
    after_procs = processes(args.get("after_processes"))
    targets = args.get("targets") or []
    claimed = args.get("claimed") or {}
    checks: list[dict[str, Any]] = []
    try:
        import psutil
        alive = lambda pid: psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE  # noqa
    except Exception:
        alive = lambda pid: any(p.get("pid") == pid for p in after_procs)  # noqa: E731
    still_running, gone = [], []
    for t in targets:
        if t.get("tool") == "process_stop" and isinstance(t.get("pid"), int):
            try:
                running = alive(t["pid"])
            except Exception:
                running = any(p.get("pid") == t["pid"] for p in after_procs)
            (still_running if running else gone).append(t)
            checks.append({"check": f"{t.get('target')} (PID {t['pid']}) stopped", "passed": not running})
        elif t.get("tool") == "model_unload":
            loaded = set((args.get("after_models") or {}).get("loaded") or [])
            ok = t.get("target") not in loaded
            (gone if ok else still_running).append(t)
            checks.append({"check": f"model {t.get('target')} unloaded", "passed": ok})
    measures = {}
    eased = []
    for key, limit in (("cpu_percent", CPU_HIGH), ("memory_percent", MEMORY_HIGH)):
        b, a = _num(before.get(key)), _num(after.get(key))
        if b is None or a is None:
            continue
        measures[key] = {"before": b, "after": a}
        if b >= limit:
            ok = a < limit or (b - a) >= 10
            eased.append(ok)
            label = "CPU" if key == "cpu_percent" else "memory"
            checks.append({"check": f"{label} {'eased' if ok else 'still high'}: {b:.0f}% → {a:.0f}%", "passed": ok})
    claimed_ok = [t for t in still_running if claimed.get(t.get("node"), {}).get("ok")]
    if claimed_ok:
        quality = Quality.CONFLICTING
        detail = (f"{claimed_ok[0].get('target')} was reported stopped, but it is still running")
    elif still_running:
        quality = Quality.FAILED
        detail = f"{still_running[0].get('target')} is still running"
    elif eased and all(eased):
        quality = Quality.VERIFIED
        detail = "; ".join(c["check"] for c in checks)
    elif eased and not any(eased):
        quality = Quality.PARTIALLY_VERIFIED if gone else Quality.FAILED
        m = measures.get("cpu_percent") or measures.get("memory_percent") or {}
        detail = (f"the change was made, but the machine is still under pressure "
                  f"({m.get('before', 0):.0f}% → {m.get('after', 0):.0f}%)")
    elif gone:
        quality = Quality.PARTIALLY_VERIFIED if eased else Quality.VERIFIED
        detail = "; ".join(c["check"] for c in checks) or "the targets are gone"
    else:
        quality = Quality.UNVERIFIED
        detail = "nothing could be compared"
    return {"quality": quality.value, "detail": detail, "checks": checks, "measures": measures,
            "resolved": quality == Quality.VERIFIED}


def verify_disk(args: dict[str, Any]) -> dict[str, Any]:
    before, after = args.get("before") or {}, args.get("after") or {}
    deleted = args.get("paths") or []
    checks = []
    absent = [p for p in deleted if not os.path.exists(p)]
    for p in deleted:
        checks.append({"check": f"{p} removed", "passed": p in absent})
    fb, fa = _num(before.get("free")), _num(after.get("free"))
    freed = (fa - fb) if fb is not None and fa is not None else None
    if freed is not None:
        checks.append({"check": f"free space {_size(fb)} → {_size(fa)}", "passed": freed >= 0})
    if deleted and len(absent) < len(deleted):
        quality, detail = Quality.FAILED, f"{len(deleted) - len(absent)} file(s) are still there"
    elif deleted:
        # JARVIS's trash is on the same disk: the space returns when the trash is emptied, so the absence of the
        # files is what can be verified now
        quality, detail = Quality.VERIFIED, f"{len(absent)} file(s) moved to the trash" + \
            (f"; free space changed by {_size(freed)}" if freed is not None else "")
    else:
        quality, detail = Quality.UNVERIFIED, "nothing was removed"
    return {"quality": quality.value, "detail": detail, "checks": checks, "freed": freed,
            "resolved": quality == Quality.VERIFIED}


def verify_backup(args: dict[str, Any]) -> dict[str, Any]:
    source = Path(os.path.expanduser(str(args.get("source") or "")))
    dest = Path(os.path.expanduser(str(args.get("destination") or "")))
    if not source.exists():
        return {"quality": Quality.UNVERIFIED.value, "detail": f"{source} can't be read", "checks": []}
    if not dest.exists():
        return {"quality": Quality.FAILED.value, "detail": f"{dest} does not exist", "checks": []}
    missing, mismatched, count = [], [], 0
    files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
    for f in files:
        rel = f.name if source.is_file() else f.relative_to(source)
        target = dest / rel if dest.is_dir() and not source.is_file() else (dest if dest.is_file() else dest / rel)
        count += 1
        if not target.exists():
            missing.append(str(rel))
        elif target.stat().st_size != f.stat().st_size:
            mismatched.append(str(rel))
    checks = [{"check": f"{count} file(s) present at the destination with the same size",
               "passed": not missing and not mismatched}]
    if missing or mismatched:
        detail = f"{len(missing)} missing and {len(mismatched)} different of {count} file(s)"
        return {"quality": Quality.FAILED.value, "detail": detail, "checks": checks,
                "missing": missing[:20], "mismatched": mismatched[:20]}
    return {"quality": Quality.VERIFIED.value, "detail": f"all {count} file(s) are at {dest} with matching sizes",
            "checks": checks, "resolved": True}


def verify_citations(args: dict[str, Any]) -> dict[str, Any]:
    """Every claim must be found, word for word, in the source it cites. Claims that can't be found are
    reported as unsupported instead of being passed on as fact."""
    claims = [c for c in (args.get("claims") or []) if isinstance(c, dict)]
    root = args.get("root")
    supported, unsupported = [], []
    cache: dict[str, str] = {}
    for c in claims:
        path = str(c.get("source") or "")
        full = path if os.path.isabs(path) or not root else os.path.join(str(root), path)
        if full not in cache:
            try:
                cache[full] = Path(full).read_text(errors="replace")
            except OSError:
                cache[full] = ""
        quote = " ".join(str(c.get("quote") or "").split())
        text = " ".join(cache[full].split())
        (supported if quote and quote in text else unsupported).append(c)
    conflicts = args.get("conflicts") or []
    if not claims:
        quality, detail = Quality.UNVERIFIED, "there were no claims to check"
    elif conflicts and supported:
        quality, detail = Quality.CONFLICTING, f"{len(conflicts)} disagreement(s) between sources"
    elif not unsupported:
        quality, detail = Quality.VERIFIED, f"all {len(supported)} claim(s) found in their sources"
    elif supported:
        quality, detail = Quality.PARTIALLY_VERIFIED, f"{len(supported)} of {len(claims)} claim(s) found in their sources"
    else:
        quality, detail = Quality.FAILED, "none of the claims could be found in the cited sources"
    return {"quality": quality.value, "detail": detail,
            "supported": [{k: c.get(k) for k in ("text", "source", "line")} for c in supported],
            "unsupported": [{k: c.get(k) for k in ("text", "source", "line", "by")} for c in unsupported],
            "checks": [{"check": "claims found in their sources", "passed": not unsupported}]}


def verify_command(args: dict[str, Any]) -> dict[str, Any]:
    """A check command run again by the verifier (e.g. "python -c 'import yaml'"): its exit code decides."""
    result = args.get("result") or {}
    expected = int(args.get("expect_exit", 0))
    what = args.get("what") or "the check"
    code = result.get("exit_code") if isinstance(result, dict) else None
    if code is None:
        return {"quality": Quality.UNVERIFIED.value, "detail": f"{what} didn't run", "resolved": None}
    if code == expected:
        return {"quality": Quality.VERIFIED.value, "detail": f"{what} passes now", "resolved": True}
    err = (result.get("stderr") or result.get("stdout") or "").strip().splitlines()
    return {"quality": Quality.FAILED.value, "detail": f"{what} still fails" + (f": {err[-1][:160]}" if err else ""),
            "resolved": False}


def verify_screen(args: dict[str, Any]) -> dict[str, Any]:
    """Visual verification: the error message is (or isn't) still on screen."""
    result = args.get("result") or {}
    passed = result.get("passed") if isinstance(result, dict) else None
    detail = (result.get("detail") if isinstance(result, dict) else "") or "the screen couldn't be checked"
    if passed is None:
        return {"quality": Quality.UNVERIFIED.value, "detail": f"not checked on screen: {detail}", "resolved": None}
    return {"quality": (Quality.VERIFIED if passed else Quality.FAILED).value, "detail": detail,
            "resolved": bool(passed)}


COMPARATORS = {"performance": verify_performance, "disk": verify_disk, "backup": verify_backup,
               "citations": verify_citations, "command": verify_command, "screen": verify_screen}


def _size(value: Any) -> str:
    n = _num(value)
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def size_text(value: Any) -> str:
    return _size(value)


def common_words(texts: list[str], n: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(_content_words(t))
    return [w for w, _ in counter.most_common(n)]
