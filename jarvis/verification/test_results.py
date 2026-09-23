"""Parse test-runner output into structured results.

"The command exited 1" is not a diagnosis. Knowing *which* tests failed lets
JARVIS report precisely, classify failures and decide what is safe to fix.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TestResults:
    framework: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)
    parsed: bool = True

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.errors == 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    def summary(self) -> str:
        if not self.parsed:
            return "test output could not be parsed"
        if self.ok:
            return f"all {self.passed} tests passed" + (f" ({self.skipped} skipped)" if self.skipped else "")
        bad = []
        if self.failed:
            bad.append(f"{self.failed} failed")
        if self.errors:
            bad.append(f"{self.errors} error{'s' if self.errors != 1 else ''}")
        return f"{', '.join(bad)}, {self.passed} passed"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        d["summary"] = self.summary()
        return d


def _num(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def parse_test_output(output: str, exit_code: int | None = None) -> TestResults:
    text = output or ""
    # pytest: "==== 2 failed, 10 passed, 1 skipped in 0.52s ====" or "10 passed in 0.1s"
    if re.search(r"\b\d+ (passed|failed)\b.*\bin [\d.]+s", text) or "short test summary" in text:
        results = TestResults("pytest", _num(r"(\d+) passed", text), _num(r"(\d+) failed", text),
                              _num(r"(\d+) errors?\b", text), _num(r"(\d+) skipped", text))
        results.failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.MULTILINE)[:50]
        return results
    # jest / vitest: "Tests:       1 failed, 5 passed, 6 total"
    m = re.search(r"Tests:\s+(.*?\d+ total)", text)
    if m:
        line = m.group(1)
        results = TestResults("jest", _num(r"(\d+) passed", line), _num(r"(\d+) failed", line), 0,
                              _num(r"(\d+) skipped", line))
        results.failures = re.findall(r"●\s+(.+?)\s*$", text, re.MULTILINE)[:50]
        return results
    # cargo: "test result: FAILED. 3 passed; 1 failed; 0 ignored"
    if "test result:" in text:
        passed = sum(int(x) for x in re.findall(r"(\d+) passed", text))
        failed = sum(int(x) for x in re.findall(r"(\d+) failed", text))
        ignored = sum(int(x) for x in re.findall(r"(\d+) ignored", text))
        results = TestResults("cargo", passed, failed, 0, ignored)
        results.failures = re.findall(r"^test (\S+) \.\.\. FAILED", text, re.MULTILINE)[:50]
        return results
    # go test: "--- FAIL: TestX" / "ok  pkg" / "FAIL pkg"
    if re.search(r"^(ok|FAIL|---)\s", text, re.MULTILINE) and ("--- " in text or re.search(r"^ok\s+\S+", text, re.M)):
        failures = re.findall(r"^--- FAIL: (\S+)", text, re.MULTILINE)
        passes = re.findall(r"^--- PASS: (\S+)", text, re.MULTILINE)
        pkg_ok = len(re.findall(r"^ok\s+\S+", text, re.MULTILINE))
        return TestResults("go", len(passes) or pkg_ok, len(failures), 0, 0, failures[:50])
    # unittest: "Ran 12 tests in 0.01s" + "FAILED (failures=2, errors=1)"
    m = re.search(r"Ran (\d+) tests? in", text)
    if m:
        total = int(m.group(1))
        failed = _num(r"failures=(\d+)", text)
        errors = _num(r"errors=(\d+)", text)
        skipped = _num(r"skipped=(\d+)", text)
        return TestResults("unittest", total - failed - errors - skipped, failed, errors, skipped,
                           re.findall(r"^(?:FAIL|ERROR): (\S+)", text, re.MULTILINE)[:50])
    results = TestResults("unknown", parsed=False)
    if exit_code is not None:
        results.parsed = False
        if exit_code != 0:
            results.errors = 1
    return results
