"""Shared fixtures. Async test functions are run on a fresh event loop without plugins."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from jarvis.clock import FakeClock
from jarvis.database.db import Database


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        funcargs = pyfuncitem.funcargs
        kwargs = {name: funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        # live model tests (real Ollama, possibly on CPU) get far longer than the in-process suite
        timeout = 900 if pyfuncitem.get_closest_marker("ollama") else 30
        asyncio.run(asyncio.wait_for(pyfuncitem.obj(**kwargs), timeout=timeout))
        return True
    return None


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def clock():
    return FakeClock()
