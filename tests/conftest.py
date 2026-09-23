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
        asyncio.run(asyncio.wait_for(pyfuncitem.obj(**kwargs), timeout=30))
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
