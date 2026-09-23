"""Secret access.

Secrets are referenced, not embedded: configuration and tool arguments carry
``secret://NAME`` references which are resolved only inside the tool that needs
them, at execution time. Models never see raw secret values (spec §83).

The default backend reads environment variables. Other backends (OS keyring,
encrypted vault) implement :class:`SecretBackend`.
"""

from __future__ import annotations

import os
from typing import Protocol

SECRET_SCHEME = "secret://"


class SecretBackend(Protocol):
    def get(self, name: str) -> str | None: ...


class EnvSecretBackend:
    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._env = env

    def get(self, name: str) -> str | None:
        env = os.environ if self._env is None else self._env
        return env.get(name)


class SecretStore:
    def __init__(self, backend: SecretBackend | None = None) -> None:
        self._backend = backend or EnvSecretBackend()

    def get(self, name: str) -> str | None:
        return self._backend.get(name)

    def resolve(self, value: str) -> str:
        """Resolve a ``secret://NAME`` reference; plain strings pass through unchanged."""
        if not value.startswith(SECRET_SCHEME):
            return value
        name = value[len(SECRET_SCHEME):]
        secret = self._backend.get(name)
        if secret is None:
            raise KeyError(f"secret {name!r} is not available")
        return secret

    def has(self, name: str) -> bool:
        return self._backend.get(name) is not None
