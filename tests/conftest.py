"""No test reads the real user environment of the machine it runs on."""

from __future__ import annotations

import pytest

from synthic import provider


@pytest.fixture(autouse=True)
def no_registry_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider, "_registry_value", lambda name: None)
