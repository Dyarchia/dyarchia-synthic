"""No test reads the real user environment, or the API key variables, of the machine it runs on."""

from __future__ import annotations

import pytest

from synthic import provider
from synthic.types import KEY_VARIABLES


@pytest.fixture(autouse=True)
def no_registry_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider, "_registry_value", lambda name: None)
    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
