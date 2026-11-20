"""Suite-wide guard: no test opens a real HTTP connection (ollama daemon or cloud). A test that needs
HTTP patches urllib.request.urlopen or _post itself, which overrides the blocker for that test only.
"""
import urllib.request

import pytest


def _blocked_urlopen(*args, **kwargs):
    raise AssertionError("real network access from a test: stub urllib.request.urlopen or _post")


@pytest.fixture(autouse=True, scope="session")
def _offline_ollama():
    patch = pytest.MonkeyPatch()
    patch.setattr(urllib.request, "urlopen", _blocked_urlopen)
    yield
    patch.undo()
