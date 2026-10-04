"""Checkout-first tests and explicit property/integration execution tiers."""

from pathlib import Path

import pytest
from hypothesis import settings

from musicians_style.provenance import import_provenance


settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("full", max_examples=200, deadline=None)
settings.load_profile("full")


def pytest_sessionstart(session):
    checked = import_provenance(Path(__file__).resolve().parents[1])
    if not checked["passed"]:
        raise pytest.UsageError(f"checkout import guard failed: {checked}")


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in Path(str(item.path)).parts:
            item.add_marker(pytest.mark.integration)
