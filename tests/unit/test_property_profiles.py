import ast
from pathlib import Path

from hypothesis import settings

from tests.property.profiles import property_settings


def test_development_caps_and_full_preserves_original_budgets():
    previous = settings.default
    try:
        settings.load_profile("dev")
        for budget in (50, 100, 200, 300):
            assert property_settings(max_examples=budget, deadline=None).max_examples == 20
        settings.load_profile("full")
        for budget in (50, 100, 200, 300):
            result = property_settings(max_examples=budget, deadline=None)
            assert result.max_examples == budget and result.deadline is None
    finally:
        # Restore the chosen CLI profile without relying on Hypothesis internals.
        settings.load_profile("dev" if previous == settings.get_profile("dev") else "full")


def test_all_existing_property_decorators_use_profile_helper():
    counts = []
    root = Path(__file__).resolve().parents[1] / "property"
    for path in root.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "settings", path.name
                if node.func.id == "property_settings":
                    counts.extend(k.value.value for k in node.keywords if k.arg == "max_examples")
    assert sorted(counts) == [50, 50, 100] + [200] * 12 + [300, 300]
