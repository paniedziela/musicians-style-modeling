import json
from pathlib import Path
from types import ModuleType

import pytest

from musicians_style.provenance import collect_provenance, import_provenance


def package_at(path):
    package = ModuleType("musicians_style")
    package.__file__ = str(path)
    package.__path__ = [str(path.parent)]
    return package


def test_guard_accepts_only_checkout_source(tmp_path):
    source = tmp_path / "src/musicians_style/__init__.py"
    source.parent.mkdir(parents=True)
    source.touch()
    checked = import_provenance(tmp_path, package_at(source))
    assert checked["passed"]
    assert checked["resolved_source_path"] == str(source.resolve())


@pytest.mark.parametrize("location", [".venv/Lib/site-packages/musicians_style/__init__.py", "other-worktree/src/musicians_style/__init__.py"])
def test_guard_rejects_stale_or_other_worktree_even_inside_checkout(tmp_path, location):
    source = tmp_path / location
    source.parent.mkdir(parents=True)
    source.touch()
    assert not import_provenance(tmp_path, package_at(source))["passed"]


def test_guard_rejects_missing_file_and_foreign_search_path(tmp_path):
    assert not import_provenance(tmp_path, ModuleType("musicians_style"))["passed"]
    source = tmp_path / "src/musicians_style/__init__.py"
    assert not import_provenance(tmp_path, package_at(source))["passed"]
    source.parent.mkdir(parents=True)
    source.touch()
    package = package_at(source)
    package.__path__.append(str(tmp_path / "foreign"))
    assert not import_provenance(tmp_path, package)["passed"]


def test_missing_import_is_diagnostic(tmp_path, monkeypatch):
    import musicians_style.provenance as module

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    result = import_provenance(tmp_path)
    assert not result["passed"] and "ModuleNotFoundError" in result["error"]


def test_checkout_provenance_is_serializable_and_records_environment():
    checkout = Path(__file__).resolve().parents[2]
    result = collect_provenance(checkout)
    assert result["import"]["passed"]
    assert result["environment"]["executable"]
    assert "editable_install_metadata" in result["environment"]
    assert result["git"]["head"]
    json.dumps(result)
