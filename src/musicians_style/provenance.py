"""Standard-library-only checkout/import provenance for opt-in research audits."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


DEPENDENCIES = (
    "musicians-style", "numpy", "scipy", "scikit-learn", "mido", "pretty_midi",
    "torch", "pytest", "hypothesis", "musif", "music21", "partitura",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_provenance(checkout: Path, package: ModuleType | None = None) -> dict[str, Any]:
    """Inspect the actual import; never add to sys.path or fix an installation."""
    checkout = checkout.resolve()
    expected = (checkout / "src" / "musicians_style").resolve()
    result: dict[str, Any] = {
        "checkout": str(checkout), "expected_package_directory": str(expected),
        "package_file": None, "resolved_source_path": None, "package_search_locations": [],
        "passed": False, "error": None,
    }
    try:
        package = package if package is not None else importlib.import_module("musicians_style")
        result["package_file"] = getattr(package, "__file__", None)
        locations = [str(Path(item).resolve()) for item in getattr(package, "__path__", ())]
        result["package_search_locations"] = locations
        if not result["package_file"]:
            raise ValueError("musicians_style.__file__ is unavailable")
        source = Path(result["package_file"]).resolve(strict=True)
        result["resolved_source_path"] = str(source)
        if not expected.is_relative_to(checkout) or not source.is_relative_to(expected):
            raise ValueError("imported package is outside the active checkout's src/musicians_style")
        if not locations or any(not Path(item).is_relative_to(expected) for item in locations):
            raise ValueError("package search locations include a different installation/worktree")
        result["passed"] = True
    except (ImportError, OSError, ValueError, TypeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def git_provenance(checkout: Path) -> dict[str, Any]:
    def query(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(checkout), *args], check=True, capture_output=True,
                text=True, encoding="utf-8", timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = query("status", "--porcelain")
    return {
        "head": query("rev-parse", "HEAD"), "branch": query("branch", "--show-current"),
        "toplevel": query("rev-parse", "--show-toplevel"),
        "git_directory": query("rev-parse", "--absolute-git-dir"),
        "common_directory": query("rev-parse", "--path-format=absolute", "--git-common-dir"),
        "status_porcelain": status,
        "dirty": bool(status) if status is not None else None,
    }


def collect_provenance(checkout: Path, package: ModuleType | None = None) -> dict[str, Any]:
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        raw = importlib.metadata.distribution("musicians-style").read_text("direct_url.json")
        editable = json.loads(raw) if raw else None
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        editable = None
    return {
        "schema": "research_audit.provenance.1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "import": import_provenance(checkout, package), "git": git_provenance(checkout),
        "environment": {
            "executable": sys.executable, "python": sys.version, "platform": platform.platform(),
            "dependencies": versions, "editable_install_metadata": editable,
        },
    }
