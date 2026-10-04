"""Local asset roots for new audits; historical experiment loaders are unchanged."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


ROOTS = {
    "data": ("MSM_DATA_ROOT", "datasets"),
    "results": ("MSM_RESULTS_ROOT", "experiments"),
    "literature": ("MSM_LITERATURE_ROOT", "Literatura"),
}


def resolve_asset_roots(
    checkout: Path,
    configuration: Mapping[str, str | Path | None] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Resolve explicit > environment > checkout defaults, independent of cwd."""
    checkout = checkout.resolve()
    configuration = configuration or {}
    environ = os.environ if environ is None else environ
    unknown = set(configuration) - set(ROOTS)
    if unknown:
        raise ValueError(f"unknown asset roots: {sorted(unknown)}")
    resolved = {}
    for name, (variable, fallback) in ROOTS.items():
        explicit = configuration.get(name)
        if explicit is not None and str(explicit).strip():
            value, source = str(explicit), "configuration"
        elif environ.get(variable, "").strip():
            value, source = environ[variable], variable
        else:
            value, source = fallback, "checkout_default"
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = checkout / path
        resolved[name] = {"path": str(path.resolve()), "source": source, "input": value}
    return resolved
