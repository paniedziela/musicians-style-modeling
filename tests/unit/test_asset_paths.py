from pathlib import Path

import pytest

from musicians_style.asset_paths import resolve_asset_roots


def test_roots_precedence_and_checkout_defaults(tmp_path):
    checkout = tmp_path / "worktree"
    roots = resolve_asset_roots(checkout, {"data": "explicit"}, {"MSM_DATA_ROOT": "ignored", "MSM_RESULTS_ROOT": str(tmp_path / "shared")})
    assert roots["data"] == {"path": str((checkout / "explicit").resolve()), "source": "configuration", "input": "explicit"}
    assert roots["results"]["path"] == str((tmp_path / "shared").resolve())
    assert roots["results"]["source"] == "MSM_RESULTS_ROOT"
    assert roots["literature"]["path"] == str((checkout / "Literatura").resolve())
    assert roots["literature"]["source"] == "checkout_default"


def test_roots_are_independent_of_cwd_and_blank_environment(tmp_path, monkeypatch):
    checkout = tmp_path / "other-worktree"
    monkeypatch.chdir(tmp_path)
    roots = resolve_asset_roots(checkout, environ={"MSM_DATA_ROOT": "  "})
    assert roots["data"]["path"] == str((checkout / "datasets").resolve())
    with pytest.raises(ValueError, match="unknown"):
        resolve_asset_roots(checkout, {"typo": "x"}, {})
