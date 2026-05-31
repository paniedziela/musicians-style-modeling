"""Testy jednostkowe logowania strukturalnego (``musicians_style.logging``).

Weryfikują *Format wpisu logu* z ``design.md`` (pola ``ts``, ``level``,
``component``, ``msg`` w każdym wpisie JSON-lines), dowiązanie nazwy komponentu
przez :func:`get_logger` oraz inicjalizację katalogu eksperymentu wraz z
zapisem commit hash (Wymagania 8.2, 8.4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from musicians_style.logging import (
    configure_logging,
    get_logger,
    init_experiment_dir,
)

REQUIRED_FIELDS = {"ts", "level", "component", "msg"}


def _read_log_lines(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def test_log_entry_has_required_fields(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "test.jsonl"
    configure_logging(level="INFO", log_file=log_file, force=True)
    try:
        log = get_logger("trainer")
        log.info("epoch progress", epoch=1, iter=100)
    finally:
        # Przywrócenie domyślnej konfiguracji (stdout) dla kolejnych testów.
        configure_logging(force=True)

    entries = _read_log_lines(log_file)
    assert len(entries) == 1
    entry = entries[0]
    assert REQUIRED_FIELDS.issubset(entry.keys())
    assert entry["component"] == "trainer"
    assert entry["level"] == "INFO"
    assert entry["msg"] == "epoch progress"
    # Dodatkowy kontekst jest zachowany obok pól wymaganych.
    assert entry["epoch"] == 1
    assert entry["iter"] == 100


def test_level_is_uppercase_and_timestamp_present(tmp_path: Path) -> None:
    log_file = tmp_path / "warn.jsonl"
    configure_logging(level="DEBUG", log_file=log_file, force=True)
    try:
        log = get_logger("acquirer")
        log.warning("file rejected", file="broken.mid", reason="invalid SMF header")
    finally:
        configure_logging(force=True)

    entry = _read_log_lines(log_file)[0]
    assert entry["level"] == "WARNING"
    assert entry["component"] == "acquirer"
    # ts powinno być łańcuchem ISO-8601 (zawiera 'T').
    assert isinstance(entry["ts"], str)
    assert "T" in entry["ts"]


def test_component_bound_in_every_entry(tmp_path: Path) -> None:
    log_file = tmp_path / "multi.jsonl"
    configure_logging(level="INFO", log_file=log_file, force=True)
    try:
        log = get_logger("parser")
        log.info("pierwszy")
        log.info("drugi")
        log.error("trzeci")
    finally:
        configure_logging(force=True)

    entries = _read_log_lines(log_file)
    assert len(entries) == 3
    assert all(e["component"] == "parser" for e in entries)
    assert all(REQUIRED_FIELDS.issubset(e.keys()) for e in entries)


@dataclass
class _DummyConfig:
    experiment_name: str


def test_init_experiment_dir_creates_structure_and_commit(tmp_path: Path) -> None:
    config = _DummyConfig(experiment_name="beatles_stargan_v1")
    exp_dir = init_experiment_dir(config, base_dir=tmp_path / "experiments")

    assert exp_dir.exists()
    assert exp_dir.name.startswith("beatles_stargan_v1_")
    # Podkatalogi zgodne z układem manifestu eksperymentu.
    for subdir in ("checkpoints", "logs", "outputs/transferred", "outputs/audio", "reports/plots"):
        assert (exp_dir / subdir).is_dir()

    git_commit = exp_dir / "git_commit.txt"
    assert git_commit.is_file()
    content = git_commit.read_text(encoding="utf-8").strip()
    # git może być niedostępny - wtedy "unknown"; w przeciwnym razie niepusty hash.
    assert content != ""


def test_init_experiment_dir_accepts_mapping(tmp_path: Path) -> None:
    exp_dir = init_experiment_dir(
        {"experiment_name": "ga_run"}, base_dir=tmp_path / "experiments"
    )
    assert exp_dir.name.startswith("ga_run_")
    assert (exp_dir / "git_commit.txt").is_file()


def test_init_experiment_dir_default_name(tmp_path: Path) -> None:
    exp_dir = init_experiment_dir({}, base_dir=tmp_path / "experiments")
    assert exp_dir.name.startswith("experiment_")
