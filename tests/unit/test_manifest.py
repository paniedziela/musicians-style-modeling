"""Testy jednostkowe *Manifestu_Zbioru* (``musicians_style.data.manifest``).

Zakres (zadanie 5.1, Wymaganie 1.7):
* :func:`compute_sha256` - determinizm i zgodność z wartością referencyjną,
* :func:`to_json` - struktura zapisanego pliku JSON (pola, format, indent),
* round-trip ``from_json(to_json(m))`` zachowujący równość manifestu,
* pomocnicze :func:`utc_now_iso` i :func:`current_git_commit`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from musicians_style.data.manifest import (
    UNKNOWN_GIT_COMMIT,
    FileEntry,
    Manifest,
    compute_sha256,
    current_git_commit,
    from_json,
    to_dict,
    to_json,
    utc_now_iso,
)


# --------------------------------------------------------------------------- #
# Pomocnicze dane
# --------------------------------------------------------------------------- #
def _sample_manifest() -> Manifest:
    return Manifest(
        artist_id="the_beatles",
        files=[
            FileEntry(
                path="yesterday.mid",
                sha256="a" * 64,
                duration_s=124.3,
                tracks=4,
                source="local",
            ),
            FileEntry(
                path="hey_jude.mid",
                sha256="b" * 64,
                duration_s=425.0,
                tracks=6,
                source="youtube",
            ),
        ],
        created_at="2025-01-15T12:00:00Z",
        git_commit="a1b2c3d",
    )


# --------------------------------------------------------------------------- #
# compute_sha256
# --------------------------------------------------------------------------- #
def test_compute_sha256_matches_reference(tmp_path: Path) -> None:
    content = b"Standard MIDI File content \x00\x01\x02"
    target = tmp_path / "song.mid"
    target.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert compute_sha256(target) == expected


def test_compute_sha256_is_deterministic(tmp_path: Path) -> None:
    target = tmp_path / "song.mid"
    target.write_bytes(b"some bytes" * 1000)

    first = compute_sha256(target)
    second = compute_sha256(target)
    assert first == second
    assert len(first) == 64
    assert all(c in "0123456789abcdef" for c in first)


def test_compute_sha256_differs_for_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.mid"
    b = tmp_path / "b.mid"
    a.write_bytes(b"content A")
    b.write_bytes(b"content B")
    assert compute_sha256(a) != compute_sha256(b)


def test_compute_sha256_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.mid"
    target.write_bytes(b"")
    assert compute_sha256(target) == hashlib.sha256(b"").hexdigest()


def test_compute_sha256_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_sha256(tmp_path / "nie_istnieje.mid")


# --------------------------------------------------------------------------- #
# to_dict / to_json struktura
# --------------------------------------------------------------------------- #
def test_to_dict_structure() -> None:
    manifest = _sample_manifest()
    data = to_dict(manifest)

    assert data["artist_id"] == "the_beatles"
    assert data["created_at"] == "2025-01-15T12:00:00Z"
    assert data["git_commit"] == "a1b2c3d"
    assert isinstance(data["files"], list)
    assert data["files"][0] == {
        "path": "yesterday.mid",
        "sha256": "a" * 64,
        "duration_s": 124.3,
        "tracks": 4,
        "source": "local",
    }


def test_to_json_writes_pretty_file(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    out = tmp_path / "manifest.json"

    returned = to_json(manifest, out)
    assert returned == out
    assert out.exists()

    raw = out.read_text(encoding="utf-8")
    # indent=2 → zagnieżdżone klucze wcięte dwiema spacjami
    assert '\n  "artist_id"' in raw
    # zapisana zawartość parsuje się z powrotem do oczekiwanej struktury
    parsed = json.loads(raw)
    assert parsed == to_dict(manifest)
    assert list(parsed.keys()) == ["artist_id", "files", "created_at", "git_commit"]


def test_to_json_creates_parent_directories(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    out = tmp_path / "nested" / "deep" / "manifest.json"
    to_json(manifest, out)
    assert out.exists()


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
def test_round_trip_to_json_then_load(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    out = tmp_path / "manifest.json"
    to_json(manifest, out)

    loaded = from_json(out)
    assert loaded == manifest


def test_round_trip_classmethods(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    out = tmp_path / "manifest.json"
    manifest.to_json(out)

    loaded = Manifest.load(out)
    assert loaded == manifest


def test_from_dict_applies_defaults_for_optional_fields() -> None:
    data = {
        "artist_id": "queen",
        "files": [
            {
                "path": "bohemian.mid",
                "sha256": "c" * 64,
                "duration_s": 354.0,
                "tracks": 8,
                "source": "local",
            }
        ],
    }
    manifest = Manifest.from_dict(data)
    assert manifest.artist_id == "queen"
    assert manifest.git_commit == UNKNOWN_GIT_COMMIT
    # created_at uzupełniony domyślnie (niepusty ISO-8601 z sufiksem Z)
    assert manifest.created_at.endswith("Z")


def test_empty_files_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(artist_id="abba", created_at="2025-01-15T12:00:00Z")
    out = tmp_path / "manifest.json"
    to_json(manifest, out)
    loaded = from_json(out)
    assert loaded.files == []
    assert loaded == manifest


# --------------------------------------------------------------------------- #
# Pomocnicze
# --------------------------------------------------------------------------- #
def test_utc_now_iso_format() -> None:
    stamp = utc_now_iso()
    assert stamp.endswith("Z")
    assert "T" in stamp
    # brak mikrosekund (sekundy zaokrąglone)
    assert "." not in stamp


def test_default_git_commit_is_unknown_or_hash() -> None:
    # current_git_commit nigdy nie rzuca - zwraca hash lub "unknown"
    commit = current_git_commit()
    assert isinstance(commit, str)
    assert commit != ""
