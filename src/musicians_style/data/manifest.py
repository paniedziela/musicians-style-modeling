"""Manifest_Zbioru - metadane *Zbioru_Stylu* w formacie JSON (Wymaganie 1.7).

Moduł definiuje struktury danych oraz funkcje pomocnicze realizujące Wymaganie
1.7: zapis metadanych *Zbioru_Stylu* (lista plików, sumy kontrolne SHA-256,
źródło, długość, liczba ścieżek) w pliku manifestu w formacie JSON.

Struktura manifestu odpowiada przykładowi z sekcji *Akwizytor_Danych* dokumentu
``design.md``::

    {
      "artist_id": "the_beatles",
      "files": [
        {"path": "yesterday.mid", "sha256": "...", "duration_s": 124.3,
         "tracks": 4, "source": "local"}
      ],
      "created_at": "2025-01-15T12:00:00Z",
      "git_commit": "a1b2c3d"
    }

Założenia projektowe:

- Struktury :class:`FileEntry` i :class:`Manifest` to ``@dataclass`` - czytelne,
  porównywalne po wartości i łatwe do (de)serializacji.
- :func:`to_json` zapisuje sformatowany JSON (``indent=2``), a komplementarna
  :func:`from_json` wczytuje go z powrotem, dając własność round-trip
  (``from_json(to_json(m)) == m``) wykorzystywaną w kolejnych zadaniach
  (DatasetAcquirer - zadanie 5.2, dataset treningowy - zadanie 9.1).
- Moduł jest zależnościowo lekki - korzysta wyłącznie z biblioteki standardowej
  (``hashlib``, ``json``, ``dataclasses``, ``datetime``, ``pathlib``,
  ``subprocess``). Nie tworzy twardej zależności od ``logging.init_experiment_dir``.
- ``created_at`` jest łańcuchem ISO-8601 w strefie UTC (z sufiksem ``Z``);
  domyślnie generowany dla chwili bieżącej, lecz może być podany jawnie dla
  determinizmu w testach (:func:`utc_now_iso`).
- ``git_commit`` jest parametrem z sensowną wartością domyślną ``"unknown"``;
  pomocnicza funkcja :func:`current_git_commit` pozwala odczytać commit hash
  repozytorium (``git rev-parse HEAD``) bez zgłaszania wyjątku.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "FileEntry",
    "Manifest",
    "compute_sha256",
    "current_git_commit",
    "to_dict",
    "to_json",
    "from_dict",
    "from_json",
    "utc_now_iso",
]

#: Wartość domyślna pola ``git_commit`` używana, gdy commit hash jest nieznany
#: (git niedostępny lub katalog nie jest repozytorium).
UNKNOWN_GIT_COMMIT = "unknown"

#: Rozmiar bloku (w bajtach) przy strumieniowym czytaniu pliku w :func:`compute_sha256`.
_SHA256_CHUNK_SIZE = 65536  # 64 KiB


@dataclass
class FileEntry:
    """Pojedynczy wpis pliku w *Manifeście_Zbioru* (Wymaganie 1.7).

    Atrybuty:
        path: ścieżka pliku MIDI (względna lub bezwzględna, jako tekst).
        sha256: suma kontrolna SHA-256 zawartości pliku (hex, małe litery).
        duration_s: długość pliku w sekundach (>= 0).
        tracks: liczba ścieżek w pliku MIDI (>= 0).
        source: źródło pliku, np. ``"local"`` lub ``"youtube"``.
    """

    path: str
    sha256: str
    duration_s: float
    tracks: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Zwraca słownik gotowy do serializacji JSON."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "duration_s": self.duration_s,
            "tracks": self.tracks,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileEntry":
        """Tworzy :class:`FileEntry` ze słownika (np. wczytanego z JSON)."""
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            duration_s=float(data["duration_s"]),
            tracks=int(data["tracks"]),
            source=str(data["source"]),
        )


@dataclass
class Manifest:
    """*Manifest_Zbioru* - metadane *Zbioru_Stylu* jednego artysty (Wymaganie 1.7).

    Atrybuty:
        artist_id: identyfikator artysty docelowego (np. ``"the_beatles"``).
        files: lista wpisów plików (:class:`FileEntry`).
        created_at: znacznik czasu utworzenia manifestu w formacie ISO-8601 UTC
            (np. ``"2025-01-15T12:00:00Z"``). Domyślnie chwila bieżąca.
        git_commit: commit hash repozytorium w chwili utworzenia manifestu;
            domyślnie :data:`UNKNOWN_GIT_COMMIT`.
    """

    artist_id: str
    files: list[FileEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: utc_now_iso())
    git_commit: str = UNKNOWN_GIT_COMMIT

    def to_dict(self) -> dict[str, Any]:
        """Zwraca słownik manifestu gotowy do serializacji JSON."""
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Manifest":
        """Tworzy :class:`Manifest` ze słownika (np. wczytanego z JSON)."""
        return from_dict(data)

    def to_json(self, path: Path | str) -> Path:
        """Zapisuje manifest do pliku JSON (delegacja do :func:`to_json`)."""
        return to_json(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "Manifest":
        """Wczytuje manifest z pliku JSON (delegacja do :func:`from_json`)."""
        return from_json(path)


def utc_now_iso() -> str:
    """Zwraca bieżący znacznik czasu w formacie ISO-8601 UTC z sufiksem ``Z``.

    Sekundy są zaokrąglane (bez mikrosekund) dla zwięzłości i zgodności z
    przykładem z ``design.md`` (np. ``"2025-01-15T12:00:00Z"``).

    Returns:
        Łańcuch ISO-8601, np. ``"2025-01-15T12:00:00Z"``.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    # ``isoformat`` zwraca ``+00:00`` dla UTC; zamieniamy na bardziej zwięzłe ``Z``.
    return now.isoformat().replace("+00:00", "Z")


def compute_sha256(path: Path | str) -> str:
    """Oblicza sumę kontrolną SHA-256 zawartości pliku (Wymaganie 1.7).

    Plik jest czytany blokami (strumieniowo), dzięki czemu funkcja działa także
    dla dużych plików bez wczytywania całości do pamięci. Wynik jest
    deterministyczny - dla tej samej zawartości pliku zwracany jest zawsze ten
    sam skrót.

    Args:
        path: ścieżka do pliku.

    Returns:
        Skrót SHA-256 w postaci heksadecymalnej (małe litery, 64 znaki).

    Raises:
        FileNotFoundError: gdy plik nie istnieje.
        OSError: w razie błędu odczytu pliku.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(cwd: Path | str | None = None) -> str:
    """Zwraca aktualny commit hash (``git rev-parse HEAD``) lub ``"unknown"``.

    Funkcja obsługuje przypadek niedostępności gita (brak binarki, katalog nie
    będący repozytorium) bez zgłaszania wyjątku - zwraca wtedy
    :data:`UNKNOWN_GIT_COMMIT`. Pozwala to zachować lekkość modułu (brak twardej
    zależności od ``logging.init_experiment_dir``).

    Args:
        cwd: katalog roboczy, w którym uruchamiane jest polecenie git
            (domyślnie bieżący katalog procesu).

    Returns:
        Commit hash (pełny) lub :data:`UNKNOWN_GIT_COMMIT`.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
    except (OSError, ValueError):
        return UNKNOWN_GIT_COMMIT
    if result.returncode != 0:
        return UNKNOWN_GIT_COMMIT
    commit = result.stdout.strip()
    return commit if commit else UNKNOWN_GIT_COMMIT


def to_dict(manifest: Manifest) -> dict[str, Any]:
    """Konwertuje :class:`Manifest` na słownik gotowy do serializacji JSON.

    Args:
        manifest: manifest do konwersji.

    Returns:
        Słownik o strukturze zgodnej z przykładem w ``design.md``.
    """
    return {
        "artist_id": manifest.artist_id,
        "files": [entry.to_dict() for entry in manifest.files],
        "created_at": manifest.created_at,
        "git_commit": manifest.git_commit,
    }


def from_dict(data: Mapping[str, Any]) -> Manifest:
    """Tworzy :class:`Manifest` ze słownika (np. wczytanego z pliku JSON).

    Args:
        data: mapowanie z kluczami ``artist_id``, ``files``, ``created_at``,
            ``git_commit``. Pola ``created_at`` i ``git_commit`` są opcjonalne -
            przy ich braku przyjmowane są wartości domyślne.

    Returns:
        Zrekonstruowany :class:`Manifest`.

    Raises:
        KeyError: gdy brakuje wymaganego pola ``artist_id``.
    """
    files_data = data.get("files", []) or []
    files = [FileEntry.from_dict(entry) for entry in files_data]
    return Manifest(
        artist_id=str(data["artist_id"]),
        files=files,
        created_at=str(data.get("created_at") or utc_now_iso()),
        git_commit=str(data.get("git_commit") or UNKNOWN_GIT_COMMIT),
    )


def to_json(manifest: Manifest, path: Path | str) -> Path:
    """Zapisuje *Manifest_Zbioru* do pliku w formacie JSON (Wymaganie 1.7).

    Plik zapisywany jest jako sformatowany JSON (``indent=2``) w kodowaniu UTF-8,
    z zachowaniem znaków spoza ASCII (``ensure_ascii=False``). Katalog docelowy
    jest tworzony, jeśli nie istnieje.

    Args:
        manifest: manifest do zapisania.
        path: ścieżka pliku wynikowego (``.json``).

    Returns:
        Ścieżka zapisanego pliku.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict(manifest)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def from_json(path: Path | str) -> Manifest:
    """Wczytuje *Manifest_Zbioru* z pliku JSON.

    Funkcja komplementarna do :func:`to_json` - razem zapewniają własność
    round-trip (``from_json(to_json(m, p))`` zwraca manifest równy ``m``).

    Args:
        path: ścieżka do pliku manifestu (``.json``).

    Returns:
        Zrekonstruowany :class:`Manifest`.

    Raises:
        FileNotFoundError: gdy plik nie istnieje.
        json.JSONDecodeError: gdy zawartość pliku nie jest poprawnym JSON.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_dict(data)
