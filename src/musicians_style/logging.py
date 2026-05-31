"""Logowanie strukturalne *Systemu* w formacie JSON-lines (Wymaganie 8.4).

Moduł konfiguruje bibliotekę :mod:`structlog` tak, aby każdy wpis logu był
pojedynczą linią JSON zawierającą **co najmniej** pola wymagane w sekcji
*Format wpisu logu* dokumentu ``design.md``:

* ``ts``        - znacznik czasowy ISO-8601 (UTC),
* ``level``     - poziom logowania (np. ``"INFO"``, ``"WARNING"``),
* ``component`` - nazwa komponentu (związana funkcją :func:`get_logger`),
* ``msg``       - treść komunikatu.

Przykład wpisu::

    {"ts": "2025-03-10T12:00:01.234Z", "level": "INFO", "component": "trainer", "msg": "epoch progress"}

Uwaga implementacyjna: plik nosi nazwę ``logging.py`` wewnątrz pakietu
``musicians_style``. W Pythonie 3 instrukcja ``import logging`` jest importem
bezwzględnym i wskazuje bibliotekę standardową, a nie ten moduł. Dla pełnej
czytelności bibliotekę standardową importujemy pod aliasem ``stdlib_logging``,
a :mod:`structlog` importujemy bezpośrednio.
"""

from __future__ import annotations

import logging as stdlib_logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Mapping

import structlog

__all__ = [
    "configure_logging",
    "get_logger",
    "init_experiment_dir",
]

# Flaga zapobiegająca wielokrotnej konfiguracji structlog (idempotencja).
_CONFIGURED: bool = False


def _uppercase_level(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Procesor structlog: zamienia wartość pola ``level`` na wielkie litery.

    ``structlog.processors.add_log_level`` zapisuje poziom małymi literami
    (``"info"``), natomiast *Format wpisu logu* w ``design.md`` używa wielkich
    liter (``"INFO"``). Procesor ujednolica tę reprezentację.
    """
    level = event_dict.get("level")
    if isinstance(level, str):
        event_dict["level"] = level.upper()
    return event_dict


def _build_processors() -> list[Any]:
    """Buduje łańcuch procesorów structlog gwarantujący wymagane pola wpisu."""
    return [
        # Dodaje pole "level" (małymi literami) na podstawie wywołanej metody.
        structlog.processors.add_log_level,
        _uppercase_level,
        # Dodaje pole "ts" jako znacznik czasu ISO-8601 w strefie UTC.
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        # Zmienia domyślny klucz "event" na "msg" zgodnie z formatem logu.
        structlog.processors.EventRenamer("msg"),
        # Renderuje cały słownik zdarzenia jako pojedynczą linię JSON.
        structlog.processors.JSONRenderer(sort_keys=True),
    ]


def configure_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    *,
    force: bool = False,
) -> None:
    """Konfiguruje globalnie :mod:`structlog` do logowania w formacie JSON-lines.

    Args:
        level: minimalny poziom logowania (``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``).
        log_file: opcjonalna ścieżka pliku, do którego dopisywane są wpisy
            (tryb append). Gdy ``None``, logi trafiają na ``stdout``.
        force: wymusza ponowną konfigurację nawet jeśli moduł był już
            skonfigurowany.

    Konfiguracja jest idempotentna - kolejne wywołania bez ``force=True`` nie
    zmieniają stanu (zapobiega to dublowaniu handlerów).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    min_level = stdlib_logging.getLevelName(level.upper())
    if not isinstance(min_level, int):
        # Nieznana nazwa poziomu - bezpieczny domyślny próg INFO.
        min_level = stdlib_logging.INFO

    stream: IO[str] | None = None
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a", encoding="utf-8")

    structlog.configure(
        processors=_build_processors(),
        wrapper_class=structlog.make_filtering_bound_logger(min_level),
        logger_factory=structlog.WriteLoggerFactory(file=stream),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(component: str) -> Any:
    """Zwraca logger structlog z trwale związaną nazwą komponentu.

    Pole ``component`` pojawia się w **każdym** wpisie wyprodukowanym przez
    zwrócony logger. Jeżeli logowanie nie zostało jeszcze skonfigurowane,
    funkcja wykonuje konfigurację domyślną (poziom ``INFO``, wyjście na
    ``stdout``).

    Args:
        component: nazwa komponentu Systemu (np. ``"trainer"``, ``"acquirer"``,
            ``"parser"``).

    Returns:
        Logger structlog z dowiązanym kontekstem ``component=<component>``.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger().bind(component=component)


def _resolve_experiment_name(config: Any) -> str:
    """Wyodrębnia nazwę eksperymentu z obiektu lub mapowania konfiguracji.

    Obsługuje zarówno dataclass/obiekt z atrybutem ``experiment_name``, jak i
    mapowanie (np. ``dict``) z kluczem ``experiment_name``. Brak nazwy skutkuje
    użyciem wartości domyślnej ``"experiment"``.
    """
    name: Any = None
    if isinstance(config, Mapping):
        name = config.get("experiment_name")
    else:
        name = getattr(config, "experiment_name", None)
    if not name:
        return "experiment"
    return str(name)


def _get_git_commit(cwd: Path) -> str:
    """Zwraca aktualny commit hash (``git rev-parse HEAD``) lub ``"unknown"``.

    Funkcja obsługuje przypadek niedostępności gita (brak binarki, katalog nie
    będący repozytorium) bez zgłaszania wyjątku - zwraca wtedy ``"unknown"``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            check=False,
        )
    except (OSError, ValueError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit if commit else "unknown"


def init_experiment_dir(
    config: Any,
    base_dir: Path | str = "experiments",
) -> Path:
    """Tworzy katalog wynikowy eksperymentu i zapisuje commit hash repozytorium.

    Tworzy katalog ``{base_dir}/{experiment_name}_{timestamp}/`` wraz z
    podkatalogami zgodnymi z układem z ``design.md`` (``checkpoints/``,
    ``logs/``, ``outputs/transferred/``, ``outputs/audio/``, ``reports/plots/``)
    oraz zapisuje plik ``git_commit.txt`` z wynikiem ``git rev-parse HEAD``
    (Wymaganie 8.2). Gdy git jest niedostępny, plik zawiera ``"unknown"`` i
    fakt ten jest odnotowywany w logu.

    Args:
        config: obiekt lub mapowanie konfiguracji zawierające
            ``experiment_name``.
        base_dir: katalog nadrzędny dla eksperymentów (domyślnie
            ``"experiments"`` względem bieżącego katalogu roboczego).

    Returns:
        Ścieżka do utworzonego katalogu eksperymentu.
    """
    log = get_logger("experiment")

    experiment_name = _resolve_experiment_name(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    experiment_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"

    # Utworzenie struktury katalogów zgodnej z układem manifestu eksperymentu.
    for subdir in (
        "checkpoints",
        "logs",
        "outputs/transferred",
        "outputs/audio",
        "reports/plots",
    ):
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

    commit = _get_git_commit(experiment_dir)
    git_commit_path = experiment_dir / "git_commit.txt"
    git_commit_path.write_text(commit + "\n", encoding="utf-8")

    if commit == "unknown":
        log.warning(
            "nie udało się ustalić commit hash (git niedostępny)",
            experiment_dir=str(experiment_dir),
        )
    else:
        log.info(
            "zainicjowano katalog eksperymentu",
            experiment_dir=str(experiment_dir),
            git_commit=commit,
        )

    return experiment_dir
