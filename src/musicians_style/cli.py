"""Interfejs wiersza poleceń (CLI) *Systemu* modelowania stylu artystycznego.

Moduł realizuje zadanie 12.1 planu implementacyjnego: jednolite CLI ``midi-style``
spinające komponenty Systemu w cztery komendy (Wymagania 5.1, 5.8, 8.2):

* ``midi-style acquire --config CONFIG`` - pozyskanie *Zbiorów_Stylu* z lokalnych
  katalogów i zapis *Manifestów_Zbioru* (Wymaganie 1).
* ``midi-style train --config CONFIG`` - trening *Modelu_GAN* (StarGAN w trybie
  ``conditional`` lub CycleGAN w trybie ``per_artist``; Wymaganie 3).
* ``midi-style infer --target_artist NAME --checkpoint CKPT --input X.mid
  --output Y.mid [--seed S]`` - inferencja transferu stylu (Wymaganie 5).
* ``midi-style evaluate --pairs PAIRS_DIR --style MANIFEST`` - ewaluacja
  obiektywna par ``(Utwór_Wejściowy, Utwór_Wyjściowy)`` względem *Zbioru_Stylu*
  (Wymaganie 6).

Architektura
------------
Sercem modułu jest funkcja :func:`main`, zwracająca **kod wyjścia** (``int``)
zamiast bezpośrednio kończyć proces. Dzięki temu CLI jest w pełni testowalne
jednostkowo bez uruchamiania podprocesów. Cienka osłona
``if __name__ == "__main__": sys.exit(main())`` przekłada zwrócony kod na kod
zakończenia procesu.

Kody wyjścia (zgodnie z sekcją *Error Handling* dokumentu ``design.md`` i
zadaniem 12.1)::

    0  - sukces,
    1  - nieobsłużony (uncaught) wyjątek (błąd programu),
    2  - błąd walidacji wejścia/konfiguracji lub logiki domeny
         (ConfigValidationError, EmptyDatasetError, UnknownArtistError,
          ArtistMismatchError, MidiValidationError, brak pliku wejściowego),
    3  - błąd zasobów (GpuOutOfMemoryError - niewystarczająca pamięć GPU).

Mapowanie wyjątków na kody wykorzystuje atrybut ``exit_code`` przenoszony przez
hierarchię :class:`~musicians_style.errors.MusiciansStyleError` (każdy podtyp
deklaruje właściwy kod). *Pipeline* i *Trener* **nie** wywołują ``sys.exit`` -
zgłaszają wyjątki domenowe, a ich tłumaczeniem na kody wyjścia zajmuje się
wyłącznie warstwa CLI.

Inicjalizacja katalogu eksperymentu (Wymaganie 8.2)
---------------------------------------------------
Każde uruchomienie inicjalizuje katalog ``experiments/{name}_{ts}/`` przez
:func:`~musicians_style.logging.init_experiment_dir` (tworzy strukturę
podkatalogów oraz ``git_commit.txt`` z ``git rev-parse HEAD``). Gdy dostępny jest
plik konfiguracyjny, jego kopia jest zapisywana jako ``config_used.<ext>``.
Katalog nadrzędny eksperymentów można nadpisać opcją ``--experiments-dir``
(wykorzystywaną m.in. w testach do wskazania katalogu tymczasowego).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from musicians_style.config import Config, load_config
from musicians_style.data.acquirer import DatasetAcquirer
from musicians_style.data.manifest import Manifest, from_json, to_json
from musicians_style.errors import (
    ConfigValidationError,
    GpuOutOfMemoryError,
    MusiciansStyleError,
)
from musicians_style.evaluation.objective import ObjectiveEvaluator
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.inference.pipeline import StyleTransferPipeline
from musicians_style.logging import get_logger, init_experiment_dir
from musicians_style.training.dataset import MultiArtistManifest
from musicians_style.training.trainer import GANTrainer

__all__ = [
    "EXIT_SUCCESS",
    "EXIT_UNCAUGHT",
    "EXIT_VALIDATION",
    "EXIT_RESOURCE",
    "build_parser",
    "main",
]

# --------------------------------------------------------------------------- #
# Kody wyjścia procesu (zadanie 12.1)
# --------------------------------------------------------------------------- #
#: Sukces.
EXIT_SUCCESS = 0
#: Nieobsłużony wyjątek (błąd programu).
EXIT_UNCAUGHT = 1
#: Błąd walidacji wejścia/konfiguracji lub logiki domeny.
EXIT_VALIDATION = 2
#: Błąd zasobów (np. niewystarczająca pamięć GPU - OOM).
EXIT_RESOURCE = 3

_DEFAULT_EXPERIMENTS_DIR = "experiments"
_MIDI_SUFFIXES = (".mid", ".midi")


# --------------------------------------------------------------------------- #
# Pomocnicze: katalog eksperymentu (Wymaganie 8.2)
# --------------------------------------------------------------------------- #
def _prepare_experiment_dir(
    config_or_name: Any,
    experiments_dir: str | Path,
    config_source: Path | None = None,
) -> Path:
    """Tworzy katalog eksperymentu i (opcjonalnie) zapisuje kopię konfiguracji.

    Deleguje utworzenie struktury katalogów oraz pliku ``git_commit.txt`` do
    :func:`~musicians_style.logging.init_experiment_dir` (Wymaganie 8.2), a
    następnie - jeżeli podano źródłowy plik konfiguracyjny - kopiuje go do
    katalogu eksperymentu jako ``config_used.<ext>`` (Wymaganie 8.2: kopia
    użytej konfiguracji).

    Args:
        config_or_name: obiekt :class:`Config` lub mapowanie/obiekt z atrybutem
            ``experiment_name`` (steruje nazwą katalogu).
        experiments_dir: katalog nadrzędny eksperymentów (``experiments`` lub
            ścieżka tymczasowa w testach).
        config_source: ścieżka źródłowego pliku konfiguracyjnego do skopiowania;
            ``None`` dla komend bez konfiguracji.

    Returns:
        Ścieżka utworzonego katalogu eksperymentu.
    """
    experiment_dir = init_experiment_dir(config_or_name, base_dir=experiments_dir)
    if config_source is not None:
        suffix = config_source.suffix or ".yaml"
        shutil.copyfile(config_source, experiment_dir / f"config_used{suffix}")
    return experiment_dir


# --------------------------------------------------------------------------- #
# Pomocnicze: wykrywanie par MIDI dla ewaluacji
# --------------------------------------------------------------------------- #
def _gather_midi(directory: Path) -> list[Path]:
    """Zwraca posortowaną listę plików MIDI (``.mid`` / ``.midi``) w katalogu."""
    found: list[Path] = []
    for suffix in _MIDI_SUFFIXES:
        found.extend(directory.glob(f"*{suffix}"))
    return sorted(found, key=lambda p: p.name)


def _discover_pairs(pairs_dir: Path | str) -> list[tuple[Path, Path]]:
    """Wykrywa pary ``(Utwór_Wejściowy, Utwór_Wyjściowy)`` w katalogu (Wymaganie 6).

    Obsługiwane są dwie konwencje nazewnicze (sprawdzane w tej kolejności):

    #. **Podkatalogi** ``input/`` i ``output/``: pary tworzone są przez
       dopasowanie identycznych nazw plików (``input/x.mid`` ↔ ``output/x.mid``).
    #. **Płaski katalog** z sufiksami: pliki ``<stem>_input.mid`` i odpowiadające
       im ``<stem>_output.mid`` (analogicznie dla ``.midi``).

    Args:
        pairs_dir: katalog zawierający pary utworów.

    Returns:
        Lista par ścieżek ``(input_midi, output_midi)`` w kolejności
        deterministycznej (alfabetycznej wg nazwy wejścia).

    Raises:
        ConfigValidationError: gdy katalog nie istnieje lub nie znaleziono w nim
            żadnej kompletnej pary (kod wyjścia 2 - walidacja wejścia).
    """
    base = Path(pairs_dir)
    if not base.is_dir():
        raise ConfigValidationError(
            f"Katalog par PAIRS_DIR nie istnieje lub nie jest katalogiem: {base}."
        )

    input_dir = base / "input"
    output_dir = base / "output"
    pairs: list[tuple[Path, Path]] = []

    if input_dir.is_dir() and output_dir.is_dir():
        # Konwencja 1: dopasowanie po nazwie pliku między input/ a output/.
        for in_path in _gather_midi(input_dir):
            out_path = output_dir / in_path.name
            if out_path.exists():
                pairs.append((in_path, out_path))
    else:
        # Konwencja 2: pliki <stem>_input.<ext> i <stem>_output.<ext>.
        for suffix in _MIDI_SUFFIXES:
            for in_path in sorted(base.glob(f"*_input{suffix}"), key=lambda p: p.name):
                out_path = base / in_path.name.replace("_input", "_output", 1)
                if out_path.exists():
                    pairs.append((in_path, out_path))

    if not pairs:
        raise ConfigValidationError(
            "Nie znaleziono par (Utwór_Wejściowy, Utwór_Wyjściowy) w katalogu "
            f"{base}. Oczekiwano podkatalogów 'input/' i 'output/' z plikami o "
            "tych samych nazwach albo plików '<nazwa>_input.mid' / "
            "'<nazwa>_output.mid'."
        )
    return pairs


# --------------------------------------------------------------------------- #
# Handlery komend
# --------------------------------------------------------------------------- #
def _cmd_acquire(args: argparse.Namespace) -> int:
    """Komenda ``acquire`` - pozyskanie *Zbiorów_Stylu* z lokalnych katalogów.

    Dla każdego artysty zadeklarowanego w ``dataset.artists`` wczytuje pliki MIDI
    przez :meth:`DatasetAcquirer.acquire_local` (walidacja SMF, długości i
    liczności - Wymaganie 1) i zapisuje *Manifest_Zbioru* (Wymaganie 1.7) do
    katalogu eksperymentu jako ``manifest_<artist_id>.json``.

    Returns:
        :data:`EXIT_SUCCESS` po pomyślnym pozyskaniu wszystkich zbiorów.
    """
    log = get_logger("cli")
    config_path = Path(args.config)
    config = load_config(config_path)
    experiment_dir = _prepare_experiment_dir(
        config, args.experiments_dir, config_path
    )

    if not config.dataset.artists:
        raise ConfigValidationError(
            "Sekcja 'dataset.artists' jest pusta - brak katalogów do pozyskania "
            "(Wymaganie 1.1)."
        )

    acquirer = DatasetAcquirer()
    saved: list[str] = []
    for artist_id, directory in config.dataset.artists.items():
        manifest = acquirer.acquire_local(directory, artist_id)
        manifest_path = experiment_dir / f"manifest_{artist_id}.json"
        to_json(manifest, manifest_path)
        saved.append(str(manifest_path))

    log.info(
        "zakończono pozyskiwanie Zbiorów_Stylu",
        experiment_dir=str(experiment_dir),
        manifests=saved,
    )
    return EXIT_SUCCESS


def _build_train_manifest(
    config: Config, acquirer: DatasetAcquirer
) -> tuple[Manifest | MultiArtistManifest, Any]:
    """Buduje manifest treningowy i strukturę ``roots`` zgodnie z trybem modelu.

    * tryb ``conditional`` → :class:`MultiArtistManifest` z manifestów wszystkich
      artystów ``model.artists`` oraz mapowanie ``roots = {artist_id → katalog}``,
    * tryb ``per_artist`` → pojedynczy :class:`Manifest` artysty docelowego oraz
      ``roots`` wskazujące katalog tego artysty.

    Raises:
        ConfigValidationError: gdy brakuje katalogu danych dla artysty
            zadeklarowanego w konfiguracji (kod wyjścia 2).
    """
    artists_dirs = config.dataset.artists

    if config.model.mode == "conditional":
        if not config.model.artists:
            raise ConfigValidationError(
                "Tryb 'conditional' wymaga niepustej listy 'model.artists'."
            )
        manifests: dict[str, Manifest] = {}
        roots: dict[str, str] = {}
        for artist_id in config.model.artists:
            directory = artists_dirs.get(artist_id)
            if directory is None:
                raise ConfigValidationError(
                    f"Brak katalogu danych dla artysty '{artist_id}' w sekcji "
                    "'dataset.artists' (Wymaganie 8.6)."
                )
            manifests[artist_id] = acquirer.acquire_local(directory, artist_id)
            roots[artist_id] = directory
        return MultiArtistManifest(manifests), roots

    # tryb per_artist
    artist_id = (
        config.model.artists[0]
        if config.model.artists
        else config.model.reference_artist
    )
    if not artist_id:
        raise ConfigValidationError(
            "Tryb 'per_artist' wymaga wskazania artysty docelowego w "
            "'model.artists'."
        )
    directory = artists_dirs.get(artist_id)
    if directory is None:
        raise ConfigValidationError(
            f"Brak katalogu danych dla artysty '{artist_id}' w sekcji "
            "'dataset.artists'."
        )
    manifest = acquirer.acquire_local(directory, artist_id)
    return manifest, directory


def _cmd_train(args: argparse.Namespace) -> int:
    """Komenda ``train`` - trening *Modelu_GAN* (Wymaganie 3).

    Wczytuje konfigurację, pozyskuje manifesty zbiorów, a następnie uruchamia
    :meth:`GANTrainer.train` z *Seedem* z konfiguracji (Wymaganie 3.6).
    *Punkty_Kontrolne* i logi treningu trafiają do katalogu eksperymentu.

    Błąd niewystarczającej pamięci GPU (:class:`GpuOutOfMemoryError`) propaguje do
    :func:`main`, gdzie jest mapowany na kod wyjścia :data:`EXIT_RESOURCE` (3).

    Returns:
        :data:`EXIT_SUCCESS` po zapisaniu ostatniego *Punktu_Kontrolnego*.
    """
    log = get_logger("cli")
    config_path = Path(args.config)
    config = load_config(config_path)
    experiment_dir = _prepare_experiment_dir(
        config, args.experiments_dir, config_path
    )

    acquirer = DatasetAcquirer()
    train_manifest, roots = _build_train_manifest(config, acquirer)

    trainer = GANTrainer(
        config,
        config.model.mode,  # type: ignore[arg-type]
        output_dir=experiment_dir,
        roots=roots,
    )
    checkpoint = trainer.train(train_manifest, seed=config.seed)

    log.info(
        "zakończono trening Modelu_GAN",
        experiment_dir=str(experiment_dir),
        checkpoint=str(checkpoint),
    )
    return EXIT_SUCCESS


def _cmd_infer(args: argparse.Namespace) -> int:
    """Komenda ``infer`` - inferencja transferu stylu *Modelu_GAN* (Wymaganie 5).

    Parametr ``--target_artist`` jest **obowiązkowy** niezależnie od trybu
    *Punktu_Kontrolnego* (Wymaganie 5.8; wymuszone przez ``required=True`` w
    parserze). Walidacja wartości ``target_artist`` względem metadanych
    *Punktu_Kontrolnego* (Wymagania 5.9, 5.10) jest wykonywana przez
    :meth:`StyleTransferPipeline.infer_gan`; ewentualne
    :class:`~musicians_style.errors.UnknownArtistError` /
    :class:`~musicians_style.errors.ArtistMismatchError` propagują do
    :func:`main` i są mapowane na kod wyjścia :data:`EXIT_VALIDATION` (2).

    Returns:
        :data:`EXIT_SUCCESS` po zapisaniu *Utworu_Wyjściowego*.
    """
    log = get_logger("cli")

    config: Config | None = None
    config_source: Path | None = None
    if args.config:
        config_source = Path(args.config)
        config = load_config(config_source)

    name_holder: Any = config if config is not None else {"experiment_name": "infer"}
    experiment_dir = _prepare_experiment_dir(
        name_holder, args.experiments_dir, config_source
    )

    pipeline = StyleTransferPipeline(config)
    output = pipeline.infer_gan(
        args.input,
        args.target_artist,
        args.checkpoint,
        args.seed,
        output_path=args.output,
    )

    log.info(
        "zakończono inferencję transferu stylu",
        experiment_dir=str(experiment_dir),
        target_artist=args.target_artist,
        output=str(output),
    )
    return EXIT_SUCCESS


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Komenda ``evaluate`` - ewaluacja obiektywna transferu stylu (Wymaganie 6).

    Wczytuje *Manifest_Zbioru* artysty docelowego, agreguje jego *Wektory_Cech*
    (:meth:`FeatureExtractor.extract_dataset`), wykrywa pary
    ``(Utwór_Wejściowy, Utwór_Wyjściowy)`` w ``--pairs`` (zob.
    :func:`_discover_pairs`), uruchamia :class:`ObjectiveEvaluator` i zapisuje
    ``objective_report.json`` do katalogu ``reports/`` eksperymentu
    (Wymaganie 6.5).

    Returns:
        :data:`EXIT_SUCCESS` po zapisaniu raportu ewaluacji.
    """
    log = get_logger("cli")

    config: Config | None = None
    config_source: Path | None = None
    if args.config:
        config_source = Path(args.config)
        config = load_config(config_source)

    name_holder: Any = (
        config if config is not None else {"experiment_name": "evaluate"}
    )
    experiment_dir = _prepare_experiment_dir(
        name_holder, args.experiments_dir, config_source
    )

    style_manifest_path = Path(args.style)
    if not style_manifest_path.is_file():
        raise ConfigValidationError(
            f"Manifest *Zbioru_Stylu* nie istnieje: {style_manifest_path}."
        )
    style_manifest = from_json(style_manifest_path)

    extractor = FeatureExtractor()
    aggregated = extractor.extract_dataset(
        style_manifest, root=style_manifest_path.parent
    )

    pairs = _discover_pairs(args.pairs)

    alpha = config.evaluation.significance_alpha if config is not None else 0.05
    min_pairs = config.evaluation.min_pairs_for_test if config is not None else 10
    evaluator = ObjectiveEvaluator(
        metric=args.metric,
        alpha=alpha,
        min_pairs_for_test=min_pairs,
    )
    report = evaluator.evaluate(pairs, aggregated)
    report_path = experiment_dir / "reports" / "objective_report.json"
    evaluator.write_report(report, report_path)

    log.info(
        "zakończono ewaluację obiektywną",
        experiment_dir=str(experiment_dir),
        n_pairs=report.n_pairs,
        report=str(report_path),
    )
    return EXIT_SUCCESS


# --------------------------------------------------------------------------- #
# Budowa parsera argumentów
# --------------------------------------------------------------------------- #
def _add_experiments_dir(parser: argparse.ArgumentParser) -> None:
    """Dodaje wspólny argument ``--experiments-dir`` (nadpisanie katalogu)."""
    parser.add_argument(
        "--experiments-dir",
        dest="experiments_dir",
        default=_DEFAULT_EXPERIMENTS_DIR,
        help=(
            "Katalog nadrzędny eksperymentów (domyślnie 'experiments'). "
            "Każde uruchomienie tworzy w nim podkatalog '{name}_{ts}/' "
            "z kopią konfiguracji i commit hashem (Wymaganie 8.2)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Buduje parser ``argparse`` z czterema podkomendami CLI (zadanie 12.1).

    Returns:
        Skonfigurowany :class:`argparse.ArgumentParser`. Każdy podparser
        ustawia ``func`` (handler) przez ``set_defaults``; brak podkomendy jest
        błędem walidacji (argparse kończy z kodem 2).
    """
    parser = argparse.ArgumentParser(
        prog="midi-style",
        description=(
            "System modelowania stylu artystycznego muzyków - transfer stylu "
            "pomiędzy plikami MIDI (StarGAN/CycleGAN + algorytm genetyczny)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- acquire ------------------------------------------------------------
    acquire_parser = subparsers.add_parser(
        "acquire",
        help="Pozyskanie Zbiorów_Stylu z lokalnych katalogów (Wymaganie 1).",
    )
    acquire_parser.add_argument(
        "--config", required=True, help="Ścieżka pliku konfiguracyjnego (YAML/JSON)."
    )
    _add_experiments_dir(acquire_parser)
    acquire_parser.set_defaults(func=_cmd_acquire)

    # -- train --------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Trening Modelu_GAN (conditional/StarGAN lub per_artist/CycleGAN).",
    )
    train_parser.add_argument(
        "--config", required=True, help="Ścieżka pliku konfiguracyjnego (YAML/JSON)."
    )
    _add_experiments_dir(train_parser)
    train_parser.set_defaults(func=_cmd_train)

    # -- infer --------------------------------------------------------------
    infer_parser = subparsers.add_parser(
        "infer",
        help="Inferencja transferu stylu (Wymaganie 5).",
    )
    # Wymaganie 5.8: target_artist jest OBOWIĄZKOWY niezależnie od trybu.
    infer_parser.add_argument(
        "--target_artist",
        required=True,
        help=(
            "Obowiązkowy identyfikator artysty docelowego (Wymaganie 5.8); "
            "walidowany względem metadanych Punktu_Kontrolnego."
        ),
    )
    infer_parser.add_argument(
        "--checkpoint", required=True, help="Ścieżka Punktu_Kontrolnego (.pt)."
    )
    infer_parser.add_argument(
        "--input", required=True, help="Ścieżka Utworu_Wejściowego (MIDI)."
    )
    infer_parser.add_argument(
        "--output", required=True, help="Ścieżka docelowa Utworu_Wyjściowego (MIDI)."
    )
    infer_parser.add_argument(
        "--seed", type=int, default=None, help="Opcjonalny Seed (determinizm)."
    )
    infer_parser.add_argument(
        "--config",
        default=None,
        help=(
            "Opcjonalna konfiguracja - geometria generatora / pianorolla musi "
            "odpowiadać Punktowi_Kontrolnemu."
        ),
    )
    _add_experiments_dir(infer_parser)
    infer_parser.set_defaults(func=_cmd_infer)

    # -- evaluate -----------------------------------------------------------
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Ewaluacja obiektywna par (Utwór_Wejściowy, Utwór_Wyjściowy).",
    )
    evaluate_parser.add_argument(
        "--pairs",
        required=True,
        help=(
            "Katalog z parami: podkatalogi 'input/' i 'output/' (dopasowanie po "
            "nazwie) albo pliki '<nazwa>_input.mid' / '<nazwa>_output.mid'."
        ),
    )
    evaluate_parser.add_argument(
        "--style",
        required=True,
        help="Ścieżka Manifestu_Zbioru artysty docelowego (JSON).",
    )
    evaluate_parser.add_argument(
        "--metric",
        choices=("euclidean", "mahalanobis"),
        default="euclidean",
        help="Metryka odległości Wektorów_Cech (domyślnie 'euclidean').",
    )
    evaluate_parser.add_argument(
        "--config",
        default=None,
        help="Opcjonalna konfiguracja (alpha, min_pairs_for_test, nazwa eksperymentu).",
    )
    _add_experiments_dir(evaluate_parser)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    return parser


# --------------------------------------------------------------------------- #
# Punkt wejścia
# --------------------------------------------------------------------------- #
def _coerce_exit_code(code: Any) -> int:
    """Normalizuje kod z :class:`SystemExit` do liczby całkowitej.

    ``argparse`` kończy działanie przez ``SystemExit`` - z kodem ``0`` dla
    ``--help`` oraz ``2`` dla błędów argumentów (w tym brakującego, obowiązkowego
    ``--target_artist``). Komunikat tekstowy (``str``) traktujemy jako błąd
    walidacji (kod 2).
    """
    if code is None:
        return EXIT_SUCCESS
    if isinstance(code, int):
        return code
    return EXIT_VALIDATION


def main(argv: Sequence[str] | None = None) -> int:
    """Punkt wejścia CLI ``midi-style`` zwracający **kod wyjścia** (zadanie 12.1).

    Funkcja parsuje argumenty, wywołuje handler wybranej podkomendy i mapuje
    zgłoszone wyjątki na kody wyjścia procesu (sekcja *Error Handling*):

    * :data:`EXIT_SUCCESS` (0) - powodzenie,
    * :data:`EXIT_RESOURCE` (3) - :class:`GpuOutOfMemoryError`,
    * :data:`EXIT_VALIDATION` (2) - :class:`MusiciansStyleError` z
      ``exit_code == 2`` (walidacja/konfiguracja/logika domeny) oraz brak pliku
      wejściowego (:class:`FileNotFoundError`),
    * :data:`EXIT_UNCAUGHT` (1) - dowolny inny, nieobsłużony wyjątek.

    Args:
        argv: lista argumentów (bez nazwy programu); ``None`` → ``sys.argv[1:]``.

    Returns:
        Kod wyjścia procesu (``int``).
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help (0) lub błąd argumentów (2)
        return _coerce_exit_code(exc.code)

    log = get_logger("cli")
    try:
        return int(args.func(args))
    except GpuOutOfMemoryError as exc:
        # Wymaganie 3.9: błąd zasobów GPU → kod wyjścia 3 wraz z sugestią.
        log.error("błąd zasobów GPU (OOM)", reason=str(exc), exit_code=exc.exit_code)
        print(f"Błąd zasobów (OOM): {exc}", file=sys.stderr)
        return EXIT_RESOURCE
    except MusiciansStyleError as exc:
        # Wyjątki domenowe niosą właściwy exit_code (zwykle 2 - walidacja).
        log.error(
            "błąd walidacji/logiki domeny",
            reason=str(exc),
            exit_code=exc.exit_code,
        )
        print(f"Błąd: {exc}", file=sys.stderr)
        return int(exc.exit_code)
    except FileNotFoundError as exc:
        # Brak pliku wejściowego (np. Punkt_Kontrolny, Manifest) - błąd wejścia.
        log.error("brak pliku wejściowego", reason=str(exc))
        print(f"Błąd: nie znaleziono pliku: {exc}", file=sys.stderr)
        return EXIT_VALIDATION
    except Exception as exc:  # noqa: BLE001 - świadomy catch-all dla kodu 1
        log.error("nieobsłużony wyjątek", reason=repr(exc))
        print(f"Nieoczekiwany błąd: {exc}", file=sys.stderr)
        return EXIT_UNCAUGHT


if __name__ == "__main__":  # pragma: no cover - cienka osłona procesu
    sys.exit(main())
