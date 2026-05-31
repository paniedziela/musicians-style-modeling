"""Loader konfiguracji eksperymentu (YAML / JSON).

Moduł realizuje Wymagania 8.1 i 8.6:

* 8.1 - System przyjmuje plik konfiguracyjny w formacie **YAML** lub **JSON**
  definiujący wszystkie parametry eksperymentu (sieć, trening, algorytm
  genetyczny, *Seed*).
* 8.6 - Plik konfiguracyjny zawiera sekcję ``model.mode`` ze zbioru
  {``conditional``, ``per_artist``}; dla ``conditional`` wymagana jest lista
  artystów ``model.artists`` (mapowanie identyfikator → *Etykieta_Artysty*).

Struktura danych odwzorowuje schemat z sekcji *Data Models* dokumentu
``design.md``. Wszystkie sekcje konfiguracji są reprezentowane jako
``@dataclass`` z domyślnymi wartościami, dzięki czemu wynik ``load_config`` jest
porównywalny (``==``) niezależnie od formatu źródłowego (YAML vs JSON) -
własność wykorzystywana przez test Property 17.

Walidacja obecności pól wymaganych (``seed``, ``model.mode``,
``model.artists`` dla trybu ``conditional`` oraz ``model.reference_artist`` dla
trybu ``per_artist``) zgłasza :class:`ConfigValidationError`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

# ``ConfigValidationError`` należy do hierarchii wyjątków implementowanej w
# module ``errors`` (zadanie 1.4). Aby uniknąć twardej zależności od kolejności
# realizacji zadań, importujemy go defensywnie; jeśli moduł nie jest jeszcze
# dostępny, definiujemy lokalny zamiennik o identycznej semantyce. Po ukończeniu
# zadania 1.4 import zadziała i klasa z ``errors`` stanie się źródłem prawdy.
try:  # pragma: no cover - ścieżka zależna od stanu repozytorium
    from musicians_style.errors import ConfigValidationError
except Exception:  # ImportError oraz brak symbolu w częściowo gotowym module

    class ConfigValidationError(Exception):
        """Błąd walidacji pliku konfiguracyjnego (zamiennik tymczasowy).

        Zostanie zastąpiony przez wersję z ``musicians_style.errors`` po
        ukończeniu zadania 1.4.
        """


__all__ = [
    "ConfigValidationError",
    "Config",
    "DatasetConfig",
    "FeaturesConfig",
    "ModelConfig",
    "PianorollConfig",
    "GeneratorConfig",
    "DiscriminatorConfig",
    "LossWeights",
    "TrainingConfig",
    "GAConfig",
    "EvaluationConfig",
    "load_config",
    "parse_config_text",
]

VALID_MODES = ("conditional", "per_artist")


# --------------------------------------------------------------------------- #
# Pomocnicze funkcje normalizacji
# --------------------------------------------------------------------------- #
def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    """Zwraca ``value`` jako słownik lub zgłasza błąd z czytelnym kontekstem."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            f"Sekcja '{context}' musi być mapą klucz→wartość, "
            f"otrzymano typ {type(value).__name__}."
        )
    return dict(value)


def _known_keys(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """Filtruje słownik do pól zadeklarowanych w danym ``@dataclass``."""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def _as_int_pair(value: Any, context: str, default: tuple[int, int]) -> tuple[int, int]:
    """Normalizuje listę dwuelementową (np. ``pitch_range``) do krotki int."""
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigValidationError(
            f"Pole '{context}' musi być listą dwóch liczb całkowitych, "
            f"otrzymano: {value!r}."
        )
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            f"Pole '{context}' musi zawierać liczby całkowite, otrzymano: {value!r}."
        ) from exc


# --------------------------------------------------------------------------- #
# Modele danych - sekcje konfiguracji
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    """Sekcja ``dataset`` - źródła i ograniczenia *Zbioru_Stylu*."""

    artists: dict[str, str] = field(default_factory=dict)
    min_files_per_artist: int = 30
    max_duration_s: int = 1800
    min_duration_s: int = 5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetConfig":
        data = _known_keys(cls, _require_mapping(data, "dataset"))
        artists = _require_mapping(data.get("artists"), "dataset.artists")
        return cls(
            artists={str(k): str(v) for k, v in artists.items()},
            min_files_per_artist=int(data.get("min_files_per_artist", 30)),
            max_duration_s=int(data.get("max_duration_s", 1800)),
            min_duration_s=int(data.get("min_duration_s", 5)),
        )


@dataclass
class FeaturesConfig:
    """Sekcja ``features`` - parametry *Ekstraktora_Cech* i pianorolla."""

    pitch_range: tuple[int, int] = (24, 108)
    interval_window: int = 12
    pianoroll_steps_per_beat: int = 4

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeaturesConfig":
        data = _known_keys(cls, _require_mapping(data, "features"))
        return cls(
            pitch_range=_as_int_pair(
                data.get("pitch_range"), "features.pitch_range", (24, 108)
            ),
            interval_window=int(data.get("interval_window", 12)),
            pianoroll_steps_per_beat=int(data.get("pianoroll_steps_per_beat", 4)),
        )


@dataclass
class PianorollConfig:
    """Pod-sekcja ``model.pianoroll`` - wymiary wejścia *Modelu_GAN*."""

    window_steps: int = 64
    pitch_range: tuple[int, int] = (24, 108)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PianorollConfig":
        data = _known_keys(cls, _require_mapping(data, "model.pianoroll"))
        return cls(
            window_steps=int(data.get("window_steps", 64)),
            pitch_range=_as_int_pair(
                data.get("pitch_range"), "model.pianoroll.pitch_range", (24, 108)
            ),
        )


@dataclass
class GeneratorConfig:
    """Pod-sekcja ``model.generator``."""

    n_residual_blocks: int = 6

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratorConfig":
        data = _known_keys(cls, _require_mapping(data, "model.generator"))
        return cls(n_residual_blocks=int(data.get("n_residual_blocks", 6)))


@dataclass
class DiscriminatorConfig:
    """Pod-sekcja ``model.discriminator``."""

    type: str = "patchgan"
    patch_size: int = 70

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiscriminatorConfig":
        data = _known_keys(cls, _require_mapping(data, "model.discriminator"))
        return cls(
            type=str(data.get("type", "patchgan")),
            patch_size=int(data.get("patch_size", 70)),
        )


@dataclass
class LossWeights:
    """Pod-sekcja ``model.loss_weights`` - wagi składników straty StarGAN."""

    cls: float = 1.0
    cycle: float = 10.0
    identity: float = 5.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LossWeights":
        data = _known_keys(cls, _require_mapping(data, "model.loss_weights"))
        return cls(
            cls=float(data.get("cls", 1.0)),
            cycle=float(data.get("cycle", 10.0)),
            identity=float(data.get("identity", 5.0)),
        )


@dataclass
class ModelConfig:
    """Sekcja ``model`` - tryb pracy i architektura *Modelu_GAN*.

    Pole ``mode`` ze zbioru {``conditional``, ``per_artist``} steruje resztą
    walidacji (Wymaganie 8.6).
    """

    mode: str = "conditional"
    artists: list[str] = field(default_factory=list)
    reference_artist: Optional[str] = None
    pianoroll: PianorollConfig = field(default_factory=PianorollConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        raw = _require_mapping(data, "model")
        data = _known_keys(cls, raw)
        artists_raw = data.get("artists") or []
        if not isinstance(artists_raw, (list, tuple)):
            raise ConfigValidationError(
                "Pole 'model.artists' musi być listą identyfikatorów artystów, "
                f"otrzymano typ {type(artists_raw).__name__}."
            )
        reference = data.get("reference_artist")
        return cls(
            mode=str(data.get("mode", "conditional")),
            artists=[str(a) for a in artists_raw],
            reference_artist=None if reference is None else str(reference),
            pianoroll=PianorollConfig.from_dict(data.get("pianoroll")),
            generator=GeneratorConfig.from_dict(data.get("generator")),
            discriminator=DiscriminatorConfig.from_dict(data.get("discriminator")),
            loss_weights=LossWeights.from_dict(data.get("loss_weights")),
        )


@dataclass
class TrainingConfig:
    """Sekcja ``training`` - hiperparametry *Pipeline_Treningu*."""

    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 0.0002
    optimizer: str = "adam"
    beta1: float = 0.5
    beta2: float = 0.999
    device: str = "cuda"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        data = _known_keys(cls, _require_mapping(data, "training"))
        return cls(
            epochs=int(data.get("epochs", 200)),
            batch_size=int(data.get("batch_size", 16)),
            learning_rate=float(data.get("learning_rate", 0.0002)),
            optimizer=str(data.get("optimizer", "adam")),
            beta1=float(data.get("beta1", 0.5)),
            beta2=float(data.get("beta2", 0.999)),
            device=str(data.get("device", "cuda")),
        )


@dataclass
class GAConfig:
    """Sekcja ``ga`` - parametry *Algorytmu_Genetycznego*."""

    population_size: int = 100
    generations: int = 200
    tournament_size: int = 3
    crossover: str = "uniform"
    mutation_sigma: dict[str, float] = field(
        default_factory=lambda: {
            "transpose_semitones": 1.5,
            "rhythm_density_factor": 0.1,
            "note_duration_factor": 0.1,
            "velocity_offset": 5.0,
        }
    )
    elitism_k: int = 2
    fitness_metric: str = "euclidean"
    stagnation_generations: int = 30

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GAConfig":
        data = _known_keys(cls, _require_mapping(data, "ga"))
        sigma_raw = _require_mapping(data.get("mutation_sigma"), "ga.mutation_sigma")
        default_sigma = {
            "transpose_semitones": 1.5,
            "rhythm_density_factor": 0.1,
            "note_duration_factor": 0.1,
            "velocity_offset": 5.0,
        }
        sigma = dict(default_sigma)
        for key, value in sigma_raw.items():
            sigma[str(key)] = float(value)
        return cls(
            population_size=int(data.get("population_size", 100)),
            generations=int(data.get("generations", 200)),
            tournament_size=int(data.get("tournament_size", 3)),
            crossover=str(data.get("crossover", "uniform")),
            mutation_sigma=sigma,
            elitism_k=int(data.get("elitism_k", 2)),
            fitness_metric=str(data.get("fitness_metric", "euclidean")),
            stagnation_generations=int(data.get("stagnation_generations", 30)),
        )


@dataclass
class EvaluationConfig:
    """Sekcja ``evaluation`` - parametry ewaluacji obiektywnej i subiektywnej."""

    significance_alpha: float = 0.05
    min_pairs_for_test: int = 10
    plot_format: str = "pdf"
    soundfont: str = "soundfonts/FluidR3_GM.sf2"
    clip_seconds: int = 15

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationConfig":
        data = _known_keys(cls, _require_mapping(data, "evaluation"))
        return cls(
            significance_alpha=float(data.get("significance_alpha", 0.05)),
            min_pairs_for_test=int(data.get("min_pairs_for_test", 10)),
            plot_format=str(data.get("plot_format", "pdf")),
            soundfont=str(data.get("soundfont", "soundfonts/FluidR3_GM.sf2")),
            clip_seconds=int(data.get("clip_seconds", 15)),
        )


@dataclass
class Config:
    """Pełna konfiguracja eksperymentu (korzeń schematu *Data Models*)."""

    seed: int = 0
    experiment_name: str = "experiment"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Buduje i waliduje :class:`Config` z surowego słownika.

        Najpierw sprawdzana jest obecność pól wymaganych na surowych danych
        (przed zastosowaniem wartości domyślnych), następnie spójność trybu
        ``model.mode`` względem ``model.artists`` / ``model.reference_artist``.
        """
        if not isinstance(data, Mapping):
            raise ConfigValidationError(
                "Konfiguracja musi być mapą najwyższego poziomu (obiekt YAML/JSON), "
                f"otrzymano typ {type(data).__name__}."
            )

        _validate_presence(data)

        config = cls(
            seed=int(data["seed"]),
            experiment_name=str(data.get("experiment_name", "experiment")),
            dataset=DatasetConfig.from_dict(data.get("dataset")),
            features=FeaturesConfig.from_dict(data.get("features")),
            model=ModelConfig.from_dict(data.get("model")),
            training=TrainingConfig.from_dict(data.get("training")),
            ga=GAConfig.from_dict(data.get("ga")),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation")),
        )

        _validate_consistency(config)
        return config


# --------------------------------------------------------------------------- #
# Walidacja
# --------------------------------------------------------------------------- #
def _validate_presence(data: Mapping[str, Any]) -> None:
    """Sprawdza obecność pól wymaganych: ``seed``, ``model``, ``model.mode``."""
    if "seed" not in data:
        raise ConfigValidationError(
            "Brak wymaganego pola 'seed' w konfiguracji eksperymentu "
            "(Wymaganie 8.1 - reprodukowalność)."
        )
    seed = data["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigValidationError(
            f"Pole 'seed' musi być liczbą całkowitą, otrzymano: {seed!r}."
        )

    if "model" not in data:
        raise ConfigValidationError(
            "Brak wymaganej sekcji 'model' w konfiguracji."
        )
    model = data["model"]
    if not isinstance(model, Mapping):
        raise ConfigValidationError(
            "Sekcja 'model' musi być mapą, "
            f"otrzymano typ {type(model).__name__}."
        )
    if "mode" not in model:
        raise ConfigValidationError(
            "Brak wymaganego pola 'model.mode' (Wymaganie 8.6); "
            f"dozwolone wartości: {', '.join(VALID_MODES)}."
        )


def _validate_consistency(config: Config) -> None:
    """Sprawdza spójność trybu modelu z pozostałymi polami sekcji ``model``."""
    mode = config.model.mode
    if mode not in VALID_MODES:
        raise ConfigValidationError(
            f"Niedozwolona wartość 'model.mode': {mode!r}; "
            f"dozwolone wartości: {', '.join(VALID_MODES)}."
        )

    if mode == "conditional":
        if not config.model.artists:
            raise ConfigValidationError(
                "Tryb 'conditional' wymaga niepustej listy 'model.artists' "
                "(mapowanie identyfikator artysty → Etykieta_Artysty, Wymaganie 8.6)."
            )
    elif mode == "per_artist":
        if not config.model.reference_artist:
            raise ConfigValidationError(
                "Tryb 'per_artist' wymaga wskazania 'model.reference_artist' "
                "(zbiór kontrolny dla treningu CycleGAN, Wymaganie 3.13)."
            )


# --------------------------------------------------------------------------- #
# Parsowanie formatów
# --------------------------------------------------------------------------- #
def _parse_yaml(text: str) -> Any:
    """Parsuje tekst YAML; zgłasza czytelny błąd, gdy brak PyYAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - zależne od środowiska
        raise ConfigValidationError(
            "Obsługa konfiguracji YAML wymaga pakietu 'PyYAML'. "
            "Zainstaluj zależności (pip install -r requirements.txt) lub użyj "
            "formatu JSON."
        ) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Błąd parsowania YAML: {exc}") from exc


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Błąd parsowania JSON: {exc}") from exc


def parse_config_text(text: str, fmt: str) -> Config:
    """Parsuje konfigurację z łańcucha znaków w podanym formacie.

    Args:
        text: zawartość pliku konfiguracyjnego.
        fmt: ``"yaml"`` lub ``"json"``.

    Funkcja pomocnicza wykorzystywana m.in. przez testy równoważności
    formatów (Property 17), gdzie konfiguracja jest serializowana w pamięci.
    """
    fmt_norm = fmt.lower().lstrip(".")
    if fmt_norm in ("yaml", "yml"):
        data = _parse_yaml(text)
    elif fmt_norm == "json":
        data = _parse_json(text)
    else:
        raise ConfigValidationError(
            f"Nieobsługiwany format konfiguracji: {fmt!r}; "
            "obsługiwane formaty to 'yaml' i 'json'."
        )
    return Config.from_dict(data)


def load_config(path: Path) -> Config:
    """Wczytuje i waliduje konfigurację eksperymentu z pliku.

    Format rozpoznawany jest po rozszerzeniu: ``.yaml`` / ``.yml`` → YAML,
    ``.json`` → JSON. Struktura jest normalizowana do zestawu ``@dataclass``
    i walidowana (obecność pól wymaganych oraz spójność trybu ``model.mode``).

    Args:
        path: ścieżka do pliku konfiguracyjnego YAML lub JSON.

    Returns:
        Znormalizowany i zwalidowany obiekt :class:`Config`.

    Raises:
        ConfigValidationError: gdy plik nie istnieje, ma nieobsługiwane
            rozszerzenie, jest niepoprawny składniowo lub nie spełnia
            ograniczeń schematu.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigValidationError(f"Plik konfiguracyjny nie istnieje: {path}")
    if not path.is_file():
        raise ConfigValidationError(
            f"Ścieżka konfiguracji nie wskazuje na plik: {path}"
        )

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        fmt = "yaml"
    elif suffix == ".json":
        fmt = "json"
    else:
        raise ConfigValidationError(
            f"Nieobsługiwane rozszerzenie pliku konfiguracyjnego: {suffix!r} "
            f"(plik: {path.name}); obsługiwane: .yaml, .yml, .json."
        )

    text = path.read_text(encoding="utf-8")
    return parse_config_text(text, fmt)
