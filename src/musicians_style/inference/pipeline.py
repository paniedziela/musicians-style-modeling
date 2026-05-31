"""Pipeline transferu stylu (inferencja) - :class:`StyleTransferPipeline`.

Moduł implementuje *Potok inferencji* z sekcji *Pipeline inferencji* dokumentu
``design.md`` (Wymaganie 5), spinając wcześniej zbudowane komponenty Systemu w
trzy metody inferencji transferu stylu:

* :meth:`StyleTransferPipeline.infer_gan` (zadanie 10.1) - inferencja
  generatywna z *Punktu_Kontrolnego* *Modelu_GAN* z walidacją ``target_artist``
  względem metadanych (Wymagania 5.1, 5.2, 5.6-5.10).
* :meth:`StyleTransferPipeline.infer_ga` (zadanie 10.2) - inferencja ewolucyjna
  oparta na :class:`~musicians_style.ga.algorithm.GeneticAlgorithm`, działająca
  na cechach *Zbioru_Stylu* artysty docelowego (Wymagania 5.1, 5.11).
* :meth:`StyleTransferPipeline.infer_combined` (zadanie 10.3) - tryb łączony
  GAN + GA, umożliwiający kolejne wykorzystanie obu metod (np. GAN → wynik →
  GA jako post-processing) (Wymaganie 5.1).

Przepływ GAN (zadanie 10.1)
===========================

``parse → Pianoroll.from_internal → G(x, c) → Pianoroll.to_internal(template=
parsed_input) → write``. Reprezentacja *pianoroll* jest stratna (nie przechowuje
*velocity*, kanałów ani meta), dlatego rekonstrukcja używa *Utworu_Wejściowego*
jako *template* - dzięki temu tempo, metrum i siatka czasowa *Utworu_Wyjściowego*
pochodzą z wejścia (Wymagania 5.6 - długość, 5.7 - struktura metryczna).

Walidacja ``target_artist`` (Wymagania 5.8-5.10)
------------------------------------------------
Parametr ``target_artist`` jest obowiązkowy niezależnie od trybu *Punktu_
Kontrolnego* (Wymaganie 5.8). Walidacja zależy od ``metadata.mode``:

* ``"conditional"`` - jeśli ``target_artist`` nie znajduje się na liście
  ``metadata.artists`` → :class:`~musicians_style.errors.UnknownArtistError`
  (Wymaganie 5.9),
* ``"per_artist"`` - jeśli ``target_artist`` różni się od ``metadata.artists[0]``
  → :class:`~musicians_style.errors.ArtistMismatchError` (Wymaganie 5.10).

Pipeline **zgłasza** te wyjątki (każdy niesie ``exit_code == 2``); tłumaczenie
na kod wyjścia procesu (``sys.exit(2)``) należy do warstwy CLI (zadanie 12.1) -
pipeline **nigdy** nie wywołuje ``sys.exit`` samodzielnie.

Przepływ GA (zadanie 10.2)
==========================

``parse → extract_dataset(Zbiór_Stylu) → GeneticAlgorithm.run →
apply_transformation(best_genome) → write``. Funkcja
:func:`~musicians_style.ga.transformation.apply_transformation` jest czysta i
zachowuje meta-zdarzenia oraz ``ticks_per_beat`` *Utworu_Wejściowego*, więc
metrum i tempo wyniku są zachowane (Wymaganie 5.7).

Tryb łączony (zadanie 10.3)
===========================

Domyślnie GAN jest stosowany jako pierwszy, a jego wynik trafia na wejście GA
jako post-processing (``order="gan_then_ga"``). Kolejność może być odwrócona
(``order="ga_then_gan"``) lub sterowana konfiguracją. W obu fazach
*Reprezentacja_Wewnętrzna* przenosi meta-zdarzenia z *Utworu_Wejściowego*, więc
struktura metryczna jest zachowywana w całym łańcuchu.

Geometria modelu i wczytanie *Punktu_Kontrolnego*
=================================================

Format *Punktu_Kontrolnego* (sekcja *Format Punktu_Kontrolnego* w ``design.md``)
przechowuje ``metadata.mode`` i ``metadata.artists``, lecz **nie** zapisuje
geometrii sieci (liczba kanałów, bloków residualnych). Aby odtworzyć generator i
wczytać ``generator_state``, pipeline przyjmuje w konstruktorze parametry
budowy generatora (``conv_dim``, ``n_residual_blocks``, ``in_channels``) oraz
geometrię *pianorolla* (przez :class:`~musicians_style.midi.pianoroll.Pianoroll`).
Liczba artystów ``N`` jest wyznaczana z ``len(metadata.artists)``. Parametry te
**muszą** odpowiadać geometrii użytej podczas treningu - w przeciwnym razie
``load_state_dict`` zgłosi błąd niezgodności kształtów. Gdy przekazany jest
:class:`~musicians_style.config.Config`, domyślne wartości geometrii są z niego
wywodzone.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch

from ..config import Config, GAConfig
from ..data.manifest import Manifest, from_json
from ..errors import ArtistMismatchError, UnknownArtistError
from ..features.extractor import FeatureExtractor
from ..features.types import AggregatedFeatures
from ..ga.algorithm import GeneticAlgorithm, History
from ..ga.transformation import apply_transformation
from ..ga.types import Genome
from ..logging import get_logger
from ..midi.parser import MidiParser
from ..midi.pianoroll import Pianoroll
from ..midi.printer import MidiPrettyPrinter
from ..midi.types import InternalRepr
from ..models.cyclegan import CycleGANGenerator
from ..models.stargan import StarGANGenerator

__all__ = ["StyleTransferPipeline", "CombinedOrder"]

#: Dozwolone tryby *Punktu_Kontrolnego* (zgodne z metadanymi treningu).
_VALID_MODES = ("conditional", "per_artist")

#: Dozwolona kolejność metod w trybie łączonym (zadanie 10.3).
CombinedOrder = Literal["gan_then_ga", "ga_then_gan"]


class StyleTransferPipeline:
    """*Potok inferencji* transferu stylu MIDI (Wymaganie 5).

    Klasa udostępnia trzy spójne metody inferencji: generatywną (GAN),
    ewolucyjną (GA) oraz łączoną (GAN + GA). Każda metoda przyjmuje
    *Utwór_Wejściowy* w formacie MIDI, obowiązkowy ``target_artist`` oraz
    *Seed*, a zwraca ścieżkę do zapisanego *Utworu_Wyjściowego* (Standard MIDI
    File). Komponenty pomocnicze (parser, pretty printer, konwerter *pianoroll*,
    *Ekstraktor_Cech*, *Algorytm_Genetyczny*) są wstrzykiwalne, co ułatwia testy
    i pozwala zbudować bardzo mały model na CPU.

    Args:
        config: opcjonalna pełna :class:`~musicians_style.config.Config`. Gdy
            podana, służy do wywiedzenia domyślnej geometrii generatora
            (``model.generator.n_residual_blocks``), geometrii *pianorolla*
            (``model.pianoroll``) oraz domyślnej konfiguracji *Algorytmu_
            Genetycznego* (``ga``). Jawne argumenty mają pierwszeństwo.
        ga_config: konfiguracja *Algorytmu_Genetycznego* używana w
            :meth:`infer_ga` / :meth:`infer_combined`; ``None`` →
            ``config.ga`` (gdy ``config`` podany) lub domyślny :class:`GAConfig`.
        pianoroll: konwerter :class:`~musicians_style.midi.pianoroll.Pianoroll`
            ustalający geometrię wejścia *Modelu_GAN* (``window_steps``,
            ``pitch_range``, ``steps_per_beat``). ``None`` → wywiedziony z
            ``config`` lub domyślny ``Pianoroll()`` (``[1, 64, 84]``).
        parser: :class:`~musicians_style.midi.parser.MidiParser`; ``None`` →
            domyślna instancja.
        printer: :class:`~musicians_style.midi.printer.MidiPrettyPrinter`;
            ``None`` → domyślna instancja.
        extractor: :class:`~musicians_style.features.extractor.FeatureExtractor`
            wykorzystywany przez ścieżkę GA do agregacji cech *Zbioru_Stylu*;
            ``None`` → domyślna instancja.
        device: urządzenie obliczeń PyTorch (``"cpu"`` domyślnie).
        conv_dim: liczba kanałów bazowych generatora (musi odpowiadać geometrii
            *Punktu_Kontrolnego*; domyślnie ``64``).
        n_residual_blocks: liczba bloków residualnych generatora; ``None`` →
            ``config.model.generator.n_residual_blocks`` lub ``6``.
        in_channels: liczba kanałów pianorolla (domyślnie ``1``).
        logger: opcjonalny logger structlog; ``None`` → ``get_logger("inference")``.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        ga_config: GAConfig | None = None,
        pianoroll: Pianoroll | None = None,
        parser: MidiParser | None = None,
        printer: MidiPrettyPrinter | None = None,
        extractor: FeatureExtractor | None = None,
        device: str | torch.device = "cpu",
        conv_dim: int = 64,
        n_residual_blocks: int | None = None,
        in_channels: int = 1,
        logger: Any | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)

        self._pianoroll = pianoroll if pianoroll is not None else self._default_pianoroll(config)
        self._parser = parser if parser is not None else MidiParser()
        self._printer = printer if printer is not None else MidiPrettyPrinter()
        self._extractor = extractor if extractor is not None else FeatureExtractor()
        self._ga = GeneticAlgorithm(extractor=self._extractor)

        if ga_config is not None:
            self._ga_config = ga_config
        elif config is not None:
            self._ga_config = config.ga
        else:
            self._ga_config = GAConfig()

        self.conv_dim = int(conv_dim)
        if n_residual_blocks is not None:
            self.n_residual_blocks = int(n_residual_blocks)
        elif config is not None:
            self.n_residual_blocks = int(config.model.generator.n_residual_blocks)
        else:
            self.n_residual_blocks = 6
        self.in_channels = int(in_channels)

        self._log = logger if logger is not None else get_logger("inference")

    # ------------------------------------------------------------------ #
    # 10.1 - Inferencja GAN
    # ------------------------------------------------------------------ #
    def infer_gan(
        self,
        x_path: Path | str,
        target_artist: str,
        checkpoint_path: Path | str,
        seed: int | None = None,
        *,
        output_path: Path | str | None = None,
        generator: torch.nn.Module | None = None,
    ) -> Path:
        """Inferencja generatywna *Modelu_GAN* (zadanie 10.1, Wymagania 5.1-5.10).

        Wczytuje *Punkt_Kontrolny*, waliduje ``target_artist`` względem jego
        metadanych (Wymagania 5.8-5.10), a następnie wykonuje przepływ
        ``parse → from_internal → G(x, c) → to_internal(template) → write``.
        *Utwór_Wyjściowy* dziedziczy tempo, metrum i siatkę czasową
        *Utworu_Wejściowego* (Wymagania 5.6, 5.7) dzięki rekonstrukcji z
        *template*.

        Args:
            x_path: ścieżka *Utworu_Wejściowego* (MIDI).
            target_artist: obowiązkowy identyfikator artysty docelowego
                (Wymaganie 5.8).
            checkpoint_path: ścieżka *Punktu_Kontrolnego* (``.pt``).
            seed: opcjonalny *Seed* (inicjalizuje generatory liczb
                pseudolosowych dla determinizmu; inferencja GAN jest jednak
                deterministyczna względem wczytanych wag).
            output_path: docelowa ścieżka *Utworu_Wyjściowego*; ``None`` →
                ``<katalog wejścia>/<nazwa>_gan.mid``.
            generator: opcjonalny, gotowy generator wstrzykiwany zamiast
                budowanego z ``generator_state`` (wykorzystywane w testach;
                metadane *Punktu_Kontrolnego* są nadal wczytywane i walidowane).

        Returns:
            Ścieżka zapisanego *Utworu_Wyjściowego* (MIDI).

        Raises:
            UnknownArtistError: tryb ``conditional`` i nieznany ``target_artist``
                (Wymaganie 5.9; ``exit_code == 2``).
            ArtistMismatchError: tryb ``per_artist`` i niezgodny ``target_artist``
                (Wymaganie 5.10; ``exit_code == 2``).
            MidiValidationError: gdy *Utwór_Wejściowy* jest uszkodzony (Wymaganie 5.5).
        """
        self._seed_everything(seed)
        parsed_input = self._parser.parse(x_path)
        output_repr = self._run_gan(
            parsed_input, target_artist, checkpoint_path, generator=generator
        )
        target = self._resolve_output_path(x_path, output_path, "gan")
        self._printer.write(output_repr, target)
        self._log.info(
            "zapisano Utwór_Wyjściowy (GAN)",
            target_artist=target_artist,
            output=str(target),
        )
        return target

    # ------------------------------------------------------------------ #
    # 10.2 - Inferencja GA
    # ------------------------------------------------------------------ #
    def infer_ga(
        self,
        x_path: Path | str,
        target_artist: str,
        style_manifest_path: Path | str,
        seed: int = 0,
        *,
        style_root: Path | str | None = None,
        ga_config: GAConfig | None = None,
        output_path: Path | str | None = None,
        log_path: Path | str | None = None,
    ) -> Path:
        """Inferencja ewolucyjna *Algorytmu_Genetycznego* (zadanie 10.2, Wymagania 5.1, 5.11).

        Wczytuje *Manifest_Zbioru* artysty docelowego, agreguje jego
        *Wektory_Cech* (:meth:`FeatureExtractor.extract_dataset`), a następnie
        uruchamia :meth:`GeneticAlgorithm.run` optymalizujący transformację
        *Utworu_Wejściowego* ku stylowi docelowemu. Najlepszy :class:`Genome`
        jest aplikowany przez
        :func:`~musicians_style.ga.transformation.apply_transformation`, a wynik
        zapisywany przez :class:`MidiPrettyPrinter`. *Algorytm_Genetyczny* nie
        wymaga osobnego treningu modelu (Wymaganie 5.11).

        Args:
            x_path: ścieżka *Utworu_Wejściowego* (MIDI).
            target_artist: obowiązkowy identyfikator artysty docelowego
                (Wymaganie 5.8); walidowany względem ``artist_id`` *Manifestu*.
            style_manifest_path: ścieżka pliku JSON *Manifestu_Zbioru* artysty
                docelowego (*Zbiór_Stylu*).
            seed: *Seed* inicjalizujący przebieg *Algorytmu_Genetycznego*
                (determinizm, Wymaganie 4.4).
            style_root: katalog bazowy do rozwiązania względnych ścieżek wpisów
                *Manifestu*; ``None`` → katalog pliku ``style_manifest_path``.
            ga_config: konfiguracja *Algorytmu_Genetycznego*; ``None`` →
                konfiguracja z konstruktora pipeline'u.
            output_path: docelowa ścieżka *Utworu_Wyjściowego*; ``None`` →
                ``<katalog wejścia>/<nazwa>_ga.mid``.
            log_path: opcjonalna ścieżka pliku ``ga.jsonl`` z historią pokoleń.

        Returns:
            Ścieżka zapisanego *Utworu_Wyjściowego* (MIDI).

        Raises:
            UnknownArtistError: gdy ``target_artist`` nie odpowiada ``artist_id``
                *Manifestu_Zbioru* (``exit_code == 2``).
        """
        parsed_input = self._parser.parse(x_path)
        output_repr, _ = self._run_ga(
            parsed_input,
            target_artist,
            style_manifest_path,
            seed,
            style_root=style_root,
            ga_config=ga_config,
            log_path=log_path,
        )
        target = self._resolve_output_path(x_path, output_path, "ga")
        self._printer.write(output_repr, target)
        self._log.info(
            "zapisano Utwór_Wyjściowy (GA)",
            target_artist=target_artist,
            output=str(target),
        )
        return target

    # ------------------------------------------------------------------ #
    # 10.3 - Tryb łączony GAN + GA
    # ------------------------------------------------------------------ #
    def infer_combined(
        self,
        x_path: Path | str,
        target_artist: str,
        checkpoint_path: Path | str,
        style_manifest_path: Path | str,
        seed: int | None = None,
        *,
        order: CombinedOrder = "gan_then_ga",
        style_root: Path | str | None = None,
        ga_config: GAConfig | None = None,
        output_path: Path | str | None = None,
        generator: torch.nn.Module | None = None,
        log_path: Path | str | None = None,
    ) -> Path:
        """Tryb łączony GAN + GA (zadanie 10.3, Wymaganie 5.1).

        Umożliwia kolejne wykorzystanie obu metod transferu. Domyślnie *Model_GAN*
        jest stosowany jako pierwszy, a jego wynik trafia na wejście *Algorytmu_
        Genetycznego* jako post-processing (``order="gan_then_ga"``); kolejność
        może zostać odwrócona (``order="ga_then_gan"``) lub sterowana
        konfiguracją. W obu fazach *Reprezentacja_Wewnętrzna* przenosi
        meta-zdarzenia *Utworu_Wejściowego*, więc struktura metryczna jest
        zachowywana w całym łańcuchu (Wymaganie 5.7).

        Args:
            x_path: ścieżka *Utworu_Wejściowego* (MIDI).
            target_artist: obowiązkowy identyfikator artysty docelowego
                (Wymaganie 5.8), walidowany w obu fazach.
            checkpoint_path: ścieżka *Punktu_Kontrolnego* (``.pt``) dla fazy GAN.
            style_manifest_path: ścieżka *Manifestu_Zbioru* dla fazy GA.
            seed: *Seed* przekazywany do obu faz (determinizm).
            order: kolejność metod - ``"gan_then_ga"`` (domyślnie) lub
                ``"ga_then_gan"``.
            style_root: katalog bazowy *Manifestu* (jak w :meth:`infer_ga`).
            ga_config: konfiguracja *Algorytmu_Genetycznego* fazy GA.
            output_path: docelowa ścieżka *Utworu_Wyjściowego*; ``None`` →
                ``<katalog wejścia>/<nazwa>_combined.mid``.
            generator: opcjonalny generator wstrzykiwany do fazy GAN (testy).
            log_path: opcjonalna ścieżka ``ga.jsonl`` fazy GA.

        Returns:
            Ścieżka zapisanego *Utworu_Wyjściowego* (MIDI).

        Raises:
            ValueError: gdy ``order`` ma nieobsługiwaną wartość.
            UnknownArtistError, ArtistMismatchError: jak w fazach składowych.
        """
        if order not in ("gan_then_ga", "ga_then_gan"):
            raise ValueError(
                f"Nieobsługiwana kolejność trybu łączonego: {order!r}; "
                "dozwolone wartości to 'gan_then_ga' lub 'ga_then_gan'."
            )

        self._seed_everything(seed)
        parsed_input = self._parser.parse(x_path)
        ga_seed = 0 if seed is None else int(seed)

        if order == "gan_then_ga":
            stage1 = self._run_gan(
                parsed_input, target_artist, checkpoint_path, generator=generator
            )
            final_repr, _ = self._run_ga(
                stage1,
                target_artist,
                style_manifest_path,
                ga_seed,
                style_root=style_root,
                ga_config=ga_config,
                log_path=log_path,
            )
        else:
            stage1, _ = self._run_ga(
                parsed_input,
                target_artist,
                style_manifest_path,
                ga_seed,
                style_root=style_root,
                ga_config=ga_config,
                log_path=log_path,
            )
            final_repr = self._run_gan(
                stage1, target_artist, checkpoint_path, generator=generator
            )

        target = self._resolve_output_path(x_path, output_path, "combined")
        self._printer.write(final_repr, target)
        self._log.info(
            "zapisano Utwór_Wyjściowy (GAN+GA)",
            target_artist=target_artist,
            order=order,
            output=str(target),
        )
        return target

    # ------------------------------------------------------------------ #
    # Rdzeń przepływów (operują na InternalRepr, bez efektów na dysku)
    # ------------------------------------------------------------------ #
    def _run_gan(
        self,
        parsed_input: InternalRepr,
        target_artist: str,
        checkpoint_path: Path | str,
        *,
        generator: torch.nn.Module | None = None,
    ) -> InternalRepr:
        """Wykonuje przepływ GAN na *Reprezentacji_Wewnętrznej* i zwraca wynik.

        ``from_internal → G(x[, c]) → to_internal(template=parsed_input)``.
        *Template* zapewnia zachowanie tempa/metrum/velocity wyjścia
        (Wymagania 5.6, 5.7).
        """
        checkpoint = self._load_checkpoint(checkpoint_path)
        metadata = self._extract_metadata(checkpoint)
        mode = metadata["mode"]
        artists = metadata["artists"]

        artist_index = self._validate_target_artist(mode, artists, target_artist)

        if generator is None:
            generator = self._build_generator(mode, len(artists), checkpoint)
        generator = generator.to(self.device)
        generator.eval()

        roll = self._pianoroll.from_internal(parsed_input)  # [T, P] float32
        x_tensor = torch.from_numpy(np.ascontiguousarray(roll, dtype=np.float32))
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, T, P]

        with torch.no_grad():
            if mode == "conditional":
                c = self._one_hot(artist_index, len(artists)).to(self.device)
                out = generator(x_tensor, c)
            else:
                out = generator(x_tensor)

        out_roll = out.detach().cpu().numpy()[0, 0]  # [T, P]
        return self._pianoroll.to_internal(out_roll, template=parsed_input)

    def _run_ga(
        self,
        parsed_input: InternalRepr,
        target_artist: str,
        style_manifest_path: Path | str,
        seed: int,
        *,
        style_root: Path | str | None = None,
        ga_config: GAConfig | None = None,
        log_path: Path | str | None = None,
    ) -> tuple[InternalRepr, History]:
        """Wykonuje przepływ GA na *Reprezentacji_Wewnętrznej* i zwraca wynik.

        Agreguje cechy *Zbioru_Stylu*, uruchamia *Algorytm_Genetyczny* i aplikuje
        najlepszy :class:`Genome` przez ``apply_transformation`` (zachowuje meta
        i ``ticks_per_beat`` wejścia, Wymaganie 5.7).
        """
        manifest, root = self._load_style_manifest(style_manifest_path, style_root)
        self._validate_style_artist(target_artist, manifest)

        aggregated: AggregatedFeatures = self._extractor.extract_dataset(
            manifest, root=root
        )
        config = ga_config if ga_config is not None else self._ga_config

        best_genome, history = self._ga.run(
            parsed_input,
            aggregated,
            config,
            int(seed),
            log_path=log_path,
        )
        self._log.info(
            "zakończono Algorytm_Genetyczny",
            target_artist=target_artist,
            generations=history.num_generations,
            stop_reason=history.stop_reason,
            best_fitness=history.best_fitness[-1] if history.best_fitness else None,
        )
        output_repr = apply_transformation(parsed_input, best_genome)
        return output_repr, history

    # ------------------------------------------------------------------ #
    # Punkt_Kontrolny: wczytanie, metadane, budowa generatora
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_checkpoint(checkpoint_path: Path | str) -> Mapping[str, Any]:
        """Wczytuje *Punkt_Kontrolny* (``torch.load`` na CPU)."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Punkt_Kontrolny nie istnieje: {path}")
        return torch.load(str(path), map_location="cpu")

    @staticmethod
    def _extract_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        """Wydobywa i waliduje sekcję ``metadata`` *Punktu_Kontrolnego*."""
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(
                "Punkt_Kontrolny nie zawiera poprawnej sekcji 'metadata' "
                "(mode, artists) - nie można zwalidować target_artist."
            )
        mode = metadata.get("mode")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Nieobsługiwany tryb Punktu_Kontrolnego: {mode!r}; "
                f"dozwolone wartości: {', '.join(_VALID_MODES)}."
            )
        artists = list(metadata.get("artists") or [])
        if not artists:
            raise ValueError(
                "Metadane Punktu_Kontrolnego nie zawierają listy 'artists' "
                "(mapowanie Etykiety_Artysty)."
            )
        return {"mode": mode, "artists": artists}

    @staticmethod
    def _validate_target_artist(
        mode: str, artists: list[str], target_artist: str
    ) -> int:
        """Waliduje ``target_artist`` względem metadanych (Wymagania 5.9, 5.10).

        Returns:
            Indeks *Etykiety_Artysty* (kolumna one-hot) artysty docelowego.

        Raises:
            UnknownArtistError: tryb ``conditional`` i nieznany artysta.
            ArtistMismatchError: tryb ``per_artist`` i niezgodny artysta.
        """
        if mode == "conditional":
            if target_artist not in artists:
                raise UnknownArtistError(target_artist, available=artists)
            return artists.index(target_artist)
        # per_artist: dokładnie jeden obsługiwany artysta (metadata.artists[0]).
        expected = artists[0]
        if target_artist != expected:
            raise ArtistMismatchError(target_artist, expected=expected)
        return 0

    def _build_generator(
        self, mode: str, num_artists: int, checkpoint: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Buduje generator zgodny z trybem i wczytuje ``generator_state``.

        Geometria (``conv_dim``, ``n_residual_blocks``, ``in_channels``) pochodzi
        z konstruktora i musi odpowiadać geometrii treningowej (zob. docstring
        modułu). ``num_artists`` wynika z ``len(metadata.artists)``.
        """
        state = checkpoint.get("generator_state")
        if state is None:
            raise ValueError(
                "Punkt_Kontrolny nie zawiera pola 'generator_state' - nie można "
                "odtworzyć generatora. Wstrzyknij gotowy generator parametrem "
                "'generator' albo wskaż poprawny Punkt_Kontrolny."
            )
        if mode == "conditional":
            generator: torch.nn.Module = StarGANGenerator(
                num_artists=num_artists,
                in_channels=self.in_channels,
                conv_dim=self.conv_dim,
                n_residual_blocks=self.n_residual_blocks,
            )
        else:
            generator = CycleGANGenerator(
                in_channels=self.in_channels,
                conv_dim=self.conv_dim,
                n_residual_blocks=self.n_residual_blocks,
            )
        generator.load_state_dict(state)
        return generator

    # ------------------------------------------------------------------ #
    # Zbiór_Stylu (ścieżka GA)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_style_manifest(
        style_manifest_path: Path | str, style_root: Path | str | None
    ) -> tuple[Manifest, Path]:
        """Wczytuje *Manifest_Zbioru* i ustala katalog bazowy względnych ścieżek."""
        manifest_path = Path(style_manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest_Zbioru stylu nie istnieje: {manifest_path}"
            )
        manifest = from_json(manifest_path)
        root = Path(style_root) if style_root is not None else manifest_path.parent
        return manifest, root

    @staticmethod
    def _validate_style_artist(target_artist: str, manifest: Manifest) -> None:
        """Waliduje zgodność ``target_artist`` z ``artist_id`` *Manifestu_Zbioru*.

        *Algorytm_Genetyczny* działa na *Zbiorze_Stylu* artysty docelowego
        (Wymaganie 5.11); jeśli żądany artysta nie odpowiada manifestowi,
        zgłaszany jest :class:`UnknownArtistError` (``exit_code == 2``), spójnie z
        walidacją ``target_artist`` w ścieżce GAN.
        """
        if target_artist != manifest.artist_id:
            raise UnknownArtistError(target_artist, available=[manifest.artist_id])

    # ------------------------------------------------------------------ #
    # Pomocnicze
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_pianoroll(config: Config | None) -> Pianoroll:
        """Buduje konwerter *pianoroll* na podstawie ``config`` lub domyślny."""
        if config is None:
            return Pianoroll()
        pr = config.model.pianoroll
        return Pianoroll(
            pitch_range=tuple(pr.pitch_range),
            steps_per_beat=config.features.pianoroll_steps_per_beat,
            window_steps=pr.window_steps,
        )

    def _one_hot(self, index: int, num_artists: int) -> torch.Tensor:
        """Buduje wektor one-hot *Etykiety_Artysty* ``[1, N]`` (typ float)."""
        label = torch.zeros((1, num_artists), dtype=torch.float32)
        label[0, index] = 1.0
        return label

    @staticmethod
    def _seed_everything(seed: int | None) -> None:
        """Inicjalizuje generatory liczb pseudolosowych (determinizm inferencji).

        Wywoływana, gdy podano ``seed`` - mimo że inferencja GAN jest
        deterministyczna względem wczytanych wag, ustawienie *Seeda* zapewnia
        spójność z resztą Systemu (Wymaganie 8.3).
        """
        if seed is None:
            return
        seed_int = int(seed)
        torch.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
        np.random.seed(seed_int)
        random.seed(seed_int)

    @staticmethod
    def _resolve_output_path(
        x_path: Path | str, output_path: Path | str | None, suffix: str
    ) -> Path:
        """Wyznacza ścieżkę *Utworu_Wyjściowego*.

        Gdy ``output_path`` jest podana, jest używana wprost; w przeciwnym razie
        wynik jest zapisywany obok wejścia z sufiksem ``_<suffix>.mid``.
        """
        if output_path is not None:
            return Path(output_path)
        source = Path(x_path)
        return source.parent / f"{source.stem}_{suffix}.mid"
