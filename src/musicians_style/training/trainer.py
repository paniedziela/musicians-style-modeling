"""GANTrainer - *Pipeline_Treningu* StarGAN / CycleGAN (zadania 9.2, 9.3, 9.4).

Moduł implementuje :class:`GANTrainer` realizujący trening *Modelu_GAN* w dwóch
trybach (Wymaganie 3.10):

* ``"conditional"`` - *Model_GAN_Warunkowany* (StarGAN): jeden generator
  ``G(x, c)`` obsługuje N artystów warunkowanych *Etykietą_Artysty* ``c``;
  dyskryminator posiada głowicę klasyfikacji domeny ``D_cls`` (Wymaganie 3.12).
* ``"per_artist"`` - *Model_GAN_Per_Artysta* (CycleGAN, tryb fallback,
  Wymaganie 3.13): niewarunkowany generator ``G(x)`` i dyskryminator z samą
  głowicą ``D_src``.

Zakres zadań 9.2 - 9.4
----------------------
Niniejszy moduł implementuje:

* (9.2) metodę :meth:`GANTrainer.train` zwracającą ścieżkę do ostatniego
  *Punktu_Kontrolnego* (Wymaganie 3.2),
* (9.2) zapis *Punktu_Kontrolnego* po każdej epoce ze wszystkimi polami z sekcji
  *Format Punktu_Kontrolnego* dokumentu ``design.md`` (Wymagania 3.4, 3.14),
* (9.2) logowanie wartości funkcji strat w każdej iteracji oraz metryk
  walidacyjnych po każdej epoce do pliku ``logs/train.jsonl`` (Wymaganie 3.8),
* (9.2) deterministyczną inicjalizację generatorów liczb pseudolosowych z
  parametru ``seed`` (Wymagania 3.6, 3.7),
* (9.2) selektywne ostrzeżenia o niewystarczającej liczności plików per-artysta
  w trybie warunkowanym (Wymaganie 3.11),
* (9.3) metodę :meth:`GANTrainer.resume` wznawiającą trening z *Punktu_Kontrolnego*
  (wczytanie ``state_dict`` modeli, optymalizatorów i stanów RNG, kontynuacja od
  zapisanej epoki, walidacja zgodności ``metadata.mode``; Wymaganie 3.5),
* (9.4) obsługę błędu braku pamięci GPU: pętla treningu jest owinięta w
  ``try/except torch.cuda.OutOfMemoryError`` i przy wystąpieniu OOM zgłasza
  :class:`~musicians_style.errors.GpuOutOfMemoryError` z sugerowaną redukcją
  ``batch_size`` (Wymaganie 3.9).

Budowa modeli i optymalizatorów (:meth:`_build_models`,
:meth:`_build_optimizers`) oraz serializacja *Punktu_Kontrolnego*
(:meth:`_build_checkpoint_dict`, :meth:`_capture_rng_states`) są wydzielone do
czystych metod pomocniczych, ponownie wykorzystywanych przy wczytywaniu stanu
w :meth:`GANTrainer.resume`.

Determinizm (Wymagania 3.6, 3.7)
--------------------------------
Metoda :meth:`train` najpierw inicjalizuje wszystkie generatory liczb
pseudolosowych z parametru ``seed`` (``torch.manual_seed``,
``numpy.random.seed``, ``random.seed``, ``torch.cuda.manual_seed_all``), a
dopiero potem buduje modele (inicjalizacja wag korzysta z globalnego RNG) i
``DataLoader`` (z osobnym, zaseedowanym ``torch.Generator``). Dzięki temu dwa
uruchomienia z tym samym ``seed``, manifestem i konfiguracją produkują
*Punkty_Kontrolne* zgodne z dokładnością do tolerancji numerycznej (MSE ≤ 1e-5),
co jest weryfikowane w property teście 9.5 (poza zakresem tego zadania).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.manifest import Manifest, current_git_commit, utc_now_iso
from ..errors import GpuOutOfMemoryError
from ..logging import get_logger
from ..midi.parser import MidiParser
from ..midi.pianoroll import Pianoroll
from ..models import losses as gan_losses
from ..models.cyclegan import CycleGANDiscriminator, CycleGANGenerator
from ..models.stargan import StarGANDiscriminator, StarGANGenerator
from .dataset import MultiArtistManifest, PianorollDataset

__all__ = ["GANTrainer", "MODEL_VERSION"]

#: Dozwolone tryby pracy *Modelu_GAN* (Wymaganie 3.10).
VALID_MODES = ("conditional", "per_artist")

#: Wersja formatu modelu zapisywana w metadanych *Punktu_Kontrolnego*.
MODEL_VERSION = "1.0"


class GANTrainer:
    """Trener *Modelu_GAN* w trybie warunkowanym (StarGAN) lub per-artysta (CycleGAN).

    Trener kapsułkuje pełną pętlę treningową: budowę datasetu pianoroll,
    inicjalizację modeli i optymalizatorów, iteracyjną optymalizację adwersaryjną
    oraz zapis *Punktów_Kontrolnych* i logów. Wymiary modelu (liczba kanałów,
    bloków residualnych, warstw dyskryminatora) są wstrzykiwalne, dzięki czemu w
    testach można zbudować bardzo mały model uczony przez jedną epokę na CPU.

    Args:
        config: pełna :class:`~musicians_style.config.Config` eksperymentu.
            Wykorzystywane są sekcje ``training`` (epoki, batch, lr, optymalizator,
            beta1/beta2, device), ``model.loss_weights`` (wagi ``λ_cls``,
            ``λ_cyc``, ``λ_id``), ``model.generator`` (liczba bloków residualnych)
            oraz ``dataset.min_files_per_artist`` (próg ostrzeżeń, Wymaganie 3.11).
        mode: ``"conditional"`` lub ``"per_artist"`` (Wymaganie 3.10).
        output_dir: katalog wynikowy eksperymentu. Trener tworzy w nim podkatalogi
            ``checkpoints/`` (Punkty_Kontrolne) i ``logs/`` (``train.jsonl``).
            Wstrzykiwalny - w testach wskazuje na katalog tymczasowy.
        roots: katalog bazowy (lub mapowanie ``artist_id → katalog``) do
            rozwijania względnych ścieżek wpisów manifestu (zob.
            :class:`~musicians_style.training.dataset.PianorollDataset`).
        pianoroll: konwerter :class:`~musicians_style.midi.pianoroll.Pianoroll`
            ustalający geometrię wejścia (``window_steps``, ``n_pitches``).
            ``None`` → domyślny ``Pianoroll()`` (``[1, 64, 84]``).
        parser: :class:`~musicians_style.midi.parser.MidiParser` przekazywany do
            datasetu; ``None`` → domyślna instancja.
        conv_dim: liczba kanałów bazowych pierwszej warstwy konwolucyjnej
            (domyślnie ``64``; w testach mała wartość dla szybkości).
        n_residual_blocks: liczba bloków residualnych generatora; ``None`` →
            wartość z ``config.model.generator.n_residual_blocks``.
        disc_layers: liczba warstw backbone'u dyskryminatora PatchGAN
            (domyślnie ``5``; dla małych pianorolli należy zmniejszyć).
        logger: opcjonalny logger structlog; ``None`` → ``get_logger("trainer")``.

    Raises:
        ValueError: gdy ``mode`` jest spoza :data:`VALID_MODES`.
    """

    def __init__(
        self,
        config: Config,
        mode: Literal["conditional", "per_artist"],
        *,
        output_dir: Path | str,
        roots: Path | str | Mapping[str, Path | str] | None = None,
        pianoroll: Pianoroll | None = None,
        parser: MidiParser | None = None,
        conv_dim: int = 64,
        n_residual_blocks: int | None = None,
        disc_layers: int = 5,
        logger: Any | None = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(
                f"Niedozwolony tryb '{mode}'; dozwolone wartości: "
                f"{', '.join(VALID_MODES)}."
            )

        self.config = config
        self.mode = mode

        # Katalogi wynikowe (wstrzykiwalne) - tworzone od razu, aby logowanie
        # i zapis Punktów_Kontrolnych nie wymagały dodatkowej inicjalizacji.
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_path = self.log_dir / "train.jsonl"

        self.device = torch.device(config.training.device)

        # Parametry datasetu / geometrii pianorolla.
        self._roots = roots
        self._pianoroll = pianoroll if pianoroll is not None else Pianoroll()
        self._parser = parser
        self._input_size = (self._pianoroll.window_steps, self._pianoroll.n_pitches)

        # Parametry architektury (wstrzykiwalne dla małych modeli testowych).
        self._conv_dim = int(conv_dim)
        self._n_residual_blocks = (
            int(n_residual_blocks)
            if n_residual_blocks is not None
            else int(config.model.generator.n_residual_blocks)
        )
        self._disc_layers = int(disc_layers)

        # Wagi składników straty generatora (Wymaganie 3.12).
        self.lambda_cls = float(config.model.loss_weights.cls)
        self.lambda_cyc = float(config.model.loss_weights.cycle)
        self.lambda_id = float(config.model.loss_weights.identity)

        self._logger = logger if logger is not None else get_logger("trainer")

        # Stan budowany w train() / _build_models() (None do czasu treningu).
        self.generator: torch.nn.Module | None = None
        self.discriminator: torch.nn.Module | None = None
        self.optimizer_g: torch.optim.Optimizer | None = None
        self.optimizer_d: torch.optim.Optimizer | None = None
        self.artists: list[str] = []
        self.num_artists: int = 0
        self.insufficient_artists: list[str] = []

    # -- API publiczne -------------------------------------------------------

    def train(self, manifest: Manifest | MultiArtistManifest, seed: int) -> Path:
        """Trenuje *Model_GAN* i zwraca ścieżkę do ostatniego *Punktu_Kontrolnego*.

        Kroki (zgodnie z sekcją *Pipeline_Treningu* w ``design.md``):

        #. inicjalizacja generatorów liczb pseudolosowych z ``seed`` (Wymaganie 3.6),
        #. budowa datasetu pianoroll i (w trybie warunkowanym) kontrola liczności
           per-artysta z selektywnym ostrzeżeniem (Wymaganie 3.11),
        #. budowa modeli i optymalizatorów (po zaseedowaniu - determinizm),
        #. pętla po epokach: optymalizacja adwersaryjna z logowaniem strat w
           każdej iteracji (Wymaganie 3.8) i metryk walidacyjnych po epoce,
        #. zapis *Punktu_Kontrolnego* po każdej epoce (Wymagania 3.4, 3.14).

        Args:
            manifest: *Manifest_Zbioru* (tryb per-artysta) lub
                :class:`MultiArtistManifest` (tryb warunkowany).
            seed: *Seed* inicjalizujący wszystkie generatory liczb pseudolosowych.

        Returns:
            Ścieżka do ostatniego zapisanego *Punktu_Kontrolnego* (``.pt``).

        Raises:
            ValueError: gdy konfiguracja treningu daje zerową liczbę epok lub
                dataset jest pusty.
        """
        self._seed_everything(seed)

        dataset = PianorollDataset(
            manifest,
            roots=self._roots,
            pianoroll=self._pianoroll,
            parser=self._parser,
        )
        if len(dataset) == 0:
            raise ValueError(
                "Dataset treningowy jest pusty - brak plików w manifeście."
            )

        self.artists = dataset.artists
        self.num_artists = dataset.num_artists

        # Wymaganie 3.11: selektywne ostrzeżenia tylko w trybie warunkowanym.
        if self.mode == "conditional":
            self.insufficient_artists = self.check_artist_counts(dataset.manifest)

        # Budowa modeli PO zaseedowaniu - inicjalizacja wag korzysta z RNG
        # (Wymaganie 3.7 - powtarzalność parametrów).
        self._build_models()
        self._build_optimizers()

        # DataLoader z osobnym, zaseedowanym generatorem (deterministyczne
        # tasowanie batchy bez modyfikacji globalnego RNG).
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed))
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            generator=loader_generator,
        )

        epochs = int(self.config.training.epochs)
        if epochs <= 0:
            raise ValueError(
                f"Liczba epok musi być dodatnia, otrzymano: {epochs!r}."
            )

        history: dict[str, list[float]] = {
            "loss_g": [],
            "loss_d": [],
            "loss_cls": [],
            "loss_cyc": [],
            "val_identity_l1": [],
        }

        self._emit(
            "INFO",
            "rozpoczęto trening",
            mode=self.mode,
            epochs=epochs,
            artists=self.artists,
            num_files=len(dataset),
            seed=int(seed),
        )

        last_checkpoint = self._train_epoch_loop(
            start_epoch=1, end_epoch=epochs, loader=loader, history=history
        )

        assert last_checkpoint is not None  # epochs > 0 gwarantuje zapis
        self._emit("INFO", "zakończono trening", checkpoint=str(last_checkpoint))
        return last_checkpoint

    def resume(
        self, checkpoint_path: Path | str, manifest: Manifest | MultiArtistManifest
    ) -> Path:
        """Wznawia trening z *Punktu_Kontrolnego* i zwraca ścieżkę ostatniego (Wymaganie 3.5).

        Metoda odtwarza pełny stan treningu zapisany przez :meth:`train` /
        :meth:`_save_checkpoint`, a następnie kontynuuje naukę **bez ponownej
        inicjalizacji parametrów** (Wymaganie 3.5):

        #. wczytuje *Punkt_Kontrolny* (``torch.load``) i waliduje zgodność
           ``metadata.mode`` z trybem trenera (niezgodność → ``ValueError``),
        #. odtwarza dataset i ``DataLoader`` z przekazanego ``manifest`` (sam
           *Punkt_Kontrolny* nie przechowuje danych źródłowych - stąd parametr
           ``manifest``),
        #. buduje modele i optymalizatory (:meth:`_build_models`,
           :meth:`_build_optimizers`) i wczytuje do nich ``state_dict``
           (generator, dyskryminator, oba optymalizatory),
        #. przywraca stany generatorów liczb pseudolosowych (``torch_cpu``,
           ``torch_cuda``, ``numpy``, ``python``) zapisane w ``rng_states``,
        #. kontynuuje pętlę epok od ``checkpoint["epoch"] + 1`` do
           ``config.training.epochs`` włącznie, zapisując *Punkt_Kontrolny* po
           każdej kolejnej epoce (z obsługą OOM - Wymaganie 3.9).

        Zachowanie brzegowe: jeżeli zapisana epoka jest już **równa lub większa**
        niż ``config.training.epochs``, nie ma kolejnych epok do wykonania.
        Metoda nie trenuje wówczas ani jednej epoki i zwraca ścieżkę
        przekazanego *Punktu_Kontrolnego* (traktując go jako ostatni dostępny),
        emitując wpis informacyjny ``"wznowienie bez dodatkowych epok"``.

        Args:
            checkpoint_path: ścieżka do pliku ``.pt`` *Punktu_Kontrolnego*
                zapisanego przez :meth:`train`.
            manifest: *Manifest_Zbioru* (tryb per-artysta) lub
                :class:`MultiArtistManifest` (tryb warunkowany) opisujący ten sam
                *Zbiór_Stylu*, na którym prowadzono pierwotny trening. Wymagany do
                odtworzenia datasetu i ``DataLoader`` - *Punkt_Kontrolny* nie
                przechowuje danych źródłowych.

        Returns:
            Ścieżka do ostatniego zapisanego *Punktu_Kontrolnego* (``.pt``). Gdy
            nie wykonano żadnej dodatkowej epoki - ścieżka ``checkpoint_path``.

        Raises:
            FileNotFoundError: gdy ``checkpoint_path`` nie istnieje.
            ValueError: gdy ``metadata.mode`` *Punktu_Kontrolnego* nie zgadza się
                z trybem trenera lub gdy dataset zbudowany z ``manifest`` jest pusty.
            GpuOutOfMemoryError: gdy w trakcie wznowionego treningu wystąpi
                ``torch.cuda.OutOfMemoryError`` (Wymaganie 3.9).
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Punkt_Kontrolny nie istnieje: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Walidacja zgodności trybu z metadanymi (Wymaganie 3.5).
        metadata = checkpoint.get("metadata", {})
        checkpoint_mode = metadata.get("mode")
        if checkpoint_mode != self.mode:
            raise ValueError(
                "Niezgodność trybu Punktu_Kontrolnego z konfiguracją trenera: "
                f"Punkt_Kontrolny zapisano w trybie {checkpoint_mode!r}, a trener "
                f"działa w trybie {self.mode!r}. Wznowienie przerwane."
            )

        # Odtworzenie datasetu i loadera (Punkt_Kontrolny nie przechowuje danych).
        dataset = PianorollDataset(
            manifest,
            roots=self._roots,
            pianoroll=self._pianoroll,
            parser=self._parser,
        )
        if len(dataset) == 0:
            raise ValueError(
                "Dataset treningowy jest pusty - brak plików w manifeście "
                "przekazanym do resume()."
            )

        self.artists = dataset.artists
        self.num_artists = dataset.num_artists
        if self.mode == "conditional":
            self.insufficient_artists = self.check_artist_counts(dataset.manifest)

        # Budowa modeli/optymalizatorów i wczytanie zapisanego stanu (Wymaganie 3.5).
        self._build_models()
        self._build_optimizers()
        assert self.generator is not None and self.discriminator is not None
        assert self.optimizer_g is not None and self.optimizer_d is not None
        self.generator.load_state_dict(checkpoint["generator_state"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state"])

        # Przywrócenie stanów RNG, aby tasowanie i losowość były kontynuacją
        # pierwotnego treningu, a nie nowym przebiegiem (Wymagania 3.5, 3.6).
        self._restore_rng_states(checkpoint.get("rng_states", {}))

        # DataLoader: ziarno wyprowadzone z seedu konfiguracji - tasowanie batchy
        # jest deterministyczne, a globalny stan RNG pochodzi z Punktu_Kontrolnego.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(self.config.seed))
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            generator=loader_generator,
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        epochs = int(self.config.training.epochs)

        # Odtworzenie historii metryk z Punktu_Kontrolnego (kontynuacja serii).
        history: dict[str, list[float]] = {
            "loss_g": [],
            "loss_d": [],
            "loss_cls": [],
            "loss_cyc": [],
            "val_identity_l1": [],
        }
        saved_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(saved_history.get(key, []))

        # Przypadek brzegowy: zapisana epoka osiągnęła już docelową liczbę epok.
        if start_epoch > epochs:
            self._emit(
                "INFO",
                "wznowienie bez dodatkowych epok",
                checkpoint_epoch=int(checkpoint["epoch"]),
                configured_epochs=epochs,
            )
            return checkpoint_path

        self._emit(
            "INFO",
            "wznowiono trening z Punktu_Kontrolnego",
            mode=self.mode,
            resume_from_epoch=int(checkpoint["epoch"]),
            start_epoch=start_epoch,
            end_epoch=epochs,
            checkpoint=str(checkpoint_path),
        )

        last_checkpoint = self._train_epoch_loop(
            start_epoch=start_epoch, end_epoch=epochs, loader=loader, history=history
        )

        assert last_checkpoint is not None  # start_epoch <= epochs gwarantuje zapis
        self._emit(
            "INFO", "zakończono wznowiony trening", checkpoint=str(last_checkpoint)
        )
        return last_checkpoint

    def check_artist_counts(
        self, manifest: MultiArtistManifest | Manifest
    ) -> list[str]:
        """Zwraca artystów z niewystarczającą liczbą plików i loguje ostrzeżenia.

        Realizuje Wymaganie 3.11: w trybie warunkowanym ostrzeżenie jest
        emitowane **wyłącznie** dla artystów, dla których liczba plików jest
        mniejsza niż próg ``dataset.min_files_per_artist`` (domyślnie 30). Jeśli
        wszyscy artyści mają wystarczającą liczbę plików, zwracana jest pusta
        lista i nie emitowane jest żadne ostrzeżenie. Trening jest kontynuowany
        niezależnie od wyniku (metoda nie zgłasza wyjątku).

        Args:
            manifest: :class:`MultiArtistManifest` (lub pojedynczy
                :class:`~musicians_style.data.manifest.Manifest`, traktowany jak
                jednoelementowy *Zbiór_Wielo_Artystyczny*).

        Returns:
            Lista ``artist_id`` (w kolejności indeksów *Etykiety_Artysty*) z
            liczbą plików poniżej progu.
        """
        if isinstance(manifest, Manifest):
            multi = MultiArtistManifest([manifest])
        else:
            multi = manifest

        threshold = int(self.config.dataset.min_files_per_artist)
        insufficient: list[str] = []
        for artist_id in multi.artists:
            count = len(multi.manifests[artist_id].files)
            if count < threshold:
                insufficient.append(artist_id)
                self._logger.warning(
                    "niewystarczająca liczność plików artysty",
                    artist=artist_id,
                    count=count,
                    required=threshold,
                )
                self._emit(
                    "WARNING",
                    "niewystarczająca liczność plików artysty",
                    artist=artist_id,
                    count=count,
                    required=threshold,
                )
        return insufficient

    # -- pętla treningowa ----------------------------------------------------

    def _train_epoch_loop(
        self,
        *,
        start_epoch: int,
        end_epoch: int,
        loader: DataLoader,
        history: dict[str, list[float]],
    ) -> Path | None:
        """Wykonuje pętlę po epokach z obsługą błędu OOM (Wymagania 3.8, 3.9).

        Pętla iteruje od ``start_epoch`` do ``end_epoch`` włącznie. Po każdej
        epoce aktualizuje ``history``, loguje metryki walidacyjne i zapisuje
        *Punkt_Kontrolny*. Wspólna dla :meth:`train` (od epoki 1) oraz
        :meth:`resume` (od ``checkpoint["epoch"] + 1``), dzięki czemu logika
        epoki i obsługa OOM nie są zduplikowane.

        Obsługa OOM (Wymaganie 3.9): cała pętla jest owinięta w
        ``try/except torch.cuda.OutOfMemoryError``. Przy wystąpieniu braku pamięci
        GPU emitowane jest ostrzeżenie z sugerowaną redukcją ``batch_size`` do
        połowy bieżącej wartości, a następnie zgłaszany jest
        :class:`~musicians_style.errors.GpuOutOfMemoryError` (kod wyjścia ``3``).
        Trener **nie** wywołuje ``sys.exit`` - przełożenie wyjątku na kod wyjścia
        procesu należy do warstwy CLI (zadanie 12.1).

        Args:
            start_epoch: numer pierwszej epoki do wykonania (1-based, włącznie).
            end_epoch: numer ostatniej epoki do wykonania (włącznie).
            loader: ``DataLoader`` dostarczający batchy ``(pianoroll, etykieta)``.
            history: słownik serii metryk modyfikowany w miejscu.

        Returns:
            Ścieżka ostatniego zapisanego *Punktu_Kontrolnego* lub ``None``, gdy
            zakres ``[start_epoch, end_epoch]`` jest pusty (brak epok do wykonania).

        Raises:
            GpuOutOfMemoryError: gdy w trakcie treningu wystąpi
                ``torch.cuda.OutOfMemoryError``.
        """
        last_checkpoint: Path | None = None
        try:
            for epoch in range(start_epoch, end_epoch + 1):
                epoch_avg = self._run_epoch(epoch, loader)
                val_metric = self._validate(loader)

                history["loss_g"].append(epoch_avg.get("loss_g", 0.0))
                history["loss_d"].append(epoch_avg.get("loss_d", 0.0))
                history["loss_cls"].append(epoch_avg.get("loss_cls", 0.0))
                history["loss_cyc"].append(epoch_avg.get("loss_cyc", 0.0))
                history["val_identity_l1"].append(val_metric)

                self._emit(
                    "INFO",
                    "metryki walidacyjne epoki",
                    epoch=epoch,
                    val_identity_l1=val_metric,
                    loss_g=epoch_avg.get("loss_g", 0.0),
                    loss_d=epoch_avg.get("loss_d", 0.0),
                )

                last_checkpoint = self._save_checkpoint(epoch, history)
        except torch.cuda.OutOfMemoryError as exc:
            raise self._handle_oom(exc) from exc

        return last_checkpoint

    def _handle_oom(
        self, exc: "torch.cuda.OutOfMemoryError"
    ) -> GpuOutOfMemoryError:
        """Loguje błąd OOM i buduje :class:`GpuOutOfMemoryError` (Wymaganie 3.9).

        Sugerowany rozmiar wsadu to połowa bieżącego ``batch_size`` (minimum 1),
        zgodnie z heurystyką remediacji z sekcji *Error Handling* (``design.md``).
        Metoda zwraca gotowy wyjątek (zamiast go zgłaszać), aby wywołujący mógł
        zachować łańcuch przyczyn ``raise ... from exc``.

        Args:
            exc: oryginalny wyjątek ``torch.cuda.OutOfMemoryError``.

        Returns:
            :class:`~musicians_style.errors.GpuOutOfMemoryError` z komunikatem
            zawierającym sugerowaną redukcję ``batch_size`` (``exit_code == 3``).
        """
        current_batch_size = int(self.config.training.batch_size)
        suggested = max(current_batch_size // 2, 1)
        self._logger.error(
            "brak pamięci GPU podczas treningu",
            current_batch_size=current_batch_size,
            suggested_batch_size=suggested,
            error=str(exc),
        )
        self._emit(
            "ERROR",
            "brak pamięci GPU podczas treningu",
            current_batch_size=current_batch_size,
            suggested_batch_size=suggested,
            error=str(exc),
        )
        return GpuOutOfMemoryError(suggested_batch_size=suggested)

    def _run_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        """Wykonuje pojedynczą epokę treningu, zwracając średnie strat z epoki."""
        assert self.generator is not None and self.discriminator is not None
        self.generator.train()
        self.discriminator.train()

        totals: dict[str, float] = defaultdict(float)
        n_iter = 0
        for it, (x, c) in enumerate(loader, start=1):
            x = x.to(self.device)
            c = c.to(self.device)

            if self.mode == "conditional":
                stats = self._train_step_conditional(x, c)
            else:
                stats = self._train_step_per_artist(x)

            n_iter += 1
            for key, value in stats.items():
                totals[key] += value

            # Wymaganie 3.8: log wartości strat w każdej iteracji.
            self._emit(
                "INFO",
                "iteracja treningu",
                epoch=epoch,
                iter=it,
                loss_g=stats["loss_g"],
                loss_d=stats["loss_d"],
                loss_cls=stats.get("loss_cls", 0.0),
                loss_cyc=stats.get("loss_cyc", 0.0),
            )

        denom = max(n_iter, 1)
        return {key: value / denom for key, value in totals.items()}

    def _train_step_conditional(
        self, x: torch.Tensor, c_real: torch.Tensor
    ) -> dict[str, float]:
        """Pojedynczy krok treningu StarGAN (tryb warunkowany).

        Realizuje straty z sekcji *Funkcje straty* (``design.md``): adwersaryjną,
        klasyfikacji domeny (``D_cls``), spójności cyklicznej i tożsamości
        (Wymaganie 3.12).
        """
        assert self.generator is not None and self.discriminator is not None
        batch = x.size(0)

        # Losowe Etykiety_Artysty docelowe (deterministyczne po zaseedowaniu).
        target_idx = torch.randint(
            0, self.num_artists, (batch,), device=self.device
        )
        c_target = F.one_hot(target_idx, num_classes=self.num_artists).float()

        # --- Dyskryminator ---
        self.optimizer_d.zero_grad(set_to_none=True)
        d_src_real, d_cls_real = self.discriminator(x)
        fake = self.generator(x, c_target)
        d_src_fake, _ = self.discriminator(fake.detach())
        loss_d_adv = gan_losses.adversarial_loss(d_src_real, d_src_fake)
        loss_cls_real = gan_losses.domain_classification_loss_real(
            d_cls_real, c_real
        )
        loss_d = loss_d_adv + self.lambda_cls * loss_cls_real
        loss_d.backward()
        self.optimizer_d.step()

        # --- Generator ---
        self.optimizer_g.zero_grad(set_to_none=True)
        fake = self.generator(x, c_target)
        d_src_fake, d_cls_fake = self.discriminator(fake)
        l_adv = gan_losses.generator_adversarial_loss(d_src_fake)
        l_cls = gan_losses.domain_classification_loss_fake(d_cls_fake, c_target)
        l_cyc = gan_losses.cycle_consistency_loss(
            self.generator, x, c_real, c_target
        )
        l_id = gan_losses.identity_loss(self.generator, x, c_real)
        loss_g = gan_losses.combined_generator_loss(
            l_adv,
            l_cls,
            l_cyc,
            l_id,
            lambda_cls=self.lambda_cls,
            lambda_cyc=self.lambda_cyc,
            lambda_id=self.lambda_id,
        )
        loss_g.backward()
        self.optimizer_g.step()

        return {
            "loss_g": float(loss_g.detach()),
            "loss_d": float(loss_d.detach()),
            "loss_adv": float(l_adv.detach()),
            "loss_cls": float(l_cls.detach()),
            "loss_cyc": float(l_cyc.detach()),
            "loss_id": float(l_id.detach()),
        }

    def _train_step_per_artist(self, x: torch.Tensor) -> dict[str, float]:
        """Pojedynczy krok treningu CycleGAN (tryb per-artysta, fallback).

        W trybie per-artysta generator nie jest warunkowany *Etykietą_Artysty*.
        Stosowana jest strata adwersaryjna oraz - dla zachowania struktury
        *Utworu_Wejściowego* - strata tożsamości ``||G(x) - x||_1`` i uproszczona
        strata cykliczna ``||G(G(x)) - x||_1`` (dwukrotna aplikacja jedynego
        generatora), zgodnie z duchem spójności cyklicznej CycleGAN przy jednym
        dostępnym kierunku transformacji.
        """
        assert self.generator is not None and self.discriminator is not None

        # --- Dyskryminator ---
        self.optimizer_d.zero_grad(set_to_none=True)
        d_src_real = self.discriminator(x)
        fake = self.generator(x)
        d_src_fake = self.discriminator(fake.detach())
        loss_d = gan_losses.adversarial_loss(d_src_real, d_src_fake)
        loss_d.backward()
        self.optimizer_d.step()

        # --- Generator ---
        self.optimizer_g.zero_grad(set_to_none=True)
        fake = self.generator(x)
        d_src_fake = self.discriminator(fake)
        l_adv = gan_losses.generator_adversarial_loss(d_src_fake)
        l_id = F.l1_loss(fake, x)
        l_cyc = F.l1_loss(self.generator(fake), x)
        loss_g = l_adv + self.lambda_cyc * l_cyc + self.lambda_id * l_id
        loss_g.backward()
        self.optimizer_g.step()

        return {
            "loss_g": float(loss_g.detach()),
            "loss_d": float(loss_d.detach()),
            "loss_adv": float(l_adv.detach()),
            "loss_cls": 0.0,  # brak głowicy D_cls w trybie per-artysta
            "loss_cyc": float(l_cyc.detach()),
            "loss_id": float(l_id.detach()),
        }

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Oblicza metrykę walidacyjną epoki: średnie ``||G(x) - x||_1``.

        Metryka mierzy, jak bardzo wyjście generatora odbiega od wejścia
        (zachowanie struktury *Utworu_Wejściowego*). Jest deterministyczna i
        tania - liczona bez gradientu po całym datasecie.
        """
        assert self.generator is not None
        was_training = self.generator.training
        self.generator.eval()

        total = 0.0
        count = 0
        for x, c in loader:
            x = x.to(self.device)
            if self.mode == "conditional":
                out = self.generator(x, c.to(self.device))
            else:
                out = self.generator(x)
            total += float(F.l1_loss(out, x).item()) * x.size(0)
            count += x.size(0)

        if was_training:
            self.generator.train()
        return total / max(count, 1)

    # -- budowa modeli i optymalizatorów (reużywalne, por. zadanie 9.3) ------

    def _build_models(self) -> None:
        """Buduje generator i dyskryminator właściwe dla wybranego trybu.

        Metoda wydzielona, aby przyszłe wznawianie treningu (zadanie 9.3) mogło
        ją wywołać przed wczytaniem ``state_dict`` z *Punktu_Kontrolnego*.
        """
        if self.mode == "conditional":
            self.generator = StarGANGenerator(
                num_artists=self.num_artists,
                conv_dim=self._conv_dim,
                n_residual_blocks=self._n_residual_blocks,
            ).to(self.device)
            self.discriminator = StarGANDiscriminator(
                num_artists=self.num_artists,
                conv_dim=self._conv_dim,
                n_layers=self._disc_layers,
                input_size=self._input_size,
            ).to(self.device)
        else:
            self.generator = CycleGANGenerator(
                conv_dim=self._conv_dim,
                n_residual_blocks=self._n_residual_blocks,
            ).to(self.device)
            self.discriminator = CycleGANDiscriminator(
                conv_dim=self._conv_dim,
                n_layers=self._disc_layers,
            ).to(self.device)

    def _build_optimizers(self) -> None:
        """Tworzy optymalizatory generatora i dyskryminatora wg konfiguracji.

        Metoda wydzielona dla potrzeb przyszłego wznawiania (zadanie 9.3 wczyta
        ``optimizer_g_state`` / ``optimizer_d_state`` do tych optymalizatorów).
        """
        assert self.generator is not None and self.discriminator is not None
        self.optimizer_g = self._make_optimizer(self.generator.parameters())
        self.optimizer_d = self._make_optimizer(self.discriminator.parameters())

    def _make_optimizer(self, params: Any) -> torch.optim.Optimizer:
        """Buduje optymalizator zgodnie z ``config.training`` (Adam lub SGD)."""
        training = self.config.training
        optimizer_name = str(training.optimizer).lower()
        lr = float(training.learning_rate)
        if optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=lr)
        # Domyślnie Adam (zgodnie z domyślną konfiguracją StarGAN/CycleGAN).
        return torch.optim.Adam(
            params, lr=lr, betas=(float(training.beta1), float(training.beta2))
        )

    # -- serializacja Punktu_Kontrolnego (reużywalne, por. zadanie 9.3) ------

    def _save_checkpoint(self, epoch: int, history: Mapping[str, list[float]]) -> Path:
        """Zapisuje *Punkt_Kontrolny* epoki jako plik ``.pt`` i zwraca jego ścieżkę."""
        checkpoint = self._build_checkpoint_dict(epoch, history)
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        self._emit("INFO", "zapisano Punkt_Kontrolny", epoch=epoch, path=str(path))
        return path

    def _build_checkpoint_dict(
        self, epoch: int, history: Mapping[str, list[float]]
    ) -> dict[str, Any]:
        """Buduje słownik *Punktu_Kontrolnego* (sekcja *Format Punktu_Kontrolnego*).

        Zawiera dokładnie pola wymagane w ``design.md`` (Wymagania 3.4, 3.14):
        ``epoch``, ``generator_state``, ``discriminator_state``,
        ``optimizer_g_state``, ``optimizer_d_state``, ``rng_states``,
        ``metadata`` oraz ``history``.
        """
        assert self.generator is not None and self.discriminator is not None
        assert self.optimizer_g is not None and self.optimizer_d is not None
        return {
            "epoch": int(epoch),
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "optimizer_g_state": self.optimizer_g.state_dict(),
            "optimizer_d_state": self.optimizer_d.state_dict(),
            "rng_states": self._capture_rng_states(),
            "metadata": self._build_metadata(),
            "history": {key: list(values) for key, values in history.items()},
        }

    def _build_metadata(self) -> dict[str, Any]:
        """Buduje metadane *Punktu_Kontrolnego* (Wymaganie 3.14).

        ``artists`` to uporządkowana lista identyfikatorów artystów (indeks
        *Etykiety_Artysty* → ``artist_id``), ``config_hash`` to skrót SHA-256
        konfiguracji, a ``git_commit`` - commit repozytorium dla reprodukowalności.
        """
        return {
            "mode": self.mode,
            "artists": list(self.artists),
            "config_hash": self._config_hash(),
            "git_commit": current_git_commit(),
            "model_version": MODEL_VERSION,
            "created_at": utc_now_iso(),
        }

    @staticmethod
    def _capture_rng_states() -> dict[str, Any]:
        """Zbiera stany generatorów liczb pseudolosowych (torch/numpy/python).

        Zwraca słownik z kluczami ``torch_cpu``, ``torch_cuda``, ``numpy``,
        ``python`` zgodnie z sekcją *Format Punktu_Kontrolnego*. Pozwoli to
        zadaniu 9.3 wznowić trening dokładnie od zapisanego stanu losowości.
        """
        cuda_states: list[torch.Tensor] = []
        if torch.cuda.is_available():
            cuda_states = list(torch.cuda.get_rng_state_all())
        return {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": cuda_states,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    @staticmethod
    def _restore_rng_states(rng_states: Mapping[str, Any]) -> None:
        """Przywraca stany generatorów liczb pseudolosowych z *Punktu_Kontrolnego*.

        Operacja odwrotna do :meth:`_capture_rng_states` (zadanie 9.3): odtwarza
        stan ``torch_cpu``, ``torch_cuda`` (gdy dostępne CUDA i zapisano stany),
        ``numpy`` oraz ``python``. Dzięki temu wznowiony trening jest kontynuacją
        tego samego strumienia losowości, a nie nowym przebiegiem (Wymagania 3.5,
        3.6). Brakujące klucze są pomijane, co zapewnia odporność na *Punkty_Kontrolne*
        zapisane w środowisku o innej dostępności urządzeń.

        Args:
            rng_states: słownik stanów RNG z *Punktu_Kontrolnego* (klucze
                ``torch_cpu``, ``torch_cuda``, ``numpy``, ``python``).
        """
        torch_cpu = rng_states.get("torch_cpu")
        if torch_cpu is not None:
            # torch.set_rng_state wymaga ByteTensor na CPU.
            torch.set_rng_state(torch.as_tensor(torch_cpu, dtype=torch.uint8))

        torch_cuda = rng_states.get("torch_cuda")
        if torch_cuda and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(torch_cuda))

        numpy_state = rng_states.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        python_state = rng_states.get("python")
        if python_state is not None:
            random.setstate(python_state)

    def _config_hash(self) -> str:
        """Oblicza deterministyczny skrót SHA-256 konfiguracji eksperymentu.

        Konfiguracja jest spłaszczana do słownika (``dataclasses.asdict``) i
        serializowana do JSON z posortowanymi kluczami, dzięki czemu skrót jest
        powtarzalny niezależnie od kolejności pól.
        """
        try:
            payload = dataclasses.asdict(self.config)
        except TypeError:
            # Awaryjnie, gdy config nie jest dataclass - użyj reprezentacji.
            payload = {"repr": repr(self.config)}
        encoded = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    # -- determinizm i logowanie --------------------------------------------

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Inicjalizuje wszystkie generatory liczb pseudolosowych (Wymaganie 3.6).

        Ustawia *Seed* w PyTorch (CPU i wszystkie urządzenia CUDA), NumPy oraz
        module ``random`` biblioteki standardowej. Dodatkowo wymusza
        deterministyczne algorytmy cuDNN (bez wpływu na obliczenia CPU).
        """
        seed_int = int(seed)
        torch.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
        np.random.seed(seed_int)
        random.seed(seed_int)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _emit(self, level: str, msg: str, **fields: Any) -> None:
        """Dopisuje pojedynczy wpis JSON-lines do ``logs/train.jsonl``.

        Każdy wpis zawiera pola wymagane w sekcji *Format wpisu logu*
        (``ts``, ``level``, ``component``, ``msg``) wzbogacone o przekazane
        metryki (np. ``loss_g``, ``loss_d``).
        """
        record: dict[str, Any] = {
            "ts": utc_now_iso(),
            "level": level,
            "component": "trainer",
        }
        record.update(fields)
        record["msg"] = msg
        with self.train_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
