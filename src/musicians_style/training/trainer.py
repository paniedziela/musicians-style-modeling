"""GANTrainer - *Pipeline_Treningu* StarGAN / CycleGAN (zadanie 9.2).

Moduł implementuje :class:`GANTrainer` realizujący trening *Modelu_GAN* w dwóch
trybach (Wymaganie 3.10):

* ``"conditional"`` - *Model_GAN_Warunkowany* (StarGAN): jeden generator
  ``G(x, c)`` obsługuje N artystów warunkowanych *Etykietą_Artysty* ``c``;
  dyskryminator posiada głowicę klasyfikacji domeny ``D_cls`` (Wymaganie 3.12).
* ``"per_artist"`` - *Model_GAN_Per_Artysta* (CycleGAN, tryb fallback,
  Wymaganie 3.13): niewarunkowany generator ``G(x)`` i dyskryminator z samą
  głowicą ``D_src``.

Zakres zadania 9.2
------------------
Niniejszy moduł implementuje **wyłącznie** zadanie 9.2:

* metodę :meth:`GANTrainer.train` zwracającą ścieżkę do ostatniego
  *Punktu_Kontrolnego* (Wymaganie 3.2),
* zapis *Punktu_Kontrolnego* po każdej epoce ze wszystkimi polami z sekcji
  *Format Punktu_Kontrolnego* dokumentu ``design.md`` (Wymagania 3.4, 3.14),
* logowanie wartości funkcji strat w każdej iteracji oraz metryk walidacyjnych
  po każdej epoce do pliku ``logs/train.jsonl`` (Wymaganie 3.8),
* deterministyczną inicjalizację generatorów liczb pseudolosowych z parametru
  ``seed`` (Wymagania 3.6, 3.7),
* selektywne ostrzeżenia o niewystarczającej liczności plików per-artysta w
  trybie warunkowanym (Wymaganie 3.11).

Zadania pokrewne realizowane w osobnych krokach planu (9.3 - wznawianie z
*Punktu_Kontrolnego*, 9.4 - obsługa błędu OOM) **nie** są tu implementowane.
Klasa jest jednak ustrukturyzowana tak, aby można je było dodać bez przeróbek:
budowa modeli i optymalizatorów (:meth:`_build_models`,
:meth:`_build_optimizers`) oraz serializacja *Punktu_Kontrolnego*
(:meth:`_build_checkpoint_dict`, :meth:`_capture_rng_states`) są wydzielone do
czystych metod pomocniczych, gotowych do ponownego użycia przy wczytywaniu
stanu.

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

        last_checkpoint: Path | None = None
        for epoch in range(1, epochs + 1):
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

        assert last_checkpoint is not None  # epochs > 0 gwarantuje zapis
        self._emit("INFO", "zakończono trening", checkpoint=str(last_checkpoint))
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
