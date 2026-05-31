"""Model_GAN_Per_Artysta - generator i dyskryminator CycleGAN (Wymagania 3.1, 3.13).

Moduł implementuje tryb **fallback** *Modelu_GAN* (Wymaganie 3.13) w postaci
sieci typu **CycleGAN** (Zhu et al., 2017; Brunner et al.,
[arXiv:1809.07575](https://arxiv.org/abs/1809.07575), sekcja *Model_GAN_Per_Artysta*
w ``design.md``). W odróżnieniu od wielodomenowego StarGAN, CycleGAN trenuje
osobną instancję modelu dla wskazanego artysty względem zbioru kontrolnego,
dlatego sieć **nie jest warunkowana** *Etykietą_Artysty* ``c`` i dyskryminator
**nie posiada** głowicy klasyfikacji domeny ``D_cls``.

Komponenty
----------
* :class:`CycleGANGenerator` - generator ``G(x)`` o strukturze warstwowej
  identycznej z :class:`~musicians_style.models.stargan.StarGANGenerator`
  (*down-sampling → bottleneck residualny → up-sampling → sigmoid*), lecz bez
  konkatenacji mapy *Etykiety_Artysty* na wejściu.
* :class:`CycleGANDiscriminator` - dyskryminator typu PatchGAN z pojedynczą
  głowicą ``D_src`` (mapa real/fake); brak głowicy ``D_cls``.

Reprezentacja wejścia
---------------------
Generator operuje na pianorollu ``x`` o kształcie ``[B, 1, T, P]`` (domyślnie
``T = 64`` kroków czasowych, ``P = 84`` wysokości C1..B7 - zob.
:mod:`musicians_style.midi.pianoroll`). Brak warunkowania oznacza, że wejściem
jest wyłącznie pianoroll (jeden kanał), bez dodatkowych kanałów *Etykiety_Artysty*.

Blok residualny :class:`ResidualBlock` jest reużywany z modułu
:mod:`musicians_style.models.stargan`, aby uniknąć duplikacji architektury
wąskiego gardła.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .stargan import ResidualBlock

__all__ = ["CycleGANGenerator", "CycleGANDiscriminator"]

# Domyślne wymiary pianorolla oczekiwane przez sieć (sekcja Model_GAN design.md):
# T = 64 kroki czasowe (cztery takty 4/4), P = 84 wysokości (C1..B7).
_DEFAULT_INPUT_SIZE = (64, 84)
# Domyślna liczba kanałów bazowych pierwszej warstwy konwolucyjnej.
_DEFAULT_CONV_DIM = 64
# Domyślna liczba bloków residualnych w wąskim gardle generatora.
_DEFAULT_RESIDUAL_BLOCKS = 6
# Domyślna liczba warstw konwolucyjnych backbone'u dyskryminatora (PatchGAN).
_DEFAULT_DISCRIMINATOR_LAYERS = 5
# Nachylenie LeakyReLU dyskryminatora (sekcja Model_GAN design.md).
_DEFAULT_LEAKY_SLOPE = 0.2


class CycleGANGenerator(nn.Module):
    """Generator niewarunkowany ``G(x)`` sieci CycleGAN (Wymagania 3.1, 3.13).

    Architektura jest **identyczna warstwowo** z
    :class:`~musicians_style.models.stargan.StarGANGenerator`, lecz pozbawiona
    warunkowania na *Etykiecie_Artysty* - wejściem jest sam pianoroll
    (``in_channels`` kanałów, domyślnie ``1``), bez konkatenowanej mapy ``c``:

    #. **Warstwa wejściowa** - Conv2D 7×7 (stride 1) + InstanceNorm + ReLU,
       mapująca ``in_channels`` kanałów wejścia na ``conv_dim`` kanałów cech.
    #. **Down-sampling** - 2 warstwy Conv2D (kernel 4, stride 2) + InstanceNorm
       + ReLU, podwajające liczbę kanałów i zmniejszające wymiary przestrzenne
       dwukrotnie na warstwę.
    #. **Wąskie gardło** - ``n_residual_blocks`` bloków :class:`ResidualBlock`.
    #. **Up-sampling** - 2 warstwy ConvTranspose2D (kernel 4, stride 2) +
       InstanceNorm + ReLU, odtwarzające pierwotne wymiary przestrzenne.
    #. **Warstwa wyjściowa** - Conv2D 7×7 (stride 1) + sigmoid, produkująca
       binarny pianoroll ``[B, in_channels, T, P]`` o wartościach w ``[0, 1]``.

    Dla domyślnych wymiarów ``T = 64`` i ``P = 84`` (oba parzyste) podwójny
    down-sampling i up-sampling odtwarzają dokładnie pierwotny kształt
    przestrzenny, więc wyjście ma wymiary identyczne z wejściem.

    Args:
        in_channels: liczba kanałów pianorolla wejściowego (domyślnie ``1``).
        conv_dim: liczba kanałów bazowych pierwszej warstwy (domyślnie ``64``).
        n_residual_blocks: liczba bloków residualnych (domyślnie ``6``).
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_dim: int = _DEFAULT_CONV_DIM,
        n_residual_blocks: int = _DEFAULT_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(
                f"in_channels musi być dodatnie, otrzymano: {in_channels!r}."
            )
        if conv_dim <= 0:
            raise ValueError(f"conv_dim musi być dodatnie, otrzymano: {conv_dim!r}.")
        if n_residual_blocks < 0:
            raise ValueError(
                "n_residual_blocks nie może być ujemne, otrzymano: "
                f"{n_residual_blocks!r}."
            )

        self.in_channels = int(in_channels)

        layers: list[nn.Module] = []

        # -- Warstwa wejściowa: in_channels -> conv_dim (bez warunkowania) ----
        layers.append(
            nn.Conv2d(
                self.in_channels,
                conv_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            )
        )
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # -- Down-sampling: 2 warstwy Conv2D stride 2 ------------------------
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim *= 2

        # -- Wąskie gardło: bloki residualne (reużycie z stargan.py) ---------
        for _ in range(int(n_residual_blocks)):
            layers.append(ResidualBlock(curr_dim))

        # -- Up-sampling: 2 warstwy ConvTranspose2D stride 2 -----------------
        for _ in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim //= 2

        # -- Warstwa wyjściowa: conv_dim -> in_channels ----------------------
        layers.append(
            nn.Conv2d(
                curr_dim,
                self.in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            )
        )
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Generuje pianoroll przekształcony w stronę stylu artysty docelowego.

        W trybie per-artysta domena docelowa jest stała (zakodowana w wagach
        wytrenowanego generatora), więc - inaczej niż w StarGAN - nie ma
        argumentu *Etykiety_Artysty* ``c``.

        Args:
            x: pianoroll wejściowy ``[B, in_channels, T, P]``. Dopuszczalny jest
                również kształt ``[B, T, P]`` (zakłada się wtedy
                ``in_channels == 1`` i dodaje wymiar kanału).

        Returns:
            Pianoroll ``[B, in_channels, T, P]`` o wartościach w ``[0, 1]``
            (wyjście sigmoidu).
        """
        if x.dim() == 3:
            # Wygoda: [B, T, P] -> [B, 1, T, P].
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(
                "x musi mieć kształt [B, C, T, P] lub [B, T, P], otrzymano "
                f"liczbę wymiarów: {x.dim()}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"x musi mieć {self.in_channels} kanał(ów), otrzymano: {x.size(1)}."
            )
        return self.main(x)


class CycleGANDiscriminator(nn.Module):
    """Dyskryminator CycleGAN typu PatchGAN z pojedynczą głowicą (Wymaganie 3.13).

    Architektura backbone'u jest identyczna z
    :class:`~musicians_style.models.stargan.StarGANDiscriminator`, lecz - zgodnie
    z trybem fallback - sieć posiada **wyłącznie** głowicę ``D_src`` (mapa
    real/fake) i **nie** posiada głowicy klasyfikacji domeny ``D_cls``:

    #. **Backbone PatchGAN** - ``n_layers`` warstw Conv2D (kernel 4, stride 2)
       z aktywacją LeakyReLU (nachylenie ``0.2``), kolejno podwajających liczbę
       kanałów. Brak normalizacji (zgodnie z konwencją PatchGAN dla 70×70).
    #. **Głowica ``D_src``** - Conv2D 3×3 (stride 1) redukująca cechy do mapy
       jednego kanału ``[B, 1, h, w]`` (prawdopodobieństwo real/fake na poziomie
       receptywnych łat ~70×70).

    Args:
        in_channels: liczba kanałów pianorolla wejściowego (domyślnie ``1``).
        conv_dim: liczba kanałów pierwszej warstwy konwolucyjnej (domyślnie ``64``).
        n_layers: liczba warstw backbone'u PatchGAN (domyślnie ``5``).
        leaky_slope: nachylenie LeakyReLU (domyślnie ``0.2``).
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_dim: int = _DEFAULT_CONV_DIM,
        n_layers: int = _DEFAULT_DISCRIMINATOR_LAYERS,
        leaky_slope: float = _DEFAULT_LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(
                f"in_channels musi być dodatnie, otrzymano: {in_channels!r}."
            )
        if conv_dim <= 0:
            raise ValueError(f"conv_dim musi być dodatnie, otrzymano: {conv_dim!r}.")
        if n_layers <= 0:
            raise ValueError(f"n_layers musi być dodatnie, otrzymano: {n_layers!r}.")

        self.in_channels = int(in_channels)
        self.n_layers = int(n_layers)

        layers: list[nn.Module] = []
        layers.append(
            nn.Conv2d(self.in_channels, conv_dim, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
        curr_dim = conv_dim
        for _ in range(1, self.n_layers):
            layers.append(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
            curr_dim *= 2
        self.backbone = nn.Sequential(*layers)

        # Głowica D_src: mapa real/fake (1 kanał, zachowuje wymiary przestrzenne).
        # Brak głowicy D_cls - to odróżnia CycleGAN od warunkowanego StarGAN.
        self.conv_src = nn.Conv2d(
            curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Ocenia pianoroll: zwraca wyłącznie mapę real/fake ``D_src``.

        Args:
            x: pianoroll ``[B, in_channels, T, P]`` lub ``[B, T, P]``
                (auto-rozszerzany do jednego kanału).

        Returns:
            Mapa PatchGAN ``D_src`` ``[B, 1, h, w]`` (logity real/fake).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(
                "x musi mieć kształt [B, C, T, P] lub [B, T, P], otrzymano "
                f"liczbę wymiarów: {x.dim()}."
            )

        h = self.backbone(x)
        return self.conv_src(h)
