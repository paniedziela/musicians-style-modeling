"""Model_GAN_Warunkowany - generator i dyskryminator StarGAN (Wymagania 3.1, 3.12).

Moduł implementuje architekturę sieci wielodomenowej typu **StarGAN**
(Choi et al., 2018), zaadaptowaną do binarnych pianorolli zgodnie z pracą
[arXiv:1809.07575](https://arxiv.org/abs/1809.07575) (sekcja *Model_GAN* w
``design.md``). Pojedynczy generator ``G(x, c)`` przekształca *Utwór_Wejściowy*
``x`` w stronę stylu artysty wskazanego *Etykietą_Artysty* ``c``, dzięki czemu
jeden model obsługuje N artystów bez przełączania *Punktów_Kontrolnych*.

Komponenty
----------
* :class:`ResidualBlock` - blok residualny (dwie warstwy Conv2D + InstanceNorm
  + ReLU z połączeniem skip) używany w wąskim gardle generatora.
* :class:`StarGANGenerator` - generator ``G(x, c)`` o strukturze
  *down-sampling → bottleneck residualny → up-sampling → sigmoid*.
* :class:`StarGANDiscriminator` - dyskryminator typu PatchGAN z dwiema
  głowicami: ``D_src`` (mapa real/fake) oraz ``D_cls`` (klasyfikacja artysty,
  Wymaganie 3.12).

Reprezentacja wejścia
---------------------
Generator operuje na pianorollu ``x`` o kształcie ``[B, 1, T, P]`` (domyślnie
``T = 64`` kroków czasowych, ``P = 84`` wysokości C1..B7 - zob.
:mod:`musicians_style.midi.pianoroll`). *Etykieta_Artysty* ``c`` to wektor
one-hot ``[B, N]``, który jest rozciągany przestrzennie do mapy ``[B, N, T, P]``
i konkatenowany z pianorollem po wymiarze kanałów → ``[B, 1 + N, T, P]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["ResidualBlock", "StarGANGenerator", "StarGANDiscriminator"]

# Domyślne wymiary pianorolla oczekiwane przez sieć (sekcja Model_GAN design.md):
# T = 64 kroki czasowe (cztery takty 4/4), P = 84 wysokości (C1..B7).
_DEFAULT_INPUT_SIZE = (64, 84)
# Domyślna liczba artystów N - zgodna z przykładową konfiguracją
# (configs/default_conditional.yaml: beatles, queen, abba). W praktyce N
# pochodzi z pola model.artists pliku konfiguracyjnego.
_DEFAULT_NUM_ARTISTS = 3
# Domyślna liczba kanałów bazowych pierwszej warstwy konwolucyjnej.
_DEFAULT_CONV_DIM = 64
# Domyślna liczba bloków residualnych w wąskim gardle generatora.
_DEFAULT_RESIDUAL_BLOCKS = 6
# Domyślna liczba warstw konwolucyjnych backbone'u dyskryminatora (PatchGAN).
_DEFAULT_DISCRIMINATOR_LAYERS = 5
# Nachylenie LeakyReLU dyskryminatora (sekcja Model_GAN design.md).
_DEFAULT_LEAKY_SLOPE = 0.2


class ResidualBlock(nn.Module):
    """Blok residualny wąskiego gardła generatora StarGAN.

    Realizuje przekształcenie ``y = x + F(x)``, gdzie ``F`` to sekwencja
    Conv2D → InstanceNorm → ReLU → Conv2D → InstanceNorm (zachowująca wymiary
    przestrzenne i liczbę kanałów). Połączenie skip stabilizuje propagację
    gradientu w głębokim wąskim gardle.

    Args:
        dim: liczba kanałów wejścia i wyjścia (niezmieniana przez blok).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim musi być dodatnie, otrzymano: {dim!r}.")
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102 - opisane w docstringu klasy
        return x + self.block(x)


class StarGANGenerator(nn.Module):
    """Generator warunkowany ``G(x, c)`` sieci StarGAN (Wymaganie 3.1).

    Architektura (sekcja *Model_GAN_Warunkowany* w ``design.md``):

    #. **Warstwa wejściowa** - Conv2D 7×7 (stride 1) + InstanceNorm + ReLU,
       mapująca ``1 + N`` kanałów wejścia (pianoroll + mapa *Etykiety_Artysty*)
       na ``conv_dim`` kanałów cech.
    #. **Down-sampling** - 2 warstwy Conv2D (kernel 4, stride 2) + InstanceNorm
       + ReLU, podwajające liczbę kanałów i zmniejszające wymiary przestrzenne
       dwukrotnie na warstwę.
    #. **Wąskie gardło** - ``n_residual_blocks`` bloków :class:`ResidualBlock`.
    #. **Up-sampling** - 2 warstwy ConvTranspose2D (kernel 4, stride 2) +
       InstanceNorm + ReLU, odtwarzające pierwotne wymiary przestrzenne.
    #. **Warstwa wyjściowa** - Conv2D 7×7 (stride 1) + sigmoid, produkująca
       binarny pianoroll ``[B, 1, T, P]`` o wartościach w ``[0, 1]``.

    Dla domyślnych wymiarów ``T = 64`` i ``P = 84`` (oba parzyste) podwójny
    down-sampling i up-sampling odtwarzają dokładnie pierwotny kształt
    przestrzenny, więc wyjście ma wymiary identyczne z wejściem.

    Args:
        num_artists: liczba artystów ``N`` (wymiar one-hot *Etykiety_Artysty*).
        in_channels: liczba kanałów pianorolla wejściowego (domyślnie ``1``).
        conv_dim: liczba kanałów bazowych pierwszej warstwy (domyślnie ``64``).
        n_residual_blocks: liczba bloków residualnych (domyślnie ``6``).
    """

    def __init__(
        self,
        num_artists: int = _DEFAULT_NUM_ARTISTS,
        in_channels: int = 1,
        conv_dim: int = _DEFAULT_CONV_DIM,
        n_residual_blocks: int = _DEFAULT_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        if num_artists <= 0:
            raise ValueError(
                f"num_artists musi być dodatnie, otrzymano: {num_artists!r}."
            )
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

        self.num_artists = int(num_artists)
        self.in_channels = int(in_channels)

        layers: list[nn.Module] = []

        # -- Warstwa wejściowa: (in_channels + N) -> conv_dim -----------------
        layers.append(
            nn.Conv2d(
                self.in_channels + self.num_artists,
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

        # -- Wąskie gardło: bloki residualne ---------------------------------
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

        # -- Warstwa wyjściowa: conv_dim -> in_channels + sigmoid ------------
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
        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Generuje pianoroll w stylu artysty ``c``.

        Args:
            x: pianoroll wejściowy ``[B, in_channels, T, P]``. Dopuszczalny jest
                również kształt ``[B, T, P]`` (zakłada się wtedy
                ``in_channels == 1`` i dodaje wymiar kanału).
            c: *Etykieta_Artysty* one-hot ``[B, N]`` (typ zmiennoprzecinkowy).

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
        if c.dim() != 2:
            raise ValueError(
                f"c musi mieć kształt [B, N], otrzymano liczbę wymiarów: {c.dim()}."
            )
        if c.size(0) != x.size(0):
            raise ValueError(
                "Rozmiar wsadu x i c musi być zgodny, otrzymano: "
                f"x={x.size(0)}, c={c.size(0)}."
            )
        if c.size(1) != self.num_artists:
            raise ValueError(
                f"c musi mieć {self.num_artists} kolumn (N), otrzymano: {c.size(1)}."
            )

        # Rozciągnięcie Etykiety_Artysty do mapy przestrzennej [B, N, T, P]
        # i konkatenacja z pianorollem po wymiarze kanałów.
        c = c.to(dtype=x.dtype)
        c_map = c.view(c.size(0), c.size(1), 1, 1)
        c_map = c_map.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c_map], dim=1)

        out = self.main(x)
        return torch.sigmoid(out)


class StarGANDiscriminator(nn.Module):
    """Dyskryminator StarGAN typu PatchGAN z dwiema głowicami (Wymaganie 3.12).

    Architektura (sekcja *Model_GAN_Warunkowany* w ``design.md``):

    #. **Backbone PatchGAN** - ``n_layers`` warstw Conv2D (kernel 4, stride 2)
       z aktywacją LeakyReLU (nachylenie ``0.2``), kolejno podwajających liczbę
       kanałów. Brak normalizacji (zgodnie z konwencją PatchGAN dla 70×70).
    #. **Głowica ``D_src``** - Conv2D 3×3 (stride 1) redukująca cechy do mapy
       jednego kanału ``[B, 1, h, w]`` (prawdopodobieństwo real/fake na poziomie
       receptywnych łat ~70×70).
    #. **Głowica ``D_cls``** - Conv2D redukująca cechy do ``[B, N, 1, 1]``,
       spłaszczana do logitów ``[B, N]`` klasyfikacji artysty.

    Aby głowica ``D_cls`` zwracała stały kształt ``[B, N]`` niezależnie od
    wahań wymiarów wejścia, jądro splotu klasyfikującego dobierane jest na
    podstawie zadeklarowanego ``input_size``; w razie rozbieżności mapa cech
    jest sprowadzana do oczekiwanego rozmiaru przez ``adaptive_avg_pool2d``.

    Args:
        num_artists: liczba klas artystów ``N`` (wymiar wyjścia ``D_cls``).
        in_channels: liczba kanałów pianorolla wejściowego (domyślnie ``1``).
        conv_dim: liczba kanałów pierwszej warstwy konwolucyjnej (domyślnie ``64``).
        n_layers: liczba warstw backbone'u PatchGAN (domyślnie ``5``).
        input_size: para ``(T, P)`` - referencyjne wymiary pianorolla używane do
            wyznaczenia jądra głowicy ``D_cls`` (domyślnie ``(64, 84)``).
        leaky_slope: nachylenie LeakyReLU (domyślnie ``0.2``).
    """

    def __init__(
        self,
        num_artists: int = _DEFAULT_NUM_ARTISTS,
        in_channels: int = 1,
        conv_dim: int = _DEFAULT_CONV_DIM,
        n_layers: int = _DEFAULT_DISCRIMINATOR_LAYERS,
        input_size: tuple[int, int] = _DEFAULT_INPUT_SIZE,
        leaky_slope: float = _DEFAULT_LEAKY_SLOPE,
    ) -> None:
        super().__init__()
        if num_artists <= 0:
            raise ValueError(
                f"num_artists musi być dodatnie, otrzymano: {num_artists!r}."
            )
        if in_channels <= 0:
            raise ValueError(
                f"in_channels musi być dodatnie, otrzymano: {in_channels!r}."
            )
        if conv_dim <= 0:
            raise ValueError(f"conv_dim musi być dodatnie, otrzymano: {conv_dim!r}.")
        if n_layers <= 0:
            raise ValueError(f"n_layers musi być dodatnie, otrzymano: {n_layers!r}.")

        self.num_artists = int(num_artists)
        self.in_channels = int(in_channels)
        self.n_layers = int(n_layers)

        layers: list[nn.Module] = []
        layers.append(
            nn.Conv2d(
                self.in_channels, conv_dim, kernel_size=4, stride=2, padding=1
            )
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

        # Wymiary mapy cech po backbone (każda warstwa stride 2: floor((W-2)/2)+1).
        feat_h = self._downsampled_size(int(input_size[0]), self.n_layers)
        feat_w = self._downsampled_size(int(input_size[1]), self.n_layers)
        if feat_h <= 0 or feat_w <= 0:
            raise ValueError(
                "input_size jest zbyt małe dla zadanej liczby warstw "
                f"(n_layers={self.n_layers}); wynikowa mapa cech ma wymiary "
                f"({feat_h}, {feat_w})."
            )
        self._feat_size = (feat_h, feat_w)

        # Głowica D_src: mapa real/fake (1 kanał, zachowuje wymiary przestrzenne).
        self.conv_src = nn.Conv2d(
            curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Głowica D_cls: redukcja do [B, N, 1, 1] -> logity [B, N].
        self.conv_cls = nn.Conv2d(
            curr_dim, self.num_artists, kernel_size=(feat_h, feat_w), bias=False
        )

    @staticmethod
    def _downsampled_size(size: int, n_layers: int) -> int:
        """Wymiar po ``n_layers`` splotach kernel 4, stride 2, padding 1."""
        for _ in range(n_layers):
            size = (size + 2 - 4) // 2 + 1
        return size

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Ocenia pianoroll: zwraca mapę real/fake i logity klasyfikacji artysty.

        Args:
            x: pianoroll ``[B, in_channels, T, P]`` lub ``[B, T, P]``
                (auto-rozszerzany do jednego kanału).

        Returns:
            Krotka ``(D_src, D_cls)``:

            * ``D_src`` - mapa PatchGAN ``[B, 1, h, w]`` (logity real/fake),
            * ``D_cls`` - logity klasyfikacji artysty ``[B, N]``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(
                "x musi mieć kształt [B, C, T, P] lub [B, T, P], otrzymano "
                f"liczbę wymiarów: {x.dim()}."
            )

        h = self.backbone(x)

        out_src = self.conv_src(h)

        # Dopasowanie wymiarów do jądra głowicy klasyfikującej, gdy wejście
        # ma inne wymiary niż referencyjne ``input_size``.
        if h.shape[-2:] != self._feat_size:
            h = F.adaptive_avg_pool2d(h, self._feat_size)
        out_cls = self.conv_cls(h)
        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))

        return out_src, out_cls
