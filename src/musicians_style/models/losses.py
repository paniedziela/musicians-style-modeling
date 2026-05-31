"""Funkcje straty *Modelu_GAN* (StarGAN / CycleGAN) - Wymaganie 3.12.

Moduł implementuje składniki funkcji celu treningu sieci generatywnej zgodnie z
sekcją *Funkcje straty* w ``design.md`` oraz pracą
[arXiv:1809.07575](https://arxiv.org/abs/1809.07575) (transfer gatunku
muzycznego z CycleGAN/StarGAN). Realizowane są cztery składniki:

* :func:`adversarial_loss` - **strata adwersaryjna** ``L_adv`` (wariant
  najmniejszych kwadratów / LSGAN, stabilniejszy numerycznie od logarytmicznego),
* :func:`domain_classification_loss_real` / :func:`domain_classification_loss_fake`
  - **strata klasyfikacji domeny** ``L_cls`` (entropia krzyżowa głowicy ``D_cls``
  dyskryminatora dla próbek rzeczywistych i generowanych),
* :func:`cycle_consistency_loss` - **strata spójności cyklicznej** ``L_cyc``
  (norma L1 rekonstrukcji ``G(G(x, c_target), c_orig) ≈ x``),
* :func:`identity_loss` - **strata tożsamości** ``L_id`` (norma L1
  ``G(x, c_orig) ≈ x``).

Łączna strata generatora (Wymaganie 3.12)::

    L = L_adv + λ_cls * L_cls + λ_cyc * L_cyc + λ_id * L_id

z domyślnymi wagami ``λ_cls = 1``, ``λ_cyc = 10``, ``λ_id = 5`` (wartości z
[1809.07575](https://arxiv.org/abs/1809.07575)). Wagi są konfigurowalne i
domyślnie pobierane z :class:`~musicians_style.config.LossWeights`
(:func:`combined_generator_loss`).

Konwencje
=========

* Wszystkie funkcje zwracają **skalarny** tensor straty (``Tensor`` o zerowej
  liczbie wymiarów), zredukowany przez średnią po wsadzie - gotowy do wywołania
  ``.backward()``.
* Generator ``G`` ma sygnaturę ``G(x, c) -> Tensor`` (StarGAN warunkowany na
  *Etykiecie_Artysty* ``c``). Funkcje strat cyklicznej i tożsamości przyjmują
  generator jako argument wywoływalny, co umożliwia wstrzyknięcie zaślepki w
  testach.
* :func:`domain_classification_loss_*` operują na **logitach** głowicy ``D_cls``
  (``[B, N]``) i przyjmują *Etykietę_Artysty* w postaci wektora one-hot
  ``[B, N]`` lub indeksów klas ``[B]``.
* Strata adwersaryjna korzysta z wariantu LSGAN: ``D_src`` traktowany jest jako
  surowa ocena (logit/aktywacja) zbliżana do 1 dla próbek rzeczywistych i do 0
  dla generowanych. Funkcja zwraca składniki dla dyskryminatora oraz - przez
  :func:`generator_adversarial_loss` - składnik dla generatora.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "DEFAULT_LAMBDA_CLS",
    "DEFAULT_LAMBDA_CYC",
    "DEFAULT_LAMBDA_ID",
    "adversarial_loss",
    "generator_adversarial_loss",
    "domain_classification_loss_real",
    "domain_classification_loss_fake",
    "cycle_consistency_loss",
    "identity_loss",
    "combined_generator_loss",
]

#: Domyślne wagi składników straty (Wymaganie 3.12, [1809.07575]).
DEFAULT_LAMBDA_CLS: float = 1.0
DEFAULT_LAMBDA_CYC: float = 10.0
DEFAULT_LAMBDA_ID: float = 5.0

#: Generator warunkowany ``G(x, c) -> Tensor`` (StarGAN). Wstrzykiwalny w testach.
Generator = Callable[[Tensor, Tensor], Tensor]


def _labels_to_index(c: Tensor) -> Tensor:
    """Konwertuje *Etykietę_Artysty* do indeksów klas dla entropii krzyżowej.

    Akceptuje zarówno wektor one-hot ``[B, N]`` (zwraca ``argmax`` po wymiarze
    klas), jak i gotowe indeksy ``[B]`` (zwraca je bez zmian, rzutując na
    ``long``).
    """
    if c.dim() == 2:
        return c.argmax(dim=1).long()
    if c.dim() == 1:
        return c.long()
    raise ValueError(
        "Etykieta_Artysty musi mieć kształt [B, N] (one-hot) lub [B] (indeksy), "
        f"otrzymano liczbę wymiarów: {c.dim()}."
    )


def adversarial_loss(d_src_real: Tensor, d_src_fake: Tensor) -> Tensor:
    """Strata adwersaryjna **dyskryminatora** ``L_adv`` (wariant LSGAN).

    Dyskryminator dąży do oceniania próbek rzeczywistych jako ``1`` i
    generowanych jako ``0``:

    ``L_adv^D = E[(D_src(x_real) - 1)²] + E[D_src(G(x, c))²]``.

    Wariant najmniejszych kwadratów (LSGAN) jest stabilniejszy niż klasyczny
    logarytmiczny i nie wymaga, by wejścia były prawdopodobieństwami (mogą być
    surowymi aktywacjami mapy PatchGAN ``D_src``).

    Args:
        d_src_real: wyjście głowicy ``D_src`` dla próbek rzeczywistych
            (dowolny kształt; redukcja przez średnią).
        d_src_fake: wyjście głowicy ``D_src`` dla próbek generowanych.

    Returns:
        Skalarny tensor straty adwersaryjnej dyskryminatora.
    """
    real_loss = torch.mean((d_src_real - 1.0) ** 2)
    fake_loss = torch.mean(d_src_fake ** 2)
    return real_loss + fake_loss


def generator_adversarial_loss(d_src_fake: Tensor) -> Tensor:
    """Składnik adwersaryjny **generatora** (wariant LSGAN).

    Generator dąży do tego, by dyskryminator oceniał próbki generowane jako
    rzeczywiste (``1``): ``L_adv^G = E[(D_src(G(x, c)) - 1)²]``.

    Args:
        d_src_fake: wyjście głowicy ``D_src`` dla próbek generowanych.

    Returns:
        Skalarny tensor adwersaryjnej straty generatora.
    """
    return torch.mean((d_src_fake - 1.0) ** 2)


def domain_classification_loss_real(d_cls: Tensor, c_real: Tensor) -> Tensor:
    """Strata klasyfikacji domeny dla próbek **rzeczywistych** (``L_cls^real``).

    Trenuje głowicę ``D_cls`` dyskryminatora do rozpoznawania *Etykiety_Artysty*
    rzeczywistej próbki: ``L_cls^real = -E[log D_cls(c_real | x_real)]``
    (entropia krzyżowa na logitach).

    Args:
        d_cls: logity klasyfikacji artysty ``[B, N]`` (wyjście głowicy ``D_cls``).
        c_real: prawdziwa *Etykieta_Artysty* - one-hot ``[B, N]`` lub indeksy
            ``[B]``.

    Returns:
        Skalarny tensor straty (uśredniona entropia krzyżowa).
    """
    target = _labels_to_index(c_real)
    return F.cross_entropy(d_cls, target)


def domain_classification_loss_fake(d_cls: Tensor, c_target: Tensor) -> Tensor:
    """Strata klasyfikacji domeny dla próbek **generowanych** (``L_cls^fake``).

    Trenuje generator (przez głowicę ``D_cls``) tak, by próbka ``G(x, c_target)``
    była klasyfikowana jako artysta docelowy ``c_target``:
    ``L_cls^fake = -E[log D_cls(c_target | G(x, c_target))]``.

    Args:
        d_cls: logity klasyfikacji artysty ``[B, N]`` dla próbki generowanej.
        c_target: docelowa *Etykieta_Artysty* - one-hot ``[B, N]`` lub indeksy
            ``[B]``.

    Returns:
        Skalarny tensor straty (uśredniona entropia krzyżowa).
    """
    target = _labels_to_index(c_target)
    return F.cross_entropy(d_cls, target)


def cycle_consistency_loss(
    g: Generator,
    x: Tensor,
    c_orig: Tensor,
    c_target: Tensor,
) -> Tensor:
    """Strata spójności cyklicznej ``L_cyc`` (norma L1, Wymaganie 3.12).

    Po przekształceniu *Utworu_Wejściowego* w stronę artysty docelowego, a
    następnie z powrotem do artysty oryginalnego, powinniśmy odtworzyć wejście:

    ``L_cyc = E[|| G(G(x, c_target), c_orig) - x ||_1]``.

    Args:
        g: generator ``G(x, c) -> Tensor`` (np. :class:`StarGANGenerator`).
        x: pianoroll wejściowy ``[B, C, T, P]`` (lub ``[B, T, P]``).
        c_orig: *Etykieta_Artysty* oryginalna ``[B, N]``.
        c_target: *Etykieta_Artysty* docelowa ``[B, N]``.

    Returns:
        Skalarny tensor straty spójności cyklicznej.
    """
    fake = g(x, c_target)
    reconstructed = g(fake, c_orig)
    return F.l1_loss(reconstructed, x)


def identity_loss(g: Generator, x: Tensor, c_orig: Tensor) -> Tensor:
    """Strata tożsamości ``L_id`` (norma L1, Wymaganie 3.12).

    Generator warunkowany na *Etykiecie_Artysty* oryginalnej powinien
    pozostawić wejście (prawie) niezmienione: ``L_id = E[|| G(x, c_orig) - x ||_1]``.

    Gdy generator zwraca dokładnie swoje wejście (zaślepka tożsamościowa) i
    ``c_target == c_orig``, strata jest równa **dokładnie 0** - co jest
    weryfikowane w testach jednostkowych (zadanie 8.4).

    Args:
        g: generator ``G(x, c) -> Tensor``.
        x: pianoroll wejściowy ``[B, C, T, P]`` (lub ``[B, T, P]``).
        c_orig: *Etykieta_Artysty* oryginalna ``[B, N]``.

    Returns:
        Skalarny tensor straty tożsamości.
    """
    identity = g(x, c_orig)
    return F.l1_loss(identity, x)


def combined_generator_loss(
    l_adv: Tensor,
    l_cls: Tensor,
    l_cyc: Tensor,
    l_id: Tensor,
    *,
    lambda_cls: float = DEFAULT_LAMBDA_CLS,
    lambda_cyc: float = DEFAULT_LAMBDA_CYC,
    lambda_id: float = DEFAULT_LAMBDA_ID,
) -> Tensor:
    """Łączna strata generatora (Wymaganie 3.12).

    Realizuje sumę ważoną składników::

        L = L_adv + λ_cls * L_cls + λ_cyc * L_cyc + λ_id * L_id

    Wagi domyślne (``λ_cls = 1``, ``λ_cyc = 10``, ``λ_id = 5``) pochodzą z
    [1809.07575](https://arxiv.org/abs/1809.07575) i odpowiadają polom
    :class:`~musicians_style.config.LossWeights`.

    Args:
        l_adv: składnik adwersaryjny generatora (:func:`generator_adversarial_loss`).
        l_cls: składnik klasyfikacji domeny dla próbek generowanych
            (:func:`domain_classification_loss_fake`).
        l_cyc: składnik spójności cyklicznej (:func:`cycle_consistency_loss`).
        l_id: składnik tożsamości (:func:`identity_loss`).
        lambda_cls: waga ``λ_cls`` straty klasyfikacji domeny.
        lambda_cyc: waga ``λ_cyc`` straty spójności cyklicznej.
        lambda_id: waga ``λ_id`` straty tożsamości.

    Returns:
        Skalarny tensor łącznej straty generatora.
    """
    return l_adv + lambda_cls * l_cls + lambda_cyc * l_cyc + lambda_id * l_id
