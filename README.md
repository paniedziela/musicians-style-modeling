# Modelowanie stylu artystycznego muzyków

Repozytorium badawcze pracy inżynierskiej dotyczącej analizy stylu kompozytorskiego
i transferu stylu pomiędzy plikami MIDI.

> **Stan projektu:** prototyp badawczy, nie gotowy system. Parser MIDI, ekstrakcja
> podstawowych cech, infrastruktura eksperymentów, ewaluacja i operatory GA są
> przetestowane jednostkowo. Potok GAN wymaga jednak przebudowy przygotowania
> danych i inferencji, a tryb nazwany `per_artist` nie implementuje obecnie pełnego
> CycleGAN-u A↔B. Nie należy interpretować dotychczasowych plików wynikowych jako
> miarodajnego wyniku eksperymentu.

## Od czego zacząć

- [Audyt repozytorium](docs/AUDYT_REPOZYTORIUM.md) — co faktycznie implementuje
  kod, zgodność z celem pracy i przyczyny słabych wyników.
- [Plan naprawczy](docs/PLAN_NAPRAWCZY.md) — kolejność prac, kryteria akceptacji
  i minimalny program eksperymentów.
- [Plan eksperymentu E1](docs/E1_PLAN.md) — przygotowanie lokalnego ASAP,
  podział bez przecieku oraz etapy E1.0–E1.4.
- [Porządkowanie literatury](docs/LITERATURA.md) — kryteria redukcji 173 pozycji,
  literatura rdzeniowa i proponowana struktura przeglądu.

## Rekomendowany zakres pracy

Rdzeniem pracy powinien być następujący, weryfikowalny ciąg:

1. przygotowanie jednorodnego korpusu symbolicznego dla kilku kompozytorów;
2. ekstrakcja i selekcja cech, które rzeczywiście rozróżniają kompozytorów;
3. walidacja cech klasyfikatorem na podziale bez przecieku utworów;
4. transfer stylu przez interpretowalną, wielokryterialną optymalizację/GA;
5. porównanie z prostymi baseline'ami, a opcjonalnie z naprawionym modelem
   neuronowym.

Takie ujęcie odpowiada formalnemu celowi pracy lepiej niż uczynienie GAN-u
jedynym artefaktem. Warunkowany GAN może pozostać eksperymentem dodatkowym.

## Aktualne komponenty

| Obszar | Implementacja | Aktualny status |
|---|---|---|
| MIDI | parser, writer, pianoroll | działa, lecz pianoroll jest stratny |
| Cechy | 42 wartości: tempo, histogramy, gęstość, długości, pauzy | działa; zestaw jest zbyt mały do tezy o stylu |
| `conditional` | pojedynczy generator wielodomenowy inspirowany StarGAN | prototyp; wymaga segmentacji, walidacji i stabilizacji |
| `per_artist` | pojedynczy generator i dyskryminator | **nie jest pełnym CycleGAN-em** |
| GA | czteroparametrowa transformacja globalna | działa technicznie; wymaga nowej funkcji celu i bogatszych operatorów |
| Ewaluacja | odległości cech, testy statystyczne, odsłuch | infrastruktura jest, brak kompletnego eksperymentu |
| CLI | `acquire`, `train`, `infer`, `evaluate` | GA i tryb łączony są dostępne tylko przez API Pythona |

## Instalacja

Projekt zakłada Python 3.10 oraz środowisko z PyTorch. Na Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Wariant Conda jest opisany w `environment.yml`. Wersję PyTorch/CUDA należy
dopasować do sterownika GPU; deklaracje w `pyproject.toml` i `requirements.txt`
nie są obecnie jednym spójnym źródłem prawdy.

## Podstawowe polecenia

```powershell
# etap E1.0: manifest ASAP i raport jakości (bez treningu)
python -m musicians_style.e1 --config configs/e1_asap.yaml

# walidacja danych i manifesty
midi-style acquire --config configs/custom_conditional.yaml

# trening prototypowego modelu warunkowanego
midi-style train --config configs/custom_conditional.yaml

# inferencja GAN
midi-style infer `
  --config configs/custom_conditional.yaml `
  --checkpoint experiments/<eksperyment>/checkpoints/<plik>.pt `
  --input <wejscie.mid> `
  --target_artist bach `
  --output <wynik.mid>

# ewaluacja przygotowanych par wejście-wyjście
midi-style evaluate `
  --pairs <katalog_par> `
  --style <manifest_artysty.json> `
  --metric mahalanobis
```

Do czasu wykonania etapów P0–P2 z [planu naprawczego](docs/PLAN_NAPRAWCZY.md)
nie warto uruchamiać długiego treningu GAN.

## Struktura

```text
src/musicians_style/
  data/          pozyskiwanie, walidacja i manifesty
  midi/          parser, writer i konwersja pianoroll
  features/      ekstrakcja cech symbolicznych
  models/        prototypy StarGAN/CycleGAN i funkcje strat
  training/      dataset i trener GAN
  ga/            algorytm genetyczny i transformacje
  inference/     GAN, GA i przepływ łączony (API)
  evaluation/    metryki obiektywne i testy odsłuchowe
tests/           testy unit/property/integration
configs/         konfiguracje przykładowe
docs/            dokumentacja audytu i plan dalszych prac
```

`datasets/`, `experiments/`, `Literatura/`, `logseq_pages/` i modele są lokalnymi
artefaktami ignorowanymi przez Git.

## Testy

```powershell
pytest
pytest -m property
pytest -m "not slow"
```

Stan audytu z 1 września 2026: **434 testy przechodzą**. Nie oznacza to jeszcze
poprawności metody badawczej: brakuje m.in. testu pełnego utworu dłuższego niż
jedno okno, prawdziwego testu zachowania długości, porównania domen A↔B oraz
walidacji jakości transferu na wydzielonym zbiorze.
