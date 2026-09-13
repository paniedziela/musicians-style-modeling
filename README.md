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
- [Plan eksperymentu E2](docs/E2_PLAN.md) — baseline transferu obecnym
  algorytmem genetycznym, pilot, pełna macierz i niezależna ewaluacja E1b.
- [Plan implementacji E3](docs/E3_PLAN.md) — odchudzony GA z lokalnymi
  operacjami i twardą ochroną melodii, metrum oraz długości.
- [Porządkowanie literatury](docs/LITERATURA.md) — kryteria redukcji 173 pozycji,
  literatura rdzeniowa i proponowana struktura przeglądu.

E1 jest zamknięty decyzją GO (`balanced accuracy E1b = 0,864`; testy E1/GA:
`105 passed`). E2 służy jako zamrożony baseline obecnego GA przed przebudową
metody w E3.

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
| GA | czteroparametrowa transformacja globalna | działa technicznie; E2 mierzy ją jako baseline, E3 ma ją przebudować |
| Ewaluacja | odległości cech, testy statystyczne, odsłuch | infrastruktura jest, brak kompletnego eksperymentu |
| CLI | `acquire`, `train`, `infer`, `evaluate`, `e1`, `e2` | tryb łączony nadal jest dostępny przez API Pythona |

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

### Webowy odsłuch MIDI

Z katalogu repozytorium uruchom lokalną aplikację (bez dodatkowego frameworka):

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m musicians_style.listener
```

Otwórz **http://127.0.0.1:8765**. Biblioteka wyszukuje MIDI rekurencyjnie;
wybierz eksperyment, wyszukaj utwór i załaduj pliki do **A** oraz **B**.
Przełączanie zachowuje pozycję odsłuchu. Dla wyników E2/E3 przy dostępnych
manifestach i zbiorze danych można jednym kliknięciem załadować oryginał.
Własne MIDI dodasz przez wybór pliku lub przeciągnięcie do okna.
Aplikacja oferuje pauzę, przewijanie, tempo, zapętlanie, podgląd nut i pobieranie
MIDI/WAV. Skróty: spacja, A/B, strzałki w lewo/prawo.

Audio powstaje przez FluidSynth przy pierwszym załadowaniu pliku. Wymagane są
`pretty_midi`, `pyFluidSynth`, systemowa biblioteka FluidSynth oraz SoundFont
(domyślnie `soundfonts/FluidR3_GM.sf2`). Zmiana tempa zmienia również wysokość
dźwięku. Limit pojedynczego MIDI: 10 MB, 15 minut, 200 000 nut.
Cache audio i wgrane pliki trafiają do `experiments/.midi-listener/`;
można go usunąć po zatrzymaniu aplikacji. Wgrane pliki są dostępne do końca
sesji serwera. Aplikacja działa lokalnie, bez CDN i zewnętrznych usług.

```powershell
# Inny katalog, SoundFont i port
.venv\Scripts\python.exe -m musicians_style.listener `
  --root experiments/e4_asap_v2 --soundfont soundfonts/FluidR3_GM.sf2 --port 8766
```

Po zainstalowaniu aktualnego pakietu dostępne jest też polecenie `midi-listen`.

```powershell
# E1.0: manifest ASAP i raport jakości (bez treningu)
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage audit

# E1.1: deterministyczne zewnętrzne i wewnętrzne splity grouped CV
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage splits

# E1.2: cache obu wariantów cech i klasyfikacja E1a
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage features
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage classify

# finalny E1.2: dodatkowo kosztowny test permutacyjny z ponownym uczeniem
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage classify `
  --retraining-permutations 99

# E1.3: 93 cechy kompozycyjne, ablacje grup i klasyfikacja E1b
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage e1b-features
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage e1b-classify `
  --importance-repeats 10

# szybszy przebieg główny E1b bez 12 dodatkowych wariantów ablacyjnych
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage e1b-classify `
  --variants composition_full `
  --importance-repeats 10 `
  --retraining-permutations 99

# oba kroki E1.3 w jednym wywołaniu; finalny przebieg dodaje test retreningowy
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage e1b `
  --importance-repeats 10 `
  --retraining-permutations 99

# E1-open: audyt dodatkowych kompozytorów i analiza odrzucania unknown
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage open-data
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage open-classify

# E1.4: zamknięcie ukończonego E1b, opcjonalnie wraz z E1-open
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage report `
  --e1b-run-dir experiments/<ukończony_e1b> `
  --open-run-dir experiments/<ukończony_e1_open>

# E2: przygotowanie, pilot, pełna macierz i raport baseline'u GA
python -m musicians_style.e2 --config configs/e2_asap.yaml --stage prepare
python -m musicians_style.e2 --config configs/e2_asap.yaml --stage pilot
python -m musicians_style.e2 --config configs/e2_asap.yaml --stage run --workers 1
python -m musicians_style.e2 --config configs/e2_asap.yaml --stage report

# E3: lokalny transfer z ochroną melodii/metrum i profilem wyłącznie z train
python -m musicians_style.e3 --config configs/e3_asap.yaml --stage prepare
python -m musicians_style.e3 --config configs/e3_asap.yaml --stage pilot --workers 2
python -m musicians_style.e3 --config configs/e3_asap.yaml --stage run --workers 2
python -m musicians_style.e3 --config configs/e3_asap.yaml --stage report `
  --e2-run-dir experiments/e2_asap

# albo E1.0-E1.2 w jednym przebiegu
python -m musicians_style.e1 --config configs/e1_asap.yaml --stage all

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

Klasyfikacja domyślnie wykonuje także analizę wrażliwości z jedną losowaną
deterministycznie próbką na grupę, przez co obejmuje 300 zadań model/fold.
`--retraining-permutations 99` dodaje osobną, kosztowną kontrolę istotności;
zwykłe `--permutations` dotyczy tylko szybkiej diagnostyki zgodności gotowych
predykcji OOF z etykietami. Postęp jest widoczny w terminalu i w
`progress.jsonl`. Każdy katalog przebiegu zawiera `run_manifest.json`, kopię
konfiguracji oraz snapshoty manifestu danych, splitów i cache'u cech. Nowy,
jawny katalog można podać przez `--run-dir`; katalog musi być pusty.

E1.3 zapisuje osobny `composition_features.json`. Kontrakt obejmuje 93 cechy z
jawną nazwą, grupą i jednostką: pitch, melody, rhythm, texture, harmony oraz
structure. Cache definiuje wariant pełny, sześć wariantów zawierających tylko
jedną grupę i sześć ablacji leave-one-group-out. Ważność permutacyjna jest
liczona wyłącznie dla niedummy modeli pełnego wariantu na niewidzianej części
każdego zewnętrznego foldu; nie uczestniczy w strojeniu modelu.

Pełny E1b wykonuje 13 wariantów × 3 modele × 25 foldów × 2 tryby analizy,
czyli 1950 zewnętrznych zadań. Dla logistic regression i random forest każde
zadanie obejmuje dodatkowo wewnętrzny grid search. Do głównego kryterium sukcesu
wystarcza przebieg `composition_full`; komplet pojedynczych grup i ablacji można
uruchomić osobno. Parametry `--variants` i `--models` przyjmują listy rozdzielone
przecinkami.

E1-open nie uczy klasy `other`. Stałe modele closed-set są trenowane wyłącznie na
Bachu, Beethovenie i Chopinie. Haydn, Mozart i Schumann kalibrują próg, natomiast
Liszt, Schubert, Rachmaninoff i Ravel są zachowani do testu. E1.4 przyjmuje tylko
ukończony katalog E1b i zapisuje raport Markdown, CSV predykcji/wykluczeń, wykresy
oraz `closure_manifest.json`; brak testu permutacyjnego z retreningiem daje jawny
status „niekompletne”, a nie decyzję GO.

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

Stan audytu z 1 września 2026: **453 testy przechodzą**. Nie oznacza to jeszcze
poprawności metody badawczej: brakuje m.in. testu pełnego utworu dłuższego niż
jedno okno, prawdziwego testu zachowania długości, porównania domen A↔B oraz
walidacji jakości transferu na wydzielonym zbiorze.
