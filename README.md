# Modelowanie stylu artystycznego muzyków

Repozytorium badawcze pracy inżynierskiej dotyczącej analizy stylu kompozytorskiego
i transferu stylu pomiędzy plikami MIDI.

> **Stan projektu (2026-10-04):** E1–E4 są zamrożonymi wynikami historycznymi.
> E3 poprawia proxy stylu względem E2, ale zgodność celu z osobnym ewaluatorem
> held-out jest słaba (r=0,044). E4.6 zakończył się walidacyjnym NO-GO.
> Aktualny zakres: dokumentacja i V2-01 (pochodzenie importów, audyt i poziomy
> testów). Kolejne eksperymenty wymagają osobnego przeglądu.

## Od czego zacząć

- [Aktualny status](docs/STATUS.md) i [Research V2 roadmap](docs/research/ROADMAP.md) — bieżące ustalenia, testy i granice implementacji.
- [Audyt repozytorium](docs/AUDYT_REPOZYTORIUM.md) — historyczny audyt; jego twierdzenia mogą być nieaktualne względem zamkniętych E1–E4.
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
- [Publiczne przykłady](examples/README.md) — krótkie porównania MIDI/WAV E3
  (Preludium Bacha i „Kotek”) oraz zagregowane wyniki i wykresy E2.

E1b uzyskał balanced accuracy 0,864. E2/E3 pozostają sparowanymi baseline'ami;
historyczny E3 dopuszczał transpozycję melodii. Nowe eksperymenty transferu
wymagają dokładnie oryginalnych chronionych wysokości, onsetów, note-off i
kolejności zdarzeń. Nie interpretuj proxy klasyfikatora jako procentu
percepcyjnego podobieństwa. Raporty historyczne pozostają niezmienione;
[STATUS](docs/STATUS.md) opisuje aktualne dowody i ograniczenia.

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
| MIDI | parser, writer, reprezentacja zdarzeń i pianoroll | pianoroll pozostaje stratny; zdarzenia są potrzebne do ścisłej ochrony |
| Cechy | legacy42, frozen custom93, E3 event67 | kontrakty historyczne pozostają bez zmian |
| E1 | grouped RF i kontrole przecieku | CLOSED / GO w trzech kompozytorach |
| E2 / E3 | globalny GA / lokalny GA z ochroną względną do transpozycji | 300 wyników każdego, zamrożone baseline'y |
| E4.6 | warunkowany GAN | validation NO-GO; brak autoryzacji kolejnego treningu |
| Research V2 | asset_paths, provenance, research_audit | tylko V2-01; bez nowego drzewa pakietów |
| CLI / odsłuch | istniejące eksperymenty i lokalny listener | demonstracja techniczna nie dowodzi jakości transferu |

## Instalacja i pochodzenie importu

Projekt używa Python 3.10 i PyTorch; zależności/CUDA opisują `pyproject.toml`,
`requirements.txt` i `environment.yml`. Nie aktualizuj zależności przy audycie
istniejącego środowiska. Na Windows, z aktywnego checkoutu:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --no-deps --no-build-isolation -e .
python -c "import musicians_style; print(musicians_style.__file__)"
```

Import musi wskazywać `src/musicians_style/__init__.py` tego checkoutu po
rozwiązaniu ścieżek, również w worktree. Kopia w `.venv/Lib/site-packages`,
choć znajduje się wewnątrz repozytorium, nie spełnia tego warunku. Przy nowym
środowisku najpierw przygotuj wymagane zależności i narzędzia budowania; powyższy
wariant editable nie pobiera ani nie aktualizuje zależności. Fallback checkout-first:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
python tools/research_audit.py --scope inventory --output experiments/<nowy-katalog>
```

Audyt sprawdza faktyczny import przed załadowaniem modułów badawczych i nie
naprawia go automatycznie. Błąd zostawia diagnostykę w nowym katalogu i zwraca
niepowodzenie. Pytest używa checkout-first `pythonpath = src` i własnego guardu.

Korzenie lokalnych zasobów dla audytu: jawne `--data-root`, `--results-root`,
`--literature-root`, następnie `MSM_DATA_ROOT`, `MSM_RESULTS_ROOT`,
`MSM_LITERATURE_ROOT`, następnie `datasets`, `experiments`, `Literatura`
checkoutu. Ścieżki względne są rozwiązywane względem checkoutu, nie bieżącego
katalogu procesu. Stare loadery eksperymentów pozostają bez zmian. W worktree
wskaż wspólne zasoby; nie kopiuj danych/wyników/PDF-ów.

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
  --root experiments/e4_asap_v3 --soundfont soundfonts/FluidR3_GM.sf2 --port 8766
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

## Testy i audyty

Domyślny test deweloperski jest szybki i używa źródeł checkoutu. Kosztowne
workflowy pozostają dostępne przez jawne poziomy, bez usuwania asercji:

```powershell
python -m pytest
python -m pytest -o addopts= tests/property --hypothesis-profile=dev
python -m pytest -o addopts= tests/property --hypothesis-profile=full
python -m pytest -o addopts= tests/unit tests/integration -m "integration or regression"
python -m pytest -o addopts= tests/unit tests/integration
python -m pytest -o addopts= tests --hypothesis-profile=full
```

Domyślnie wykluczone: property, integration, regression i slow. Property dev:
limit 20 przykładów na każdą własność; full (domyślny po jawnym wybraniu
property): zachowane oryginalne budżety 50/100/200/300. Integracja/regresja:
trening, checkpoint→MIDI, HTTP oraz kosztowne wykresy/raporty. Tanie sprawdzenia
kształtów modeli i strat pozostają w szybkim poziomie. `-o addopts=` usuwa
wykluczenia; ostatnie polecenie jest pełnym, niefiltrowanym przebiegiem.

Weryfikacja V2-01: fast 452 passed / 64 deselected w 19,59 s; property dev
17 passed w 9,34 s; pełne unit/integration/regression 527 passed w 42,89 s.
Nie uruchomiono pełnych budżetów property; sprawdzono ich zachowanie osobno.
Brak nowych badań naukowych/treningu/transferu w tym przebiegu.

```powershell
python tools/research_audit.py --scope inventory --output experiments/<nowy-inventory>
python tools/research_audit.py --scope baseline --output experiments/<nowy-baseline>
```

Inventory zapisuje środowisko, zasoby i poziomy testów. Baseline dodatkowo
weryfikuje 150 źródeł, podziały/snapshoty, 600 wyników E2/E3 i E4.6 checkpoint/
walidację. Katalog wyjściowy musi być nowy. Audyt nie ekstraktuje cech, nie
oblicza nowych metryk treści, nie trenuje i nie regeneruje MIDI. JSON/report
zawierają jawne błędy/braki oraz ograniczenia historycznego pochodzenia.
Duże audyty naukowe produkują artefakty poza pytest. Szczegóły i następne
bramki przeglądu: [ROADMAP](docs/research/ROADMAP.md).
