# Implementation Plan: Modelowanie stylu artystycznego muzyków

## Overview

Plan implementacyjny dla *Systemu* modelowania stylu artystycznego muzyków zgodnie z `requirements.md` i `design.md`. Każde zadanie buduje na poprzednich i kończy się integracją komponentów z resztą Systemu, bez pozostawiania kodu osieroconego. Implementacja prowadzona jest w **Python 3.10+** zgodnie z wyborem technologicznym z `design.md` (PyTorch ≥ 2.0, mido, pretty_midi, Hypothesis, scipy, structlog).

Zadania oznaczone `*` (np. `2.2*`) są opcjonalnymi sub-zadaniami testowymi (testy jednostkowe, property-based, integracyjne) i mogą zostać pominięte dla szybszego MVP. Główne zadania implementacyjne (bez `*`) muszą być zrealizowane.

Każde zadanie property-based testowe odwołuje się do konkretnej właściwości z sekcji *Correctness Properties* w `design.md` oraz do kryterium akceptacji z `requirements.md`. Punkty kontrolne (Checkpoint) gwarantują inkrementalną walidację.

## Tasks

- [x] 1. Konfiguracja projektu i podstawowa infrastruktura
  - [x] 1.1 Utworzenie struktury katalogów i plików manifestu projektu
    - Utworzyć układ katalogów `src/musicians_style/{data,midi,features,models,training,ga,inference,evaluation}`, `tests/{unit,property,integration}`, `configs/`, `experiments/`
    - Utworzyć `pyproject.toml` z deklaracją Python 3.10+, `requirements.txt` oraz `environment.yml` z dokładnymi wersjami zależności (PyTorch ≥ 2.0, numpy, scipy, mido, pretty_midi, structlog, pyyaml, hypothesis, pytest, fluidsynth-binding)
    - Dodać plik `.gitignore` dla `experiments/`, `__pycache__/`, `*.pt`
    - Skonfigurować `pytest.ini` z markerem `property` i ścieżką `tests/`
    - _Requirements: 8.5_

  - [x] 1.2 Implementacja loadera konfiguracji YAML/JSON
    - Utworzyć `src/musicians_style/config.py` z klasami `Config`, `ModelConfig`, `TrainingConfig`, `GAConfig`, `EvaluationConfig` jako `@dataclass`
    - Zaimplementować `load_config(path: Path) -> Config` rozpoznający format po rozszerzeniu (YAML/JSON) i normalizujący strukturę danych
    - Walidować obecność wymaganych pól (`seed`, `model.mode`, `model.artists` dla trybu `conditional`, `model.reference_artist` dla `per_artist`); dla niespójnych konfiguracji rzucać `ConfigValidationError`
    - Utworzyć `configs/default_conditional.yaml`, `configs/default_per_artist.yaml`, `configs/default_ga.yaml` zgodnie ze schematem z sekcji *Data Models* design.md
    - _Requirements: 8.1, 8.6_

  - [ ]* 1.3 Property test P17 - równoważność konfiguracji YAML i JSON
    - **Property 17: Równoważność konfiguracji YAML i JSON**
    - **Validates: Requirements 8.1**
    - W `tests/property/test_config_formats.py` zaimplementować strategię `config_dict_strategy` generującą poprawne słowniki konfiguracyjne
    - Test sprawdza, że `load_config(yaml_dump(c))` i `load_config(json_dump(c))` produkują bit-identyczne struktury po normalizacji
    - `@settings(max_examples=200, deadline=None)`, tag `# Feature: musicians-style-modeling, Property 17: Równoważność YAML/JSON`

  - [x] 1.4 Implementacja hierarchii wyjątków i logowania strukturalnego
    - Utworzyć `src/musicians_style/errors.py` zawierający `MusiciansStyleError`, `MidiValidationError` (z asercją niepustego `message`), `EmptyDatasetError`, `UnknownArtistError`, `ArtistMismatchError`, `GpuOutOfMemoryError`, `ConfigValidationError`
    - Skonfigurować `structlog` z formatterem JSON-lines w `src/musicians_style/logging.py` z funkcją `get_logger(component: str)`
    - Logger SHALL zawierać `ts`, `level`, `component`, `msg` w każdym wpisie (sekcja *Format wpisu logu* w design.md)
    - Zapisywać commit hash (`git rev-parse HEAD`) do `experiments/{name}/git_commit.txt` w funkcji `init_experiment_dir(config)`
    - _Requirements: 8.2, 8.4_

- [x] 2. Parser i pretty printer MIDI
  - [x] 2.1 Definicja Reprezentacji_Wewnętrznej (InternalRepr)
    - W `src/musicians_style/midi/types.py` zaimplementować `@dataclass(frozen=True)` dla `NoteEvent`, `MetaEvent`, `InternalRepr` z polami zgodnie z sekcją *Components and Interfaces* design.md
    - Utworzyć funkcję porządkującą `event_key(e)` (lex po `(tick, channel, pitch, velocity)`) używaną do deterministycznego sortowania zdarzeń
    - Zachować immutowalność (pola `notes`, `meta` jako krotki) dla potrzeb testów własnościowych
    - _Requirements: 5.3_

  - [x] 2.2 Implementacja MidiParser
    - W `src/musicians_style/midi/parser.py` zaimplementować klasę `MidiParser` z metodami `parse(path)`, `parse_bytes(data)`, `validate(path)`
    - `validate` SHALL operować na surowych bajtach przed jakimkolwiek przetwarzaniem treści (przed obliczaniem długości czasowej) - kontrola nagłówka `MThd`, długości bloków, struktury chunków
    - Przy wykryciu uszkodzenia zgłaszać `MidiValidationError` z opisem miejsca i typu; jeśli niemożliwe zlokalizować - zwracać niepustą stałą `"wykryto uszkodzenie struktury pliku"`
    - Wykorzystać `mido` dla parsowania niskopoziomowego, konwertować zdarzenia do `NoteEvent`/`MetaEvent` i zwracać `InternalRepr` z deterministycznie posortowanymi krotkami
    - _Requirements: 5.3, 5.5_

  - [x] 2.3 Implementacja MidiPrettyPrinter
    - W `src/musicians_style/midi/printer.py` zaimplementować klasę `MidiPrettyPrinter` z metodami `write(repr_, path)` oraz `to_bytes(repr_)`
    - Generować plik MIDI zgodny ze specyfikacją SMF 1.0 (format 0 lub 1) na podstawie `InternalRepr.smf_format`
    - Zachowywać deterministyczną kolejność zdarzeń z `event_key`, kodować `tempo`, `time_signature`, `key_signature`, `program_change` jako odpowiednie meta-zdarzenia
    - _Requirements: 5.2, 5.3_

  - [ ]* 2.4 Property test P1 - round-trip parsera i pretty printera
    - **Property 1: Round-trip parsera i pretty printera MIDI**
    - **Validates: Requirements 5.4, 11.1**
    - W `tests/property/test_midi_roundtrip.py` zaimplementować strategię `midi_internal_repr` (sekcja *Strategie generatorów Hypothesis*)
    - Test: `parse(write(r))` zwraca `r'` semantycznie równoważne `r` (ta sama lista zdarzeń nutowych z dokładnością do kolejności zdarzeń o identycznym znaczniku czasu)
    - `@settings(max_examples=200, deadline=None)`

  - [ ]* 2.5 Property test P13 - niepusty opis błędu walidacji MIDI
    - **Property 13: Niepusty opis błędu walidacji MIDI**
    - **Validates: Requirements 5.5**
    - W `tests/property/test_midi_validation.py` zaimplementować strategię `malformed_midi_bytes = st.binary(min_size=1, max_size=4096)`
    - Test: `MidiParser.validate(b)` albo zwraca sukces, albo zgłasza `MidiValidationError` z niepustym `message`; żadne wywołanie nie skutkuje nieobsłużonym wyjątkiem (TypeError, AttributeError, IndexError)
    - `@settings(max_examples=300, deadline=None)`

  - [ ]* 2.6 Testy jednostkowe parsera/printera dla przypadków brzegowych
    - W `tests/unit/test_midi_parser.py` testować: pliki SMF format 0 i 1, plik z pojedynczą nutą, plik z meta-zdarzeniami `key_signature` i `time_signature`, plik bez tempa (sprawdzenie domyślnego 120 BPM przez Ekstraktor_Cech)
    - Testować przypadki uszkodzonych plików: skrócony nagłówek, niespójna długość chunku
    - _Requirements: 5.2, 5.5_

  - [x] 2.7 Implementacja konwertera Pianoroll
    - W `src/musicians_style/midi/pianoroll.py` zaimplementować klasę `Pianoroll` z metodami `from_internal(repr_, window_steps)` i `to_internal(pianoroll, template)`
    - Konwersja `from_internal`: macierz binarna `[T × P]` z krokiem 16th note, zakres wysokości z konfiguracji (`pitch_range: [24, 108]`)
    - Konwersja `to_internal`: rekonstrukcja przy użyciu `template` zachowującego tempo, metrum, kanały i velocity (Wymaganie 5.7)
    - Połączyć z `MidiParser` i `MidiPrettyPrinter` w spójny pipeline `parse → Pianoroll → ... → InternalRepr → write`
    - _Requirements: 5.3, 5.7_

- [x] 3. Ekstraktor cech muzycznych
  - [x] 3.1 Definicja FeatureVector i AggregatedFeatures
    - W `src/musicians_style/features/types.py` zaimplementować `@dataclass(frozen=True) FeatureVector` z polami: `tempo_bpm`, `key`, `pitch_class_histogram` (shape (12,)), `interval_histogram` (shape (25,)), `note_density_per_s`, `mean_note_duration_s`, `std_note_duration_s`, `rest_ratio`
    - Zaimplementować `as_array() -> np.ndarray` dla obliczeń odległości
    - Zaimplementować `AggregatedFeatures` z polami `mean`, `median`, `std`, `covariance`
    - W `src/musicians_style/features/constants.py` zdefiniować stałą `NEUTRAL_FEATURE_VECTOR` używaną dla pustych plików
    - _Requirements: 2.1, 2.3, 2.7_

  - [x] 3.2 Implementacja FeatureExtractor (ekstrakcja per plik)
    - W `src/musicians_style/features/extractor.py` zaimplementować klasę `FeatureExtractor` z metodą `extract(repr_: InternalRepr) -> FeatureVector`
    - Obliczać tempo (jeśli brak meta-zdarzenia → 120 BPM + log), tonację profilem Krumhansla-Schmucklera, histogram klas wysokości (znormalizowany do sumy 1), histogram interwałów (różnice w monofonicznym śladzie), gęstość nut/s, średnią i odchylenie długości nuty, proporcję pauz
    - Dla pustych `InternalRepr` (zero nut) zwracać `NEUTRAL_FEATURE_VECTOR` + log
    - W przypadku częściowego błędu numerycznego (Wymaganie 2.8) logować nazwę pliku, problem i częściowe wartości; zwracać częściowy wektor
    - _Requirements: 2.1, 2.2, 2.5, 2.7, 2.8_

  - [ ]* 3.3 Property test P2 - idempotencja Ekstraktora_Cech
    - **Property 2: Idempotencja Ekstraktora_Cech**
    - **Validates: Requirements 2.4, 11.2**
    - W `tests/property/test_features_idempotence.py` testować, że `extract(r) == extract(r)` bit-identycznie dla losowo generowanych `InternalRepr`
    - `@settings(max_examples=200, deadline=None)`

  - [ ]* 3.4 Property test P3 - walidność Wektora_Cech
    - **Property 3: Walidność Wektora_Cech**
    - **Validates: Requirements 2.1, 2.3**
    - W `tests/property/test_features_validity.py` testować niezmienniki: `tempo_bpm > 0`, `rest_ratio ∈ [0, 1]`, `sum(pitch_class_histogram) == 1 ± 1e-9`, `len(extract(r).as_array())` jest stała niezależnie od długości wejścia
    - `@settings(max_examples=200, deadline=None)`

  - [ ]* 3.5 Property test P4 - wektor neutralny dla pustych plików
    - **Property 4: Wektor neutralny dla pustych plików MIDI**
    - **Validates: Requirements 2.7, 11.8**
    - W `tests/property/test_features_empty.py` używając `midi_internal_repr(max_notes=0)` weryfikować, że `extract(r) == NEUTRAL_FEATURE_VECTOR` bez zgłaszania wyjątku
    - `@settings(max_examples=100, deadline=None)`

  - [ ]* 3.6 Property test P6 - równoważność transpozycji
    - **Property 6: Równoważność transpozycji**
    - **Validates: Requirements 11.3**
    - W `tests/property/test_features_transpose.py` dla losowych `InternalRepr` i `k ∈ [-12, +12]` weryfikować, że `extract(transpose(r, k)).pitch_class_histogram` jest cyklicznym przesunięciem o `k mod 12` pozycji
    - Funkcja pomocnicza `transpose(r, k)` w `tests/property/helpers.py` dodaje `k` do `pitch` każdej nuty (z odrzuceniem nut spoza zakresu)
    - `@settings(max_examples=200, deadline=None)`

  - [x] 3.7 Implementacja agregacji cech zbioru (extract_dataset)
    - Rozszerzyć `FeatureExtractor` o `extract_dataset(manifest: Manifest) -> AggregatedFeatures` obliczające średnią, medianę, odchylenie standardowe i macierz kowariancji każdej cechy
    - Macierz kowariancji wykorzystywana w odległości Mahalanobisa w `Funkcji_Dopasowania` i ewaluacji
    - _Requirements: 2.6_

- [ ] 4. Checkpoint - parser MIDI i ekstraktor cech działają
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Akwizytor danych i Manifest_Zbioru
  - [x] 5.1 Implementacja Manifest_Zbioru i walidacji liczności
    - W `src/musicians_style/data/manifest.py` zaimplementować `@dataclass Manifest` z polami `artist_id`, `files: list[FileEntry]`, `created_at`, `git_commit`
    - `FileEntry` zawiera `path`, `sha256`, `duration_s`, `tracks`, `source`
    - Funkcja `compute_sha256(path)` oraz `to_json(manifest, path)` zapisująca manifest w formacie JSON
    - _Requirements: 1.7_

  - [x] 5.2 Implementacja DatasetAcquirer (lokalny katalog)
    - W `src/musicians_style/data/acquirer.py` zaimplementować `DatasetAcquirer.acquire_local(directory: Path) -> Manifest`
    - Walidować liczność (`< 30` → ostrzeżenie + log z minimalną wymaganą liczbą), zgodność ze SMF (pominięcie + log z nazwą pliku i przyczyną), długość w przedziale `[5 s, 30 min]` (odrzucenie + log)
    - Po walidacji jeśli liczba poprawnych plików = 0, zgłosić `EmptyDatasetError` z opisem przyczyny i zakończyć działanie z niezerowym kodem wyjścia
    - Wykorzystać `MidiParser.validate` do sprawdzania zgodności ze SMF
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 5.3 Implementacja akwizytora YouTube
    - Rozszerzyć `DatasetAcquirer` o `acquire_youtube(ids: list[str], max_duration_s: int = 15) -> Manifest`
    - Wykorzystać `yt-dlp` jako bibliotekę zewnętrzną; pobierać fragmenty audio i konwertować do MIDI poprzez zewnętrzny komponent (lub odrzucać poza zakresem pracy - log + skip)
    - Ograniczyć długość każdego fragmentu do nie więcej niż 15 sekund
    - _Requirements: 1.6_

  - [ ]* 5.4 Testy jednostkowe walidacji Akwizytora_Danych
    - W `tests/unit/test_acquirer.py` testować scenariusze: katalog z 29 plikami (ostrzeżenie), plik o długości 31 minut (odrzucenie), katalog pusty (EmptyDatasetError + exit 2)
    - Mock `MidiParser.validate` dla testów izolowanych
    - _Requirements: 1.1, 1.2, 1.4, 1.5_

  - [ ]* 5.5 Property test P16 - selektywne ostrzeżenia w trybie warunkowanym
    - **Property 16: Selektywne ostrzeżenia o niewystarczającej liczności w trybie warunkowanym**
    - **Validates: Requirements 3.11**
    - W `tests/property/test_dataset_warnings.py` strategia `multi_artist_count_strategy` generująca słowniki `{artist_id: file_count}`
    - Test: zbiór ostrzeżeń ma dokładnie te `artist_id`, dla których `file_count < 30`; pusty gdy wszyscy mają ≥ 30
    - `@settings(max_examples=300, deadline=None)`

- [ ] 6. Algorytm genetyczny
  - [x] 6.1 Definicja Genome i implementacja apply_transformation
    - W `src/musicians_style/ga/types.py` zaimplementować `@dataclass(frozen=True) Genome` z parametrami `transpose_semitones`, `rhythm_density_factor`, `note_duration_factor`, `velocity_offset` (parametry rzeczywiste, dopuszczalne wartości ujemne)
    - W `src/musicians_style/ga/transformation.py` zaimplementować `apply_transformation(x: InternalRepr, g: Genome) -> InternalRepr` jako czystą funkcję
    - Transpozycja: dodanie `round(g.transpose_semitones)` do `pitch`, modyfikacja gęstości i długości nut przez przemnożenie czasów, modyfikacja velocity z saturacją do `[0, 127]`
    - Stała `IDENTITY_GENOME = Genome(0, 1.0, 1.0, 0.0)`
    - _Requirements: 4.1_

  - [x] 6.2 Implementacja Funkcji_Dopasowania
    - W `src/musicians_style/ga/fitness.py` zaimplementować funkcję `fitness(genome, x_input, target_aggregated, metric)` zwracającą `-distance(extract(apply_transformation(x_input, genome)), target_aggregated.mean)`
    - Wsparcie dla odległości euklidesowej i Mahalanobisa (używającej `target_aggregated.covariance`)
    - Funkcja deterministyczna względem wejść (brak losowości)
    - _Requirements: 4.3_

  - [x] 6.3 Implementacja operatorów genetycznych
    - W `src/musicians_style/ga/operators.py` zaimplementować: `tournament_select(population, fitnesses, k, rng)`, `single_point_crossover(p1, p2, rng)`, `uniform_crossover(p1, p2, rng)` (BLX-α dla parametrów rzeczywistych z α=0.5), `gaussian_mutate(genome, sigma_per_param, rng)`
    - Wszystkie operatory akceptują `numpy.random.Generator` jako parametr (brak globalnego RNG)
    - Operatory zwracają nowe instancje `Genome`, nigdy nie modyfikują wejścia
    - _Requirements: 4.2_

  - [x] 6.4 Główna pętla GeneticAlgorithm z elitaryzmem i warunkami stopu
    - W `src/musicians_style/ga/algorithm.py` zaimplementować klasę `GeneticAlgorithm` z metodą `run(x_input, style_aggregated, config: GAConfig, seed: int) -> tuple[Genome, History]`
    - Inicjalizacja populacji z `numpy.random.Generator(seed)`, ewaluacja, pętla pokoleń: selekcja → krzyżowanie → mutacja → ewaluacja → elitaryzm (k_elite najlepszych przechodzi bezpośrednio)
    - Warunki stopu: `max_generations` lub stagnacja przez `stagnation_generations`
    - Logowanie `(best_fitness, mean_fitness, worst_fitness, best_genome)` w `ga.jsonl` w każdym pokoleniu
    - _Requirements: 4.4, 4.5, 4.6, 4.7_

  - [ ]* 6.5 Property test P5 - niezmienność Ekstraktora_Cech na transformację tożsamościową
    - **Property 5: Niezmienność Ekstraktora_Cech na transformację tożsamościową**
    - **Validates: Requirements 11.7**
    - W `tests/property/test_features_identity.py` weryfikować, że `extract(apply_transformation(r, IDENTITY_GENOME)) == extract(r)` bit-identycznie
    - `@settings(max_examples=200, deadline=None)`

  - [ ]* 6.6 Property test P7 - determinizm Algorytmu_Genetycznego
    - **Property 7: Determinizm Algorytmu_Genetycznego**
    - **Validates: Requirements 4.4, 11.4**
    - W `tests/property/test_ga_determinism.py` dla losowych `seed × small_dataset × ga_config` weryfikować, że dwa wywołania `ga.run(x, D, c, seed=s)` zwracają bit-identyczne końcowe populacje i historie dopasowania
    - `@settings(max_examples=50, deadline=None)` (mniej iteracji ze względu na koszt)

  - [ ]* 6.7 Property test P8 - monotoniczność elitaryzmu
    - **Property 8: Monotoniczność Algorytmu_Genetycznego w trybie elitaryzmu**
    - **Validates: Requirements 4.7, 11.5**
    - W `tests/property/test_ga_elitism.py` z `elitism_k ≥ 1` weryfikować dla każdej pary `(n, n+1)`, że `best_fitness[n+1] >= best_fitness[n]`
    - `@settings(max_examples=50, deadline=None)`

  - [ ]* 6.8 Property test P9 - operatory zachowują liczność populacji
    - **Property 9: Operatory genetyczne zachowują liczność populacji**
    - **Validates: Requirements 4.1, 4.2**
    - W `tests/property/test_ga_operators.py` dla losowej `population_strategy` o rozmiarze N i konfiguracji operatorów weryfikować, że `next_generation(P, config).size == N` oraz że każdy gen mieści się w zadeklarowanych zakresach
    - `@settings(max_examples=200, deadline=None)`

- [ ] 7. Checkpoint - algorytm genetyczny działa deterministycznie
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Modele GAN (StarGAN i CycleGAN)
  - [x] 8.1 Implementacja StarGAN generatora i dyskryminatora
    - W `src/musicians_style/models/stargan.py` zaimplementować `StarGANGenerator(nn.Module)` (down-sampling 2× Conv2D + InstanceNorm + ReLU, 6 bloków residualnych, up-sampling 2× ConvTranspose2D, sigmoid) z forward `forward(x: Tensor, c: Tensor) -> Tensor`
    - Zaimplementować `StarGANDiscriminator(nn.Module)` PatchGAN 70×70 z dwiema głowicami: `D_src` (real/fake) i `D_cls` (klasyfikacja artysty), forward zwraca tuple `(Tensor, Tensor)`
    - Wejście generatora: pianoroll `[T=64, P=84]` skonkatenowany po kanałach z mapą Etykiety_Artysty `c ∈ {0,1}^N`
    - _Requirements: 3.1, 3.12_

  - [x] 8.2 Implementacja CycleGAN (tryb fallback)
    - W `src/musicians_style/models/cyclegan.py` zaimplementować `CycleGANGenerator` (architektura warstwowa identyczna jak StarGAN, lecz bez warunkowania) i `CycleGANDiscriminator` (PatchGAN bez głowicy `D_cls`)
    - Forward generatora: `forward(x: Tensor) -> Tensor`
    - _Requirements: 3.1, 3.13_

  - [x] 8.3 Implementacja funkcji strat GAN
    - W `src/musicians_style/models/losses.py` zaimplementować: `adversarial_loss(D_src_real, D_src_fake)`, `domain_classification_loss_real(D_cls, c_real)`, `domain_classification_loss_fake(D_cls, c_target)`, `cycle_consistency_loss(G, x, c_orig, c_target)`, `identity_loss(G, x, c_orig)`
    - Łączna strata: `L = L_adv + λ_cls * L_cls + λ_cyc * L_cyc + λ_id * L_id` z λ_cls=1, λ_cyc=10, λ_id=5 (z konfiguracji)
    - _Requirements: 3.12_

  - [ ]* 8.4 Testy jednostkowe modeli GAN
    - W `tests/unit/test_gan_models.py` testować poprawność wymiarów wyjścia generatora dla losowego `[B, T, P]` i `[B, N]`, poprawność wymiarów obu głowic dyskryminatora, poprawność forward CycleGAN
    - Testować, że identity loss = 0 gdy `c_target == c_orig` i generator zwraca wejście
    - _Requirements: 3.1_

- [ ] 9. Pipeline treningu GAN
  - [x] 9.1 Implementacja datasetu pianoroll i loadera
    - W `src/musicians_style/training/dataset.py` zaimplementować `PianorollDataset(torch.utils.data.Dataset)` przyjmujący `Manifest` lub `MultiArtistManifest` i zwracający `(pianoroll, etykieta_artysty)`
    - Wykorzystać `Pianoroll.from_internal` przez `MidiParser` na lokalnych plikach MIDI
    - Klasa `MultiArtistManifest` agregująca manifesty per-artist z mapowaniem artysta → indeks Etykiety_Artysty
    - _Requirements: 3.2_

  - [x] 9.2 Implementacja GANTrainer z trybem warunkowanym i per-artysta
    - W `src/musicians_style/training/trainer.py` zaimplementować klasę `GANTrainer` z parametrem `mode: Literal["conditional", "per_artist"]` i metodą `train(manifest, seed) -> Path`
    - Zapis Punktu_Kontrolnego po każdej epoce z polami: `epoch`, `generator_state`, `discriminator_state`, `optimizer_g_state`, `optimizer_d_state`, `rng_states (torch_cpu/cuda, numpy, python)`, `metadata.{mode, artists, config_hash, git_commit, model_version}`, `history`
    - Logowanie funkcji strat generatora i dyskryminatora w każdej iteracji oraz metryk walidacyjnych po każdej epoce do `train.jsonl`
    - Inicjalizacja seedów: `torch.manual_seed`, `numpy.random.seed`, `random.seed`, `torch.cuda.manual_seed_all` z parametru `seed`
    - W trybie warunkowanym sprawdzać per-artist liczność i emitować ostrzeżenia tylko dla artystów z `< 30` plikami; trening kontynuowany
    - _Requirements: 3.2, 3.4, 3.6, 3.8, 3.10, 3.11, 3.14_

  - [ ] 9.3 Implementacja wznawiania treningu z Punktu_Kontrolnego
    - Metoda `GANTrainer.resume(checkpoint_path) -> Path` wczytuje `state_dict` generatora, dyskryminatora, optymalizatorów oraz stany RNG, kontynuuje od zapisanej epoki
    - Walidować zgodność `metadata.mode` z aktualną konfiguracją
    - _Requirements: 3.5_

  - [ ] 9.4 Obsługa błędu OOM w GANTrainer
    - Owijać pętlę treningu w `try/except torch.cuda.OutOfMemoryError`; przy wystąpieniu logować, sugerować redukcję `batch_size` (np. `current_batch_size // 2`), zgłaszać `GpuOutOfMemoryError` i kończyć z exit code 3
    - _Requirements: 3.9_

  - [ ]* 9.5 Property test P14 - determinizm Pipeline_Treningu GAN
    - **Property 14: Determinizm Pipeline_Treningu GAN**
    - **Validates: Requirements 3.6, 3.7**
    - W `tests/property/test_gan_determinism.py` (oznaczyć jako `@pytest.mark.slow`) dla `seed × tiny_manifest × 1 epoch` weryfikować, że dwa uruchomienia `GANTrainer.train(D, c, seed=s)` zwracają Punkty_Kontrolne o parametrach zgodnych z dokładnością `MSE ≤ 1e-5`
    - `@settings(max_examples=10, deadline=None)` (test kosztowny - tiny model, 1 epoka)

  - [ ]* 9.6 Testy jednostkowe pipeline'u treningu
    - W `tests/unit/test_trainer.py` testować: zapis i wczytanie Punktu_Kontrolnego, zachowanie metadanych (`mode`, `artists`), walidację konfiguracji, ostrzeżenie selektywne przy małej liczności jednego artysty
    - _Requirements: 3.4, 3.5, 3.11, 3.14_

- [ ] 10. Pipeline transferu stylu (inferencja)
  - [ ] 10.1 Implementacja inferencji GAN z walidacją target_artist
    - W `src/musicians_style/inference/pipeline.py` zaimplementować `StyleTransferPipeline.infer_gan(x_path, target_artist, checkpoint_path, seed)`
    - Wczytać Punkt_Kontrolny, walidować `target_artist`: jeśli `mode == "conditional"` i `target_artist not in metadata.artists` → zgłosić `UnknownArtistError(requested, available=metadata.artists)` + exit 2; jeśli `mode == "per_artist"` i `target_artist != metadata.artists[0]` → `ArtistMismatchError` + exit 2
    - Przetwarzanie: `parse → Pianoroll.from_internal → G(x, c) → Pianoroll.to_internal(template=parsed_input) → write`
    - _Requirements: 5.1, 5.2, 5.6, 5.7, 5.8, 5.9, 5.10_

  - [ ] 10.2 Implementacja inferencji GA
    - Metoda `StyleTransferPipeline.infer_ga(x_path, target_artist, style_manifest_path, seed)` używająca `GeneticAlgorithm.run` na cechach `Zbioru_Stylu` artysty docelowego
    - Po znalezieniu najlepszego `Genome` aplikować `apply_transformation(x_input, best_genome)` i zapisywać wynik przez `MidiPrettyPrinter`
    - _Requirements: 5.1, 5.11_

  - [ ] 10.3 Implementacja trybu łączonego GAN+GA
    - Metoda `StyleTransferPipeline.infer_combined(x_path, target_artist, checkpoint_path, style_manifest_path, seed)` umożliwiająca kolejne lub równoległe wykorzystanie obu metod (np. GAN → wynik → GA jako post-processing)
    - Wybór trybu na podstawie konfiguracji
    - _Requirements: 5.1_

  - [ ]* 10.4 Property test P11 - zachowanie długości Utworu_Wyjściowego
    - **Property 11: Zachowanie długości czasowej Utworu_Wyjściowego**
    - **Validates: Requirements 5.6, 11.6**
    - W `tests/property/test_pipeline_duration.py` z mockiem GAN (np. `MockGenerator` zwracający wejście zsumowane z szumem) testować, że `duration(y) / duration(x) ∈ [0.95, 1.05]` dla ścieżek GAN, GA i combined
    - `@settings(max_examples=100, deadline=None)`

  - [ ]* 10.5 Property test P12 - zachowanie struktury metrycznej
    - **Property 12: Zachowanie struktury metrycznej**
    - **Validates: Requirements 5.7**
    - W `tests/property/test_pipeline_meter.py` weryfikować, że oznaczenie metrum w `y` jest identyczne z metrum w `x` oraz liczba taktów się zachowuje przy domyślnej konfiguracji inferencji
    - `@settings(max_examples=100, deadline=None)`

  - [ ]* 10.6 Property test P18 - etykieta artysty wpływa na wyjście Modelu_GAN_Warunkowanego
    - **Property 18: Etykieta artysty wpływa na wyjście Modelu_GAN_Warunkowanego**
    - **Validates: Requirements 11.9**
    - W `tests/property/test_stargan_label_effect.py` (manualne uruchomienie z trenowanym checkpointem) weryfikować, że `dist(extract(G(x, c1)), extract(G(x, c2))) ≥ ε` dla `c1 ≠ c2`
    - Próg `ε` zdefiniowany w konfiguracji testów; `@settings(max_examples=20, deadline=None)`

  - [ ]* 10.7 Property test P19 - kierunkowość warunkowania
    - **Property 19: Kierunkowość warunkowania Modelu_GAN_Warunkowanego**
    - **Validates: Requirements 11.10**
    - W `tests/property/test_stargan_directionality.py` (manualne) weryfikować, że średnia odległość `extract(G(x, c_A))` od stylu `A` jest ≤ średniej odległości od stylu pozostałych artystów
    - `@settings(max_examples=20, deadline=None)`

- [ ] 11. Ewaluator obiektywny i subiektywny
  - [x] 11.1 Implementacja funkcji odległości
    - W `src/musicians_style/evaluation/distance.py` zaimplementować `euclidean(f1, f2_or_aggregated)` i `mahalanobis(f1, f2_or_aggregated, covariance)` jako czyste funkcje deterministyczne
    - Walidować: skończoność, nieujemność wyniku, identyczne wymiary wejść
    - _Requirements: 6.1, 6.2_

  - [x] 11.2 Implementacja ObjectiveEvaluator z testami statystycznymi
    - W `src/musicians_style/evaluation/objective.py` zaimplementować klasę `ObjectiveEvaluator` z metodą `evaluate(pairs, style_aggregated) -> EvaluationReport`
    - Obliczać dla każdej pary `(input, output)`: `dist_to_style_before`, `dist_to_style_after`, `dist_input_output`
    - Reguła doboru testu: `n ≥ 10` → test t-Studenta dla prób zależnych jeśli różnice mają rozkład bliski normalnemu (Shapiro-Wilk), w przeciwnym razie test rang Wilcoxona; α = 0.05; pomijać statystyki opisowe gdy n ≥ 10
    - `n < 10` → log warning + raport tylko statystyk opisowych, statistical_test = None
    - Raportować wartości p w `objective_report.json`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 11.3 Generowanie wykresów porównawczych
    - W `src/musicians_style/evaluation/plots.py` zaimplementować generację histogramów i wykresów pudełkowych cech `Utworu_Wejściowego`, `Utworu_Wyjściowego`, `Zbioru_Stylu` używając `matplotlib`
    - Zapisywać w formacie PDF lub EPS zgodnie z konfiguracją (`evaluation.plot_format`)
    - _Requirements: 6.6_

  - [x] 11.4 Implementacja SubjectiveEvaluator z FluidSynth
    - W `src/musicians_style/evaluation/subjective.py` zaimplementować `SubjectiveEvaluator.prepare_listening_set(n_pairs=10)` zwracający `ListeningSet` z parami `(input, output)` + fragmenty referencyjne `Zbioru_Stylu`
    - Metoda `render_to_audio(midi_path, soundfont, clip_seconds=15)` używająca `FluidSynth` do generacji WAV (44.1 kHz, 16 bit) - ujednolicony SoundFont
    - Dla próbek z chronionego `Zbioru_Stylu` ograniczać długość do 15 s
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 11.5 Implementacja formularza ankietowego i analizy odpowiedzi
    - Metoda `build_form(listening_set, form_type: Literal["ABX", "MOS"]) -> FormSpec` generująca specyfikację formularza
    - Metoda `analyze_responses(responses)` obliczająca średnią, odchylenie standardowe, przedziały ufności (bez przycinania do skali [1, 5]); test istotności gdy n_respondents ≥ 20
    - _Requirements: 7.4, 7.5_

  - [ ]* 11.6 Property test P10 - walidność funkcji odległości
    - **Property 10: Walidność funkcji odległości w ewaluacji**
    - **Validates: Requirements 6.1, 6.2**
    - W `tests/property/test_distance_validity.py` strategia `feature_vector_strategy` generująca losowe wektory cech
    - Test: `distance(f1, f2)` jest skończona, nieujemna, deterministyczna; testować zarówno euclidean jak i Mahalanobisa z losową dodatnio-półokreśloną macierzą kowariancji
    - `@settings(max_examples=200, deadline=None)`

  - [ ]* 11.7 Testy jednostkowe ewaluatora
    - W `tests/unit/test_objective_evaluator.py` testować: scenariusz `n=5` (brak testu, log warning), `n=10` (test t-Studenta), `n=10` z różnicami nie-normalnymi (test Wilcoxona), poprawne raportowanie wartości p
    - _Requirements: 6.3, 6.4, 6.5_

- [ ] 12. CLI i integracja end-to-end
  - [ ] 12.1 Implementacja CLI z komendami acquire, train, infer, evaluate
    - W `src/musicians_style/cli.py` zaimplementować z `argparse` lub `click` komendy: `midi-style acquire --config CONFIG`, `midi-style train --config CONFIG`, `midi-style infer --target_artist NAME --checkpoint CKPT --input X.mid --output Y.mid [--seed S]`, `midi-style evaluate --pairs PAIRS_DIR --style MANIFEST`
    - Komenda `infer` SHALL wymagać obowiązkowego parametru `target_artist` niezależnie od trybu Punktu_Kontrolnego
    - Kody wyjścia: 0 (sukces), 1 (uncaught), 2 (walidacja wejścia/konfiguracji), 3 (zasoby/OOM)
    - Każde uruchomienie SHALL inicjalizować katalog eksperymentu `experiments/{name}_{ts}/` z kopią konfiguracji i commit hashem
    - _Requirements: 5.1, 5.8, 8.2_

  - [ ]* 12.2 Testy integracyjne CLI
    - W `tests/integration/test_cli.py` testować pełne ścieżki: `acquire` → `train` (z tiny config, 1 epoka) → `infer` → `evaluate`
    - Testować scenariusze błędów: `target_artist` spoza listy → exit 2 + lista dostępnych etykiet, `target_artist` niezgodny z trybem `per_artist` → exit 2
    - _Requirements: 5.8, 5.9, 5.10_

  - [ ]* 12.3 Property test P15 - reprodukowalność end-to-end
    - **Property 15: Reprodukowalność end-to-end**
    - **Validates: Requirements 8.3**
    - W `tests/property/test_e2e_reproducibility.py` (manualne, oznaczyć `@pytest.mark.slow`) dla `seed × tiny_config × tiny_dataset` weryfikować, że dwa pełne uruchomienia (trening + inferencja) zwracają `Utwory_Wyjściowe` o `Wektorach_Cech` zgodnych z dokładnością `1e-5`
    - `@settings(max_examples=5, deadline=None)`

  - [ ]* 12.4 Smoke test wydajnościowy
    - W `tests/integration/test_resource_smoke.py` (manualne, oznaczyć `@pytest.mark.slow`) weryfikować: ekstrakcja cech pliku 5-minutowego < 10 s na CPU, zużycie VRAM < 6 GB podczas treningu z domyślnym batch size
    - _Requirements: 2.5, 9.4_

- [ ] 13. Końcowy checkpoint - cały Pipeline integracyjny
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Zadania oznaczone `*` są opcjonalne i mogą zostać pominięte dla szybszego MVP. Każde z nich realizuje testy (jednostkowe, property-based, integracyjne).
- Każde zadanie referuje konkretne kryteria akceptacji z `requirements.md` w polu `_Requirements:_`.
- Zadania property-based testowe (P1-P19) jawnie odwołują się do właściwości z sekcji *Correctness Properties* w `design.md` oraz do walidowanych kryteriów w `requirements.md`.
- Punkty kontrolne (zadania 4, 7, 13) gwarantują inkrementalną walidację po zakończeniu kluczowych komponentów.
- Wymaganie 10 (`Dokument_Dyplomowy`) nie jest realizowane jako zadanie kodujące - obejmuje przygotowanie dokumentu LaTeX poza zakresem niniejszego planu (manualne opracowanie i recenzja merytoryczna). Skrypty pomocnicze do eksportu tabel i wykresów dla pracy są wytwarzane w ramach zadań 11.3 i 11.5.
- Wymagania zasobowe 9.1, 9.2, 9.3 walidowane są przez ręczne uruchomienie smoke testu (12.4) i charakter wytworzonej infrastruktury (`device: cuda|cpu` w konfiguracji, brak twardych zależności od GPU w pipeline).
- Wszystkie property testy używają `Hypothesis` z `@settings(max_examples ≥ 100)` dla testów lekkich i `max_examples ≥ 20` dla testów ciężkich (GAN training).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4", "2.1"] },
    { "id": 2, "tasks": ["1.3", "2.2", "2.3", "3.1", "5.1"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "2.7", "3.2", "5.2", "6.1", "8.1", "8.2"] },
    { "id": 4, "tasks": ["3.3", "3.4", "3.5", "3.6", "3.7", "5.3", "5.4", "5.5", "6.2", "6.3", "8.3", "8.4"] },
    { "id": 5, "tasks": ["6.4", "6.5", "6.8", "9.1"] },
    { "id": 6, "tasks": ["6.6", "6.7", "9.2", "11.1"] },
    { "id": 7, "tasks": ["9.3", "9.4", "9.5", "9.6", "10.1", "10.2", "11.2", "11.6"] },
    { "id": 8, "tasks": ["10.3", "10.4", "10.5", "11.3", "11.4", "11.7"] },
    { "id": 9, "tasks": ["10.6", "10.7", "11.5", "12.1"] },
    { "id": 10, "tasks": ["12.2", "12.3", "12.4"] }
  ]
}
```
