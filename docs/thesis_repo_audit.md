# Audyt repozytorium w kontekście pracy „Modelowanie stylu artystycznego muzyków”

Data audytu: **13 września 2026**.

**Analizowany commit HEAD: `f58f084963cfe8281a22b7c0ec88aebaac667d59`** (`add E4 improvement handoff`, 2026-09-04 01:32:23 +0200). Przed audytem `git status --short` był pusty. Hash oznacza wersję kodu i dokumentacji śledzonej przez Git po metadanej anonimizacji historii. Analiza obejmuje także zastane, ignorowane przez Git lokalne dane i eksperymenty; ich zawartości nie da się odtworzyć wyłącznie z tego commita. Nie jest to hash kodu użytego we wszystkich historycznych treningach.

Zakres: źródła, konfiguracje, testy, historia Git, dokumentacja, lokalne manifesty/cache/splity, JSON/CSV/JSONL, checkpointy E4, wykresy i próbki MIDI, struktura notebooków dostawcy danych oraz materiały pomocnicze. Nie przeprowadzono treningu, nowych eksperymentów muzycznych, odsłuchu ani refaktoryzacji. Nie modyfikowano zastanych wyników. Zapisane niżej liczby eksperymentalne pochodzą z istniejących artefaktów; kontrole integralności i uruchomienie testów są czynnościami audytu.

Oznaczenia: **potwierdzone** — bezpośredni odczyt kodu lub artefaktu; **interpretacja** — ocena znaczenia tego dowodu; **nieustalone** — repo nie dostarcza wystarczającej informacji. Status GO/NO-GO jest decyzją konkretnego protokołu, nie ogólnym orzeczeniem o skuteczności transferu.

## 1. Executive summary

Projekt obecnie realizuje dwa powiązane zadania: stylometrię trzech kompozytorów na score MIDI ASAP oraz przekształcanie utworów między tymi domenami. Obok historycznego GAN-u i globalnego GA istnieją już: E1 z klasyfikacją 93 cech kompozycyjnych, E2 jako baseline globalnego GA, E3 jako GA z lokalnymi transformacjami i ograniczeniami treści oraz E4 jako osobny, segmentowany GAN warunkowany kompozytorem.

**Poziom zaawansowania:** prototyp badawczy z ukończonymi eksperymentami i materiałem do rozdziału wynikowego, lecz bez potwierdzonego systemu transferu na dowolnym wejściowym MIDI. E1: GO; E2: ukończony baseline; E3: GO z ograniczeniami po analizie post-hoc; E4.5: validation NO-GO, bez outer testu. E4.6 jest propozycją w handoffie, nie implementacją.

**Największy atut:** połączenie korpusu fortepianowego, grupowania na poziomie dzieł, jawnych cech, niezależnego względem celu GA klasyfikatora i zapisanych wyników dla tych samych 300 zadań E2/E3. E1b random forest osiągnął balanced accuracy **0,8642**, macro-F1 **0,8696**; istnieją predykcje OOF, macierz pomyłek i importance.

**Największa luka:** nie wykazano jeszcze silnego, muzycznie wiarygodnego przesunięcia stylu przy zachowaniu wszystkich istotnych aspektów treści. E3 poprawia proxy celu średnio o **0,0377**, ale zgodność własnej funkcji celu z tym proxy jest bardzo słaba (Pearson **0,0440**). Zachowanie melodii oznacza ochronę heurystycznej najwyższej nuty na onset, z dopuszczoną transpozycją; nie oznacza zachowania harmonii, fraz ani rzeczywistej melodii w wielogłosie.

**Ocena kierunku — interpretacja:** kierunek feature/statistics + kontrolowana optymalizacja jest sensownym i obecnie najtańszym rdzeniem pracy inżynierskiej. E4 może być wartościowym negatywnym eksperymentem dodatkowym. Nie ma uzasadnienia, aby na tym etapie uzależniać ukończenie pracy od nowego Transformera lub modelu pretrained. Uczciwa teza dotyczy rozróżniania repertuarów i ograniczonego transferu cech, a nie uniwersalnego „stylu artysty”.

Wcześniejszy `docs/AUDYT_REPOZYTORIUM.md` opisuje stan z 1 września. Jego stwierdzenia o braku klasyfikatora i kompletnych wyników są już historyczne; uwagi o starym potoku GAN nadal mają podstawę w kodzie.

## 2. Mapa repozytorium

| Element | Rola i aktualna istotność | Status / zastrzeżenie |
|---|---|---|
| `src/musicians_style/midi/` | Parser SMF, niemutowalne zdarzenia nutowe, writer, historyczny pianoroll | Wspólny fundament. Pianoroll i parser mają określone straty informacji |
| `features/`, `evaluation/distance.py` | Historyczny wektor 42 wartości i odległości | Nadal używane przez E1a/E2 i stare API; nie zastępować ich w zamrożonym baseline |
| `e1/` | Audyt ASAP, grouped nested CV, 93 cechy, open-set, raport zamknięcia | Najmocniejszy komponent stylometrii |
| `ga/`, `e2/` | Globalna transformacja czterogenowa, zamrożony eksperyment i diagnostyka | Baseline; wyniki pokazują istotną degradację czasu/rytmu |
| `e3/` | Taktowanie, Skyline, profile train-only, pięciogenowy GA, ograniczenia, raport post-hoc | Główna metoda do wykorzystania i krytycznej oceny |
| `e4/` | Czterotaktowe onset/frame, sampler, mały GAN, selection, dekoder, bramka validation/test | Nowszy potok neuronowy; aktualny wynik NO-GO |
| `models/`, `training/` | Pierwotne generatory StarGAN/CycleGAN i trener | Historyczny, nadal wykonywalny potok; E4 korzysta z własnych implementacji |
| `inference/pipeline.py`, `cli.py` | Historyczne GAN/GA/combined przez API, polecenia acquire/train/infer/evaluate | Nie jest publicznym interfejsem transferu E3/E4. E1–E4 mają własne `python -m ...` |
| `data/`, `evaluation/subjective.py`, `plots.py` | Manifesty, pozyskiwanie plików/YouTube, rendering, formularze odsłuchu i wykresy | Pomocnicza infrastruktura; brak odnalezionych wyników badania słuchowego |
| `configs/e1_asap.yaml`–`e4_asap.yaml` | Protokoły E1–E4 | Istotne; YAML bieżący nie zawsze odzwierciedla efektywne CLI historycznego runu |
| `configs/default_*.yaml`, `custom_conditional.yaml` | Konfiguracje pierwotnego prototypu | Historyczne/demo; `custom_conditional` wskazuje Kaggle i CUDA |
| `datasets/asap-dataset-1.2/` | Lokalna kopia score/performance MIDI, XML, metadanych i anotacji | E1–E4 używają wybranych score MIDI, nie wykonawczych MIDI/audio |
| `datasets/derived/e1_asap/` | Manifesty, quality report, splity, cache 42/93 cech, rozszerzenie open-set | Kluczowe dane pochodne, ignorowane przez Git |
| `datasets/kaggle_midi_classic_music/`, `datasets/outputs/` | Stary korpus i dziewięć ręcznych wyników inferencji | Archiwum prototypu, słabsza kontrola jednorodności i pochodzenia |
| `experiments/` | 27 lokalnych katalogów runów; wyniki i checkpointy | Ignorowane przez Git. Backupy E2/E3 nie są replikacjami |
| `docs/results/E1.md`–`E4.md` | Śledzone podsumowania wyników | Wygodne do pracy, ale weryfikować z surowymi JSON i ich ograniczeniami |
| `docs/*PLAN*`, `E3_LIT_UZASADNIENIE.md`, `E4_IMPROVEMENT_HANDOFF.md` | Plany, motywacja i proponowane dalsze warianty | Oddzielać plan od wykonania; E4 handoff jest najnowszy |
| `tests/unit`, `tests/property`, `tests/integration` | Regresje implementacji i własności; integracja YouTube | Nie zastępują ewaluacji naukowej. W audycie uruchomiono tylko unit |
| `.kiro/specs/`, `.kiro/steering/` | Pierwotne wymagania, design i lista prac | Dokumentacja pomocnicza, nie dowód zakończenia eksperymentów |
| `Literatura/`, `Literatura - arkusz.csv`, `logseq_pages/` | 26 PDF-ów, arkusz literatury i 20 stron notatek | Tło i dawny zakres obejmujący też audio; brak wyników nowych modeli w tych materiałach |
| `soundfonts/FluidR3_GM.sf2` | Zasób syntezy do odsłuchu | Zastany zasób, brak odnalezionych WAV wynikowych |
| `build/lib/`, `.venv/` | Kopie pakietu i środowisko | Nie traktować jako źródła aktualnego kodu; wykryto starą instalację pakietu |

Nie znaleziono własnych notebooków eksperymentalnych projektu. Cztery `.ipynb` są w `datasets/asap-dataset-1.2/util/`: `check_dataset_quality`, `create_json`, `full_ann_pipeline`, `usage_example`. Mają zapisane outputy bez bloków typu error, lecz dotyczą narzędzi/anotacji dostawcy danych; nie dokumentują treningów E1–E4. Nie uruchamiano ich i nie potwierdzono reprodukowalności ich outputów.

## 3. Dotychczasowe eksperymenty

Chronologia kodu: maj 2026 — pierwotny GAN/GA/API; lokalne runy nazwane datą 29 lipca — pierwszy GAN; 1–2 września — audyt i E1; 2–3 września — E2/E3; 2–4 września — E4.3, E4.5 i handoff. Daty katalogów, UTC logów i daty commitów w +0200 nie są jednym zegarem. Kolejność ukończenia E2/E3 również nie odpowiada numerom: pełny E3 ukończono przed ponownym pełnym uruchomieniem E2.

### 3.1. Pierwotny conditional GAN na korpusie Kaggle

- **Cel:** wielodomenowe przekształcenie MIDI do Bacha, Chopina lub Mozarta; adaptacja idei StarGAN.
- **Dane:** manifesty w `experiments/custom_conditional_2026-07-29_211645/`: 131/136/87 plików, łącznie 354. `source=local`, dane w `datasets/kaggle_midi_classic_music/`. Etykiety z katalogów artystów, nie z niezależnej walidacji autorstwa.
- **Reprezentacja/preprocessing:** pojedyncza binarna macierz sustain `[1,64,84]`, pitch `[24,108)`, 4 kroki na ćwierćnutę. Każdy plik daje tylko pierwsze 64 kroki; zwykle cztery takty 4/4. Ścieżki są scalane, velocity/kanał nie wchodzą do tensora.
- **Model i I/O:** `models/stargan.py`, `Trainer._train_step_conditional`; `G(fragment, one_hot_cel)` → fragment. Target losowany z wszystkich klas, także źródłowej. Inferencja historyczna również obejmuje jedno okno.
- **Trening:** seed 42, batch 16, Adam, LR 0,0002, 200 planowanych epok. Run `...214352` ma 528 linii logu, 21 checkpointów i dwie iteracje epoki 22; nie ma dowodu ukończenia 200 epok ani przyczyny przerwania.
- **Ewaluacja:** adversarial/classification/cycle/identity losses; `val_identity_l1` na tym samym loaderze co trening, z etykietą źródłową. Brak niezależnego splitu i miarodajnej ewaluacji zmiany stylu.
- **Wyniki:** log epoki 1/21: loss G **4,3938/1,8675**, loss D **1,6158/0,1495**, identity L1 **0,04962/0,06536**. Dziewięć `datasets/outputs/*.mid` istnieje, lecz nie ma sidecaru wiążącego je z konkretnym checkpointem, inputem, poleceniem i seedem.
- **Wniosek:** potwierdzony trening prototypu i spadek strat; brak dowodu skutecznego transferu. Heterogeniczny repertuar i ucięcie wejścia ograniczają wartość naukową wyników.

Pozostałe 14 katalogów `custom_conditional_*` zawierają manifesty albo tylko konfigurację/hash i puste podkatalogi. Brak logu nie pozwala orzec, czy były błędem, pustą inferencją czy przerwanym uruchomieniem. Nie są niezależnymi wynikami treningu.

### 3.2. `per_artist` i przepływ combined

**Cel zapisany w kodzie:** per-artist transformacja i kombinacja GAN+GA. Dane per-artist to jeden manifest jednej domeny, reprezentacja jest historycznym sustain pianorollem. Jest jeden G i jeden D, cycle to `G(G(x))≈x`, identity to `G(x)≈x`, optymalizacja Adam. Nie ma dwóch przeciwnych generatorów i dwóch domen właściwych pełnemu CycleGAN. Combined wykonuje sekwencję GAN→GA lub GA→GAN przez API.

**Status:** implementacje i testy istnieją; nie znaleziono runu treningowego `per_artist`, kompletnego porównania combined ani ich wyników eksperymentalnych. Nie przedstawiać ich jako wykonanych metod porównawczych. Literatura źródłowa o symbolicznym CycleGAN dotyczy transferu **gatunków**, co dodatkowo nie jest równoznaczne z kompozytorem ([Brunner et al.](https://arxiv.org/abs/1809.07575)).

### 3.3. E1.0/E1.1 — audyt danych i podziały

**Cel:** jednorodny korpus fortepianowych score MIDI i podział bez przecięcia dzieł. **Dane:** ASAP, Bach/Beethoven/Chopin. **Preprocessing:** unikalne `(composer,title)`, wybór kanonicznego score, parsowanie i SHA, grupowanie dzieł. **Algorytm:** audyt jakości i `StratifiedGroupKFold`; bez treningu. **I/O:** metadane/MIDI → manifest, quality report, powtarzane nested splity. **Ocena:** liczebności, integralność, grupy/SHA, outliery. **Artefakty:** `datasets/derived/e1_asap/{manifest,quality_report,splits}.json`.

**Wynik:** 150 zaakceptowanych próbek, 0 wykluczeń i brak grup identycznego SHA; 87 grup. 5 repeatów × 5 outer foldów, w każdym 3 inner foldy. Audyt ponownie sprawdził wszystkie zapisane przecięcia sample/group/SHA: **0**. Wniosek: poprawny techniczny fundament pod split na poziomie zdefiniowanych grup, bez gwarancji braku wszystkich muzycznych podobieństw między grupami.

### 3.4. E1a — klasyfikacja historycznych cech

**Cel:** ustalić, ile sygnału kompozytora niosą stare cechy i cechy pozbawione zależności od czasu wykonania. **Dane:** te same 150 score MIDI. **Reprezentacja:** `legacy_full` 42 wartości; `legacy_score_only` 38 po usunięciu tempo, density/s, mean/std duration/s. **Preprocessing:** cache; filtr wariancji wewnątrz pipeline, scaler tylko dla logistic regression. **Modele:** Dummy most-frequent, balanced logistic regression (`C` grid), balanced RF 300 drzew (`max_features`, `min_samples_leaf` grid). **I/O:** wektor utworu → kompozytor/probability. **Trening:** nested grouped CV 5×5 z inner 3. **Ocena:** OOF balanced accuracy/macro-F1, CI i permutacje zgodności gotowych predykcji.

**Artefakt:** `experiments/e1_asap_2026-09-01_215646/e1a_results.json`, `e1a_predictions.json`. Zapisany schema `e1.2.0` jest starszy niż bieżący kod `e1.2.1`: brak dowodu finalnego testu z ponownym uczeniem. RF full **0,7695**, RF score-only **0,7307**. Wniosek: sygnał jest obecny, ale wąski zestaw gorzej odróżnia Chopina. Permutacje gotowych OOF nie zastępują testu null z retreningiem.

### 3.5. E1b — 93 cechy kompozycyjne

**Cel:** potwierdzić bardziej kompozycyjny sygnał i uzyskać proxy do transferu. **Dane:** ASAP 150/87 grup. **Reprezentacja:** 93 wartości pitch/melody/rhythm/texture/harmony/structure w jednostkach score, bez velocity i tempo jako predyktorów. **Preprocessing/modele/trening:** jak E1a, z 5 seedami splitów, analizą wszystkich próbek i jednej próbki na grupę oraz outer-held-out permutation importance (10 powtórzeń).

**Ewaluacja:** balanced accuracy, macro-F1, klastrowe CI, confusion matrices, analiza form/błędów; 99 permutacji etykiet grup z retreningiem dla niedummy modeli w pierwszym predeclared repeat i stałych pipeline'ach. Ten test nie odtwarza całego nested strojenia po każdej permutacji.

**Wyniki:** run `e1_asap_e1b_2026-09-02_005705_449734` ukończony, 150 fold tasks i poprawna kontrola 3555 predykcji OOF across modele i tryby. RF full **BA 0,864240**, CI **[0,837808;0,890590]**, F1 **0,869562**, permutation p **0,01**. RF one-sample/group **BA 0,839152**. Raport E1.4 daje GO. Wcześniejszy run `...225113_456415` ma status `failed`, błąd **KeyboardInterrupt**.

**Ważne:** cache deklaruje 13 wariantów cech, ale ukończony run wykonywał **wyłącznie `composition_full`**. Nie znaleziono ukończonych sześciu single-group ani sześciu leave-one-group-out ablacji. Nazwa istniejącego `ablations.png` nie dowodzi ich wykonania. Wniosek: 93 cechy rozróżniają te repertuary; nie wykazano niezależności stylu od epoki, formy czy źródła notacji.

### 3.6. E1-open i E1.4

**Cel:** zakres stosowalności klasyfikatora i formalne zamknięcie E1. **Dane:** open manifest 214 score MIDI; trzy znane klasy 150 próbek, calibration Haydn/Mozart/Schumann 27, test Liszt/Schubert/Rachmaninoff/Ravel 37. **Reprezentacja:** 93 cechy. **Algorytm:** closed-set LR/RF, próg max-class confidence; brak uczenia klasy `other`. Calibration known z foldów 0/1; osobne unknown composers do calibration/test. Są także rotacje leave-one-known-composer-unknown.

**Trening/ocena:** grouped protokół pierwszego repeat, kalibracja progu z warunkiem known TPR 0,95; AUROC/AUPRC, recall unknown/known, coverage. **Wyniki:** dwa ukończone runy open; finalny `...231352_526004` ma 8 summaries (2 modele × external + 3 rotacje). External RF: AUROC **0,742**, AUPRC **0,883**, unknown recall **0,108**, known recall **0,978**, coverage **0,953**. LR unknown recall **0**. E1.4 tworzy report/CSV/PNG i closure manifest, nie nowy model.

**Wniosek:** open-set wykrywanie obcego kompozytora jest słabe mimo dobrego closed-set. Nie wolno przenosić BA 0,864 na twierdzenie o dowolnym artyście/wejściu.

### 3.7. E2 — globalny GA jako baseline transferu

**Cel:** zamrożony pomiar starego GA. **Dane:** test każdego z pięciu outer foldów **repeat 0**, oba inne cele dla 150 wejść → 300 zadań. Pilot: 3 źródła × 2 cele × 3 seedy = 18. **Reprezentacja:** pełne MIDI w `InternalRepr`, fitness 42 cechy, niezależna ocena 93 cech E1b.

**Preprocessing/model:** profile średniej/kowariancji celu wyłącznie z train danego foldu. Cztery geny: globalna transpozycja, mnożnik onsetów, mnożnik durations i offset velocity. Population 100, max 200 generations, tournament 3, BLX-α pod nazwą `uniform`, mutation, elitism 2, stagnation 30. Nie ma uczenia generatywnego; GA optymalizuje każdy input osobno. **I/O:** score MIDI + profil celu → pełne zmienione MIDI.

**Ocena:** RF 300 drzew train-only, Δp_target, target hits, distances grup E1b, content, round-trip, fitness i runtime. CI bootstrap grup źródłowych; sign permutations 999 i Holm per direction. Pilot kontroluje deterministyczne powtórzenie.

**Wyniki:** `experiments/e2_asap/`, 300/300 complete, Δp **0,027666** (podsumowanie 0,028), CI **[0,020;0,036]**, p **0,001**; target hit **4,3%→4,7%**. Średni length error **0,370**, tylko **6%** w ±5%, onset-F1 **0,365**, contour **0,990**. Wszystkie zadania zakończone stagnacją. Mean runtime/task **111,017 s**, wall run **8393,796 s**, 4 workers.

**Wniosek:** GA skutecznie poprawia własny fitness, ale duży fitness gain (**16,661**) nie oznacza dużej zmiany stylu; wysoki contour przy degradacji czasu pokazuje słabość samej metryki konturu. Direction Beethoven→Chopin ma Δp **0,081**, Chopin→Bach **−0,004**. Wynik jest użytecznym negatywnym baseline, a nie udanym transferem z zachowaniem treści.

### 3.8. E3 — lokalne operacje i ograniczenia treści

**Cel:** lepszy kompromis styl/treść niż E2. **Dane/I/O:** identyczna pełna macierz 300 zadań; pilot 6, seed 1729. **Reprezentacja:** pełne `InternalRepr`, rzeczywiste metrum, Skyline najwyższej nuty każdego onsetu; **67** cech własnego celu, inne niż 93 cechy ewaluatora. **Preprocessing:** profile target train-only z udowodnioną rozłącznością sample/group/SHA.

**Model:** pięć globalnych sił sterujących lokalnym wykonaniem: transpozycja ±6, onset movement akompaniamentu wewnątrz taktu, durations akompaniamentu, thinning albo octave doubling. Genotyp nie ma osobnych parametrów dla każdego taktu. Cel to równoważna średnia zysków z RMS standaryzowanych grup pitch/rhythm/texture. Population 32, max 60 generations, stagnation 12, elitism 2, tournament 3, cache MIDI/genomów. Deb ranking: feasible przed infeasible; identity jako punkt startowy/fallback.

**Ograniczenia bieżącego kodu:** zachowanie chronionych nut po transpozycji, meta/TPB/SMF, length ≤ krok TPB/16 i ≤5%, note count 0,9–1,1, polyphony ≤ max(source,target train), brak wzrostu zero-duration, semantic round-trip. Nie ma twardego zachowania chroma ani harmonii.

**Ocena/wyniki:** RF train-only 93 cechy, content i porównanie sparowane E2; `results.json` 300 complete, `closure_analysis.json` post-hoc. Δp **0,037705**, CI **[0,028496;0,046044]**, p **0,0001**; E3−E2 **0,010040**, CI **[0,001339;0,022403]**, p **0,0395**. Onset-F1 **0,982615**, contour mean **0,964662**, median **1**, chroma **0,719536**. Length error mean **0,000001744**, max **0,000424391** — wyświetlane w podsumowaniu 0,0000 nie oznacza idealnej równości wszystkich długości. Wall full invocation **17149,341 s**, 4 workers.

**Wynik historyczny vs korekta:** raport GO powstał po korekcie kontroli round-trip i feasible identity. Zapisane `results.json` i output MIDI nie zostały przez tę analizę zmienione. Closure raportuje 271 exact oraz 300 semantic/parseable/revised feasible. Są to wyniki protokołu closure, a nie dowód, że historyczny GA już stosował dokładnie bieżący kontrakt. `reporting._measure_outputs` korzysta ze starego flag `constraints.roundtrip_valid`; dla pozostałych sprawdza semantykę tylko identity względem źródła i usuwa jego naruszenie polyphony. Pełne odtworzenie pre-serialization obiektów wszystkich kandydatów nie jest zapisane.

**Wniosek:** wyraźnie lepsze zachowanie czasu i mały, kierunkowo nierówny zysk proxy. 185/67/48 zadań z Δp dodatnim/zerowym/ujemnym. W 48 własny cel poprawił się, a proxy spadło. Raportowe `identity_outputs=45` faktycznie liczy `style_gain==0`, więc nazwa nie dowodzi bitowej identity. E3 nie jest bezstratną reharmonizacją ani potwierdzoną imitacją kompozytora.

### 3.9. E4.3 smoke i E4.5 pełny trening/walidacja

**Cel:** sprawdzić, czy naprawiony potok wielodomenowy GAN pozwala na dalszy kierunek neuronowy. **Dane:** ASAP, repeat/outer/inner fold 0. Po wykluczeniach: train **71** utworów, validation **43**, outer test **28**; train nonempty segments Bach/Beethoven/Chopin **349/1559/862**.

**Reprezentacja/preprocessing:** dwa kanały onset/frame `[2,64,84]`; 4 takty/okno, **16 kroków na takt**, pitch `[24,108)`, padding mask, carry sustain między oknami. To inne odwzorowanie czasu niż stare 4 steps/beat. Metadane i mapa taktów idą ścieżką pomocniczą. BalancedComposerSampler równoważy kompozytorów. Trening unpaired, target zawsze różny od source.

**Model/I/O:** mały conditional conv GAN, conv_dim 32, 3 residual blocks, PatchDiscriminator i composer head, G(fragment,target one-hot) → dwa logits; stitcher składa pełne MIDI. Brak jawnego content/style latent i brak modelu pretrained. **Trening:** Adam LR 0,0002, batch16, λcls/cyc/id **1/10/10**, maskowany L1 dla cycle/identity, seed1729, max30/patience5. Checkpoint selection na 3 validation źródłach/6 transferach; progi onset/frame kalibrowane osobno przez identity cell-F1.

**Artefakty:** `experiments/e4_asap` schema e4.3.0 to historyczny smoke z loss-selected best; nie pełny wynik porównawczy. `e4_asap_v2` schema e4.5.0 ma audit, smoke, snapshots, checkpointy, progress i validation MIDI. Bezpieczny odczyt wag potwierdził best epoch **1**, last **6**. Best progi **onset 0,3 / frame 0,2**, identity cell-F1 **1,0/1,0** na selection — nie event-F1 ani wynik transferu.

**Ocena:** E1b RF train-only na validation, style group gains, content/length/polyphony/fallback i zamrożone bramki. Validation: **86** kierunkowych outputów, mean Δp **0,005369**, 5/6 kierunków dodatnich; contour median **0,923248** <0,95, onset-F1 **0,854297**, chroma **0,999569**. Fallback **0/2752**, orphan starts **19916**, aktywne pitches na końcach okien **7873**. Pitch/rhythm/texture mean gains **+0,008594/−3,269917/−0,091442**. FAIL dla stylu grup, melodii i polyphony; decision **NO-GO**.

**Wniosek:** technicznie działający transfer pełnych plików nie spełnił kryteriów muzycznych. Spadek generator loss **2,645→1,029** towarzyszył pogorszeniu selection. Nie znaleziono `outer_test.json`; nie uruchamiano go w audycie. Protokół blokuje obliczenie transferu testowego przed validation GO. Dane testowe były jednak czytane do audytu/segmentacji — „outer test zamknięty” oznacza brak ewaluacji transferu, nie fizyczny brak odczytu MIDI (szczegóły w §12).

## 4. Dane i reprezentacja

### Korpusy i etykiety

| Korpus | Faktyczne użycie | Jednostka / ograniczenia |
|---|---|---|
| Lokalny ASAP 1.2 | E1–E4, wyłącznie `metadata.csv:midi_score` | 150 wybranych score; folder zawiera też performance MIDI i XML, które nie są wejściem tych modeli |
| Open rozszerzenie ASAP | E1-open | 214 score; unknown calibration i test są różnymi kompozytorami |
| Lokalny Kaggle classic MIDI | Pierwotny GAN i ręczne próby | Lokalnie 1417 plików o rozszerzeniu `.mid` bez względu na wielkość liter; stare manifesty wybierają 354 |
| `datasets/outputs/` | Dziewięć wyników starej inferencji | Brak kompletnego provenance; to nie dataset testowy |
| YouTube | Implementacja akwizycji i testy/mocki | Nie odnaleziono badawczego korpusu użytego w E1–E4 z transkrypcji YouTube |

Liczebność całego folderu MIDI nie jest liczebnością eksperymentu. README dostawcy ASAP ma niejednolite sumy opisowe/tabelaryczne; źródłem liczebności audytu są manifesty i realnie wybrane score, nie arbitralna liczba z README. Etykiety ASAP pochodzą z metadanych; forma jest inferowana z tytułu do analiz pomocniczych, nie jest podawana klasyfikatorowi jako string.

| Kompozytor E1 | Próbki | Grupy |
|---|---:|---:|
| Bach | 59 | 30 |
| Beethoven | 57 | 28 |
| Chopin | 34 | 29 |
| Razem | 150 | 87 |

Grupy Bacha łączą Prelude/Fugue o tym samym BWV; Beethoven — części jednej sonaty; Chopin — części sonaty rozpoznane regexem. Inne tytuły zwykle pozostają osobnymi grupami. Wspólne opusy/cykle, warianty o innym tytule lub podobne aranżacje nie są automatycznie równoważne tej definicji dzieła.

### Co jest reprezentowane i tracone

- **InternalRepr:** tick, duration_ticks, pitch, velocity, channel; TPB i SMF oraz tempo/time_signature/key_signature/program_change w meta. Nie zachowuje całego oryginalnego SMF: parser pomija m.in. control_change/pedał, pitch bend, tekst i identyfikację ścieżek. Writer odtwarza własną organizację ścieżek; equality wewnętrzna nie oznacza byte-equality ani pełnego zachowania wykonania.
- **42 cechy:** tempo, 12 pitch-class, 25 interwałów ograniczonych do ±12, density/s, mean/std duration/s, rest_ratio. Nie mają velocity/instrumentacji. Interwały starego ekstraktora są z sekwencji posortowanych nut, co w polifonii może mieszać głosy.
- **93 cechy:** score-only pitch/melody/rhythm/texture/harmony/structure; rytm w beats i względem metrum. Melody to najwyższy pitch na wspólnym onset; harmony z globalnego duration chroma i współrozpoczynających nut, nie pełnej analizy wszystkich współbrzmiących głosów. Structure opisuje powtarzalność/ngramy/lag, nie jawne granice fraz czy formę.
- **E3 67 cech celu:** onset bins16, duration bins6, pitch-class12, melody-relative intervals25 z clippingiem, chord sizes8. Część opisuje akompaniament wybrany przez Skyline. Różne materiały mogą mieć taki sam histogram; nie koduje globalnej kolejności motywów.
- **Historyczny GAN:** sustain binary, jeden początek utworu, zlewanie powtarzanych nut tej samej wysokości, brak dynamiki/głosów, stała siatka.
- **E4:** onset/frame poprawiają retrigger i stitching, ale quantization ma zmienną rozdzielczość w beats: 16 komórek dla każdego taktu, niezależnie od metrum. Triole/ornamenty i overlapy tej samej wysokości tracą szczegóły. Zerowe duration są pomijane w tensorze; pitches poza zakresem stitcher kopiuje z source. Velocity/channel są dobierane z najbliższej source nuty o tej samej wysokości (fallback 64/channel0), nie generowane w stylu celu. Meta jest kopiowane, także tonacja przy zmianie pitches.

**Ryzyko leakage:** zapisane E1 splity i konstrukcja profili E2/E3 mają silną kontrolę sample/group/SHA i fold-local fit; nie znaleziono bezpośredniego przecięcia tych identyfikatorów. Pozostają: repertuar/form/epoka/source confounding, niepełne grupowanie cykli, exact SHA zamiast muzycznej deduplikacji oraz adaptacja kolejnych metod po obejrzeniu wyników na tym samym korpusie. Dla historycznego GAN brak wydzielonego splitu jest potwierdzony. Dla E4 selection i thresholds używają validation zgodnie z planem; pełna validation nie jest już niezależnym testem wyboru modelu. Standardowe uzasadnienie grouped CV i train-only transformacji opisuje [dokumentacja scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html).

## 5. Co obecnie oznacza „styl”

| Ścieżka | Operacyjna definicja stylu | Co rzeczywiście mierzy |
|---|---|---|
| E1 | Etykieta Bach/Beethoven/Chopin i 93 deskryptory repertuaru | Rozróżnialność kompozytorów w closed-set, pośredni sygnał stylu |
| Stary GA/E2 | Mean/covariance 42 cech korpusu celu | Podobieństwo statystyczne, częściowo zależne od zapisów tempo/rytmu |
| E3 | Train-only profil 67 kontrolowalnych cech, trzy grupy | Podobieństwo akompaniamentu/faktury/statystyk, nie pełne autorstwo |
| Stary GAN/E4 | One-hot kompozytora i composer-head D | Rozkład domenowy wyuczony z nieparowanych okien |
| Niezależny ewaluator | Probability RF na 93 cechach | Proxy celu; Δp nie jest skalą percepcyjnego „procentu stylu” |

Nie ma osobnego style encodera, agregatora referencji ani wyodrębnionego embeddingu artysty, którym można sterować bez retreningu. Hidden activations GAN-u nie są zbadanym/disentangled latent style. Dodanie nowego kompozytora do one-hot wymaga zmiany modelu i treningu. Repo nie implementuje stylu gatunkowego jako osobnych etykiet eksperymentalnych, mimo historycznych odniesień do genre transfer.

Styl nie jest mierzony bezpośrednio przez ocenę ekspertów. Wszystkie obecne wyniki style similarity są statystycznymi proxy. E1-open pokazuje, że dobre closed-set rozpoznanie nie gwarantuje rozpoznania źródła obcego.

## 6. Co obecnie oznacza „content”

**Jawne content-preservation istnieje w E3; E4 ma straty rekonstrukcji i bramki treści. Pierwotny GA/E2 nie ma jawnego celu ani twardych ograniczeń treści.** Nie należy uogólniać braków starego prototypu na całe dzisiejsze repo.

| Aspekt | Obecny mechanizm | Granica dowodu |
|---|---|---|
| Melodia | E3 chroni najwyższą nutę każdego onsetu, jej tick/duration/velocity/channel; dopuszcza wspólną transpozycję | Heurystyka może wskazywać akompaniament w polifonii; nie gwarantuje rzeczywistego głosu melodycznego |
| Kontur | Trigram multiset Jaccard znaków interwału najwyższych onsetów; E4 gate median ≥0,95 | Ignoruje rozmiary interwałów, wiele elementów kolejności, tonację i dopasowanie w czasie |
| Harmonia | Globalne duration-weighted chroma cosine i deskryptory harmony | Diagnostyka, nie constraint E3. Chroma jest globalne i nie mierzy progresji |
| Rytm | E3 nie rusza timing chronionych nut; ogranicza lokalne onsety/duration akompaniamentu; onset-F1 | Onset-F1 widzi unikalne czasy, nie przypisane pitches/głosy/durations |
| Takty | Mapa z metrum, E3 długość/meta, E4 długość/bar_count i stitching | Zachowanie liczby taktów nie dowodzi zachowania fraz lub muzycznej formy |
| Całe MIDI | Round-trip/parser, liczniki nut, TPB/meta, historyczne feature input-output distances | To poprawność techniczna lub statystyki, nie pełna równoważność utworów |
| Frazy/motywy | Cechy repetition/lag; brak jawnego constraint | Nie ma segmentacji fraz, alignment motywów ani osobnej oceny struktury długiej |

Praktyczny content w E3 to „pozostaw heuristic melody i timing z niewielką zmianą liczby nut, pozwól zmieniać akompaniament i transponować”. W E4 cycle/identity L1 wymuszają przybliżoną rekonstruowalność tensora; nie gwarantują zachowania melodii w zdekodowanym MIDI.

## 7. Ewaluacja

### A. Style similarity

Istnieją: Euclidean/Mahalanobis do target mean przed/po, fitness gains, E3 grouped standardized RMS gains, E1b standardized distances sześciu grup w E2 i trzech w E4, probability target przed/po, **Δp_target**, probability drop source, target-class hit rates i wyniki kierunkowe. E1 BA/macro-F1 i importance walidują sygnał cech, ale same nie oceniają przeniesienia stylu. D_cls cross-entropy jest składnikiem treningu, nie niezależnym dowodem similarity.

Brakuje: niezależnej walidacji percepcyjnej/eksperckiej, sprawdzonej kalibracji probabilities, drugiego proxy o innym biasie i testu przenoszenia na external-source MIDI. Suma fitness gain i Δp nie ma wspólnej jednostki.

### B. Content preservation

Istnieją: relative/tick length error i tolerancja, onset-F1, melody contour trigram Jaccard, chroma cosine (shared E3/E4), liczba/ratio nut, bar_count/empty bars, meta/TPB/SMF preservation, protected-melody constraint E3, cycle/identity L1 i identity cell-F1. Feature input-output distance starego ewaluatora jest co najwyżej ogólnym proxy statystycznej zmiany.

Brakuje: event-level melody pitch/onset/duration alignment, transposition-aware oceny z jawną polityką tonacji, lokalnego harmony similarity, phrase/motif alignment i wyników ogonów rozkładów jako warunku akceptacji. Same średnie/mediany pozwalają ukryć pojedyncze poważne degradacje.

### C. General music quality

Istnieją diagnostyki: niepustość, zero-duration, mean/max polyphony, udział pustych taktów, orphan frames, aktywne pitches na końcach segmentów, note-count ratio oraz poprawny parse/round-trip. Nie są pełną miarą muzyczności. `subjective.py` ma rendering/ankiety, ale nie znaleziono wypełnionych ocen słuchaczy ani analizy jakości artystycznej. Brak zapisanej ewaluacji płynności fraz, harmonic correctness, voice-leading lub czytelności zapisu.

### D. Inne

E1: confusion matrices, BA/macro-F1, form errors, OOF coverage, grouped CI, fixed-OOF association permutations i retrained group-label permutation test. Open-set: AUROC/AUPRC, known/unknown recall, coverage, threshold, FPR przy target known TPR, known/unknown balanced accuracy, accuracy zaakceptowanych known i open-set confusion matrix oraz rotacje. **W AUROC/AUPRC klasą dodatnią jest known (`y_known=1`), więc AUPRC 0,883 nie jest average precision wykrywania unknown.** GA: generation0/identity gains, stagnation, gene-boundary rates, unique candidates/cache hits, runtime i determinism. Infrastruktura historyczna: Shapiro na różnicach, paired t-test albo Wilcoxon; E2/E3: clustered bootstrap, sign permutations i Holm. Zależne utwory/kierunki trzeba analizować na poziomie grup, nie traktować 300 zadań jako 300 niezależnych kompozycji.

**Problem agregacji:** E2 bootstrap losuje grupy w stratach kompozytora i zachowuje ich zadania. E3 closure najpierw uśrednia zadania każdej grupy, a potem bootstrapuje równo ważone grupy, choć wyświetlana główna średnia jest liczona po zadaniach. CI E3 i displayed mean nie opisują dokładnie tego samego ważenia. Test sign E2 też używa średnich grup. Przy wykorzystaniu istniejących tabel trzeba opisać oba estimandy; przyszły wspólny raport powinien stosować jedno ważenie. To nie jest powód do potajemnego nadpisania historycznych CI.

**Dodatkowe ograniczenie CI E1:** `_cluster_bootstrap_ci` tworzy klastry `(repeat,composer,group_id)` i losuje je osobno w stratach `(repeat,composer)`. To rozdziela powtarzane predykcje tych samych dzieł między repeatami zamiast losować jedno dzieło razem ze wszystkimi jego predykcjami. Potwierdzony jest taki algorytm; interpretacja: może zaniżać niepewność względem generalizacji na nowe dzieła. Zapisane CI należy podpisać dokładną metodą, a nie deklarować ich jako bezwarunkowej miary niepewności populacyjnej. Nie wyliczano tu alternatywnych przedziałów.

## 8. Wyniki nadające się do pracy

### Gotowa tabela klasyfikacji z istniejących JSON

| Zestaw / analiza | Model | Balanced accuracy | Macro-F1 | 95% CI BA |
|---|---|---:|---:|---|
| E1a legacy_full 42 | Dummy | 0,2805 | 0,2418 | [0,2411;0,3194] |
| E1a legacy_full 42 | LR | 0,6406 | 0,6405 | [0,6070;0,6730] |
| E1a legacy_full 42 | RF | 0,7695 | 0,7736 | [0,7431;0,7971] |
| E1a legacy_score_only 38 | LR | 0,6071 | 0,6064 | [0,5716;0,6428] |
| E1a legacy_score_only 38 | RF | 0,7307 | 0,7284 | [0,7045;0,7588] |
| E1b composition_full 93 | LR | 0,7981 | 0,8014 | [0,7653;0,8323] |
| E1b composition_full 93 | RF | 0,8642 | 0,8696 | [0,8378;0,8906] |
| E1b jedna próbka/grupa | LR | 0,7745 | 0,7719 | [0,7375;0,8115] |
| E1b jedna próbka/grupa | RF | 0,8392 | 0,8371 | [0,8064;0,8713] |

Źródła: `experiments/e1_asap_2026-09-01_215646/e1a_results.json:summaries`, `experiments/e1_asap_e1b_2026-09-02_005705_449734/e1b_results.json:summaries`. To zestawienie zastanych liczb, nie nowa klasyfikacja ani nowy test istotności różnicy E1a/E1b. Dummy pooled OOF poniżej 1/3 nie jest błędem odczytu: najczęstsza klasa train zmienia się między grouped foldami. Nie podmieniać wyniku na teoretyczny baseline 1/3.

Macierz E1b RF, rows true / columns predicted, kolejność Bach/Beethoven/Chopin:

| True \\ predicted | Bach | Beethoven | Chopin |
|---|---:|---:|---:|
| Bach | 292 | 2 | 1 |
| Beethoven | 0 | 259 | 26 |
| Chopin | 8 | 44 | 118 |

Suma **750 = 150 utworów × 5 repeatów**, nie 750 niezależnych utworów. Asymetria Beethoven/Chopin jest widoczna, Bach rozpoznawany prawie bezbłędnie. To zgodne także z confounding repertuaru: formy Prelude/Fugue są bardzo charakterystyczne dla Bacha w tej próbie.

### Gotowa tabela transferu; E4 oznaczać osobno

| Metryka | E2 outer OOF repeat0 | E3 outer OOF repeat0 | E4.5 inner validation |
|---|---:|---:|---:|
| Zadania | 300 | 300 | 86 |
| Mean Δp_target | 0,0277 | 0,0377 | 0,0054 |
| CI 95% Δp | [0,020;0,036] | [0,0285;0,0460] | Brak zapisanego CI |
| Onset-F1 mean | 0,365 | 0,9826 | 0,8543 |
| Contour trigram mean | 0,990 | 0,9647 | Nie podano w raporcie |
| Contour trigram median | Nie podano tu | 1,0000 | 0,9232 |
| Length error mean | 0,370 | 0,000001744 | Gate length PASS; nie zastępować średniej wynikiem gate |
| Chroma cosine mean | Brak tej metryki w podsumowaniu E2 | 0,7195 | 0,9996 |
| Decyzja | Baseline ukończony | GO z ograniczeniami, closure post-hoc | NO-GO |

E2/E3 mają te same źródła/cele/foldy i nadają się do sparowanego porównania. **E4 nie jest wynikiem na tym samym zbiorze ani ewaluatorem o identycznym fit-set**, więc nie wyciągać z tej tabeli testowej przewagi GA nad GAN-em. Oddzielić panel walidacyjny lub jawnie opisać różnicę. Wyższy chroma E4 nie równoważy jego problemów z rhythm/polyphony/melody.

### Inwentaryzacja artefaktów do tabel i rycin

| Artefakt / lokalizacja | Eksperyment i zawartość | Przydatność / zastrzeżenia |
|---|---|---|
| `datasets/derived/e1_asap/quality_report.json`, `manifest.json` | Liczebności, grupy, długości/note counts, outliery, SHA | Tak: charakterystyka korpusu; outlier to diagnostyka, nie automatyczne wykluczenie |
| `.../e1a_results.json`, `e1a_predictions.json` | Sześć model/variant summaries, fold scores i OOF | Tak: baseline cech; starszy schema i brak finalnego retraining testu |
| `.../e1b_results.json`, `e1b_predictions.json` | Modele, sensitivity, confidence, confusion, importance, form errors | Tak: główny wynik stylometrii; tylko full variant |
| `.../e1b.../report/predictions.csv`, `exclusions.csv`, `closure_manifest.json` | Eksport predykcji i formalna decyzja E1.4 | Tak: wygodne tabele i provenance; nie są nowym niezależnym runem |
| `.../e1b.../report/plots/confusion_matrix.png`, `fold_scores.png` | Macierz i outer fold scores | Tak; opisać repeat pooling i zależność wyników |
| `.../report/plots/feature_importance.png` | Held-out permutation importance cech/grup | Tak; korelacja cech ogranicza interpretację pojedynczych importance |
| `.../report/plots/ablations.png` | Wykres wariantów obecnych w wynikach | Warunkowo: nie opisywać jako wykonane leave-one-group-out |
| `.../e1_open_results.json`, `e1_open_predictions.json` | External unknowns i rotacje | Tak: ograniczenia closed-set; nie pomijać unknown recall 0,108 |
| `experiments/e2_asap/{results.json,results.csv,analysis.json}` | 300 zadań, gain/style/content, genes/runtime | Tak: baseline; style proxy i poważna degradacja czasu |
| `e2_asap/plots/style_gain_by_direction.png` | Sześć kierunków i CI | Tak: nierównomierność efektu |
| `e2_asap/plots/style_gain_vs_length_error.png`, `length_error.png` | Kompromis styl/długość | Tak: przykład braku content constraint |
| `e2_asap/plots/genome_distributions.png`, `gene_boundary_rates.png` | Rozkłady genów/saturacja | Tak: interpretowalna diagnostyka; offset velocity nieobecny w fitness |
| `e2_asap/plots/runtime_and_stagnation.png`, `tasks/*/ga.jsonl` | Czas i przebiegi GA | Tak: koszt optymalizacji; opis sprzętu/logów niekompletny |
| `experiments/e3_asap/results.json`, `closure_analysis.json` | 300 zadań, E3−E2, constraints, ogony content | Tak: główne porównanie; wyraźnie zaznaczyć post-hoc closure i różne ważenie CI |
| `e3_asap/tasks/*/ga.jsonl`, `result.json` | Historie gain/feasible/cache/genome per task | Tak: diagnostyka zbieżności, materiał do późniejszej wizualizacji istniejących danych; gotowych PNG E3 nie znaleziono |
| `e2_asap/tasks/*/output.mid`, `e3_asap/tasks/*/output.mid` | 300 pełnych outputów każdej metody plus piloty | Tak: przykłady/odsłuch; wybór według jawnej reguły, nie tylko najlepszych |
| `experiments/e4_asap_v2/{audit.json,segments.jsonl}` | Segmenty, occupancy, pitch/metrum, wykluczenia | Tak: kontrola reprezentacji; audit obejmuje również source test |
| `e4_asap_v2/progress.jsonl` | Sześć epok, loss G i selection_key | Tak: krzywa loss vs decoded quality; brak zapisanych epoch-average składowych loss D/adv/cls/cyc/id |
| `e4_asap_v2/{best.pt,last.pt}` | Checkpointy epoch1/6, geometry, thresholds, optimizer | Tak: reprodukcja inferencji, nie sam wynik muzyczny; checkpoint best **2791090 B** |
| `e4_asap_v2/{validation.json,validation_midi/,report.md}` | 86 validation outputs, trzy niespełnione bramki | Tak: negatywny eksperyment; brak outer-test comparison |
| `docs/results/E1.md`–`E4.md` | Zwięzłe raporty zamknięcia | Tak: źródło orientacyjne, nie zastępuje surowych danych |
| `custom_conditional_...214352/logs/train.jsonl` | Pierwotne loss curves, identity L1 epoch1–21 | Tak: opis porażki/prototypu; identity na train |
| `datasets/outputs/*.mid` | Dziewięć ręcznych prób | Tylko ilustracyjnie, brak pełnego provenance i quantitative evaluation |

Nie znaleziono gotowych publikacyjnych loss plots E4 ani danych ocen słuchaczy. Nie tworzono nowych wykresów/wyników, aby uzupełnić tę sekcję. Existing PNG są rastrowe; ewentualny późniejszy eksport tych samych danych do wektora jest pracą redakcyjną, nie nowym eksperymentem.

## 9. Ocena kierunku pracy

Oceny ryzyka i kosztu poniżej są **interpretacją inżynierską względem zastanego repo**, nie benchmarkiem przyszłych modeli ani estymacją konkretnej liczby GPU-hours.

| Kierunek | Co już istnieje | Co pozostało | Ryzyko / koszt | Mierzalność i sens pracy |
|---|---|---|---|---|
| **A. Cechy/statystyki + klasyczne ML / GA** | ASAP/splity, 42/93/67 cech, LR/RF, E2/E3, constraints, histories i 600 pełnych outputów | Doprecyzować zakres treści, naprawić rozbieżności raportowania, kontrola wpływu transpozycji, prosty search baseline, odsłuch | Najniższe ryzyko; CPU, ale E3 pełny run trwał ~4,76 h przy 4 workers. GA ma koszt per input | Najlepszy stosunek wykonanej pracy do braków. Łączy projekt reprezentacji, algorytm i kontrolowane porównanie. Mały efekt może być rzetelnym wynikiem |
| **B. Latent / autoencoder / Transformer transfer** | MIDI/parser/splity/metryki i infrastruktura E4; działający conv GAN nie jest latent AE | Tokenizacja/sekwencje, AE/VAE lub Transformer, podział content/style, trening i kontrola decode; wariant naprawy E4 wymaga mniej prac niż nowy VAE | Wysokie ryzyko dla nowej architektury; trening GPU i tuning. E4 ujawnił, że decode/selection są istotne niezależnie od architektury | Możliwe do zmierzenia obecnymi proxy, ale disentanglement nie wynika z rekonstrukcji. Sensowne rozszerzenie, słabszy wybór minimum |
| **C. Model pretrained + composer conditioning / steering** | Tylko dane, parser i evaluator; lokalne `.pt` są własnymi GAN-ami, nie pretrained symbolic backbone | Wybrać model/checkpoint, sprawdzić domenę/tokenizer, zgodność korpusu treningowego ze splitem, source-preserving edit, conditioning/adaptation, ewaluacja i reproducibility | Największa niepewność integracji. Zamrożona inferencja może ograniczyć koszt, ale skuteczny steering/fine-tuning i leakage nie są rozstrzygnięte | Generowanie conditioned nie wystarcza do transferu istniejącego content. Brak zastanych wyników uzasadniających ten kierunek jako rdzeń |

[Music Transformer](https://arxiv.org/abs/1809.04281) jest przykładem modelowania sekwencji muzycznych; sama zdolność generowania nie dowodzi rozdzielenia kompozytora i treści. [MuseMorphose](https://arxiv.org/abs/2105.04090) realizuje Transformer VAE i sterowanie rhythmic intensity/polyphony w długich utworach pop piano; to przydatny kontrast architektoniczny, ale nie gotowy dowód transferu Bacha/Beethovena/Chopina. Te źródła wyjaśniają różnice z zadaniem repo, nie stanowią rekomendacji konkretnego checkpointu.

**Rekomendacja:** A jako rdzeń, E4.5 jako eksperyment dodatkowy z NO-GO. Co najwyżej jeden jasno oddzielony wariant naprawy E4, jeżeli czas pozwala. Nie przepisywać parsera/splitów/GA ani nie dodawać jednocześnie VAE i pretrained steering do minimalnego zakresu.

## 10. Minimalny sensowny zakres dalszej pracy

Proponowany zakres: **interpretowalna modyfikacja akompaniamentu fortepianowych score MIDI między trzema znanymi repertuarami przy ochronie heurystycznej melodii i czasu**. Tak sformułowany cel odpowiada istniejącemu E3 i nie obiecuje dowolnego artysty, wykonania, obsady czy zachowania pełnej harmonii.

- **Baselines:** identity jako zerowa zmiana; istniejący E2 jako globalny baseline. Dodać prostą, z góry ustaloną transformację/search (np. ograniczoną samą transpozycję lub deterministyczny mały grid na tym samym genotypie), aby oddzielić wartość GA od wartości nowych operatorów. Wynik takiego baseline obecnie nie istnieje.
- **Jedna główna metoda:** E3 z zamrożonym kontraktem operatorów i constraints; bez nowego modelu. Ustalić przed runem, czy pitch content jest chroniony absolutnie, czy z dopuszczoną globalną transpozycją.
- **Style metric:** mean i per-direction Δp_target RF E1b train-only, pomocniczo standardized gains; nie optymalizować bezpośrednio tego samego RF, który jest końcowym ewaluatorem.
- **Content metrics:** istniejące onset-F1, length, contour, chroma oraz event-level porównanie chronionej melodii (pitch/onset/duration, z jawną polityką transpozycji). Raportować distribution/quantile/min i odsetek naruszeń, nie tylko medianę. Chroma raportować jako diagnostykę, dopóki zakres dopuszcza zmianę harmonii/tonacji.
- **Protokół:** zachować grouped split dzieł i train-only profile/fit. Przyszłe wybory dokonywać na development/pilocie; nie na ponownie oglądanym outer materiale. Zastane E2/E3 uczciwie nazwać porównaniem na znanym już benchmarku. Jeśli zakres obejmuje generalizację, dobrać osobny niewykorzystany external-source set i zamrozić go przed jakimkolwiek tuningiem.
- **Minimalne eksperymenty:** E1 full i sensitivity już istnieją; E2 vs E3 już istnieje. Najbardziej wartościowe uzupełnienie to kontrola E3 z transpose=0 / transpose-only oraz prosty non-GA baseline przy tych samych constraints. Mały, jawnie dobrany odsłuch identity/E2/E3 sprawdzi sens muzyczny niezależnie od RF. E4 NO-GO można opisać bez kolejnego treningu.

Przed nową ewaluacją zamrozić jedną definicję ważenia zadań/grup dla mean/CI/testu i wersję kontraktu round-trip. Nowe warianty zapisywać w nowych katalogach i raportach. Nie aktualizować wstecz istniejących wyników, aby udawały ten protokół.

## 11. Możliwe rozszerzenia

### MUST HAVE

Jawna definicja styl/treść i ograniczeń zakresu; kompletne pochodzenie artefaktów; oddzielenie bieżącego kodu od historycznych runów; wspólny estimand mean/CI; kontrola transpozycji; przynajmniej identity/E2/E3 i niezależne od optymalizacji style proxy; jawne raportowanie przypadków degradacji i wyników negatywnych.

### SHOULD HAVE

Pełne lub wybrane uprzednio ablacje grup E1b, niezależny prosty search baseline, niewielki odsłuch z ukrytymi nazwami metod, sensitivity po seedach transferu, lokalna melody/harmony ocena, mały external-source sanity set oraz prosty interfejs transferu E3 z sidecar provenance. Każdy wynik dodatkowy wymaga osobnego runu, nie dopisania planu jako wykonania.

### OPTIONAL / jeśli zostanie czas

Jeden E4.6 zgodny z handoffem w nowym katalogu; rendering WAV i wizualizacje partytur; lepszy melody extractor; token-based pretrained model albo reference-guided style encoder jako osobne rozszerzenie zakresu. Kolejny pełny neural model nie jest warunkiem sensownej pracy.

## 12. Problemy techniczne

| Priorytet | Problem potwierdzony / ryzyko | Dowód i znaczenie |
|---|---|---|
| Wysoki | Stara zainstalowana kopia pakietu | Bez PYTHONPATH import prowadzi do `.venv/lib/site-packages/musicians_style`; testy nie widzą E1–E4. Oceniać aktualny `src`, nie przypadkowy build |
| Wysoki | Niepełne provenance commita eksperymentów | E2 run manifest zapisuje przedpublikacyjny `9f7177e...` (publiczny odpowiednik po anonimizacji: `11ea480...`), który nie zawiera `e2/experiment.py`; cache E1b zapisuje przedpublikacyjny `08e4cc6...` (publiczny odpowiednik: `a574b46...`), który nie zawiera `composition_features.py`. HEAD sam nie obejmuje dirty/untracked kodu z chwili uruchomienia |
| Wysoki | Słabsze provenance E3/E4 | E3 run/task manifest nie zapisuje code_commit ani kompletnego environment/source snapshot. E4 prepared status miał commit, lecz kolejne statusy go zastępują; checkpointy nie mają commit. Geometry/schema/input snapshots nie identyfikują całego kodu |
| Wysoki | Repertuar/forma/epoka confounding | E1 form errors, prawie idealny Bach, mała i nierówna próba. To nie bezpośredni split leakage, ale ogranicza twierdzenie o samym stylu |
| Wysoki | Cel E3 słabo zgodny z proxy | `closure_analysis.objective_alignment`: r=0,04395, 48 regressions proxy przy poprawie celu. Nie traktować zbieżności celu jako jakości transferu |
| Wysoki | E3 zmiana closure po runie | `reporting._measure_outputs` koryguje historyczne flags/identity polyphony. GO opisuje revised feasible; nie przedstawiać go jako niezmienionej pierwotnej bramki |
| Wysoki | Mean/CI/test mają różne ważenie | E3 `reporting._group_values/_bootstrap_ci` vs `_stats`; E2 cluster bootstrap vs group sign test. Podawać jednostkę agregacji |
| Wysoki | CI E1 rozdziela to samo dzieło między repeatami | `e1/classification._cluster_bootstrap_ci`: klastry repeat/composer/group, strata repeat/composer. Potencjalne zawężenie niepewności; nie oceniono jego wielkości |
| Wysoki | E4 NO-GO i niediagnostyczny loss log | `training.masked_l1` liczy dodatnie i puste cells równoważnie; frame occupancy train ~3,5–4,0%. `gan_step` zwraca składowe, lecz harness zapisuje tylko mean G; przyczyna regresu nie jest w pełni rozstrzygnięta |
| Średni | Checkpoint selection E4 mała i skokowa | `_selection_pieces`: jedna mediana-length piece/composer; `_selection_key`: fractions co 1/6. Pełna validation ocenia 43 źródła i inny agregat gate |
| Średni | Zamknięty outer test nie oznacza fizycznego nieodczytu | `audit_e4` iteruje train/validation/test, `prepare_e4` i `train_e4._pieces` kodują także test. Nie widać użycia tych outputów do gradients/selection, ale test source statistics były dostępne. Handoffowy „fizycznie nieodczytany” warunek nie opisuje istniejącej implementacji |
| Średni | Blokada testu E4 nie gwarantuje jednokrotności ani powiązania bramki z checkpointem | `_evaluate_scope('test')` sprawdza `validation.summary.passed`, nie sprawdza w tym warunku zgodności checkpoint hash i nie odmawia nadpisania istniejącego outer_test. Nowa walidacja/zmiana checkpointu wymaga ostrożności; obecnie nie ma outer wyniku |
| Średni | E4 snapshots nie są wyłącznym wejściem dalszych etapów | Etapy używają `_inputs(config)` z bieżących wskazanych plików; prepare kopiuje snapshoty i przepisuje segments/status. Nie należy uruchamiać prepare w historycznym katalogu dla samego sprawdzenia |
| Średni | E3 `identity_outputs` to nazwa mocniejsza od obliczenia | Closure liczy objective==0. Nie dowodzi to zerowych genów ani niezmienionych nut |
| Średni | Kopiowanie meta tonacji przy transpozycji | `ga/transformation.py`, `e3/transformation.py` zachowują meta. Pitch może zmienić tonację, choć key_signature pozostaje wejściowe |
| Średni | Offset velocity E2 ma słaby/zerowy sygnał celu | Nieobecny w 42 cechach. Saturacja do velocity0 może sprawić, że zapisane note_on jest odczytane jako note_off; sama parsowalność nie gwarantuje zachowania nut |
| Średni | Metryki treści mogą premiować zmiany trywialne | Unikalne onset times bez pitches; contour ze znaków i bag-of-ngrams; globalna chroma bez kolejności. E2 contour0,990 przy length error0,370 jest konkretnym przykładem |
| Średni | Straty parsera i reprezentacji | Brak control_change/pedału/track identity, FIFO dla overlap same pitch; E4 scala wysokość across channels. Semantyka wewnętrzna nie pokrywa wszystkiego z oryginalnego MIDI |
| Średni | Konfiguracja akwizycji częściowo pozorna | Historyczny acquirer filtruje stałymi `MIN_DURATION_S/MAX_DURATION_S`, nie granicami YAML; skan lokalny jest płaski. Bieżące default wartości mogą działać przypadkiem zgodnie z YAML |
| Średni | Niespójne zależności | `pyproject` torch2.2.2+cu121 vs requirements torch2.2.2 vs Conda; brak jednego locka i pełnego HW/version inventory. Lokalnie Python3.10.20, numpy1.26.4, sklearn1.5.1, torch2.2.2+cu121, CUDA dostępne |
| Średni | Ścieżki zależne od katalogu roboczego i lokalnych zasobów | E3/E4 `_resolve` używa `Path.cwd()`, a nie katalogu YAML; domyślne ASAP/derived/run paths i `soundfonts/FluidR3_GM.sf2` są zapisane w konfiguracji/kodzie. Snapshoty zawierają absolutne `D:\\Studia\\...`; przeniesienie repo wymaga jawnego wskazania zasobów |
| Średni | Determinizm GPU/resume niepełny | Seedy są obecne w E1–E4 i lokalny RNG GA; E4 nie zapisuje/odtwarza pełnego RNG state i nie wymusza deterministic algorithms. Resume nie musi być bitowo równy ciągłemu treningowi |
| Niski | Martwe pomocnicze funkcje i duplikaty | E3 `experiment._group_means/_bootstrap_group_ci/_sign_permutation_p/_holm` nie mają znalezionych odwołań po przeniesieniu raportu. E2 ma lokalne content helpers obok shared content; potencjał rozjazdu definicji |
| Niski | Dokumentacja nie odzwierciedla API/stanu | README tabela nadal mówi o 42 cechach i brakach kompletnych eksperymentów; `cli.py` rejestruje tylko acquire/train/infer/evaluate. Polecenia E1–E4 są modułowe, nie z tej tabeli CLI |
| Niski | Lokalne kopie wyników mylone z powtórzeniami | `e2_asap_backup/results.json` i `e3_asap_backup/results.json` są byte-identyczne z głównymi; nie zwiększają N/seeds/repeatów |
| Niski | Literatura i stare notatki są szerokie | Lokalne audio-synthesis PDF i notatki o MFCC/Essentia nie dokumentują obecnego symbolic transfer. Nie przedstawiać rozważanych bibliotek/modeli jako użytych |

**Kontrola testów wykonana w audycie:**

1. `.venv\Scripts\python.exe -m pytest -q tests/unit` — **20 błędów collection**: import starego pakietu, brak E1–E4/shared content/nowych funkcji distance. Nie są to wyniki assertów bieżącego kodu.
2. Po ustawieniu procesowego `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/unit` — **458 passed, 14 warnings, 48,40 s**. Ostrzeżenia dotyczą bibliotek/deprecations; bez fail testów. Nie uruchamiano całego property/integration suite ani długiego treningu, więc nie podaje się tej liczby jako wyniku wszystkich testów repo.
3. Sprawdzenie 25 outer foldów i ich inner partitions: brak przecięcia sample/group/SHA. Sprawdzenie SHA 300 output MIDI E2 i 300 E3: **0 niezgodności** względem zapisanych wyników. E2/E3 backup results są identyczne z głównymi. Potwierdzono obecność 86 validation MIDI E4 i brak outer_test.json.
4. Checkpointy E4 odczytano na CPU z `weights_only=True` bez inferencji/treningu. Historyczny epoch021 nie dał się odczytać tym trybem z powodu obiektu NumPy; nie przełączano na pełne ładowanie pickle. Wnioski o jego historycznych lossach pochodzą z JSONL, nie z odtworzonego modelu.

## 13. Rekomendowane następne kroki

Maksymalnie pięć działań, w kolejności wartości dla pracy względem kosztu:

1. **Zamrozić zakres i udokumentować pochodzenie.** Przyjąć A/E3 jako rdzeń, E4 NO-GO jako dodatkowy wynik. Zarchiwizować lokalne dane/runy wraz z istniejącymi hashami, pełnym source/environment snapshotem i jawnym opisem unknown provenance; poprawić mapę aktualnych poleceń w osobnej późniejszej zmianie.
2. **Przygotować rozdział wynikowy z już zapisanych danych.** E1a/E1b/sensitivity/confusion/importance, E1-open i sparowane E2/E3; pokazać negatywne kierunki i tail content. Opisać post-hoc E3 i ważenie mean/CI. E4 oddzielić jako validation failure.
3. **Rozstrzygnąć definicję treści i wpływ transpozycji.** Na development części przygotować event-level melody check oraz transpose-only/transpose=0 kontrolę; ustalić, czy utrata chroma jest dopuszczona. To sprawdzi, ile zysku E3 wynika z prostego przesuwania tonacji.
4. **Wykonać jeden minimalny, zamrożony pakiet porównawczy.** Identity, istniejący E2, E3 i tani non-GA search baseline z jednym wspólnym protokołem; dołączyć mały odsłuch i, jeśli deklarowana jest generalizacja, osobny external-source set. Nowe katalogi, bez nadpisywania E2/E3 i bez strojenia na E4 outer test.
5. **Dopiero przy zapasie czasu zdecydować o jednym E4.6.** Handoff wskazuje diagnostykę D_cls/loss, większe selection, sparse-aware reconstruction i joint decode calibration. Jeden nowy run, bez obniżania gate po wynikach i bez uzależniania ukończenia pracy od GO.

## Appendix A — Evidence map

| Twierdzenie | Plik / funkcja / wynik | Dowód | Pewność |
|---|---|---|---|
| Commit analizowanego kodu | `git rev-parse HEAD`, `git log -1` | `f58f084963cfe8281a22b7c0ec88aebaac667d59`; clean przed audytem | Wysoka |
| Local artifacts poza commitem | `.gitignore`, lokalne drzewa danych/runów | Ignorowane datasets/experiments/PDF/checkpoints są obecne | Wysoka |
| Obecny pipeline różni się od starego audytu | `e1/`–`e4/`, `docs/results/` | Ukończone E1–E3 i validation E4; audit z 1 IX mówi o wcześniejszych brakach | Wysoka |
| 150 score / 87 grup | `datasets/derived/e1_asap/quality_report.json` | 59/57/34 samples, 30/28/29 groups | Wysoka |
| E1 nie używa performance MIDI | `e1/asap._read_candidates`, manifest.selection | `metadata.midi_score`, `score MIDI only` | Wysoka |
| Brak bezpośredniego group/SHA leakage | `e1/splits._assert_disjoint`, zapisane splity | Kontrola audytu: 0 przecięć w outer/inner | Wysoka dla identyfikatorów |
| Grupowanie nie obejmuje wszystkich możliwych zależności | `e1/asap.group_id_for` | Regex BWV/sonata, fallback do slug tytułu | Wysoka kod; średnia skala ryzyka |
| E1b RF BA 0,86424 | Ukończony `e1b_results.json:summaries` | BA/F1/CI, OOF validation passed | Wysoka |
| Nie wykonano full ablacji w ukończonym E1b | `e1b_results.protocol.feature_variants`, run parameters | Lista zawiera tylko composition_full | Wysoka dla odnalezionych runów |
| Open-set słaby | `docs/results/E1.md`, finalny open JSON | RF unknown recall0,108, LR0 | Wysoka |
| E2 działa pełnymi MIDI i globalnym GA | `ga/transformation`, `e2/experiment` | Globalne pitch/onset/duration/velocity; profile train-only | Wysoka |
| E2 degradacja czasu | `e2_asap/analysis.json`, `docs/results/E2.md` | Mean length error0,370; onset0,365 | Wysoka |
| E3 ma jawne content constraints | `e3/objective.validate_constraints`, `structure.analyse_structure` | Protected melody, length, note count/meta/polyphony/roundtrip | Wysoka dla bieżącego kodu |
| E3 cel to 67 cech | `e3/profile.style_vector` | 16+6+12+25+8; trzy grupy | Wysoka |
| E3 style gain mały | `closure_analysis.delta_p_target` | Mean0,037705; 48 negative proxy | Wysoka |
| Cel E3 słabo zgodny z RF | `closure_analysis.objective_alignment` | Pearson0,043951, 48 objective-improved/proxy-worsened | Wysoka |
| E3 content nie jest idealnie bezstratny | `closure_analysis.content` | Chroma0,719536, min contour0,185637, nonzero max length error | Wysoka |
| GO E3 po revised closure | `e3/reporting._measure_outputs`, `docs/results/E3.md` | Reklasyfikacja flags identity/semantic/polyphony | Wysoka |
| Raportowy identity count nie sprawdza genome | `e3/reporting.build_closure_report` | `identity_outputs = sum(objective == 0)` | Wysoka |
| E4 używa pełnych utworów przez segmenty | `e4/segmentation.encode_piece/stitch_segments` | 4 bars, onset/frame, carry, full source end | Wysoka |
| E4 one-hot, bez latent/style encoder | `e4/model.ConditionalGenerator.forward` | concat z composer one-hot; brak reference encoder | Wysoka |
| E4 best epoch1, last6 | `e4_asap_v2/best.pt`, `last.pt`, progress | Bezpieczny odczyt metadata i epok | Wysoka |
| E4 validation NO-GO | `validation.json.summary`, `status.json` | Style groups/melody/polyphony false | Wysoka |
| Brak wyniku outer test | Drzewo `e4_asap_v2/`, `report.md` | Brak outer_test.json/output dir; test blocked | Wysoka dla lokalnych artefaktów |
| Source test był czytany przed GO | `audit_e4`, `prepare_e4`, `train_e4._pieces` | Wszystkie partitions kodowane/audytowane | Wysoka kod; brak dowodu leakage gradients |
| Stary GAN przycina do jednego okna | `midi/pianoroll.from_internal`, `training/dataset`, `inference._run_gan` | Stałe64; pojedyncze from_internal/to_internal | Wysoka |
| Stare identity validation na train | `Trainer._train_epochs`, `_validate` | Przekazanie tego samego loadera | Wysoka |
| Old GAN21 epok i2 iteracje | `custom...214352/logs/train.jsonl`, checkpoints | Ostatni checkpoint021, końcowe iter epoki22 | Wysoka |
| `per_artist` nie pełny CycleGAN | `Trainer._train_step_per_artist` | Jeden G/D i G(G(x)); brak runów | Wysoka |
| Provenance HEAD nie wystarcza | Cache E1b/E2 manifest i `git cat-file -e` | Zapisane commity nie zawierają modułów eksperymentu | Wysoka; przyczyna dirty-state nieustalona |
| Backup nie jest powtórzeniem | E2/E3 `results.json` vs backup | Byte-identical | Wysoka |
| Tests aktualnego source działają | `tests/unit`, komenda z PYTHONPATH | 458 passed /48,40 s; bez source path20 import errors | Wysoka |
| Brak dowodu jakości percepcyjnej | `evaluation/subjective.py`, inwentarz artefaktów | Jest kod, brak ocen słuchaczy | Wysoka dla odnalezionych plików |
| Kierunek A najkorzystniejszy | Ukończone E1–E3 vs stan B/C | Znacznie większy zakres gotowych komponentów i wyników | Interpretacja, średnia |

## Appendix B — Experiment inventory

Każdy lokalny katalog runu wymieniono poniżej. Podkroki E1–E4 opisane w §3 nie zawsze mają własny katalog. Puste podkatalogi `reports/outputs/checkpoints` nie są wynikiem.

| Run / etap | Status potwierdzony | Wynik / komentarz |
|---|---|---|
| `custom_conditional_2026-07-29_211645` | Akwizycja manifestów | 131 Bach,136 Chopin,87 Mozart; bez train log |
| `custom_conditional_2026-07-29_212218` | Tylko config/hash | Brak wyniku; cel uruchomienia nieustalony |
| `custom_conditional_2026-07-29_214352` | Trening częściowy | 21 checkpointów,21 complete epochs +2 iter epoch22 |
| `custom_conditional_2026-07-29_223552` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_223852` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_223904` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_224131` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_224146` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_224201` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_224415` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_224431` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_224447` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_225909` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_225924` | Tylko config/hash | Brak wyniku |
| `custom_conditional_2026-07-29_225939` | Tylko config/hash | Brak wyniku |
| `e1_asap_2026-09-01_215646` | E1a wyniki zapisane | 42/38 cech ×3 modele; starszy schema |
| `e1_asap_e1b_2026-09-01_225113_456415` | failed | KeyboardInterrupt; bez finalnego results |
| `e1_asap_e1b_2026-09-02_005705_449734` | completed, closure GO | Full93 features,3 modele,2 analyses; brak12 ablacji |
| `e1_asap_open_2026-09-01_231036_077090` | completed | Open-set +rotacje; wcześniejszy run |
| `e1_asap_open_2026-09-01_231352_526004` | completed | Open-set +rotacje; dołączony do E1 closure |
| `e2_asap` | full_completed | Pilot18 +full300; JSON/CSV/plots/output MIDI |
| `e2_asap_backup` | Kopia ukończonego E2 | Results identyczne; brak dodatkowego repeat |
| `e3_asap` | completed, closure GO_WITH_LIMITATIONS | Pilot6 +full300; closure post-hoc |
| `e3_asap_backup` | Kopia ukończonego E3 | Results identyczne; bez closure_analysis |
| `e3_smoke` | pilot_completed | Sześć krótkich/pilotowych zadań; nie pełna macierz |
| `e4_asap` | e4.3.0 smoke_passed | Historyczny best epoch2, loss-selected; brak validation/test |
| `e4_asap_v2` | e4.5.0 validation_no_go | Smoke6,train6 epochs,best1,validation86; outer brak |
| E1.0/E1.1 w `datasets/derived/e1_asap` | Artefakty gotowe | Manifest150/87 i splity5×5×3 |
| E1.4 w finalnym E1b `report/` | Ukończone zamknięcie | CSV,PNG,closure manifest; nie niezależny trening |
| Dziewięć prób `datasets/outputs` | Output MIDI istnieją | Bez przypisania run/checkpoint/seed |
| `per_artist`, combined | Kod/testy, brak odnalezionych runów | Nie wykazane eksperymentalnie |
| E4.6 / E4_asap_v3 | Tylko handoff/plan | Nie ma implementacji ani wyników tego wariantu |
| AE/VAE/Transformer/pretrained steering | Brak implementacji/runów | Rozważane kierunki, nie zastane eksperymenty |
| Subjective listening study | Infrastruktura, brak wyników | Nie znaleziono ocen słuchaczy |

## Appendix C — Questions / unknowns

- Jaki dokładnie source tree i dirty diff uruchomił E1 cache, E2, E3 oraz E4? Część zapisanych HEAD nie zawiera odpowiednich modułów; same daty i schema nie rozwiązują tego.
- Dlaczego pierwotny GAN zatrzymał się po dwóch iteracjach epoki22 i czym były puste `custom_conditional` runy?
- Z jakich checkpointów/inputów/poleceń pochodzą dziewięć ręcznych output MIDI i czy wszystkie przechodziły ocenę słuchową? Repo nie zapisuje wiązania.
- Czy lokalny Kaggle corpus jest dokładnie publiczną wersją datasetu, z jaką licencją i źródłami poszczególnych plików? Nazwa folderu i source=local nie identyfikują wydania.
- Czy między różnymi group_id są duplikaty muzyczne, części wspólnego cyklu albo aranżacje niewykrywalne SHA? Exact hash kontroluje tylko identyczne pliki.
- Czy 93 cechy rozróżniają kompozytora po kontrolowaniu formy/epoki/źródła notacji? Brak zrównoważonego korpusu i ukończonych ablacji nie pozwala tego ustalić.
- Jakie są pełne event-level melody/harmony/phrase zachowanie i jakość percepcyjna E3? Istniejące proxy nie odpowiadają jednoznacznie.
- Ile raportowych zero-style-gain przypadków E3 jest rzeczywistą identity, a ile równoważnością cech/celu? `identity_outputs` nie rozstrzyga.
- Na ile revised semantic validity closure E3 odpowiada wszystkim historycznym pre-serialization kandydatom? Nie ma zapisanych kompletnych obiektów zdarzeń każdego kandydata.
- Czy głowa D_cls E4 poprawnie odróżnia kompozytorów na prawdziwych validation oknach? Brak zapisanej accuracy/confusion i epoch loss decomposition.
- Dlaczego identity cell-F1 E4 na małym selection jest1,0, a decoded transfer pogarsza kontur/rytm? Nie ma event-level identity benchmarku całej validation rozdzielającego model i decode.
- Jakie są dokładne GPU/CPU/driver, koszty przygotowania/cache i end-to-end czasy poszczególnych etapów E4? Timestampy epok nie są pełnym pomiarem treningu; prepared-to-epoch1 zawiera nieustalone operacje.
- Czy E3/E4 generalizują na źródło spoza Bacha/Beethovena/Chopina, inne metrum/polifonię albo melody-only MIDI użytkownika? Nie ma takich wyników. Techniczna możliwość podania tensora nie jest dowodem generalizacji.
- Czy praca ma chronić absolutne wysokości/tonację i harmonię, czy wyłącznie relatywną melodię i timing przy zmianie akompaniamentu? To decyzja zakresu dalszej pracy; obecne E3 dopuszcza transpozycję.
- Czy istnieją poza workspace niezapisane eksperymenty, ablacje, oceny słuchaczy lub wcześniejsze wersje wyników? Audyt nie ma dowodu ich istnienia i nie wlicza ich do stanu repo.
