# E4 — handoff do poprawy implementacji po pierwszym validation NO-GO

## Cel dokumentu

Ten dokument przekazuje stan E4 po pierwszym pełnym treningu i walidacji oraz
opisuje minimalny, metodologicznie uczciwy zakres ewentualnego wariantu E4.6.
Rozdziela poprawę eksperymentu od szerszego celu użytkowego: nauczenia stylu na
zbiorze utworów kompozytora i nałożenia go na wejściowe MIDI spoza zbioru
trzech domen.

## Stan zastany

- Punkt odniesienia kodu: commit `8ea3a73` (`first E4 training results`).
- Implementacja bazowa E4.5: commit `2bba874`.
- Run: `experiments/e4_asap_v2`.
- Schemat: `e4.5.0`.
- Audit: PASS; 0 błędów parsera, 8 jawnych wykluczeń.
- Smoke: PASS; 6/6 pełnych plików przeszło zapis i parse.
- Trening zatrzymał się po 6 epokach (`patience=5`).
- `best.pt` pochodzi z epoki 1.
- Validation: NO-GO.
- `outer_test.json` nie istnieje. Outer test pozostaje zamknięty i nie wolno go
  używać podczas prac nad E4.6.
- Worktree był czysty przed utworzeniem tego dokumentu.

Artefakty źródłowe:

- `experiments/e4_asap_v2/progress.jsonl` — przebieg epok;
- `experiments/e4_asap_v2/best.pt` i `last.pt` — epoka 1 i 6;
- `experiments/e4_asap_v2/validation.json` — 86 wyników kierunkowych;
- `experiments/e4_asap_v2/validation_midi/` — zapisane wyniki;
- `docs/results/E4.md` — raport NO-GO;
- `docs/E4_PLAN.md` — zamrożony protokół E4.5.

Nie nadpisywać `e4_asap_v2`. Każda zmiana selection, lossu, zbioru
walidacyjnego lub dekodera wymaga nowej wersji schematu, nowego katalogu runu i
nowego raportu, np. `e4.6.0` oraz `experiments/e4_asap_v3`.

## Co pokazał trening

`selection_key` w E4.5 ma znaczenie:

```text
[
  structural_valid_rate,
  fraction_passing_melody_and_polyphony,
  negative_fallback_rate,
  mean_delta_p_target
]
```

Przebieg:

| Epoka | selection_key | generator_loss |
|---:|---|---:|
| 1 | `[1.0, 0.667, 0.0, 0.0050]` | 2.645 |
| 2 | `[1.0, 0.667, 0.0, 0.0039]` | 1.387 |
| 3 | `[1.0, 0.167, -0.006, -0.0118]` | 1.382 |
| 4 | `[1.0, 0.0, -0.006, -0.0012]` | 1.319 |
| 5 | `[1.0, 0.0, -0.006, -0.0140]` | 1.209 |
| 6 | `[1.0, 0.0, 0.0, -0.0155]` | 1.029 |

Early stopping zadziałał zgodnie z kodem. Spadek sumarycznego lossu nie oznaczał
poprawy transferu ani treści. Po pierwszej epoce jakość dekodowanego materiału
spadała.

Pełna walidacja `best.pt`:

- średnie `delta_p_target = 0.00537`;
- 5/6 kierunków ma dodatnią średnią;
- mediana podobieństwa melodii: `0.92325`, poniżej `0.95`;
- onset-F1: `0.85430`;
- chroma cosine: `0.99957`;
- pitch group gain: `+0.00859`;
- rhythm group gain: `-3.26992`;
- texture group gain: `-0.09144`;
- 11/86 wyników przekracza limit polifonii, maksymalnie o 4 głosy;
- 19 916 osieroconych początków frame;
- fallback identity: 0/2752 segmentów.

Największy regres pojedynczej cechy dotyczy `syncopated_note_ratio`. W profilach
train jego średnia wynosi około `0.011–0.022`, natomiast w wynikach kierowanych
do poszczególnych domen około `0.17–0.24`, miejscami do `0.77`. Dekoder tworzy
zbyt wiele długich nut rozpoczynających się poza pulsem. Duży ujemny rhythm gain
nie jest więc wyłącznie artefaktem standaryzacji ewaluatora.

## Ograniczenia obecnego wyboru checkpointu

1. Selection używa tylko jednego utworu na kompozytora: trzech źródeł i sześciu
   transferów. Druga składowa klucza zmienia się skokowo co `1/6`.
2. Lokalny warunek selection wymaga `melody >= 0.95` dla każdego wyniku i łączy
   go z polifonią. Końcowa bramka używa mediany melodii oraz osobnego warunku
   polifonii. Selection nie odpowiada więc dokładnie końcowej decyzji.
3. Log zawiera tylko całkowity generator loss. Nie zapisuje lossu
   dyskryminatora, adversarial, classification, cycle ani identity.
4. Nie mierzymy accuracy głowy `D_cls` na prawdziwych fragmentach validation.
   Jeżeli czterotaktowe pianorolle nie zawierają dla niej stabilnego sygnału
   domeny, generator nie otrzymuje użytecznego kierunku stylu.
5. Rekonstrukcja L1 jest uśredniana po bardzo rzadkim tensorze. Aktywne komórki
   stanowią około 3.5–4% reprezentacji, więc dobre przewidywanie zer może
   obniżać loss mimo pogarszania onsetów i durations.
6. Progi onset/frame są kalibrowane niezależnie przez cell-level F1 na identity.
   Nie optymalizują wspólnie jakości zdekodowanych zdarzeń, długości nut ani
   liczby osieroconych frame.

## Zalecany wariant E4.6

### P0 — reprodukowalność i izolacja

- Zwiększyć `SCHEMA_VERSION` do `e4.6.0`.
- Użyć nowego `run_dir`, domyślnie `experiments/e4_asap_v3`.
- Nie ładować checkpointów E4.5 po zmianie lossu lub geometrii selection.
- Zachować ten sam `repeat=0`, outer fold i inner fold.
- Nie czytać outer testu przed nowym validation GO.
- Zapisać w snapshotcie także wersję polityki selection i definicję lossu.

### P1 — diagnostyka treningu

Rozszerzyć `progress.jsonl` per epoka o średnie:

- `discriminator_loss`;
- `generator_loss`;
- `adversarial_loss`;
- `classification_loss`;
- `cycle_loss`;
- `identity_loss`;
- threshold i identity F1 osobno dla onset/frame;
- occupancy i onset density wejścia, identity oraz transferu;
- orphan-frame rate, a nie tylko bezwzględną liczbę;
- pełne składowe selection summary.

Dodać ewaluację `D_cls` na prawdziwych fragmentach train/validation: balanced
accuracy i macierz pomyłek per kompozytor. Nie zwiększać `lambda_cls` w ciemno.
Jeśli `D_cls` nie rozpoznaje domen ponad rozsądny baseline, najpierw poprawić lub
wstępnie wytrenować klasyfikator domeny.

### P2 — stabilniejszy selection i early stopping

- Wybrać deterministycznie 3 utwory na kompozytora, np. najbliższe dolnemu
  kwartylowi, medianie i górnemu kwartylowi liczby segmentów. Daje to 9 źródeł i
  18 transferów bez kosztu pełnej validation po każdej epoce.
- Zastąpić `fraction_passing_melody_and_polyphony` ciągłym raportem naruszeń.
- Oddzielić twardą poprawność techniczną od budżetu treści i wyniku stylu.
- Dopasować warunki selection do agregacji z finalnej bramki: mediana melodii,
  globalny fallback rate, odsetek i wielkość przekroczeń polifonii.
- Gdy ograniczenia treści są spełnione, wybierać checkpoint według liczby
  dodatnich kierunków, zgodności grup stylu i średniego `delta_p_target`.
- Gdy żaden checkpoint nie jest feasible, zachować najmniejsze znormalizowane
  naruszenie jako wynik diagnostyczny, ale nie oznaczać go jako GO.
- Ustawić `min_epochs=10`, `patience=10`, `max_epochs=30`. Patience zaczynać
  liczyć dopiero po `min_epochs`.

Nie obniżać finalnego progu melodii po obejrzeniu E4.5. Samo obniżenie go do
`0.90` nie naprawiłoby regresu rytmu, tekstury ani polifonii.

### P3 — loss odporny na rzadkość pianorolla

Zastąpić jednakowo ważony reconstruction L1 wersją ważoną kanałowo i po klasie:

- osobna waga onset i frame;
- większa kara dla dodatnich komórek niż dla zer;
- wagi wyznaczone wyłącznie z train i ograniczone maksymalną wartością, aby
  rzadkie onsety nie destabilizowały gradientów;
- ten sam kontrakt dla identity i cycle.

Pierwszy wariant powinien być minimalny: weighted L1 albo BCE-with-logits z
ograniczonym `pos_weight`, bez jednoczesnej wymiany całej architektury. Dodać
małą karę za zmianę onset density i frame occupancy tylko wtedy, gdy logi
potwierdzą dalszą eksplozję długości nut.

Test jednostkowy musi wykazać, że błąd na aktywnym onset ma większy wpływ na
loss niż taka sama liczba błędów na pustych komórkach.

### P4 — dekodowanie

- Kalibrować progi na większym, stałym selection, a nie trzech utworach.
- Oceniać każdą parę progów po pełnym identity decode: onset-F1, duration/frame
  F1, melody similarity, orphan-frame rate i polifonia.
- Nie dobierać progu wyłącznie osobnym cell-level F1 kanałów.
- Zachować istniejący carry sustainu pomiędzy oknami i kontrakt
  `onset => frame`.
- Nie wprowadzać arbitralnego maksymalnego czasu nuty. Najpierw ograniczyć
  przyczynę długich frame przez loss i wspólną kalibrację.

### P5 — jeden kontrolowany przebieg

Po implementacji uruchomić jeden nowy run z seedem `1729`. Nie wykonywać grid
searchu. Jeżeli E4.6 nadal nie przejdzie validation, zapisać NO-GO i zakończyć
wątek GAN zamiast wielokrotnie dostrajać go do inner validation.

## Pierwotny cel produktu a obecne E4

### Obecne E4 nie jest transferem 1:1

Trening jest **unpaired**. Nie istnieją pary typu „ta sama melodia w stylu Bacha
i Chopina”, a generator nie wyszukuje jednego utworu docelowego i nie kopiuje z
niego stylu. Uczy się na wielu niezależnych czterotaktowych fragmentach trzech
domen. Jeden generator otrzymuje pianoroll wejścia oraz etykietę celu:

```text
G(fragment_wejściowy, target_composer_id) -> fragment_w_stylu_celu
```

Etykieta `target_composer_id` jest stałym one-hotem. „Styl Bacha” jest więc
niejawnym rozkładem wyuczonym ze wszystkich bachowskich fragmentów train, nie
wektorem policzonym jawnie z jednego utworu ani parą 1:1.

W ewaluacji każdy held-out utwór jednego z trzech kompozytorów jest przekształcany
do dwóch pozostałych domen. Stąd sześć kierunków. Jest to **closed-set,
many-to-many domain transfer**:

```text
Bach       -> Beethoven, Chopin
Beethoven  -> Bach, Chopin
Chopin     -> Bach, Beethoven
```

Zamknięty charakter dotyczy źródła i celu: wejścia train/validation/test także
pochodzą tylko od Bacha, Beethovena i Chopina. Nie testowaliśmy utworów innego
kompozytora ani MIDI użytkownika.

### Czy obecny model może technicznie przyjąć dowolne MIDI?

Częściowo tak. `_infer_piece()` podczas inferencji potrzebuje zakodowanego
utworu i indeksu kompozytora docelowego; nie używa etykiety kompozytora
źródłowego. Parser, `encode_piece()` i stitcher potrafią podzielić pełny utwór
na okna. Rdzeń może więc wykonać:

```text
nieznane MIDI -> target Bach/Beethoven/Chopin
```

Nie oznacza to jednak potwierdzonej generalizacji:

- nie ma publicznego polecenia ani API `transfer` dla zewnętrznego pliku;
- wejście musi mieścić się w kontrakcie score MIDI, metrum, długości taktów i
  zakresu wysokości albo otrzymać jawne ostrzeżenia/fallback;
- model nigdy nie był oceniony na źródłach spoza trzech domen;
- silny distribution shift może spowodować znacznie gorszy wynik;
- obecna ewaluacja E1b zna tylko trzy klasy docelowe i mierzy podobieństwo do
  nich, nie ogólną jakość muzyczną.

### Minimalne rozszerzenie zgodne z pierwotnym zamysłem

Jeśli cel brzmi: „style są z góry znane i uczone na korpusach trzech artystów,
ale wejściem może być obce MIDI”, nie trzeba od razu wymieniać architektury.
Po uzyskaniu stabilnego modelu należy dodać:

```text
python -m musicians_style.e4 transfer --config configs/e4_asap_v3.yaml --checkpoint experiments/e4_asap_v3/best.pt --input path/to/content.mid --target Bach --output path/to/result.mid
```

Tryb powinien:

1. parsować i audytować wejście;
2. raportować nuty poza zakresem oraz nietypowe takty;
3. kodować wszystkie okna;
4. używać zamrożonych progów checkpointu;
5. zapisywać MIDI i sidecar JSON z diagnostyką;
6. sprawdzać ponowny parse, długość, metadane, puste wyniki i fallback;
7. nie wymagać `sample_id`, wpisu w manifeście ani etykiety źródła.

Dodać osobny **external-source sanity set**, niebiorący udziału w treningu ani
wyborze checkpointu. Może zawierać kilku innych kompozytorów i syntetyczne MIDI
o znanej melodii. Raportować te same metryki treści oraz `p_target` E1b, ale nie
mieszać tego wyniku z zamrożonym outer testem E4.

Jeśli natomiast celem jest dodawanie nowego artysty wyłącznie przez podanie
zbioru jego utworów, bez ponownego treningu modelu, obecny one-hot nie wystarcza.
Potrzebny byłby osobny style encoder/agregator:

```text
z_style = S(zbiór utworów artysty)
content = E(melodia wejściowa)
output = G(content, z_style)
```

To jest większa zmiana badawcza: reference-guided lub few-shot style transfer,
a nie poprawka E4. Nie powinna być dokładana do obecnej pracy pod nazwą E4.6
bez zmiany pytania badawczego i zakresu ewaluacji.

## Pliki przewidziane do zmiany

- `src/musicians_style/e4/training.py` — ważony reconstruction loss i metryki;
- `src/musicians_style/e4/experiment.py` — selection, patience, diagnostyka i
  wersjonowanie;
- `src/musicians_style/e4/evaluation.py` — ciągłe naruszenia i D_cls metrics;
- `src/musicians_style/e4/__main__.py` — opcjonalny tryb `transfer`;
- `configs/e4_asap_v3.yaml` — nowy zamrożony protokół;
- `tests/unit/e4/` — regresje lossu, selection, dekodera i external input;
- `docs/E4_PLAN.md` — addendum E4.6, bez przepisywania historycznego E4.5;
- `docs/results/E4_v3.md` — nowy raport, jeśli run zostanie wykonany.

## Kryteria odbioru implementacji

- Stare testy E4.5 nadal przechodzą.
- Selection jest deterministyczny i nie używa outer testu.
- Test pokazuje wyższy koszt błędnego onsetu niż pustej komórki.
- Progress zapisuje wszystkie składowe lossu i selection.
- `min_epochs` i `patience` mają test graniczny.
- Kalibracja progów używa wyłącznie validation selection.
- Zmieniony protokół nie ładuje checkpointu E4.5.
- Smoke zapisuje 6/6 poprawnych pełnych MIDI.
- Opcjonalny `transfer` działa bez `sample_id` i klasy źródłowej oraz zwraca
  sidecar diagnostyczny.
- Outer test pozostaje fizycznie nieodczytany do validation GO.

## Czego nie robić

- Nie obniżać progu `0.95` tylko dlatego, że E4.5 osiągnęło `0.923`.
- Nie uznawać spadku generator loss za dowód poprawy muzycznej.
- Nie zwiększać tylko `patience` i nie nadpisywać istniejącego runu.
- Nie stroić kolejnych wariantów na outer test.
- Nie nazywać modelu wiernym StarGAN-em ani transferem stylu dowolnego artysty.
- Nie twierdzić, że obecna walidacja dowodzi działania na dowolnym MIDI.

## Polecenia kontrolne

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q tests/unit/e4
.\.venv\Scripts\python.exe -m pytest -q tests/unit
```

Przed nowym runem sprawdzić `git status`, wersję schematu, nowy `run_dir` i brak
`outer_test.json` w nowym katalogu.
