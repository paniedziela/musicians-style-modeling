# Plan eksperymentu E1 — rozróżnianie stylu kompozytorskiego w ASAP

## Decyzja

Do rozpoczęcia E1 nie jest potrzebny kolejny zbiór. Lokalny ASAP zawiera dość
utworów Bacha, Beethovena i Chopina, aby przeprowadzić pierwszy kontrolowany
eksperyment klasyfikacyjny. W E1 używane będą wyłącznie **score MIDI** — nie
performance MIDI, audio ani powiązane pliki MAESTRO.

Implementację należy wykonać w odseparowanych etapach E1.0–E1.4. Pierwszy etap
nie trenuje jeszcze klasyfikatora: buduje manifest, sprawdza pliki i zamraża
protokół. Dzięki temu błędu w przygotowaniu danych nie pomylimy z wynikiem
modelu.

## Stan lokalnego zbioru

Sprawdzony katalog: `datasets/asap-dataset-1.2/`.

| Element | Wynik audytu struktury i metadanych |
|---|---:|
| Wszystkie fizyczne `midi_score.mid` | 235 |
| Wszystkie fizyczne `xml_score.musicxml` | 235 |
| Bach — unikalne `(composer, title)` | 59 |
| Beethoven — unikalne `(composer, title)` | 57 |
| Chopin — unikalne `(composer, title)` | 34 |
| Razem próbek przed kontrolą jakości | 150 |
| Ścieżki score MIDI wskazane dla tych tytułów | 158 |
| Brakujące score MIDI wskazane w metadanych | 0 |
| Pliki poprawnie odczytane przez obecny `MidiParser` | 150/150 |
| Pliki z nieskończonymi wartościami obecnych cech | 0 |
| Grupy identycznych SHA wśród próbek kanonicznych | 0 |

Różnica 158 ścieżek względem 150 tytułów nie oznacza ośmiu dodatkowych dzieł.
ASAP zawiera warianty tego samego materiału dopasowane do wykonań z inną obsługą
repetycji, np. `17-1` i `17-1_no_repeat`. Sam README zbioru zaleca uznanie pary
`(composer, title)` za identyfikator unikalnego utworu.

Wstępne grupowanie powiązanych części daje:

| Kompozytor | Próbki | Grupy do podziału |
|---|---:|---:|
| Bach | 59 | 30 |
| Beethoven | 57 | 28 |
| Chopin | 34 | 29 |

Grupy są mniej liczne niż próbki, ponieważ nie rozdzielamy między foldy m.in.
preludium i fugi o tym samym numerze BWV ani części tej samej sonaty.

### Identyfikacja wersji danych

Początkowy manifest powinien zapisać co najmniej następujący fingerprint:

```text
metadata.csv SHA-256: b327d92e498e281b381161ee387dacfd4b1c360c9c240d9107171f78e571b2a5
README.md   SHA-256: 4c0cf082cbed77fa4e1e471db0c5e4352aca8c3a1a6b025f8225c12228a638af
LICENSE.md  SHA-256: af662157a4f2e47b5fefde500141853e1f7d160ba0dda95546229b5e60a93f6d
```

Katalog nazywa się `1.2`, ale dołączony README w sekcji „Versions” nadal określa
wersję bieżącą jako `v1.1`. Przed cytowaniem danych w pracy trzeba zachować URL i
nazwę pobranego archiwum/tagu. Fingerprint pozwoli odtworzyć eksperyment nawet
przy tej niespójności opisu. Licencja danych to CC BY-NC-SA 4.0; surowych danych
nie należy commitować do repozytorium.

## Pytanie i hipoteza E1

**Pytanie:** czy cechy symboliczne score MIDI pozwalają odróżniać utwory Bacha,
Beethovena i Chopina na dziełach niewidzianych podczas uczenia?

**Hipoteza robocza:** klasyfikator oparty na cechach symbolicznych osiąga
balanced accuracy istotnie wyższe od poziomu losowego `1/3` w podziale po
grupach dzieł.

E1 jest testem poprawności danych i przydatności cech, a nie jeszcze transferem
stylu. Nie uruchamia GAN-u ani algorytmu genetycznego.

Wynik należy opisywać jako rozróżnialność trzech podzbiorów ASAP. Korpus jest
silnie skorelowany z formą i epoką: Bach to głównie preludia/fugi, Beethoven —
części sonat, a Chopin — mieszanka etiud i większych form. Wysoka trafność sama
w sobie nie dowodzi, że model nauczył się „czystej” tożsamości kompozytora.

## Kontrakt danych

### Jednostka próbki

Jedna próbka to jeden unikalny rekord `(composer, title)` i jeden kanoniczny
`midi_score`.

Reguła wyboru wariantu jest deterministyczna:

1. zebrać wszystkie `midi_score` dla `(composer, title)`;
2. preferować ścieżkę bez sufiksu `_no_repeat` i `_no_2_repeat`;
3. pozostałe wersje zapisać w `alternate_score_paths`, ale nie używać ich jako
   kolejnych próbek;
4. nie kopiować ani nie modyfikować surowego pliku.

### Identyfikatory

Manifest E1 powinien zawierać:

- `sample_id` — stabilny slug z `(composer, title)`;
- `composer`, `title`, `score_path` i `alternate_score_paths`;
- `group_id` używany przy podziale;
- SHA-256 pliku oraz fingerprint metadanych ASAP;
- format SMF, TPB, liczbę ścieżek, nut i zdarzeń meta;
- zakres wysokości, maksymalną polifonię i długość w taktach/beatach;
- wynik walidacji oraz ewentualny `exclusion_reason`;
- wersję schematu manifestu i commit kodu.

Konserwatywne `group_id`:

- Bach: wspólna grupa dla `Prelude_bwv_N` i `Fugue_bwv_N`; Italian Concerto ma
  własną grupę;
- Beethoven: wszystkie dostępne części `Piano_Sonatas_N-*` w grupie sonaty `N`;
- Chopin: wszystkie części tej samej sonaty w jednej grupie; pozostałe tytuły
  stanowią osobne grupy.

### Kontrola jakości

Każdy kanoniczny plik musi przejść:

1. kontrolę istnienia, rozszerzenia i SHA-256;
2. walidację struktury SMF i próbę odczytu obecnym `MidiParser`;
3. kontrolę `ticks_per_beat > 0`, co najmniej jednej nuty, poprawnych pitch,
   velocity i czasów trwania;
4. kontrolę skończoności wszystkich wyliczonych cech;
5. wykrywanie identycznych SHA wewnątrz i między klasami;
6. raport wartości odstających, ale bez automatycznego odrzucania tylko dlatego,
   że utwór jest krótki, długi albo gęsty.

Wykluczenia muszą być zapisane z przyczyną. Nie wolno po obejrzeniu wyników
klasyfikacji usuwać trudnych próbek.

## Cechy i warianty eksperymentu

### E1a — baseline na obecnym ekstraktorze

Najpierw wykorzystać istniejący 42-elementowy wektor, aby otrzymać szybki wynik
diagnostyczny. Uruchomić dwie ablacje:

- `legacy_full`: wszystkie obecne cechy;
- `legacy_score_only`: bez `tempo_bpm`, `note_density_per_s` oraz średniej i
  odchylenia długości w sekundach.

Drugi wariant ogranicza sygnały zależne od arbitralnego tempa zapisanego w score
MIDI. Histogram klas wysokości, histogram interwałów i `rest_ratio` mogą zostać.

### E1b — główny zestaw cech kompozycyjnych

Cechy rytmiczne należy liczyć w beatach lub względem metrum, nie w sekundach.
Minimalne grupy:

- **pitch:** zakres, średnia, odchylenie, entropia i histogram klas wysokości;
- **melody:** interwały kierunkowe/bezwzględne, kontur i krótkie n-gramy;
- **rhythm:** długości oraz IOI w beatach, pozycje metryczne onsetów, synkopa;
- **texture:** średnia/maksymalna polifonia, rozmiary akordów, udział akordów;
- **harmony:** chroma i proste przejścia/stabilność tonalna;
- **structure:** powtarzalność krótkich wzorców, z zachowaniem ostrożności dla
  różnej długości dzieł.

Każda cecha musi mieć nazwę, grupę, jednostkę i test na prostym syntetycznym MIDI.
Osobno raportować wynik grup cech oraz ich ablacje. Velocity pozostaje poza
głównym wariantem, ponieważ badamy zapis kompozycji, a nie interpretację
wykonawcy.

## Modele i protokół walidacji

Modele minimalne:

1. `DummyClassifier` — strategia prior/most-frequent;
2. regresja logistyczna z regularyzacją i `class_weight="balanced"`;
3. random forest z wagami klas.

Dla regresji skalowanie, usuwanie cech stałych i dobór hiperparametrów muszą być
częścią `Pipeline` dopasowywanego wyłącznie na danych treningowych.

Protokół główny:

- 5 powtórzeń 5-fold `StratifiedGroupKFold` z zapisanymi seedami;
- `group_id` jako grupa — żadna grupa nie może trafić do dwóch foldów;
- wewnętrzne 3-fold grouped CV do doboru hiperparametrów;
- wykorzystanie wszystkich zaakceptowanych 150 próbek, bez redukowania Bacha i
  Beethovena do 34 utworów;
- `balanced_accuracy` jako metryka główna, `macro_F1` jako druga;
- macierz pomyłek z predykcji out-of-fold;
- 95% przedział ufności liczony z bootstrapu klastrowego po `group_id`;
- test permutacyjny etykiet na poziomie grup jako kontrola istotności;
- zapis predykcji każdej próbki i każdego zewnętrznego foldu.

Analiza wrażliwości: powtórzyć ocenę, losując po jednej próbce z każdej grupy.
Zmniejsza to przewagę kompozytorów reprezentowanych wieloma częściami tych samych
sonat i pokazuje, czy wynik nie jest skutkiem liczby movements.

Analiza wrażliwości obejmuje ponowne dopasowanie modeli i wewnętrzny dobór
hiperparametrów na zredukowanym zbiorze, a nie tylko przeliczenie metryk na
podzbiorze gotowych predykcji. W każdym powtórzeniu wybór próbki jest
deterministyczny względem zapisanego seeda i wspólny dla wszystkich modeli oraz
wariantów cech.

Rozróżniamy dwie kontrole permutacyjne:

- szybka permutacja etykiet względem gotowych predykcji OOF jest wyłącznie
  diagnostyką zgodności i nie stanowi testu całej procedury uczenia;
- raportowany test z ponownym uczeniem permutuje etykiety całych `group_id`, po
  każdej permutacji ponownie dopasowuje wcześniej ustalone pipeline'y i używa
  pierwszego, wcześniej zapisanego powtórzenia splitów. Hiperparametry kontroli
  są stałe (`C=1` dla logistic oraz `max_features="sqrt"`,
  `min_samples_leaf=1` dla RF), dzięki czemu nie są dobierane na podstawie
  obserwowanych lub permutowanych etykiet. Finalny przebieg E1.2 powinien używać
  co najmniej 99 takich permutacji.

Nie tworzymy jednego stałego test setu przy tak małej liczbie grup. Zewnętrzne
foldy pełnią rolę niewidzianych testów, a ich predykcje są agregowane dopiero po
zakończeniu wszystkich foldów.

### Obserwowalność długich przebiegów

Klasyfikacja musi raportować postęp co najmniej na poziomie wariantu cech,
modelu, powtórzenia i zewnętrznego foldu. Dla każdego dopasowania zapisujemy
zdarzenie rozpoczęcia i zakończenia, czas wykonania oraz balanced accuracy foldu.
Postęp jest widoczny w terminalu i równolegle dopisywany do `progress.jsonl`,
aby można było odróżnić kosztowne strojenie od zawieszenia procesu. Log postępu
jest artefaktem pomocniczym; autorytatywne metryki nadal pochodzą z kompletnego
`e1a_results.json` zapisywanego atomowo po zakończeniu przebiegu.

### Analiza dodatkowa: rozpoznawanie kompozytora spoza zbioru (open set)

Główny E1 pozostaje klasyfikacją **zamkniętą** Bacha, Beethovena i Chopina. Jest
to właściwy protokół dla pytania, czy zdefiniowane cechy rozróżniają te trzy
podzbiory na niewidzianych dziełach. Wyniku nie należy jednak interpretować jako
zdolności rozpoznania dowolnego kompozytora: klasyfikator zamknięty zawsze
przypisze jedną z trzech znanych etykiet, także dla materiału spoza domeny.

Po E1b należy wykonać osobną analizę `E1-open`, bez włączania jej do głównego
kryterium sukcesu E1:

- model bazowy jest uczony wyłącznie na trzech klasach E1;
- wynik może zostać odrzucony jako `unknown` na podstawie wcześniej ustalonego
  progu pewności lub odległości od rozkładu cech klas znanych;
- próg dobiera się bez użycia kompozytorów przeznaczonych do końcowego testu
  open-set;
- dodatkowych kompozytorów ASAP dzieli się **po kompozytorach** na zbiór
  kalibracyjny i testowy, aby ten sam nie wystąpił po obu stronach;
- nie tworzy się jednej uczonej klasy `other`, ponieważ łączyłaby niejednorodne
  style i zamieniałaby problem z powrotem w zwykłą klasyfikację zamkniętą;
- jako kontrolę wykonuje się rotacyjne leave-one-composer-out dla trzech klas E1:
  dwie klasy są znane, a trzecia pełni rolę `unknown`;
- raport obejmuje AUROC/AUPRC dla `known` kontra `unknown`, false-positive rate
  dla ustalonego true-positive rate, recall klasy `unknown`, pokrycie oraz
  balanced accuracy wśród zaakceptowanych próbek znanych.

ASAP zawiera dodatkowy materiał m.in. Liszta, Schuberta, Haydna i Schumanna,
więc tę analizę można przygotować bez pobierania nowego korpusu. Najpierw trzeba
jednak objąć te dane tym samym audytem kanonizacji, grupowania i kontroli SHA co
trzy klasy główne. Analiza ma charakter testu odporności i zakresu stosowalności,
nie dowodu atrybucji autorstwa.

Zamrożony podział E1-open:

- kalibracja progu unknown: Haydn, Mozart i Schumann (27 próbek);
- końcowy test unknown: Liszt, Schubert, Rachmaninoff i Ravel (37 próbek);
- znane próbki z foldów 0–1 pierwszego powtórzenia służą do kalibracji progu,
  a foldy 2–4 do raportowanej oceny;
- każda próbka unknown jest deterministycznie przypisana do dokładnie jednego
  modelu foldowego, dzięki czemu jej confidence nie pochodzi z uprzywilejowanego
  ensemble'u i jest porównywalne z predykcją OOF próbki znanej;
- logistic regression (`C=1`) i random forest (`max_features="sqrt"`,
  `min_samples_leaf=1`) mają stałe hiperparametry, bez nested grid searchu;
- próg maksymalizuje balanced accuracy known/unknown pod warunkiem zachowania
  co najmniej 95% znanych próbek w części kalibracyjnej.

Pierwszy przebieg diagnostyczny potwierdził, że confidence nie wystarcza jeszcze
do niezawodnego odrzucania. Random forest osiągnął AUROC 0,742, ale przy
zamrożonym progu recall klasy unknown wyniósł tylko 0,108; logistic regression
osiągnęła AUROC 0,676 i recall unknown 0,000. Wynik pozostaje analizą zakresu
stosowalności i nie zmienia kryterium sukcesu closed-set E1.

## Etapy implementacji i bramki jakości

### E1.0 — środowisko i audyt surowych danych

Zakres:

- wykorzystanie istniejącego `.venv` i uzupełnienie zależności klasyfikacyjnych;
- adapter `metadata.csv` specyficzny dla ASAP;
- manifest 150 kandydatów, wybór wersji kanonicznych i grupowanie;
- raport jakości oraz testy adaptera.

Istniejący `.venv` (Python 3.10.20) zawiera `numpy` i `mido`. Audyt potwierdził,
że wszystkie 150 kanonicznych plików przechodzą przez obecny parser i ekstraktor
bez wyjątku oraz bez wartości `NaN/inf`. W środowisku brakuje `scikit-learn`,
którego nie ma również w deklaracjach zależności projektu. Dodanie i przypięcie
tej zależności należy wykonać w E1.0, bez uruchamiania treningu GAN.

**Go/no-go:** wszystkie pliki są rozliczone jako accepted/excluded; zero
nieudokumentowanych wyjątków parsera; brak identycznego SHA między klasami;
każda klasa zachowuje co najmniej 30 próbek. Jeśli warunek nie jest spełniony,
naprawiamy adapter/parser albo jawnie zmieniamy zakres przed E1.1.

### E1.1 — deterministyczne splity

Zakres:

- implementacja i zapis zewnętrznych/wewnętrznych foldów;
- asercje rozłączności `sample_id`, `group_id` i SHA;
- tabela liczności klas oraz grup dla każdego foldu.

**Go/no-go:** żaden `group_id` ani SHA nie przecieka między train/test, każda
próbka ma dokładnie jedną predykcję out-of-fold w każdym powtórzeniu.

### E1.2 — E1a, obecne cechy

Zakres:

- ekstrakcja i cache cech z numerem schematu;
- dummy, logistic regression i random forest;
- porównanie `legacy_full` z `legacy_score_only`;
- kompletny raport metryk i predykcji;
- terminalowy postęp dopasowań oraz strukturalny `progress.jsonl` z czasami
  wykonania foldów.

**Go/no-go:** pipeline jest deterministyczny dla tego samego seeda, cechy
walidacyjne nie uczestniczą w skalowaniu ani wyborze hiperparametrów, dummy daje
wynik zgodny z oczekiwanym poziomem odniesienia. Walidator OOF potwierdza brak
brakujących, nadmiarowych i zduplikowanych predykcji oraz zgodność foldu,
kompozytora, `group_id` i SHA. Finalny katalog zawiera zakończony
`run_manifest.json`, snapshoty wejść i analizę jednej próbki na grupę.

### E1.3 — E1b, cechy kompozycyjne

Zakres:

- nowe cechy mierzone w beatach/metrum wraz z testami;
- klasyfikacja tym samym protokołem;
- ablacje grup cech i permutation importance liczone wewnątrz foldów;
- analiza pomyłek i form muzycznych.

**Kryterium sukcesu E1:** dolna granica 95% CI balanced accuracy najlepszego
wcześniej zdefiniowanego modelu jest powyżej `1/3`, test permutacyjny odrzuca
hipotezę braku związku, a rezultat nie opiera się wyłącznie na cechach tempa.

### E1.4 — zamknięcie eksperymentu

Zakres:

- raport wyników, ograniczeń i decyzji o przejściu do E2;
- konfiguracja, seedy, wersje bibliotek, commit i fingerprint danych;
- tabela wszystkich wykluczeń oraz predykcji;
- wykresy macierzy pomyłek, wyników foldów, ablacji i ważności cech.
- jawne ograniczenie wyniku closed-set oraz, jeśli ukończono `E1-open`, osobny
  raport skuteczności odrzucania nieznanych kompozytorów.

Wynik negatywny jest poprawnym rezultatem E1: oznacza konieczność zmiany cech lub
korpusu przed użyciem klasyfikatora jako funkcji celu GA.

## Planowane artefakty

```text
configs/e1_asap.yaml                  # śledzona konfiguracja
src/musicians_style/e1/               # adapter, audyt, splity, klasyfikacja
tests/unit/e1/                         # testy bez dużego zbioru
datasets/derived/e1_asap/             # manifest/cache; ignorowane przez Git
experiments/e1_asap_<timestamp>/      # wyniki uruchomienia; ignorowane przez Git
docs/results/E1.md                    # śledzone podsumowanie po eksperymencie
```

Surowy `datasets/asap-dataset-1.2/` pozostaje niezmieniony. Kod nie powinien
przenosić ani zmieniać nazw jego plików.

## Stan implementacji

- **E1.0 — wykonany:** adapter ASAP, manifest, raport jakości i bramka audytu.
- **E1.1 — wykonany:** deterministyczne 5×5 `StratifiedGroupKFold`, wewnętrzne
  3-fold grouped CV, kontrole przecieku `sample_id`/`group_id`/SHA i liczności.
- **E1.2 — implementacja domknięta:** wersjonowany cache `legacy_full` oraz
  `legacy_score_only`, nested CV dla dummy/logistic/RF, walidowane predykcje OOF,
  metryki, stratyfikowany bootstrap klastrowy, ponownie trenowana analiza jednej
  próbki na grupę, test permutacyjny z ponownym uczeniem i pełna proweniencja
  przebiegu. Do zamrożenia wyniku potrzebny jest finalny przebieg z co najmniej
  99 permutacjami retreningowymi.
- **E1.3 — implementacja domknięta:** wersjonowany kontrakt 93 cech
  kompozycyjnych mierzonych w beatach/metrum, testy syntetyczne, wariant pełny,
  wyniki pojedynczych grup i ablacje leave-one-group-out. Klasyfikacja korzysta
  z tego samego nested grouped CV co E1.2, liczy permutation importance na
  zewnętrznych foldach oraz zapisuje analizę błędów według form wyprowadzonych z
  tytułów ASAP. Do zamrożenia wyniku potrzebny jest finalny przebieg E1b z co
  najmniej 99 permutacjami retreningowymi.
- **E1-open — implementacja i pierwszy przebieg ukończone:** rozszerzony audyt
  zaakceptował 214/214 plików, rozłączne zbiory kompozytorów kalibracyjnych i
  testowych są zamrożone, a raport obejmuje AUROC/AUPRC, FPR przy zadanym TPR,
  recall unknown, pokrycie, jakość zaakceptowanych znanych próbek i trzy kontrole
  leave-one-composer-out.
- **E1.4 — implementacja domknięta, oczekuje na finalny E1b:** generator wymaga
  ukończonego `run_manifest.json`, sprawdza kryterium sukcesu, zapisuje CSV-y,
  cztery wykresy, proweniencję, ograniczenie closed-set i opcjonalną sekcję
  E1-open. Nieukończony test permutacyjny daje status „niekompletne”.

Nie ma potrzeby pobierania teraz MAESTRO, GiantMIDI-Piano ani innego zbioru.
