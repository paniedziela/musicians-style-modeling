# Inferencja E3 i eksperymentalna E4.6 na własnym MIDI

Uruchamiaj w katalogu głównym repozytorium. Nowa inferencja wykorzystuje istniejące
`E3GeneticAlgorithm.run`, bez klasyfikatora E1, treningu i eksperymentów zbiorczych.
E4.6 korzysta z lokalnego checkpointu oraz istniejących `encode_piece` i
`_infer_piece` (łącznie z progami dekodera i fallbackiem). Nie uruchamia
treningu, kalibracji progów ani testu zewnętrznego.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
.venv\Scripts\python.exe -m musicians_style.e3 infer --help
```

Jednorazowo przygotuj profile (istniejące lokalne dane ASAP i splity E1;
wyłącznie outer train powtórzenia 0, foldu 0, sprawdzone SHA i rozłączność
sample/group/SHA względem test). To agregacja istniejących cech E3, bez uczenia modelu:

```powershell
.venv\Scripts\python.exe -m musicians_style.e3 prepare-profiles
.venv\Scripts\python.exe -m musicians_style.e3 styles
```

Profile zostają w `inference_workspace/profiles`. Nie powtarzaj przygotowania,
jeśli istnieją. Inny zestaw można zapisać przez `--profiles-dir` i `--fold`.
Inferencja potrzebuje tylko profili, kodu i zależności Python; nie czyta korpusu.
Nie tworzy profilu z wejścia. Odrzuca wejście o SHA należącym do train profilu;
nie wykrywa wszystkich aranżacji i duplikatów muzycznych o innych bajtach.

Pełne domyślne wyszukiwanie (32 osobniki, 60 generacji, seed 1729, stagnacja 12):

```powershell
.venv\Scripts\python.exe -m musicians_style.e3 infer --input "C:\muzyka\utwor.mid" --target Beethoven --output "inference_workspace/moj_wynik.mid"
```

Szybka konfiguracja demonstracyjna (nie powtórzenie historycznej ewaluacji):

```powershell
.venv\Scripts\python.exe -m musicians_style.e3 infer --input "examples/e3_bach_prelude/input_bach_bwv868.mid" --target Beethoven --output "inference_workspace/demo/nowy_wynik.mid" --generations 3 --population-size 8 --seed 1729
```

Obok MIDI powstaje JSON z pochodzeniem profilu, hashami, seedem, konfiguracją,
genotypem, transpozycją, historią, ograniczeniami i istniejącymi miarami treści.
Pliki istniejące nie są nadpisywane. Status `unchanged` oznacza niezmieniony utwór,
a nie udany transfer. `transformed` potwierdza zmianę, nie jakość muzyczną.
`normalized_only` oznacza usunięcie zdarzeń wykonawczych bez zmiany nut.
Zysk funkcji celu E3 nie jest procentem podobieństwa do kompozytora.

## Obsługiwane dane

- SMF 0/1, dodatnie PPQ, nuty sparowane w tej samej ścieżce; bez SMPTE i SMF 2.
- 1–10 000 nut, do 10 MB i 15 minut; do 10 000 taktów. To konserwatywne limity
  nowej inferencji, nie deklaracja ograniczeń historycznego eksperymentu.
- Metrum musi być dokładnie wyrażalne w tickach. Zmiany metrum rozpoczynają
  nowy segment; brak metrum oznacza założenie 4/4. Nie ma automatycznej kwantyzacji.
- Domyślny tryb `preserve` zachowuje `control_change` (w tym sustain CC64),
  `pitchwheel` i kanałowy `aftertouch`: kanały, wartości i absolutne ticki.
  Kontrolery nie wpływają na nutowe funkcje celu ani metryki, ale wpływają na
  odsłuch. Nie przeliczamy działania sustainu na długości nut. CC i zmiany programu są
  odtwarzane w porządku źródłowych ścieżek; w tym samym ticku przed nutami.
  Przy scalaniu ścieżek nie gwarantujemy identycznego porządku względem nut.
- Opcja `--midi-policy score-only` (także w formularzu) jawnie usuwa wszystkie
  dodatkowe zdarzenia wykonawcze, w tym SysEx i polifoniczny aftertouch.
  Zachowuje czasy nut, tempo i programy instrumentów; nie naprawia nut ani
  nie kwantyzuje E3. Raport wymienia typy i liczby usuniętych zdarzeń.
- `--midi-policy strict` przywraca pierwotne odrzucanie zdarzeń wykonawczych.
  SysEx/polifoniczny aftertouch są odrzucane również w `preserve` — nie
  kopiujemy bez interpretacji komunikatów mogących zmieniać strój/instrument.
- Perkusja kanału 10 pozostaje nieobsługiwana, także w `score-only`.
- Kanały, programy instrumentów, tempo i wybrane metadane są zachowane,
  ale ścieżki scalane; tekst, nazwy ścieżek i końcowa cisza nie są zachowane.
  Pozostałe pomijane metadane są wymieniane w ostrzeżeniach.
- Nakładanie nut musi przejść kontrolę parse/write/parse. Polifonia wyniku jest
  ograniczona istniejącą regułą E3: maksimum z wejścia i korpusu profilu.
- Profile są fortepianowe. Techniczna obsługa wielu kanałów nie dowodzi
  skuteczności dla instrumentacji innej niż fortepian.
- Skyline chroni czas i długość jednej najwyższej nuty na onset, nie całej
  melodii rozpoznanej muzycznie. Globalna transpozycja zmienia także jej wysokość.
  Po niezmienionej optymalizacji tonacje w eksporcie są transponowane;
  JSON rozdziela ograniczenia oceniane przed korektą i metryki eksportu.
- Ograniczenie liczby nut E3 wynosi ±10%, długości: jednocześnie 5% i PPQ/16.
  Dla monofonii większość operatorów lokalnych nie ma zastosowania.

E2 pozostaje poza nowym interfejsem; wymaga osobnych profili i funkcji celu.

## Eksperymentalny model E4.6

```powershell
.venv\Scripts\python.exe -m musicians_style.e4 styles
.venv\Scripts\python.exe -m musicians_style.e4 infer --input "C:\muzyka\utwor.mid" --target Beethoven --output "inference_workspace/moj_wynik_e4.mid"
```

Domyślnie: `experiments/e4_asap_v3/best.pt`, epoka 9, zamrożone progi
onset **0,7** i frame **0,5**, CPU, 2 wątki. Opcjonalnie `--checkpoint`
wskazuje inny zgodny lokalny checkpoint E4.6. Sprawdzamy wersję, geometrię,
mapowanie stylów, wagi i progi. E4.5 z `e4_asap_v2` ma inną architekturę
i jest odrzucane. Nie pobieramy nowych modeli. Seed i parametry GA dotyczą
wyłącznie E3 — E4 wykonuje deterministyczny forward z zapisanymi wagami.

**E4.6 nie przeszło walidacji jakości (NO-GO).** Udostępnienie inferencji
nie zmienia tego wyniku i nie odblokowuje outer test. Nie wyznaczamy
prawdopodobieństwa kompozytora ani procentu podobieństwa. Nie jest to
ewaluacja generalizacji na niezależnym zbiorze.

Ograniczenia E4: jeden kanał nutowy, do 256 segmentów (1024 takty), przynajmniej
16 ticków na takt/fragment metrum, co najmniej jedna dodatnia nuta w zakresie
MIDI 24–107. Siatka to 16 kroków/takt, okna po 4 takty. Nuty poza zakresem są
kopiowane; zerowe długości wewnątrz zakresu pomijane. Nakładające się nuty tej
samej wysokości mogą się scalić. Brak gwarancji zachowania melodii, polifonii
i liczby nut. Kanał i velocity są przypisywane przez istniejący dekoder
z najbliższej nuty tej samej wysokości (nowe wysokości: kanał 0, velocity 64).
Tonacja w metadanych pochodzi ze źródła i nie musi odpowiadać wygenerowanym nutom.

JSON zawiera SHA checkpointu, epokę, progi, diagnostykę fallbacku, miary treści
oraz miary samej rekonstrukcji reprezentacji. `change_origin=representation_only`
oznacza wynik identyczny z wejściem przepuszczonym przez siatkę, bez wykazanej
zmiany modelowej. `identity` oznacza nuty niezmienione; `model_and_representation`
oznacza różnicę względem wejścia i rekonstrukcji, bez potwierdzenia jakości.

## Lokalny interfejs webowy

```powershell
.venv\Scripts\python.exe -m musicians_style.listener --root . --soundfont soundfonts/FluidR3_GM.sf2 --cache inference_workspace/audio_cache --profiles-dir inference_workspace/profiles --inference-workspace inference_workspace/web --port 8765
```

Otwórz `http://127.0.0.1:8765`. Kliknij **Dodaj MIDI**, wybierz plik i załaduj
go przyciskiem **A**. W panelu „Przekształć plik A” wybierz **metodę** i styl,
pozostaw „Zachowaj kontrolery (w tym sustain CC64) i pitch bend”. Dla E3 opcjonalnie zmień
seed i budżet, następnie **Uruchom: A → B**. Parametry GA znikają dla E4.6.
Panel pokazuje generację E3 lub segment E4 i stan końcowy. Działa jedno zadanie w tle naraz;
odświeżenie strony w tej samej karcie odtwarza śledzenie bieżącego zadania.
Nie zamykaj serwera w trakcie optymalizacji. Ctrl+C kończy serwer; przerwane
zadanie nie jest wznawiane. Wgrane pliki są lokalne, bez usług zewnętrznych.

Po zakończeniu użyj **Pobierz wynik MIDI**, rozwiń raport, a w listenerze
porównaj A/B, piano roll i przewijanie. Oryginał zadania trafia do A, wynik do B.
Zostaw tempo **1×**. Synteza jest oddzielnym etapem — jej błąd nie blokuje
pobrania poprawnego MIDI. Potrzebne są lokalne `pretty_midi`, `pyFluidSynth`,
biblioteka FluidSynth i `soundfonts/FluidR3_GM.sf2`; należy je przygotować przed
uruchomieniem syntezy.

Każde zadanie zapisuje `input.mid`, `output.mid` i `output.json` w nowym
`inference_workspace/web/<identyfikator>/`. HTTP nie przyjmuje ścieżki wyjścia.
Upload i WAV są w `inference_workspace/audio_cache`. Stan zadań jest w pamięci
serwera; po restarcie wyniki zostają na dysku i można je otworzyć z biblioteki.
E4 jest dostępne, gdy checkpoint jest obecny i zgodny; jego brak nie blokuje E3.
Brak profili E3 nie blokuje E4. Argument serwera `--e4-checkpoint` wskazuje
plik wyłącznie podczas lokalnego uruchamiania, nigdy przez żądanie HTTP.
**Po aktualizacji zatrzymaj poprzedni serwer, uruchom go ponownie i odśwież
stronę (Ctrl+F5)** — działający proces Python nie przeładowuje kodu automatycznie.

## Gotowy przykład i scenariusz 2–3 minuty

W [`examples/e3_bach_prelude`](../examples/e3_bach_prelude/) znajduje się krótki
Preludium C-dur BWV 868 J.S. Bacha z ASAP oraz rzeczywisty wynik pełnego zadania
E3 dla stylu Beethoven (fold 2, seed 1729). Wejście ma 417 nut, a wynik 439;
oba trwają około 38 s. Pliki WAV umożliwiają odsłuch bez lokalnej syntezy,
a `result.json` i `provenance.json` opisują metryki oraz pochodzenie danych.
To historyczny wynik eksperymentu, nie rezultat pokazanej wyżej szybkiej
konfiguracji demonstracyjnej.

Dodatkowo [`examples/e3_kotek`](../examples/e3_kotek/) zawiera to samo krótkie
wejście oraz trzy przygotowane wyniki: Bach, Beethoven i Chopin. Jest to wygodne
porównanie odsłuchowe; dla tych plików nie zachowały się raporty ani dokładne
konfiguracje przebiegów, co zaznaczono w ich opisie pochodzenia.

1. Uruchom listener. Dla krótkiej listy demo zamień `--root .` na
   `--root examples/e3_bach_prelude`; pozostałe argumenty pozostają takie same.
2. Załaduj `input_bach_bwv868.mid` do A, a `output_beethoven.mid` do B.
3. Przełącz A/B, pokaż piano roll i odsłuch. W przygotowanym wyniku wystąpiła
   transpozycja **+4 półtony**, liczba nut wzrosła z 417 do 439, a czas nutowego
   materiału pozostał bez zmian.
4. Pokaż `result.json`: zysk celu wyniósł około **0,167**, a zmiana udziału
   docelowego stylu około **0,0219**. To miary modelu, nie procent podobieństwa
   do kompozytora ani dowód udanego transferu.
5. Opcjonalnie uruchom szybki przebieg 3 generacje / 8 osobników i pokaż postęp,
   status oraz raport. Wynik może różnić się od przygotowanego pełnego przebiegu.
6. Jako drugi odsłuch otwórz `examples/e3_kotek` i porównaj trzy style dla
   identycznego wejścia.

## Weryfikacja i granice sprawdzenia

**58 testów przeszło** (14 ostrzeżeń przestarzałych API zależności).
Testy E3/E4, listenera, zdarzeń MIDI i miar treści obejmują CLI, deterministyczność,
niepoprawne/brakujące wejście, brak profilu, identity, brak dopuszczalnego wyniku,
zgodność HTTP/CLI, pobieranie, poprawkę tonacji i ochronę danych treningowych.
Testy wykorzystują małe syntetyczne dane; przypadek niedopuszczalnego końcowego
wyniku wymuszono atrapą wyniku algorytmu. Nowe testy sprawdzają zachowanie
CC/pitch bend/aftertouch w SMF 0 i 1, jawne usuwanie zdarzeń bez przesunięcia
nut, zgodność E4 web/serwis, niezgodny checkpoint i brak progów.
Testy jednostkowe E4 używają syntetycznego checkpointu z zerową resztą modelu;
nie są dowodem jakości wytrenowanych wag.

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/unit/e3 tests/unit/e4 tests/unit/test_listener.py tests/unit/test_inference_midi_events.py tests/unit/test_content_metrics.py
```

Osobno wykonano rzeczywisty przepływ HTTP: upload → inferencja → pobranie
identycznego z CLI MIDI → odczyt piano roll → synteza WAV oryginału i wyniku.
Publiczne WAV-y mają 22 050 Hz: para z `examples/e3_bach_prelude` trwa około
39,0 s, a pliki z `examples/e3_kotek` około 18,67 s (z wybrzmieniem).
Nie wykonano oceny słuchowej ani zewnętrznego testu E4.
