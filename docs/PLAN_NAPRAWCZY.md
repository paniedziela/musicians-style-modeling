# Plan naprawczy

Plan jest ułożony tak, aby po każdym etapie powstał samodzielny, opisywalny
wynik pracy. Długi trening GAN jest celowo odsunięty do chwili, gdy dane,
metryki i baseline'y będą wiarygodne.

## Definicja wyniku pracy

Hipoteza badawcza powinna mieć dwie części:

1. wybrany zestaw cech symbolicznych pozwala rozróżniać utwory Bacha, Chopina i
   Mozarta lepiej niż baseline losowy;
2. transformacja optymalizowana w kierunku artysty zwiększa zgodność z jego
   profilem stylu, zachowując rozpoznawalną melodię i długość wejścia.

Wynik negatywny dla GAN-u nadal jest wartościowy, jeżeli zostanie rzetelnie
porównany z baseline'ami i wyjaśniony.

## P0. Zatrzymanie obecnego eksperymentu

Nie wznawiać treningu od epoki 21 przed wykonaniem etapów P0–P2. Zachować:

- checkpoint `epoch_021.pt` jako artefakt diagnostyczny;
- konfigurację, log i dziewięć wyników;
- tabelę obecnych rezultatów jako „eksperyment zerowy / nieudany”.

Nie przedstawiać go jako końcowego modelu. W pracy może posłużyć do opisania,
dlaczego segmentacja i protokół ewaluacji są krytyczne.

**Kryterium zakończenia:** jeden opisany rekord eksperymentu z wersją kodu,
manifestami, konfiguracją, czasem treningu i znanymi ograniczeniami.

## P1. Uporządkowanie danych

### 1. Ujednolicić domenę

Najmniej ryzykowny wariant to **solo piano dla wszystkich kompozytorów**.
Nie mieszać symfonii Mozarta, mszy Bacha i miniatur fortepianowych Chopina w
jednym eksperymencie o stylu kompozytorskim.

Możliwe źródła:

- obecny korpus po ręcznym filtrowaniu i udokumentowaniu pochodzenia;
- [MAESTRO](https://magenta.withgoogle.com/datasets/maestro) — dane fortepianowe,
  metadane kompozytorów i gotowy split, ale ograniczona liczba kompozytorów;
- [GiantMIDI-Piano](https://doi.org/10.5334/tismir.80) — 10 855 solowych utworów
  fortepianowych i 2786 kompozytorów, lecz MIDI pochodzą z automatycznej
  transkrypcji i wymagają kontroli jakości.

### 2. Zbudować audytowalny manifest

Dla każdego pliku zapisać co najmniej:

- kompozytora, tytuł dzieła, część/movement i źródło;
- SHA-256, długość, metrum, liczbę ścieżek i programy MIDI;
- liczbę nut, zakres wysokości, udział nut poza zakresem modelu;
- informację „score MIDI” vs „performance/transcription MIDI”;
- powód przyjęcia lub odrzucenia.

Wyszukiwanie rekursywne powinno być jawnie konfigurowalne. Duplikaty po SHA-256
oraz możliwe duplikaty tego samego dzieła należy grupować przed podziałem.

### 3. Podział bez przecieku

Podział `train/validation/test` wykonywać po **dziełach**, nie po losowych
fragmentach. Wszystkie części lub transpozycje jednego utworu muszą znaleźć się
w jednym foldzie. Zalecany jest stratified group split i co najmniej kilka
powtórzeń lub cross-validation.

**Kryteria zakończenia P1:**

- co najmniej trzy porównywalne domeny;
- brak identycznych SHA pomiędzy splitami;
- żaden utwór nie występuje w więcej niż jednym splicie;
- raport liczności dzieł i fragmentów per kompozytor;
- ręczny odsłuch losowej próby plików z każdej domeny.

## P2. Analiza i walidacja cech — rdzeń celu pracy

### 1. Rozszerzyć cechy

Obecne tempo, histogramy klas wysokości/interwałów, gęstość, długości i pauzy
są dobrym minimum. Należy dodać cechy obejmujące:

- zakres, centrum i entropię wysokości;
- rozkład interwałów bezwzględnych i kierunków ruchu;
- n-gramy interwałowe lub kontury melodyczne;
- synkopę, onsety na pozycjach metrycznych, IOI i wzorce rytmiczne;
- polifonię, rozmiary akordów i równoczesności;
- przejścia harmoniczne/chroma, dysonans i stabilność tonalną;
- powtarzalność motywów i autokorelację;
- cechy per głos/ścieżka, jeżeli źródło wiarygodnie je zachowuje.

Nie trzeba implementować wszystkiego samodzielnie. Warto porównać własny
ekstraktor z [musif](https://musif.didone.eu/) lub cechami `music21`/jSymbolic.
`musif` wspiera MIDI, choć najwyższą jakość analiz daje zwykle MusicXML/Kern.

### 2. Znormalizować i wybrać cechy

- dopasować skalowanie wyłącznie na foldzie treningowym;
- usunąć stałe i prawie stałe cechy;
- kontrolować silną współliniowość histogramów;
- raportować ważność permutation importance lub współczynniki modelu;
- wykonać ablację grup cech: pitch, rhythm, harmony, texture, dynamics.

### 3. Klasyfikacja

Minimalne modele:

- baseline majority/random;
- regresja logistyczna z regularyzacją;
- random forest lub gradient boosting;
- opcjonalnie mały klasyfikator pianoroll onset+sustain.

Metryki: macro-F1, balanced accuracy, macierz pomyłek i przedziały ufności.
Nie wybierać cech i hiperparametrów na zbiorze testowym.

**Kryteria zakończenia P2:**

- balanced accuracy wyraźnie ponad baseline'em w group cross-validation;
- lista cech istotnych i stabilnych pomiędzy foldami;
- analiza, które pary kompozytorów są mylone;
- porównanie z modelem bez metadanych wykonawczych (tempo/velocity), aby
  odróżnić styl kompozycji od interpretacji wykonawcy.

## P3. Naprawa reprezentacji i segmentacji GAN

Ten etap jest potrzebny niezależnie od wyboru StarGAN/CycleGAN.

### Dataset

1. Podzielić utwór według taktów na kolejne frazy czterotaktowe.
2. Nie ograniczać się do pierwszego fragmentu.
3. Odrzucać lub oznaczać okna puste; raportować zapełnienie.
4. Zachować mapowanie `work_id`, `start_bar`, `end_bar`, split i artist.
5. Balansować liczbę fraz artystów samplerem, bez duplikowania testu.
6. Dla ostatniego fragmentu używać paddingu i maski.

Oryginalny muzyczny CycleGAN używał kolejnych czterotaktowych fraz 64×84,
utworów 4/4, jednej ścieżki fortepianowej i osobnego train/test. To minimum,
którego brakuje w obecnym potoku.

### Reprezentacja

Preferowany wariant ma co najmniej dwa kanały:

- onset;
- sustain/aktywność nuty.

Opcjonalnie trzeci kanał velocity. Zapobiega to zlewaniu sąsiednich powtórzeń
tej samej wysokości w jedną długą nutę. Zakres wysokości i kwantyzacja muszą być
wyliczone z raportu danych, nie tylko skopiowane z artykułu.

### Inferencja całego utworu

1. Rozbić wejście na te same okna taktowe.
2. Przetworzyć każde okno z tym samym targetem.
3. Złożyć je na oryginalnej osi czasu.
4. Usunąć padding ostatniego okna.
5. Rozstrzygnąć graniczne sustainy i powtarzane onsety.
6. Przy zachowaniu metrum wymusić identyczną liczbę taktów.

Prosty overlap-add jest możliwy, lecz dla binarnych onsetów potrzebna jest jawna
reguła łączenia, nie średnia wartości bez kalibracji.

### Dekodowanie

Próg 0,5 zastąpić progiem kalibrowanym na walidacji albo dekodowaniem
uwzględniającym docelową gęstość. Dodać ograniczenia:

- minimalna/maksymalna długość nuty;
- limit polifonii zgodny z danymi;
- zakaz zerowego wyniku dla niepustego wejścia;
- kontrola nut brzmiących przez wiele taktów;
- osobne odtwarzanie onsetów i sustainów.

**Kryteria zakończenia P3:**

- dataset zawiera wszystkie kwalifikujące się frazy;
- round-trip segmentacja→scalenie bez generatora zachowuje nuty i długość;
- wejście 100-taktowe daje wynik 100-taktowy (± jeden krok kwantyzacji);
- każdy niepusty testowy fragment daje niepusty wynik;
- raportowane są occupancy, onset density i udział pustych okien per domena.

## P4. Wybór modelu neuronowego

### Wariant rekomendowany: warunkowany model jako eksperyment dodatkowy

Po P1–P3 można zachować jeden generator wielodomenowy, ale należy:

- losować target różny od źródła dla kroku transferu;
- identity liczyć osobno dla etykiety źródłowej;
- używać zbalansowanych batchy domen;
- dodać prawdziwy validation loader;
- monitorować styl, treść i poprawność, nie tylko straty;
- zachowywać `best` według złożonej metryki, nie każdy pełny checkpoint;
- zapisać pełną geometrię i konfigurację preprocessing w checkpoint.

Nie nazywać go wiernym StarGAN-em, jeżeli funkcja adversarial lub trening
różnią się od publikacji; bezpieczna nazwa to „warunkowany wielodomenowy GAN
inspirowany StarGAN”.

### Jeżeli potrzebny jest CycleGAN

Zaimplementować dokładnie:

- domenę A i B;
- `G_AB`, `G_BA`;
- `D_A`, `D_B`;
- dwie straty adversarial;
- dwa cykle A→B→A i B→A→B;
- identity w obu domenach;
- osobne batch'e niesparowanych danych.

Wariant ma sens dla jednej dobrze zdefiniowanej pary, np. Chopin↔Mozart piano.
Nie jest naturalnym sposobem obsługi wielu kompozytorów.

### Alternatywa Transformer/VAE

[MuseMorphose](https://github.com/YatingMusic/MuseMorphose) jest dobrym punktem
odniesienia dla dłuższych struktur i kontroli taktowej, a
[MidiTok](https://miditok.readthedocs.io/en/latest/) upraszcza tokenizację MIDI.
To jednak osobny projekt badawczy. Wybrać go tylko wtedy, gdy:

- P2 jest zakończone;
- dostępny jest jednorodny, większy korpus;
- można poświęcić czas na tokenizację i trening sekwencyjny;
- styl kompozytora zostanie zdefiniowany jako jawny warunek, a nie tylko dwa
  atrybuty MuseMorphose.

**Kryterium go/no-go:** mały model musi po kilku epokach pokonać identity i
prostą transformację w co najmniej jednej metryce stylu, bez pogorszenia
metryki treści poza ustalony limit. Jeśli nie — GAN/Transformer pozostaje
wynikiem negatywnym i nie pochłania reszty harmonogramu.

## P5. Przebudowa GA

### Nowa funkcja celu

Zamiast jednej odległości do średniej:

```text
score = style_gain
        - λ_content * content_distance
        - λ_length * length_error
        - λ_invalid * musical_invalidity
```

Gdzie `style_gain` to poprawa względem wejścia, nie tylko bezwzględna odległość
wyniku od centroidu. Cechy muszą być standaryzowane na zbiorze treningowym.
Jeszcze lepszy wariant to optymalizacja wielokryterialna Pareto (np. NSGA-II),
która nie ukrywa kompromisu styl–treść w jednej arbitralnej wadze.

### Bogatszy, ale kontrolowany genotyp

Rozważyć parametry lokalne per takt/fraza:

- transpozycja diatoniczna i zmiany stopni skali;
- wybór wzorca rytmicznego z biblioteki artysty;
- kontrolowane przesunięcia onsetów;
- ornamentacja/dodanie lub usunięcie nut przejściowych;
- zagęszczanie/redukcja akordów;
- długości i artykulacja;
- velocity tylko dla performance MIDI.

Każdy operator powinien mieć ograniczenie zachowania melodii przewodniej. Jeśli
melodia nie jest wydzielona, najpierw potrzebny jest prosty algorytm ekstrakcji
melodii (np. najwyższy głos z regułami ciągłości).

### Długość

Nie skalować globalnie wszystkich ticków w zakresie 0,5–2,0, jeżeli wymagane
jest ±5%. Zmiany rytmu wykonywać wewnątrz niezmiennych granic taktów, a na końcu
normalizować długość frazy dokładnie do oryginału.

**Kryteria zakończenia P5:**

- identity jest członkiem populacji/baseline'em;
- wynik poprawia styl względem wejścia na zbiorze testowym;
- liczba taktów i metrum są zachowane;
- metryka melodii nie pogarsza się ponad ustalony próg;
- eksperyment ma ablację operatorów i wag celu.

## P6. Ewaluacja końcowa

### Baseline'y

Co najmniej:

1. brak transformacji (identity);
2. losowa transpozycja w dozwolonym zakresie;
3. dopasowanie tylko tonacji/tempa;
4. obecny czterogenowy GA;
5. ulepszony GA;
6. model neuronowy, jeśli przejdzie go/no-go.

### Metryki stylu

- zmiana prawdopodobieństwa niezależnego klasyfikatora docelowego;
- standaryzowana odległość cech do rozkładu artysty;
- macro-F1/balanced accuracy klasyfikatora na realnych danych;
- cechy per grupa: melody, rhythm, harmony, texture.

Klasyfikator do ewaluacji nie powinien być tą samą głową, którą trenowano GAN.

### Metryki treści i poprawności

- podobieństwo konturu melodii lub n-gramów interwałowych;
- odległość chroma po wyrównaniu;
- precyzja/recall onsetów w tolerancji czasowej;
- błąd długości i liczby taktów;
- udział pustych taktów, zakleszczonych nut i przekroczonej polifonii;
- poprawność otwarcia pliku w MuseScore/DAW.

### Odsłuch

Ślepy test ABX lub MOS na tych samych wejściach i znormalizowanym SoundFouncie.
Pytania osobno o:

- rozpoznawalność melodii;
- podobieństwo do artysty docelowego;
- naturalność/muzyczność;
- preferencję ogólną.

Nie łączyć tych pytań w jedną ocenę. Randomizować kolejność, raportować liczbę
słuchaczy i przedziały ufności.

**Kryterium końcowe:** metoda wygrywa z identity w mierze stylu, zachowuje
długość w tolerancji, nie przegrywa wyraźnie w rozpoznawalności melodii i nie
produkuje pustych lub strukturalnie uszkodzonych plików.

## P7. Porządek techniczny repo

Po ustaleniu metody:

- jedno źródło zależności i osobne instrukcje CPU/CUDA;
- schema wersjonowana dla manifestu, cech i checkpointu;
- każdy eksperyment zawiera manifesty/splity, config, commit, log, metryki i
  indeks wyników;
- checkpoint `last` oraz `best`, opcjonalnie retencja co N epok;
- błędne uruchomienie zapisuje `status.json` z przyczyną;
- CLI otrzymuje `features`, `classify`, `infer-ga`, `infer-combined`, `audit`;
- specyfikację Kiro oznaczyć jako historyczną albo zaktualizować po decyzji
  metodologicznej;
- dodać testy P11/P12/P18/P19 na rzeczywistym długim przepływie.

## Minimalna macierz eksperymentów

| ID | Dane | Metoda | Cel |
|---|---|---|---|
| E0 | obecne, pierwsze okno | checkpoint ep. 21 | udokumentowany wynik negatywny |
| E1 | jednorodne piano, split po dziełach | cechy + logistic/RF | wykazać rozróżnialność stylu |
| E2 | jak E1 | obecny GA | baseline transferu |
| E3 | jak E1 | ulepszony GA | główny transfer interpretowalny |
| E4 | pełna segmentacja | mały conditional GAN | smoke/go-no-go |
| E5 | pełna segmentacja | najlepszy model neuronowy | tylko po sukcesie E4 |
| E6 | ustalony test | identity/E2/E3/E5 | porównanie obiektywne i odsłuch |

Najpierw E1. Bez niego nie wiadomo, czy dane i wybrane cechy w ogóle zawierają
sygnał odpowiadający formalnemu celowi pracy.
