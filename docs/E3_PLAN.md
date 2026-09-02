# Plan implementacji E3 — transfer kontrolowany przez ograniczenia treści

## Decyzja w jednym zdaniu

E3 będzie małym, interpretowalnym rozszerzeniem GA: ma zmieniać przede wszystkim
akompaniament wewnątrz istniejących taktów, a melodię, metrum i długość chronić
jako twarde ograniczenia. Nie implementujemy teraz GAN-u, NSGA-II ani genotypu
z osobnym genem dla każdej nuty lub każdego taktu.

## Po co są E1, E2 i E3

Najprostszy model mentalny jest następujący:

| Etap | Pytanie | Rola |
|---|---|---|
| E1 | Czy z MIDI da się rozpoznać kompozytora? | buduje „miernik stylu” i potwierdza, że dane zawierają sygnał |
| E2 | Co potrafi obecny czteroparametrowy GA? | zamrożony punkt odniesienia, także jeśli wynik jest słaby |
| E3 | Czy kontrolowana edycja poprawia styl bez psucia melodii? | główna proponowana metoda pracy |
| E4+ | Czy model neuronowy daje coś więcej? | opcjonalne rozszerzenie, nie warunek ukończenia rdzenia pracy |

E2 nie musi być dobrą metodą. Jego wartość polega na pokazaniu, dlaczego samo
globalne przesunięcie wysokości, onsetów, długości i velocity nie wystarcza.
Pilot już pokazuje ten mechanizm: średnie `Δp_target` jest dodatnie, lecz małe
(`0,0215`), żaden z 18 wyników nie został sklasyfikowany jako kompozytor docelowy,
średni błąd długości wynosi `26,2%`, a onset-F1 tylko `0,341`. GA poprawia więc
własny historyczny fitness, ale często robi to przez rozciągnięcie lub ściśnięcie
całego utworu. To uzasadnia E3; nie oznacza awarii implementacji E2.

## Co jest konieczne, a co byłoby overkillem

Konieczne dla tezy pracy są:

1. brak przecieku utworów testowych do profilu stylu;
2. jawna, mierzalna poprawa stylu względem wejścia i E2;
3. ochrona melodii, długości, metrum i poprawności MIDI;
4. powtarzalny eksperyment na tej samej macierzy 300 par.

Na tym etapie nie są konieczne:

- NSGA-II ani pełny front Pareto;
- lokalny genotyp o zmiennej długości;
- ornamentacja i generowanie nowych fraz;
- optymalizacja velocity, ponieważ ASAP score MIDI opisuje kompozycję, nie
  interpretację wykonawczą;
- duży grid search wag funkcji celu;
- ponowne uruchamianie pełnego nested CV z E1;
- E4/E5 i naprawa GAN-u przed zamknięciem E3.

## Pytanie badawcze i hipoteza

**Pytanie:** czy GA z operacjami lokalnymi w takcie i twardymi ograniczeniami
treści daje większy przyrost prawdopodobieństwa stylu docelowego niż obecny GA,
zachowując melodię i strukturę czasową wejścia?

**Hipoteza:** E3 ma wyższe `Δp_target` niż E2 w sparowanych zadaniach, przy 100%
wyników w tolerancji długości 5%, zachowanym metrum i wysokim podobieństwie
melodii.

Negatywny wynik nadal jest wynikiem badawczym. Nie dokładamy kolejnych operatorów
po obejrzeniu testowych rezultatów tylko po to, aby uzyskać dodatnią liczbę.

## Zamrożony kontrakt danych

- Te same 150 kanonicznych score MIDI ASAP i `repeat=0` splitów E1.
- Ta sama macierz 150 wejść × 2 cele = 300 zadań.
- Profil celu powstaje wyłącznie z treningowej części danego foldu.
- Klasyfikator ewaluacyjny jest trenowany wyłącznie na train danego foldu,
  identycznie jak w E2.
- E2 pozostaje niezmienione. E3 zapisuje osobny schemat, konfigurację i katalog.
- Wybór operatorów, ograniczeń i budżetu GA zostaje zamrożony przed pełnym runem.

## Minimalna metoda E3

### 1. Struktura utworu i chroniona melodia

Najpierw utwór jest dzielony na takty z użyciem `ticks_per_beat` i zdarzeń
`time_signature`. Zmiana metrum tworzy nowy odcinek; nie jest ignorowana.
Ostatni takt może być niepełny.

Melodia jest wyznaczana deterministycznie jako najwyższa nuta każdego onsetu;
ewentualny remis rozstrzyga stabilny identyfikator nuty. W E3-MVP nuty melodii mogą przejść wyłącznie wspólną
transpozycję całego utworu. Ich onsety i długości nie są modyfikowane. Po
cofnięciu globalnej transpozycji sekwencja wysokości, kontur i rytm melodii mają
być identyczne z wejściem.

To jest celowo prostsze od pełnej separacji głosów. Maska melodii i wszystkie
niejednoznaczne przypadki są zapisywane, aby ograniczenie dało się skontrolować.

### 2. Profil stylu celu

Dla każdego `(fold, target_composer)` budowany jest profil tylko z plików train:

- średnia i bezpieczne odchylenie standardowe wybranych cech E1b;
- histogram pozycji onsetów w 16 równych polach każdego taktu;
- histogramy długości nut;
- rozkład rozmiaru akordów i polifonii;
- rozkład klas wysokości oraz interwałów akompaniamentu.

Profil zawiera fingerprint listy plików treningowych. Test ma wykazać, że żaden
`sample_id`, `group_id` ani SHA z testu nie uczestniczy w jego budowie.

Do celu optymalizacji wchodzą tylko cechy, na które implementowane operatory
mogą świadomie wpływać: pitch-class/interval, metrical onset/duration oraz
chord-size/polyphony. Pełne 93 cechy E1b pozostają w ewaluacji. Dzięki temu GA
nie jest karany za cechy strukturalne, których małym zestawem operatorów nie
potrafi zmienić.

### 3. Mały genotyp i operatory

Genotyp E3-MVP ma pięć parametrów:

| Gen | Zakres | Znaczenie |
|---|---:|---|
| `transpose_semitones` | całkowite `[-6, 6]` | wspólna transpozycja zachowująca kontur |
| `rhythm_strength` | `[0, 1]` | udział onsetów akompaniamentu dopasowanych do typowych pozycji celu |
| `duration_strength` | `[0, 1]` | siła dopasowania długości akompaniamentu do rozkładu celu |
| `thin_strength` | `[0, 1]` | kontrolowana redukcja nadmiarowych nut akordu |
| `double_strength` | `[0, 1]` | kontrolowane zdublowanie istniejących nut w oktawie |

Operacje są deterministyczne dla `(genome, input, target_profile, seed)`.
Wybór nut do modyfikacji nie zależy od kolejności obiektów w pamięci. Onset nigdy
nie opuszcza swojego taktu, długość nuty jest dodatnia i przycinana do granicy
taktu. Dodanie oktawy jest dozwolone tylko w zakresie MIDI, poniżej chronionej
melodii danego onsetu i do limitu polifonii profilu celu.

`thin_strength` i `double_strength` nie mogą działać jednocześnie na tym samym
onsecie. To ogranicza przestrzeń poszukiwania i ułatwia interpretację.
Ornamentacja, zmiana melodii i swobodne generowanie wysokości pozostają poza MVP.

### 4. Ograniczenia zamiast wielu arbitralnych wag

Nie zaczynamy od sumy kilkunastu kar. Kandydat jest **feasible**, tylko jeśli:

- MIDI jest niepuste i przechodzi round-trip;
- `ticks_per_beat`, format SMF, metrum i pozostałe meta są zachowane;
- długość wyjścia różni się najwyżej o jeden krok kwantyzacji i zawsze mieści
  się w tolerancji 5%;
- onsety i długości chronionej melodii są zachowane;
- kontur melodii po cofnięciu transpozycji jest identyczny;
- liczba nut pozostaje w przedziale `[0,9; 1,1]` liczby wejściowej;
- nie rośnie liczba nut zerowej długości i nie jest przekroczony limit
  polifonii wynikający z train profilu.

Kandydaci niespełniający ograniczeń przegrywają z każdym kandydatem feasible.
Wśród feasible maksymalizujemy grupowo zrównoważony przyrost stylu:

```text
style_gain = mean_group(
    RMS_z_distance(input, target) - RMS_z_distance(output, target)
)
```

Standaryzacja `z` używa wyłącznie średniej i odchylenia train. Dzielenie na grupy
zapobiega wygraniu celu tylko dlatego, że jedna grupa ma więcej cech. Identity
jest jawnie pierwszym osobnikiem populacji i wynikiem awaryjnym: E3 nie zwraca
transformacji o ujemnym `style_gain`.

### 5. Budżet wyszukiwania

Punkt startowy to populacja `32`, maksymalnie `60` pokoleń, elita `2`, turniej
`3` i stop po `12` pokoleniach stagnacji. To 3–10 razy mniej ocen niż konfiguracja
E2, zależnie od chwili zatrzymania. Parametry są zamrażane przed technicznym
pilotem; pilot nie służy do wybierania konfiguracji o najlepszym wyniku testowym.
Jeżeli ujawni błąd lub nieakceptowalny koszt, zmiana protokołu wymaga nowej wersji
konfiguracji i powtórzenia całego pilota.

W historii zapisujemy liczbę unikalnych kandydatów i cache trafień. Ocena tego
samego zgenotypowanego wyniku w obrębie zadania jest cache'owana. Jest to ważne,
bo część parametrów ciągłych po kwantyzacji daje identyczne MIDI.

## Etapy implementacji

### E3.0 — zamknięcie diagnozy E2

1. Dokończyć 300 zadań E2 i wygenerować `docs/results/E2.md`.
2. Zamrozić tabelę: `Δp_target`, długość, onset-F1, melodia, polifonia, geny na
   granicach, pokolenia i czas.
3. Potwierdzić, że identyfikatory 300 zadań E3 można sparować 1:1 z E2.

Nie blokuje to pisania kodu E3. Blokuje jedynie zamrożenie finalnej konfiguracji
i rozpoczęcie pełnego E3.

### E3.1 — reprezentacja taktów i melodia

Pliki:

```text
src/musicians_style/e3/structure.py
src/musicians_style/e3/types.py
tests/unit/e3/test_structure.py
```

Implementacja: granice taktów, obsługa zmian metrum, maska melodii, stabilne
identyfikatory nut oraz walidacja niejednoznacznych onsetów.

**Bramka:** identity daje bitowo ten sam `InternalRepr`; długi syntetyczny utwór
i utwór ze zmianą metrum zachowują wszystkie nuty i granice.

### E3.2 — profil train-only i operatory

Pliki:

```text
src/musicians_style/e3/profile.py
src/musicians_style/e3/transformation.py
tests/unit/e3/test_profile.py
tests/unit/e3/test_transformation.py
```

Najpierw implementujemy rytm/duration, potem texture. Każdy operator można
wyłączyć wartością `0`, co musi być testowane jako identity. Nie dodajemy kolejnej
operacji, dopóki poprzednia nie spełnia ograniczeń melodii, taktu i długości.

**Bramka:** wszystkie operatory są deterministyczne, nie korzystają z testowych
próbek i osobno zachowują inwarianty.

### E3.3 — funkcja celu i GA

Pliki:

```text
src/musicians_style/e3/objective.py
src/musicians_style/e3/algorithm.py
tests/unit/e3/test_objective.py
tests/unit/e3/test_algorithm.py
```

E2 i obecny `musicians_style.ga` pozostają nietknięte jako reprodukowalny
baseline. E3 dostaje własne typy, ranking feasible/infeasible, identity w
populacji, cache ocen i historię składników celu.

**Bramka:** dla syntetycznego profilu algorytm znajduje znaną poprawę, nigdy nie
zwraca wyniku gorszego od identity i jest deterministyczny dla tego samego seeda.

### E3.4 — harness eksperymentu

Pliki:

```text
configs/e3_asap.yaml
src/musicians_style/e3/experiment.py
src/musicians_style/e3/__main__.py
tests/unit/e3/test_experiment.py
```

CLI zachowuje etapy E2: `prepare`, `pilot`, `run`, `report`. Manifest, wznowienie,
atomowe zapisy i `--workers` mają ten sam kontrakt. Wspólną ewaluację można
wydzielić z E2 dopiero wtedy, gdy test regresyjny potwierdzi identyczne metryki;
nie refaktoryzujemy E2 profilaktycznie.

Pilot E3 to tylko sześć kierunków × jeden seed. Jedno z sześciu zadań jest
powtarzane dla kontroli determinizmu — nie wszystkie zadania jak w E2.

**Bramka:** 6/6 poprawnych wyników, powtórzone zadanie identyczne, wznowienie
pomija wyniki, a oszacowanie czasu pełnego runu jest zapisane w manifeście.

### E3.5 — pełny eksperyment i raport

Uruchomienie:

```powershell
python -m musicians_style.e3 --config configs/e3_asap.yaml --stage prepare
python -m musicians_style.e3 --config configs/e3_asap.yaml --stage pilot --workers 2
python -m musicians_style.e3 --config configs/e3_asap.yaml --stage run --workers 2
python -m musicians_style.e3 --config configs/e3_asap.yaml --stage report `
  --e2-run-dir experiments/e2_asap
```

Raport porównuje E3 z wejściem i sparowanym zadaniem E2. Główna analiza to 95%
bootstrap CI klastrowane po `group_id` oraz test znaku/permutacyjny na średnich
grupowych. Sześć kierunków jest pokazane opisowo z korektą Holma; nie tworzymy
osobnej hipotezy dla każdej cechy.

Minimalna ablacja jest uruchamiana dopiero po pełnym wyniku i tylko na sześciu
zadaniach pilota: wyłączenie kolejno rytmu, duration i texture. Nie powtarzamy
300 zadań dla każdej wersji operatora. Jeśli E3 nie poprawia stylu, ablacja nie
jest potrzebna do decyzji głównej.

## Kryteria sukcesu i bezpieczny wynik negatywny

E3 jest **udanym transferem**, jeśli łącznie:

- średnia sparowana różnica `Δp_target(E3) - Δp_target(E2)` jest dodatnia, a
  95% CI klastrowane nie obejmuje zera;
- `Δp_target(E3)` względem identity ma dodatnią dolną granicę 95% CI;
- 100% wyników przechodzi round-trip i zachowuje meta;
- 100% wyników mieści się w błędzie długości 5%;
- mediana podobieństwa trigramów konturu melodii wynosi co najmniej `0,95`;
- żaden wynik nie jest gorszy od identity według zamrożonego celu E3.

Jeśli styl się nie poprawia, E3 jest zamkniętym wynikiem negatywnym: raportujemy,
że mały, bezpieczny zestaw operatorów nie wystarczył. Dopiero wtedy podejmowana
jest jawna decyzja, czy dodać jeden konkretny operator, czy przejść do E4. Nie
rozszerzamy automatycznie zakresu.

## Testy: które są naprawdę potrzebne

Testy automatyczne mają pozostać krótkie i nie używać całego ASAP:

| Rodzaj | Zakres | Oczekiwany koszt |
|---|---|---:|
| jednostkowe | takty, melodia, każdy operator, cel, ranking | sekundy |
| jeden test end-to-end | małe syntetyczne MIDI, 4 osobniki × 2 pokolenia | sekundy |
| test harnessu | 2–3 zadania z atrapą wykonawcy | sekundy |
| pilot | 6 realnych zadań, uruchamiany ręcznie | minuty |
| pełny run | 300 realnych zadań, uruchamiany raz | godziny |

Pilot i pełny run są eksperymentami, a nie częścią `pytest`. Nie dodajemy długich
testów property opartych na setkach losowań, jeśli ten sam inwariant można
sprawdzić kilkoma celowymi przykładami granicznymi.

## Planowane artefakty

```text
configs/e3_asap.yaml
src/musicians_style/e3/
tests/unit/e3/
experiments/e3_asap/          # ignorowane przez Git
docs/results/E3.md
```

Po E3 wymagany jest jeden raport porównawczy E2–E3. E4/GAN pozostaje osobną,
opcjonalną decyzją i nie jest częścią tego planu implementacji.
