# Plan eksperymentu E2 — baseline transferu obecnym GA

## Status i cel

E1 został zamknięty decyzją GO: balanced accuracy najlepszego wariantu E1b
wyniosła `0,864` (95% CI klastrowane `[0,838; 0,891]`). E2 mierzy, czy
istniejący, czterogenowy algorytm genetyczny potrafi przesunąć utwór w stronę
profilu innego kompozytora. E2 jest baseline'em dla E3, dlatego nie zmieniamy
jego genotypu, operatorów ani funkcji dopasowania.

Wynik dodatni, zerowy albo ujemny jest poprawnym wynikiem E2. Nie stosujemy
bramki GO/NO-GO; przejście do E3 wynika z kompletności pomiaru i diagnozy
ograniczeń obecnej metody.

## Zamrożony protokół

- Źródłem jest ten sam audytowany ASAP score MIDI i te same grupowe splity E1.
- Używamy pierwszego powtórzenia splitów (`repeat=0`) i pięciu foldów.
- W każdym foldzie profil celu powstaje wyłącznie z treningowych utworów celu.
- Każdy testowy utwór jest przekształcany do dwóch pozostałych kompozytorów;
  macierz główna ma 150 × 2 = 300 zadań i sześć kierunków.
- GA pozostaje skonfigurowany jako: populacja 100, 200 pokoleń, turniej 3,
  uniform/BLX-α, elita 2, stagnacja 30, Mahalanobis, seed `1729`.
- Fitness używa historycznego wektora 42 cech. Cechy E1b służą wyłącznie do
  niezależnej ewaluacji wyniku.
- Pseudoodwrotność kowariancji Mahalanobisa jest przygotowywana raz dla
  każdego profilu `(fold, kompozytor)` i współdzielona przez oceny osobników;
  test regresyjny potwierdza zgodność numeryczną z dotychczasowym API.

### Pilot

W foldzie 0 wybieramy dla każdego kompozytora testową próbkę najbliższą medianie
liczby nut. Sześć kierunków uruchamiamy dla seedów `1729`, `2718`, `3141`, czyli
18 zadań. Pilot sprawdza poprawność, determinizm, możliwość wznowienia i koszt
obliczeń; nie służy do strojenia parametrów.

### Niezależna ewaluacja

W każdym foldzie uczymy od zera Random Forest E1b tylko na train:
300 drzew, `class_weight=balanced`, `max_features=0.5`,
`min_samples_leaf=2`, `random_state=1729`, pełny wariant 93 cech.

Metryką główną jest:

```text
Δp_target = P(target | output) − P(target | input)
```

Raportujemy także prawdopodobieństwo źródła, standaryzowane odległości do celu
w grupach `pitch`, `melody`, `rhythm`, `texture`, `harmony`, `structure`, oraz
metryki treści: onset-F1 (tolerancja `1/16` beatu), podobieństwo trigramów
konturu melodii, błąd długości, zmianę liczby nut, polifonię, nuty zerowej
długości i poprawność round-trip MIDI.

## Uruchamianie i artefakty

```powershell
python -m musicians_style.e2 --config configs/e2_asap.yaml --stage prepare
python -m musicians_style.e2 --config configs/e2_asap.yaml --stage pilot
python -m musicians_style.e2 --config configs/e2_asap.yaml --stage run --workers 1
python -m musicians_style.e2 --config configs/e2_asap.yaml --stage report
```

`--stage all` wykonuje wszystkie etapy po kolei. `prepare` zapisuje manifest 18
zadań pilota i 300 zadań głównych oraz snapshoty wejść. Każde zadanie ma własny
katalog z `result.json`, `output.mid`, `genotype.json`, `history.json` i `ga.jsonl`; zakończone zadania są
pomijane przy kolejnym uruchomieniu, a nieudane można ponowić. Rekordy i raporty
są zapisywane atomowo. `--run-dir` pozwala wskazać osobny katalog przebiegu,
a `--workers N` włącza równoległość procesową (domyślnie `1`).

Katalog przebiegu zawiera ponadto `run_manifest.json`, `progress.jsonl`,
`results.json`, `results.csv`, `analysis.json` i wykresy (w tym rozkłady genów,
granice zakresów oraz koszt/stagnację). Raport końcowy jest
kopiowany do `docs/results/E2.md` dopiero po ukończeniu wszystkich 300 zadań.

## Analiza statystyczna i ograniczenia

Dla `Δp_target` podajemy średnią, medianę, 95% bootstrap CI klastrowane po
`group_id` (2000 replik) oraz dwustronny test permutacyjny znaku na poziomie
grup (999 permutacji). Wyniki kierunkowe mają korektę Holma. Pokazujemy również
fitness końcowy względem identity i najlepszego osobnika generacji 0, rozkład
genów, trafianie w granice i powody zatrzymania.

Klasyfikator E1b jest proxy stylu, nie dowodem atrybucji autorstwa. Historyczny
fitness zawiera tempo, a `velocity_offset` nie wpływa na jego 42 cechy. Globalne
skalowanie onsetów i długości może znacząco naruszać długość oraz rytm utworu.

## Kryteria akceptacji

- manifest wejściowy i fingerprinty E1 są zgodne;
- pilot ma 18/18 poprawnych, skończonych i deterministycznych wyników;
- przebieg główny ma 300/300 rozliczonych zadań bez przecieku train/test;
- każdy wynik ma poprawne MIDI, historię GA i komplet metryk;
- profil celu i ewaluator używają wyłącznie danych treningowych foldu;
- raport, CSV, JSON i wykresy są kompletne, a testy projektu przechodzą.

## Stan implementacji

Interfejs `musicians_style.e2`, konfiguracja, manifest zadań, wznowienie,
techniczna bramka pilota, analiza i generator raportu są zaimplementowane.
Pełna macierz nie jest uruchamiana automatycznie podczas testów repozytorium;
po wykonaniu 18 zadań pilota można wznowić ją poleceniem `--stage run`.
