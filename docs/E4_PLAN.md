# Plan implementacji E4 — mały warunkowany GAN i bramka go/no-go

## Decyzja w jednym zdaniu

E4 będzie ograniczonym eksperymentem decyzyjnym na jednym zamrożonym foldzie
ASAP: ma sprawdzić, czy mały wielodomenowy GAN potrafi przesunąć pełne utwory w
kierunku wskazanego kompozytora bez obcinania czasu, pustych wyników i utraty
chronionej treści. Nie jest to jeszcze strojenie najlepszego modelu neuronowego
ani pełne porównanie E6.

Model należy nazywać **warunkowanym GAN-em inspirowanym StarGAN**, a nie
StarGAN-em ani CycleGAN-em. Zachowujemy ideę jednego generatora `G(x, c)` i
dyskryminatora z głowami real/fake oraz domeny, ale reprezentacja, cel
adwersaryjny i zastosowanie do MIDI są własną adaptacją.

## Stan implementacji (2026-09-03)

Etapy E4.0–E4.5 są zaimplementowane w schemacie `e4.5.0`: audyt danych,
segmentacja i scalenie, trening, kalibracja progów, wybór checkpointu na
validation, pełna ewaluacja sześciu kierunków, bramka go/no-go oraz osobno
blokowany outer test. Testy jednostkowe obejmują również stan sustainu między
oknami i blokadę testu przed validation GO.

Historyczny katalog `experiments/e4_asap` zawiera niekompatybilny przebieg E4.3,
w którym `best.pt` wybierano po stracie treningowej. Nie wolno interpretować go
jako wyniku E4.4/E4.5. Nowy protokół zapisuje artefakty do
`experiments/e4_asap_v2`; wynik badawczy powstaje dopiero po wykonaniu treningu
i walidacji. Etap `test` wolno uruchomić tylko wtedy, gdy `validation.json`
zawiera `passed: true`.

## Addendum E4.6 (2026-09-16)

Pierwsza walidacja E4.5 zakończyła się NO-GO i ujawniła collapse warunku celu:
11 z 43 par wyników dla dwóch różnych kompozytorów docelowych było identycznych,
a mediana podobieństwa zdarzeń nutowych wyniosła `0,9946`. Pliki technicznego
smoke miały dodatkowo od 9 do 32 razy więcej nut niż wejścia.

E4.6 jest izolowanym wariantem w schemacie `e4.6.0`, konfiguracji
`configs/e4_asap_v3.yaml` i katalogu `experiments/e4_asap_v3`. Nie ładuje
checkpointów E4.5. Zmiany protokołu:

- warunek kompozytora steruje affine parameters po InstanceNorm
  (conditional InstanceNorm/FiLM), zamiast być stałą mapą usuwaną przez
  normalizację;
- generator przewiduje residual względem bezpiecznego logitowego identity;
- identity i cycle używają ważonego L1 z wagami dodatnich onset/frame
  wyznaczonymi wyłącznie z train i ograniczonymi do wartości `12`;
- selection używa trzech kwantyli długości utworów na kompozytora i jawnie
  premiuje różnicę pomiędzy dwoma celami;
- finalna bramka odrzuca średnie podobieństwo zdarzeń dwóch celów `>= 0,99`;
- progress zapisuje wszystkie składowe lossu, wagi rekonstrukcji oraz balanced
  accuracy i macierz pomyłek głowy klasyfikującej dyskryminatora;
- smoke sprawdza także gęstość nut, skrajne wysokości i polifonię, a nie tylko
  możliwość odczytu oraz długość pliku.

Pierwszy smoke E4.6 usunął wcześniejszy black-MIDI failure: sześć wyników ma
`0,70–1,00x` liczby nut wejścia, nie generuje nut na granicach 24/107 i zachowuje
zakres wysokości źródła. Nie jest to jeszcze wynik walidacji ani dowód transferu.

Polecenia nowego wariantu:

```powershell
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage smoke
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage train
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage evaluate
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage report
```

## Rola E4 w pracy

| Etap | Rola |
|---|---|
| E1 | niezależny miernik rozróżnialności Bacha, Beethovena i Chopina |
| E2 | historyczny baseline prostych transformacji globalnych |
| E3 | główna, interpretowalna metoda transferu |
| E4 | tani test, czy model neuronowy daje sygnał uzasadniający E5 |
| E5 | dopiero po sukcesie E4: stabilizacja i szerszy eksperyment neuronowy |

E1 osiągnął balanced accuracy `0,864` na podziale grupowym, więc istnieje
niezależny sygnał stylometryczny, którego można użyć do ewaluacji E4. E4 nie
zastępuje E3 i nie może opóźniać zamknięcia jego raportu. Implementację można
przygotować wcześniej, ale kosztowny trening rozpoczynamy dopiero po zamrożeniu
wyniku E3 albo po jawnej decyzji o przejściu dalej mimo wyniku negatywnego.

## Pytanie badawcze i hipoteza

**Pytanie:** czy mały, wielodomenowy GAN uczony na czterotaktowych fragmentach
score MIDI zwiększa prawdopodobieństwo kompozytora docelowego względem wejścia,
jednocześnie zachowując liczbę taktów, metrum, długość i rozpoznawalną treść?

**Hipoteza robocza:** na wcześniej ustalonym zbiorze walidacyjnym średnie
`delta_p_target` będzie dodatnie w co najmniej czterech z sześciu kierunków, a
wyniki przejdą te same twarde kontrole struktury co E3. Jest to bramka
inżyniersko-badawcza, nie końcowa hipoteza statystyczna pracy.

Wynik negatywny jest dopuszczalny. Jeżeli model nie przejdzie bramki, raport E4
zamyka wariant GAN i E5 nie jest realizowane.

## Co przejmujemy z literatury, a co jest naszą decyzją

| Decyzja | Podparcie | Status |
|---|---|---|
| cztery kolejne takty jako kontekst i pianoroll `64 x 84` | Brunner et al. [L1] | mocna inspiracja, ale u nas metra nie ograniczają się do 4/4 |
| jeden generator dla wielu domen, etykieta celu i głowa klasyfikacji domeny | StarGAN [L2] | mocne architektonicznie; publikacja dotyczy obrazów |
| osobne kanały onset i aktywności nuty | Kim et al. [L3] | mocne dla reprezentacji kompozytora, nie dowodzi skuteczności transferu |
| osobna ocena stylu i zachowania treści | Cífka et al. [L4] | bardzo mocne metodologicznie |
| niezależny klasyfikator do pomiaru przesunięcia domeny | Brunner et al. [L1] | użyteczne, lecz nie może być jedyną metryką [L4] |
| zbalansowanie liczby fragmentów domen | Brunner et al. [L1] | mocne jako ochrona przed skrótem klasowym |
| `conv_dim=32`, trzy bloki residualne, limity epok i progi dekodowania | nasza konfiguracja E4 | zamrozić przed testem; nie przypisywać literaturze |
| normalizacja każdego taktu do 16 pozycji przy zmiennym metrum | nasza adaptacja ASAP | wymaga audytu i testu round-trip |

Brunner et al. użyli czterech kolejnych taktów 4/4, szesnastu kroków na takt i
84 wysokości, otrzymując próbkę `64 x 84`; oceniali transfer osobnym
klasyfikatorem gatunku [L1]. Nie kopiujemy jednak ich wniosków wprost:
gatunek nie jest kompozytorem, ich model jest dwudomenowym CycleGAN-em, a korpus
ASAP ma zmienne metrum.

StarGAN uzasadnia pojedynczy generator warunkowany etykietą celu oraz
dyskryminator rozpoznający autentyczność i domenę [L2]. Oryginalna publikacja
używa innego celu adwersaryjnego i danych obrazowych, dlatego zgodność nazw i
opis ograniczeń są częścią kontraktu E4.

Kim et al. pokazali, że rozdzielenie onsetów od aktywności nut jest użyteczne w
klasyfikacji kompozytora na symbolicznym MIDI [L3]. Ich kanał aktywności zawiera
velocity; w E4 świadomie go binaryzujemy, ponieważ używamy score MIDI. Taka
reprezentacja ma też funkcję techniczną: odróżnia kolejne uderzenia tej samej
wysokości od jednej długiej nuty, czego obecny jednokanałowy pianoroll nie
potrafi.

## Zamrożony kontrakt danych

- Ten sam manifest 150 score MIDI ASAP co w E1–E3: Bach, Beethoven i Chopin.
- Tylko `repeat=0`, `outer fold=0`.
- Train i validation pochodzą odpowiednio z train/validation `inner fold=0`;
  outer test pozostaje nietknięty do jednorazowej oceny po przejściu walidacji.
- Jednostką podziału pozostaje `group_id`; fragmenty jednego dzieła nigdy nie
  przechodzą pomiędzy train, validation i test.
- Model widzi wyłącznie score MIDI. Velocity nie jest kanałem stylu.
- Nie używamy starego korpusu Bach/Chopin/Mozart ani checkpointu epoki 21.
- Nie wykonujemy transpozycji ani innych augmentacji w E4. Ich wpływ byłby
  dodatkową zmienną, której mały eksperyment go/no-go nie potrzebuje.

Wstępny odczyt obecnego manifestu pokazuje zakres wysokości `22..103` i tylko
3 nuty poza `[24,108)` na 365 680 zdarzeń. E4.0 ma zapisać ten audyt jako
artefakt; dopiero jego wynik zamraża zakres. Domyślnie pozostaje `[24,108)`,
zgodny z geometrią 84 wysokości z [L1]. Nuty poza zakresem nie mogą znikać po
cichu: raportujemy je per kompozytor i zachowujemy z wejścia przy składaniu.

Ten sam wstępny odczyt wskazuje, że tylko 33 ze 150 plików pozostają w 4/4
przez cały utwór. Ograniczenie korpusu do 4/4 byłoby więc silne i nierówne między
kompozytorami; stąd taktowe mapowanie czasu, a nie skopiowanie preprocessingu
[L1] bez adaptacji.

## Minimalna metoda E4

### 1. Segmentacja taktowa

1. Użyć metrum i granic taktów z `e3.structure`, zamiast dzielić utwór co stałe
   16 ćwierćnut.
2. Tworzyć niepokrywające się okna czterech kolejnych taktów (`stride=4`).
3. Każdy takt odwzorować na 16 równych pozycji, więc tensor ma 64 kroki także
   dla 3/4, 6/8 i zmian metrum. Metadane okna przechowują dokładne granice
   ticków potrzebne do odwrotnego mapowania.
4. Ostatnie niepełne okno dopełniać zerami; maska wskazuje prawdziwe kroki.
   Padding nie uczestniczy w stratach rekonstrukcji ani w metrykach.
5. Nutę przecinającą granicę okna przyciąć w lokalnym tensorze, ale zapisać
   stan kontynuacji. Przy scalaniu onset należy do okna, w którym nuta się
   zaczęła, a aktywność na granicy służy do odtworzenia sustainu.
6. Okna puste zachować w manifeście i raporcie, lecz nie podawać ich jako
   samodzielnych próbek treningowych. Niepuste wejście nie może dać pustego
   wyniku.

Normalizacja taktu do 16 pozycji jest świadomym kompromisem. Zachowuje położenie
metryczne i stały kształt sieci, ale ogranicza rozdzielczość triol i złożonych
metrów. Audyt E4.0 raportuje błąd kwantyzacji onsetu i duration per metrum. Jeśli
95. percentyl bezwzględnego błędu onsetu przekroczy połowę lokalnego kroku,
błąd końca nuty przekroczy jeden krok albo pojawią się nieobsługiwane granice,
implementacja zatrzymuje się przed treningiem; nie wykluczamy tych przypadków
dopiero po obejrzeniu wyników modelu.

### 2. Reprezentacja onset + frame

Próbka ma kształt `[2, 64, 84]`:

- kanał `onset`: `1` tylko w kroku rozpoczęcia nuty;
- kanał `frame`: `1` we wszystkich krokach jej aktywności, łącznie z onsetem.

Maska czasu jest osobnym tensorem, nie trzecim kanałem generowanym przez model.
Generator zwraca dwa logity, a sigmoid jest stosowany w stracie i dekoderze.
Pozwala to użyć stabilniejszej straty z logitami i kalibrować kanały osobno.

### 3. Model i funkcje straty

Minimalna architektura:

- `G(x, c_target)`: encoder–3 bloki residualne–decoder, `conv_dim=32`;
- `D(x)`: wspólny backbone PatchGAN i dwie głowy: real/fake oraz kompozytor;
- conditioning przez przestrzenne dołączenie one-hot celu, jak w [L2];
- output generatora: dwa kanały bez końcowego sigmoidu.

Minimalny cel generatora:

```text
L_G = L_adv
    + lambda_cls * L_cls_fake
    + lambda_cyc * L1_masked(G(G(x, c_target), c_source), x)
    + lambda_id  * L1_masked(G(x, c_source), x)
```

Dla kroku transferu `c_target` zawsze różni się od `c_source`. Identity jest
liczone osobnym przejściem z etykietą źródłową. Głowa klasyfikacyjna
dyskryminatora uczy się kompozytora wyłącznie na prawdziwych fragmentach;
generator jest oceniany względem etykiety celu na fragmentach sztucznych.

E4 zachowuje least-squares adversarial loss z obecnego prototypu, ponieważ jest
to mała adaptacja możliwa do zweryfikowania bez wdrażania pełnego WGAN-GP.
Właśnie ta różnica uniemożliwia nazywanie modelu wiernym StarGAN-em.

### 4. Sampling i trening

- Batch ma równy udział trzech kompozytorów dzięki samplerowi; nie duplikujemy
  validation ani testu.
- Jeden zamrożony seed `1729` dla głównego pilota.
- Adam: `lr=0.0002`, `beta1=0.5`, `beta2=0.999`; są to wartości startowe zgodne
  z eksperymentem [L1], nie przedmiot przeszukiwania.
- `batch_size=16`, maksymalnie 30 epok, zapis tylko `last.pt` i `best.pt`.
- Walidacja po każdej epoce na stałej liście pełnych utworów i sześciu
  kierunkach; early stopping po 5 epokach bez poprawy.
- Nie robimy grid searchu. Dopuszczalna jest jedna korekta techniczna po smoke
  teście (np. batch po OOM), zapisana w manifeście przed treningiem właściwym.

Straty treningowe służą do diagnostyki, nie są kryterium sukcesu. `best.pt`
wybieramy leksykograficznie: najpierw kompletność i poprawność MIDI, potem
spełnienie limitów treści, a dopiero wśród wyników feasible najwyższe średnie
`delta_p_target` na validation.

### 5. Dekodowanie i pełny utwór

1. Progi onset/frame kalibrować na identity reconstruction zbioru validation,
   osobno dla obu kanałów, na małej zamrożonej siatce `0.1..0.9`.
2. Onset zawsze rozpoczyna nową nutę, nawet gdy `frame` był aktywny w poprzednim
   kroku. Dzięki temu sąsiednie powtórzenia tej samej wysokości nie zlewają się.
3. `frame` bez onsetu na początku okna może jedynie kontynuować nutę aktywną na
   końcu poprzedniego okna; w innym przypadku jest ignorowany i liczony jako
   błąd dekodowania.
4. Wynik poza zakresem modelu, tempo, metrum, program i inne meta pochodzą z
   wejścia. Końcowy tick i liczba taktów są wymuszane przez oryginalną mapę okien.
5. Dla niepustego okna pusty output uruchamia bezpieczny fallback identity dla
   tego okna i zostaje jawnie policzony. Fallback zapobiega uszkodzeniu pliku,
   ale nie jest zaliczany jako sukces modelu.
6. Składanie zwraca jeden pełny `InternalRepr`; dopiero potem następują zapis,
   ponowne parsowanie i ewaluacja.

### 6. Ewaluacja

Styl i treść raportujemy osobno, zgodnie z argumentem Cifki et al., że idealny
wynik tylko w jednym z tych wymiarów jest łatwy, lecz bezużyteczny [L4].

**Styl:**

- `delta_p_target` niezależnego klasyfikatora E1; dla wyboru checkpointu jest
  on uczony tylko na inner train, a przy outer test — na całym outer train;
- zmiana standaryzowanej odległości grup cech `pitch`, `rhythm`, `texture` do
  profilu celu z train;
- wyniki per sześć kierunków, nie tylko średnia globalna.

**Treść i poprawność:**

- błąd długości i zgodność liczby taktów/metrum;
- podobieństwo trigramów konturu melodii jak w E2/E3;
- onset-F1 oraz chroma cosine względem wejścia;
- odsetek pustych taktów, fallbacków identity, osieroconych frame'ów,
  zakleszczonych nut i przekroczeń polifonii;
- ponowne otwarcie każdego zapisanego MIDI parserem.

Klasyfikator E1 jest niezależny od głowy `D_cls`, ale nadal może reagować na
artefakty generacji zamiast muzycznie przekonującego stylu. Dlatego jego wynik
nie wystarcza bez metryk cech, treści i kontroli strukturalnych; podobne
ograniczenie ewaluacji klasyfikatorem opisują [L1] i [L4]. Odsłuch formalny
należy do E6. W E4 wystarczy ręczny odsłuch sześciu wcześniej ustalonych par
wejście–wynik jako kontrola błędów niewidocznych w metrykach.

## Etapy implementacji i bramki

### E4.0 — audyt danych i zamrożenie geometrii

Zaimplementować tryb `audit`, który bez treningu zapisuje:

- liczbę taktów i czterotaktowych okien per split i kompozytor;
- rozkład metrów, długości okien, occupancy, onset density i pustych okien;
- listę nietypowych długości taktów (w ćwierćnutach), w szczególności poza
  zakresem `0,5..12`, aby wychwycić pickupy i artefakty meta-zdarzeń;
- pitch min/max i nuty poza zakresem;
- błąd kwantyzacji onset/duration przy mapowaniu takt -> 16 kroków;
- listę odrzuceń z jednoznacznym powodem.

**Bramka:** brak przecieku `group_id`, brak błędów parsera, co najmniej 100
niepustych okien każdego kompozytora w train oraz zaakceptowany raport błędu
kwantyzacji. Każde oflagowane metrum musi mieć przed treningiem zamrożoną regułę
obsługi albo jawnego odrzucenia całego dzieła. Niespełnienie bramki zmienia
reprezentację przed treningiem, a nie prowadzi do cichego filtrowania domeny.

### E4.1 — segmentacja, reprezentacja i scalenie

Zaimplementować struktury `Segment`, `SegmentMap`, encoder, decoder i stitcher.
Najpierw przejść round-trip bez generatora na danych syntetycznych, potem na
jednym długim utworze ASAP z każdego kompozytora.

**Bramka:** identyczna liczba taktów i końcowy tick; żadna nuta nie znika poza
jawną kwantyzacją lub zakresem; ponowny parse jest poprawny; powtórzenia tej
samej wysokości pozostają oddzielnymi onsetami.

### E4.2 — dataset i model

Zbudować dataset z indeksu segmentów, sampler zbalansowany po kompozytorach,
małe modele `G` i `D` oraz pojedynczy krok uczenia.

**Bramka:** target różny od source w każdym kroku transferu, zgodne kształty,
skończone straty i gradienty obu modeli oraz brak danych validation/test w
loaderze treningowym.

### E4.3 — harness, checkpoint i inferencja

Nowy harness ma etapy `audit`, `prepare`, `smoke`, `train`, `evaluate`, `test`, `report`,
atomowe artefakty, `status.json`, wznowienie z `last.pt` i pełną geometrię
preprocessingu w checkpointcie. Nie rozszerzamy starego globalnego CLI, dopóki
E4 nie przejdzie go/no-go.

**Bramka smoke:** dwie epoki na małym, zbalansowanym podzbiorze; sześć pełnych
plików przechodzi inferencję, zapis i ponowny parse; żaden wynik nie jest
skrócony. Smoke sprawdza przepływ, nie jakość stylu.

### E4.4 — pilot validation i decyzja go/no-go

Uruchomić jeden trening na `inner fold=0`, wybrać `best.pt` bez użycia outer
test i wygenerować zamrożony zestaw validation.

E4 przechodzi do jednorazowej ewaluacji testowej tylko wtedy, gdy łącznie:

- średnie `delta_p_target > 0` i co najmniej 4/6 kierunków ma dodatnią średnią;
- model poprawia co najmniej jedną grupę cech stylu bez pogorszenia wszystkich
  pozostałych;
- 100% plików zachowuje metrum, liczbę taktów i przechodzi round-trip parsera;
- błąd długości wynosi najwyżej jeden lokalny krok kwantyzacji;
- mediana podobieństwa trigramów melodii wynosi co najmniej `0,95`;
- żaden niepusty plik nie jest pusty po złożeniu;
- fallback identity dotyczy najwyżej 1% niepustych okien.

Progi treści są celowo zgodne z ambicją E3 i zostają zamrożone przed treningiem.
Nie dostrajamy ich po zobaczeniu wyników.

Jeżeli wynik jest wyraźnie ujemny, kończymy E4. Jeżeli jedna miara leży dokładnie
na granicy i log wskazuje losowość treningu, wolno wykonać jeden seed
potwierdzający. Nie uruchamiamy automatycznie wielu seedów ani grid searchu.

### E4.5 — jednorazowy test i raport

Po przejściu validation zamrozić checkpoint i progi, a następnie wykonać raz
inferencję wszystkich zadań outer test: każde wejście do dwóch pozostałych
kompozytorów. Raport pokazuje E4 względem identity i, jeśli dostępne, sparowanych
wyników E3. Bez ponownego wyboru modelu na podstawie testu.

Pozytywny outer test pozwala zaplanować E5. Negatywny test zamyka E4 jako wynik
negatywny; nie wracamy do validation z nową architekturą pod wpływem testu.

## Testy — tylko konieczne

| Test | Po co istnieje | Koszt |
|---|---|---:|
| segmentacja -> scalenie długiego, mieszanego metrycznie MIDI | wykrywa główny błąd starego potoku: pierwsze okno i skrócenie | sekundy |
| onset/frame na powtórzeniu wysokości i nucie przez granicę | chroni semantykę nowej reprezentacji | sekundy |
| izolacja splitów i liczba segmentów | zapobiega przeciekowi i pomijaniu dalszych okien | sekundy |
| jeden krok `G/D` | target != source, kształty, maska, skończone gradienty | sekundy |
| jeden syntetyczny end-to-end CLI | checkpoint -> pełne MIDI -> parse | sekundy/minuta |

Nie dodajemy pełnego ASAP do `pytest`, testów każdej warstwy konwolucyjnej,
property tests z setkami losowań ani testów jakości muzycznej. `smoke`, trening i
pilot są eksperymentami uruchamianymi ręcznie. Dodatkowy test regresyjny powstaje
tylko wtedy, gdy implementacja ujawni konkretny błąd, którego nie obejmuje pięć
powyższych przypadków.

## Plan plików i ponowne użycie kodu

```text
configs/e4_asap.yaml
src/musicians_style/e4/
  __init__.py
  __main__.py
  segmentation.py       # SegmentMap, onset/frame, stitcher
  dataset.py             # indeks fragmentów i sampler domen
  model.py               # małe G/D; bez modyfikowania historycznego E0
  training.py            # straty, checkpoint best/last, resume
  experiment.py          # audit/prepare/smoke/train/evaluate/report
  evaluation.py          # E1b, metryki treści i zamrożona bramka
tests/unit/e4/
experiments/e4_asap_v2/  # ignorowane przez Git
docs/results/E4.md
```

Ponownie wykorzystać parser/printer MIDI, `e3.structure`, manifest i splity E1
oraz ewaluator E1–E3. Nie importować `PianorollDataset`, `GANTrainer` ani
`StyleTransferPipeline` jako rdzenia E4, ponieważ utrwalają kontrakt jednego
okna i jednego kanału. Stary kod pozostaje niezmieniony jako odtwarzalny zapis
E0; wspólne elementy wydzielamy dopiero wtedy, gdy realnie redukują duplikację i
mają mały test regresyjny.

## Artefakty i minimalne polecenia

Każdy run zapisuje snapshot konfiguracji i splitów, commit, seed, wersje schematów,
`audit.json`, `segments.jsonl`, `progress.jsonl`, `last.pt`, `best.pt`, progi
dekodowania, metryki per plik/kierunek oraz `status.json`.

Aktualne wywołania E4.6 (`configs/e4_asap.yaml` pozostaje historycznym
snapshotem E4.5 i jest celowo niezgodny z nowym schematem):

```powershell
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage audit
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage prepare
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage smoke
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage train
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage evaluate
# Tylko po validation GO:
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage test
python -m musicians_style.e4 --config configs/e4_asap_v3.yaml --stage report
```

## Kryterium zamknięcia E4

E4 jest zakończone, gdy istnieje odtwarzalny raport z jednoznaczną decyzją:

- **GO:** validation i outer test spełniają kryteria, więc E5 może rozszerzyć
  stabilizację, seedy i porównanie architektur;
- **NO-GO:** model nie daje mierzalnej poprawy stylu przy wymaganej ochronie
  treści albo potok pozostaje strukturalnie zawodny; E5 zostaje skreślone;
- **wynik technicznie nieważny:** wyłącznie gdy nie udało się spełnić bramki
  danych/round-trip. Taki stan wymaga naprawy potoku, nie interpretacji jakości
  GAN-u.

## Literatura bezpośrednio użyta w planie

- **[L1]** G. Brunner, Y. Wang, R. Wattenhofer, S. Zhao,
  [*Symbolic Music Genre Transfer with CycleGAN*](https://arxiv.org/abs/1809.07575),
  2018 — czterotaktowe próbki `64 x 84`, balans domen, CycleGAN dla
  symbolicznego MIDI i ewaluacja osobnym klasyfikatorem.
- **[L2]** Y. Choi et al.,
  [*StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation*](https://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html),
  CVPR 2018 — jeden generator wielodomenowy, conditioning celem, głowa domeny i
  rekonstrukcja do domeny źródłowej.
- **[L3]** S. Kim et al.,
  [*Deep Composer Classification Using Symbolic Representation*](https://arxiv.org/abs/2010.00823),
  2020 — dwukanałowa reprezentacja onset/frame w klasyfikacji kompozytorów.
- **[L4]** O. Cífka, U. Şimşekli, G. Richard,
  [*Supervised Symbolic Music Style Translation Using Synthetic Data*](https://archives.ismir.net/ismir2019/paper/000071.pdf),
  ISMIR 2019 — konieczność osobnego pomiaru style fit i content preservation
  oraz ograniczenia klasyfikatora jako jedynej miary stylu.
