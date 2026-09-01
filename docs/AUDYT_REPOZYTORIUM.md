# Audyt repozytorium i decyzja metodologiczna

Data audytu: 1 września 2026.

## Wniosek wykonawczy

Projekt ma wartościowy fundament techniczny, ale obecny eksperyment nie pozwala
jeszcze stwierdzić, że modeluje ani przenosi styl kompozytora. Słabe rezultaty nie
wynikają przede wszystkim z „podania surowych plików MIDI”. Kod faktycznie je
parsuje i zamienia na pianoroll. Problem polega na tym, **jak** je przygotowuje:
z każdego utworu wykorzystuje tylko pierwsze 64 kroki, czyli zwykle cztery takty,
a podczas inferencji również generuje tylko jedno takie okno.

Najbezpieczniejsza droga do realizacji formalnego celu pracy to uczynienie
analizy i walidacji cech kompozytorskich głównym eksperymentem, a następnie użycie
tych cech w interpretowalnym transferze opartym na optymalizacji/GA. GAN powinien
być wariantem rozszerzonym, dopóki nie spełni minimalnych kryteriów jakości.

## 1. Cel pracy a to, co bada repo

Cel zapisany w lokalnych materiałach brzmi w skrócie:

1. opracować zestaw cech pozwalających rozróżnić styl wybranych muzyków;
2. po wyodrębnieniu cech opracować metodę nakładania stylu na melodię.

Repo zawiera ekstraktor 42 wartości, lecz nie zawiera najważniejszego
eksperymentu potwierdzającego punkt 1: klasyfikacji kompozytorów, selekcji cech,
analizy ważności ani ablacji. Większość specyfikacji Kiro koncentruje się na
budowie GAN-u i testach poprawności kodu. To przesunęło ciężar z pytania
badawczego („które cechy rozróżniają styl?”) na artefakt („czy klasy i API
spełniają wymagania?”).

### Ocena zgodności

| Element celu | Stan | Ocena |
|---|---|---|
| zbiór dla kilku muzyków | Bach/Chopin/Mozart | jest, ale korpus jest niejednorodny |
| ekstrakcja cech | 42 cechy liczbowe | implementacja działa, zakres jest wąski |
| wykazanie rozróżnialności | brak klasyfikatora i walidacji | brak kluczowego dowodu |
| nakładanie stylu | StarGAN, pseudo-CycleGAN, GA | prototypy bez miarodajnej walidacji |
| zachowanie melodii | straty L1 i deklarowane ±5% długości | niespełnione dla pełnych utworów |
| porównanie metod | infrastruktura ewaluacji | brak kompletnego eksperymentu |

## 2. Co faktycznie zaimplementowano

### Tryb `conditional`

Jest to autorska adaptacja idei StarGAN do binarnego pianorolla:

- jeden generator `G(x, c)` warunkowany etykietą kompozytora;
- jeden PatchGAN z głową real/fake i klasyfikacją kompozytora;
- straty adversarial, classification, cycle i identity.

To nie jest modyfikacja CycleGAN-u w ścisłym sensie, lecz hybryda architektury
StarGAN z reprezentacją i częścią motywacji zaczerpniętą z pracy o symbolicznym
transferze gatunku. Oryginalny StarGAN był metodą wielodomenowego transferu
obrazów i stosował cel WGAN z gradient penalty. Tutaj adversarial loss jest
wariantem least-squares bez gradient penalty, więc najdokładniejsza nazwa to
„wielodomenowy GAN inspirowany StarGAN”. Skuteczność tej adaptacji dla MIDI musi
zostać osobno wykazana.

### Tryb `per_artist`

Nazwa i dokumentacja sugerują CycleGAN A↔B, lecz kod ma:

- jeden generator `G`;
- jeden dyskryminator `D`;
- dane wyłącznie jednego artysty;
- „cykl” liczony jako `G(G(x)) ≈ x`.

Pełny CycleGAN wymaga dwóch domen, dwóch generatorów `G_AB`, `G_BA` oraz dwóch
dyskryminatorów. Oryginalna implementacja muzyczna także opisuje dwa GAN-y
trenowane wspólnie. Obecny tryb `per_artist` jest auto-mapowaniem domeny i nie
powinien być nazywany CycleGAN-em ani używany jako wynik pracy.

### Algorytm genetyczny

GA jest niezależną metodą, nie modułem uczącym GAN. Osobnik ma cztery parametry
globalne:

- transpozycję;
- skalowanie czasu rozpoczęcia wszystkich nut;
- skalowanie długości wszystkich nut;
- przesunięcie velocity.

Fitness to ujemna odległość od średniego 42-elementowego wektora cech artysty.
Jest to technicznie poprawna optymalizacja, ale przestrzeń transformacji jest za
uboga, by imitować styl kompozytorski: nie zmienia lokalnych interwałów,
motywów, harmonii, prowadzenia głosów ani charakterystycznych figur rytmicznych.

## 3. Twarde dane z dotychczasowego eksperymentu

### Dane treningowe

Manifesty z eksperymentu `custom_conditional_2026-07-29_211645`:

| Artysta | Pliki po walidacji | Unikalne SHA-256 | Mediana długości | Średnie zapełnienie pierwszego okna |
|---|---:|---:|---:|---:|
| Bach | 131 | 130 | 223,9 s | 2,852% |
| Chopin | 136 | 132 | 157,5 s | 2,877% |
| Mozart | 87 | 85 | 247,0 s | 3,366% |

W katalogu danych znajduje się 1426 plików MIDI, lecz akwizytor wyszukuje pliki
tylko płasko. Zagnieżdżone katalogi są pomijane. To nie musi być błędem, ale jest
nieudokumentowaną decyzją o doborze próby.

Korpus miesza różne obsady i gatunki: m.in. utwory fortepianowe, chorały, msze,
suity orkiestrowe i symfonie. Po scaleniu ścieżek model może rozpoznawać rodzaj
obsady, fakturę lub źródło pliku zamiast stylu kompozytora. Chopin jest silnie
związany z fortepianem, podczas gdy dane Bacha i Mozarta są bardziej
zróżnicowane — to klasyczny czynnik zakłócający.

### Trening

Eksperyment `custom_conditional_2026-07-29_214352`:

- 354 pliki, batch 16, 23 iteracje na epokę;
- wykonano 21 pełnych epok z planowanych 200 i dwie iteracje epoki 22;
- zapisano 21 checkpointów po około 235 MB, łącznie około 4,7 GB;
- `loss_g`: 4,3938 → 1,8675;
- `loss_d`: 1,6158 → 0,1495;
- raportowane `val_identity_l1`: 0,0496 → 0,0654.

`val_identity_l1` nie pochodzi z wydzielonego zbioru walidacyjnego. Jest liczona
na tym samym loaderze co trening i dla docelowej etykiety równej oryginalnej.
Nie mierzy powodzenia transferu do innego kompozytora. Sam spadek strat nie jest
dowodem jakości muzycznej ani zmiany stylu.

### Wyniki MIDI

Dziewięć zapisanych wyników ma następujący zakres:

- Bach: 0 nut dla wszystkich trzech wejść;
- Chopin: 29–49 nut, koniec po 14,25 ćwierćnuty;
- Mozart: 72–76 nut, koniec po 15,25 ćwierćnuty.

Limit wynika bezpośrednio z okna `64 steps / 4 steps_per_beat = 16 beats`.
Rozmiar pliku MIDI nie jest miarą długości, lecz tutaj parser potwierdza, że
wyniki rzeczywiście kończą się przed szesnastą ćwierćnutą.

## 4. Przyczyny ciszy, zapętleń i skrócenia

### P0 — jedno okno zamiast datasetu fragmentów

`PianorollDataset` traktuje jeden plik jako jedną próbkę. `from_internal()`
tworzy macierz 64×84 i przycina wszystko poza końcem okna. Model widzi wyłącznie
początek każdego utworu, choć mediany długości źródeł wynoszą kilka minut.

Skutki:

- setki taktów nie uczestniczą w treningu;
- model uczy się charakterystyki początków utworów;
- efektywna liczba próbek to 354, a nie tysiące fraz;
- nie ma losowania ani augmentacji fragmentów;
- inferencja pełnego utworu nie jest zdefiniowana.

Oryginalny projekt CycleGAN dla muzyki dzielił utwory na frazy czterech kolejnych
taktów, filtrował metrum i tworzył osobne próbki 64×84. Aktualny kod przejął
rozmiar macierzy, ale nie przejął krytycznego etapu segmentacji.

### P0 — inferencja obcina pełny utwór

Przepływ `parse → from_internal → G → to_internal` jest wykonywany raz. Writer
nie ma czego „przedłużyć”, bo dostaje tylko macierz jednego okna. Metadane
oryginału pozostają, lecz nie odtwarzają utraconych nut.

### P0 — skrajna rzadkość pianorolla i próg 0,5

Aktywne komórki stanowią średnio około 3% macierzy. Wyjście sigmoidu jest
binaryzowane stałym progiem 0,5. Przy niezrównoważonej reprezentacji model może
minimalizować część strat, produkując głównie zera. Dla innych domen pojedyncze
wysokie aktywacje są zlepiane w powtarzalne pasma nut, co brzmi jak zapętlenie.

Brakuje:

- osobnych kanałów onset/sustain;
- kalibracji progu na walidacji;
- karania pustych wyników i patologicznych długich nut;
- kontroli liczby onsetów, polifonii i aktywności taktów;
- maski dla paddingu i obsługi granic fraz.

### P0 — niejednorodne domeny

Po połączeniu wszystkich ścieżek do jednego binarnego pianorolla tracone są
instrument, kanał i velocity. Jednocześnie liczba głosów i gęstość pozostają,
więc klasyfikator domeny może wykorzystywać obsadę jako skrót. Taki model nie
odpowiada na pytanie o styl kompozytorski.

### P1 — zachowanie długości istnieje tylko w specyfikacji

Wymaganie ±5% i property P11 są opisane w `design.md`, ale odpowiedniego testu
property nie ma. Test jednostkowy używa krótkiego wejścia mieszczącego się w
jednym oknie i generatora tożsamościowego, dlatego nie wykrywa obcięcia.

### P1 — fitness GA miesza nieporównywalne skale

Odległość euklidesowa jest liczona na surowych cechach: tempo około 100–200,
histogramy 0–1, gęstość i długości w innych skalach. Tempo może zdominować
odległość, mimo że genotyp go nie zmienia. Mahalanobis używa pseudoodwrotności,
ale przy skorelowanych histogramach pozostaje wrażliwy i nie zastępuje
standaryzacji ani doboru cech.

`rhythm_density_factor` skaluje wszystkie ticki w zakresie 0,5–2,0, więc sam GA
może skrócić albo podwoić utwór. Jest to sprzeczne z deklarowanym zachowaniem
długości ±5%.

### P1 — konfiguracja nie steruje całym kodem

- CLI treningu nie przekazuje geometrii pianorolla z konfiguracji do trenera;
- akwizytor używa stałych 5 s / 30 min zamiast wartości z konfiguracji;
- `discriminator.patch_size` i część deklarowanych pól nie wpływają na model;
- pełna geometria modelu nie jest zapisywana w metadanych checkpointu;
- CLI nie udostępnia GA ani trybu łączonego, mimo że istnieją w API.

### P2 — porządek repo i reprodukowalność

- katalogi po nieudanych komendach zostają z samą konfiguracją i commitem;
- każdy checkpoint zawiera stany Adam i zajmuje około 235 MB;
- treningowy katalog eksperymentu nie zawiera kopii manifestów;
- trzy pliki zależności podają nie w pełni spójne warianty PyTorch;
- specyfikacja Kiro ma wiele niezakończonych checkpointów, choć duża część kodu
  istnieje;
- istnieje 173-pozycyjny arkusz literatury, lecz tylko 26 lokalnych PDF-ów i
  brak statusu „przeczytane / użyte / odrzucone z powodem”.

## 5. Co mówią testy

Pełny przebieg testów zakończył się wynikiem:

```text
434 passed, 16 warnings in 106.88s
```

To dobry sygnał jakości komponentów, ale testy koncentrują się na kontraktach,
kształtach tensorów, serializacji i deterministyczności. Nie wykrywają głównych
błędów badawczych:

- brak testu pliku dłuższego niż 64 kroki;
- brak weryfikacji, że wszystkie frazy utworu trafiają do datasetu;
- brak realnego P11/P12 dla pełnego pipeline'u;
- brak testu dwóch domen i dwóch kierunków CycleGAN;
- brak kryterium niepustości i muzycznej poprawności wyniku;
- brak niezależnej walidacji zmiany stylu.

## 6. Czy podejście nadaje się do pracy?

### Tak, po zmianie hierarchii celów

Warto zachować:

- symboliczną reprezentację MIDI jako rozsądny kompromis zasobów i
  interpretowalności;
- parser, manifesty i infrastrukturę ewaluacji;
- ekstraktor jako punkt startowy;
- GA jako interpretowalny baseline/metodę główną po rozszerzeniu;
- warunkowany GAN jako eksperyment dodatkowy.

### Nie, jeżeli jedynym dowodem ma być obecny GAN

Aktualny system nie pokazuje, które cechy są stylistyczne, nie rozdziela stylu
od obsady i repertuaru, nie przetwarza całych utworów i nie ma wiarygodnej
walidacji transferu. Dalsze 179 epok na tych samych danych najpewniej utrwali
problem zamiast go naprawić.

## 7. Rekomendowana decyzja architektoniczna

| Wariant | Zgodność z celem | Ryzyko | Rekomendacja |
|---|---:|---:|---|
| cechy + klasyfikator + wielokryterialny GA | bardzo wysoka | niskie/średnie | **rdzeń pracy** |
| naprawiony warunkowany StarGAN | średnia | wysokie | eksperyment dodatkowy |
| wierna reprodukcja muzycznego CycleGAN A↔B | średnia | średnie/wysokie | tylko jako baseline dwóch domen |
| MuseMorphose / Transformer VAE | średnia | wysokie | inspiracja lub wariant, nie szybka podmiana |
| model na surowym audio | niska dla obecnego zakresu | bardzo wysokie | odrzucić |

MuseMorphose obsługuje pełniejsze sekwencje i sterowanie atrybutami takt po takcie,
ale jego oficjalne wagi dotyczą popowego fortepianu oraz atrybutów takich jak
intensywność rytmiczna i polifonia, nie tożsamości kompozytora. Integracja wymaga
nowej tokenizacji, danych i celu treningowego. Nie jest „gotowcem”, który bez
adaptacji rozwiąże tę pracę.

## 8. Źródła użyte do weryfikacji architektury

- [Brunner et al., Symbolic Music Genre Transfer with CycleGAN](https://arxiv.org/abs/1809.07575)
- [oficjalne repozytorium CycleGAN Music Style Transfer](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer)
- [Choi et al., StarGAN, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html)
- [Wu i Yang, MuseMorphose](https://arxiv.org/abs/2105.04090)
- [oficjalna implementacja MuseMorphose](https://github.com/YatingMusic/MuseMorphose)
