# Plan redukcji i uporządkowania literatury

## Stan zastany

- arkusz `Literatura - arkusz.csv`: 173 rekordy;
- katalog `Literatura/`: 26 PDF-ów i plik indeksowy;
- pozycje obejmują jednocześnie teorię muzyki, audio synthesis, MIR, GAN-y,
  transfer symboliczny, generowanie, emocje, biblioteki i luźne projekty;
- znaczna część listy pochodzi z lat 2016–2021, a statusy „do wykorzystania” są
  niepełne;
- występują możliwe duplikaty i pozycje opisane jako „brak dostępu”, „raczej
  nie”, „amatorska” lub niezwiązane z celem.

Nie warto streszczać wszystkich 173 rekordów. Potrzebny jest jawny protokół
screeningu, dzięki któremu redukcja będzie obroniona metodologicznie.

## Pytania, którym ma służyć literatura

1. Jak definiuje się i mierzy styl kompozytorski w danych symbolicznych?
2. Które cechy pozwalają rozróżniać kompozytorów i jak je walidować?
3. Jak reprezentować MIDI bez mylenia stylu kompozytora z wykonaniem i obsadą?
4. Jak realizuje się transfer stylu przy zachowaniu melodii/treści?
5. Jak ewaluować jednocześnie zmianę stylu, zachowanie treści i muzyczność?
6. Jaką rolę może pełnić optymalizacja ewolucyjna?

Każda pozycja pozostawiona w rdzeniu powinna odpowiadać przynajmniej na jedno z
tych pytań.

## Kryteria włączenia

Priorytet otrzymują:

- prace recenzowane lub oficjalne preprinty z kodem/danymi;
- symboliczne dane muzyczne (MIDI, MusicXML, Kern), nie surowe audio;
- klasyfikacja kompozytorów, stylometria muzyczna, interpretowalne cechy;
- transfer stylu, który mierzy zachowanie treści;
- prace z jawnym splitem, baseline'ami i metrykami;
- narzędzia rzeczywiście rozważane w implementacji.

## Kryteria odrzucenia lub przeniesienia do tła

Do archiwum `C — poza zakresem` przenieść:

- syntezę surowego audio bez związku z cechami kompozytorskimi: WaveNet,
  MelNet, WaveFlow, DiffWave, Jukebox i podobne;
- transfer barwy/timbre, miksowanie i odpowiedzi impulsowe;
- pozycje bez pełnego tekstu, jeśli nie wnoszą unikalnej tezy;
- blogi, prezentacje i amatorskie projekty zastępowane publikacją pierwotną;
- powtórzenia tej samej pracy/preprintu/dema;
- ogólne przeglądy GAN, jeśli wystarczy jedno źródło podstawowe;
- prace o generowaniu, które nie dotyczą ani stylu, ani ewaluacji potrzebnej w
  eksperymencie.

Nie trzeba usuwać rekordów fizycznie. Wystarczy nadać im status i nie cytować w
głównym przeglądzie.

## Proponowana literatura rdzeniowa — warstwa A

### A1. Definicja i identyfikacja stylu

1. [Simonetta, Style-based Composer Identification and Attribution of Symbolic Music Scores: a Systematic Survey (2025)](https://doi.org/10.5334/tismir.240)
   — najważniejszy aktualny punkt startowy; porządkuje cechy, reprezentacje,
   protokoły walidacji i ryzyka autorstwa.
2. [Zhang et al., Symbolic Music Representations for Classification Tasks (ISMIR 2023)](https://archives.ismir.net/ismir2023/paper/000101.pdf)
   — porównuje pianoroll, sekwencje i grafy m.in. w klasyfikacji kompozytora.
3. [Kim et al., Deep Composer Classification Using Symbolic Representation](https://arxiv.org/abs/2010.00823)
   — użyteczny baseline onset + sustain i przykład ewaluacji kompozytora.
4. `shalit13.pdf` — *Modeling Musical Influence with Topic Models*; pozostawić,
   jeśli w tekście zostanie wykorzystane rozróżnienie wpływu i stylu.

### A2. Ekstrakcja cech

5. [Llorens et al., musif: a Python package for symbolic music feature extraction](https://arxiv.org/abs/2307.01120)
6. [Simonetta et al., Optimizing Feature Extraction for Symbolic Music](https://archives.ismir.net/ismir2023/paper/000095.pdf)
   — porównanie narzędzi i argument za rozszerzeniem obecnych 42 cech.
7. `01-PodstawyNotacji.pdf` lub `zm.pdf` — wybrać **jedno** podstawowe źródło
   terminologii muzycznej, nie oba jako równorzędne filary.

### A3. Transfer stylu

8. `1809.07575.pdf` / [Brunner et al., Symbolic Music Genre Transfer with CycleGAN](https://arxiv.org/abs/1809.07575)
   — główny punkt odniesienia dla obecnego pianorolla; koniecznie opisać pełne
   preprocessing i ograniczenia gatunek≠kompozytor.
9. [Choi et al., StarGAN (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html)
   — źródło architektury wielodomenowej; zaznaczyć, że dotyczy obrazów.
10. [Wu i Yang, MuseMorphose](https://arxiv.org/abs/2105.04090)
    — dłuższa forma i sterowanie atrybutami na poziomie taktów; ważny kontrast
    wobec stałego okna GAN.
11. *MIDI-VAE: Modeling Dynamics and Instrumentation of Music with Applications
    to Style Transfer* — pozostawić, jeśli dynamika/instrumentacja wejdą do
    zakresu; inaczej warstwa B.

### A4. Optymalizacja ewolucyjna

12. `evolutionary-music-composition-system-with-statistically-modeled-criteria_1798.pdf`
    — obowiązkowe źródło wskazane w formalnej literaturze pracy.
13. *Musical Style Modification as an Optimization Problem* — priorytet do
    pozyskania pełnego tekstu i porównania funkcji celu.
14. *Classification and Generation of Composer-Specific Music Using Global
    Feature Models and Variable Neighborhood Search* — bezpośrednio łączy
    cechy globalne, klasyfikację i optymalizację.

### A5. Dane i tokenizacja

15. [MAESTRO](https://magenta.withgoogle.com/datasets/maestro) — fortepianowe
    MIDI z kompozytorem i oficjalnymi splitami.
16. [GiantMIDI-Piano](https://doi.org/10.5334/tismir.80) — duży korpus piano;
    omówić automatyczną transkrypcję jako źródło błędu.
17. [MidiTok](https://miditok.readthedocs.io/en/latest/) — tylko jeśli zostanie
    wybrany wariant sekwencyjny/Transformer.

## Warstwa B — tło, maksymalnie kilka cytowań na temat

- `1709.01620.pdf`, *Deep Learning Techniques for Music Generation – A Survey*;
- jedna aktualniejsza praca przeglądowa o generowaniu symbolicznym;
- Goodfellow, Bengio, Courville — podstawy uczenia głębokiego/GAN;
- ogólna praca o ewaluacji modeli generatywnych w muzyce;
- jedna praca o VAE, jedna o Transformerach i jedna o GAN-ach, jeśli są potrzebne
  do porównania paradygmatów;
- `thesis.pdf`, *Probabilistic Models for Music*, tylko dla potrzebnego tła;
- etyka/prawa autorskie jako krótka osobna sekcja, nie rdzeń techniczny.

## Warstwa C — archiwum

Z lokalnych PDF-ów do tej warstwy najpewniej trafią:

- `1609.03499.pdf` — WaveNet;
- `1906.01083.pdf` — MelNet;
- `1912.01219.pdf` — WaveFlow;
- `2003.02836.pdf` — GAN dla audio;
- `2009.09761.pdf` — DiffWave;
- `jukebox.pdf`;
- prace o MFCC, impulse response, time-scale modification i emocjach, o ile
  końcowy zakres pozostaje wyłącznie symbolicznym stylem kompozytorskim.

To wartościowe publikacje, lecz nie odpowiadają bezpośrednio na pytania pracy.

## Nowy schemat arkusza

Obecne kolumny warto zastąpić lub uzupełnić o:

| Kolumna | Dozwolone wartości / opis |
|---|---|
| `id` | stabilny klucz, np. DOI/arXiv/slug |
| `citation` | pełny opis bibliograficzny |
| `year` | rok publikacji |
| `peer_reviewed` | tak/nie |
| `modality` | symbolic/audio/score/image |
| `task` | identification/features/transfer/generation/evaluation/data |
| `method` | handcrafted/CNN/GAN/VAE/Transformer/evolutionary |
| `dataset` | użyte dane i liczność |
| `evaluation` | split, metryki, odsłuch |
| `code_data` | link i status dostępności |
| `relevance` | A-core/B-background/C-out |
| `status` | inbox/screened/read/cited/rejected |
| `one_sentence_claim` | jedna teza używana w pracy |
| `limitations` | najważniejsze ograniczenie |
| `thesis_section` | docelowy rozdział |
| `reject_reason` | wymagane dla C |

## Procedura redukcji

1. Deduplikacja po DOI, arXiv i tytule.
2. Screening tytułu/abstraktu według sześciu pytań badawczych.
3. Nadanie A/B/C i powodu dla C.
4. Pełne czytanie tylko A; dla B czytać sekcje potrzebne do konkretnego akapitu.
5. Dla każdej A uzupełnić jedną tezę, metodę, dane, wynik i ograniczenie.
6. Zbudować macierz porównawczą metod, zamiast szeregu niezależnych streszczeń.
7. Po napisaniu rozdziału usunąć cytowania, które nie wspierają żadnego zdania.

Docelowo rdzeń może liczyć około 15–25 publikacji, a cała bibliografia pracy
około 30–45 pozycji. To nie jest sztywny limit; ważniejsze jest, by każda
pozycja pełniła określoną funkcję.

## Proponowany układ przeglądu

1. Styl kompozytorski i jego operacjonalizacja.
2. Reprezentacje symboliczne: score, performance MIDI, pianoroll, tokeny.
3. Cechy stylometryczne i klasyfikacja kompozytora.
4. Transfer stylu: optymalizacja, VAE/Transformer, GAN/CycleGAN.
5. Zachowanie treści i ewaluacja muzyczna.
6. Luka badawcza i uzasadnienie wybranej metody.

Z przeglądu powinien logicznie wynikać wybór: jednorodne MIDI fortepianowe,
walidowane cechy, wielokryterialny GA jako metoda główna i warunkowany GAN jako
porównanie opcjonalne.
