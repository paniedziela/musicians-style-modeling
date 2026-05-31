# Requirements Document

## Introduction

Niniejszy dokument specyfikuje wymagania dla pracy inżynierskiej zatytułowanej *Modelowanie stylu artystycznego muzyków* (ang. *Musicians artistic style modeling*). Praca obejmuje dwa równoległe artefakty: dokument dyplomowy (część teoretyczna i opis eksperymentów) oraz oprogramowanie badawcze realizujące transfer stylu artystycznego pomiędzy utworami zapisanymi w formacie MIDI.

System badawczy (dalej zwany *Systemem*) przyjmuje na wejściu zbiór plików MIDI reprezentujących twórczość artysty docelowego (źródło stylu) oraz pojedynczy plik MIDI utworu wejściowego (treść), a na wyjściu generuje nowy plik MIDI utworu wejściowego ze zmodyfikowanymi cechami, tak aby przypominał styl artysty docelowego. System składa się z trzech głównych komponentów: *Ekstraktora_Cech* (analiza cech muzycznych zbioru), *Modelu_GAN* (generatywna sieć neuronowa typu CycleGAN trenowana na danych MIDI artysty) oraz *Algorytmu_Genetycznego* (alternatywna lub uzupełniająca metoda transformacji MIDI optymalizująca cechy stylu). Dobór reprezentacji symbolicznej (MIDI), a nie surowego audio, wynika z potrzeby kompromisu pomiędzy zasobami obliczeniowymi dostępnymi w warunkach pracy inżynierskiej a jakością merytoryczną wyników.

Niniejsze wymagania obejmują zarówno funkcjonalność oprogramowania, jak i wymagania formalne dotyczące samego dokumentu dyplomowego, a także wymagania związane z walidacją eksperymentalną oraz właściwości poprawnościowe (correctness properties) możliwe do weryfikacji metodą testowania opartego na własnościach (PBT).

Nagłówki strukturalne (Requirements Document, Introduction, Glossary, Requirements, Requirement N, User Story) zostały zapisane w języku angielskim ze względu na wymagania szablonu narzędzia specyfikacji, natomiast treść merytoryczna (definicje, historie użytkownika, kryteria akceptacji) jest prowadzona w języku polskim zgodnie z przyjętymi zasadami redakcyjnymi.

## Glossary

- **System**: kompletne oprogramowanie badawcze realizujące transfer stylu artystycznego pomiędzy plikami MIDI (zbiorcza nazwa wszystkich komponentów).
- **Plik_MIDI**: plik w formacie Standard MIDI File (SMF) w wersji 0 lub 1, zgodny ze specyfikacją MIDI 1.0, zawierający co najmniej jedną ścieżkę z komunikatami *Note On* oraz *Note Off*.
- **Zbiór_Stylu**: kolekcja plików MIDI jednego artysty docelowego, wykorzystywana jako źródło stylu (uczenie modelu i ekstrakcja cech statystycznych).
- **Utwór_Wejściowy**: pojedynczy plik MIDI dostarczony przez użytkownika, na który ma zostać nałożony styl artysty docelowego.
- **Utwór_Wyjściowy**: plik MIDI wygenerowany przez System, będący transformacją *Utworu_Wejściowego* w kierunku stylu *Zbioru_Stylu*.
- **Ekstraktor_Cech**: komponent Systemu odpowiedzialny za obliczanie wektora cech muzycznych pliku MIDI (tempo, tonacja, rozkład wysokości dźwięków, rozkład interwałów, gęstość rytmiczna, statystyki harmoniczne).
- **Wektor_Cech**: wielowymiarowa reprezentacja liczbowa pliku MIDI, składająca się ze zdefiniowanych cech muzycznych obliczonych przez *Ekstraktor_Cech*.
- **Model_GAN**: generatywna sieć przeciwstawna realizująca transfer stylu na danych symbolicznych, zaadaptowana z pracy *Symbolic Music Genre Transfer with CycleGAN* (arXiv:1809.07575) i pokrewnych rozwiązań wielodomenowych typu StarGAN. Może być realizowana w jednym z dwóch trybów: jako *Model_GAN_Warunkowany* (tryb domyślny) albo jako *Model_GAN_Per_Artysta* (tryb fallback). Operuje na pianorollach lub macierzach symbolicznych pochodzących z plików MIDI.
- **Model_GAN_Warunkowany**: rozszerzenie *Modelu_GAN* w postaci sieci wielodomenowej typu StarGAN, w której pojedynczy generator G(x, c) przekształca *Utwór_Wejściowy* x w stronę stylu artysty wskazanego *Etykietą_Artysty* c. Domyślny tryb pracy Systemu, pozwalający obsłużyć N artystów jednym modelem bez przełączania *Punktów_Kontrolnych*.
- **Model_GAN_Per_Artysta**: tryb fallback Modelu_GAN w postaci osobnej instancji klasycznego CycleGAN (transfer A↔B) wytrenowanej dla pojedynczego artysty docelowego względem zbioru kontrolnego. Stosowany, gdy zasoby obliczeniowe nie pozwalają na trening warunkowany lub gdy użytkownik wymusza ten tryb w pliku konfiguracyjnym.
- **Etykieta_Artysty**: wektor identyfikujący artystę docelowego (kodowanie one-hot lub embedding) podawany jako warunek do generatora *Modelu_GAN_Warunkowanego*. Pozwala wybrać artystę docelowego inferencji bez przełączania modelu.
- **Zbiór_Wielo_Artystyczny**: kolekcja *Zbiorów_Stylu* dla N artystów (N ≥ 2) wraz z towarzyszącym mapowaniem artysta → *Etykieta_Artysty*, używana do treningu *Modelu_GAN_Warunkowanego*.
- **Algorytm_Genetyczny**: ewolucyjny algorytm optymalizujący transformację *Utworu_Wejściowego* na podstawie funkcji dopasowania bazującej na cechach statystycznych *Zbioru_Stylu* (zgodnie z metodyką pracy *Evolutionary music composition system with statistically modeled criteria*, Kowalczuk et al., 2017). Algorytm jest z natury ogólny - artysta docelowy jest wskazywany przez wybór *Zbioru_Stylu* podawanego jako parametr i nie wymaga osobnego treningu.
- **Pianoroll**: dwuwymiarowa macierz binarna lub liczbowa o wymiarach (czas × wysokość dźwięku), reprezentująca aktywność nut w dyskretnych krokach czasowych.
- **Funkcja_Dopasowania**: funkcja oceny *Algorytmu_Genetycznego*, mierząca odległość *Wektora_Cech* osobnika od średniego *Wektora_Cech* *Zbioru_Stylu*.
- **Seed**: liczba całkowita inicjalizująca generatory liczb pseudolosowych w komponentach Systemu.
- **Test_Odsłuchowy**: subiektywna ocena percepcyjna wykonana przez ludzkich słuchaczy w formie ankiety MOS (*Mean Opinion Score*) lub testu ABX.
- **Próba_Statystyczna**: zbiór wyników eksperymentalnych analizowany testami statystycznymi (test t-Studenta, test Wilcoxona) w celu weryfikacji istotności różnic.
- **Dokument_Dyplomowy**: tekst pracy inżynierskiej w formacie LaTeX (wybrana klasa dokumentu) wraz z bibliografią, rysunkami i tabelami.
- **Pipeline_Treningu**: zautomatyzowany proces obejmujący wczytanie *Zbioru_Stylu*, ekstrakcję cech, trening *Modelu_GAN* i zapis punktów kontrolnych modelu.
- **Punkt_Kontrolny**: zapisany na dysku stan parametrów *Modelu_GAN* w trakcie treningu, umożliwiający wznowienie treningu lub późniejszą inferencję.
- **GPU_Konsumencki**: pojedynczy procesor graficzny klasy konsumenckiej z pamięcią VRAM nie mniejszą niż 6 GB (np. NVIDIA GeForce GTX 1060 6 GB lub nowszy).

## Requirements

### Requirement 1: Pozyskiwanie i przygotowanie zbioru uczącego

**User Story:** Jako autor pracy inżynierskiej, chcę zebrać zbiór plików MIDI dla wybranych artystów, aby uzyskać reprezentatywne dane do treningu modelu i ekstrakcji cech stylu.

#### Acceptance Criteria

1. THE System SHALL akceptować na wejściu katalog zawierający co najmniej 30 plików MIDI dla jednego artysty docelowego.
2. WHEN katalog zawiera mniej niż 30 plików MIDI, THE System SHALL zwrócić ostrzeżenie wskazujące minimalną wymaganą liczność zbioru i kontynuować przetwarzanie.
3. WHEN plik wejściowy nie jest zgodny ze specyfikacją Standard MIDI File, THE System SHALL pominąć ten plik, zapisać wpis w logu z nazwą pliku i przyczyną odrzucenia oraz kontynuować przetwarzanie pozostałych plików.
4. THE System SHALL akceptować pliki MIDI o czasie trwania w przedziale [5 sekund, 30 minut] oraz odrzucać jako niepoprawne pliki o czasie trwania krótszym niż 5 sekund lub dłuższym niż 30 minut, z zapisem wpisu w logu zawierającego nazwę pliku i przyczynę odrzucenia.
5. IF po zakończeniu walidacji wszystkich plików wejściowych liczba poprawnych plików MIDI wynosi zero - niezależnie od przyczyny (brak plików w katalogu, odrzucenie z powodu niezgodności ze specyfikacją, odrzucenie z powodu przekroczenia dopuszczalnego zakresu długości lub dowolne inne kryterium filtrowania) - THEN THE System SHALL zakończyć działanie z niezerowym kodem błędu i komunikatem opisującym przyczynę.
6. WHERE użytkownik dostarcza listę identyfikatorów utworów z serwisu YouTube, THE System SHALL pobierać fragmenty audio o długości nie większej niż 15 sekund każdy, zgodnie z przyjętym protokołem projektu.
7. THE System SHALL zapisywać metadane *Zbioru_Stylu* (lista plików, sumy kontrolne SHA-256, źródło, długość, liczba ścieżek) w pliku manifestu w formacie JSON.

### Requirement 2: Ekstrakcja cech muzycznych

**User Story:** Jako badacz, chcę wyodrębnić mierzalne cechy muzyczne z plików MIDI artysty, aby móc statystycznie scharakteryzować jego styl i porównać artystów między sobą.

#### Acceptance Criteria

1. THE Ekstraktor_Cech SHALL obliczać dla każdego pliku MIDI co najmniej następujące cechy: tempo średnie (BPM, wartość ściśle dodatnia), tonacja dominująca (skala krzyżyków lub bemoli), histogram klas wysokości dźwięków (12 kategorii), histogram interwałów melodycznych, gęstość nut na sekundę, średnia długość nuty, odchylenie standardowe długości nuty oraz proporcja pauz w przedziale [0,0; 1,0].
2. WHEN plik MIDI nie zawiera informacji o tempie, THE Ekstraktor_Cech SHALL przyjąć tempo domyślne 120 BPM zgodnie ze specyfikacją MIDI i odnotować ten fakt w logu.
3. THE Ekstraktor_Cech SHALL zwracać *Wektor_Cech* o stałej długości niezależnie od długości pliku wejściowego.
4. THE Ekstraktor_Cech SHALL produkować deterministyczne wyniki, to znaczy wielokrotne uruchomienie dla tego samego pliku wejściowego SHALL zwracać identyczne wartości cech.
5. THE Ekstraktor_Cech SHALL przetwarzać pojedynczy plik MIDI o długości 5 minut w czasie nie większym niż 10 sekund na pojedynczym rdzeniu CPU klasy konsumenckiej.
6. WHEN ekstrakcja cech kończy się dla całego *Zbioru_Stylu*, THE System SHALL obliczać i zapisywać agregowany wektor statystyczny zbioru (średnia, mediana, odchylenie standardowe każdej cechy).
7. IF plik MIDI zawiera wyłącznie pauzy lub jest pusty, THEN THE Ekstraktor_Cech SHALL zwrócić *Wektor_Cech* wypełniony wartościami neutralnymi zdefiniowanymi w specyfikacji projektu i odnotować ten fakt w logu.
8. IF plik MIDI zawiera dane nutowe, ale ekstrakcja nie produkuje sensownego *Wektora_Cech* z powodów technicznych (degenerowane wartości liczbowe, niespójna struktura wewnętrzna), THEN THE Ekstraktor_Cech SHALL zapisać w logu nazwę pliku, opis problemu oraz częściowe wartości obliczone do momentu wystąpienia problemu.

### Requirement 3: Trening modelu generatywnego (GAN)

**User Story:** Jako autor pracy, chcę wytrenować sieć generatywną typu CycleGAN/StarGAN na danych MIDI artystów docelowych, aby uzyskać model zdolny do transferu stylu pomiędzy utworami w reprezentacji symbolicznej, z możliwością obsługi wielu artystów jednym wytrenowanym modelem.

#### Acceptance Criteria

1. THE Model_GAN SHALL być implementowany jako sieć wielodomenowa typu StarGAN (*Model_GAN_Warunkowany*) w trybie domyślnym albo jako sieć typu CycleGAN (*Model_GAN_Per_Artysta*) w trybie fallback, operująca na pianorollach o wymiarach zdefiniowanych w specyfikacji projektu (długość okna czasowego, zakres wysokości dźwięków).
2. THE Pipeline_Treningu SHALL uczyć *Model_GAN* na *Zbiorze_Stylu* (tryb per-artysta) lub *Zbiorze_Wielo_Artystycznym* (tryb warunkowany) z wykorzystaniem pojedynczego *GPU_Konsumenckiego* lub procesora CPU.
3. WHEN trening jest uruchamiany na GPU z 6 GB pamięci VRAM, THE Pipeline_Treningu SHALL kończyć jedną epokę dla zbioru o liczności 100 plików MIDI w czasie nie większym niż 30 minut.
4. THE Pipeline_Treningu SHALL zapisywać *Punkt_Kontrolny* po każdej epoce treningu, zawierający parametry generatora, dyskryminatora, optymalizatora oraz numer epoki.
5. WHEN trening jest wznawiany z istniejącego *Punktu_Kontrolnego*, THE Pipeline_Treningu SHALL kontynuować naukę od zapisanego stanu bez ponownej inicjalizacji parametrów.
6. THE Pipeline_Treningu SHALL akceptować *Seed* jako parametr wejściowy i wykorzystywać go do inicjalizacji generatorów liczb pseudolosowych w bibliotece numerycznej (NumPy) i frameworku uczenia głębokiego (PyTorch).
7. WHEN ten sam *Seed* oraz ta sama wersja *Zbioru_Stylu* są użyte, THE Pipeline_Treningu SHALL produkować *Punkty_Kontrolne* o parametrach zgodnych z dokładnością do tolerancji numerycznej zdefiniowanej w specyfikacji (różnica średniego błędu kwadratowego nie większa niż 1e-5).
8. THE Pipeline_Treningu SHALL zapisywać do logu wartości funkcji straty generatora i dyskryminatora w każdej iteracji oraz metryki walidacyjne po każdej epoce.
9. IF trening napotka błąd niewystarczającej pamięci GPU, THEN THE Pipeline_Treningu SHALL zwrócić komunikat błędu wskazujący wymaganą redukcję rozmiaru wsadu (batch size) i zakończyć działanie z niezerowym kodem wyjścia.
10. THE Pipeline_Treningu SHALL przyjmować z pliku konfiguracyjnego tryb pracy modelu (`conditional` jako wartość domyślna lub `per_artist` jako fallback) i odpowiednio konfigurować architekturę sieci oraz pętlę treningową.
11. WHEN tryb warunkowany jest aktywny ORAZ liczba artystów w *Zbiorze_Wielo_Artystycznym* wynosi N, THE Pipeline_Treningu SHALL wymagać dla każdego artysty co najmniej 30 plików MIDI (zgodnie z Wymaganiem 1, kryterium 1) i zwracać ostrzeżenie wyłącznie dla tych artystów, dla których liczba plików jest faktycznie niewystarczająca (mniejsza niż 30); jeśli wszyscy artyści w zbiorze posiadają wystarczającą liczbę plików, żadne ostrzeżenie nie SHALL być zwracane. Trening SHALL być kontynuowany również z artystami, dla których wystąpiło ostrzeżenie.
12. WHEN tryb warunkowany jest aktywny, THE Model_GAN_Warunkowany SHALL stosować stratę klasyfikacji domeny (domain classification loss) zgodnie z architekturą StarGAN obok strat adversarial i cycle-consistency.
13. WHEN użytkownik wybiera tryb per-artysta w pliku konfiguracyjnym, THE Pipeline_Treningu SHALL trenować osobny CycleGAN dla wskazanego artysty względem zbioru kontrolnego (zbiór ogólny lub inny artysta wskazany w polu `model.reference_artist` pliku konfiguracyjnego).
14. THE Pipeline_Treningu SHALL zapisywać w metadanych każdego *Punktu_Kontrolnego* typ wytrenowanego modelu (wartość `conditional` lub `per_artist`) oraz listę obsługiwanych *Etykiet_Artysty* wraz z mapowaniem etykieta → identyfikator artysty.

### Requirement 4: Działanie algorytmu genetycznego

**User Story:** Jako badacz, chcę dysponować algorytmem genetycznym jako alternatywną metodą transferu stylu, aby uzyskać kompromis pomiędzy nakładem zasobów obliczeniowych a jakością wyniku, oraz aby porównać jego efektywność z modelem GAN.

#### Acceptance Criteria

1. THE Algorytm_Genetyczny SHALL operować na populacji osobników reprezentujących transformacje *Utworu_Wejściowego* zakodowane jako wektory parametrów rzeczywistych (transpozycja w półtonach, modyfikacja gęstości rytmicznej, modyfikacja długości nut), przy czym poszczególne parametry mogą przyjmować wartości ujemne reprezentujące transformacje odwrotne (np. transpozycja w dół, redukcja gęstości rytmicznej, skrócenie długości nut).
2. THE Algorytm_Genetyczny SHALL stosować operatory genetyczne: selekcję turniejową, krzyżowanie jednopunktowe lub jednorodne, mutację gaussowską parametrów rzeczywistych.
3. THE Funkcja_Dopasowania SHALL obliczać odległość euklidesową lub odległość Mahalanobisa pomiędzy *Wektorem_Cech* osobnika a agregowanym wektorem statystycznym *Zbioru_Stylu*.
4. WHEN użytkownik podaje *Seed*, THE Algorytm_Genetyczny SHALL produkować deterministyczny przebieg ewolucji (identyczna populacja końcowa dla tych samych danych wejściowych i tej samej liczby pokoleń).
5. THE Algorytm_Genetyczny SHALL kończyć działanie po osiągnięciu zdefiniowanej liczby pokoleń lub po stagnacji wartości najlepszego dopasowania przez liczbę pokoleń określoną w konfiguracji.
6. THE Algorytm_Genetyczny SHALL zapisywać historię wartości funkcji dopasowania (najlepsza, średnia, najgorsza) dla każdego pokolenia w pliku logu.
7. WHILE algorytm jest uruchomiony w trybie elitaryzmu, THE Algorytm_Genetyczny SHALL zachowywać monotoniczność najlepszej wartości funkcji dopasowania (najlepszy osobnik kolejnego pokolenia SHALL mieć dopasowanie nie gorsze niż najlepszy osobnik pokolenia poprzedniego).

### Requirement 5: Transfer stylu i parser MIDI

**User Story:** Jako użytkownik końcowy, chcę móc nałożyć styl wybranego artysty na dowolny utwór wejściowy i otrzymać wynikowy plik MIDI, aby ocenić jakość transferu i wykorzystać wynik do dalszych eksperymentów.

#### Acceptance Criteria

1. THE System SHALL przyjmować jako wejście inferencji wytrenowany *Punkt_Kontrolny* lub konfigurację *Algorytmu_Genetycznego* lub oba te artefakty równocześnie, *Utwór_Wejściowy* w formacie MIDI oraz opcjonalny *Seed*; gdy oba artefakty są dostępne, THE System SHALL umożliwiać ich łączne wykorzystanie lub automatyczny wybór jednego z nich na podstawie konfiguracji.
2. THE System SHALL produkować *Utwór_Wyjściowy* w formacie Standard MIDI File zgodny ze specyfikacją MIDI 1.0, dający się otworzyć w sekwencerach standardu (MuseScore, Ableton Live, FL Studio).
3. THE System SHALL zawierać parser MIDI przekształcający plik MIDI na wewnętrzną reprezentację (pianoroll lub lista zdarzeń) oraz pretty printer odwrotnie generujący plik MIDI z reprezentacji wewnętrznej.
4. WHEN parser otrzymuje poprawny plik MIDI, a następnie pretty printer zapisuje plik wynikowy i parser ponownie wczytuje plik wynikowy, THE System SHALL produkować reprezentację wewnętrzną semantycznie równoważną oryginalnej (ta sama lista zdarzeń nutowych z dokładnością do kolejności zdarzeń o identycznym znaczniku czasu).
5. WHEN parser otrzymuje plik wejściowy, THE System SHALL wykonać walidację struktury pliku przed jakimkolwiek przetwarzaniem treści (w tym przed obliczaniem długości czasowej) i w przypadku wykrycia uszkodzenia zwrócić błąd zawierający opis miejsca i typu uszkodzenia bez zgłaszania nieobsłużonego wyjątku; IF charakter uszkodzenia uniemożliwia precyzyjne wskazanie miejsca lub typu, THEN THE System SHALL zwrócić ogólny opis błędu (np. "wykryto uszkodzenie struktury pliku") tak, aby każde wystąpienie błędu walidacji było opatrzone niepustym opisem.
6. THE System SHALL zachowywać w *Utworze_Wyjściowym* tę samą długość czasową co *Utwór_Wejściowy* z tolerancją nie większą niż ±5%.
7. THE System SHALL zachowywać w *Utworze_Wyjściowym* metryczną strukturę taktów *Utworu_Wejściowego* (liczba taktów, oznaczenie metrum), o ile konfiguracja inferencji nie wskazuje inaczej.
8. WHEN użytkownik uruchamia inferencję, THE System SHALL wymagać obowiązkowego parametru `target_artist` niezależnie od trybu zapisanego *Punktu_Kontrolnego* oraz walidować wartość tego parametru względem metadanych *Punktu_Kontrolnego*.
9. IF wartość parametru `target_artist` nie znajduje się na liście *Etykiet_Artysty* zapisanej w metadanych *Punktu_Kontrolnego*, THEN THE System SHALL zwrócić błąd zawierający listę dostępnych *Etykiet_Artysty* i zakończyć działanie z niezerowym kodem wyjścia, bez wykonywania inferencji.
10. WHEN inferencja jest uruchamiana z *Punktem_Kontrolnym* zapisanym w trybie per-artysta (`per_artist`), THE System SHALL walidować, że wartość parametru `target_artist` odpowiada artyście zapisanemu w metadanych *Punktu_Kontrolnego*; IF wartości się nie zgadzają, THEN THE System SHALL zwrócić błąd i zakończyć działanie z niezerowym kodem wyjścia.
11. THE System SHALL akceptować *Algorytm_Genetyczny* jako alternatywną metodę inferencji niezależnie od trybu *Modelu_GAN*, przyjmując *Zbiór_Stylu* artysty docelowego jako parametr i nie wymagając osobnego treningu modelu generatywnego.

### Requirement 6: Ewaluacja obiektywna

**User Story:** Jako autor pracy, chcę dysponować obiektywnymi metrykami porównującymi styl wygenerowany przez System ze stylem artysty docelowego, aby ilościowo ocenić skuteczność transferu i przedstawić wyniki w pracy.

#### Acceptance Criteria

1. THE System SHALL obliczać dla każdego *Utworu_Wyjściowego* odległość *Wektora_Cech* od agregowanego wektora *Zbioru_Stylu* artysty docelowego.
2. THE System SHALL obliczać dla każdego *Utworu_Wyjściowego* odległość *Wektora_Cech* od *Wektora_Cech* *Utworu_Wejściowego*, w celu kontroli zachowania struktury źródłowej.
3. WHEN ewaluacja kończy się dla wszystkich utworów testowych ORAZ liczba poprawnych pomiarów odległości jest nie mniejsza niż 10 par (≥ 10) (przed transferem, po transferze), THE System SHALL przeprowadzać test statystyczny (test t-Studenta dla prób zależnych lub test rang Wilcoxona) porównujący odległości od stylu docelowego przed i po transferze, z poziomem istotności α równym 0,05; THE System SHALL pomijać raportowanie statystyk opisowych w tej sytuacji.
4. IF liczba poprawnych pomiarów odległości jest mniejsza niż 10 par (< 10), THEN THE System SHALL pominąć test statystyczny, zapisać w logu ostrzeżenie o niewystarczającej liczności próby i zaraportować jedynie statystyki opisowe.
5. THE System SHALL raportować wartości p uzyskane w testach statystycznych w pliku wyjściowym ewaluacji.
6. THE System SHALL generować wykresy porównawcze (histogramy, wykresy pudełkowe) cech *Utworu_Wejściowego*, *Utworu_Wyjściowego* oraz *Zbioru_Stylu* w formacie umożliwiającym osadzenie w *Dokumencie_Dyplomowym* (PDF lub EPS).

### Requirement 7: Ewaluacja subiektywna (testy odsłuchowe)

**User Story:** Jako oceniający (komisja, słuchacze ankiety), chcę móc subiektywnie ocenić, czy *Utwór_Wyjściowy* jest rozpoznawalny jako utwór w stylu artysty docelowego, aby zweryfikować jakość systemu w sposób uzupełniający metryki obiektywne.

#### Acceptance Criteria

1. THE System SHALL przygotowywać zestaw próbek odsłuchowych zawierający co najmniej 10 par (*Utwór_Wejściowy*, *Utwór_Wyjściowy*) wraz z fragmentami referencyjnymi *Zbioru_Stylu*.
2. THE System SHALL umożliwiać eksport próbek odsłuchowych do formatu audio (WAV, 44,1 kHz, 16 bit) z wykorzystaniem zewnętrznego syntezatora MIDI o ujednoliconej konfiguracji (FluidSynth z określonym SoundFontem).
3. WHERE próbka odsłuchowa pochodzi z chronionego prawem autorskim *Zbioru_Stylu*, THE System SHALL ograniczać długość próbki do 15 sekund zgodnie z przyjętym protokołem projektu.
4. THE System SHALL przygotowywać formularz ankietowy (test ABX lub MOS w skali 1–5) umożliwiający respondentom ocenę podobieństwa stylistycznego oraz jakości muzycznej *Utworu_Wyjściowego*; statystyki obliczane z odpowiedzi (np. średnia, odchylenie standardowe, przedziały ufności) NIE muszą mieścić się w przedziale [1; 5] i SHALL być raportowane bez przycinania do granic skali.
5. WHEN ankieta zostanie wypełniona przez liczbę respondentów nie mniejszą niż 20 (≥ 20), THE System SHALL przeprowadzić analizę statystyczną wyników (średnia, odchylenie standardowe, test istotności).

### Requirement 8: Reprodukowalność i logowanie

**User Story:** Jako autor pracy oraz jako recenzent, chcę móc odtworzyć eksperymenty opisane w pracy, aby zweryfikować poprawność wyników.

#### Acceptance Criteria

1. THE System SHALL przyjmować plik konfiguracyjny zarówno w formacie YAML, jak i w formacie JSON, definiujący wszystkie parametry eksperymentu (parametry sieci, hiperparametry treningu, parametry algorytmu genetycznego, *Seed*).
2. THE System SHALL zapisywać dla każdego eksperymentu kopię użytego pliku konfiguracyjnego oraz wersję kodu (commit hash repozytorium Git) w katalogu wynikowym.
3. WHEN dwa uruchomienia używają identycznego pliku konfiguracyjnego, identycznego *Zbioru_Stylu* i identycznego *Seed*, THE System SHALL produkować *Utwory_Wyjściowe* równoważne z dokładnością do tolerancji numerycznej zdefiniowanej w specyfikacji.
4. THE System SHALL zapisywać logi wykonania w formacie strukturalnym (JSON lines lub CSV) zawierające znaczniki czasowe, poziom logowania i komunikat.
5. THE System SHALL udostępniać plik *requirements.txt* lub *environment.yml* z dokładnymi wersjami wszystkich zależności bibliotek Python.
6. THE plik konfiguracyjny SHALL zawierać sekcję `model.mode` przyjmującą wartości ze zbioru {`conditional`, `per_artist`} oraz - dla wartości `conditional` - listę artystów `model.artists` definiującą mapowanie identyfikator artysty → *Etykieta_Artysty* używaną podczas treningu *Modelu_GAN_Warunkowanego*.

### Requirement 9: Wymagania niefunkcjonalne dotyczące zasobów

**User Story:** Jako student dysponujący ograniczonymi zasobami obliczeniowymi (komputer osobisty, brak dostępu do klastra obliczeniowego), chcę aby cały eksperyment był wykonalny na pojedynczej maszynie konsumenckiej, aby praca była realizowalna w terminie.

#### Acceptance Criteria

1. THE System SHALL działać poprawnie niezależnie od konkretnej konfiguracji sprzętowej, w szczególności SHALL działać na maszynie z 16 GB pamięci RAM oraz pojedynczym *GPU_Konsumenckim* (konfiguracja referencyjna), bez wykluczania konfiguracji o większej lub mniejszej liczbie zasobów.
2. THE System SHALL działać w trybie wyłącznie CPU (bez GPU), z dopuszczalnym wzrostem czasu treningu nie większym niż 10-krotny względem wariantu z GPU.
3. THE System SHALL kończyć pełny cykl badawczy (przygotowanie zbioru, trening modelu GAN, inferencja, ewaluacja) dla jednego artysty w czasie nie większym niż 72 godziny pracy ciągłej na referencyjnej maszynie z *GPU_Konsumenckim*.
4. THE System SHALL zużywać nie więcej niż 6 GB pamięci VRAM podczas treningu z domyślnym rozmiarem wsadu zdefiniowanym w pliku konfiguracyjnym.

### Requirement 10: Wymagania dotyczące dokumentu pracy inżynierskiej

**User Story:** Jako autor pracy inżynierskiej, chcę aby dokument spełniał wymogi formalne pracy dyplomowej, aby praca została przyjęta do obrony.

#### Acceptance Criteria

1. THE Dokument_Dyplomowy SHALL zawierać objętość pomiędzy 30 a 50 stron znormalizowanych zgodnie z przyjętym szablonem redakcyjnym.
2. THE Dokument_Dyplomowy SHALL zawierać co najmniej 7 cytowań do recenzowanych źródeł literaturowych.
3. THE Dokument_Dyplomowy SHALL być napisany w formie bezosobowej i biernej (zostało wykonane, zaprojektowano).
4. THE Dokument_Dyplomowy SHALL stosować spis treści o głębokości zagłębień nie większej niż 3 poziomy.
5. THE Dokument_Dyplomowy SHALL zawierać wstęp o objętości nie mniejszej niż 1,5 strony oraz zakończenie o objętości nie mniejszej niż 1,5 strony.
6. THE Dokument_Dyplomowy SHALL zawierać każdy rozdział o objętości nie mniejszej niż 5 stron.
7. THE Dokument_Dyplomowy SHALL zawierać rozdział teoretyczny dotyczący teorii muzyki i podstaw uczenia maszynowego, umieszczony przed rozdziałem opisującym implementację.
8. THE Dokument_Dyplomowy SHALL cytować następujące pozycje literaturowe: Goodfellow et al. *Deep Learning* (2016), Kowalczuk et al. *Evolutionary music composition system with statistically modeled criteria* (2017), Brunner et al. *Symbolic Music Genre Transfer with CycleGAN* (2018), Cífka et al. *Groove2Groove* (2020), van den Oord et al. *WaveNet* (2016).
9. THE Dokument_Dyplomowy SHALL zawierać sekcję opisującą parametry zaprojektowanej sieci neuronowej (architektura warstw, liczba parametrów, hiperparametry treningu, funkcja straty, optymalizator).
10. THE Dokument_Dyplomowy SHALL zawierać sekcję wyników eksperymentalnych z wynikami testów statystycznych (próby statystyczne) oraz tabelami zawierającymi wartości metryk obiektywnych i subiektywnych.
11. THE Dokument_Dyplomowy SHALL zapisywać nazwy własne (zespołów, aplikacji, bibliotek) kursywą lub pisownią z wielkiej litery zgodnie z przyjętymi zasadami redakcyjnymi.
12. THE Dokument_Dyplomowy SHALL być przygotowany w systemie LaTeX z wykorzystaniem właściwej klasy dokumentu zgodnie z wymaganiami redakcyjnymi.

### Requirement 11: Właściwości poprawnościowe (correctness properties)

**User Story:** Jako programista weryfikujący poprawność implementacji, chcę dysponować zestawem właściwości testowalnych metodą property-based testing, aby automatycznie wykrywać regresje w komponentach Systemu.

#### Acceptance Criteria

1. FOR ALL poprawnych plików MIDI, THE System SHALL spełniać własność round-trip: parsowanie pliku, a następnie pretty-print i ponowne parsowanie SHALL produkować reprezentację wewnętrzną semantycznie równoważną oryginalnej (ta sama lista zdarzeń nutowych z dokładnością do kolejności zdarzeń o identycznym znaczniku czasu).
2. FOR ALL poprawnych plików MIDI, THE Ekstraktor_Cech SHALL spełniać własność idempotencji: dwukrotne wywołanie ekstrakcji dla tego samego pliku SHALL zwracać identyczny *Wektor_Cech*.
3. FOR ALL poprawnych plików MIDI i wszystkich liczb całkowitych *k* z zakresu od –12 do +12, THE Ekstraktor_Cech SHALL spełniać własność równoważności transpozycji: histogram klas wysokości dźwięków pliku transponowanego o *k* półtonów SHALL być cyklicznym przesunięciem histogramu klas wysokości pliku oryginalnego o *k* pozycji.
4. FOR ALL ustalonych *Seed* i identycznych konfiguracji wejściowych, THE Algorytm_Genetyczny SHALL spełniać własność determinizmu: dwa niezależne uruchomienia SHALL produkować identyczne populacje końcowe.
5. FOR ALL pokoleń *n* w trybie elitaryzmu, THE Algorytm_Genetyczny SHALL spełniać własność monotoniczności: najlepsza wartość *Funkcji_Dopasowania* w pokoleniu *n+1* SHALL być nie gorsza niż w pokoleniu *n*.
6. FOR ALL poprawnych *Utworów_Wejściowych* i *Utworów_Wyjściowych* zwróconych przez System, THE System SHALL spełniać własność zachowania długości: stosunek długości *Utworu_Wyjściowego* do długości *Utworu_Wejściowego* SHALL mieścić się w przedziale [0,95; 1,05].
7. FOR ALL poprawnych plików MIDI, THE Ekstraktor_Cech SHALL spełniać własność niezmienności na transformację tożsamościową: wektor cech pliku poddanego transformacji tożsamościowej SHALL być identyczny z wektorem cech pliku oryginalnego.
8. FOR ALL plików MIDI o zerowej gęstości nut (pliki puste), THE Ekstraktor_Cech SHALL zwracać deterministyczny wektor wartości neutralnych zdefiniowany w specyfikacji projektu, bez zgłaszania wyjątku.
9. WHEN tryb warunkowany jest aktywny, FOR ALL par *Etykiet_Artysty* c1 ≠ c2 obsługiwanych przez ten sam *Punkt_Kontrolny* *Modelu_GAN_Warunkowanego* oraz dla danego *Utworu_Wejściowego* x, THE System SHALL produkować *Utwory_Wyjściowe* G(x, c1) oraz G(x, c2) o różnych *Wektorach_Cech*, przy czym odległość euklidesowa pomiędzy tymi wektorami SHALL być nie mniejsza niż próg ε zdefiniowany w specyfikacji projektu (właściwość weryfikująca, że *Etykieta_Artysty* faktycznie wpływa na wynik generacji).
10. WHEN tryb warunkowany jest aktywny ORAZ kierunkowość warunkowania jest aktywna, FOR ALL artystów A obecnych w *Zbiorze_Wielo_Artystycznym* obsługiwanym przez wczytany *Punkt_Kontrolny* *Modelu_GAN_Warunkowanego*, THE Model_GAN_Warunkowany SHALL produkować *Utwory_Wyjściowe* G(x, c_A), których średnia odległość *Wektora_Cech* od agregowanego *Wektora_Cech* *Zbioru_Stylu* artysty A SHALL być nie większa niż średnia odległość od agregowanych *Wektorów_Cech* *Zbiorów_Stylu* pozostałych artystów obecnych w *Zbiorze_Wielo_Artystycznym* (właściwość kierunkowości warunkowania).
