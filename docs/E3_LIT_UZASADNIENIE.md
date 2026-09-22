Tak — rdzeń E3 ma mocne oparcie w literaturze. Trzeba tylko wyraźnie oddzielić inspiracje naukowe od naszych własnych decyzji implementacyjnych. Literatura nie mówi: „użyj dokładnie pięciu genów, populacji 32 i 60 pokoleń”.

| Decyzja E3 | Oparcie | Ocena |
|---|---|---|
| Transfer jako optymalizacja cech uczonych z korpusu | Zalkow et al.; Herremans et al.; Kowalczuk et al. | mocne |
| Małe, lokalne zmiany zamiast zastępowania całego utworu | Zalkow et al. | mocne |
| Zachowanie długości i treści wejścia | Zalkow et al.; Cífka et al.; Brunner et al. | mocne |
| Osobny pomiar zmiany stylu i zachowania treści | Cífka et al.; Brunner et al. | bardzo mocne |
| Klasyfikator jako miernik stylu | Herremans et al.; Brunner et al. | mocne, ale nie powinien być jedyną metryką |
| Rozwiązania feasible wygrywają z naruszającymi ograniczenia | Deb | klasyczna metoda optymalizacji ograniczonej |
| Profil celu tylko z train i zamrożenie konfiguracji | Simonetta; Cawley i Talbot | mocne metodologicznie |
| Najwyższa nuta onsetu jako melodia | literatura o Skyline | akceptowalny baseline, ale nie pewna prawda |
| Pięć konkretnych genów i ich zakresy | nasza konstrukcja | wymaga uzasadnienia i ablacji |
| `32 × 60`, stagnacja 12, progi 0,9–1,1 | nasza konfiguracja | uzasadnić kosztem E2 i zamrozić przed runem |

Najważniejsze źródła

1. Zalkow, Brand i Graf — *Musical Style Modification as an Optimization Problem*

To najbardziej bezpośredni odpowiednik E3. Autorzy:

- uczą charakterystyki stylu z korpusu;
- traktują zmianę stylu jako optymalizację wielu celów;
- stosują lokalne modyfikacje wysokości i długości nut;
- zmieniają długości par nut tak, aby suma długości pozostała stała;
- argumentują, że małe modyfikacje pomagają zachować cechy oryginału.

Ich metoda jest monofoniczna i używa lokalnego searchu, więc nie jest gotowym algorytmem dla naszego fortepianu, ale bardzo dobrze uzasadnia ogólną konstrukcję E3. [Zalkow, Brand, Graf, 2016](https://www.audiolabs-erlangen.de/content/05_fau/assistant/zalkow/01_publications/ZalkowBrandGraf16_StyleOpt_ICMC.pdf)

2. Herremans, Sörensen i Martens — klasyfikator jako funkcja celu

Autorzy wyekstrahowali globalne cechy utworów Bacha, Haydna i Beethovena, zbudowali klasyfikator kompozytora, a prawdopodobieństwo z regresji logistycznej włączyli do funkcji celu algorytmu Variable Neighborhood Search generującego kontrapunkt.

To jest niemal dokładne uzasadnienie ciągu:

```text
cechy → klasyfikacja kompozytora → funkcja celu optymalizacji
```

czyli naszego E1 → E3. [Herremans, Sörensen, Martens, 2015](https://qmro.qmul.ac.uk/xmlui/handle/123456789/11794?show=full), DOI: [10.1162/COMJ_a_00316](https://doi.org/10.1162/COMJ_a_00316).

3. Kowalczuk, Tatara i Bąk — kryteria statystyczne i regułowe

Ta praca uzasadnia połączenie:

- cech muzycznych;
- kryteriów regułowych, czyli naszych ograniczeń poprawności;
- kryteriów statystycznych wyuczonych z istniejących utworów;
- algorytmu ewolucyjnego.

Autorzy modelują kryteria rozkładami Gaussa z parametrami oszacowanymi na rozpoznanych utworach. To jest bliskie naszemu profilowi `mean/std` i odległościom standaryzowanym. [Kowalczuk, Tatara, Bąk, 2017](https://mostwiedzy.pl/pl/publication/download/1/evolutionary-music-composition-system-with-statistically-modeled-criteria_1798.pdf), DOI: [10.1007/978-3-319-60699-6_70](https://doi.org/10.1007/978-3-319-60699-6_70).

4. Deb — twarde ograniczenia zamiast strojenia kar

Deb opisuje bezparametrowe reguły porównywania kandydatów:

- rozwiązanie dopuszczalne wygrywa z niedopuszczalnym;
- spośród dopuszczalnych wybiera się lepszą wartość celu;
- spośród niedopuszczalnych preferuje się mniejsze naruszenie ograniczeń.

To bezpośrednio uzasadnia decyzję, aby nie dobierać wielu arbitralnych `λ`, lecz oddzielić `feasible` od `infeasible`. [Deb, 2000, *An Efficient Constraint Handling Method for Genetic Algorithms*](https://doi.org/10.1016/S0045-7825(99)00389-8).

5. Cífka, Şimşekli i Richard — styl oraz treść muszą być mierzone osobno

Autorzy podkreślają, że doskonały wynik tylko w stylu albo tylko w zachowaniu treści jest łatwy, lecz bezużyteczny. Ewaluują oba aspekty osobno, stosują profil stylu oparty na relacjach czasowo-interwałowych i używają wejścia jako baseline’u.

Praca dotyczy właśnie transformacji akompaniamentu przy zachowaniu struktury harmonicznej, więc dobrze podpiera decyzję, aby E3 modyfikował głównie akompaniament. [Cífka, Şimşekli, Richard, ISMIR 2019](https://archives.ismir.net/ismir2019/paper/000071.pdf)

Co ważne, autorzy ostrzegają też, że wysoki wynik klasyfikatora dowodzi jedynie obecności cech rozróżniających styl, a nie pełnej zgodności muzycznej. Dlatego `Δp_target` nie powinno być jedynym dowodem sukcesu.

6. Brunner et al. — zachowanie struktury podczas transferu

W symbolicznym CycleGAN-ie autorzy dodali dodatkowe mechanizmy chroniące strukturę wejścia i oceniali transfer oddzielnym klasyfikatorem gatunku. Wyniki pokazały też typowe napięcie: mocna zmiana domeny może nadmiernie przekształcić oryginał. [Brunner et al., 2018](https://arxiv.org/abs/1809.07575)

To wspiera ograniczenia melodii i struktury, nawet jeśli E3 nie korzysta z GAN-u.

Co wymaga poprawienia w planie

Największa słabość dotyczy określenia „chroniona melodia”. Najwyższa nuta każdego onsetu to znany algorytm Skyline, ale nie zawsze odpowiada melodii — może przeskakiwać na akompaniament, a melodia może znajdować się w niższym głosie. Jest to dobry, tani baseline, ale nie ground truth. [Ozcan, Isikhan i Alpkocak, 2005](https://doi.org/10.1109/ISM.2005.77)

W pracy należałoby napisać:

> Jako operacyjny estymator melodii zastosowano heurystykę Skyline, a nie pełną separację głosów. Ograniczenie zachowania melodii odnosi się zatem do melodii wyznaczonej przez tę heurystykę.

Dobrze byłoby ręcznie sprawdzić jej poprawność na niewielkiej, wcześniej ustalonej próbie, np. 15–30 fragmentów. Inaczej algorytm i metryka mogą zgodnie „chronić” niewłaściwą linię.

Druga rzecz to seedy. Powtórzenie tego samego seeda sprawdza determinizm kodu, ale nie odporność algorytmu na losowość. Wytyczne dotyczące porównywania algorytmów optymalizacyjnych zalecają jawne raportowanie konfiguracji, kosztu, losowości i statystycznie uczciwe porównania. [Beiranvand, Hare i Lucet, 2017](https://doi.org/10.1007/s11081-017-9366-1)

Rozsądny kompromis koszt–wiarygodność:

- pełne 300 zadań wykonać dla jednego zamrożonego seeda;
- sześć reprezentatywnych kierunków uruchomić dla tych samych trzech seedów co E2;
- raportować tę małą analizę jako odporność na seed, nie jako główny test.

Trzecia rzecz: dokładne zakresy genów, `32 × 60`, stagnacja 12, limit liczby nut ±10% i próg melodii 0,95 są naszymi decyzjami. Powinny zostać opisane jako:

> Parametry przedeksperymentalne ustalone na podstawie ograniczeń muzycznych, kosztu zmierzonego w E2 oraz potrzeby zachowania małej, interpretowalnej przestrzeni poszukiwania.

Nie należy twierdzić, że te wartości „wynikają z literatury”.

Najlepsze krótkie uzasadnienie E3 do pracy mogłoby brzmieć:

> Zaproponowaną metodę sformułowano jako ograniczoną optymalizację cech symbolicznych. Podejście bazuje na pracach przedstawiających modyfikację stylu jako iteracyjne, lokalne przekształcanie zdarzeń nutowych względem profilu wyuczonego z korpusu oraz na zastosowaniu modeli klasyfikacyjnych i statystycznych jako składników funkcji celu. Zachowanie melodii, struktury czasowej i poprawności MIDI potraktowano jako ograniczenia dopuszczalności, zgodnie z regułami obsługi ograniczeń w algorytmach genetycznych. Skuteczność oceniana jest osobno w wymiarach dopasowania stylu i zachowania treści.

W [LITERATURA.md](LITERATURA.md) są już wymienione trzy najważniejsze pozycje dla tego uzasadnienia: Zalkow, Herremans oraz Kowalczuk. Plan E3 warto uzupełnić o Deba, Cífkę, źródło Skyline oraz wytyczne benchmarkowania algorytmów optymalizacyjnych.
