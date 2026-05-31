# Modelowanie stylu artystycznego muzyków

System badawczy realizujący transfer stylu artystycznego pomiędzy utworami w
formacie MIDI. Praca inżynierska, Wydział Elektroniki, Telekomunikacji i
Informatyki Politechniki Gdańskiej.

System składa się z trzech głównych komponentów:

- **Ekstraktor_Cech** - analiza cech muzycznych zbioru plików MIDI,
- **Model_GAN** - generatywna sieć neuronowa (StarGAN warunkowany / CycleGAN per-artysta),
- **Algorytm_Genetyczny** - ewolucyjna metoda transformacji MIDI optymalizująca cechy stylu.

## Wymagania

- Python >= 3.10
- Zależności w `requirements.txt` (pip) lub `environment.yml` (conda)

## Instalacja

```bash
# wariant pip
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .[dev]

# wariant conda
conda env create -f environment.yml
conda activate musicians-style
```

## Struktura projektu

```
src/musicians_style/     # kod źródłowy pakietu
  data/                  # Akwizytor_Danych, Manifest_Zbioru
  midi/                  # Parser_MIDI, Pretty_Printer_MIDI, Pianoroll
  features/              # Ekstraktor_Cech
  models/                # StarGAN, CycleGAN, funkcje strat
  training/              # Pipeline_Treningu
  ga/                    # Algorytm_Genetyczny
  inference/             # transfer stylu
  evaluation/            # ewaluacja obiektywna i subiektywna
tests/                   # unit / property / integration
configs/                 # pliki konfiguracyjne YAML/JSON
experiments/             # katalogi wynikowe per uruchomienie
```

## Testy

```bash
pytest                   # wszystkie testy
pytest -m property       # tylko testy własnościowe (Hypothesis)
pytest -m "not slow"     # pominięcie testów kosztownych
```
