# E3 — Bach, Preludium BWV 868 → Beethoven

Krótki, około 38-sekundowy przykład wybrany z pełnego eksperymentu E3. Wejściem
jest score MIDI Preludium Bacha BWV 868 z ASAP, a wynikiem istniejąca
transformacja w kierunku profilu Beethovena.

## Pliki

- `input_bach_bwv868.mid`, `output_beethoven.mid` — wejście i wynik E3;
- `input_bach_bwv868.wav`, `output_beethoven.wav` — odsłuch A/B;
- `result.json` — skrócony, przenośny raport istniejącego zadania E3;
- `provenance.json` — pochodzenie, licencja i sposób utworzenia artefaktów;
- `SHA256SUMS` — sumy kontrolne plików MIDI, WAV i JSON.

WAV-y są mono, 16-bit PCM, 22 050 Hz i mają około 39,0 s wraz z
wybrzmieniem. Zostały wyrenderowane lokalnie z użyciem `FluidR3_GM.sf2`; sam
SoundFont nie jest częścią repozytorium.

E3 zachowało czas utworu i onsety, zwiększając liczbę nut z 417 do 439.
Transpozycja wyniku wynosi `+4` półtony. Dodatni zysk funkcji celu i wzrost
proxy klasyfikatora nie są samodzielnym dowodem udanego transferu stylu.

Źródłowy score MIDI pochodzi z ASAP i jest redystrybuowany na warunkach
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Pełne przypisanie oraz opis zmian znajdują się w `provenance.json`.

Do porównania MIDI w lokalnym interfejsie:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m musicians_style.listener `
  --root examples/e3_bach_prelude `
  --soundfont soundfonts/FluidR3_GM.sf2
```
