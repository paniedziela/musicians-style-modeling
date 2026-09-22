# E3 — lokalny przykład „Kotek”

Krótki przykład A/B/C/D: jedno wejście i trzy istniejące warianty E3 nazwane
według stylu docelowego — Bach, Beethoven i Chopin.

## Pliki

- `input.mid` / `input.wav` — lokalne wejście `KOTEK.MID`;
- `output_bach.*`, `output_beethoven.*`, `output_chopin.*` — trzy wyniki;
- `summary.json` — przenośne metryki treści i sumy MIDI;
- `provenance.json` — jawny stan wiedzy o pochodzeniu plików;
- `SHA256SUMS` — sumy kontrolne plików MIDI, WAV i JSON.

WAV-y są mono, 16-bit PCM, 22 050 Hz i mają około 18,67 s wraz z
wybrzmieniem. Zostały wyrenderowane lokalnie z użyciem `FluidR3_GM.sf2`.

Pierwotne źródło lokalnego `KOTEK.MID` ani dokładne konfiguracje historycznych
trzech transformacji nie były zapisane obok plików. Ten brak jest zachowany
jawnie w `provenance.json`; przykładu nie należy przedstawiać jako
reprodukowalnego wyniku eksperymentalnego.

Do porównania w listenerze:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m musicians_style.listener `
  --root examples/e3_kotek `
  --soundfont soundfonts/FluidR3_GM.sf2
```
