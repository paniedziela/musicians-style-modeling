# E2 — zagregowane wyniki baseline'u GA

Katalog zawiera wyłącznie zbiorcze artefakty pełnego przebiegu E2: sześć
wykresów i krótkie `summary.json`. Nie zawiera źródłowych MIDI, wyników per
utwór, ścieżek lokalnych ani pełnego katalogu eksperymentu.

E2 był zamrożonym baseline'em czteroparametrowego algorytmu genetycznego i nie
miał bramki GO/NO-GO. Średni wzrost prawdopodobieństwa stylu docelowego był
dodatni, lecz metoda istotnie pogarszała rytm i długość materiału. Wykresy należy
czytać razem z ograniczeniami opisanymi w
[`docs/results/E2.md`](../../docs/results/E2.md).

Najważniejsze wykresy:

- `style_gain_by_direction.png` — wynik w sześciu kierunkach transferu;
- `style_gain_vs_length_error.png` — relacja zysku proxy i błędu długości;
- `length_error.png` — rozkład błędu długości;
- `genome_distributions.png` i `gene_boundary_rates.png` — diagnostyka genotypu;
- `runtime_and_stagnation.png` — koszt i przyczyny zatrzymania.
