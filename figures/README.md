# Figures

| File | Used in | Shows |
|---|---|---|
| `fig_conf_progression.pdf` | Conference paper | F-measure and AP of every detector in the progressive study: (a) all 15 circuits, (b) the nine small circuits where TWEC-TD completed |
| `fig_conf_percircuit.pdf` | Conference paper | Per-circuit F-measure, CA kNN-50 vs LOF + CBLOF stacking |
| `fig_journal_progression.pdf` | Journal paper | F-measure and AP from the baselines through CA kNN-50 and each TW-CA-kNN component |
| `slide_progression_line.png` | Slides | F-measure and AP across every method tried, ending at TW-CA-kNN |

The PDFs are sized for one IEEE column (3.5 in) with 8 pt Times-compatible text; the PNGs are 300 dpi previews.

## Regenerating

```bash
cd figures
python make_figs.py         # fig_conf_progression, fig_journal_progression
python make_percircuit.py   # fig_conf_percircuit (reads results/progression/progression_per_circuit.csv)
python make_slide_line.py   # slide_progression_line
```

All values are means of 20 runs per circuit under the fixed contamination rate C, except the final TW-CA-kNN step, which uses the automatic flag count.
