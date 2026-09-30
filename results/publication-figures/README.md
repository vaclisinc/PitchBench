# Manuscript Figures 1–4: sources and redesigned figures

Figures 1–4 were redrawn on 2026-10-01 from the manuscript task taxonomy and saved submission evidence.
This changes presentation, not evaluation scores. The original plots and CSVs
remain intact. No model calls, smoothing, new confidence intervals, or new
experimental comparisons were introduced.

## Source map

| Figure | Saved evidence | Original plotting implementation |
| --- | --- | --- |
| 1: task hierarchy | `PitchBench-paper/exp-figure.jpg`; task definitions in [EXPERIMENTS.md](../../EXPERIMENTS.md) | Original editable source unavailable; reconstructed with [hierarchy.py](../../src/pitchbench/analysis/hierarchy.py) |
| 2: A1 prediction distributions | Six `paper/evaluation/_<model>/pitchbench_a1_single_pitch_id/<run>/results_<model>.csv` files; exact paths and SHA-256 hashes in [figure_sources.json](figure_sources.json) | [a1.py](../../src/pitchbench/analysis/a1.py), `extract_a1_data`, `_predicted_midi`, `plot_a1_heatmap` |
| 3: accuracy by pitch | [accuracies_by_pitch.csv](../../paper/figures-and-tables/accuracies_by_pitch.csv) | [overview.py](../../src/pitchbench/analysis/overview.py), `plot_accuracy_by_note` |
| 4: accuracy by representation | [accuracies_by_notation.csv](../../paper/figures-and-tables/accuracies_by_notation.csv) | [overview.py](../../src/pitchbench/analysis/overview.py), `plot_accuracy_by_notation` |

The Figure 3/4 CSVs match those in the manuscript repository byte-for-byte.
They are frozen historical aggregates, not the updated Table 1 builder's
outputs. Figure 3 retains all 300 cells (six models, two formats, 25 pitches).
Figure 4 retains all 30 cells, including the historical ANY aggregate with its
original denominator. ANY is displayed separately because it is an aggregate,
not a fifth requested response notation. Its score is not an arithmetic mean
of the four displayed formats. No uncertainty estimates were supplied in these
source CSVs, so none are drawn.

Figure 2 retains all 18 model/format panels and uses the original A1 parser,
nearest-semitone binning, and column normalization over valid predictions
within MIDI 29–89. Each model has 1,159 recordings (61 pitches × 19 timbres).
The displayed probabilities are conditional on an in-range parsed answer;
they must not be read as unconditional accuracy. Titles now disclose rounded
in-range response coverage. [Coverage counts](figure2_response_coverage.csv)
account for every response, including invalid and out-of-range predictions.
[Sparse cell counts](figure2_prediction_counts.csv) preserve all nonzero cells;
omitted cells are zero, not missing observations. The diagonal means a match
after nearest-semitone rounding, not the stricter Hz scoring tolerance.

## Regeneration

The new generator is [publication.py](../../src/pitchbench/analysis/publication.py).
Use the project's Python environment with NumPy and Matplotlib installed:

```bash
PYTHONPATH=src python -m pitchbench.analysis.publication \
  --output-dir /ABSOLUTE/OUTPUT/DIRECTORY \
  --qa-scripts /ABSOLUTE/PATH/TO/nature-figure/scripts
```

It validates complete, unique aggregate coverage and A1 response accounting;
exports editable PDF/SVG and 600-dpi PNG; and records source/parser/generator
hashes and library versions. `--qa-scripts` enables the strict render-time
alignment gate. Omitting it renders the same figures but marks alignment as
unaudited. The original aggregate-generating functions remain available above.

Compact PDFs are committed under [figures/](figures/). SVGs, PNG previews and
the compiled manuscript are under the VacLab output directory
`outputs/pitchbench-paper/figure-redesign-20261001/`.

## Validation and design

All figures use the manuscript's 139.7-mm text width. Figure 2 uses a common
sequential probability scale; Figure 3 uses consistent model colours and
marker shapes; Figure 4 uses an annotated accuracy matrix with best-format
scores in bold. See [figure-contract.md](figure-contract.md).

The retained [QA records](qa/) show that all three figures pass panel alignment
(1.5-pt tolerance), PDF text and rendered collision checks. Minimum PDF text
sizes are 5.3, 5.6 and 5.5 pt respectively. Source preflight has no failures.
Its two warnings are reviewed: the width is supplied through `WIDTH_IN = 5.5`
rather than a literal `figsize`, and no TIFF is exported because the manuscript
uses PDF with a 600-dpi PNG preview. Figure 4's contained fill overlays are
intentional numeric labels inside shaded table cells.

Every panel and each figure embedded in the compiled manuscript were visually
inspected. The manuscript compiles to 14 pages; body text ends on page 11,
references begin on page 12, and the prompt-sensitivity table stays in the
appendix. Existing text/font warnings outside these figures remain. Manuscript
Q1/Q2 paragraphs are unchanged by this figure-only revision.

## Figure 1: editable task hierarchy

The original pyramid has been replaced by a three-level task map: Atomic,
Contextual, and Melodic. All 28 tasks remain in their original six categories
(A=3, B=5, C=4, D=8, E=6, F=2). Short labels were checked against the manuscript
Task Hierarchy and EXPERIMENTS.md; [the mapping](figure1_task_hierarchy.tasks.csv)
records every displayed task. The connectors represent conceptual progression,
not observed performance, empirical difficulty, or proven prerequisites.

```bash
PYTHONPATH=src python -m pitchbench.analysis.hierarchy \
  --output-dir /ABSOLUTE/OUTPUT/DIRECTORY \
  --qa-scripts /ABSOLUTE/PATH/TO/nature-figure/scripts
```

[PDF](figures/figure1_task_hierarchy.pdf) and
[editable SVG](figures/figure1_task_hierarchy.svg) are committed. The PNG preview
and compiled manuscript are in
`outputs/pitchbench-paper/figure1-redesign-20261001/`. See the
[contract](figure1-contract.md) and [source hashes](figure1_task_hierarchy.sources.json).
Figure 1 uses 139.7 × 96.5 mm, the existing manuscript text width. Alignment,
PDF text and collision checks pass; the smallest glyph is 5.6 pt. All six
category blocks were visually checked. The 71 fully contained text/fill
relationships are intentional labels within panels and numbered level markers.
Source preflight has 19 passes, no failures, and two reviewed warnings:
139.7 mm is intentional for the NeurIPS manuscript (rather than the checker’s
89/183-mm Nature defaults), and TIFF is unnecessary for this PDF-based paper.
No research questions or result paragraphs are changed by the replacement.
