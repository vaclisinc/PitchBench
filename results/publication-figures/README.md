# Manuscript Figures 1–4: sources and redesigned figures

Figures 1–4 were redrawn on 2026-10-01 from the task definitions and saved submission evidence.
This changes presentation, not evaluation scores. The original plots and CSVs
remain intact. No model calls, smoothing, new confidence intervals, or new
experimental comparisons were introduced.

## Source map

| Figure | Saved evidence | Original plotting implementation |
| --- | --- | --- |
| 1: task hierarchy | `PitchBench-paper/exp-figure.jpg`; task definitions in [EXPERIMENTS.md](../../EXPERIMENTS.md) | Original editable source unavailable; pyramid reconstructed with [pyramid.py](../../src/pitchbench/analysis/pyramid.py) |
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
references begin on page 11, and the prompt-sensitivity table stays in the
appendix. Existing text/font warnings outside these figures remain. Manuscript
Q1/Q2 paragraphs are unchanged by this figure-only revision.

## Figure 1: redesigned pyramid

**Keep the pyramid.** The owner requires the original three-tier structure:
A1 as the foundation, A2/A3 as the variation blocks above it; four adjacent
B/C/D/E wedges in the middle; F at the apex. All 28 tasks remain in the side
legends, with A/B/C/F on the left and D/E on the right. This revision changes
colour, typography, separators, spacing and export quality. It does not turn
the taxonomy into a task list or flowchart. Areas do not encode counts,
performance or measured difficulty.

```bash
PYTHONPATH=src python -m pitchbench.analysis.pyramid \
  --output-dir /ABSOLUTE/OUTPUT/DIRECTORY \
  --qa-scripts /ABSOLUTE/PATH/TO/nature-figure/scripts
```

- [PDF](figures/figure1_pitch_pyramid.pdf) and
  [editable SVG](figures/figure1_pitch_pyramid.svg).
- [Task mapping](figure1_pitch_pyramid.tasks.csv),
  [source hashes](figure1_pitch_pyramid.sources.json),
  [figure contract](figure1-pyramid-contract.md).
- PNG preview and compiled manuscript:
  `outputs/pitchbench-paper/figure1-pyramid-20261001/`.

The figure is 139.7 × 73.7 mm, matching manuscript text width. The PDF contains
all 28 distinct task IDs; A1–A3 and F1–F2 are intentionally repeated inside the
pyramid. Minimum text is 5.2 pt. The alignment gate records a single canvas as
not applicable. PDF font checks pass, and the collision audit has no failures.
Its five fill-edge warnings are reviewed false positives: bounding rectangles
of adjacent trapezoids overlap, but all eight middle text boxes are completely
inside their own actual wedge polygons (verified geometrically in the
[content check](qa/figure1-pyramid-content-check.json)). All six categories,
three tiers, and the complete manuscript page were visually inspected.

Source validation reports 19 passes, no failures and two reviewed warnings:
139.7 mm intentionally follows the NeurIPS manuscript rather than the checker's
89/183-mm Nature defaults, and TIFF is unnecessary for this PDF-based paper.
The SVG has only trailing serialization whitespace removed; path coordinates
are unchanged. The manuscript compiles to 14 pages with the conclusion on
page 11. The compact revision changes only the four figure PDF assets in the manuscript; main.tex and all prose/captions are byte-identical to the pre-compaction version.

## Compact revision: figure height only

The owner specified a 10-page body limit and requested figure compaction before
any prose edits. Figure 1 now places Level 1/2/3 on the left with tier brackets,
retaining the pyramid and all 28 tasks. A1's side label is shortened to “Pitch
identification” under “Single note”; the pyramid still says “A1 SINGLE PITCH”.
All other task meanings, panels and quantitative values remain unchanged.

| Figure | Previous height (in) | Current height (in) | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 3.90 | 2.90 | 25.6% |
| 2 | 3.62 | 2.75 | 24.0% |
| 3 | 2.08 | 1.72 | 17.3% |
| 4 | 2.05 | 1.50 | 26.8% |

All retain 5.5-in width. Total saved figure height is 2.78 in (7.06 cm; 23.9%).
Minimum rendered text sizes are 5.2, 5.3, 5.6, and 5.5 pt. Font checks pass;
Figures 2–4 pass alignment and collision checks. Figure 1 has one canvas
(alignment not applicable) and two reviewed bounding-rectangle fill warnings;
all eight middle label boxes are inside their actual wedge polygons. A1
prediction counts and coverage CSVs are byte-identical to the previous revision;
Figure 3/4 source hashes are unchanged. Every panel and all four manuscript
figure pages were visually inspected.

The manuscript still has **11 body pages**, so the 10-page limit is not yet met.
Figure 4, Limitations and Conclusion remain on page 11; references also start
there. No prose, captions, tables, margins or manuscript font sizes were edited.
Content reductions are recommendations only, pending the owner's choice.
See [validation](qa/compact-validation.json), [pagination](qa/compact-pagination.json)
and [contract](compact-figure-contract.md). Current previews and compiled PDF:
`outputs/pitchbench-paper/compact-figures-20261001/`.

## Current pyramid label and spacing refinement

The pale slate, blue, lavender, sage, peach and teal pyramid retains its wider geometry. A1/A2/A3 now share 6 pt normal text; the apex shows only F. All 28 task descriptions remain in the side legends. D/E descriptions sit lower beside the contextual tier. The canvas extends right to 5.75 × 2.9 inches; the manuscript scales it to 5.5 × 2.774 inches, slightly reducing its height.

The generator checks all 68 exterior legend text boxes with 2.5 pt padding against actual polygons. All eight middle label boxes fit inside their own wedges. The PDF collision audit has zero failures and one reviewed bounding-rectangle warning for Sequence, whose text is inside its own wedge. Minimum text is 5.3 pt in the export and 5.07 pt in the manuscript. Alignment is not applicable to this single canvas. Source checks have no failures; manuscript-specific width and vector PDF delivery explain the two advisory warnings.

Only Figure 1 changes; prose and Figures 2–4 are untouched. The compiled manuscript remains 14 total pages, with body text ending on page 11. The full figure page was visually inspected. Latest preview and compiled PDF: `outputs/pitchbench-paper/pyramid-bold-20261001/`.

Figure 1 now uses Hanken Grotesk Regular (400) and Bold (700), using the vaclis.net family with stronger emphasis as requested. The open-licensed fonts are bundled under `src/pitchbench/analysis/fonts/` and embedded in the PDF; SVG text remains editable. Every PDF text run was verified as Hanken Grotesk, including equal Regular A1/A2/A3 labels. The typography update preserves geometry, page count and all 28 tasks; font and actual-polygon clearance checks pass.
