# Figure contract

Backend: existing Python/matplotlib workflow. Target: the manuscript's 139.7 mm text width; PDF/SVG editable text, PNG at 600 dpi; minimum 5 pt at final size. Existing empirical values and parser/aggregation semantics are preserved. No new evaluation, uncertainty estimates, smoothing, or model calls.

## Figure 1
Task-hierarchy schematic, not a numeric plot. The original editable source was unavailable. The owner requires a pyramid: the vector reconstruction now preserves the A base, B–E middle wedges and F apex. See figure1-pyramid-contract.md and src/pitchbench/analysis/pyramid.py.

## Figure 2 — quantitative grid
Question: do predicted pitches follow ground truth, and does this depend on model/notation?
Six model groups, each with MIDI/SPN/Hz conditional prediction distributions. Keep all 18 panels. Reuse the A1 prediction parser and nearest-semitone binning, source range MIDI 29–89. Preserve the old figure's normalization conditional on a valid in-range prediction; record all excluded invalid/out-of-range answers and show in-range coverage for every panel. Use a shared 0–1 sequential colour scale and model headings outside the data. No confidence intervals: these are full empirical response distributions, not means across independent seeds.

## Figure 3 — quantitative comparison
Question: does accuracy vary by ground-truth pitch and requested notation?
Two complementary panels, MIDI and SPN, retain all six models and all 25 C3–C5 pitches from the saved aggregate. Shared model colours, marker differentiation, external legend, common 0–100% axes. Emphasize A4 only via its tick/reference region. No uncertainty is supplied in the source aggregate; do not invent intervals.

## Figure 4 — quantitative comparison
Question: which requested representation elicits the highest accuracy for each model?
Annotated accuracy matrix (all six models × four requested formats) with the saved ANY aggregate as a separate aligned column. This separates an aggregate score from requested output formats. Preserve all 30 source values and original denominators; no recomputation of historical scores. Exact numbers and a common 0–100 scale make rankings readable without oblique labels or 30 repeated bar labels. No uncertainty is supplied.

Evidence: Figures 3 and 4 source CSVs match byte-for-byte across the code and manuscript repositories. Figure 2 uses the six saved A1 evaluation runs, 1,159 recordings each. Figure-level source hashes and parser hashes are saved with exports.
