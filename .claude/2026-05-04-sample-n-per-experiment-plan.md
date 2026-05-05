# Sample N per experiment

**Date:** 2026-05-04
**Status:** complete

## Goal
Add a global `--sample-n` flag that randomly draws a fixed number of audio
stimuli per experiment, so every experiment runs at the same sample size
(fairer ability comparison) and so iteration cycles don't require running the
full cross-product. Sampling must be deterministic and recorded in result
metadata.

## Context

### Decisions locked in
- **Sampling unit:** 1 audio file (= 1 record row). All prompt variants for that
  audio still run. Matches the natural unit of every existing script.
- **Strategy:** stratified by the experiment's primary axis (typically
  `source`). Each script declares its stratification key. Falls back to flat
  uniform sampling only if a script has no meaningful axis.
- **CLI surface:** global `--sample-n` and `--sample-seed` in `run.py`,
  forwarded via the existing `extra_argv` mechanism (same dual-parse pattern as
  `--models`). Each in-scope script also registers them in its own
  `_parse_args()` so direct script invocation still works.
- **Default seed:** reuse `--sample-seed` value (default `42`) for the sampling
  RNG. Independent from each script's existing `--seed` (which controls
  stimulus construction, not which stimuli to keep).
- **Preview/download:** when `--sample-n` is set, only generate audio for the
  sampled stimuli. Without it, behaviour is unchanged (generate all).

### In scope
27 experiment scripts:
- `a1`–`a5`, `b1`–`b5`, `c1`–`c4`, `d1`–`d7`, `e1`–`e3`, `f1`, `f2`, `z1`

### Out of scope
- `f1` (embedding geometry / PCA), `f2` (token logits), `f3` (kNN oracle).
  These probes operate over whole pools, not per-query iteration, so
  `--sample-n` doesn't have a meaningful interpretation here. Leave them
  untouched.

### z1 interaction
`z1` already has `--n-per-family`. When `--sample-n` is passed, it
**supersedes** `--n-per-family`: the full NSynth valid pool is the universe,
stratified by `instrument_family_str`, and `N` items total are drawn balanced
across families. Without `--sample-n`, the existing `--n-per-family` path is
unchanged.

### Reproducibility hooks (CLAUDE.md rules)
Every results JSON must record (via `get_run_metadata`):
- `sample_n` (int or null)
- `sample_seed` (int)
- `total_available` (int — full enumeration size before sampling)
- `stratified_by` (str — the axis key used)
The summary `.txt` must echo these so a human reading results knows whether
they're looking at a sampled run.

### Stratification keys per script (initial proposal)
| Script(s) | Strata key |
|---|---|
| `a1`, `a2`, `a3`, `a4`, `a5` | `source` |
| `b1`, `b2`, `b4` | `source` |
| `b3`, `b5` | `source` (trial × source × pitch product) |
| `c1`, `c2`, `c3`, `c4` | `source` |
| `d1`, `d2`, `d3`, `d4`, `d5`, `d6` | `source` |
| `d7` | `(n_notes, source)` (composite key — preserve length × source coverage) |
| `e1`, `e2` | `source` |
| `e3` | `(source, background)` |
| `f1` | `(n_voices, source)` |
| `f2` | `chorale` |
| `z1` | `instrument_family_str` |

Final keys to be confirmed when each script is touched — the table is the
starting point, not the spec.

### Files this touches
- `src/pitchbench/experiments/helpers/sampling.py` — new
- `src/pitchbench/experiments/run.py` — flag + forwarding
- `src/pitchbench/experiments/helpers/results.py` — extend
  `get_run_metadata()` callers' contract OR (cleaner) accept a sampling-info
  dict
- 27 scripts under `src/pitchbench/experiments/scripts/`
- `tests/` — new `test_sampling.py`
- `README.md` — document the flags

## Action items

- [x] Implement `helpers/sampling.py` with
      `stratified_sample(items, n, key_fn, seed) -> list[dict]`. Guarantees:
      (a) deterministic for fixed `(items, n, key_fn, seed)`,
      (b) per-stratum allocation is `floor(n / num_strata)` with the remainder
      distributed in sorted-key order,
      (c) raises if any stratum is too small to fulfil its quota (caller
      decides whether to fall back). Add `tests/test_sampling.py` covering
      determinism, exact stratum counts, and remainder distribution.
- [x] Add `--sample-n` (int, default `None`) and `--sample-seed` (int,
      default `42`) to `run.py`. Forward through `extra` so each script
      receives them via its existing `parse_known_args` path. Update the
      module docstring at the top of `run.py` with a usage example.
- [x] Cluster 1 — single-pitch source × pitch scripts (`a1`, `a2`, `a3`, `a4`,
      `a5`, `e1`, `e2`): factor each into a `build_stimuli() -> list[dict]`
      step before the iteration loop, apply `stratified_sample` when
      `--sample-n` is set, iterate the (possibly subsetted) list. Update
      `preview()` to only render the sampled subset. Add metadata fields.
- [x] Cluster 2 — chord scripts (`c1`, `c2`, `c3`, `c4`): same refactor; key =
      `source`.
- [x] Cluster 3 — onset/timing scripts (`b1`, `b2`, `b3`, `b4`, `b5`): same
      refactor; trial-based scripts include the trial index in the flat list.
- [x] Cluster 4 — sequence scripts (`d1`–`d7`): same refactor;
      `d7` uses composite key `(n_notes, source)`.
- [x] Cluster 5 — environmental + ensemble (`e3`, `f1`, `f2`): apply per the
      stratification table.
- [x] `z1` — wire `--sample-n` to override `--n-per-family` semantics; key =
      `instrument_family_str`. Validate against the existing NSynth-valid pool.
- [x] Extend `get_run_metadata()` (or add a sibling helper) so every save
      records `sample_n`, `sample_seed`, `total_available`, `stratified_by`.
      Echo these in `summary_lines` for the human-readable `.txt`.
- [x] Smoke test end-to-end:
      `pitchbench --id a1 --sample-n 10 --preview` shows exactly 10 stimuli
      and is reproducible across reruns; same for one trial-based script
      (e.g. `d7`) and `z1`.
- [x] Update `README.md` with a "Sampling" section showing the two flags and a
      worked example.
