# Config → CLI flag overhaul

**Date:** 2026-05-04
**Status:** planned

## Goal
Make every hyperparameter in `src/pitchbench/config.py` overridable from the
command line, so an experiment run can be reconfigured (sample rate, MIDI
range, tone duration, …) without code edits, and the chosen values are
recorded for reproducibility. Today most config values are only changeable by
editing the file or (for a few) via env var.

## Context

### Why now
Today's flag surface is uneven: `--models`, `--sources`, `--seed`,
`--n-trials`, `--n-notes`, `--n-per-family`, `--top-k`, `--k`, `--chorales`
exist on various scripts, while audio params (`SAMPLE_RATE`, `NOTE_ON_DUR`,
`RELEASE_DUR`, `FADE_OUT`, `TARGET_PEAK`), pitch defaults
(`DEFAULT_DURATIONS_MS`, `DEFAULT_PITCHES`, `DEFAULT_MIDI_MIN`,
`DEFAULT_MIDI_MAX`), and `DEFAULT_MODEL` have no flag at all. A unified
mechanism turns config into a first-class experiment input.

### Relationship to other plans
- `2026-05-04-sample-n-per-experiment-plan.md` introduces `--sample-n` /
  `--sample-seed` directly in `run.py`. Once **this** plan lands, those flags
  should migrate to use the same machinery rather than being bespoke.

### Open design decisions (resolve in Phase 1)
1. **Which constants are flag-able?**
   - **Yes:** audio params (`SAMPLE_RATE`, `NOTE_ON_DUR`, `RELEASE_DUR`,
     `FADE_OUT`, `TARGET_PEAK`), pitch defaults (`DEFAULT_*`), `DEFAULT_MODEL`.
   - **Probably no (keep env-var only):** `SF2_PATH`, `MODEL_URLS`,
     `OPENROUTER_BASE_URL`, runtime dirs (`DATA_DIR`, …) — these are
     deployment-environment concerns, not experiment hyperparameters.
   - **Open:** structured constants like `GM_PROGRAMS_V1`, `WAVEFORMS`,
     `MODELS` — these are catalogues, not scalars; not obvious how a single
     flag should override them.
2. **Override mechanism.** Two candidates:
   - **(a) Mutate `pitchbench.config` attributes at CLI parse time**, before
     `importlib.import_module(...)` of the experiment script. Pro: scripts
     don't change. Con: import-time constants captured in a module
     (e.g. `MIDI_MIN = config.DEFAULT_MIDI_MIN` at top-of-file in many
     scripts) snapshot the *original* value before mutation, so this only
     works if scripts read `config.X` lazily.
   - **(b) Pass a resolved-config object** through `run_experiment(name,
     extra, config)` and have scripts read from it. Pro: explicit and clean.
     Con: large refactor — every script currently imports config directly.
   - Likely answer: (a) plus a one-time pass to convert top-of-file
     `MIDI_MIN = config.X` into lazy `config.X` reads at call sites.
3. **Argparse generation.** Hand-write per constant, or auto-generate from a
   typed schema (e.g. a `pydantic` / `dataclass` description of config)? Hand-
   written is faster to land; auto-generated is sturdier as config grows.
4. **Result-JSON logging.** Every overridden value must end up in
   `metadata` of every result file. Mechanism: `get_run_metadata()` snapshots
   the resolved config (or just the changed-from-default subset) and embeds
   it. Decide whether to log full config or just the diff.
5. **Per-script `DEFAULT_SEED` / `DEFAULT_N_TRIALS` etc.** — these are config-
   shaped constants currently scattered across scripts. Phase 3 migrates them
   into `config.py` so they flow through the same flag machinery.

### Files this touches (final scope)
- `src/pitchbench/config.py` — possibly add a typed schema/dataclass.
- `src/pitchbench/experiments/run.py` — argparse generation, override logic.
- `src/pitchbench/experiments/helpers/results.py` — log resolved config.
- All 30 scripts (lazy-read pass, then per-script `DEFAULT_*` migration in
  Phase 3).
- `tests/` — override + reproducibility tests.
- `README.md`.

## Phase 1: Design + thin slice
**Purpose:** Pin down the override mechanism by building it end-to-end for a
small, representative set of constants and proving it works without breaking
any existing experiment.

### Action items
- [ ] Audit `config.py` and produce a written list (in this file under
      "Decisions locked in") of: (a) flag-able scalar constants, (b)
      env-var-only constants, (c) structured constants deferred to later.
- [ ] Decide override mechanism (option a vs b above) and document the
      decision with the rationale. Assume option (a) unless something blocks
      it.
- [ ] Audit all 30 scripts for top-of-file snapshots (`MIDI_MIN =
      config.DEFAULT_MIDI_MIN` and friends) and list which need to become
      lazy `config.X` reads.
- [ ] Implement override plumbing in `run.py`: parse known config flags,
      mutate `pitchbench.config` attributes before importing the experiment
      module. Use a small allow-list of constants for this phase:
      `DEFAULT_MIDI_MIN`, `DEFAULT_MIDI_MAX`, `SAMPLE_RATE`, `DEFAULT_MODEL`.
- [ ] Convert top-of-file snapshots in scripts that touch the four pilot
      constants to lazy reads.
- [ ] Extend `get_run_metadata()` to record resolved config values for the
      pilot set (full snapshot of those four).
- [ ] Add `tests/test_config_overrides.py`: assert that
      `pitchbench --id a1 --midi-min 60 --midi-max 72 --preview` produces
      exactly 13 pitches per source and that the resulting metadata records
      `midi_min=60, midi_max=72`.
- [ ] Update `README.md` with the new override pattern (one example).
- [ ] Confirm no regression: `pitchbench --id a1 --preview` and
      `pitchbench --id d7 --preview` produce identical output bytes to a
      pre-change baseline.

## Phase 2: Roll out to all flag-able config constants
**Purpose:** Once the mechanism is proven on the pilot set, extend the
allow-list to cover every flag-able scalar in `config.py`, doing the
top-of-file lazy-read conversion for any remaining scripts.

_Action items will be finalized when Phase 1 completes._

## Phase 3: Migrate per-script `DEFAULT_*` into `config.py`
**Purpose:** Pull the `DEFAULT_SEED`, `DEFAULT_N_TRIALS`, `DEFAULT_TOP_K`,
`DEFAULT_K`, `DEFAULT_N_PER_FAMILY`, `DEFAULT_CHORALES` constants currently
defined inside individual scripts into `config.py` (namespaced per-experiment
or globally, TBD), so they flow through the same override machinery and can
be set globally in one place.

_Action items will be finalized when Phase 2 completes._

## Out of scope
- Loading config from a YAML/TOML file. (Could be a follow-up — same
  override machinery would back it.)
- Replacing env-var-driven settings (`MODEL_URLS`, paths) with flags. Those
  remain env-only because they describe the runtime environment, not the
  experiment.
