"""
Shared runner for category-B experiments (timing + pitch-at-time tasks).

Each cat-B script (b1..b5) reduces to a CatBSpec dataclass plus a small
``build_conditions``/``wav_for``/``gt_timestamps_fn`` (and prompts).

The category contains two task families:

* **pitch tasks** (b1, b4) — predict the pitch of a tone (b1: in silence;
  b4: at a queried time inside a sequence). Scored with the standard
  4-format (MIDI / SPN / Doremi / Hz) record from
  ``standard_pitch_record``, then stripped of continuous Hz / legacy ABC
  columns just like cat-A.

* **timing tasks** (b2, b3, b5) — predict timestamps. The unified
  ``score_timestamps`` function handles single-pair and multi-pair
  predictions identically: ``correct = 1`` iff the prediction count
  matches the GT count *and* every pair is within
  ``config.BENCHMARK_TIMESTAMP_TOLERANCE_MS``. Per-stimulus records keep
  the diagnostic columns (``iou``, ``within_100ms_both``,
  ``onset_s_pred``, ``valid``, …) for offline audit, but **only**
  ``correct`` flows into the headline summary / ``accuracies_<model>.csv``
  via the auto-marginals (the diagnostic columns deliberately don't end
  in ``_correct``).

The lifecycle (argparse → sampling → audio → dispatch → save_results)
mirrors :mod:`cat_a` and reuses ``cat_a._sample_conditions`` and
``cat_a._strip_for_csv`` to avoid duplication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable

import pitchbench.config as config
import pitchbench.sound.engine as engine
from pitchbench.model.query import (
    get_model_info, query_alm, query_four_formats,
)
from pitchbench.experiments.helpers.audit import (
    audit_line, pitch_record_audit_str,
)
from pitchbench.model.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    POINT_PITCH_FORMATS,
    PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI, PROMPT_SPN,
    midi_to_note, parse_mm_ss_cc, standard_pitch_record, timing_metrics,
)
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies, get_run_metadata, make_run_dir,
    save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import (
    apply_default_sampling, export_sampled_conditions_csv, sampling_summary_lines,
)

# Reuse cat-A primitives — the strip-rules, pair-expansion sampler, and
# CSV layout are identical across categories.
from pitchbench.experiments.helpers.cat_a import (
    _strip_for_csv as _strip_for_csv_pitch,
)


PITCH_FORMATS:  tuple[str, ...] = POINT_PITCH_FORMATS
TIMING_METRIC:  str             = "correct"


# ── Spec ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CatBSpec:
    """Per-experiment configuration for the cat-B unified runner."""
    exp_name:            str
    build_conditions_fn: Callable[[], list[dict]]
    wav_fn:              Callable[[dict], Path]

    task_type: str  # "pitch" or "timing"

    # ── pitch tasks (b1, b4) ─────────────────────────────────────
    prompt_prefix:  str | None = None
    prompts_fn:     Callable[[dict], dict[str, str]] | None = None

    # ── timing tasks (b2, b3, b5) ────────────────────────────────
    prompt:           str | Callable[[dict], str] | None = None
    gt_timestamps_fn: Callable[[dict], list[float]] | None = None
    tolerance_ms:     int | None = None  # None ⇒ config default

    # ── shared (mirrors CatASpec) ────────────────────────────────
    primary_filter: Callable[[dict], bool] | None = None
    pair_key:       tuple[str, ...] = ()
    record_extras:  tuple[str, ...] = ()
    label_fn:       Callable[[dict], str] | None = None
    metadata_fn:    Callable[[], dict] | None = None

    def __post_init__(self) -> None:
        if self.task_type not in ("pitch", "timing"):
            raise ValueError(f"{self.exp_name}: task_type must be 'pitch' or 'timing'")
        if self.task_type == "pitch":
            if (self.prompt_prefix is None) == (self.prompts_fn is None):
                raise ValueError(
                    f"{self.exp_name}: pitch tasks need exactly one of "
                    "prompt_prefix or prompts_fn"
                )
        else:  # timing
            if self.prompt is None:
                raise ValueError(f"{self.exp_name}: timing tasks need a prompt")
            if self.gt_timestamps_fn is None:
                raise ValueError(
                    f"{self.exp_name}: timing tasks need a gt_timestamps_fn"
                )

    @property
    def effective_tolerance_ms(self) -> int:
        return (
            self.tolerance_ms
            if self.tolerance_ms is not None
            else int(config.BENCHMARK_TIMESTAMP_TOLERANCE_MS)
        )


# ── Unified timing scorer (single + sequence) ────────────────────────────────

def score_timestamps(
    gt:           list[float],
    pred:         list[float],
    tolerance_ms: int,
) -> int:
    """Binary correctness for a list of GT timestamps vs predicted timestamps.

    Returns ``1`` iff ``len(pred) == len(gt)`` and every paired
    ``(gt[i], pred[i])`` differs by at most ``tolerance_ms`` milliseconds.
    Order-preserving — the cat-B prompts all instruct the model to emit
    timestamps in chronological order. Used uniformly by:

    * b2 — single tone (2 timestamps: onset, offset)
    * b3 — named target among distractors (2 timestamps)
    * b5 — every note in a sequence (2 × n_notes timestamps)
    """
    if len(pred) != len(gt):
        return 0
    tol_s = tolerance_ms / 1000.0
    return int(all(abs(g - p) <= tol_s for g, p in zip(gt, pred)))


# ── Prompt assembly ──────────────────────────────────────────────────────────

def _pitch_prompts(spec: CatBSpec, cond: dict) -> dict[str, str]:
    if spec.prompts_fn is not None:
        out = spec.prompts_fn(cond)
        return {fmt: out.get(fmt, "") for fmt in PITCH_FORMATS}
    pref = spec.prompt_prefix or ""
    return {
        "midi":   pref + PROMPT_MIDI,
        "spn":    pref + PROMPT_SPN,
        "doremi": pref + PROMPT_DOREMI,
        "hz":     pref + PROMPT_HZ,
    }


def _timing_prompt(spec: CatBSpec, cond: dict) -> str:
    p = spec.prompt
    return p(cond) if callable(p) else (p or "")


# ── Source-type fallback ─────────────────────────────────────────────────────

def _src_type(source: str) -> str:
    return "waveform" if source in config.WAVEFORMS else "instrument"


# ── Per-stimulus record builders ─────────────────────────────────────────────

def _pitch_record(spec: CatBSpec, c: dict, wav: str, prompts: dict, results: tuple) -> dict:
    raw_m, raw_s, raw_d, raw_h = results
    extras = {k: c[k] for k in spec.record_extras if k in c}
    rec = standard_pitch_record(
        wav=wav,
        source=c["source"],
        source_type=c.get("source_type", _src_type(c["source"])),
        midi_gt=c["midi"],
        raw_midi=raw_m, raw_spn=raw_s, raw_doremi=raw_d, raw_hz=raw_h,
        prompt_midi=prompts["midi"], prompt_spn=prompts["spn"],
        prompt_doremi=prompts["doremi"], prompt_hz=prompts["hz"],
        **extras,
    )
    # Drop the same Hz/ABC continuous columns cat-A drops.
    return _strip_for_csv_pitch([rec])[0]


def _timing_record(spec: CatBSpec, c: dict, wav: str, prompt: str, raw_response: str) -> dict:
    """Per-stimulus record for a timing task.

    Keeps full diagnostic columns from ``timing_metrics`` (when applicable —
    only meaningful when there are exactly 2 GT timestamps); the auto-
    marginals only ever pick up ``correct`` because the diagnostic
    column names don't end in ``_correct``.
    """
    extras = {k: c[k] for k in spec.record_extras if k in c}
    gt    = (spec.gt_timestamps_fn or (lambda _: []))(c)
    pred  = parse_mm_ss_cc(raw_response or "")
    tol_ms  = spec.effective_tolerance_ms
    correct = score_timestamps(gt, pred, tol_ms)

    rec: dict[str, Any] = {
        **extras,
        "source":       c["source"],
        "source_type":  c.get("source_type", _src_type(c["source"])),
        "n_gt":         len(gt),
        "n_pred":       len(pred),
        "correct":      correct,
        "raw_response": (raw_response or "").strip(),
        "wav":          wav,
        "prompt":       prompt,
    }

    # Two-timestamp tasks (b2, b3): expose the legacy diagnostic block so
    # offline analysis still has IoU / abs_error / within_100ms_both /
    # within_500ms_both. These intentionally do NOT feed the summary —
    # their column names don't end in `_correct`.
    if len(gt) == 2:
        on_gt, off_gt = gt[0], gt[1]
        on_p          = pred[0] if len(pred) >= 1 else None
        off_p         = pred[1] if len(pred) >= 2 else None
        m = timing_metrics(on_gt, off_gt, on_p, off_p)
        rec.update({
            "onset_s_gt":        round(on_gt, 4),
            "offset_s_gt":       round(off_gt, 4),
            "onset_s_pred":      on_p,
            "offset_s_pred":     off_p,
            "iou":               m["iou"],
            "abs_error_on":      m["abs_error_on"],
            "abs_error_off":     m["abs_error_off"],
            "within_100ms_both": m["within_100ms_both"],
            "within_500ms_both": m["within_500ms_both"],
            "valid":             int(m["valid"]),
        })
    return rec


# ── Summary ──────────────────────────────────────────────────────────────────

def _accuracy(records: list[dict], col: str) -> float:
    if not records:
        return 0.0
    vals = [r[col] for r in records if isinstance(r.get(col), (int, float, bool))]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def compute_summary(
    records:        list[dict],
    task_type:      str,
    primary_filter: Callable[[dict], bool] | None,
) -> dict[str, Any]:
    """Build the cat-B summary dict.

    Layout mirrors cat-A's ``compute_summary``:
      - ``total``, ``condition_n``, ``accuracy`` always present.
      - When ``primary_filter`` is set, also ``baseline_n`` and
        ``accuracy_baseline``.
      - For pitch tasks, ``accuracy`` is ``{midi, spn, doremi, hz, n}``.
      - For timing tasks, ``accuracy`` is ``{correct, n}`` (single-key dict
        so the auto-flattener emits one ``accuracy.correct`` row in the
        accuracies CSV — matching the ``by_<iv>.<value>.correct`` rows).
    """
    total = len(records)
    if primary_filter is None:
        primary, secondary = records, []
    else:
        primary, secondary = [], []
        for r in records:
            (primary if primary_filter(r) else secondary).append(r)

    def _acc_for(rs: list[dict]) -> dict[str, Any]:
        if task_type == "pitch":
            return {
                "n": len(rs),
                **{fmt: _accuracy(rs, f"{fmt}_correct") for fmt in PITCH_FORMATS},
            }
        return {"n": len(rs), TIMING_METRIC: _accuracy(rs, TIMING_METRIC)}

    out: dict[str, Any] = {
        "total":       total,
        "condition_n": len(primary),
        "accuracy":    _acc_for(primary),
    }
    if primary_filter is not None:
        out["baseline_n"]        = len(secondary)
        out["accuracy_baseline"] = _acc_for(secondary)
    return out


def _format_summary_lines(
    spec:        CatBSpec,
    summary:     dict[str, Any],
    sample_info: dict | None,
) -> list[str]:
    """Uniform two-column text block (Primary / Baseline if applicable)."""
    has_baseline = "accuracy_baseline" in summary
    lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli (total)     : {summary['total']}",
    ]
    metrics = (
        list(PITCH_FORMATS) if spec.task_type == "pitch" else [TIMING_METRIC]
    )
    acc_p = summary["accuracy"]
    if has_baseline:
        acc_b = summary["accuracy_baseline"]
        lines += [
            f"  Stimuli (condition) : {summary['condition_n']}",
            f"  Stimuli (baseline)  : {summary['baseline_n']}",
            "",
            f"  {'Metric':>8}  {'Condition':>9}  {'Baseline':>9}",
            f"  {'─' * 32}",
        ]
        for m in metrics:
            lines.append(
                f"  {m.upper():>8}  {acc_p[m]:>9.1%}  {acc_b[m]:>9.1%}"
            )
        lines += [
            "",
            "  Headline 'accuracy' is computed on the condition subset only "
            "(see CatBSpec.primary_filter).",
        ]
    else:
        lines += [
            f"  Stimuli (condition) : {summary['condition_n']}  (all records)",
            "",
            f"  {'Metric':>8}  {'Accuracy':>9}",
            f"  {'─' * 22}",
        ]
        for m in metrics:
            lines.append(f"  {m.upper():>8}  {acc_p[m]:>9.1%}")
    if spec.task_type == "timing":
        lines.append(
            f"  (correctness tolerance = ±{spec.effective_tolerance_ms} ms)"
        )
    return lines


# ── Sampling with optional pair-expansion ────────────────────────────────────

def _sample_conditions(
    spec:        CatBSpec,
    all_conds:   list[dict],
    sample_n:    int | None,
    sample_seed: int,
) -> tuple[list[dict], dict]:
    """Stratified sampling, optionally pair-expanded around primary records.

    When ``spec.pair_key`` is set together with ``spec.primary_filter``,
    sampling runs on primaries only and each sampled stim brings every
    record sharing its ``pair_key`` value (the matched baseline twin).
    Otherwise this is just a pass-through to ``apply_default_sampling``.
    """
    if not (spec.pair_key and spec.primary_filter is not None):
        return apply_default_sampling(
            spec.exp_name, all_conds, sample_n, sample_seed,
        )

    primaries = [c for c in all_conds if spec.primary_filter(c)]
    sampled_primaries, s_meta = apply_default_sampling(
        spec.exp_name, primaries, sample_n, sample_seed,
    )
    sampled_keys = {tuple(c[k] for k in spec.pair_key) for c in sampled_primaries}
    paired = [c for c in all_conds if tuple(c[k] for k in spec.pair_key) in sampled_keys]

    s_meta = dict(s_meta)
    s_meta["pair_key"]        = list(spec.pair_key)
    s_meta["sampled_primary"] = len(sampled_primaries)
    s_meta["sampled_total"]   = len(paired)
    s_meta["total_available"] = len(all_conds)
    return paired, s_meta


# ── Argparse ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview",     action="store_true")
    parser.add_argument("--models",      nargs="+", metavar="MODEL")
    parser.add_argument("--sample-n",    type=int, default=None, metavar="N")
    parser.add_argument("--sample-seed", type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


# ── Per-model run ────────────────────────────────────────────────────────────

def run_one_model(
    spec:        CatBSpec,
    model_name:  str,
    conds:       list[dict],
    run_dir:     Path,
    sample_info: dict | None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {model_name}")

    # Phase 1 — generate audio sequentially.
    jobs: list[dict] = []
    for c in conds:
        try:
            wav = str(spec.wav_fn(c))
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        if spec.task_type == "pitch":
            prompts = _pitch_prompts(spec, c)
            jobs.append({"wav": wav, "cond": c, "prompts": prompts})
        else:
            prompt = _timing_prompt(spec, c)
            jobs.append({"wav": wav, "cond": c, "prompt": prompt})

    # Phase 2 — dispatch.
    label = spec.label_fn or (lambda j: f"{midi_to_note(j['cond'].get('midi', 0)):>4} {j['cond']['source']}")

    if spec.task_type == "pitch":
        def _query_pitch(job: dict) -> dict:
            r_m, r_s, r_d, r_h = query_four_formats(
                model_name, job["wav"],
                job["prompts"]["midi"], job["prompts"]["spn"],
                job["prompts"]["doremi"], job["prompts"]["hz"],
                verbose=False,
            )
            return _pitch_record(
                spec, job["cond"], job["wav"], job["prompts"],
                (r_m["result"], r_s["result"], r_d["result"], r_h["result"]),
            )
        raw = dispatch(
            jobs, _query_pitch,
            model_name=model_name,
            label_fn=lambda j: label(j),
            result_label_fn=lambda j, r: pitch_record_audit_str(r, label=label(j)),
        )
    else:
        def _query_timing(job: dict) -> dict:
            out = query_alm(model_name, job["wav"], job["prompt"])
            return _timing_record(
                spec, job["cond"], job["wav"], job["prompt"], out["result"] or "",
            )
        raw = dispatch(
            jobs, _query_timing,
            model_name=model_name,
            label_fn=lambda j: label(j),
            result_label_fn=lambda j, r: audit_line(
                label(j),
                gt=f"n_gt={r['n_gt']}",
                pred=f"n_pred={r['n_pred']}",
                correct=bool(r["correct"]),
            ),
        )

    full_records: list[dict] = [r for r in raw if r is not None]

    # Phase 3 — summary + lines.
    summary       = compute_summary(full_records, spec.task_type, spec.primary_filter)
    summary_lines = _format_summary_lines(spec, summary, sample_info)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    # Phase 4 — save (no plots).
    extra_meta = spec.metadata_fn() if spec.metadata_fn else {}
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        task_type=spec.task_type,
        tolerance_ms=spec.effective_tolerance_ms if spec.task_type == "timing" else None,
        primary_filter_set=spec.primary_filter is not None,
        record_extras=list(spec.record_extras),
        **extra_meta,
        **(sample_info or {}),
    )
    formats = PITCH_FORMATS if spec.task_type == "pitch" else ()
    extra_metrics = () if spec.task_type == "pitch" else (TIMING_METRIC,)
    save_results(
        spec.exp_name, model_name, full_records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=formats,
        extra_metrics=extra_metrics,
    )
    return summary


def run_cat_b_experiment(spec: CatBSpec, *, mode: str = "run") -> dict | None:
    """Top-level cat-B entry point. ``mode`` ∈ {"preview", "run"}."""
    engine.set_exp(spec.exp_name)
    args = _parse_args()
    if args.preview:
        mode = "preview"

    all_conds = spec.build_conditions_fn()
    conds, s_meta = _sample_conditions(spec, all_conds, args.sample_n, args.sample_seed)


    n_skipped = 0
    for c in conds:
        try:
            spec.wav_fn(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            n_skipped += 1

    if mode == "preview":
        export_sampled_conditions_csv(spec.exp_name, conds)
        print(f"Experiment : {spec.exp_name}")
        print(f"Stimuli    : {len(conds)}  (skipped: {n_skipped})")
        print(f"Audio dir  : {config.AUDIO_DIR}/{spec.exp_name}")
        for line in sampling_summary_lines(s_meta):
            print(line)
        print("\nRun without --preview to query the model(s).")
        
        # for c in conds:
        #     print(f"\n  Condition example:\n    {c}\n    WAV path: {spec.wav_fn(c)}")
        return None

    if not args.models:
        raise SystemExit("--models is required: specify one or more model URLs or slugs")
    target_models = args.models
    print(f"Experiment : {spec.exp_name}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(spec.exp_name)
    all_summaries: dict[str, dict] = {}
    for m in target_models:
        all_summaries[m] = run_one_model(spec, m, conds, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, spec.exp_name)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))


# ── Generate / evaluate from Parquet ────────────────────────────────────────

import json as _json

from pitchbench.experiments.helpers.music import (
    PC_TO_SOLFEGE, midi_to_freq,
)


def generate_cat_b_data(
    spec:        CatBSpec,
    *,
    sample_n:    int | None = None,
    sample_seed: int = config.DEFAULT_SAMPLE_SEED,
) -> int:
    """Generate audio and write/append data.parquet for cat-B experiments.

    Pitch tasks produce four prompt columns (midi/spn/doremi/hz) and GT pitch
    columns.  Timing tasks produce a single prompt column and a
    ``gt_timestamps_json`` column (JSON array of floats in seconds).
    When ``sample_n`` is given, only that many conditions are generated.
    """
    from pitchbench.experiments.helpers.data import append_dataset, dataset_path
    from pitchbench.experiments.helpers.sampling import apply_default_sampling

    parquet_path = dataset_path(spec.exp_name)
    existing_paths: set[str] = set()
    if parquet_path.exists():
        import pandas as _pd
        existing_paths = set(_pd.read_parquet(parquet_path)["audio_path"].astype(str).tolist())

    engine.set_exp(spec.exp_name)
    conds = spec.build_conditions_fn()
    if sample_n is not None:
        conds, _ = apply_default_sampling(spec.exp_name, conds, sample_n, sample_seed)
    rows: list[dict] = []
    for c in tqdm(conds, desc=spec.exp_name, unit="cond", leave=False):
        try:
            wav_path = spec.wav_fn(c)
        except ValueError as exc:
            tqdm.write(f"  [SKIP] {exc}")
            continue
        audio_path_str = str(wav_path)
        if audio_path_str in existing_paths:
            continue

        row: dict = {"audio_path": audio_path_str, "condition_json": _json.dumps(c), **c}

        if spec.task_type == "pitch":
            prompts = _pitch_prompts(spec, c)
            midi_val: int = c["midi"]
            row.update({
                "prompt_midi":        prompts["midi"],
                "prompt_spn":         prompts["spn"],
                "prompt_doremi":      prompts["doremi"],
                "prompt_hz":          prompts["hz"],
                "gt_midi":            midi_val,
                "gt_spn":             midi_to_note(midi_val),
                "gt_doremi":          PC_TO_SOLFEGE.get(midi_val % 12, "?"),
                "gt_hz":              round(midi_to_freq(midi_val), 4),
            })
        else:
            prompt = _timing_prompt(spec, c)
            gt_ts  = (spec.gt_timestamps_fn or (lambda _: []))(c)
            row.update({
                "prompt":              prompt,
                "gt_timestamps_json":  _json.dumps(gt_ts),
            })

        rows.append(row)
        existing_paths.add(audio_path_str)

    n = append_dataset(parquet_path, rows)
    print(f"  {spec.exp_name}: {n} new rows written → {parquet_path}")
    return n


def evaluate_cat_b_from_parquet(
    spec:        CatBSpec,
    model_name:  str,
    run_dir:     Path,
    sample_info: dict | None = None,
    *,
    model_label: str | None = None,
    filter_conditions: bool = False,
) -> dict:
    """Evaluate ``model_name`` on the pre-generated dataset for a cat-B spec."""
    from pitchbench.experiments.helpers.data import (
        dataset_path, filter_rows_to_conditions, read_dataset,
    )

    rows = read_dataset(dataset_path(spec.exp_name))
    # Ordinary evaluation consumes the stored sample independently of local
    # synthesis capabilities. Only analysis presets request condition filtering.
    if filter_conditions:
        expected = spec.build_conditions_fn()
        before = len(rows)
        rows = filter_rows_to_conditions(rows, expected)
        print(f"  Filtered to analysis config: {len(rows)} / {before} rows")
    total = len(rows)
    rows, sample_info = apply_default_sampling(
        spec.exp_name, rows,
        cli_sample_n=(sample_info or {}).get("sample_n"),
        cli_seed=(sample_info or {}).get("sample_seed", config.DEFAULT_SAMPLE_SEED),
    )
    if len(rows) < total:
        print(f"  Sampling: {len(rows)} rows")
    info = get_model_info(model_name)
    print(f"\n  Model : {model_name}")

    jobs: list[dict] = []
    for row in rows:
        cond: dict = row.get("_condition") or {
            k: v for k, v in row.items()
            if not k.startswith("prompt") and not k.startswith("gt_")
            and k not in ("audio_path", "condition_json", "_condition",
                          "gt_timestamps_json")
        }
        j: dict = {"wav": row["audio_path"], "cond": cond, "row": row}
        if spec.task_type == "pitch":
            j["prompts"] = {
                "midi":   row.get("prompt_midi", ""),
                "spn":    row.get("prompt_spn", ""),
                "doremi": row.get("prompt_doremi", ""),
                "hz":     row.get("prompt_hz", ""),
            }
        else:
            j["prompt"] = row.get("prompt", "")
            j["gt_ts"]  = _json.loads(row.get("gt_timestamps_json") or "[]")
        jobs.append(j)

    label = spec.label_fn or (
        lambda j: f"{midi_to_note(j['cond'].get('midi', 0)):>4} {j['cond'].get('source', '')}"
    )

    if spec.task_type == "pitch":
        def _query_pitch(job: dict) -> dict:
            c = job["cond"]; row = job["row"]
            r_m, r_s, r_d, r_h = query_four_formats(
                model_name, job["wav"],
                job["prompts"]["midi"], job["prompts"]["spn"],
                job["prompts"]["doremi"], job["prompts"]["hz"],
                verbose=False,
            )
            return _pitch_record(spec, c, job["wav"], job["prompts"],
                                 (r_m["result"], r_s["result"], r_d["result"], r_h["result"]))
        raw = dispatch(jobs, _query_pitch, model_name=model_name,
                       label_fn=lambda j: label(j),
                       result_label_fn=lambda j, r: pitch_record_audit_str(r, label=label(j)))
    else:
        def _query_timing(job: dict) -> dict:
            out = query_alm(model_name, job["wav"], job["prompt"])
            c   = job["cond"]; row = job["row"]
            extras = {k: c.get(k) for k in spec.record_extras if k in c}
            gt = job["gt_ts"]
            pred = parse_mm_ss_cc(out["result"] or "")
            tol_ms = spec.effective_tolerance_ms
            correct = score_timestamps(gt, pred, tol_ms)
            rec: dict = {
                **extras,
                "source":       c.get("source", ""),
                "source_type":  c.get("source_type", ""),
                "n_gt":         len(gt),
                "n_pred":       len(pred),
                "correct":      correct,
                "raw_response": (out["result"] or "").strip(),
                "wav":          job["wav"],
                "prompt":       job["prompt"],
            }
            if len(gt) == 2:
                on_gt, off_gt = gt[0], gt[1]
                on_p  = pred[0] if len(pred) >= 1 else None
                off_p = pred[1] if len(pred) >= 2 else None
                m = timing_metrics(on_gt, off_gt, on_p, off_p)
                rec.update({
                    "onset_s_gt": round(on_gt, 4), "offset_s_gt": round(off_gt, 4),
                    "onset_s_pred": on_p, "offset_s_pred": off_p,
                    "iou": m["iou"], "abs_error_on": m["abs_error_on"],
                    "abs_error_off": m["abs_error_off"],
                    "within_100ms_both": m["within_100ms_both"],
                    "within_500ms_both": m["within_500ms_both"],
                    "valid": int(m["valid"]),
                })
            return rec
        raw = dispatch(jobs, _query_timing, model_name=model_name,
                       label_fn=lambda j: label(j),
                       result_label_fn=lambda j, r: audit_line(
                           label(j), gt=f"n_gt={r['n_gt']}",
                           pred=f"n_pred={r['n_pred']}", correct=bool(r["correct"])))

    full_records: list[dict] = [r for r in raw if r is not None]
    summary       = compute_summary(full_records, spec.task_type, spec.primary_filter)
    summary_lines = _format_summary_lines(spec, summary, sample_info)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    stem_name = model_label or model_name
    extra_meta = spec.metadata_fn() if spec.metadata_fn else {}
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        **({"model_label": model_label} if model_label else {}),
        task_type=spec.task_type,
        tolerance_ms=spec.effective_tolerance_ms if spec.task_type == "timing" else None,
        primary_filter_set=spec.primary_filter is not None,
        record_extras=list(spec.record_extras),
        **extra_meta,
        **(sample_info or {}),
    )
    formats       = PITCH_FORMATS if spec.task_type == "pitch" else ()
    extra_metrics = () if spec.task_type == "pitch" else (TIMING_METRIC,)
    save_results(
        spec.exp_name, stem_name, full_records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=formats,
        extra_metrics=extra_metrics,
    )
    return summary
