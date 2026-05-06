"""
Shared runner for category-C experiments (simultaneous-pitch tasks).

Each cat-C script (c1..c4) reduces to a CatCSpec dataclass plus a small
``build_conditions``/``wav_for``/``prompts_fn``/``record_fn`` quartet.

The category covers four task families that all stimuli-render through
``engine.chord(midis, source_arg, duration_ms)``:

* **interval** (c1) — semitones between two simultaneous tones.
* **count**    (c2) — distinct pitch count of a chord.
* **set**      (c4) — which pitches are sounding (4-format: MIDI/SPN/Doremi/Hz).
* **quality**  (c4) — chord quality, optionally also the root.

The lifecycle (argparse → sampling → audio → dispatch → save_results)
mirrors :mod:`cat_a` and :mod:`cat_b`. ``CatCSpec.headline_metrics`` lists
the metric names whose ``<name>_correct`` columns flow into
``summary["accuracy"]`` and the headline accuracies CSV. Auxiliary
diagnostic columns (``off_by``, ``tp``/``fp``/``fn``, ``recall``,
``precision``) live in the per-stimulus records CSV but stay invisible
to the auto-marginals because their column names do not end in
``_correct``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies, get_run_metadata, make_run_dir,
    save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import (
    apply_default_sampling, export_sampled_conditions_csv, sampling_summary_lines,
)


# ── Spec ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CatCSpec:
    """Per-experiment configuration for the cat-C unified runner.

    Each condition produces ONE record. ``prompts_fn`` returns a mapping
    ``{prompt_name: prompt_text}`` whose keys are queried in turn (the
    runner caches one query per prompt). ``record_fn`` is then called with
    the condition, the wav path, and the raw responses keyed by the same
    prompt names; it must return a dict with at least one ``<name>_correct``
    integer column for each entry in ``headline_metrics``.
    """
    exp_name:            str
    build_conditions_fn: Callable[[], list[dict]]
    wav_fn:              Callable[[dict], Path]

    task_type: str  # "interval" | "count" | "set" | "quality"

    prompts_fn: Callable[[dict], dict[str, str]]
    record_fn:  Callable[[dict, str, dict[str, str]], dict]

    # Names whose `<name>_correct` columns flow into summary["accuracy"]
    # (and therefore into accuracies_<model>.csv). Order = display order.
    headline_metrics: tuple[str, ...] = ()

    # Shared (mirrors CatASpec / CatBSpec).
    primary_filter: Callable[[dict], bool] | None = None
    pair_key:       tuple[str, ...] = ()
    record_extras:  tuple[str, ...] = ()
    label_fn:       Callable[[dict], str] | None = None
    metadata_fn:    Callable[[], dict] | None = None

    def __post_init__(self) -> None:
        if self.task_type not in ("interval", "count", "set", "quality"):
            raise ValueError(
                f"{self.exp_name}: task_type must be one of "
                "'interval', 'count', 'set', 'quality'"
            )
        if not self.headline_metrics:
            raise ValueError(
                f"{self.exp_name}: headline_metrics must list at least one name"
            )


# ── Source-type fallback ─────────────────────────────────────────────────────

def _src_type(source: str) -> str:
    return "waveform" if source in config.WAVEFORMS else "instrument"


# ── Summary ──────────────────────────────────────────────────────────────────

def _accuracy(records: list[dict], col: str) -> float:
    if not records:
        return 0.0
    vals = [r[col] for r in records if isinstance(r.get(col), (int, float, bool))]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def compute_summary(
    records:          list[dict],
    headline_metrics: tuple[str, ...],
    primary_filter:   Callable[[dict], bool] | None,
) -> dict[str, Any]:
    """Build the cat-C summary dict.

    Layout mirrors cat-A / cat-B:

    * ``total``, ``condition_n``, ``accuracy`` always present.
    * When ``primary_filter`` is set, also ``baseline_n`` and
      ``accuracy_baseline``.
    * ``accuracy = {n, <m1>, <m2>, ...}`` where each ``<m>`` is averaged
      over records whose ``<m>_correct`` column is numeric (records lacking
      the column are skipped, so c4-style "root only on
      root_and_quality" metrics average correctly).
    """
    total = len(records)
    if primary_filter is None:
        primary, secondary = records, []
    else:
        primary, secondary = [], []
        for r in records:
            (primary if primary_filter(r) else secondary).append(r)

    def _acc_for(rs: list[dict]) -> dict[str, Any]:
        return {
            "n": len(rs),
            **{m: _accuracy(rs, f"{m}_correct") for m in headline_metrics},
        }

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
    spec:        CatCSpec,
    summary:     dict[str, Any],
    sample_info: dict | None,
) -> list[str]:
    """Uniform two-column text block (Primary / Baseline if applicable)."""
    has_baseline = "accuracy_baseline" in summary
    lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli (total)     : {summary['total']}",
    ]
    metrics = list(spec.headline_metrics)
    acc_p   = summary["accuracy"]
    if has_baseline:
        acc_b = summary["accuracy_baseline"]
        lines += [
            f"  Stimuli (condition) : {summary['condition_n']}",
            f"  Stimuli (baseline)  : {summary['baseline_n']}",
            "",
            f"  {'Metric':>10}  {'Condition':>9}  {'Baseline':>9}",
            f"  {'─' * 34}",
        ]
        for m in metrics:
            lines.append(
                f"  {m.upper():>10}  {acc_p[m]:>9.1%}  {acc_b[m]:>9.1%}"
            )
        lines += [
            "",
            "  Headline 'accuracy' is computed on the condition subset only "
            "(see CatCSpec.primary_filter).",
        ]
    else:
        lines += [
            f"  Stimuli (condition) : {summary['condition_n']}  (all records)",
            "",
            f"  {'Metric':>10}  {'Accuracy':>9}",
            f"  {'─' * 23}",
        ]
        for m in metrics:
            lines.append(f"  {m.upper():>10}  {acc_p[m]:>9.1%}")
    return lines


# ── Sampling with optional pair-expansion ────────────────────────────────────

def _sample_conditions(
    spec:        CatCSpec,
    all_conds:   list[dict],
    sample_n:    int | None,
    sample_seed: int,
) -> tuple[list[dict], dict]:
    """Stratified sampling, optionally pair-expanded around primary records."""
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
    spec:        CatCSpec,
    model_name:  str,
    conds:       list[dict],
    run_dir:     Path,
    sample_info: dict | None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1 — generate audio sequentially.
    jobs: list[dict] = []
    for c in conds:
        try:
            wav = str(spec.wav_fn(c))
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        prompts = spec.prompts_fn(c)
        jobs.append({"wav": wav, "cond": c, "prompts": prompts})

    # Phase 2 — dispatch (one query per prompt key per cond → one record per cond).
    label = spec.label_fn or (lambda j: f"{j['cond'].get('source', '?'):>12}  task={spec.task_type}")

    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        prompts = job["prompts"]
        responses: dict[str, str] = {}
        for name, prompt in prompts.items():
            out = query_alm(model_name, job["wav"], prompt)
            responses[name] = (out["result"] or "").strip()
        record = spec.record_fn(c, job["wav"], responses)
        # Inject `prompt_<name>` columns + record_extras (carried from cond).
        for name, prompt in prompts.items():
            record.setdefault(f"prompt_{name}", prompt)
        for k in spec.record_extras:
            record.setdefault(k, c.get(k))
        record.setdefault("source", c.get("source"))
        record.setdefault(
            "source_type",
            c.get("source_type", _src_type(str(c.get("source", "")))),
        )
        record.setdefault("wav", job["wav"])
        return record

    def _audit_str(record: dict) -> str:
        """Per-metric ``<name>=<pred> ✅|❌`` blocks.

        The label already carries cond info (the ground truth is implicit in
        e.g. ``n=4 maj7 root=C4``), so the audit just needs to show the
        prediction and the correctness symbol per headline metric.
        """
        blocks: list[str] = []
        for m in spec.headline_metrics:
            pred = record.get(f"{m}_pred")
            ok   = bool(record.get(f"{m}_correct"))
            sym  = "✅" if ok else "❌"
            blocks.append(f"{m}={pred!s} {sym}")
        return "  | ".join(blocks)

    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: label(j),
        result_label_fn=lambda j, r: f"{label(j)}  {_audit_str(r)}",
    )
    full_records: list[dict] = [r for r in raw if r is not None]

    # Phase 3 — summary + lines.
    summary       = compute_summary(full_records, spec.headline_metrics, spec.primary_filter)
    summary_lines = _format_summary_lines(spec, summary, sample_info)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    # Phase 4 — save (no plots in this pass).
    extra_meta = spec.metadata_fn() if spec.metadata_fn else {}
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        task_type=spec.task_type,
        headline_metrics=list(spec.headline_metrics),
        primary_filter_set=spec.primary_filter is not None,
        record_extras=list(spec.record_extras),
        **extra_meta,
        **(sample_info or {}),
    )
    save_results(
        spec.exp_name, model_name, full_records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=tuple(f"{m}_correct" for m in spec.headline_metrics),
    )
    return summary


def run_cat_c_experiment(spec: CatCSpec, *, mode: str = "run") -> dict | None:
    """Top-level cat-C entry point. ``mode`` ∈ {"preview", "run"}."""
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
        return None

    target_models = args.models or list(config.MODELS)
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
