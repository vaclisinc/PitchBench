"""
Shared runner for category-A experiments (single-pitch identification under
various stimulus conditions).

Each cat-A script (a1..a5) reduces to a CatASpec dataclass plus a small
``build_conditions`` and ``wav_for`` function. This module owns the rest:
argparse, sampling, dispatch, prompt assembly, scoring (via
``standard_pitch_record``), summary aggregation, the four prediction-vs-truth
confidence-interval plots, and the ``save_results`` call.

Per-stimulus CSV columns are filtered to discrete values only — the
continuous Hz columns (`hz_pred`, `hz_gt`, `hz_abs_error`, `hz_ratio`) and
the legacy `abc_*` aliases are stripped just before write. The full
unstripped record list is kept in memory long enough to draw the plots.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
from pitchbench.experiments.helpers.audit import pitch_record_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    POINT_PITCH_FORMATS,
    PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI, PROMPT_SPN,
    extract_solfege, midi_to_note, note_to_midi,
    solfege_pc_to_midi, standard_pitch_record,
)
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies, get_run_metadata, make_run_dir,
    save_comparison, save_results, _safe_stem,
)
from pitchbench.experiments.helpers.sampling import (
    apply_default_sampling, sampling_summary_lines,
)


FORMATS: tuple[str, ...] = POINT_PITCH_FORMATS


# ── Spec ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CatASpec:
    """Per-experiment configuration for the cat-A unified runner.

    Either ``prompt_prefix`` or ``prompts_fn`` must be provided.
    """
    exp_name:            str
    build_conditions_fn: Callable[[], list[dict]]
    wav_fn:              Callable[[dict], Path]

    # Prompt construction (one of the two must be set):
    prompt_prefix:  str | None = None
    prompts_fn:     Callable[[dict], dict[str, str]] | None = None

    # Filter for headline accuracy; non-matching records still feed by_* rows.
    primary_filter: Callable[[dict], bool] | None = None

    # When set together with ``primary_filter``, sampling is performed on
    # primary conditions only and each sampled primary is expanded to include
    # every condition that shares its ``pair_key`` value (typically the
    # matching control / baseline). Used to enforce "always run the matched
    # pair" semantics: e.g. a2's anchored stim and its no-reference baseline
    # are always sampled together.
    pair_key:       tuple[str, ...] = ()

    # Keys copied from each condition dict into the per-stimulus record.
    record_extras:  tuple[str, ...] = ()

    # Short progress-display string for dispatch().
    label_fn:       Callable[[dict], str] | None = None

    # Optional extra metadata fields for the JSON file.
    metadata_fn:    Callable[[], dict] | None = None

    def __post_init__(self) -> None:
        if (self.prompt_prefix is None) == (self.prompts_fn is None):
            raise ValueError(
                f"{self.exp_name}: exactly one of prompt_prefix or prompts_fn must be set"
            )


# ── Prompt assembly ──────────────────────────────────────────────────────────

def _build_prompts(spec: CatASpec, cond: dict) -> dict[str, str]:
    """Return ``{format: prompt}`` for a single condition."""
    if spec.prompts_fn is not None:
        out = spec.prompts_fn(cond)
        # Ensure every format has a key (default to empty string for missing).
        return {fmt: out.get(fmt, "") for fmt in FORMATS}
    pref = spec.prompt_prefix or ""
    return {
        "midi":   pref + PROMPT_MIDI,
        "spn":    pref + PROMPT_SPN,
        "doremi": pref + PROMPT_DOREMI,
        "hz":     pref + PROMPT_HZ,
    }


# ── CSV record filtering ─────────────────────────────────────────────────────

# Continuous-valued columns and legacy ABC aliases are stripped before the
# per-stimulus CSV is written. The unstripped records remain in memory for
# the plot helper.
_CSV_DROP_KEYS: frozenset[str] = frozenset({
    "hz_gt", "hz_pred", "hz_abs_error", "hz_ratio",
    "abc_gt", "abc_pred", "abc_correct",
})


def _strip_for_csv(records: list[dict]) -> list[dict]:
    """Drop continuous Hz fields and ABC aliases from each record."""
    return [
        {k: v for k, v in r.items() if k not in _CSV_DROP_KEYS}
        for r in records
    ]


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
    primary_filter: Callable[[dict], bool] | None,
) -> dict[str, Any]:
    """Build the cat-A summary dict.

    ``accuracy`` is the headline number: averaged over the **primary** subset
    (records matching ``primary_filter``). When ``primary_filter`` is ``None``,
    primary == all records and there is no secondary subset.

    When ``primary_filter`` is set, the dict additionally carries
    ``secondary_n`` and ``accuracy_secondary`` so the control / no-reference
    accuracies are visible alongside the headline numbers — they're reported
    in the .txt summary and flow through the auto-marginals into the
    accuracies CSV (`by_<discriminator>.<value>.<format>` rows).

    Returns::

        {
            "total":        <total record count>,
            "condition_n":  <count used for accuracy>,
            "accuracy":     {"n": int, "midi": r, "spn": r, "doremi": r, "hz": r},
            # only when primary_filter is not None:
            "baseline_n":        <other count>,
            "accuracy_baseline": {"n": int, "midi": r, "spn": r, "doremi": r, "hz": r},
        }
    """
    total = len(records)
    if primary_filter is None:
        return {
            "total":        total,
            "condition_n":  total,
            "accuracy":  {
                "n": total,
                **{fmt: _accuracy(records, f"{fmt}_correct") for fmt in FORMATS},
            },
        }

    primary:   list[dict] = []
    secondary: list[dict] = []
    for r in records:
        (primary if primary_filter(r) else secondary).append(r)
    return {
        "total":             total,
        "condition_n":       len(primary),
        "baseline_n":        len(secondary),
        "accuracy":          {
            "n": len(primary),
            **{fmt: _accuracy(primary,   f"{fmt}_correct") for fmt in FORMATS},
        },
        "accuracy_baseline": {
            "n": len(secondary),
            **{fmt: _accuracy(secondary, f"{fmt}_correct") for fmt in FORMATS},
        },
    }


def format_summary_lines(
    spec:        CatASpec,
    summary:     dict[str, Any],
    sample_info: dict | None,
) -> list[str]:
    """Uniform text-summary block for the .txt output.

    When the spec has a ``primary_filter`` the table shows two columns:
    Primary (the headline accuracy) and Secondary (control / baseline) so
    that ablation numbers are visible without opening the CSV.
    """
    n_total = summary["total"]
    acc_p   = summary["accuracy"]
    has_baseline = "accuracy_baseline" in summary

    lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli (total)     : {n_total}",
    ]
    if has_baseline:
        lines += [
            f"  Stimuli (condition) : {summary['condition_n']}",
            f"  Stimuli (baseline)  : {summary['baseline_n']}",
            "",
            f"  {'Format':>6}  {'Condition':>9}  {'Baseline':>9}",
            f"  {'─' * 30}",
        ]
        acc_b = summary["accuracy_baseline"]
        for fmt in FORMATS:
            lines.append(
                f"  {fmt.upper():>6}  {acc_p[fmt]:>9.1%}  {acc_b[fmt]:>9.1%}"
            )
        lines.append("")
        lines.append(
            "  Headline 'accuracy' is computed on the condition subset only "
            "(see CatASpec.primary_filter)."
        )
    else:
        lines += [
            f"  Stimuli (condition) : {summary['condition_n']}  (all records)",
            "",
            f"  {'Format':>6}  {'Accuracy':>9}",
            f"  {'─' * 18}",
        ]
        for fmt in FORMATS:
            lines.append(f"  {fmt.upper():>6}  {acc_p[fmt]:>9.1%}")
    return lines


# ── Confidence-interval plots ────────────────────────────────────────────────

def _bootstrap_ci(
    values:   list[float],
    *,
    n_boot:   int = 1000,
    alpha:    float = 0.05,
    seed:     int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_lo, ci_hi)``. Empty input gives ``(nan, nan, nan)``."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    import numpy as np
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _predicted_midi(record: dict, fmt: str) -> float | None:
    """Convert the record's prediction in ``fmt`` to a (float) MIDI value."""
    if fmt == "midi":
        v = record.get("midi_pred")
        return float(v) if isinstance(v, (int, float)) else None
    if fmt == "spn":
        spn = record.get("spn_pred")
        if not isinstance(spn, str) or not spn:
            return None
        m = note_to_midi(spn)
        return float(m) if m is not None else None
    if fmt == "doremi":
        raw = record.get("raw_doremi") or ""
        pc  = extract_solfege(raw)
        if pc is None:
            return None
        gt  = record.get("midi_gt")
        if gt is None:
            return None
        return float(solfege_pc_to_midi(int(gt), int(pc)))
    if fmt == "hz":
        hz = record.get("hz_pred")
        if not isinstance(hz, (int, float)) or hz <= 0:
            return None
        return 12.0 * math.log2(float(hz) / 440.0) + 69.0
    return None


def save_pitch_confidence_plots(
    records:    list[dict],
    plots_dir:  Path,
    model_name: str,
) -> None:
    """Write four PNGs: ``pitch_ci_<format>_<safe_model>.png``.

    Each plot is predicted-MIDI vs ground-truth-MIDI with a 95 % bootstrap
    CI band. x-axis tick labels are ``<midi> (<spn>)``, e.g. ``60 (C4)``.
    """
    if not records:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [warn] matplotlib not available; skipping cat-A CI plots")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_stem(model_name)

    gts_all = sorted({int(r["midi_gt"]) for r in records if isinstance(r.get("midi_gt"), int)})
    if not gts_all:
        return

    for fmt in FORMATS:
        means: list[float] = []
        lo:    list[float] = []
        hi:    list[float] = []
        ns:    list[int]   = []
        n_parsed = 0
        n_total  = 0
        for gt in gts_all:
            preds = [
                _predicted_midi(r, fmt)
                for r in records if int(r.get("midi_gt", -1)) == gt
            ]
            n_total  += len(preds)
            kept = [p for p in preds if p is not None]
            n_parsed += len(kept)
            m, l, h = _bootstrap_ci(kept)
            means.append(m); lo.append(l); hi.append(h); ns.append(len(kept))

        fig, ax = plt.subplots(figsize=(max(8, len(gts_all) * 0.5), 5.5))

        # y = x reference
        ax.plot(gts_all, gts_all, color="lightgray", linewidth=1.2,
                linestyle="--", label="y = x (perfect)")

        # CI band — only spans points where we have at least one parsed pred.
        valid = [i for i, m in enumerate(means) if not math.isnan(m)]
        if valid:
            xs = [gts_all[i] for i in valid]
            ax.fill_between(
                xs, [lo[i] for i in valid], [hi[i] for i in valid],
                color="steelblue", alpha=0.20, label="95% bootstrap CI",
            )
            ax.plot(xs, [means[i] for i in valid],
                    marker="o", markersize=4, linewidth=1.4, color="steelblue",
                    label="mean predicted MIDI")

        ax.set_xticks(gts_all)
        ax.set_xticklabels(
            [f"{m}\n({midi_to_note(m)})" for m in gts_all],
            fontsize=8, rotation=0,
        )
        ax.set_xlabel("Ground-truth MIDI (note)")
        ax.set_ylabel("Predicted MIDI")
        ax.set_title(
            f"{fmt.upper()} prediction vs. truth — {model_name}\n"
            f"(parsed {n_parsed}/{n_total})"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        out = plots_dir / f"pitch_ci_{fmt}_{safe}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)


# ── Sampling with optional pair-expansion ────────────────────────────────────

def _sample_conditions(
    spec:        CatASpec,
    all_conds:   list[dict],
    sample_n:    int | None,
    sample_seed: int,
) -> tuple[list[dict], dict]:
    """Stratified sampling with optional ``pair_key`` expansion.

    When ``spec.pair_key`` is set together with ``spec.primary_filter``,
    sampling runs on the primary subset only and each sampled primary stim
    is expanded to include every condition that shares its ``pair_key``
    value (i.e. its matched control / baseline). Otherwise this is just a
    pass-through to :func:`apply_default_sampling`.
    """
    if not (spec.pair_key and spec.primary_filter is not None):
        return apply_default_sampling(
            spec.exp_name, all_conds, sample_n, sample_seed,
        )

    # Sample on primary records only, then expand to matched pairs.
    primaries = [c for c in all_conds if spec.primary_filter(c)]
    sampled_primaries, s_meta = apply_default_sampling(
        spec.exp_name, primaries, sample_n, sample_seed,
    )
    sampled_keys = {tuple(c[k] for k in spec.pair_key) for c in sampled_primaries}
    paired = [c for c in all_conds if tuple(c[k] for k in spec.pair_key) in sampled_keys]

    # Reflect pair-expansion in the sampling metadata so the .txt summary
    # shows the actual stimulus count after expansion.
    s_meta = dict(s_meta)
    s_meta["pair_key"]        = list(spec.pair_key)
    s_meta["sampled_primary"] = len(sampled_primaries)
    s_meta["sampled_total"]   = len(paired)
    s_meta["total_available"] = len(all_conds)
    return paired, s_meta


# ── Argparse + entry points ──────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview",     action="store_true")
    parser.add_argument("--models",      nargs="+", metavar="MODEL")
    parser.add_argument("--sample-n",    type=int, default=None, metavar="N")
    parser.add_argument("--sample-seed", type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def run_one_model(
    spec:        CatASpec,
    model_name:  str,
    conds:       list[dict],
    run_dir:     Path,
    sample_info: dict | None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1 — generate audio sequentially (engine cache makes this fast).
    jobs: list[dict] = []
    for c in conds:
        try:
            wav = str(spec.wav_fn(c))
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        prompts = _build_prompts(spec, c)
        jobs.append({"wav": wav, "cond": c, "prompts": prompts})

    # Phase 2 — dispatch model queries.
    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        prompts = job["prompts"]
        r_m, r_s, r_d, r_h = query_four_formats(
            model_name, job["wav"],
            prompts["midi"], prompts["spn"], prompts["doremi"], prompts["hz"],
            verbose=False,
        )
        extras = {k: c[k] for k in spec.record_extras if k in c}
        return standard_pitch_record(
            wav=job["wav"],
            source=c["source"],
            source_type=c.get(
                "source_type",
                "waveform" if c["source"] in config.WAVEFORMS else "instrument",
            ),
            midi_gt=c["midi"],
            raw_midi=r_m["result"], raw_spn=r_s["result"],
            raw_doremi=r_d["result"], raw_hz=r_h["result"],
            prompt_midi=prompts["midi"], prompt_spn=prompts["spn"],
            prompt_doremi=prompts["doremi"], prompt_hz=prompts["hz"],
            **extras,
        )

    label = spec.label_fn or (lambda j: f"{midi_to_note(j['cond']['midi']):>4} {j['cond']['source']}")
    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: label(j),
        result_label_fn=lambda j, r: pitch_record_audit_str(r, label=label(j)),
    )
    full_records: list[dict] = [r for r in raw if r is not None]

    # Phase 3 — summary + summary lines.
    summary       = compute_summary(full_records, spec.primary_filter)
    summary_lines = format_summary_lines(spec, summary, sample_info)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    # Phase 4 — plots from FULL records (Hz columns intact).
    save_pitch_confidence_plots(full_records, run_dir / "plots", model_name)

    # Phase 5 — strip continuous columns and write the JSON / TXT / CSV trio.
    csv_records = _strip_for_csv(full_records)

    extra_meta = spec.metadata_fn() if spec.metadata_fn else {}
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        primary_filter_set=spec.primary_filter is not None,
        record_extras=list(spec.record_extras),
        **extra_meta,
        **(sample_info or {}),
    )
    save_results(
        spec.exp_name, model_name, csv_records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=FORMATS,
    )
    return summary


def run_cat_a_experiment(spec: CatASpec, *, mode: str = "run") -> dict | None:
    """Top-level cat-A entry point. ``mode`` ∈ {"preview", "run"}."""
    engine.set_exp(spec.exp_name)
    args = _parse_args()
    if args.preview:
        mode = "preview"

    all_conds = spec.build_conditions_fn()
    conds, s_meta = _sample_conditions(spec, all_conds, args.sample_n, args.sample_seed)

    # Generate audio in both modes (preview = "stimuli only, no queries").
    n_skipped = 0
    for c in conds:
        try:
            spec.wav_fn(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            n_skipped += 1

    if mode == "preview":
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
