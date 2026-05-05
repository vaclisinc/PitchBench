"""
Shared runner for category-F experiments (polyphonic-line transcription).

Both cat-F scripts ask the model to transcribe a single designated voice
from an n-part polyphonic mix:

* **melodic_line** (f1) — synthetic n-part polyphony with controllable
  rhythm/register/timbre. The target voice always has exactly N notes.
* **chorale_voice** (f2) — real Bach chorales sourced via music21, sliced
  to the longest no-cross segment per voice. Variable note counts.

Both produce one record per condition with three multi-format
``<format>_seq_correct`` columns (MIDI / SPN / Doremi). Per-position
correctness is preserved as a list-string diagnostic
``<format>_per_pos`` (no ``_correct`` suffix → invisible to the
auto-marginals).

The lifecycle (argparse → sampling → audio → dispatch → save_results)
mirrors :mod:`cat_d`. ``score_polyphonic_record`` does all the
parsing+scoring+record assembly so each cat-F script reduces to a
``build_conditions`` / ``wav_for`` / ``prompts_for`` triple plus a
``CatFSpec`` instance.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.audit import melody_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    PC_TO_SOLFEGE,
    any_format_correct,
    extract_all_notes,
    extract_all_solfege,
    midi_to_freq,
    midi_to_note,
    semitone_distance,
)
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies, get_run_metadata, make_run_dir,
    save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import (
    apply_default_sampling, sampling_summary_lines,
)


HEADLINE_METRICS: tuple[str, ...] = ("midi_seq", "spn_seq", "doremi_seq", "hz_seq", "any_seq")


# ── Spec ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CatFSpec:
    """Per-experiment configuration for the cat-F unified runner.

    Each condition produces ONE record. ``prompts_fn`` returns a mapping
    ``{format_name: prompt_text}`` whose keys are queried in turn.
    ``record_fn`` receives the cond, the wav path, and the raw responses
    and must return a dict with at least ``midi_seq_correct``,
    ``spn_seq_correct``, and ``doremi_seq_correct``. In practice both
    f1 and f2 just delegate to :func:`score_polyphonic_record`.
    """
    exp_name:            str
    build_conditions_fn: Callable[[], list[dict]]
    wav_fn:              Callable[[dict], Path]

    task_type: str  # "melodic_line" | "chorale_voice"

    prompts_fn: Callable[[dict], dict[str, str]]
    record_fn:  Callable[[dict, str, dict[str, str]], dict]

    record_extras:  tuple[str, ...] = ()
    label_fn:       Callable[[dict], str] | None = None
    metadata_fn:    Callable[[], dict] | None = None

    # Headline metrics are fixed for cat-F: all three formats.
    headline_metrics: tuple[str, ...] = HEADLINE_METRICS

    def __post_init__(self) -> None:
        if self.task_type not in ("melodic_line", "chorale_voice"):
            raise ValueError(
                f"{self.exp_name}: task_type must be 'melodic_line' or 'chorale_voice'"
            )


# ── Source-type fallback ─────────────────────────────────────────────────────

def _src_type(source: str) -> str:
    return "waveform" if source in config.WAVEFORMS else "instrument"


# ── Response parsing (variable n) ────────────────────────────────────────────

def _pad(seq: list, n: int) -> list:
    out = list(seq[:n])
    while len(out) < n:
        out.append(None)
    return out


def parse_midi_seq(text: str, n: int) -> list[int | None]:
    nums = [int(m) for m in re.findall(r"\b(\d{1,3})\b", text or "") if 0 <= int(m) <= 127]
    return _pad(nums, n)


def parse_spn_seq(text: str, n: int) -> list[str | None]:
    return _pad(extract_all_notes(text or ""), n)


def parse_doremi_seq(text: str, n: int) -> list[int | None]:
    return _pad(extract_all_solfege(text or ""), n)


def parse_hz_seq(text: str, n: int) -> list[float | None]:
    nums: list[float] = []
    for m in re.finditer(r"\d+(?:\.\d+)?", text or ""):
        v = float(m.group(0))
        if 16.0 <= v <= 20000.0:
            nums.append(v)
    return _pad(nums, n)


# ── Scoring ──────────────────────────────────────────────────────────────────

def _score_midi(gt: list[int], pred: list[int | None]) -> tuple[list[bool | None], int]:
    per_pos = [
        (gt[i] == pred[i] if pred[i] is not None else None)
        for i in range(len(gt))
    ]
    return per_pos, int(all(v is True for v in per_pos))


def _score_spn(gt_midi: list[int], pred: list[str | None]) -> tuple[list[bool | None], int]:
    gt_spn = [midi_to_note(m) for m in gt_midi]
    per_pos: list[bool | None] = []
    for gt, pr in zip(gt_spn, pred):
        if pr is None:
            per_pos.append(None)
        else:
            dist = semitone_distance(gt, pr)
            per_pos.append(dist == 0 if dist is not None else False)
    return per_pos, int(all(v is True for v in per_pos))


def _score_doremi(gt_midi: list[int], pred_pcs: list[int | None]) -> tuple[list[bool | None], int]:
    gt_pcs = [m % 12 for m in gt_midi]
    per_pos: list[bool | None] = []
    for gt_pc, pred_pc in zip(gt_pcs, pred_pcs):
        if pred_pc is None:
            per_pos.append(None)
        else:
            diff = abs(gt_pc - pred_pc)
            per_pos.append(min(diff, 12 - diff) == 0)
    return per_pos, int(all(v is True for v in per_pos))


def _score_hz(gt_midi: list[int], pred: list[float | None]) -> tuple[list[bool | None], int]:
    per_pos: list[bool | None] = []
    for gt_m, pr in zip(gt_midi, pred):
        if pr is None or pr <= 0:
            per_pos.append(None)
        else:
            ratio = pr / midi_to_freq(gt_m)
            per_pos.append(0.99 <= ratio <= 1.01)
    return per_pos, int(all(v is True for v in per_pos))


# ── Per-stim record builder shared by f1 and f2 ──────────────────────────────

def score_polyphonic_record(
    cond:      dict,
    wav:       str,
    responses: dict[str, str],
) -> dict[str, Any]:
    """Parse the four raw responses against ``cond["target_pitches"]`` and
    return the standard cat-F record. Both f1 and f2 use this directly.

    Required keys on ``cond``: ``target_pitches`` (list[int]),
    ``source`` (str — used for source_type fallback), ``sources`` (list).
    """
    target_pitches: list[int] = list(cond["target_pitches"])
    n = len(target_pitches)
    raw_midi   = responses.get("midi", "") or ""
    raw_spn    = responses.get("spn", "")  or ""
    raw_doremi = responses.get("doremi", "") or ""
    raw_hz     = responses.get("hz", "") or ""

    pred_midi   = parse_midi_seq(raw_midi, n)
    pred_spn    = parse_spn_seq(raw_spn, n)
    pred_doremi = parse_doremi_seq(raw_doremi, n)
    pred_hz     = parse_hz_seq(raw_hz, n)

    midi_pos, midi_sc = _score_midi(target_pitches, pred_midi)
    spn_pos,  spn_sc  = _score_spn(target_pitches, pred_spn)
    dor_pos,  dor_sc  = _score_doremi(target_pitches, pred_doremi)
    hz_pos,   hz_sc   = _score_hz(target_pitches, pred_hz)
    any_sc = any_format_correct((midi_sc, spn_sc, dor_sc, hz_sc))

    target_spn    = [midi_to_note(m) for m in target_pitches]
    target_doremi = [PC_TO_SOLFEGE[m % 12] for m in target_pitches]
    target_hz     = [round(midi_to_freq(m), 4) for m in target_pitches]

    return {
        "n_target":           n,
        "target_midi_gt":     str(target_pitches),
        "target_spn_gt":      str(target_spn),
        "target_doremi_gt":   str(target_doremi),
        "target_hz_gt":       str(target_hz),
        # MIDI
        "midi_pred":          str(pred_midi),
        "midi_per_pos":       str(midi_pos),
        "midi_seq_correct":   midi_sc,
        # SPN
        "spn_pred":           str(pred_spn),
        "spn_per_pos":        str(spn_pos),
        "spn_seq_correct":    spn_sc,
        # Doremi
        "doremi_pred":        str(pred_doremi),
        "doremi_per_pos":     str(dor_pos),
        "doremi_seq_correct": dor_sc,
        # Hz
        "hz_pred":            str(pred_hz),
        "hz_per_pos":         str(hz_pos),
        "hz_seq_correct":     hz_sc,
        # Any
        "any_seq_correct":    any_sc,
        # Raw
        "raw_midi":           raw_midi.strip(),
        "raw_spn":            raw_spn.strip(),
        "raw_doremi":         raw_doremi.strip(),
        "raw_hz":             raw_hz.strip(),
    }


# ── Summary ──────────────────────────────────────────────────────────────────

def _accuracy(records: list[dict], col: str) -> float:
    if not records:
        return 0.0
    vals = [r[col] for r in records if isinstance(r.get(col), (int, float, bool))]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def compute_summary(records: list[dict]) -> dict[str, Any]:
    """Layout mirrors cat-A / cat-B / cat-C / cat-D headline summary."""
    total = len(records)
    return {
        "total":       total,
        "condition_n": total,
        "accuracy": {
            "n":      total,
            **{m: _accuracy(records, f"{m}_correct") for m in HEADLINE_METRICS},
        },
    }


def _format_summary_lines(
    summary:     dict[str, Any],
    sample_info: dict | None,
) -> list[str]:
    acc = summary["accuracy"]
    lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli (total)     : {summary['total']}",
        f"  Stimuli (condition) : {summary['condition_n']}  (all records)",
        "",
        f"  {'Metric':>14}  {'Accuracy':>9}",
        f"  {'─' * 27}",
    ]
    for m in HEADLINE_METRICS:
        lines.append(f"  {m.upper():>14}  {acc[m]:>9.1%}")
    lines.append("")
    lines.append(
        "  (Headline = full-sequence exact match for each format. Per-position "
        "diagnostics live in <format>_per_pos in the records CSV.)"
    )
    return lines


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
    spec:        CatFSpec,
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

    # Phase 2 — dispatch (3 queries per cond → one record per cond).
    label = spec.label_fn or (lambda j: f"{j['cond'].get('source_label', '?')}  task={spec.task_type}")

    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        prompts = job["prompts"]
        responses: dict[str, str] = {}
        for name, prompt in prompts.items():
            out = query_alm(model_name, job["wav"], prompt)
            responses[name] = (out["result"] or "").strip()
        record = spec.record_fn(c, job["wav"], responses)
        # Inject prompt_<name> + record_extras + a few standard columns.
        for name, prompt in prompts.items():
            record.setdefault(f"prompt_{name}", prompt)
        for k in spec.record_extras:
            record.setdefault(k, c.get(k))
        # source / source_type for the auto-marginals to pick up.
        src = c.get("source") or c.get("source_label") or ""
        record.setdefault("source", src)
        record.setdefault("source_type", _src_type(str(src)))
        record.setdefault("wav", job["wav"])
        return record

    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: label(j),
        result_label_fn=lambda j, r: melody_audit_str(r, label=label(j)),
    )
    full_records: list[dict] = [r for r in raw if r is not None]

    # Phase 3 — summary + lines.
    summary       = compute_summary(full_records)
    summary_lines = _format_summary_lines(summary, sample_info)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    # Phase 4 — save (no bespoke plots; uniform plots run inside save_results).
    extra_meta = spec.metadata_fn() if spec.metadata_fn else {}
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        task_type=spec.task_type,
        headline_metrics=list(HEADLINE_METRICS),
        record_extras=list(spec.record_extras),
        **extra_meta,
        **(sample_info or {}),
    )
    save_results(
        spec.exp_name, model_name, full_records, summary, metadata, summary_lines,
        run_dir=run_dir,
        formats=(),
        extra_metrics=tuple(f"{m}_correct" for m in HEADLINE_METRICS),
    )
    return summary


def run_cat_f_experiment(spec: CatFSpec, *, mode: str = "run") -> dict | None:
    """Top-level cat-F entry point. ``mode`` ∈ {"preview", "run"}."""
    engine.set_exp(spec.exp_name)
    args = _parse_args()
    if args.preview:
        mode = "preview"

    all_conds = spec.build_conditions_fn()
    conds, s_meta = apply_default_sampling(
        spec.exp_name, all_conds, args.sample_n, args.sample_seed,
    )

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
    print(f"Stimuli    : {len(conds)}  × 4 formats = {len(conds) * 4} queries/model")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(spec.exp_name)
    all_summaries: dict[str, dict] = {}
    for m in target_models:
        all_summaries[m] = run_one_model(spec, m, conds, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, spec.exp_name)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))
