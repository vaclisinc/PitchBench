"""
Experiment 12 — Anchored pitch identification
A reference tone is played first, then the target tone. The prompt tells the
model the identity of the reference. Tests whether the model can use that
anchor to identify the target more accurately — i.e. whether it can combine
relative pitch perception with an absolute reference.

Baseline condition: target tone played alone (no reference), same prompts.

Intervals tested: −12, −7, −5, −4, −3, −2, −1, 0, +1, +2, +3, +4, +5, +7, +12 semitones.

Three prompt variants: MIDI, ABC, Doremi.

Usage:
    python experiments/run.py exp_12_anchored_pitch
    python experiments/run.py exp_12_anchored_pitch --preview
    python experiments/run.py exp_12_anchored_pitch --models audio_flamingo_next_instruct
"""

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_three_formats
from pitchbench.experiments.helpers.music import (
    midi_to_note, midi_to_solfege,
    standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.plots import save_accuracy_plots
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results
from pitchbench.experiments.helpers.sampling import sampling_meta, sampling_summary_lines, stratified_sample

EXP_NAME = Path(__file__).stem

# Reference pitches spanning the middle range
REFERENCE_PITCHES: list[int] = [52, 55, 60, 64, 67, 69, 72]
#                               E3  G3  C4  E4  G4  A4  C5

INTERVALS: list[int] = [-12, -7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 12]

SOURCES: list[str] = list(config.WAVEFORMS) + list(config.GM_PROGRAMS_V1.keys())

# TONE_DURATION_MS = 1_500
TONE_DURATION_MS = 4_000
GAP_MS           = 500    # silence between reference and target

CONDITIONS: list[str] = ["anchored", "baseline"]   # with or without reference tone


def _make_prompt(variant: str, ref_midi: int, condition: str) -> str:
    ref_note    = midi_to_note(ref_midi)
    ref_solfege = midi_to_solfege(ref_midi)
    ref_midi_s  = str(ref_midi)

    if condition == "baseline":
        if variant == "midi":
            return ("This audio contains a single musical note. "
                    "What is the MIDI note number (integer 0–127)? "
                    "Reply with ONLY the integer.")
        elif variant == "abc":
            return ("This audio contains a single musical note. "
                    "What is the note name and octave, e.g. C4, F#3? "
                    "Reply with ONLY the note name.")
        else:  # doremi
            return ("This audio contains a single musical note. "
                    "What is the solfège syllable (fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B)? "
                    "Reply with the syllable and accidental (if necessary).")
    else:  # anchored
        if variant == "midi":
            return (f"You will hear two tones separated by a silence. "
                    f"The FIRST tone is MIDI note {ref_midi_s}. "
                    f"What is the MIDI note number of the SECOND tone? "
                    f"Reply with ONLY the integer.")
        elif variant == "abc":
            return (f"You will hear two tones separated by a silence. "
                    f"The FIRST tone is {ref_note}. "
                    f"What is the note name of the SECOND tone, e.g. C4, F#3? "
                    f"Reply with ONLY the note name.")
        else:  # doremi
            return (f"You will hear two tones separated by a silence. "
                    f"The FIRST tone is '{ref_solfege}' (fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B). "
                    f"What is the solfège syllable of the SECOND tone? "
                    f"Reply with the syllable and accidental (if necessary).")


# ── Conditions / stimulus generation ─────────────────────────────────────────

def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for ref_midi in REFERENCE_PITCHES:
        for interval in INTERVALS:
            tgt_midi = ref_midi + interval
            if tgt_midi < 0 or tgt_midi > 127:
                continue
            tgt_note    = midi_to_note(tgt_midi)
            tgt_solfege = midi_to_solfege(tgt_midi)
            for src in SOURCES:
                for cond in CONDITIONS:
                    rows.append({
                        "condition":   cond,
                        "ref_midi":    ref_midi,
                        "ref_note":    midi_to_note(ref_midi),
                        "interval":    interval,
                        "tgt_midi":    tgt_midi,
                        "tgt_note":    tgt_note,
                        "tgt_solfege": tgt_solfege,
                        "source":      src,
                    })
    return rows


def _get_wav(c: dict) -> Path:
    if c["condition"] == "anchored":
        return engine.sequence([c["ref_midi"], c["tgt_midi"]], c["source"], TONE_DURATION_MS, GAP_MS)
    else:
        return engine.tone(c["tgt_midi"], c["source"], TONE_DURATION_MS)


def generate_stimuli(conds: list[dict]) -> None:
    for c in conds:
        _get_wav(c)


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav = str(_get_wav(c))
        print(f"    {c['condition']:8s} r={c['ref_note']:3s} i={c['interval']:+3d} {c['source']}")
        prompt_midi   = _make_prompt("midi",   c["ref_midi"], c["condition"])
        prompt_abc    = _make_prompt("abc",    c["ref_midi"], c["condition"])
        prompt_doremi = _make_prompt("doremi", c["ref_midi"], c["condition"])
        raw_midi, raw_abc, raw_doremi = query_three_formats(
            model_name, wav, prompt_midi, prompt_abc, prompt_doremi
        )
        rec = standard_pitch_record(
            wav=wav, source=c["source"],
            source_type="waveform" if c["source"] in config.WAVEFORMS else "instrument",
            midi_gt=c["tgt_midi"],
            raw_midi=raw_midi, raw_abc=raw_abc, raw_doremi=raw_doremi,
            prompt_midi=prompt_midi, prompt_abc=prompt_abc, prompt_doremi=prompt_doremi,
            condition=c["condition"],
            ref_midi=c["ref_midi"], ref_note=c["ref_note"],
            interval=c["interval"],
        )
        records.append(rec)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    per_cond_var: dict[str, dict] = {}
    for cond in CONDITIONS:
        sub_all = [r for r in records if r["condition"] == cond]
        per_cond_var[cond] = {
            "midi":   round(sum(r["midi_correct"]   for r in sub_all) / len(sub_all), 4) if sub_all else 0.0,
            "abc":    round(sum(r["abc_correct"]    for r in sub_all) / len(sub_all), 4) if sub_all else 0.0,
            "doremi": round(sum(r["doremi_correct"] for r in sub_all) / len(sub_all), 4) if sub_all else 0.0,
        }

    per_interval: dict[int, dict] = {}
    for iv in INTERVALS:
        sub_a = [r for r in records if r["interval"] == iv and r["condition"] == "anchored"]
        sub_b = [r for r in records if r["interval"] == iv and r["condition"] == "baseline"]
        if sub_a:
            per_interval[iv] = {
                "anchored_abc":  round(sum(r["abc_correct"] for r in sub_a) / len(sub_a), 4),
                "baseline_abc":  round(sum(r["abc_correct"] for r in sub_b) / len(sub_b), 4) if sub_b else 0.0,
            }

    summary = {
        "total":        n,
        "per_cond_var": per_cond_var,
        "per_interval": {str(k): v for k, v in per_interval.items()},
    }

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources : {SOURCES}",
        f"  Stimuli : {n}  ({len(REFERENCE_PITCHES)} refs × {len(INTERVALS)} intervals × "
        f"{len(SOURCES)} sources × {len(CONDITIONS)} conditions)",
        "",
        f"  {'Condition':10s}  {'MIDI%':>7}  {'ABC%':>7}  {'Doremi%':>8}",
        f"  {'─' * 38}",
    ]
    for cond, vd in per_cond_var.items():
        summary_lines.append(
            f"  {cond:10s}  {vd.get('midi', 0):>7.1%}  "
            f"{vd.get('abc', 0):>7.1%}  {vd.get('doremi', 0):>8.1%}"
        )
    summary_lines += ["", "  ABC accuracy by interval (anchored | baseline):"]
    for iv, d in per_interval.items():
        delta = d["anchored_abc"] - d["baseline_abc"]
        summary_lines.append(
            f"    i={iv:+3d}: {d['anchored_abc']:.0%} | {d['baseline_abc']:.0%}"
            f"  (Δ={delta:+.0%})"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        reference_pitches=REFERENCE_PITCHES, intervals=INTERVALS,
        sources=SOURCES, tone_duration_ms=TONE_DURATION_MS, gap_ms=GAP_MS,
        conditions=CONDITIONS,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    save_accuracy_plots(
        wide_to_long_records(records), run_dir, model_name,
        instrument_key="source",
        pitch_key="midi_gt",
        prompt_key="prompt_variant",
        accuracy_key="exact_match",
    )
    return summary


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=42,   metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    all_conds = build_conditions()
    conds = all_conds  # rename: save the full list
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    generate_stimuli(conds)
    n_audio = len(conds)
    print(f"Experiment   : {EXP_NAME}")
    print(f"Sources      : {SOURCES}")
    print(f"Ref pitches  : {[midi_to_note(m) for m in REFERENCE_PITCHES]}")
    print(f"Intervals    : {INTERVALS}")
    print(f"Conditions   : {CONDITIONS}")
    print(f"Audio files  : {n_audio}")
    print(f"Queries/model: {n_audio * 3}  (3 prompt variants)")
    print(f"Audio dir    : {config.AUDIO_DIR}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    all_conds = build_conditions()
    conds = all_conds  # rename: save the full list
    if args.sample_n is not None:
        conds = stratified_sample(all_conds, args.sample_n, lambda c: c["source"], seed=args.sample_seed)
    s_meta = sampling_meta(len(all_conds), "source", args.sample_n, args.sample_seed)
    generate_stimuli(conds)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Audio files: {len(conds)}  × 3 variants = {len(conds)*3} queries/model")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, conds, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, EXP_NAME)


if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
