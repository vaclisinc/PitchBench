"""
Experiment 02 — Hidden pitch in silence
A single musical note is embedded at a specific position inside a long silent
audio clip (default: 60 seconds).  The prompt tells the model exactly when the
note occurs; the task is to identify its pitch.

This tests whether models can extract pitch from a brief event in an otherwise
quiet recording — analogous to identifying a note in a live recording with
long rests, or finding a signal buried in silence.

Parameters swept:
  • position  — where the tone starts: 5 s, 15 s, 30 s, 45 s, 55 s
  • pitch     — 7 representative MIDI notes spanning C3–C5
  • source    — all waveforms (instruments if FluidSynth available)
  • variant   — MIDI, ABC, Doremi

Baseline: same tone played in isolation (no surrounding silence) — same prompt
variants, compare accuracy with vs without context silence.

Usage:
    python experiments/run.py exp_2_pitch_in_silence
    python experiments/run.py exp_2_pitch_in_silence --preview
    python experiments/run.py exp_2_pitch_in_silence --models audio_flamingo_next_instruct
"""

import argparse
from pathlib import Path
from typing import Any

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
from pitchbench.experiments.helpers.audit import pitch_record_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    PROMPT_ABC, PROMPT_MIDI, PROMPT_DOREMI,
    midi_to_note, midi_to_solfege,
    standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.plots import (
    save_accuracy_plots, save_combined_iv_plot, save_per_format_iv_plots,
)
from pitchbench.experiments.helpers.results import get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies
from pitchbench.experiments.helpers.sampling import apply_default_sampling, sampling_summary_lines

EXP_NAME = Path(__file__).stem

# Representative pitches — wide range
PITCHES: list[int] = config.DEFAULT_PITCHES

# Positions where the tone is placed inside the silent clip (ms)
TONE_POSITIONS_MS: list[int] = config.DEFAULT_TONE_POSITIONS_MS

TONE_DURATION_MS = config.DEFAULT_DURATION_MS    # note duration
TOTAL_SILENCE_MS  = config.DEFAULT_TOTAL_DUR_MS  # total clip length

CONDITIONS: list[str] = ["hidden", "baseline"]   # with/without surrounding silence

SOURCES: list[str] = config.ALL_SOURCES    # instruments added when FluidSynth available


def _make_prompt(variant: str, pos_ms: int, dur_ms: int, total_ms: int, condition: str) -> str:
    pos_s  = pos_ms  / 1000
    dur_s  = dur_ms  / 1000
    end_s  = (pos_ms + dur_ms) / 1000
    total_s = total_ms / 1000

    if condition == "baseline":
        if variant == "midi":
            return ("This audio contains a single musical note. "
                    "What is the MIDI note number (integer 0–127)? "
                    "Reply with ONLY the integer.")
        elif variant == "abc":
            return ("This audio contains a single musical note. "
                    "What is the note name and octave, e.g. C4, F#3? "
                    "Reply with ONLY the note name.")
        elif variant == "doremi":
            return ("This audio contains a single musical note. "
                    "What is the solfège syllable and accidental (if needed) (fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B)? "
                    "Reply with the syllable and accidental (if necessary).")
        else:  # hz
            return ("This audio contains a single musical note. "
                    "What is the pitch frequency in Hertz? "
                    "Reply with ONLY a number (the frequency in Hz). Nothing else.")
    else:  # hidden
        context = (
            f"You will hear a {total_s:.0f}-second audio clip. "
            f"A single musical note sounds in the clip; the rest is silence. "
        )
        if variant == "midi":
            return context + ("What is the MIDI note number (integer 0–127) of that note? "
                              "Reply with ONLY the integer.")
        elif variant == "abc":
            return context + ("What is the note name and octave of that note, e.g. C4, F#3? "
                              "Reply with ONLY the note name.")
        elif variant == "doremi":
            return context + ("What is the solfège syllable and accidental (if needed) (fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B) "
                              "of that note? Reply with the syllable and accidental (if necessary).")
        else:  # hz
            return context + ("What is the pitch frequency of that note in Hertz? "
                              "Reply with ONLY a number (the frequency in Hz). Nothing else.")


def build_conditions(sources: list[str]) -> list[dict]:
    rows: list[dict] = []
    for src in sources:
        for midi in PITCHES:
            for pos_ms in TONE_POSITIONS_MS:
                for cond in CONDITIONS:
                    rows.append({
                        "source":    src,
                        "midi":      midi,
                        "note":      midi_to_note(midi),
                        "pos_ms":    pos_ms,
                        "condition": cond,
                    })
    return rows


def _get_wav(c: dict) -> Path:
    if c["condition"] == "hidden":
        return engine.tone_in_silence(
            c["midi"], c["source"],
            tone_start_ms=c["pos_ms"],
            tone_dur_ms=TONE_DURATION_MS,
            total_dur_ms=TOTAL_SILENCE_MS,
        )
    else:
        return engine.tone(c["midi"], c["source"], TONE_DURATION_MS)


def run_one_model(
    model_name: str,
    conds: list[dict],
    sources: list[str],
    run_dir: Path,
    sample_info: dict | None = None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1: generate audio sequentially.
    jobs: list[dict[str, Any]] = []
    for c in conds:
        try:
            wav = _get_wav(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        prompt_midi   = _make_prompt("midi",   c["pos_ms"], TONE_DURATION_MS, TOTAL_SILENCE_MS, c["condition"])
        prompt_abc    = _make_prompt("abc",    c["pos_ms"], TONE_DURATION_MS, TOTAL_SILENCE_MS, c["condition"])
        prompt_doremi = _make_prompt("doremi", c["pos_ms"], TONE_DURATION_MS, TOTAL_SILENCE_MS, c["condition"])
        prompt_hz     = _make_prompt("hz",     c["pos_ms"], TONE_DURATION_MS, TOTAL_SILENCE_MS, c["condition"])
        jobs.append({
            "wav": wav, "cond": c,
            "prompt_midi": prompt_midi, "prompt_abc": prompt_abc,
            "prompt_doremi": prompt_doremi, "prompt_hz": prompt_hz,
        })

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict[str, Any]) -> dict[str, Any]:
        c = job["cond"]
        r_m, r_a, r_d, r_h = query_four_formats(
            model_name, str(job["wav"]),
            job["prompt_midi"], job["prompt_abc"], job["prompt_doremi"], job["prompt_hz"],
            verbose=False,
        )
        return standard_pitch_record(
            wav=job["wav"], source=c["source"],
            source_type="waveform" if c["source"] in config.WAVEFORMS else "instrument",
            midi_gt=c["midi"],
            raw_midi=r_m["result"], raw_abc=r_a["result"],
            raw_doremi=r_d["result"], raw_hz=r_h["result"],
            prompt_midi=job["prompt_midi"], prompt_abc=job["prompt_abc"],
            prompt_doremi=job["prompt_doremi"], prompt_hz=job["prompt_hz"],
            condition=c["condition"], pos_ms=c["pos_ms"],
        )

    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"{j['cond']['condition']:8s} {midi_to_note(j['cond']['midi']):4s} @{j['cond']['pos_ms']/1000:.0f}s {j['cond']['source']}",
        result_label_fn=lambda j, r: pitch_record_audit_str(r, label=f"{j['cond']['condition']:8s} {midi_to_note(j['cond']['midi']):4s} @{j['cond']['pos_ms']/1000:.0f}s {j['cond']['source']}"),
    )
    records: list[dict[str, Any]] = [r for r in raw if r is not None]

    # ── Summary ───────────────────────────────────────────────────────────────
    per_cond: dict[str, dict] = {}
    for cond in CONDITIONS:
        sub_all = [r for r in records if r["condition"] == cond]
        per_cond[cond] = {
            "n":      len(sub_all),
            "midi":   round(sum(r["midi_correct"]   for r in sub_all) / len(sub_all), 4) if sub_all else 0.0,
            "abc":    round(sum(r["abc_correct"]    for r in sub_all) / len(sub_all), 4) if sub_all else 0.0,
            "doremi": round(sum(r["doremi_correct"] for r in sub_all) / len(sub_all), 4) if sub_all else 0.0,
            "hz":     round(sum(r["hz_correct"]     for r in sub_all) / len(sub_all), 4) if sub_all else 0.0,
        }

    per_position: dict[int, dict] = {}
    for pos in TONE_POSITIONS_MS:
        h = [r for r in records if r["condition"] == "hidden" and r["pos_ms"] == pos]
        b = [r for r in records if r["condition"] == "baseline"]
        if h:
            per_position[pos] = {}
            for v in ("midi", "abc", "doremi", "hz"):
                col = f"{v}_correct"
                per_position[pos][f"hidden_{v}"]   = round(sum(r[col] for r in h) / max(1, len(h)), 4)
                per_position[pos][f"baseline_{v}"] = round(sum(r[col] for r in b) / max(1, len(b)), 4) if b else None

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Sources   : {sources}",
        f"  Pitches   : {PITCHES}",
        f"  Positions : {[p // 1000 for p in TONE_POSITIONS_MS]} s",
        f"  Clip      : {TOTAL_SILENCE_MS // 1000} s total, tone {TONE_DURATION_MS // 1000} s",
        "",
        f"  {'Condition':10s}  {'n':>5}  {'MIDI%':>7}  {'ABC%':>7}  {'Doremi%':>8}  {'Hz%':>6}",
        f"  {'─' * 54}",
    ]
    for cond, vd in per_cond.items():
        summary_lines.append(
            f"  {cond:10s}  {vd.get('n', 0):>5}  {vd.get('midi', 0):>7.1%}  "
            f"{vd.get('abc', 0):>7.1%}  {vd.get('doremi', 0):>8.1%}  "
            f"{vd.get('hz', 0):>6.1%}"
        )
    summary_lines += ["", "  Accuracy by position — hidden | baseline  (MIDI / ABC / Doremi / Hz):"]
    for pos, d in per_position.items():
        parts = []
        for v in ("midi", "abc", "doremi"):
            h_val = d.get(f"hidden_{v}", 0.0)
            b_val = d.get(f"baseline_{v}")
            b_s   = f"{b_val:.0%}" if b_val is not None else "—"
            delta = (h_val - b_val) if b_val is not None else None
            delta_s = f"(Δ{delta:+.0%})" if delta is not None else ""
            parts.append(f"{h_val:.0%}|{b_s}{delta_s}")
        summary_lines.append(f"    @{pos//1000:2d}s:  " + "  /  ".join(parts))

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    summary = {
        "total":        len(records),
        "per_cond":     per_cond,
        "per_position": {str(k): v for k, v in per_position.items()},
    }
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        pitches=PITCHES, positions_ms=TONE_POSITIONS_MS,
        tone_duration_ms=TONE_DURATION_MS, total_silence_ms=TOTAL_SILENCE_MS,
        sources=sources, conditions=CONDITIONS,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    save_accuracy_plots(
        wide_to_long_records(records), run_dir, model_name,
        instrument_key="source", pitch_key="midi_gt",
        prompt_key="prompt_variant", accuracy_key="exact_match",
    )
    save_per_format_iv_plots(records, run_dir, model_name, iv_key="condition", iv_label="Condition")
    save_combined_iv_plot(records, run_dir, model_name, iv_key="condition", iv_label="Condition")
    return {
        f"hidden_{v}":   per_cond.get("hidden",   {}).get(v, 0.0) for v in ("midi", "abc", "doremi", "hz")
    } | {
        f"baseline_{v}": per_cond.get("baseline", {}).get(v, 0.0) for v in ("midi", "abc", "doremi", "hz")
    }


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by source)")
    parser.add_argument("--sample-seed",  type=int, default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    all_conds = build_conditions(SOURCES)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    print(f"Experiment   : {EXP_NAME}")
    print(f"Sources      : {SOURCES}")
    print(f"Pitches      : {[midi_to_note(m) for m in PITCHES]}")
    print(f"Positions    : {[p // 1000 for p in TONE_POSITIONS_MS]} s  (inside {TOTAL_SILENCE_MS // 1000}s clip)")
    print(f"Conditions   : {CONDITIONS}")
    print(f"Audio files  : {len(conds)}  (generating …)")
    for c in conds:
        try:
            _get_wav(c)
        except ValueError as exc:
            print(f"  [SKIP] {exc}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print(f"Queries/model: {len(conds) * 4}")
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args    = _parse_args()
    target_models = args.models or list(config.MODELS)
    sources = list(SOURCES)
    all_conds = build_conditions(sources)
    conds, s_meta = apply_default_sampling(EXP_NAME, all_conds, args.sample_n, args.sample_seed)
    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, conds, sources, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
