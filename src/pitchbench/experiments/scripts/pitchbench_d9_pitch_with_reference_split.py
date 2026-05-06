"""
d9 — Pitch with reference (split-audio variant of d7).

Same hypothesis as d7 — does an explicit reference tone help the model nail
the target pitch — but the reference and target are delivered as TWO
SEPARATE audio inputs (Audio 1 = reference, Audio 2 = target) instead of
being concatenated into a single ``sequence`` WAV.

This isolates "the model can't temporally segment the two tones" from
"the model can't use a reference at all": d7 packages both tones in one
stream + tells the model which one is first via text, while d9 lets the
chat template attach two distinct audio blocks. Comparing d7-anchored vs.
d9 should show how much of d7's accuracy comes from the linguistic anchor
versus the structured input format.

Anchored only — d7 already provides the no-reference baseline; running it
here would just be duplication.

Universal IVs: source, source_type, midi (target).
Experiment-specific IVs: ref_midi, interval.

Reuses ``config.pitchbench_d7_*`` for stimulus parameters so the two
experiments are directly comparable.

Usage::
    pitchbench --id d9 --preview
    pitchbench --id d9 --models audio_flamingo_next_instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import (
    get_model_info, query_four_formats_multi,
)
from pitchbench.experiments.helpers.audit import pitch_record_audit_str
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    midi_to_freq, midi_to_note, midi_to_solfege, standard_pitch_record,
)
from pitchbench.experiments.helpers.results import (
    extract_format_accuracies, get_run_metadata, make_run_dir,
    save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import (
    apply_default_sampling, sampling_summary_lines,
)


EXP_NAME          = Path(__file__).stem
SOURCES           = config.pitchbench_d7_SOURCES
REFERENCE_PITCHES = config.pitchbench_d7_REFERENCE_PITCHES
INTERVALS         = config.pitchbench_d7_INTERVALS
TONE_DURATION_MS  = config.pitchbench_d7_TONE_DURATION_MS

FORMATS = ("midi", "spn", "doremi", "hz")


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for ref_midi in REFERENCE_PITCHES:
        for interval in INTERVALS:
            tgt_midi = ref_midi + interval
            if tgt_midi < 0 or tgt_midi > 127:
                continue
            for src in SOURCES:
                rows.append({
                    "source":      src,
                    "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":        tgt_midi,
                    "ref_midi":    ref_midi,
                    "interval":    interval,
                })
    return rows


def wavs_for(c: dict) -> tuple[Path, Path]:
    """Return (reference_wav, target_wav) — two separate single-tone files."""
    ref = engine.tone(c["ref_midi"], c["source"], TONE_DURATION_MS)
    tgt = engine.tone(c["midi"],     c["source"], TONE_DURATION_MS)
    return ref, tgt


def prompts_for(c: dict) -> dict[str, str]:
    ref_midi = c["ref_midi"]
    ref_note = midi_to_note(ref_midi)
    ref_solf = midi_to_solfege(ref_midi)
    ref_hz   = f"{midi_to_freq(ref_midi):.2f}"
    return {
        "midi": (
            f"You will be given two audio inputs. The FIRST audio is a reference "
            f"tone whose MIDI note number is {ref_midi}: <AUDIO1>. The SECOND "
            f"audio is the target tone: <AUDIO2>. What is the MIDI note number "
            f"of the target tone (the second audio)? "
            f"Reply with ONLY the integer."
        ),
        "spn": (
            f"You will be given two audio inputs. The FIRST audio is a reference "
            f"tone — note name {ref_note}: <AUDIO1>. The SECOND audio is the "
            f"target tone: <AUDIO2>. What is the note name and octave of the "
            f"target tone (the second audio), e.g. C4, F#3? "
            f"Reply with ONLY the note name in Scientific Pitch Notation."
        ),
        "doremi": (
            f"You will be given two audio inputs. The FIRST audio is a reference "
            f"tone — solfège syllable '{ref_solf}' "
            f"(fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B): <AUDIO1>. The "
            f"SECOND audio is the target tone: <AUDIO2>. What is the solfège "
            f"syllable and accidental (if needed) of the target tone "
            f"(the second audio)? Reply with ONLY the syllable and accidental."
        ),
        "hz": (
            f"You will be given two audio inputs. The FIRST audio is a reference "
            f"tone at {ref_hz} Hz: <AUDIO1>. The SECOND audio is the target "
            f"tone: <AUDIO2>. What is the pitch frequency of the target tone "
            f"(the second audio) in Hertz? "
            f"Reply with ONLY a number (the frequency in Hz)."
        ),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preview",     action="store_true")
    p.add_argument("--models",      nargs="+", metavar="MODEL")
    p.add_argument("--sample-n",    type=int, default=None, metavar="N")
    p.add_argument("--sample-seed", type=int,
                   default=config.DEFAULT_SAMPLE_SEED, metavar="SEED")
    args, _ = p.parse_known_args()
    return args


def _label(j: dict) -> str:
    c = j["cond"]
    return (
        f"r={midi_to_note(c['ref_midi']):>3} "
        f"i={c['interval']:+3d} {c['source']}"
    )


def _run_one_model(
    model_name:  str,
    conds:       list[dict],
    run_dir:     Path,
    sample_info: dict | None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    # Phase 1 — generate audio (engine cache makes this fast).
    jobs: list[dict] = []
    for c in conds:
        ref_wav, tgt_wav = wavs_for(c)
        jobs.append({
            "audios":  [str(ref_wav), str(tgt_wav)],
            "cond":    c,
            "prompts": prompts_for(c),
        })

    # Phase 2 — dispatch model queries.
    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        prompts = job["prompts"]
        r_m, r_s, r_d, r_h = query_four_formats_multi(
            model_name, job["audios"],
            prompts["midi"], prompts["spn"], prompts["doremi"], prompts["hz"],
            verbose=False,
        )
        return standard_pitch_record(
            wav=job["audios"][1],            # target wav is the "primary" record path
            source=c["source"],
            source_type=c["source_type"],
            midi_gt=c["midi"],
            raw_midi=r_m["result"], raw_spn=r_s["result"],
            raw_doremi=r_d["result"], raw_hz=r_h["result"],
            prompt_midi=prompts["midi"], prompt_spn=prompts["spn"],
            prompt_doremi=prompts["doremi"], prompt_hz=prompts["hz"],
            ref_midi=c["ref_midi"],
            interval=c["interval"],
            ref_wav=job["audios"][0],
        )

    raw = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=_label,
        result_label_fn=lambda j, r: pitch_record_audit_str(r, label=_label(j)),
    )
    records: list[dict] = [r for r in raw if r is not None]

    # Phase 3 — summary.
    n_total = len(records)

    def _acc(fmt: str) -> float:
        if not records:
            return 0.0
        vals = [r[f"{fmt}_correct"] for r in records
                if isinstance(r.get(f"{fmt}_correct"), (int, float, bool))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "total":       n_total,
        "condition_n": n_total,
        "accuracy":    {"n": n_total, **{fmt: _acc(fmt) for fmt in FORMATS}},
    }
    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli : {n_total}",
        "",
        f"  {'Format':>6}  {'Accuracy':>9}",
        f"  {'─' * 18}",
    ]
    for fmt in FORMATS:
        summary_lines.append(
            f"  {fmt.upper():>6}  {summary['accuracy'][fmt]:>9.1%}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name,
        model_info=info,
        record_extras=["ref_midi", "interval", "ref_wav"],
        **(sample_info or {}),
    )
    save_results(
        EXP_NAME, model_name, records, summary, metadata, summary_lines,
        run_dir=run_dir, formats=FORMATS,
    )
    return summary


def _run(mode: str) -> dict | None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    if args.preview:
        mode = "preview"

    all_conds = build_conditions()
    conds, s_meta = apply_default_sampling(
        EXP_NAME, all_conds, args.sample_n, args.sample_seed,
    )

    # Always materialise the audio in both modes (preview = stimuli only).
    for c in conds:
        wavs_for(c)

    if mode == "preview":
        print(f"Experiment : {EXP_NAME}")
        print(f"Stimuli    : {len(conds)}  (each = ref wav + target wav)")
        print(f"Audio dir  : {config.AUDIO_DIR}/{EXP_NAME}")
        for line in sampling_summary_lines(s_meta):
            print(line)
        print("\nRun without --preview to query the model(s).")
        return None

    targets = args.models or list(config.MODELS)
    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(targets)}")
    print(f"Stimuli    : {len(conds)}")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    summaries: dict[str, dict] = {}
    for m in targets:
        summaries[m] = _run_one_model(m, conds, run_dir, s_meta)
    save_comparison(run_dir, summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(summaries.keys()))


def preview() -> None:
    _run("preview")


def run() -> dict | None:
    return _run("run")


if __name__ == "__main__":
    run()
