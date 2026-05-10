"""
d7d — Pitch with reference, split-audio format, a1-style pitch list.

Companion to d7b: same split-audio design (reference and target delivered as
two separate audio inputs) but target pitches come from an explicit list
(``pitchbench_d7d_PITCHES``) rather than being derived from
``ref_midi + interval``.  This lets the experiment share exactly the same
target pitch set as a1, so results are directly comparable.

In analysis mode (MODE=ANALYSIS, q1 preset) the pitch list is set to the 7
ablation pitches [29, 38, 47, 56, 65, 74, 83], giving 2 ref × 7 pitches × 6
sources = 84 conditions — the same 42 notes per reference that a1 uses in the
ablation study.

The interval is computed as ``tgt_midi − ref_midi`` and stored for analysis.

Anchored only — d7c already provides the no-reference baseline.

Model compatibility
-------------------
Each target model is probed once at the start of the run by issuing a tiny
multi-audio call.  Models whose server returns 501 (or whose API path raises
``NotImplementedError``) are skipped; the run continues with the supported
subset.

Usage::
    pitchbench --id d7d --preview
    pitchbench --id d7d --models openrouter/google/gemini-2.5-flash
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import (
    get_model_info, query_alm_multi, query_four_formats_multi,
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
    apply_default_sampling, sampling_summary_lines, export_sampled_conditions_csv,
)


EXP_NAME          = Path(__file__).stem
SOURCES           = config.pitchbench_d7d_SOURCES
REFERENCE_PITCHES = config.pitchbench_d7d_REFERENCE_PITCHES
PITCHES           = config.pitchbench_d7d_PITCHES
TONE_DURATION_MS  = config.pitchbench_d7d_TONE_DURATION_MS

FORMATS = ("midi", "spn", "doremi", "hz")


def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for ref_midi in REFERENCE_PITCHES:
        for tgt_midi in PITCHES:
            interval = tgt_midi - ref_midi
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
            f"You will hear two audios. Audio 1 is a reference tone whose MIDI "
            f"note number is {ref_midi}. Audio 2 is the target tone. What is "
            f"the MIDI note number of Audio 2? Reply with ONLY the integer.\n"
            f"Audio 1: <AUDIO1>\nAudio 2: <AUDIO2>"
        ),
        "spn": (
            f"You will hear two audios. Audio 1 is a reference tone — note "
            f"name {ref_note}. Audio 2 is the target tone. What is the note "
            f"name and octave of Audio 2, e.g. C4, F#3? Reply with ONLY the "
            f"note name in Scientific Pitch Notation.\n"
            f"Audio 1: <AUDIO1>\nAudio 2: <AUDIO2>"
        ),
        "doremi": (
            f"You will hear two audios. Audio 1 is a reference tone — solfège "
            f"syllable '{ref_solf}' (fixed-do: do=C re=D mi=E fa=F sol=G la=A "
            f"si=B). Audio 2 is the target tone. What is the solfège syllable "
            f"and accidental (if needed) of Audio 2? Reply with ONLY the "
            f"syllable and accidental.\n"
            f"Audio 1: <AUDIO1>\nAudio 2: <AUDIO2>"
        ),
        "hz": (
            f"You will hear two audios. Audio 1 is a reference tone at "
            f"{ref_hz} Hz. Audio 2 is the target tone. What is the pitch "
            f"frequency of Audio 2 in Hertz? Reply with ONLY a number "
            f"(the frequency in Hz).\n"
            f"Audio 1: <AUDIO1>\nAudio 2: <AUDIO2>"
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


def _probe_multi_audio_supported(model_name: str, probe_audios: list[str]) -> bool:
    try:
        query_alm_multi(
            model_name, probe_audios,
            prompt="<AUDIO1>\n<AUDIO2>\nReply with one word.",
            max_new_tokens=10,
            timeout_s=60.0,
        )
        return True
    except NotImplementedError:
        return False


def _label(j: dict) -> str:
    c = j["cond"]
    return (
        f"r={midi_to_note(c['ref_midi']):>3} "
        f"t={midi_to_note(c['midi']):>3} {c['source']}"
    )


def _run_one_model(
    model_name:  str,
    conds:       list[dict],
    run_dir:     Path,
    sample_info: dict | None,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    jobs: list[dict] = []
    for c in conds:
        ref_wav, tgt_wav = wavs_for(c)
        jobs.append({
            "audios":  [str(ref_wav), str(tgt_wav)],
            "cond":    c,
            "prompts": prompts_for(c),
        })

    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        prompts = job["prompts"]
        r_m, r_s, r_d, r_h = query_four_formats_multi(
            model_name, job["audios"],
            prompts["midi"], prompts["spn"], prompts["doremi"], prompts["hz"],
            verbose=False,
        )
        return standard_pitch_record(
            wav=job["audios"][1],
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

    for c in conds:
        wavs_for(c)

    if mode == "preview":
        export_sampled_conditions_csv(EXP_NAME, conds)
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

    probe_ref, probe_tgt = wavs_for(conds[0])
    probe_audios = [str(probe_ref), str(probe_tgt)]

    print("\nProbing multi-audio support...")
    supported: list[str] = []
    for m in targets:
        print(f"  {m:50}", end=" ", flush=True)
        if _probe_multi_audio_supported(m, probe_audios):
            print("OK")
            supported.append(m)
        else:
            print("not supported — skipping")

    if not supported:
        print("\nNo target model supports multi-audio queries; nothing to run.")
        return None

    run_dir = make_run_dir(EXP_NAME)
    summaries: dict[str, dict] = {}
    for m in supported:
        summaries[m] = _run_one_model(m, conds, run_dir, s_meta)
    save_comparison(run_dir, summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(summaries.keys()))


def preview() -> None:
    _run("preview")


def run() -> dict | None:
    return _run("run")


if __name__ == "__main__":
    run()
