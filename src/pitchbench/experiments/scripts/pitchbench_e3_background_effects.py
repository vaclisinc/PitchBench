"""
e3 — Background-noise effects on pitch identification.

Question: do additive backgrounds (white noise + real-world recordings)
impair the ALM's absolute pitch hearing?

Universal IVs: duration_ms, midi, source.
Experiment-specific IVs:
    background:  {white_noise, church-bells, crowd-noise, rain, street-noise}
    snr_db:      {30, 20, 0, -6}

Backgrounds:
    white_noise           — Gaussian, deterministic per seed=0
    church-bells / crowd-noise / rain / street-noise
                          — loaded from data/downloaded/background/<name>.mp3,
                            truncated to the fragment length (no looping).

Each mix is normalised to peak 0.9 after combining tone + scaled background.
4-format pitch-ID prompts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_four_formats
from pitchbench.experiments.helpers.music import (
    PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI, PROMPT_SPN,
    midi_to_note, standard_pitch_record,
)
from pitchbench.experiments.helpers.results import (
    get_run_metadata, make_run_dir, save_comparison, save_results,
)
from pitchbench.experiments.helpers.sampling import sampling_meta, sampling_summary_lines, stratified_sample

EXP_NAME = Path(__file__).stem

BACKGROUNDS: list[str] = [
    "white_noise",
    "church-bells", "crowd-noise", "rain", "street-noise",
]
SNR_DB:      list[float] = [30.0, 20.0, 0.0, -6.0]
SOURCES:     list[str]   = config.ALL_SOURCES

PROMPT_PREFIX = (
    "This audio contains a single sustained musical note mixed with a "
    "background sound. Identify the PITCH of the note, ignoring the "
    "background. "
)
PROMPT_MIDI_FULL   = PROMPT_PREFIX + PROMPT_MIDI
PROMPT_SPN_FULL    = PROMPT_PREFIX + PROMPT_SPN
PROMPT_DOREMI_FULL = PROMPT_PREFIX + PROMPT_DOREMI
PROMPT_HZ_FULL     = PROMPT_PREFIX + PROMPT_HZ


def build_conditions(durations_ms: list[int], pitches: list[int], sources: list[str]) -> list[dict]:
    rows: list[dict] = []
    for src in sources:
        for midi in pitches:
            for dur in durations_ms:
                for bg in BACKGROUNDS:
                    for snr in SNR_DB:
                        rows.append({
                            "source":      src,
                            "duration_ms": dur,
                            "midi":        midi,
                            "background":  bg,
                            "snr_db":      snr,
                        })
    return rows


def _wav_for(c: dict) -> Path:
    return engine.tone_with_background(
        c["midi"], c["source"], c["duration_ms"], c["background"], c["snr_db"],
    )


def run_one_model(model_name: str, conds: list[dict], run_dir: Path, sample_info: dict | None = None) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        try:
            wav = str(_wav_for(c))
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
            continue
        r_m, r_s, r_d, r_h = query_four_formats(
            model_name, wav,
            PROMPT_MIDI_FULL, PROMPT_SPN_FULL, PROMPT_DOREMI_FULL, PROMPT_HZ_FULL,
        )
        rec = standard_pitch_record(
            wav=wav,
            source=c["source"],
            source_type="waveform" if c["source"] in config.WAVEFORMS else "instrument",
            midi_gt=c["midi"],
            raw_midi=r_m["result"], raw_spn=r_s["result"],
            raw_doremi=r_d["result"], raw_hz=r_h["result"],
            prompt_midi=PROMPT_MIDI_FULL, prompt_spn=PROMPT_SPN_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL, prompt_hz=PROMPT_HZ_FULL,
            duration_ms=c["duration_ms"],
            background=c["background"],
            snr_db=c["snr_db"],
        )
        rec["model_params_midi"] = r_m["model_params"]
        records.append(rec)

    n = len(records)
    summary: dict[str, float | int] = {"total": n}
    for fmt in ("midi", "spn", "doremi", "hz"):
        col = f"{fmt}_correct"
        summary[f"acc_{fmt}"] = round(sum(r[col] for r in records) / max(1, n), 4)
    summary["acc_midi_within_1"] = round(
        sum(r["midi_within_1"] for r in records) / max(1, n), 4
    )

    summary_lines = sampling_summary_lines(sample_info or {}) + [
        f"  Stimuli   : {n}",
        f"  Backgrnds : {BACKGROUNDS}",
        f"  SNR (dB)  : {SNR_DB}",
        "",
    ]
    for fmt in ("midi", "spn", "doremi", "hz"):
        summary_lines.append(f"  {fmt.upper():>6}  {summary[f'acc_{fmt}']:.1%}")
    summary_lines.append(f"  MIDI±1  {summary['acc_midi_within_1']:.1%}")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SOURCES, pitches=config.DEFAULT_PITCHES,
        durations_ms=config.DEFAULT_DURATIONS_MS,
        backgrounds=BACKGROUNDS, snr_db=SNR_DB,
        prompt_midi=PROMPT_MIDI_FULL, prompt_spn=PROMPT_SPN_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL, prompt_hz=PROMPT_HZ_FULL,
        **(sample_info or {}),
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL")
    parser.add_argument("--sources", nargs="+", metavar="SRC", default=None)
    parser.add_argument("--sample-n",     type=int, default=None, metavar="N",
                        help="Draw N stimuli (stratified by (source, background))")
    parser.add_argument("--sample-seed",  type=int, default=42,   metavar="SEED")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args    = _parse_args()
    sources = args.sources or SOURCES
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(
            all_conds, args.sample_n,
            lambda c: (c["source"], c["background"]),
            seed=args.sample_seed,
        )
    s_meta = sampling_meta(len(all_conds), "(source, background)", args.sample_n, args.sample_seed)
    for c in conds:
        try: _wav_for(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {sources}")
    print(f"Stimuli    : {len(conds)}")
    print(f"Audio dir  : {config.AUDIO_DIR}/{EXP_NAME}")
    for line in sampling_summary_lines(s_meta):
        print(line)
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args    = _parse_args()
    target_models = args.models or list(config.MODELS)
    sources = args.sources or SOURCES
    all_conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources)
    conds = all_conds
    if args.sample_n is not None:
        conds = stratified_sample(
            all_conds, args.sample_n,
            lambda c: (c["source"], c["background"]),
            seed=args.sample_seed,
        )
    s_meta = sampling_meta(len(all_conds), "(source, background)", args.sample_n, args.sample_seed)
    for c in conds:
        try: _wav_for(c)
        except ValueError: pass

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")
    for line in sampling_summary_lines(s_meta):
        print(line)

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for m in target_models:
        all_summaries[m] = run_one_model(m, conds, run_dir, s_meta)
    save_comparison(run_dir, all_summaries, EXP_NAME)


if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
