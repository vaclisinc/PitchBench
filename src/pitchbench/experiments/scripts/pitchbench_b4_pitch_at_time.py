"""
b4 — Pitch identification at a specific time within a sequence.

Question: among multiple non-overlapping notes, can the ALM correctly
identify the pitch sounding at a queried time?

Universal IVs: duration_ms (per note), midi (the queried target pitch),
                source.
Experiment-specific IVs:
    n_notes:         {3, 5}
    query_pos:       index of the queried note within the sequence
                     (always within the note interior — query_time = onset
                     of the note + duration/2)

Fixed conditions: 20 s clip; non-overlapping; equal level; gaps drawn from
[300, 1500] ms with a per-cell seed; 4 pitch-ID prompt variants.

Scoring: standard 4-format (MIDI / SPN / Doremi / Hz).
"""

from __future__ import annotations

import argparse
import random
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

EXP_NAME = Path(__file__).stem

N_NOTES_OPTS:  list[int] = [3, 5]
TOTAL_DUR_MS              = 20_000
GAP_MIN_MS, GAP_MAX_MS    = 30, 1500
DEFAULT_SEED = config.DEFAULT_SEED

SOURCES: list[str] = config.ALL_SOURCES


def _query_str(secs: float) -> str:
    minutes = int(secs // 60)
    seconds = secs - minutes * 60
    return f"{minutes}:{seconds:05.2f}"


def _prompt_set(query_time_s: float) -> tuple[str, str, str, str]:
    qs = _query_str(query_time_s)
    prefix = (
        f"This audio contains a sequence of musical notes separated by silence. "
        f"Identify the pitch that is sounding at exactly {qs}. "
    )
    return (
        prefix + PROMPT_MIDI,
        prefix + PROMPT_SPN,
        prefix + PROMPT_DOREMI,
        prefix + PROMPT_HZ,
    )


def build_conditions(
    durations_ms: list[int], pitches: list[int], sources: list[str], seed: int,
) -> list[dict]:
    rng_seed = seed
    rows: list[dict] = []
    for src in sources:
        for tgt in pitches:
            for n in N_NOTES_OPTS:
                for dur in durations_ms:
                    cell_seed = (rng_seed ^ hash((src, tgt, n, dur))) & 0xFFFFFFFF
                    sub_rng   = random.Random(cell_seed)
                    distractors = [p for p in pitches if p != tgt]
                    other_pitches = sub_rng.sample(distractors, n - 1)
                    midis_seq = list(other_pitches)
                    insert_at = sub_rng.randint(0, n - 1)
                    midis_seq.insert(insert_at, tgt)
                    gaps = [sub_rng.randint(GAP_MIN_MS, GAP_MAX_MS) for _ in range(n + 1)]
                    total = sum(gaps) + dur * n
                    if total > TOTAL_DUR_MS:
                        continue
                    onsets: list[int] = []
                    cursor = gaps[0]
                    for _ in range(n):
                        onsets.append(cursor)
                        cursor += dur + gaps[len(onsets)]
                    target_idx = midis_seq.index(tgt)
                    query_time_s = (onsets[target_idx] + dur / 2) / 1000
                    rows.append({
                        "source":       src,
                        "midi":         tgt,
                        "duration_ms":  dur,
                        "n_notes":      n,
                        "midi_seq":     midis_seq,
                        "onsets_ms":    onsets,
                        "target_idx":   target_idx,
                        "query_time_s": round(query_time_s, 3),
                        "seed":         cell_seed,
                    })
    return rows


def _wav_for(c: dict) -> Path:
    triples = [(m, c["onsets_ms"][i], c["duration_ms"]) for i, m in enumerate(c["midi_seq"])]
    path, _ = engine.clip_with_notes(triples, c["source"], TOTAL_DUR_MS, name_hint="b4")
    return path


def run_one_model(model_name: str, conds: list[dict], run_dir: Path) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    records: list[dict] = []
    for c in conds:
        wav = str(_wav_for(c))
        p_m, p_s, p_d, p_h = _prompt_set(c["query_time_s"])
        r_m, r_s, r_d, r_h = query_four_formats(model_name, wav, p_m, p_s, p_d, p_h)
        rec = standard_pitch_record(
            wav=wav, source=c["source"],
            source_type="waveform" if c["source"] in config.WAVEFORMS else "instrument",
            midi_gt=c["midi"],
            raw_midi=r_m["result"], raw_spn=r_s["result"],
            raw_doremi=r_d["result"], raw_hz=r_h["result"],
            prompt_midi=p_m, prompt_spn=p_s, prompt_doremi=p_d, prompt_hz=p_h,
            duration_ms=c["duration_ms"],
            n_notes=c["n_notes"],
            query_time_s=c["query_time_s"],
            seed=c["seed"],
        )
        rec["model_params_midi"] = r_m["model_params"]
        records.append(rec)

    n = len(records)
    summary: dict[str, float | int] = {"total": n}
    for fmt in ("midi", "spn", "doremi", "hz"):
        col = f"{fmt}_correct"
        summary[f"acc_{fmt}"] = round(sum(r[col] for r in records) / max(1, n), 4)
    summary["acc_midi_within_1"] = round(sum(r["midi_within_1"] for r in records) / max(1, n), 4)

    summary_lines = [f"  Stimuli : {n}", ""]
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
        n_notes_opts=N_NOTES_OPTS, total_dur_ms=TOTAL_DUR_MS,
        gap_range_ms=[GAP_MIN_MS, GAP_MAX_MS],
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--models",  nargs="+", metavar="MODEL")
    parser.add_argument("--sources", nargs="+", metavar="SRC", default=None)
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    sources = args.sources or SOURCES
    conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources, args.seed)
    for c in conds:
        try: _wav_for(c)
        except ValueError as exc:
            print(f"    [SKIP] {exc}")
    print(f"Experiment : {EXP_NAME}")
    print(f"Sources    : {sources}")
    print(f"Stimuli    : {len(conds)}")
    print(f"Audio dir  : {config.AUDIO_DIR}/{EXP_NAME}")
    print("\nRun without --preview to query the model(s).")


def run() -> None:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    sources = args.sources or SOURCES
    conds = build_conditions(config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, sources, args.seed)
    for c in conds:
        try: _wav_for(c)
        except ValueError: pass

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)}")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for m in target_models:
        all_summaries[m] = run_one_model(m, conds, run_dir)
    save_comparison(run_dir, all_summaries, EXP_NAME)


if __name__ == "__main__":
    args = _parse_args()
    (preview if args.preview else run)()
