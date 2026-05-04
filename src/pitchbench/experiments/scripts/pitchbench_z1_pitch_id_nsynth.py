"""
Experiment 03 — Pitch recognition on NSynth (real instrument sounds)
Tests whether models can identify pitch from real instrument recordings.

Three prompt conditions per sample:
  - MIDI:   ask for the MIDI note number    (e.g. 60, 54)
  - ABC:    ask for the note name + octave  (e.g. "C4", "F#3")
  - Doremi: ask for the solfege syllable and accidental (if needed)    (e.g. "do", "sol#")

Usage:
    python experiments/run.py exp_3_nsynth
    python experiments/run.py exp_3_nsynth --preview
    python experiments/run.py exp_3_nsynth --models audio_flamingo_3 audio_flamingo_next_instruct
    python experiments/run.py exp_3_nsynth --n-per-family 5 --seed 123
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pitchbench.config as config
from pitchbench.experiments.helpers.api import get_model_info, query_three_formats
from pitchbench.experiments.helpers.music import (
    PROMPT_ABC, PROMPT_MIDI, PROMPT_DOREMI,
    midi_to_note,
    standard_pitch_record, wide_to_long_records,
)
from pitchbench.experiments.helpers.plots import save_accuracy_plots
from pitchbench.experiments.helpers.results import exp_data_dir, get_run_metadata, make_run_dir, save_comparison, save_results

EXP_NAME = Path(__file__).stem

DEFAULT_N_PER_FAMILY = 10
DEFAULT_SEED = config.DEFAULT_SEED

PROMPT_DOREMI_FULL = "Listen to this audio recording of a single musical note. " + PROMPT_DOREMI
PROMPT_ABC_FULL    = "Listen to this audio recording of a single musical note. " + PROMPT_ABC
PROMPT_MIDI_FULL   = "Listen to this audio recording of a single musical note. " + PROMPT_MIDI


def load_nsynth_examples() -> dict:
    meta_path = config.NSYNTH_VALID_DIR / "examples.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"NSynth metadata not found at {meta_path}\n"
            "Download the NSynth validation set and place it under _datasets/NSynth/nsynth-valid/"
        )
    with open(meta_path) as f:
        return json.load(f)


def sample_per_family(examples: dict, n: int, seed: int) -> list[dict]:
    """Return n items per instrument family, chosen deterministically with the given seed."""
    by_family: dict[str, list[dict]] = {}
    for key, meta in examples.items():
        fam = meta["instrument_family_str"]
        by_family.setdefault(fam, []).append({**meta, "note_str": key})

    rng = random.Random(seed)
    selected = []
    for fam in sorted(by_family):
        pool = by_family[fam]
        chosen = rng.sample(pool, min(n, len(pool)))
        selected.extend(chosen)
    return selected


def run_one_model(
    model_name: str,
    sample: list[dict[str, Any]],
    n_per_family: int,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    info = get_model_info(model_name)

    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")
    print(f"  URL   : {config.MODEL_URLS[model_name]}")

    audio_dir = config.NSYNTH_VALID_DIR / "audio"
    records: list[dict] = []

    for item in sample:
        wav = str(audio_dir / f"{item['note_str']}.wav")
        gt_midi = item["pitch"]

        print(f"    {item['note_str']}")
        raw_midi, raw_abc, raw_doremi = query_three_formats(
            model_name, wav,
            PROMPT_MIDI_FULL, PROMPT_ABC_FULL, PROMPT_DOREMI_FULL,
            verbose=True,
        )

        record = standard_pitch_record(
            wav=wav,
            source=item["note_str"],
            source_type="instrument",
            midi_gt=gt_midi,
            raw_midi=raw_midi,
            raw_abc=raw_abc,
            raw_doremi=raw_doremi,
            prompt_midi=PROMPT_MIDI_FULL,
            prompt_abc=PROMPT_ABC_FULL,
            prompt_doremi=PROMPT_DOREMI_FULL,
            note_str=item["note_str"],
            instrument_family=item["instrument_family_str"],
            instrument_source=item["instrument_source_str"],
        )
        records.append(record)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(records)
    midi_correct   = sum(r["midi_correct"]   for r in records)
    abc_correct    = sum(r["abc_correct"]    for r in records)
    doremi_correct = sum(r["doremi_correct"] for r in records)
    midi_within_1  = sum(r["midi_within_1"]  for r in records)

    summary = {
        "total_samples": n,
        "n_per_family":  n_per_family,
        "seed":          seed,
        "midi_correct":    midi_correct,
        "midi_within_1":   midi_within_1,
        "abc_correct":     abc_correct,
        "doremi_correct":  doremi_correct,
    }

    summary_lines = [
        f"  Total samples            : {n}  ({n_per_family}/family, seed={seed})",
        f"",
        f"  [MIDI — integer]",
        f"    Exact match            : {midi_correct} / {n}  ({midi_correct/n:.1%})",
        f"    Within 1               : {midi_within_1} / {n}",
        f"",
        f"  [ABC — note name]",
        f"    Exact match            : {abc_correct} / {n}  ({abc_correct/n:.1%})",
        f"",
        f"  [Doremi — solfege]",
        f"    Exact match            : {doremi_correct} / {n}  ({doremi_correct/n:.1%})",
    ]

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    metadata = get_run_metadata(
        model_name=model_name,
        model_info=info,
        nsynth_split="valid",
        n_per_family=n_per_family,
        seed=seed,
        prompt_midi=PROMPT_MIDI_FULL,
        prompt_abc=PROMPT_ABC_FULL,
        prompt_doremi=PROMPT_DOREMI_FULL,
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)

    long_records = wide_to_long_records(records)
    save_accuracy_plots(
        long_records, run_dir, model_name,
        instrument_key="instrument_family",
        pitch_key="midi_gt",
        prompt_key="prompt_variant",
        accuracy_key="exact_match",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Show sample selection, do not query models")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help=f"Model slugs to test (default: all). Available: {list(config.MODELS)}")
    parser.add_argument("--n-per-family", type=int, default=DEFAULT_N_PER_FAMILY,
                        help=f"Samples per instrument family (default: {DEFAULT_N_PER_FAMILY})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for sampling (default: {DEFAULT_SEED})")
    args, _ = parser.parse_known_args()
    return args


def run() -> None:
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    examples = load_nsynth_examples()
    sample = sample_per_family(examples, args.n_per_family, args.seed)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Samples    : {len(sample)}  ({args.n_per_family}/family, seed={args.seed})")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict[str, Any]] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(model_name, sample, args.n_per_family, args.seed, run_dir)
    save_comparison(run_dir, all_summaries, EXP_NAME)


def preview() -> None:
    args = _parse_args()
    examples = load_nsynth_examples()
    sample = sample_per_family(examples, args.n_per_family, args.seed)
    print(f"NSynth preview — {len(sample)} samples "
          f"({args.n_per_family} per family, seed={args.seed})\n")
    for item in sample:
        wav = config.NSYNTH_VALID_DIR / "audio" / f"{item['note_str']}.wav"
        status = "OK" if wav.exists() else "MISSING"
        print(f"  [{status}] {item['note_str']:40s}  "
              f"pitch={item['pitch']:3d}  "
              f"({midi_to_note(item['pitch'])})  "
              f"family={item['instrument_family_str']}")
    print(f"\nRun without --preview to query the model(s).")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
