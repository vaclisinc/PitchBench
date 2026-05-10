"""
Upload the staged PitchBench dataset to a Hugging Face dataset repo.

Run `build_hf_dataset.py` first to stage `pitchbench_hf/<exp>/{*.wav,metadata.jsonl}`.
Then run this script:

    huggingface-cli login                       # if not already
    python upload_to_hf.py
    python upload_to_hf.py --private            # private first
    python upload_to_hf.py --dry-run            # write README, no push
    python upload_to_hf.py --repo other/Name    # override target repo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Lines reproduced from each experiment's docstring (one-line summary).
EXP_DESCRIPTIONS = {
    "pitchbench_a1_single_pitch_id":              "Identify the pitch of a single tone.",
    "pitchbench_d7a_pitch_with_reference":       "Identify a target pitch given a named reference tone.",
    "pitchbench_d7b_pitch_with_reference_split": "Identify a target pitch given a named reference tone (two-clip format).",
    "pitchbench_a3_single_pitch_by_duration":     "Pitch ID across very short to very long tone durations.",
    "pitchbench_e5_vibrato":    "Pitch ID with vibrato (rate × depth sweep).",
    "pitchbench_e6_slightly_off":    "Pitch ID when the tone is detuned by a fraction of a semitone.",
    "pitchbench_b1_single_pitch_within_silence":      "Pitch ID when the tone is hidden in a long silent stimulus.",
    "pitchbench_b3_timestamp_single_pitch":   "Predict the onset/offset times of a single tone in silence.",
    "pitchbench_b4_timestamp_specific_pitch": "Predict onset/offset of a specific named target among distractors.",
    "pitchbench_b2_pitch_at_timestamp":         "Identify which pitch is sounding at a given timestamp.",
    "pitchbench_b5_timestamp_multiple_pitches":     "Predict onset/offset for every note in a sequence.",
    "pitchbench_c2_chord_dyad_interval":         "Identify the interval (in semitones) of a two-note dyad.",
    "pitchbench_c1_chord_count_pitches":     "Count the number of simultaneous pitches in a chord.",
    "pitchbench_c4_chord_pitches":        "List every pitch in a chord (dyad / triad / seventh).",
    "pitchbench_c3_chord_quality":         "Classify chord quality (major, minor, dim, aug, 7th, sus, …).",
    "pitchbench_d1_sequence_count_pitches":       "Count the number of distinct pitches in a sequential passage.",
    "pitchbench_d2_dyad_lower_higher_difference":      "Decide whether the second tone is higher or lower (cents-scale).",
    "pitchbench_d6_sequence_dyad_interval":       "Identify the interval between two sequentially played pitches.",
    "pitchbench_d3_contour_discrete":      "Describe the up/down contour of a discrete-step melody.",
    "pitchbench_d4_contour_continuous":    "Describe the contour of a continuous pitch glide.",
    "pitchbench_d5_sequence_ranking_by_pitch":         "Rank N tones (small cents-scale differences) from low to high.",
    "pitchbench_d8_sequence_pitches":          "Transcribe every pitch in a melodic sequence.",
    "pitchbench_a2_single_pitch_by_loudness":              "Pitch ID at varying loudness levels.",
    "pitchbench_e1_audio_effects":         "Pitch ID under audio effects (reverb, EQ, clip, saturation, …).",
    "pitchbench_e2_background":    "Pitch ID embedded in real-world background noise (rain, crowd, …).",
    "pitchbench_e3_harmonic_saturation":   "Pitch ID under increasing harmonic-saturation drive.",
    "pitchbench_e4_time_stretching":          "Pitch ID with resample (pitch-shift) vs time-stretch (pitch preserved).",
    "pitchbench_f1_melodic_line_atonal":       "Identify the pitch sequence of one part within a polyphonic mix.",
    "pitchbench_f2_melodic_line_tonal":      "Identify a target voice in a four-part Bach chorale rendering.",
}


def _format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


_PARQUET_NAME = "test-00000-of-00001.parquet"
_D7B = "pitchbench_d7b_pitch_with_reference_split"


def _count_rows(exp_dir: Path) -> int:
    parquet = exp_dir / _PARQUET_NAME
    if parquet.exists():
        import pyarrow.parquet as pq
        return pq.read_metadata(str(parquet)).num_rows
    meta = exp_dir / "metadata.jsonl"
    if meta.exists():
        return sum(1 for _ in meta.open())
    return 0


def write_readme(staging: Path) -> Path:
    """Write a dataset card README.md at staging/README.md.

    Uses Parquet data_files paths so HF uses the standard Parquet reader
    for all configs (including d7b which has audio_1 + audio_2 columns).
    """
    configs = []
    for exp_dir in sorted(p for p in staging.iterdir() if p.is_dir()):
        if not (exp_dir / _PARQUET_NAME).exists() and not (exp_dir / "metadata.jsonl").exists():
            continue
        n_rows = _count_rows(exp_dir)
        configs.append({"name": exp_dir.name, "rows": n_rows})

    yaml_configs = "\n".join(
        f"  - config_name: {c['name']}\n"
        f"    data_files:\n"
        f"      - split: test\n"
        f"        path: {c['name']}/{_PARQUET_NAME}\n"
        for c in configs
    )

    body_table = "\n".join(
        f"| `{c['name']}` | {c['rows']:,} | {EXP_DESCRIPTIONS.get(c['name'], '')} |"
        for c in configs
    )
    total_rows = sum(c["rows"] for c in configs)

    readme = f"""---
license: cc-by-4.0
task_categories:
  - audio-classification
  - audio-text-to-text
language:
  - en
tags:
  - audio
  - music
  - pitch
  - benchmark
  - alm
  - audio-language-model
pretty_name: PitchBench
size_categories:
  - 1K<n<10K
configs:
{yaml_configs}---

# PitchBench

A benchmark for testing what audio / acoustic signals **Audio Language Models (ALMs)** do
and don't understand. PitchBench probes pitch perception across {len(configs)} controlled
experiments — single-pitch ID, onsets/offsets, chords, sequences, contour, audio
effects, and polyphonic streams.

Each row is one **(audio, question, answer)** triple: a short WAV stimulus, the
question (`prompt*`) asked of the model, and the ground-truth answer fields
(experiment-specific column names).

## Quick start

```python
from datasets import load_dataset

# Load one experiment (configurations match experiment IDs)
ds = load_dataset("REPO_PLACEHOLDER", "pitchbench_a1_single_pitch_id", split="test")
print(ds[0]["audio"], ds[0]["prompt_midi"], ds[0]["gt_midi"])

# d7b has two audio columns: audio_1 (reference tone) and audio_2 (target tone)
ds_d7b = load_dataset("REPO_PLACEHOLDER",
                      "pitchbench_d7b_pitch_with_reference_split", split="test")
print(ds_d7b[0]["audio_1"], ds_d7b[0]["audio_2"], ds_d7b[0]["gt_midi"])

# Iterate every experiment
import datasets
for cfg in datasets.get_dataset_config_names("REPO_PLACEHOLDER"):
    ds = load_dataset("REPO_PLACEHOLDER", cfg, split="test")
    print(cfg, len(ds))
```

## Experiments ({len(configs)} configs, {total_rows:,} total stimuli)

| Config | # rows | Question |
|---|---:|---|
{body_table}

## Schema

Every row has:

- `audio` — the audio stimulus (16 kHz mono). `pitchbench_d7b_pitch_with_reference_split` has `audio_1` (reference tone) and `audio_2` (target tone) instead.
- `prompt` *or* one or more of `prompt_midi`, `prompt_spn`, `prompt_abc`,
  `prompt_doremi`, `prompt_hz` — the question(s) put to the model.
- Experiment-specific ground-truth fields (e.g. `midi`, `n`, `interval_st`,
  `chord_quality_gt`, `pattern_gt`, `traj_name`, `midi_sequence`, …).

## Reproducibility

Stimuli are generated deterministically from the configuration in
`pitchbench.config` (`EVAL=True` benchmark constants). The subset published
here is the seeded stratified sample used in the paper — reproduced by
`apply_default_sampling(EXP_NAME, all_conds, None, seed=42)` over the output
of each experiment's `build_conditions(...)`. Source code is provided as
supplementary material with the submission (anonymized for review).

## License

Released under **CC-BY-4.0**.

## Citation

```bibtex
@misc{{pitchbench2026,
  title  = {{PitchBench: A Benchmark for Pitch Understanding in Audio Language Models}},
  author = {{Anonymous Authors}},
  year   = {{2026}},
  note   = {{Under review at NeurIPS 2026 Datasets \\& Benchmarks Track}},
}}
```
"""
    out = staging / "README.md"
    out.write_text(readme, encoding="utf-8")
    return out


def upload(staging: Path, repo_id: str, private: bool, token: str | None) -> None:
    import shutil
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    # upload_large_folder caches resume state at <staging>/.cache/huggingface/upload/.
    # That cache is keyed per-folder, not per-repo: if the target repo has
    # changed since the last attempt, the cache will incorrectly mark files as
    # already committed and the upload will silently push only the README.
    # Stamp the target repo and invalidate when it changes.
    cache_dir = staging / ".cache" / "huggingface" / "upload"
    stamp = staging / ".cache" / "huggingface" / "_target_repo"
    prev = stamp.read_text().strip() if stamp.exists() else None
    if prev and prev != repo_id and cache_dir.exists():
        print(f"Target repo changed ({prev} → {repo_id}); clearing resume cache.")
        shutil.rmtree(cache_dir)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(repo_id)

    print(f"Uploading {staging} → {repo_id} (this can take a while)…")
    api.upload_large_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Verify: list the repo and ensure files actually landed.
    files = api.list_repo_files(repo_id, repo_type="dataset")
    n_wav = sum(1 for f in files if f.endswith(".wav"))
    n_meta = sum(1 for f in files if f.endswith("metadata.jsonl"))
    print(f"\nRepo now contains: {len(files)} files ({n_wav} wav, {n_meta} metadata.jsonl).")
    if n_wav == 0:
        raise SystemExit(
            "No WAV files in the remote repo — upload silently failed. "
            "Try removing the resume cache and re-running:\n"
            f"  rm -rf {cache_dir}\n  python upload_to_hf.py"
        )
    print(f"Done. Dataset is at https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--staging", type=Path, default=Path("pitchbench_hf"))
    p.add_argument("--repo", default="pitchbench-authors/PitchBench",
                   help="Target HF repo (default: pitchbench-authors/PitchBench)")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private (flip to public later in the UI).")
    p.add_argument("--token", default=None,
                   help="HF token (defaults to cached login from `huggingface-cli login`).")
    p.add_argument("--dry-run", action="store_true",
                   help="Write README.md but skip the upload.")
    args = p.parse_args()

    if not args.staging.exists():
        raise SystemExit(
            f"Staging dir {args.staging!s} not found. "
            f"Run `python build_hf_dataset.py --out {args.staging}` first."
        )

    readme = write_readme(args.staging)
    # Patch in the actual repo id for the Quick-start examples.
    readme.write_text(
        readme.read_text().replace("REPO_PLACEHOLDER", args.repo),
        encoding="utf-8",
    )
    print(f"Wrote {readme}")

    if args.dry_run:
        print("--dry-run set; skipping upload.")
        return

    upload(args.staging, args.repo, args.private, args.token)


if __name__ == "__main__":
    main()
