"""
Build a Hugging Face audiofolder dataset from the PitchBench benchmark.

For each experiment, this script:
  1. Calls build_conditions() with the same defaults the experiments use.
  2. Applies the deterministic stratified sampling
     (config.EXPERIMENT_DEFAULTS, seed=config.DEFAULT_SAMPLE_SEED).
  3. Resolves the audio path for every sampled condition by calling the same
     engine functions the experiments use (cached, so no audio is regenerated
     when files already exist on disk).
  4. Writes <out>/<exp_name>/metadata.jsonl and symlinks each WAV alongside.

The output is an HF audiofolder layout. Pass it to `huggingface_hub.upload_*`
or `datasets-cli` to publish.

Usage:
    python build_hf_dataset.py --out pitchbench_hf
    python build_hf_dataset.py --out pitchbench_hf --only a1 d1 d7
    python build_hf_dataset.py --out pitchbench_hf --copy   # copy instead of symlink
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.sampling import apply_default_sampling


# ── Experiments included in the benchmark dataset ────────────────────────────
# Skipped:
#   f1, f2, f3 — embedding/logit probes; outputs not user-facing audio
#   z1         — uses NSynth (external dataset, not redistributable here)

BENCH_EXPERIMENTS = [
    "pitchbench_a1_pitch_id",
    "pitchbench_a2_pitch_with_reference",
    "pitchbench_a3_pitch_by_duration",
    "pitchbench_a4_pitch_with_vibrato",
    "pitchbench_a5_pitch_slightly_off",
    "pitchbench_b1_pitch_in_silence",
    "pitchbench_b2_onset_offset_single",
    "pitchbench_b3_onset_offset_specific",
    "pitchbench_b4_pitch_at_time",
    "pitchbench_b5_onset_offset_each",
    "pitchbench_c1_dyad_interval",
    "pitchbench_c2_chord_pitch_count",
    "pitchbench_c3_chord_pitch_id",
    "pitchbench_c4_chord_quality",
    "pitchbench_d1_seq_pitch_count",
    "pitchbench_d2_pitch_difference",
    "pitchbench_d3_interval_id_seq",
    "pitchbench_d4_contour_discrete",
    "pitchbench_d5_contour_continuous",
    "pitchbench_d6_pitch_ranking",
    "pitchbench_d7_seq_pitch_id",
    "pitchbench_e1_loudness",
    "pitchbench_e2_audio_effects",
    "pitchbench_e3_background_effects",
    "pitchbench_e4_harmonic_saturation",
    "pitchbench_e5_time_stretch",
    "pitchbench_f1_melodic_line_id",
    "pitchbench_f2_chorale_voice_id",
]


# ── Per-experiment build-conditions args + audio-path resolvers ──────────────

def _build_args(exp: str, mod) -> tuple:
    """Args to pass to mod.build_conditions(), matching what each module's preview() uses."""
    if exp == "pitchbench_a1_pitch_id":
        return (mod.ALL_SOURCES,)
    if exp == "pitchbench_a2_pitch_with_reference":
        return ()
    if exp == "pitchbench_a3_pitch_by_duration":
        return ()
    if exp == "pitchbench_a4_pitch_with_vibrato":
        return (mod.DURATIONS_MS, mod.PITCHES, mod.SOURCES)
    if exp == "pitchbench_a5_pitch_slightly_off":
        return ([config.DEFAULT_DURATION_MS], config.DEFAULT_PITCHES, mod.SOURCES)
    if exp == "pitchbench_b1_pitch_in_silence":
        return (mod.SOURCES,)
    if exp in (
        "pitchbench_b2_onset_offset_single",
    ):
        return (config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, mod.SOURCES)
    if exp in (
        "pitchbench_b3_onset_offset_specific",
        "pitchbench_b4_pitch_at_time",
        "pitchbench_b5_onset_offset_each",
    ):
        return (config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES, mod.SOURCES, mod.DEFAULT_SEED)
    if exp == "pitchbench_c1_dyad_interval":
        return (config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES)
    if exp == "pitchbench_c2_chord_pitch_count":
        return (config.DEFAULT_DURATIONS_MS, mod.DEFAULT_SEED)
    if exp == "pitchbench_c3_chord_pitch_id":
        return (mod.SOURCES,)
    if exp == "pitchbench_c4_chord_quality":
        return (config.DEFAULT_DURATIONS_MS,)
    if exp == "pitchbench_d1_seq_pitch_count":
        return (config.DEFAULT_DURATIONS_MS, mod.DEFAULT_N_TRIALS, mod.DEFAULT_SEED)
    if exp == "pitchbench_d2_pitch_difference":
        return (config.DEFAULT_DURATIONS_MS, mod.SEPARATION_MS, mod.DEFAULT_N_TRIALS, mod.DEFAULT_SEED)
    if exp == "pitchbench_d3_interval_id_seq":
        return (config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES)
    if exp == "pitchbench_d4_contour_discrete":
        return (config.DEFAULT_DURATIONS_MS, config.DEFAULT_PITCHES)
    if exp == "pitchbench_d5_contour_continuous":
        return (mod.SOURCES,)
    if exp == "pitchbench_d6_pitch_ranking":
        return (config.DEFAULT_DURATIONS_MS, mod.DEFAULT_N_TRIALS, mod.DEFAULT_SEED)
    if exp == "pitchbench_d7_seq_pitch_id":
        seqs = mod.build_sequences(mod.N_NOTES_LIST, mod.DEFAULT_N_TRIALS, mod.DEFAULT_SEED)
        return (seqs, mod.SOURCES)
    if exp == "pitchbench_e1_loudness":
        return ()
    if exp == "pitchbench_e2_audio_effects":
        return ()
    if exp == "pitchbench_e3_background_effects":
        return (mod.DURATIONS_MS, config.DEFAULT_PITCHES, mod.SOURCES)
    if exp == "pitchbench_e4_harmonic_saturation":
        return ()
    if exp == "pitchbench_e5_time_stretch":
        return ()
    if exp == "pitchbench_f1_melodic_line_id":
        return (mod.N_TRIALS, mod.DEFAULT_SEED)
    if exp == "pitchbench_f2_chorale_voice_id":
        return (mod.DEFAULT_CHORALES, mod.DEFAULT_SEED)
    raise KeyError(f"No build args for {exp}")


def _audio_path(exp: str, mod, c: dict) -> Path:
    """Resolve the (cached) audio path for a single condition."""
    if exp == "pitchbench_a1_pitch_id":
        return engine.tone(c["midi"], c["source"], mod.TONE_DURATION_MS)
    if exp == "pitchbench_a2_pitch_with_reference":
        return mod._get_wav(c)
    if exp == "pitchbench_a3_pitch_by_duration":
        return engine.tone(c["midi"], c["source"], c["duration_ms"])
    if exp == "pitchbench_b1_pitch_in_silence":
        return mod._get_wav(c)
    if exp == "pitchbench_b2_onset_offset_single":
        path, _gt = mod._wav_for(c)  # b2 returns (path, (on_s, off_s))
        return path
    if exp == "pitchbench_c3_chord_pitch_id":
        return engine.chord(c["midi_notes"], c["source"], mod.TONE_DURATION_MS)
    if exp == "pitchbench_d5_contour_continuous":
        return mod._get_wav(c)
    if exp == "pitchbench_d7_seq_pitch_id":
        return engine.sequence(c["midi_sequence"], c["source"], mod.TONE_MS, mod.GAP_MS)
    if exp == "pitchbench_e1_loudness":
        return engine.tone_at_volume(c["midi"], c["source"], mod.TONE_MS, c["loudness_db"])
    if exp == "pitchbench_e2_audio_effects":
        return engine.tone_with_effect(
            c["midi"], c["source"], mod.TONE_MS,
            c["effect"], mod.EFFECTS[c["effect"]], c["noise_seed"],
        )
    if exp == "pitchbench_e4_harmonic_saturation":
        return engine.tone_with_effect(
            c["midi"], c["source"], mod.TONE_MS,
            c["saturation_level"], mod.SATURATIONS[c["saturation_level"]], c["noise_seed"],
        )
    if exp == "pitchbench_e5_time_stretch":
        return engine.tone_time_modified(
            c["original_midi"], c["source"], mod.TONE_MS,
            c["mode"], c["factor"],
        )
    if exp == "pitchbench_f1_melodic_line_id":
        return engine.polyphonic_mix(
            list(zip(c["all_notes"], c["sources"])),
            c["total_ms"],
            name_hint=mod._cond_hint(c),
        )
    if exp == "pitchbench_f2_chorale_voice_id":
        hint = f"{c['chorale_slug']}_x{c['x']}_{c['inst_cfg']}_{c['source_label']}"
        return engine.polyphonic_mix(
            list(zip(c["all_notes"], c["sources"])),
            c["total_ms"],
            name_hint=hint,
        )
    # Everything else uses a module-level _wav_for(c)
    return mod._wav_for(c)


# ── Prompt resolution (the "question") ───────────────────────────────────────

def _prompts(exp: str, mod, c: dict) -> dict:
    """Return {prompt_field_name: prompt_string} for one condition.

    Most experiments have either a single PROMPT or four format-specific
    prompts (MIDI / ABC|SPN / Doremi / Hz). A few build their prompt
    dynamically from the condition (e.g. b3 asks about a specific target note;
    b4 asks about a specific timestamp; d6/d7 depend on the sequence length).
    """
    # Single canonical prompt — most experiments
    if hasattr(mod, "PROMPT") and isinstance(mod.PROMPT, str):
        return {"prompt": mod.PROMPT}

    # Four-format pitch-id prompts (a1, a3, e1, e2): MIDI/ABC/Doremi/Hz
    if all(hasattr(mod, f"PROMPT_{k}_FULL") for k in ("MIDI", "ABC", "DOREMI", "HZ")):
        return {
            "prompt_midi":   mod.PROMPT_MIDI_FULL,
            "prompt_abc":    mod.PROMPT_ABC_FULL,
            "prompt_doremi": mod.PROMPT_DOREMI_FULL,
            "prompt_hz":     mod.PROMPT_HZ_FULL,
        }

    # Four-format with SPN instead of ABC (a4, a5, e3, e4, e5)
    if all(hasattr(mod, f"PROMPT_{k}_FULL") for k in ("MIDI", "SPN", "DOREMI", "HZ")):
        return {
            "prompt_midi":   mod.PROMPT_MIDI_FULL,
            "prompt_spn":    mod.PROMPT_SPN_FULL,
            "prompt_doremi": mod.PROMPT_DOREMI_FULL,
            "prompt_hz":     mod.PROMPT_HZ_FULL,
        }

    # Per-experiment dynamic builders
    if exp == "pitchbench_a2_pitch_with_reference":
        return {
            "prompt_midi":   mod._make_prompt("midi",   c["ref_midi"], c["condition"]),
            "prompt_abc":    mod._make_prompt("abc",    c["ref_midi"], c["condition"]),
            "prompt_doremi": mod._make_prompt("doremi", c["ref_midi"], c["condition"]),
            "prompt_hz":     mod._make_prompt("hz",     c["ref_midi"], c["condition"]),
        }
    if exp == "pitchbench_b1_pitch_in_silence":
        return {
            "prompt_midi":   mod._make_prompt("midi",   c["pos_ms"], mod.TONE_DURATION_MS, mod.TOTAL_SILENCE_MS, c["condition"]),
            "prompt_abc":    mod._make_prompt("abc",    c["pos_ms"], mod.TONE_DURATION_MS, mod.TOTAL_SILENCE_MS, c["condition"]),
            "prompt_doremi": mod._make_prompt("doremi", c["pos_ms"], mod.TONE_DURATION_MS, mod.TOTAL_SILENCE_MS, c["condition"]),
            "prompt_hz":     mod._make_prompt("hz",     c["pos_ms"], mod.TONE_DURATION_MS, mod.TOTAL_SILENCE_MS, c["condition"]),
        }
    if exp == "pitchbench_b3_onset_offset_specific":
        from pitchbench.experiments.helpers.music import midi_to_note
        target = f"{midi_to_note(c['midi'])} (MIDI {c['midi']})"
        return {"prompt": mod._prompt_for(target)}
    if exp == "pitchbench_b4_pitch_at_time":
        # Four prompts at the target time.
        query_time_s = c["query_time_s"]
        pm, ps, pd, ph = mod._prompt_set(query_time_s)
        return {"prompt_midi": pm, "prompt_spn": ps, "prompt_doremi": pd, "prompt_hz": ph}
    if exp == "pitchbench_c3_chord_pitch_id":
        return {
            "prompt_midi":   mod.PROMPT_MIDI,
            "prompt_abc":    mod.PROMPT_ABC,
            "prompt_doremi": mod.PROMPT_DOREMI,
        }
    if exp == "pitchbench_c4_chord_quality":
        return {
            "prompt_quality_only":     mod.PROMPT_QUALITY_ONLY,
            "prompt_root_and_quality": mod.PROMPT_ROOT_AND_QUALITY,
        }
    if exp == "pitchbench_d6_pitch_ranking":
        return {"prompt": mod._prompt(c["n_notes"])}
    if exp == "pitchbench_d7_seq_pitch_id":
        n = c["n_notes"]
        return {
            "prompt_midi":   mod.make_prompt_midi(n),
            "prompt_abc":    mod.make_prompt_abc(n),
            "prompt_doremi": mod.make_prompt_doremi(n),
            "prompt_hz":     mod.make_prompt_hz(n),
        }
    if exp == "pitchbench_f1_melodic_line_id":
        n, x, cfg, srcs = c["n"], c["x"], c["inst_cfg"], c["sources"]
        return {
            "prompt_midi":   mod.make_prompt_midi(n, x, cfg, srcs),
            "prompt_spn":    mod.make_prompt_spn(n, x, cfg, srcs),
            "prompt_doremi": mod.make_prompt_doremi(n, x, cfg, srcs),
        }
    if exp == "pitchbench_f2_chorale_voice_id":
        x, n_target, cfg, srcs = c["x"], c["n_target"], c["inst_cfg"], c["sources"]
        return {
            "prompt_midi":   mod.make_prompt_midi(x, n_target, cfg, srcs),
            "prompt_spn":    mod.make_prompt_spn(x, n_target, cfg, srcs),
            "prompt_doremi": mod.make_prompt_doremi(x, n_target, cfg, srcs),
        }
    return {}


# ── Metadata serialisation ────────────────────────────────────────────────────

def _jsonable(v):
    if isinstance(v, Path):
        return v.name
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return str(v)  # last-ditch fallback for numpy scalars etc.


def _row_for(exp: str, mod, c: dict, audio_path: Path) -> dict:
    row = {"file_name": audio_path.name}
    # Ground-truth + stimulus parameters from the condition dict
    for k, v in c.items():
        if k.startswith("_"):
            continue
        row[k] = _jsonable(v)
    # Question(s)
    for k, v in _prompts(exp, mod, c).items():
        row[k] = v
    return row


# ── Main per-experiment build ────────────────────────────────────────────────

def build_one(exp: str, out_root: Path, copy: bool = False) -> tuple[int, int]:
    """Stage one experiment's audio + metadata. Returns (n_rows, n_missing)."""
    print(f"\n=== {exp} ===")
    mod = importlib.import_module(f"pitchbench.experiments.scripts.{exp}")
    engine.set_exp(exp)

    args = _build_args(exp, mod)
    all_conds = mod.build_conditions(*args)
    sampled, meta = apply_default_sampling(
        exp, all_conds, cli_sample_n=None, cli_seed=config.DEFAULT_SAMPLE_SEED,
    )
    print(
        f"  conditions: {len(all_conds):>5} total → "
        f"{len(sampled):>4} sampled "
        f"(stratified by {meta.get('stratified_by')!r}, seed={meta.get('sample_seed')})"
    )

    out_dir = out_root / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing = 0
    for c in sampled:
        try:
            ap = Path(_audio_path(exp, mod, c))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            missing += 1
            print(f"    [SKIP] {exc}")
            continue
        if not ap.exists():
            missing += 1
            print(f"    [MISSING WAV] {ap}")
            continue

        # Stage WAV (symlink by default; copy if requested).
        target = out_dir / ap.name
        if not target.exists():
            if copy:
                shutil.copy2(ap, target)
            else:
                try:
                    os.symlink(os.path.realpath(ap), target)
                except OSError:
                    shutil.copy2(ap, target)
        rows.append(_row_for(exp, mod, c, ap))

    # Write metadata.jsonl
    meta_path = out_dir / "metadata.jsonl"
    with meta_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  wrote {len(rows)} rows → {meta_path.relative_to(out_root.parent) if out_root.parent in meta_path.parents else meta_path}")
    if missing:
        print(f"  {missing} conditions had no audio (skipped)")
    return len(rows), missing


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("pitchbench_hf"),
                   help="Output staging directory (default: pitchbench_hf/)")
    p.add_argument("--only", nargs="+", metavar="ID",
                   help="Only build these experiment IDs (e.g. a1 d1 d7)")
    p.add_argument("--copy", action="store_true",
                   help="Copy WAVs instead of symlinking (slower, 18 GB).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    target = BENCH_EXPERIMENTS
    if args.only:
        wanted = {x.lower() for x in args.only}
        target = [
            e for e in BENCH_EXPERIMENTS
            if any(e.startswith(f"pitchbench_{w}_") for w in wanted)
        ]
        if not target:
            print(f"No experiments matched {args.only}")
            sys.exit(1)

    totals = {"rows": 0, "missing": 0, "ok": 0, "fail": 0}
    for exp in target:
        try:
            n_rows, n_miss = build_one(exp, args.out, copy=args.copy)
            totals["rows"] += n_rows
            totals["missing"] += n_miss
            totals["ok"] += 1
        except Exception as exc:
            totals["fail"] += 1
            print(f"  [FAIL] {exp}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Built {totals['ok']}/{len(target)} experiments")
    print(f"Total rows: {totals['rows']:,}")
    if totals["missing"]:
        print(f"Missing audio: {totals['missing']}")
    if totals["fail"]:
        print(f"Failed: {totals['fail']}")
    print(f"Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
