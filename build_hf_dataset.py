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
from pitchbench.experiments.helpers.music import (
    midi_to_freq, midi_to_note, midi_to_solfege,
)
from pitchbench.experiments.helpers.sampling import apply_default_sampling


# ── Experiments included in the benchmark dataset ────────────────────────────
# Skipped:
#   g1, g2, g3 — embedding/logit probes (analysis track); outputs not user-facing audio
#   z1         — uses NSynth (external dataset, not redistributable here)

BENCH_EXPERIMENTS = [
    "pitchbench_a1_single_pitch_id",
    "pitchbench_d7a_pitch_with_reference",
    "pitchbench_d7b_pitch_with_reference_split",
    "pitchbench_a3_single_pitch_by_duration",
    "pitchbench_e5_vibrato",
    "pitchbench_e6_slightly_off",
    "pitchbench_b1_single_pitch_within_silence",
    "pitchbench_b3_timestamp_single_pitch",
    "pitchbench_b4_timestamp_specific_pitch",
    "pitchbench_b2_pitch_at_timestamp",
    "pitchbench_b5_timestamp_multiple_pitches",
    "pitchbench_c2_chord_dyad_interval",
    "pitchbench_c1_chord_count_pitches",
    "pitchbench_c4_chord_pitches",
    "pitchbench_c3_chord_quality",
    "pitchbench_d1_sequence_count_pitches",
    "pitchbench_d2_dyad_lower_higher_difference",
    "pitchbench_d6_sequence_dyad_interval",
    "pitchbench_d3_contour_discrete",
    "pitchbench_d4_contour_continuous",
    "pitchbench_d5_sequence_ranking_by_pitch",
    "pitchbench_d8_sequence_pitches",
    "pitchbench_a2_single_pitch_by_loudness",
    "pitchbench_e1_audio_effects",
    "pitchbench_e2_background",
    "pitchbench_e3_harmonic_saturation",
    "pitchbench_e4_time_stretching",
    "pitchbench_f1_melodic_line_atonal",
    "pitchbench_f2_melodic_line_tonal",
]


# ── Per-experiment build-conditions args + audio-path resolvers ──────────────

def _build_args(exp: str, mod) -> tuple:
    """Every benchmark ``build_conditions()`` now takes zero positional args."""
    return ()


def _audio_path(exp: str, mod, c: dict) -> Path:
    """Resolve the (cached) audio path for a single condition.

    Every benchmark experiment exposes a public ``wav_for(c)`` that returns a
    Path (or a (Path, gt) tuple in the b-series onset/offset experiments).
    Two-clip experiments (``wavs_for``) are handled by ``_audio_paths``.
    """
    out = mod.wav_for(c)
    # b2/b3/b5 wav_for returns (path, gt_timestamps) — keep just the path.
    if isinstance(out, tuple):
        out = out[0]
    return Path(out)


def _audio_paths(exp: str, mod, c: dict) -> tuple[Path, Path | None]:
    """Return (primary_wav, ref_wav_or_None).

    For two-clip experiments (wavs_for), returns (target, reference).
    For single-clip experiments (wav_for), returns (path, None).
    """
    if hasattr(mod, "wavs_for"):
        ref, tgt = mod.wavs_for(c)
        return Path(tgt), Path(ref)
    return _audio_path(exp, mod, c), None


# ── Prompt resolution (the "question") ───────────────────────────────────────

def _prompts(exp: str, mod, c: dict) -> dict:
    """Return {prompt_field_name: prompt_string} for one condition.

    Resolution order:
      1. ``mod.prompts_for(c)`` — canonical hook. Returns a dict keyed by
         short format name (``midi``, ``spn``, ``doremi``, ``hz``, ``main``,
         ``quality_only``, ``root_and_quality``, …). We prefix each key with
         ``prompt_`` (and rename ``main`` → ``prompt``).
      2. ``mod.prompt_for(c)`` — single-prompt hook (b3-style).
      3. ``mod.PROMPT_PREFIX`` (or ``mod.SPEC.prompt_prefix``) — cat_a / cat_e
         style: glue the prefix onto the four canonical PROMPT_* strings.
    """
    if hasattr(mod, "prompts_for"):
        raw = mod.prompts_for(c)
        if isinstance(raw, str):
            return {"prompt": raw}
        out = {}
        for k, v in raw.items():
            out["prompt" if k == "main" else f"prompt_{k}"] = v
        return out

    if hasattr(mod, "prompt_for"):
        return {"prompt": mod.prompt_for(c)}

    if hasattr(mod, "PROMPT") and isinstance(mod.PROMPT, str):
        return {"prompt": mod.PROMPT}

    pref = getattr(mod, "PROMPT_PREFIX", None)
    if not isinstance(pref, str):
        spec = getattr(mod, "SPEC", None)
        pref = getattr(spec, "prompt_prefix", None) if spec is not None else None
    if isinstance(pref, str):
        from pitchbench.experiments.helpers.music import (
            PROMPT_DOREMI, PROMPT_HZ, PROMPT_MIDI, PROMPT_SPN,
        )
        return {
            "prompt_midi":   pref + PROMPT_MIDI,
            "prompt_spn":    pref + PROMPT_SPN,
            "prompt_doremi": pref + PROMPT_DOREMI,
            "prompt_hz":     pref + PROMPT_HZ,
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


# Prompt-key normalisation so the four formats line up across experiments.
# Some scripts call the note-letter format "abc", others "spn"; some call the
# solfège format "doremi"; some call frequency "hz". On HF we expose them as a
# single, consistent set: midi / abc / solfege / freq.
_PROMPT_RENAME = {
    "prompt_spn":    "prompt_abc",
    "prompt_doremi": "prompt_solfege",
    "prompt_hz":     "prompt_freq",
}


def _gt_for_midi(midi: int) -> dict:
    """Four-format ground truth for a single MIDI pitch."""
    return {
        "gt_midi":    int(midi),
        "gt_abc":     midi_to_note(midi),
        "gt_solfege": midi_to_solfege(midi),
        "gt_freq":    round(midi_to_freq(midi), 2),
    }


def _gt_fields(exp: str, c: dict) -> dict:
    """Build the gt_* block for one condition.

    Experiment-specific cases are handled first (early return).
    The general fallback covers A-series, B1, B2, C3, C4, D7a/b, D8, E-series.
    """
    gt: dict = {}

    # B3: onset position of a hidden tone (answer = when, not what)
    if exp == "pitchbench_b3_timestamp_single_pitch":
        if "pos_ms" in c:
            gt["gt_pos_ms"] = c["pos_ms"]
        return gt  # midi kept in "others" via _EXP_GT_SOURCE_UNHIDE

    # B4: onset/offset of a named target note among distractors
    if exp == "pitchbench_b4_timestamp_specific_pitch":
        if "target_onset_ms" in c:
            gt["gt_target_onset_ms"] = c["target_onset_ms"]
        if "target_offset_ms" in c:
            gt["gt_target_offset_ms"] = c["target_offset_ms"]
        return gt  # midi kept in "others" via _EXP_GT_SOURCE_UNHIDE

    # B5: all note onsets/offsets
    if exp == "pitchbench_b5_timestamp_multiple_pitches":
        if "onsets_ms" in c:
            gt["gt_onsets_ms"] = c["onsets_ms"]
        return gt

    # C1: count simultaneous pitches
    if exp == "pitchbench_c1_chord_count_pitches":
        if "n" in c:
            gt["gt_n"] = c["n"]
        return gt

    # C2: dyad interval identification
    if exp == "pitchbench_c2_chord_dyad_interval":
        if "interval_st" in c:
            gt["gt_interval_st"] = c["interval_st"]
        return gt

    # D1: sequential pitch count
    if exp == "pitchbench_d1_sequence_count_pitches":
        if "n" in c:
            gt["gt_n"] = c["n"]
        return gt

    # D2: binary higher/lower judgment
    if exp == "pitchbench_d2_dyad_lower_higher_difference":
        if "answer_gt" in c:
            gt["gt_answer"] = c["answer_gt"]
        return gt

    # D3: discrete melodic contour
    if exp == "pitchbench_d3_contour_discrete":
        if "pattern" in c:
            gt["gt_pattern"] = c["pattern"]
        return gt

    # D4: continuous pitch trajectory (condition already stores gt_seq)
    if exp == "pitchbench_d4_contour_continuous":
        if "gt_seq" in c:
            gt["gt_seq"] = c["gt_seq"]
        return gt

    # D5: pitch ranking
    if exp == "pitchbench_d5_sequence_ranking_by_pitch":
        if "answer_gt" in c:
            gt["gt_answer"] = c["answer_gt"]
        return gt

    # D6: sequential dyad interval (signed semitones)
    if exp == "pitchbench_d6_sequence_dyad_interval":
        if "signed_st" in c:
            gt["gt_signed_st"] = c["signed_st"]
        return gt

    # F1/F2: target voice pitch sequence, expanded to all four formats
    if exp in ("pitchbench_f1_melodic_line_atonal", "pitchbench_f2_melodic_line_tonal"):
        tp = c.get("target_pitches")
        if isinstance(tp, (list, tuple)) and tp:
            gt["gt_seq_midi"]    = [int(m)                    for m in tp]
            gt["gt_seq_abc"]     = [midi_to_note(m)           for m in tp]
            gt["gt_seq_solfege"] = [midi_to_solfege(m)        for m in tp]
            gt["gt_seq_freq"]    = [round(midi_to_freq(m), 2) for m in tp]
        return gt

    # ── General fallback (A-series, B1, B2, C3, C4, D7a/b, D8, E-series) ──

    # Single-pitch GT (perceived pitch, after any manipulation)
    midi = c.get("midi")
    if isinstance(midi, int):
        gt.update(_gt_for_midi(midi))

    # Chord — list of MIDI notes
    chord = c.get("midi_notes")
    if isinstance(chord, (list, tuple)) and chord and all(isinstance(m, int) for m in chord):
        notes = sorted(chord)
        gt["gt_midi_notes"]    = notes
        gt["gt_abc_notes"]     = [midi_to_note(m)           for m in notes]
        gt["gt_solfege_notes"] = [midi_to_solfege(m)        for m in notes]
        gt["gt_freq_notes"]    = [round(midi_to_freq(m), 2) for m in notes]

    # Sequence — ordered list of MIDI notes
    seq = c.get("midi_sequence")
    if isinstance(seq, (list, tuple)) and seq and all(isinstance(m, int) for m in seq):
        gt["gt_midi_sequence"]    = list(seq)
        gt["gt_abc_sequence"]     = [midi_to_note(m)           for m in seq]
        gt["gt_solfege_sequence"] = [midi_to_solfege(m)        for m in seq]
        gt["gt_freq_sequence"]    = [round(midi_to_freq(m), 2) for m in seq]

    # Chord-quality (c3)
    if "chord_quality_gt" in c:
        gt["gt_quality"] = c["chord_quality_gt"]
    elif "quality" in c:
        gt["gt_quality"] = c["quality"]

    # Onset/offset in seconds (b2)
    if "onset_s" in c:
        gt["gt_onset_s"] = c["onset_s"]
    if "offset_s" in c:
        gt["gt_offset_s"] = c["offset_s"]

    # Pitch-at-time (b2 query timestamp)
    if "query_time_s" in c:
        gt["gt_query_time_s"] = c["query_time_s"]

    # Interval-only fallback (not used by c2/d6 which have early returns above)
    if "interval" in c and "ref_midi" not in c:
        gt["gt_interval"] = c["interval"]

    return gt


# Keys that feed gt_* in the general fallback — suppress from "others".
_GT_SOURCE_KEYS = {
    "midi", "midi_notes", "midi_sequence",
    "quality", "chord_quality_gt",
    "onset_s", "offset_s", "query_time_s",
}

# Per-experiment source keys that feed gt_* — suppress from "others" for that exp only.
_EXP_GT_SOURCE_KEYS: dict[str, set] = {
    "pitchbench_b3_timestamp_single_pitch":       {"pos_ms"},
    "pitchbench_b4_timestamp_specific_pitch":     {"target_onset_ms", "target_offset_ms"},
    "pitchbench_b5_timestamp_multiple_pitches":   {"onsets_ms"},
    "pitchbench_c1_chord_count_pitches":          {"n"},
    "pitchbench_c2_chord_dyad_interval":          {"interval_st"},
    "pitchbench_d1_sequence_count_pitches":       {"n"},
    "pitchbench_d2_dyad_lower_higher_difference": {"answer_gt"},
    "pitchbench_d3_contour_discrete":             {"pattern"},
    "pitchbench_d4_contour_continuous":           {"gt_seq"},
    "pitchbench_d5_sequence_ranking_by_pitch":    {"answer_gt"},
    "pitchbench_d6_sequence_dyad_interval":       {"signed_st"},
    "pitchbench_f1_melodic_line_atonal":          {"target_pitches"},
    "pitchbench_f2_melodic_line_tonal":           {"target_pitches"},
}

# Per-experiment keys to UN-suppress from _GT_SOURCE_KEYS so they appear in "others".
# B3/B4: midi is the stimulus note (not the answer), so it belongs in "others".
_EXP_GT_SOURCE_UNHIDE: dict[str, set] = {
    "pitchbench_b3_timestamp_single_pitch":   {"midi"},
    "pitchbench_b4_timestamp_specific_pitch": {"midi"},
}

# Keys derivable from `source` (or otherwise redundant) — drop from HF rows.
_DROP_KEYS = {"source_type"}


def _safe_filename(name: str) -> str:
    """Sanitize a filename for HF: replace characters that break URL parsers."""
    return name.replace("~", "-")


def _row_for(exp: str, mod, c: dict, audio_path: Path) -> dict:
    """Emit a row in the canonical HF schema:

        audio | gt_* | prompt_* | source | {others}
    """
    row: dict = {"file_name": _safe_filename(audio_path.name)}

    # Ground truth
    gt = _gt_fields(exp, c)
    row.update(gt)

    # Prompts (renamed to the unified midi/abc/solfege/freq scheme)
    for k, v in _prompts(exp, mod, c).items():
        row[_PROMPT_RENAME.get(k, k)] = v

    # Source first among the remaining columns
    if "source" in c:
        row["source"] = c["source"]

    # Everything else, in declaration order, minus what we've already shown.
    skip = (
        set(row.keys())
        | _GT_SOURCE_KEYS
        | _DROP_KEYS
        | _EXP_GT_SOURCE_KEYS.get(exp, set())
    ) - _EXP_GT_SOURCE_UNHIDE.get(exp, set())
    for k, v in c.items():
        if k.startswith("_") or k in skip:
            continue
        row[k] = _jsonable(v)

    return row


# ── HF presentation order: real instruments before raw waveforms ─────────────
_INSTR_FIRST = list(config.GM_PROGRAMS_V1.keys()) + list(config.WAVEFORMS)
_SOURCE_RANK = {s: i for i, s in enumerate(_INSTR_FIRST)}


def _source_sort_key(row: dict) -> int:
    head = str(row.get("source", "")).split("+", 1)[0]
    return _SOURCE_RANK.get(head, len(_INSTR_FIRST))


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

    def _stage(src: Path, dst_dir: Path) -> Path:
        dst = dst_dir / _safe_filename(src.name)
        if not dst.exists():
            if copy:
                shutil.copy2(src, dst)
            else:
                try:
                    os.symlink(os.path.realpath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)
        return dst

    rows: list[dict] = []
    missing = 0
    for c in sampled:
        try:
            ap, ref_ap = _audio_paths(exp, mod, c)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            missing += 1
            print(f"    [SKIP] {exc}")
            continue
        if not ap.exists():
            missing += 1
            print(f"    [MISSING WAV] {ap}")
            continue
        if ref_ap is not None and not ref_ap.exists():
            missing += 1
            print(f"    [MISSING REF WAV] {ref_ap}")
            continue

        _stage(ap, out_dir)
        if ref_ap is not None:
            _stage(ref_ap, out_dir)

        row = _row_for(exp, mod, c, ap)
        if ref_ap is not None:
            row["file_name_ref"] = ref_ap.name
        rows.append(row)

    rows.sort(key=_source_sort_key)  # real instruments first, waveforms last

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
