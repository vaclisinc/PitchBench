"""
Experiment 14 — Token logits
Inspects the probability distribution over pitch-relevant tokens at generation
step 1.  Tests whether the model puts mass near the correct pitch even when it
outputs the wrong token.

Sources: piano, violin, flute (engine-rendered via FluidSynth).
Pitches:  F1 (29), G2 (43), A3 (57), B4 (71), C5 (72).

For each (source, MIDI, prompt_variant) triple the experiment:
  • Queries the model via /generate_with_probs to get the top-K token probs
  • Maps step-1 tokens to MIDI values (or pitch classes) depending on variant:
      - MIDI:    integer tokens 0–127
      - ABC:     note-name tokens e.g. "C4", "F#3" → MIDI distance
      - Solfège: syllable and accidental (if needed) tokens "do"/"re"/… → pitch-class distance (enharmonics merged)
  • Computes: P(exact), P(within±1), P(within±2), P(within±6), P(within±12),
              entropy over pitch tokens, and the argmax ("predicted") pitch

Analyses:
  1. Probability-mass-vs-distance: how does mass fall off with semitone distance?
  2. Entropy heatmap: uncertainty by source and MIDI pitch
  3. Cross-model comparison of P(exact) and entropy
  4. Does Solfège shift mass toward the correct pitch relative to ABC?
  5. Per-condition distributions: full token-probability line plots (MIDI 0–127,
     ABC mapped to MIDI, Solfège pitch classes 0–11) with true-value marker.

Usage:
    python experiments/run.py exp_14_token_logits
    python experiments/run.py exp_14_token_logits --preview
    python experiments/run.py exp_14_token_logits --models audio_flamingo_next_instruct
    python experiments/run.py exp_14_token_logits --top-k 200
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

import pitchbench.config as config
import pitchbench.generation.engine as engine
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    FLAT_TO_SHARP, NOTE_NAMES, SOLFEGE_TO_PC,
    extract_midi, extract_note, extract_solfege,
    midi_to_note, midi_to_solfege, note_to_midi,
    semitone_distance, solfege_pc_distance,
)
from pitchbench.experiments.helpers.plots import save_accuracy_plots
from pitchbench.experiments.helpers.results import exp_data_dir, get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies

EXP_NAME = Path(__file__).stem

SELECTED_SOURCES: list[str] = config.ALL_SOURCES
# F1=29, G2=43, A3=57, B4=71, C5=72
SELECTED_MIDI_PITCHES: list[int] = [29, 43, 57, 71, 72]
# SELECTED_MIDI_PITCHES: list[int] = [29]
TONE_DURATION_MS = 2000

DEFAULT_TOP_K   = 200
MAX_NEW_TOKENS  = 32


PROMPT_MIDI_EXP14 = (
    "What is the MIDI note number (an integer)? The note number of the played note is between 10 and 99."
    "Reply with ONLY the integer. Nothing else. Output only the answer."
)

PROMPT_ABC_EXP14 = (
    "What is the note name and octave? "
    "Reply with ONLY the note name, for example: C4, F#3, Bb5. Nothing else. Output only the answer."
)

PROMPT_DOREMI_EXP14 = (
    "What is the solfege syllable and accidental (if needed) of this pitch? "
    "Use fixed-do (do=C, re=D, mi=E, fa=F, sol=G, la=A, si=B). "
    "Reply with the syllable and accidental (if needed) and accidental (if necessary). Nothing else. Output only the answer."
)

PROMPTS: dict[str, str] = {
    "midi":   PROMPT_MIDI_EXP14,
    "abc":    PROMPT_ABC_EXP14,
    "doremi": PROMPT_DOREMI_EXP14,
}

try:
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False




# ── Stimuli catalogue ─────────────────────────────────────────────────────────

def build_conditions() -> list[dict]:
    rows: list[dict] = []
    for src in SELECTED_SOURCES:
        for midi in SELECTED_MIDI_PITCHES:
            try:
                wav = engine.tone(midi, src, TONE_DURATION_MS)
            except ValueError as exc:
                print(f"  [SKIP] {src} MIDI {midi}: {exc}")
                continue
            rows.append({
                "source":      src,
                "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                "midi":        midi,
                "note":        midi_to_note(midi),
                "doremi_pc":   midi % 12,
                "wav":         str(wav),
            })
    return rows


# ── Token analysis helpers ────────────────────────────────────────────────────

_MIDI_RE     = re.compile(r"^\d{1,3}$")
_NOTE_RE     = re.compile(r"^([A-Ga-g][#b♯♭]?\d?)$")
_SOLFEGE_RE  = re.compile(
    r"^(do|re|mi|fa|sol|la|si|ti)$", re.IGNORECASE
)


def _token_to_midi(token: str) -> int | None:
    """Map a raw token string to a MIDI number, or None if not a pitch token."""
    tok = token.strip()
    if _MIDI_RE.match(tok):
        v = int(tok)
        return v if 0 <= v <= 127 else None
    return None


def _token_to_midi_abc(token: str) -> int | None:
    """Map a note-name token to MIDI (with octave) or pitch class (×12) if no octave."""
    tok = token.strip()
    if not _NOTE_RE.match(tok):
        return None
    # Try to parse as full note (e.g. C4, F#3)
    tok_norm = tok[:-1].upper().replace("♯", "#").replace("♭", "b")
    tok_norm = FLAT_TO_SHARP.get(tok_norm, tok_norm)
    if tok_norm in NOTE_NAMES:
        # Has octave digit?
        if tok[-1].isdigit():
            midi = note_to_midi(tok_norm + tok[-1])
            return midi
        # No octave — return pitch class as sentinel (pc + 1000)
        return NOTE_NAMES.index(tok_norm) + 1000
    return None


def _token_to_pc_solfege(token: str) -> int | None:
    """Map a solfège syllable and accidental (if needed) token to pitch class (0–11), or None."""
    tok = token.strip().lower()
    return SOLFEGE_TO_PC.get(tok)


def _build_midi_dist_multistep(all_steps: list[list[dict[str, Any]]]) -> dict[int, float]:
    """Aggregate MIDI number probabilities across consecutive generation steps.

    The model server returns per-step top-K distributions along the greedy
    decoding path.  A MIDI value n can be tokenized as one token ("30") or as
    multiple digit tokens ("3", "0").  This function attributes probability to
    each integer 0-127 by following the greedy prefix and branching at each step:

        P(n) += P(greedy_0) × … × P(greedy_{t-1}) × P(token_t)
                where greedy_0…token_t concatenates to str(n)

    If the model generates a sentence before the number (e.g. "The note is 69"),
    we locate the last run of decimal-digit greedy tokens and start aggregation
    there, using the conditional distribution at that position.
    """
    if not all_steps:
        return {}

    # Find where the last contiguous decimal-digit run starts in the greedy path
    greedy_toks = [
        max(s, key=lambda t: t["prob"])["token"].strip()
        for s in all_steps if s
    ]
    start_step = 0
    for i in range(len(greedy_toks) - 1, -1, -1):
        if greedy_toks[i].isdecimal():
            start_step = i
            while start_step > 0 and greedy_toks[start_step - 1].isdecimal():
                start_step -= 1
            break

    dist: dict[int, float] = {}
    greedy_prefix_digits = ""
    greedy_prefix_prob = 1.0

    for step_tokens in all_steps[start_step:]:
        if not step_tokens:
            break
        for t in step_tokens:
            tok = t["token"].strip()
            prob = float(t["prob"])
            candidate = greedy_prefix_digits + tok
            if candidate.isdecimal():
                val = int(candidate)
                if 0 <= val <= 127:
                    dist[val] = dist.get(val, 0.0) + greedy_prefix_prob * prob

        greedy = max(step_tokens, key=lambda t: t["prob"])
        greedy_tok = greedy["token"].strip()
        if not greedy_tok.isdecimal():
            break
        greedy_prefix_digits += greedy_tok
        greedy_prefix_prob *= float(greedy["prob"])

    return dist


def _semitone_dist_midi(gt_midi: int, pred_midi: int) -> int:
    return abs(gt_midi - pred_midi)


def _semitone_dist_abc(gt_midi: int, pred_midi_or_pc: int) -> int:
    """Distance between gt MIDI and predicted MIDI (or pitch-class if pred ≥ 1000)."""
    if pred_midi_or_pc >= 1000:
        pred_pc = pred_midi_or_pc - 1000
        gt_pc   = gt_midi % 12
        d = abs(gt_pc - pred_pc)
        return min(d, 12 - d)
    return abs(gt_midi - pred_midi_or_pc)


def _semitone_dist_solfege(gt_midi: int, pred_pc: int) -> int:
    return solfege_pc_distance(gt_midi, pred_pc)


def _pitch_mass_metrics(
    all_steps: list[list[dict[str, Any]]],
    gt_midi: int,
    variant: str,
) -> dict[str, Any]:
    """
    Compute probability mass metrics from the model's per-step token distributions.

    For the MIDI variant, probabilities are aggregated across consecutive steps
    so that multi-token representations (e.g. "3"+"0" for 30) are combined with
    single-token ones (e.g. "30").  ABC and Solfège use only step 1.

    Returns:
      {n_pitch_tokens, total_pitch_mass, p_exact, p_within_1, p_within_2,
       p_within_6, p_within_12, entropy_pitch, argmax_dist, argmax_token}
    """
    step1_tokens = all_steps[0] if all_steps else []
    pitch_tokens: list[tuple[str, float, int]] = []   # (token_text, prob, semitone_dist)

    if variant == "midi":
        midi_dist = _build_midi_dist_multistep(all_steps)
        for midi_val, prob in midi_dist.items():
            d = _semitone_dist_midi(gt_midi, midi_val)
            pitch_tokens.append((str(midi_val), prob, d))

    else:
        for t in step1_tokens:
            tok_text = t["token"]
            prob     = float(t["prob"])

            if variant == "abc":
                pred = _token_to_midi_abc(tok_text)
                if pred is not None:
                    dist = _semitone_dist_abc(gt_midi, pred)
                    pitch_tokens.append((tok_text, prob, dist))

            else:  # doremi
                pred_pc = _token_to_pc_solfege(tok_text)
                if pred_pc is not None:
                    dist = _semitone_dist_solfege(gt_midi, pred_pc)
                    pitch_tokens.append((tok_text, prob, dist))

    if not pitch_tokens:
        return {
            "n_pitch_tokens":   0,
            "total_pitch_mass": 0.0,
            "p_exact":          0.0,
            "p_within_1":       0.0,
            "p_within_2":       0.0,
            "p_within_6":       0.0,
            "p_within_12":      0.0,
            "entropy_pitch":    None,
            "argmax_dist":      None,
            "argmax_token":     None,
        }

    total = sum(p for _, p, _ in pitch_tokens)
    p_exact    = sum(p for _, p, d in pitch_tokens if d == 0) / total if total else 0
    p_within_1 = sum(p for _, p, d in pitch_tokens if d <= 1)  / total if total else 0
    p_within_2 = sum(p for _, p, d in pitch_tokens if d <= 2)  / total if total else 0
    p_within_6 = sum(p for _, p, d in pitch_tokens if d <= 6)  / total if total else 0
    p_within_12= sum(p for _, p, d in pitch_tokens if d <= 12) / total if total else 0

    # Entropy over pitch tokens (normalized probabilities)
    probs_norm = [p / total for _, p, _ in pitch_tokens if p > 0]
    entropy = -sum(p * math.log2(p) for p in probs_norm if p > 0)

    argmax_tok, argmax_prob, argmax_dist = max(pitch_tokens, key=lambda x: x[1])

    return {
        "n_pitch_tokens":   len(pitch_tokens),
        "total_pitch_mass": round(total, 6),
        "p_exact":          round(p_exact,     4),
        "p_within_1":       round(p_within_1,  4),
        "p_within_2":       round(p_within_2,  4),
        "p_within_6":       round(p_within_6,  4),
        "p_within_12":      round(p_within_12, 4),
        "entropy_pitch":    round(entropy, 4),
        "argmax_dist":      argmax_dist,
        "argmax_token":     argmax_tok,
    }


def _score_response(variant: str, raw: str, gt_midi: int) -> dict:
    """Standard exact-match scoring used for comparison."""
    if variant == "midi":
        pred = extract_midi(raw)
        err  = abs(pred - gt_midi) if pred is not None else None
        return {"exact_match": int(err == 0) if err is not None else 0,
                "within_1": int(err is not None and err <= 1)}
    elif variant == "abc":
        pred = extract_note(raw)
        gt_note = midi_to_note(gt_midi)
        dist = semitone_distance(gt_note, pred) if pred else None
        return {"exact_match": int(dist == 0) if dist is not None else 0,
                "within_1": int(dist is not None and dist <= 1)}
    else:  # doremi
        pred_pc = extract_solfege(raw)
        dist    = solfege_pc_distance(gt_midi, pred_pc) if pred_pc is not None else None
        return {"exact_match": int(dist == 0) if dist is not None else 0,
                "within_1": int(dist is not None and dist <= 1)}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_mass_vs_distance_plot(
    records: list[dict],
    run_dir: Path,
    model_name: str,
) -> None:
    if not _MPL:
        return

    max_dist = 25
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, variant in zip(axes, ("midi", "abc", "doremi")):
        sub = [r for r in records if r["prompt_variant"] == variant]
        if not sub:
            continue

        # Group by semitone distance bucket
        buckets: dict[int, list[float]] = {d: [] for d in range(max_dist + 1)}
        for r in sub:
            dist = r.get("argmax_dist")
            if dist is not None and dist <= max_dist:
                buckets[dist].append(r.get("p_exact", 0))

        # Average p_within_X at each exact distance level
        dist_bins = list(range(max_dist + 1))
        p_exact_vals    = [float(np.mean([r["p_exact"]    for r in sub if r.get("argmax_dist") == d] or [0])) for d in dist_bins]
        p_within_12_vals= [float(np.mean([r["p_within_12"] for r in sub if r.get("argmax_dist") == d] or [0])) for d in dist_bins]

        ax.plot(dist_bins, p_exact_vals,     label="P(exact)",      marker="o", markersize=3)
        ax.plot(dist_bins, p_within_12_vals, label="P(within±12)", marker="s", markersize=3, linestyle="--")
        ax.set_title(f"{variant.upper()}", fontsize=11)
        ax.set_xlabel("Argmax token distance (semitones)")
        ax.set_ylabel("Probability mass" if variant == "midi" else "")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Token probability mass — {model_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / f"mass_vs_distance_{model_name}.png", dpi=120)
    plt.close(fig)


def _save_entropy_heatmap(
    records: list[dict],
    run_dir: Path,
    model_name: str,
    sources: list[str],
) -> None:
    if not _MPL:
        return

    unique_midi = sorted(set(r["midi"] for r in records))
    for variant in ("midi", "abc", "doremi"):
        sub = [r for r in records if r["prompt_variant"] == variant]
        if not sub:
            continue

        mat = np.full((len(sources), len(unique_midi)), np.nan)
        for r in sub:
            si = sources.index(r["source"]) if r["source"] in sources else -1
            mi = unique_midi.index(r["midi"])
            if si >= 0 and r.get("entropy_pitch") is not None:
                mat[si, mi] = r["entropy_pitch"]

        fig, ax = plt.subplots(figsize=(max(10, len(unique_midi) * 0.25), max(4, len(sources) * 0.4)))
        im = ax.imshow(mat, aspect="auto", origin="upper", cmap="YlOrRd", vmin=0)
        plt.colorbar(im, ax=ax, label="Entropy (bits)")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources, fontsize=8)
        step = max(1, len(unique_midi) // 10)
        ax.set_xticks(range(0, len(unique_midi), step))
        ax.set_xticklabels([str(m) for m in unique_midi[::step]], fontsize=8)
        ax.set_xlabel("MIDI note"); ax.set_ylabel("Source")
        ax.set_title(f"Pitch-token entropy ({variant.upper()}) — {model_name}")
        fig.tight_layout()
        fig.savefig(run_dir / f"entropy_heatmap_{variant}_{model_name}.png", dpi=120)
        plt.close(fig)


def _save_p_exact_comparison_plot(
    records: list[dict],
    run_dir: Path,
    model_name: str,
) -> None:
    """Bar chart: P(exact) and P(within±12) per prompt variant."""
    if not _MPL:
        return

    variants  = ["midi", "abc", "doremi"]
    p_exact   = []
    p_within12= []
    for v in variants:
        sub = [r for r in records if r["prompt_variant"] == v]
        ve  = [r["p_exact"]    for r in sub if r.get("p_exact")    is not None]
        vw  = [r["p_within_12"] for r in sub if r.get("p_within_12") is not None]
        p_exact.append(float(np.mean(ve)) if ve else 0.0)
        p_within12.append(float(np.mean(vw)) if vw else 0.0)

    x = np.arange(len(variants))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, p_exact,    w, label="P(exact)",      color="steelblue")
    ax.bar(x + w/2, p_within12, w, label="P(within±12)", color="salmon")
    ax.set_xticks(x); ax.set_xticklabels([v.upper() for v in variants])
    ax.set_ylabel("Mean probability mass"); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Step-1 token probability mass by prompt variant\n{model_name}")
    fig.tight_layout()
    fig.savefig(run_dir / f"p_exact_by_variant_{model_name}.png", dpi=120)
    plt.close(fig)


# ── Per-condition distribution plots ─────────────────────────────────────────

_SOLFEGE_PC_LABELS = ["do", "", "re", "", "mi", "fa", "", "sol", "", "la", "", "si"]


def _save_distribution_plots(
    records: list[dict],
    run_dir: Path,
    model_name: str,
) -> None:
    """One figure per instrument: 5 rows (notes) × 3 cols (variants).
    Each subplot shows the full probability distribution over all possible answer
    values with a red dashed line marking the true value.
      MIDI/ABC : line+fill over MIDI 0–127
      Solfège  : bar chart over 12 pitch classes (enharmonics already merged)
    """
    if not _MPL:
        return

    out_dir = run_dir / "distributions"
    out_dir.mkdir(exist_ok=True)

    sources = sorted({r["source"] for r in records})
    midi_pitches = sorted({r["midi"] for r in records})
    variants = ["midi", "abc", "doremi"]
    variant_colors = {"midi": "#4C72B0", "abc": "#DD8452", "doremi": "#55A868"}

    for src in sources:
        n_rows = len(midi_pitches)
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 3 + 1))
        if n_rows == 1:
            axes = [axes]

        for row_idx, midi in enumerate(midi_pitches):
            note_name = midi_to_note(midi)
            for col_idx, variant in enumerate(variants):
                ax = axes[row_idx][col_idx]

                rec = next(
                    (r for r in records
                     if r["source"] == src and r["midi"] == midi
                     and r["prompt_variant"] == variant),
                    None,
                )
                step1     = (rec.get("step1_tokens")    or []) if rec else []
                all_steps = (rec.get("all_step_tokens") or []) if rec else []

                color = variant_colors[variant]

                if variant == "midi":
                    dist: dict[int, float] = _build_midi_dist_multistep(all_steps)
                    x = list(range(10,100))
                    y = [dist.get(xi, 0.0) for xi in x]
                    ax.plot(x, y, linewidth=1.0, color=color)
                    ax.fill_between(x, 0, y, alpha=0.25, color=color)
                    ax.axvline(midi, color="red", linewidth=1.5, linestyle="--")
                    ax.set_xlim(0, 127)
                    ax.set_xlabel("MIDI value", fontsize=7)

                elif variant == "abc":
                    dist = {}
                    for t in (step1 or []):
                        pred = _token_to_midi_abc(t["token"])
                        if pred is not None and pred < 1000:
                            dist[pred] = dist.get(pred, 0.0) + float(t["prob"])
                    x = list(range(128))
                    y = [dist.get(xi, 0.0) for xi in x]
                    ax.plot(x, y, linewidth=1.0, color=color)
                    ax.fill_between(x, 0, y, alpha=0.25, color=color)
                    ax.axvline(midi, color="red", linewidth=1.5, linestyle="--")
                    ax.set_xlim(0, 127)
                    ax.set_xlabel("MIDI (from note name)", fontsize=7)

                else:  # doremi
                    dist_pc: dict[int, float] = {}
                    for t in (step1 or []):
                        pred_pc = _token_to_pc_solfege(t["token"])
                        if pred_pc is not None:
                            dist_pc[pred_pc] = dist_pc.get(pred_pc, 0.0) + float(t["prob"])
                    x_pc = list(range(12))
                    y_pc = [dist_pc.get(xi, 0.0) for xi in x_pc]
                    ax.bar(x_pc, y_pc, color=color, alpha=0.75)
                    ax.axvline(midi % 12, color="red", linewidth=1.5, linestyle="--")
                    ax.set_xticks(x_pc)
                    ax.set_xticklabels(_SOLFEGE_PC_LABELS, rotation=45, ha="right", fontsize=6)
                    ax.set_xlabel("Pitch class (doremi)", fontsize=7)

                true_label = note_name if variant != "doremi" else _SOLFEGE_PC_LABELS[midi % 12]
                ax.set_title(
                    f"{variant.upper()} | {note_name}  (true: {true_label})",
                    fontsize=8,
                )
                ax.set_ylabel("Prob.", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.25)

        fig.suptitle(
            f"Token probability distributions — {src} — {model_name}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"distributions_{src}_{model_name}.png", dpi=130)
        plt.close(fig)

    print(f"  Distribution plots → {out_dir}/")


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    run_dir: Path,
    top_k: int,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    sources = sorted({c["source"] for c in conds})

    # Phase 1: build job list (one per cond × variant). Audio is already cached
    # in `c["wav"]` from build_conditions().
    jobs: list[dict] = []
    for c in conds:
        for variant, prompt in PROMPTS.items():
            jobs.append({"cond": c, "variant": variant, "prompt": prompt})

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        variant = job["variant"]
        prompt  = job["prompt"]
        wav     = c["wav"]
        all_steps: list[list[dict[str, Any]]] = []
        step1:     list[dict[str, Any]]        = []
        try:
            resp      = query_alm(
                model_name, wav, prompt,
                max_new_tokens=MAX_NEW_TOKENS, top_k=top_k, mode="probs",
            )
            raw       = resp.get("result") or ""
            all_steps = resp.get("top_tokens") or []
            step1     = all_steps[0] if all_steps else []
            mass      = _pitch_mass_metrics(all_steps, c["midi"], variant)
            verbal    = _score_response(variant, raw, c["midi"])
        except Exception as exc:
            raw    = ""
            mass   = {k: None for k in [
                "n_pitch_tokens", "total_pitch_mass", "p_exact", "p_within_1",
                "p_within_2", "p_within_6", "p_within_12", "entropy_pitch",
                "argmax_dist", "argmax_token",
            ]}
            verbal = {"exact_match": 0, "within_1": 0}

        return {
            "source":           c["source"],
            "source_type":      c["source_type"],
            "midi":             c["midi"],
            "note":             c["note"],
            "wav":              wav,
            "raw_response":     raw.strip() if raw else "",
            "prompt_variant":   variant,
            "prompt":           prompt,

            "step1_tokens":     step1,
            "all_step_tokens":  all_steps,
            # standard plot keys
            "instrument":       c["source"],
            "exact_match":      verbal["exact_match"],
            **mass,
        }

    raw_results = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"[{j['variant']:6s}] {j['cond']['note']:4s} {j['cond']['source']:14s}",
    )
    records: list[dict] = [r for r in raw_results if r is not None]

    # ── Summary ───────────────────────────────────────────────────────────────
    per_variant: dict[str, dict] = {}
    for v in PROMPTS:
        sub = [r for r in records if r["prompt_variant"] == v]
        if not sub:
            per_variant[v] = {}
            continue
        per_variant[v] = {
            "verbal_exact":   round(sum(r["exact_match"] for r in sub) / len(sub), 4),
            "p_exact":        round(float(np.mean([r["p_exact"]    for r in sub if r["p_exact"]    is not None])), 4),
            "p_within_1":     round(float(np.mean([r["p_within_1"] for r in sub if r["p_within_1"] is not None])), 4),
            "p_within_12":    round(float(np.mean([r["p_within_12"]for r in sub if r["p_within_12"]is not None])), 4),
            "entropy_mean":   round(float(np.mean([r["entropy_pitch"] for r in sub if r["entropy_pitch"] is not None])), 4) if any(r["entropy_pitch"] is not None for r in sub) else None,
        }

    summary = {
        "total": len(records),
        "per_variant": per_variant,
    }

    summary_lines = [
        f"  Stimuli : {len(records) // len(PROMPTS)}  ({len(sources)} sources × {len(SELECTED_MIDI_PITCHES)} pitches)",
        "",
        f"  {'Variant':>8}  {'Verbal%':>8}  {'P(exact)':>9}  {'P(±1)':>7}  {'P(±12)':>8}  {'Entropy':>8}",
        f"  {'─' * 58}",
    ]
    for v, d in per_variant.items():
        if not d:
            continue
        summary_lines.append(
            f"  {v.upper():>8}  {d.get('verbal_exact', 0):>8.1%}  "
            f"{d.get('p_exact', 0):>9.3f}  {d.get('p_within_1', 0):>7.3f}  "
            f"{d.get('p_within_12', 0):>8.3f}  {d.get('entropy_mean', 0) or 0:>8.3f}"
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _save_mass_vs_distance_plot(records, run_dir, model_name)
    _save_entropy_heatmap(records, run_dir, model_name, sources)
    _save_p_exact_comparison_plot(records, run_dir, model_name)
    _save_distribution_plots(records, run_dir, model_name)
    save_accuracy_plots(
        records, run_dir, model_name,
        instrument_key="source",
        pitch_key="midi",
        prompt_key="prompt_variant",
        accuracy_key="exact_match",
    )

    # ── Save results ──────────────────────────────────────────────────────────
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=SELECTED_SOURCES,
        midi_pitches=SELECTED_MIDI_PITCHES,
        tone_duration_ms=TONE_DURATION_MS,
        top_k=top_k, max_new_tokens=MAX_NEW_TOKENS,
        prompts=PROMPTS,
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)

    flat_summary = {
        "verbal_exact_midi":   per_variant.get("midi",   {}).get("verbal_exact"),
        "verbal_exact_abc":    per_variant.get("abc",    {}).get("verbal_exact"),
        "verbal_exact_doremi": per_variant.get("doremi", {}).get("verbal_exact"),
        "p_exact_midi":        per_variant.get("midi",   {}).get("p_exact"),
        "p_exact_abc":         per_variant.get("abc",    {}).get("p_exact"),
        "p_exact_doremi":      per_variant.get("doremi", {}).get("p_exact"),
        "p_within_12_midi":    per_variant.get("midi",   {}).get("p_within_12"),
        "entropy_midi":        per_variant.get("midi",   {}).get("entropy_mean"),
    }
    return flat_summary


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--top-k",   type=int, default=DEFAULT_TOP_K,
                        help=f"Top-K tokens to request per generation step (default: {DEFAULT_TOP_K})")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    engine.set_exp(EXP_NAME)
    conds = build_conditions()
    sources = sorted({c["source"] for c in conds})
    note_names = [midi_to_note(m) for m in SELECTED_MIDI_PITCHES]
    print(f"Experiment   : {EXP_NAME}")
    print(f"Sources      : {sources}")
    print(f"Pitches      : {note_names}  (MIDI {SELECTED_MIDI_PITCHES})")
    print(f"Conditions   : {len(conds)} × {len(PROMPTS)} variants = {len(conds)*len(PROMPTS)} queries/model")
    print(f"Audio engine : generation.engine.tone  ({TONE_DURATION_MS} ms)")
    print(f"matplotlib   : {'available' if _MPL else 'NOT installed'}")
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    engine.set_exp(EXP_NAME)
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    conds    = build_conditions()
    data_dir = exp_data_dir(EXP_NAME)   # reserved for future caching

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)} × {len(PROMPTS)} variants = {len(conds)*len(PROMPTS)} queries/model")
    print(f"Top-K      : {args.top_k}")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(
            model_name, conds, run_dir, top_k=args.top_k,
        )
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
