"""
Embedding distance analysis — exp_4 companion.

For each pair of sine-wave tones that differ by Δ cents, fetch the audio
encoder embedding from a running model server and compute the cosine distance
between them.  Plots embedding distance vs. Δ (cents) alongside the
behavioural accuracy curve from exp_4.

Usage:
    # requires a running model server (python model/api.py)
    python analysis/embedding_distance.py
    python analysis/embedding_distance.py --model audio_flamingo_next_instruct
    python analysis/embedding_distance.py --base A4 --n-per-delta 5

The /embed endpoint must be available on the model server.
Results are saved to analysis/results/embedding_distance/<timestamp>/.
"""

import argparse
import json
import os
import sys
import wave
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from experiments.scripts.exp_4_relative_pitch import (
    BASE_FREQS,
    DELTA_CENTS,
    TONE_DURATION,
    cents_above,
    write_sequence_wav,
)

RESULTS_DIR = Path(__file__).parent / "results" / "embedding_distance"

DEFAULT_MODEL  = "audio_flamingo_next_instruct"
DEFAULT_BASE   = "A4"
DEFAULT_N      = 3    # tone pairs per delta condition
DEFAULT_SEED   = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_distance(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(1.0 - np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def l2_distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def write_single_wav(freq_hz: float, path: Path) -> None:
    sr = config.SAMPLE_RATE
    n  = int(sr * TONE_DURATION)
    t  = np.linspace(0, TONE_DURATION, n, endpoint=False)
    fade = int(sr * 0.05)
    env  = np.ones(n, dtype=np.float32)
    env[:fade]  = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    audio = (0.8 * np.sin(2 * np.pi * freq_hz * t) * env * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(audio.tobytes())


def fetch_embedding(model_name: str, wav_path: str) -> list[float]:
    url = config.MODEL_URLS[model_name]
    with open(wav_path, "rb") as f:
        resp = requests.post(
            f"{url}/embed",
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ── Main analysis ─────────────────────────────────────────────────────────────

def run(model_name: str, base_name: str, n_per_delta: int, seed: int) -> None:
    base_hz = BASE_FREQS[base_name]
    url     = config.MODEL_URLS[model_name]

    # verify server is up
    try:
        info = requests.get(f"{url}/health", timeout=10).json()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"\nModel server not reachable at {url}\n"
            f"Start it with:  python model/api.py  (or the appropriate variant)"
        )

    # check /embed is available
    rng = random.Random(seed)
    tmp_dir = Path("/tmp/pitchbench_embed")
    tmp_dir.mkdir(exist_ok=True)

    print(f"Model  : {model_name}")
    print(f"Server : {url}")
    print(f"Base   : {base_name} ({base_hz:.1f} Hz)")
    print(f"Deltas : {DELTA_CENTS}\n")

    records = []
    for delta in DELTA_CENTS:
        high_hz = cents_above(base_hz, delta)
        cosine_dists, l2_dists = [], []

        for trial in range(n_per_delta):
            # slight jitter around base to avoid caching identical files
            jitter = rng.uniform(-0.01, 0.01)  # ±0.01 Hz, negligible perceptually
            low_f  = base_hz + jitter
            high_f = high_hz + jitter

            low_wav  = tmp_dir / f"{base_name}_{delta}c_t{trial}_low.wav"
            high_wav = tmp_dir / f"{base_name}_{delta}c_t{trial}_high.wav"
            write_single_wav(low_f,  low_wav)
            write_single_wav(high_f, high_wav)

            emb_low  = fetch_embedding(model_name, str(low_wav))
            emb_high = fetch_embedding(model_name, str(high_wav))

            cd = cosine_distance(emb_low, emb_high)
            l2 = l2_distance(emb_low, emb_high)
            cosine_dists.append(cd)
            l2_dists.append(l2)

        mean_cos = float(np.mean(cosine_dists))
        mean_l2  = float(np.mean(l2_dists))
        records.append({
            "base_name":         base_name,
            "base_hz":           base_hz,
            "delta_cents":       delta,
            "delta_hz":          round(high_hz - base_hz, 4),
            "n_trials":          n_per_delta,
            "cosine_dist_mean":  round(mean_cos, 6),
            "cosine_dist_std":   round(float(np.std(cosine_dists)), 6),
            "l2_dist_mean":      round(mean_l2, 4),
            "l2_dist_std":       round(float(np.std(l2_dists)), 4),
        })
        print(f"  Δ={delta:>5} c  ({high_hz - base_hz:>7.2f} Hz)  "
              f"cos={mean_cos:.4f}  l2={mean_l2:.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace(" ", "_")

    payload = {
        "metadata": {
            "model": model_name,
            "model_info": info,
            "base_name": base_name,
            "base_hz": base_hz,
            "delta_cents": DELTA_CENTS,
            "n_per_delta": n_per_delta,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        },
        "records": records,
    }
    json_path = run_dir / f"emb_dist_{slug}_{base_name}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    _save_plot(records, run_dir, slug, base_name, base_hz)
    print(f"\nSaved → {run_dir}/")


def _save_plot(records: list[dict], run_dir: Path, slug: str, base_name: str, base_hz: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    deltas    = [r["delta_cents"]      for r in records]
    cos_means = [r["cosine_dist_mean"] for r in records]
    cos_stds  = [r["cosine_dist_std"]  for r in records]
    l2_means  = [r["l2_dist_mean"]     for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Embedding distance vs. Δ pitch — {slug}\nBase: {base_name} ({base_hz:.0f} Hz)")

    for ax, means, stds, label, color in [
        (ax1, cos_means, cos_stds, "Cosine distance", "tab:blue"),
        (ax2, l2_means,  [0]*len(l2_means), "L2 distance",     "tab:orange"),
    ]:
        ax.errorbar(deltas, means, yerr=stds, fmt="o-", color=color,
                    linewidth=2, capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel("Δ (cents)")
        ax.set_ylabel(label)
        ax.set_xticks(deltas)
        ax.set_xticklabels([str(d) for d in deltas], fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = run_dir / f"emb_dist_{slug}_{base_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved  → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=list(config.MODELS),
                        help=f"Model slug (default: {DEFAULT_MODEL})")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        choices=list(BASE_FREQS),
                        help=f"Base frequency name (default: {DEFAULT_BASE})")
    parser.add_argument("--n-per-delta", type=int, default=DEFAULT_N,
                        help=f"Tone pairs per delta condition (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run(args.model, args.base, args.n_per_delta, args.seed)
