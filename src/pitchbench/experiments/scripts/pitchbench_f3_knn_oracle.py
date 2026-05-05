"""
Experiment 15 — kNN Oracle (proof-of-concept)
Demonstrates that a simple k-Nearest-Neighbour classifier on audio encoder
embeddings can achieve higher pitch accuracy than the model's own verbal output,
proving that pitch information *is* present in the embedding space but is not
reliably decoded into text.

Pipeline:
  1. Collect audio embeddings via /embed for all (source, MIDI) pairs.
  2. Build a labelled reference set from one half of pitches per source (fixed
     seed); use the other half as queries.
  3. For each query, retrieve the nearest neighbour by cosine similarity and
     report its MIDI label as the kNN prediction.
  4. Optionally run the LLM's verbal prediction on the same query stimuli and
     compare the two accuracies.

Evaluation conditions:
  - within_source:      reference and query from the same source
  - cross_source:       reference from source A, query from source B
  - waveform_to_instr:  reference = all waveforms, query = all instruments
  - instr_to_waveform:  reference = all instruments, query = all waveforms

Embeddings are loaded from the exp_13 cache if available, otherwise collected
fresh and cached in data/<exp_name>/embeddings_<model>.npz.

Usage:
    python experiments/run.py exp_15_knn_oracle
    python experiments/run.py exp_15_knn_oracle --preview
    python experiments/run.py exp_15_knn_oracle --models audio_flamingo_next_instruct
    python experiments/run.py exp_15_knn_oracle --no-verbal
    python experiments/run.py exp_15_knn_oracle --k 3
"""

import argparse
import random
from pathlib import Path

import numpy as np

import pitchbench.config as config
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.music import (
    extract_midi, extract_note, extract_solfege,
    midi_to_note, midi_to_solfege,
    semitone_distance, solfege_pc_distance,
)
from pitchbench.experiments.helpers.results import exp_data_dir, get_run_metadata, make_run_dir, save_comparison, save_results, extract_format_accuracies

EXP_NAME = Path(__file__).stem
EXP13_NAME = "pitchbench_f1_embedding_geometry"   # may reuse embedding cache

MIDI_MIN = config.DEFAULT_MIDI_MIN
MIDI_MAX = config.DEFAULT_MIDI_MAX
MIDI_PITCHES: list[int] = list(range(MIDI_MIN, MIDI_MAX + 1))
SPLIT_SEED = config.DEFAULT_SEED
DEFAULT_K  = 1

VERBAL_PROMPTS: dict[str, str] = {
    "midi": (
        "This audio contains a single musical note. "
        "What is the MIDI note number (integer 0–127)? "
        "Reply with ONLY the integer."
    ),
    "abc": (
        "This audio contains a single musical note. "
        "What is the note name and octave, e.g. C4, F#3? "
        "Reply with ONLY the note name."
    ),
    "doremi": (
        "This audio contains a single musical note. "
        "What is the solfège syllable and accidental (if needed) (fixed-do: do=C re=D mi=E fa=F sol=G la=A si=B)? "
        "Reply with the syllable and accidental (if needed) and accidental (if necessary)."
    ),
}

try:
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# ── Stimuli catalogue ─────────────────────────────────────────────────────────

def _discover_stimuli_dir() -> tuple[Path, list[str]]:
    v1 = config.STIMULI_DIR / "v1"
    if v1.exists():
        sources = sorted(d.name for d in v1.iterdir() if d.is_dir())
        if sources:
            return v1, sources
    v0 = config.STIMULI_DIR / "v0"
    if not v0.exists():
        raise FileNotFoundError(
            f"No stimuli found. Generate them first:\n"
            f"  python generation/generate_stimuli_v1.py"
        )
    sources = sorted(d.name for d in v0.iterdir() if d.is_dir() and d.name != "labels.csv")
    return v0, sources


def build_conditions(stimuli_root: Path, sources: list[str]) -> list[dict]:
    rows: list[dict] = []
    for src in sources:
        src_dir = stimuli_root / src
        for midi in MIDI_PITCHES:
            wav = src_dir / f"{midi}.wav"
            if wav.exists():
                rows.append({
                    "source":      src,
                    "source_type": "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":        midi,
                    "note":        midi_to_note(midi),
                    "wav":         str(wav),
                })
    return rows


# ── Embedding collection / cache ──────────────────────────────────────────────

def _load_or_collect_embeddings(
    model_name: str,
    conds: list[dict],
    data_dir: Path,
) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Try exp_13 cache first, then exp_15 cache, then collect fresh.
    Returns (X, midi_labels, source_labels).
    """
    # Prefer exp_13 cache (same embedding content)
    exp13_cache = config.DATA_DIR / EXP13_NAME / f"embeddings_{model_name}.npz"
    exp15_cache = data_dir / f"embeddings_{model_name}.npz"

    for cache in (exp13_cache, exp15_cache):
        if cache.exists():
            print(f"  Loading embeddings from {cache.parent.name}/{cache.name}")
            z = np.load(cache, allow_pickle=True)
            X = z["X"]
            midi_labels   = list(z["midi_labels"])
            source_labels = list(z["source_labels"])
            # verify dimensions match conds
            if len(X) == len(conds):
                return X, midi_labels, source_labels
            print(f"    Cache size mismatch ({len(X)} vs {len(conds)}) — re-collecting.")

    print(f"  Collecting {len(conds)} embeddings …")

    # Phase 2: dispatch HTTP queries with bounded concurrency. (Phase 1 — audio
    # gen — is already done; stimuli are on disk in `c["wav"]`.)
    def _query_one(c: dict) -> dict:
        vec = query_alm(model_name, c["wav"], mode="embed")["embedding"]
        return {"vec": vec, "midi": c["midi"], "source": c["source"]}

    raw_results = dispatch(
        conds, _query_one,
        model_name=model_name,
        label_fn=lambda c: f"{c['source']:10s} midi={c['midi']:>3}",
    )

    vecs: list[list[float]] = []
    midi_labels: list[int] = []
    source_labels: list[str] = []
    for r in raw_results:
        if r is None:
            continue
        vecs.append(r["vec"])
        midi_labels.append(r["midi"])
        source_labels.append(r["source"])

    X = np.array(vecs, dtype=np.float32)
    np.savez_compressed(
        exp15_cache,
        X=X,
        midi_labels=np.array(midi_labels),
        source_labels=np.array(source_labels, dtype=object),
    )
    print(f"  Saved embeddings → {exp15_cache.name}  shape={X.shape}")
    return X, midi_labels, source_labels


# ── Reference / query split ───────────────────────────────────────────────────

def _make_split(
    midi_pitches: list[int],
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Randomly split MIDI pitches 50/50 into reference and query sets."""
    rng = random.Random(seed)
    shuffled = list(midi_pitches)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]


# ── kNN retrieval ─────────────────────────────────────────────────────────────

def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xn = X / norms
    return Xn @ Xn.T


def knn_predict(
    ref_X: np.ndarray,
    ref_labels: list[int],
    query_X: np.ndarray,
    k: int = 1,
) -> list[int]:
    """Return predicted MIDI labels for each query via cosine kNN in ref set."""
    ref_norms   = np.linalg.norm(ref_X,   axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_X, axis=1, keepdims=True)
    ref_norms   = np.where(ref_norms   < 1e-10, 1e-10, ref_norms)
    query_norms = np.where(query_norms < 1e-10, 1e-10, query_norms)
    ref_n   = ref_X   / ref_norms
    query_n = query_X / query_norms

    sims = query_n @ ref_n.T          # (n_query, n_ref)
    ref_arr = np.array(ref_labels)

    preds: list[int] = []
    for row in sims:
        top_k_idx = np.argsort(row)[::-1][:k]
        top_k_labels = ref_arr[top_k_idx]
        # majority vote (if k>1)
        pred = int(np.bincount(top_k_labels).argmax()) if k > 1 else int(top_k_labels[0])
        preds.append(pred)
    return preds


def _accuracy_stats(gt: list[int], pred: list[int]) -> dict:
    if not gt:
        return {"exact": None, "within_1": None, "within_12": None, "n": 0}
    dists = [abs(g - p) for g, p in zip(gt, pred)]
    n = len(dists)
    return {
        "exact":     round(sum(d == 0  for d in dists) / n, 4),
        "within_1":  round(sum(d <= 1  for d in dists) / n, 4),
        "within_12": round(sum(d <= 12 for d in dists) / n, 4),
        "n": n,
    }


# ── Verbal baseline ───────────────────────────────────────────────────────────

def _score_verbal(variant: str, raw: str, gt_midi: int) -> int:
    """1 if exact match, else 0."""
    if variant == "midi":
        pred = extract_midi(raw)
        return int(pred is not None and abs(pred - gt_midi) == 0)
    elif variant == "abc":
        pred = extract_note(raw)
        gt_note = midi_to_note(gt_midi)
        dist = semitone_distance(gt_note, pred) if pred else None
        return int(dist is not None and dist == 0)
    else:
        pred_pc = extract_solfege(raw)
        dist    = solfege_pc_distance(gt_midi, pred_pc) if pred_pc is not None else None
        return int(dist is not None and dist == 0)


def run_verbal_baseline(
    model_name: str,
    query_conds: list[dict],
) -> dict[str, float]:
    """Return exact-match accuracy per variant for the query set."""
    # Phase 1: build job list (one per cond × variant).
    jobs: list[dict] = []
    for c in query_conds:
        for variant, prompt in VERBAL_PROMPTS.items():
            jobs.append({"cond": c, "variant": variant, "prompt": prompt})

    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(job: dict) -> dict:
        c       = job["cond"]
        variant = job["variant"]
        prompt  = job["prompt"]
        raw = query_alm(model_name, c["wav"], prompt)["result"]
        return {"variant": variant, "score": _score_verbal(variant, raw, c["midi"])}

    raw_results = dispatch(
        jobs, _query_one,
        model_name=model_name,
        label_fn=lambda j: f"verbal[{j['variant']:6s}] {j['cond']['note']:4s} {j['cond']['source']:14s}",
    )

    results: dict[str, list[int]] = {v: [] for v in VERBAL_PROMPTS}
    for r in raw_results:
        if r is None:
            continue
        results[r["variant"]].append(r["score"])

    return {
        v: round(sum(acc) / len(acc), 4) if acc else 0.0
        for v, acc in results.items()
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_knn_vs_verbal_plot(
    knn_results: dict,
    verbal_results: dict | None,
    run_dir: Path,
    model_name: str,
) -> None:
    if not _MPL:
        return

    cond_names = list(knn_results.keys())
    knn_accs   = [knn_results[c].get("exact") or 0.0 for c in cond_names]

    fig, ax = plt.subplots(figsize=(max(8, len(cond_names) * 1.2), 5))
    x = np.arange(len(cond_names))
    width = 0.35 if verbal_results else 0.6

    bars_knn = ax.bar(x - width/2 if verbal_results else x, knn_accs,
                      width, label="kNN oracle", color="steelblue")

    if verbal_results:
        verbal_avg = np.mean(list(verbal_results.values()))
        bars_verb = ax.bar(x + width/2, [verbal_avg] * len(cond_names),
                           width, label="LLM verbal (avg)", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"kNN oracle vs. LLM verbal pitch accuracy\n{model_name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars_knn:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.0%}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(run_dir / f"knn_vs_verbal_{model_name}.png", dpi=120)
    plt.close(fig)


def _save_cross_source_heatmap(
    cross_results: dict[str, dict[str, dict]],
    run_dir: Path,
    model_name: str,
) -> None:
    """cross_results[ref_src][query_src] = {exact, ...}"""
    if not _MPL or not cross_results:
        return
    sources = sorted(cross_results.keys())
    n = len(sources)
    mat = np.full((n, n), np.nan)
    for i, ref in enumerate(sources):
        for j, qry in enumerate(sources):
            acc = cross_results.get(ref, {}).get(qry, {}).get("exact")
            if acc is not None:
                mat[i, j] = acc

    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    im = ax.imshow(mat, aspect="auto", origin="upper", cmap="YlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Exact-match accuracy")
    ax.set_xticks(range(n)); ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(sources, fontsize=8)
    ax.set_xlabel("Query source"); ax.set_ylabel("Reference source")
    ax.set_title(f"Cross-source kNN accuracy\n{model_name}")
    fig.tight_layout()
    fig.savefig(run_dir / f"cross_source_heatmap_{model_name}.png", dpi=120)
    plt.close(fig)


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    data_dir: Path,
    run_dir: Path,
    k: int,
    run_verbal: bool,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    X, midi_labels, source_labels = _load_or_collect_embeddings(
        model_name, conds, data_dir
    )
    print(f"  Embeddings : {X.shape}  (k={k})")

    y       = np.array(midi_labels)
    s_arr   = np.array(source_labels)
    sources = sorted(set(source_labels))

    ref_midi, qry_midi = _make_split(MIDI_PITCHES, seed=SPLIT_SEED)
    ref_midi_set = set(ref_midi)
    qry_midi_set = set(qry_midi)

    # ── Within-source condition ───────────────────────────────────────────────
    print("  Running: within-source kNN …")
    within_results: dict[str, dict] = {}
    for src in sources:
        src_mask  = s_arr == src
        ref_mask  = src_mask & np.array([m in ref_midi_set for m in midi_labels])
        qry_mask  = src_mask & np.array([m in qry_midi_set for m in midi_labels])
        if not ref_mask.any() or not qry_mask.any():
            continue
        ref_X  = X[ref_mask]
        ref_y  = list(y[ref_mask])
        qry_X  = X[qry_mask]
        gt_y   = list(y[qry_mask])
        preds  = knn_predict(ref_X, ref_y, qry_X, k=k)
        within_results[src] = _accuracy_stats(gt_y, preds)
        sym = "✓" if within_results[src]["exact"] > 0.5 else "~"
        print(f"    within  {src:14s}  exact={within_results[src]['exact']:.0%}  {sym}")

    # ── Cross-source condition ────────────────────────────────────────────────
    print("  Running: cross-source kNN …")
    cross_results: dict[str, dict[str, dict]] = {}
    for ref_src in sources:
        cross_results[ref_src] = {}
        ref_mask = s_arr == ref_src
        ref_X    = X[ref_mask]
        ref_y    = list(y[ref_mask])
        for qry_src in sources:
            if qry_src == ref_src:
                continue
            qry_mask = s_arr == qry_src
            qry_X    = X[qry_mask]
            gt_y     = list(y[qry_mask])
            preds    = knn_predict(ref_X, ref_y, qry_X, k=k)
            cross_results[ref_src][qry_src] = _accuracy_stats(gt_y, preds)

    # ── Waveform ↔ instrument cross conditions ────────────────────────────────
    print("  Running: waveform/instrument cross-domain kNN …")
    wf_sources   = [s for s in sources if s in config.WAVEFORMS]
    inst_sources = [s for s in sources if s not in config.WAVEFORMS]

    cross_domain: dict[str, dict] = {}

    def _pool_masked(src_list: list[str], midi_set: set | None = None) -> tuple[np.ndarray, list[int]]:
        masks = []
        for s in src_list:
            m = s_arr == s
            if midi_set:
                m = m & np.array([mi in midi_set for mi in midi_labels])
            masks.append(m)
        full_mask = np.zeros(len(s_arr), dtype=bool)
        for m in masks:
            full_mask |= m
        return X[full_mask], list(y[full_mask])

    if wf_sources and inst_sources:
        ref_X, ref_y = _pool_masked(wf_sources, ref_midi_set)
        qry_X, gt_y  = _pool_masked(inst_sources, qry_midi_set)
        preds = knn_predict(ref_X, ref_y, qry_X, k=k)
        cross_domain["waveform_to_instr"] = _accuracy_stats(gt_y, preds)
        print(f"    waveform→instr  exact={cross_domain['waveform_to_instr']['exact']:.0%}")

        ref_X, ref_y = _pool_masked(inst_sources, ref_midi_set)
        qry_X, gt_y  = _pool_masked(wf_sources, qry_midi_set)
        preds = knn_predict(ref_X, ref_y, qry_X, k=k)
        cross_domain["instr_to_waveform"] = _accuracy_stats(gt_y, preds)
        print(f"    instr→waveform  exact={cross_domain['instr_to_waveform']['exact']:.0%}")

    # ── Leave-one-out overall ─────────────────────────────────────────────────
    print("  Running: overall leave-one-out kNN …")
    sim = (lambda A: (lambda n: (A / np.where(n < 1e-10, 1e-10, n)) @ (A / np.where(n < 1e-10, 1e-10, n)).T)(
        np.linalg.norm(A, axis=1, keepdims=True)))(X)
    np.fill_diagonal(sim, -2.0)
    preds_loo = [int(y[int(np.argmax(sim[i]))]) for i in range(len(y))]
    loo_stats = _accuracy_stats(list(y), preds_loo)
    print(f"    LOO overall  exact={loo_stats['exact']:.0%}  within±1={loo_stats['within_1']:.0%}")

    # ── Verbal baseline (optional) ────────────────────────────────────────────
    verbal_accuracy: dict[str, float] | None = None
    if run_verbal:
        print("  Running: LLM verbal baseline on query set …")
        # Use per-source query conditions (matching the within-source kNN query set)
        query_conds = [c for c in conds if c["midi"] in qry_midi_set]
        verbal_accuracy = run_verbal_baseline(model_name, query_conds)
        print(f"  Verbal accuracy: " +
              "  ".join(f"{v}={a:.1%}" for v, a in verbal_accuracy.items()))

    # ── Flatten for the bar plot ──────────────────────────────────────────────
    knn_bar: dict[str, dict] = {"loo_overall": loo_stats}
    knn_bar.update({f"within_{s}": d for s, d in within_results.items()})
    knn_bar.update(cross_domain)

    # ── Summary ───────────────────────────────────────────────────────────────
    within_avg = np.mean([d["exact"] for d in within_results.values() if d.get("exact") is not None])
    cross_flat = [v for row in cross_results.values() for v in row.values() if v.get("exact") is not None]
    cross_avg  = float(np.mean([d["exact"] for d in cross_flat])) if cross_flat else None

    summary_lines = [
        f"  Sources     : {len(sources)} ({len(wf_sources)} waveforms, {len(inst_sources)} instruments)",
        f"  Ref pitches : {len(ref_midi)}  |  Query pitches: {len(qry_midi)}  (seed={SPLIT_SEED})",
        f"  k           : {k}",
        "",
        f"  LOO overall     : exact={loo_stats['exact']:.1%}  within±1={loo_stats['within_1']:.1%}",
        f"  Within-source   : exact={within_avg:.1%} (avg over {len(within_results)} sources)",
    ]
    if cross_avg is not None:
        summary_lines.append(
            f"  Cross-source    : exact={cross_avg:.1%} (avg over {len(cross_flat)} source pairs)"
        )
    for cond_name, cd in cross_domain.items():
        summary_lines.append(
            f"  {cond_name:20s}: exact={cd['exact']:.1%}  within±1={cd['within_1']:.1%}"
        )
    if verbal_accuracy:
        summary_lines += [
            "",
            "  LLM verbal (query set):",
        ]
        for v, a in verbal_accuracy.items():
            delta = a - loo_stats["exact"]
            summary_lines.append(f"    {v.upper():>8}: {a:.1%}  (vs kNN LOO: {delta:+.1%})")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    # ── Build per-source records for CSV ──────────────────────────────────────
    records: list[dict] = []
    for src in sources:
        ws = within_results.get(src, {})
        records.append({
            "source":               src,
            "source_type":          "waveform" if src in config.WAVEFORMS else "instrument",
            "knn_within_exact":     ws.get("exact"),
            "knn_within_within1":   ws.get("within_1"),
            "n_query":              ws.get("n"),
        })

    summary = {
        "total_embeddings": len(X),
        "loo_overall":      loo_stats,
        "within_source_avg":float(within_avg),
        "cross_source_avg": cross_avg,
        "cross_domain":     cross_domain,
        "verbal_accuracy":  verbal_accuracy,
        "per_source_within":within_results,
    }

    # ── Plots ─────────────────────────────────────────────────────────────────
    _save_knn_vs_verbal_plot(knn_bar, verbal_accuracy, run_dir, model_name)
    _save_cross_source_heatmap(cross_results, run_dir, model_name)

    # ── Save results ──────────────────────────────────────────────────────────
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        sources=sources, midi_range=[MIDI_MIN, MIDI_MAX],
        ref_midi=ref_midi, qry_midi=qry_midi,
        split_seed=SPLIT_SEED, k=k,
        run_verbal=run_verbal,
    )
    save_results(EXP_NAME, model_name, records, summary, metadata, summary_lines, run_dir=run_dir)

    flat = {
        "knn_loo_exact":          loo_stats["exact"],
        "knn_loo_within1":        loo_stats["within_1"],
        "knn_within_source_avg":  float(within_avg),
        "knn_cross_source_avg":   cross_avg if cross_avg is not None else float("nan"),
    }
    if verbal_accuracy:
        flat["verbal_midi"]   = verbal_accuracy.get("midi",   0.0)
        flat["verbal_abc"]    = verbal_accuracy.get("abc",    0.0)
        flat["verbal_doremi"] = verbal_accuracy.get("doremi", 0.0)
    return flat


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",   action="store_true")
    parser.add_argument("--no-verbal", action="store_true",
                        help="Skip the LLM verbal baseline (faster)")
    parser.add_argument("--k",         type=int, default=DEFAULT_K,
                        help=f"Number of nearest neighbours (default: {DEFAULT_K})")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    stimuli_root, sources = _discover_stimuli_dir()
    conds = build_conditions(stimuli_root, sources)
    ref_midi, qry_midi = _make_split(MIDI_PITCHES, seed=SPLIT_SEED)
    wf   = [s for s in sources if s in config.WAVEFORMS]
    inst = [s for s in sources if s not in config.WAVEFORMS]

    print(f"Experiment   : {EXP_NAME}")
    print(f"Stimuli root : {stimuli_root}  ({len(sources)} sources)")
    print(f"Sources      : {sources}")
    print(f"Waveforms    : {wf}")
    print(f"Instruments  : {inst}")
    print(f"MIDI range   : {MIDI_MIN}–{MIDI_MAX}  ({len(MIDI_PITCHES)} pitches)")
    print(f"Ref set      : {sorted(ref_midi)}  ({len(ref_midi)} pitches, seed={SPLIT_SEED})")
    print(f"Query set    : {sorted(qry_midi)}  ({len(qry_midi)} pitches)")
    print(f"Total conds  : {len(conds)}")
    print("\nRun without --preview to start the experiment.")


def run() -> dict:
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    stimuli_root, sources = _discover_stimuli_dir()
    conds    = build_conditions(stimuli_root, sources)
    data_dir = exp_data_dir(EXP_NAME)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Sources    : {len(sources)}  |  Stimuli: {len(conds)}  |  k={args.k}")
    print(f"Verbal     : {'disabled' if args.no_verbal else 'enabled'}")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        all_summaries[model_name] = run_one_model(
            model_name, conds, data_dir, run_dir,
            k=args.k,
            run_verbal=not args.no_verbal,
        )
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
