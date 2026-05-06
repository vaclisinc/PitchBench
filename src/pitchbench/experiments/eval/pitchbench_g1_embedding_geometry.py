"""
Experiment 13 — Embedding geometry
Probes the audio encoder's embedding space to see whether pitch is represented
geometrically.  Uses the /embed endpoint to extract mean-pooled embeddings for
every (source, MIDI) combination, then runs five analyses:

  1. PCA  — 2-D projection coloured by MIDI pitch and by source
  2. Linear probe — LinearSVC on embeddings predicts MIDI class (49 classes)
  3. k-NN retrieval — leave-one-out, cosine similarity; within-source & cross-source
  4. Cosine-similarity heatmap — average pairwise similarity as a function of
     semitone distance
  5. Octave periodicity — are embeddings separated by exactly 12 semitones more
     similar than those separated by other intervals?

Stimuli: stimuli/v1/{source}/{midi}.wav (full instrument + waveform set).
Falls back to stimuli/v0 (piano / violin / flute) if v1 has not been generated.

Embeddings are cached in data/<exp_name>/embeddings_<model>.npz so the model
server only needs to be queried once per model.

Usage:
    python experiments/run.py exp_13_embedding_geometry
    python experiments/run.py exp_13_embedding_geometry --preview
    python experiments/run.py exp_13_embedding_geometry --models audio_flamingo_next_instruct
    python experiments/run.py exp_13_embedding_geometry --no-cache
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import pitchbench.config as config
from pitchbench.experiments.helpers.api import get_model_info, query_alm
from pitchbench.experiments.helpers.dispatcher import dispatch
from pitchbench.experiments.helpers.results import _safe_stem, exp_data_dir, get_run_metadata, make_run_dir, save_comparison, extract_format_accuracies

EXP_NAME = Path(__file__).stem

# Data-generation parameters (sourced from config.pitchbench_g1_*)
MIDI_PITCHES = config.pitchbench_g1_PITCHES
SOURCES      = config.pitchbench_g1_SOURCES
MIDI_MIN     = min(MIDI_PITCHES)
MIDI_MAX     = max(MIDI_PITCHES)

# Colours for PCA scatter
try:
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

try:
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import umap
    _UMAP = True
except ImportError:
    _UMAP = False


# ── Stimuli catalogue ─────────────────────────────────────────────────────────

def _discover_stimuli_dir() -> tuple[Path, list[str]]:
    """Return (stimuli_root, list_of_sources) preferring v1 over v0."""
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
                    "source":       src,
                    "source_type":  "waveform" if src in config.WAVEFORMS else "instrument",
                    "midi":         midi,
                    "wav":          str(wav),
                })
    return rows


# ── Embedding collection ──────────────────────────────────────────────────────

def collect_embeddings(
    model_name: str,
    conds: list[dict],
    data_dir: Path,
    use_cache: bool = True,
) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Return (X, midi_labels, source_labels).
    X shape: (N, D) — one row per (source, midi) pair.
    Embeddings are cached to <data_dir>/embeddings_<model>.npz.
    """
    cache = data_dir / f"embeddings_{model_name}.npz"
    if use_cache and cache.exists():
        print(f"  Loading cached embeddings from {cache.name}")
        z = np.load(cache, allow_pickle=True)
        return z["X"], list(z["midi_labels"]), list(z["source_labels"])

    print(f"  Collecting {len(conds)} embeddings …")

    # Phase 1 (no audio gen needed — stimuli already on disk).
    # Phase 2: dispatch HTTP queries with bounded concurrency.
    def _query_one(c: dict) -> dict | None:
        vec = query_alm(model_name, c["wav"], mode="embed")["embedding"]
        return {"vec": vec, "midi": c["midi"], "source": c["source"]}

    raw_results = dispatch(
        conds, _query_one,
        model_name=model_name,
        label_fn=lambda c: f"{c['source']:10s} midi={c['midi']:>3}",
    )
    # Note: NotImplementedError raised inside the worker propagates through the
    # dispatcher's exception handling (recorded as None and traceback printed).
    # Other failures also become None and are silently dropped.

    vecs: list[list[float]] = []
    midi_labels: list[int] = []
    source_labels: list[str] = []
    for r in raw_results:
        if r is None:
            continue
        vecs.append(r["vec"])
        midi_labels.append(r["midi"])
        source_labels.append(r["source"])

    if not vecs:
        raise RuntimeError(
            f"All {len(conds)} embedding requests failed for {model_name}. "
            "Check the model server logs."
        )

    X = np.array(vecs, dtype=np.float32)
    np.savez_compressed(
        cache,
        X=X,
        midi_labels=np.array(midi_labels),
        source_labels=np.array(source_labels, dtype=object),
    )
    print(f"  Saved embeddings → {cache.name}  shape={X.shape}")
    return X, midi_labels, source_labels


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Return N×N cosine similarity matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xn = X / norms
    return Xn @ Xn.T


def analysis_linear_probe(X: np.ndarray, midi_labels: list[int]) -> dict:
    """5-fold cross-validated LinearSVC on MIDI class prediction."""
    if not _SKLEARN:
        return {"error": "scikit-learn not installed"}
    if len(set(midi_labels)) < 2:
        return {"error": "fewer than 2 unique classes"}

    y = np.array(midi_labels)
    n_classes = len(set(midi_labels))
    chance = 1.0 / n_classes

    min_class_count = int(min(np.bincount(y - y.min())))
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        return {"error": f"too few samples per class for CV (min={min_class_count})"}

    pipe = make_pipeline(
        StandardScaler(),
        LinearSVC(max_iter=5000, C=0.1),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy":  float(np.mean(scores)),
        "std_accuracy":   float(np.std(scores)),
        "chance_accuracy": round(chance, 4),
        "lift_over_chance": float(np.mean(scores) / chance),
        "cv_scores":      [round(float(s), 4) for s in scores],
        "n_splits":       n_splits,
    }


def analysis_knn(
    X: np.ndarray,
    midi_labels: list[int],
    source_labels: list[str],
) -> dict:
    """
    Leave-one-out k-NN (k=1) using cosine similarity.
    Reports:
      - overall:       all pairs (excluding self)
      - within_source: NN restricted to same source
      - cross_source:  NN restricted to different sources
    """
    n = len(midi_labels)
    y = np.array(midi_labels)
    s = np.array(source_labels)

    sim = _cosine_sim_matrix(X)           # N×N
    np.fill_diagonal(sim, -2.0)           # exclude self

    results: dict[str, dict] = {}

    def _knn_acc(mask: np.ndarray, label: str) -> dict:
        correct = within1 = within_oct = total = 0
        for i in range(n):
            row_mask = mask[i]
            if not row_mask.any():
                continue
            row = np.where(row_mask, sim[i], -3.0)
            j = int(np.argmax(row))
            dist = abs(int(y[i]) - int(y[j]))
            correct   += int(dist == 0)
            within1   += int(dist <= 1)
            within_oct += int(dist % 12 == 0)
            total += 1
        return {
            "exact_match": round(correct / total, 4) if total else None,
            "within_1st":  round(within1  / total, 4) if total else None,
            "within_octave": round(within_oct / total, 4) if total else None,
            "n": total,
        }

    overall_mask   = np.ones((n, n), dtype=bool)
    np.fill_diagonal(overall_mask, False)
    results["overall"] = _knn_acc(overall_mask, "overall")

    within_mask = np.equal.outer(s, s) & ~np.eye(n, dtype=bool)
    results["within_source"] = _knn_acc(within_mask, "within_source")

    cross_mask = ~np.equal.outer(s, s)
    results["cross_source"] = _knn_acc(cross_mask, "cross_source")

    # Per-source breakdown
    per_source: dict[str, dict] = {}
    for src in sorted(set(source_labels)):
        src_idx = np.where(s == src)[0]
        if len(src_idx) < 2:
            continue
        mask_src = np.zeros((n, n), dtype=bool)
        mask_src[np.ix_(src_idx, src_idx)] = True
        np.fill_diagonal(mask_src, False)
        per_source[src] = _knn_acc(mask_src, src)

    results["per_source"] = per_source
    return results


def analysis_cosine_vs_interval(
    X: np.ndarray,
    midi_labels: list[int],
) -> dict:
    """
    Average cosine similarity as a function of semitone distance (0–48).
    Computes all upper-triangle pairs.
    """
    sim = _cosine_sim_matrix(X)
    y   = np.array(midi_labels)

    bins: dict[int, list[float]] = {d: [] for d in range(49)}
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(int(y[i]) - int(y[j]))
            bins[d].append(float(sim[i, j]))

    return {
        str(d): {
            "mean": round(float(np.mean(v)), 6) if v else None,
            "std":  round(float(np.std(v)),  6) if v else None,
            "n":    len(v),
        }
        for d, v in bins.items() if v
    }


def analysis_octave_periodicity(cosine_vs_interval: dict) -> dict:
    """
    Compare average cosine similarity at octave intervals (12, 24, 36, 48)
    vs. non-octave intervals.  Tests the hypothesis that embeddings are
    octave-periodic.
    """
    octave_sims = []
    non_octave_sims = []

    for d_str, stats in cosine_vs_interval.items():
        d = int(d_str)
        if stats["mean"] is None:
            continue
        if d > 0 and d % 12 == 0:
            octave_sims.append(stats["mean"])
        elif d > 0:
            non_octave_sims.append(stats["mean"])

    sim_at_0  = cosine_vs_interval.get("0", {}).get("mean")
    sim_at_12 = cosine_vs_interval.get("12", {}).get("mean")

    return {
        "sim_at_same_pitch":    sim_at_0,
        "sim_at_12st":          sim_at_12,
        "mean_octave_sims":     round(float(np.mean(octave_sims)),     6) if octave_sims     else None,
        "mean_non_octave_sims": round(float(np.mean(non_octave_sims)), 6) if non_octave_sims else None,
        "octave_lift":          round(
            float(np.mean(octave_sims)) - float(np.mean(non_octave_sims)), 6
        ) if octave_sims and non_octave_sims else None,
    }


def analysis_pca(
    X: np.ndarray,
    midi_labels: list[int],
    source_labels: list[str],
) -> dict:
    """PCA 2-D projection; return component variance ratios."""
    if not _SKLEARN:
        return {"error": "scikit-learn not installed"}
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(10, X.shape[1], X.shape[0]))
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    return {
        "variance_ratio_pc1": round(float(evr[0]), 4),
        "variance_ratio_pc2": round(float(evr[1]), 4) if len(evr) > 1 else None,
        "variance_ratio_top5": round(float(evr[:5].sum()), 4),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_pca_plots(
    X: np.ndarray,
    midi_labels: list[int],
    source_labels: list[str],
    run_dir: Path,
    model_name: str,
) -> None:
    if not _MPL or not _SKLEARN:
        return
    from sklearn.decomposition import PCA

    pca  = PCA(n_components=2)
    X2   = pca.fit_transform(X)
    evr  = pca.explained_variance_ratio_

    # ── by MIDI pitch ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X2[:, 0], X2[:, 1],
        c=midi_labels, cmap="plasma", s=18, alpha=0.7,
    )
    plt.colorbar(scatter, ax=ax, label="MIDI note")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    ax.set_title(f"Embedding PCA — by MIDI pitch\n{model_name}")
    fig.tight_layout()
    fig.savefig(run_dir / f"pca_by_pitch_{model_name}.png", dpi=120)
    plt.close(fig)

    # ── by source ─────────────────────────────────────────────────────────────
    sources = sorted(set(source_labels))
    colours = plt.get_cmap("tab20", len(sources))
    src_idx = {s: i for i, s in enumerate(sources)}
    fig, ax = plt.subplots(figsize=(10, 7))
    for src in sources:
        mask = [i for i, s in enumerate(source_labels) if s == src]
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   color=colours(src_idx[src]), label=src, s=18, alpha=0.7)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    ax.set_title(f"Embedding PCA — by source\n{model_name}")
    fig.tight_layout()
    fig.savefig(run_dir / f"pca_by_source_{model_name}.png", dpi=120)
    plt.close(fig)

    # ── UMAP (optional) ───────────────────────────────────────────────────────
    if _UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        Xu = reducer.fit_transform(X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        s1 = ax1.scatter(Xu[:, 0], Xu[:, 1], c=midi_labels, cmap="plasma", s=18, alpha=0.7)
        plt.colorbar(s1, ax=ax1, label="MIDI note")
        ax1.set_title(f"UMAP — by MIDI pitch\n{model_name}")

        for src in sources:
            mask = [i for i, s in enumerate(source_labels) if s == src]
            ax2.scatter(Xu[mask, 0], Xu[mask, 1],
                        color=colours(src_idx[src]), label=src, s=18, alpha=0.7)
        ax2.legend(loc="upper right", fontsize=7, ncol=2)
        ax2.set_title(f"UMAP — by source\n{model_name}")

        fig.tight_layout()
        fig.savefig(run_dir / f"umap_{model_name}.png", dpi=120)
        plt.close(fig)


def _save_cosine_heatmap(
    X: np.ndarray,
    midi_labels: list[int],
    source_labels: list[str],
    run_dir: Path,
    model_name: str,
) -> None:
    if not _MPL:
        return

    # Average embedding per MIDI pitch
    unique_midi = sorted(set(midi_labels))
    avg_embs: list[np.ndarray] = []
    for m in unique_midi:
        idx = [i for i, lbl in enumerate(midi_labels) if lbl == m]
        avg_embs.append(X[idx].mean(axis=0))

    Xavg = np.stack(avg_embs)
    sim  = _cosine_sim_matrix(Xavg)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, aspect="auto", origin="upper", cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    tick_step = 6
    ticks = list(range(0, len(unique_midi), tick_step))
    tick_lbls = [str(unique_midi[t]) for t in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(tick_lbls, fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(tick_lbls, fontsize=8)
    ax.set_xlabel("MIDI note"); ax.set_ylabel("MIDI note")
    ax.set_title(f"Avg cosine similarity between pitches\n{model_name}")
    fig.tight_layout()
    fig.savefig(run_dir / f"cosine_heatmap_{model_name}.png", dpi=120)
    plt.close(fig)


def _save_cosine_vs_interval_plot(
    cosine_by_interval: dict,
    run_dir: Path,
    model_name: str,
) -> None:
    if not _MPL:
        return

    dists = sorted(int(d) for d in cosine_by_interval)
    means = [cosine_by_interval[str(d)]["mean"] for d in dists if cosine_by_interval[str(d)]["mean"] is not None]
    dists_valid = [d for d in dists if cosine_by_interval[str(d)]["mean"] is not None]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dists_valid, means, marker="o", markersize=4, linewidth=1.5, color="steelblue")

    # Highlight octave intervals
    for oct_d in [12, 24, 36, 48]:
        if oct_d in dists_valid:
            idx = dists_valid.index(oct_d)
            ax.axvline(x=oct_d, color="tomato", linestyle="--", alpha=0.5, linewidth=1)
            ax.scatter([oct_d], [means[idx]], color="tomato", zorder=5, s=60)

    ax.set_xlabel("Semitone distance")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title(f"Cosine similarity vs. semitone distance\n{model_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / f"cosine_vs_interval_{model_name}.png", dpi=120)
    plt.close(fig)


def _save_knn_accuracy_plot(
    knn_results: dict,
    run_dir: Path,
    model_name: str,
) -> None:
    if not _MPL:
        return
    per_source = knn_results.get("per_source", {})
    if not per_source:
        return

    sources = sorted(per_source)
    accs    = [per_source[s].get("exact_match") or 0.0 for s in sources]

    fig, ax = plt.subplots(figsize=(max(8, len(sources) * 0.6), 5))
    bars = ax.bar(sources, accs, color="steelblue", edgecolor="white", width=0.7)

    overall = knn_results.get("within_source", {}).get("exact_match")
    if overall is not None:
        ax.axhline(overall, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"within-source avg ({overall:.1%})")
        ax.legend(fontsize=9)

    ax.set_ylim(0, 1)
    ax.set_ylabel("k-NN accuracy (exact MIDI)")
    ax.set_xlabel("Source")
    ax.set_title(f"Leave-one-out k=1 NN accuracy per source\n{model_name}")
    plt.xticks(rotation=45, ha="right", fontsize=9)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / f"knn_accuracy_{model_name}.png", dpi=120)
    plt.close(fig)


# ── Run one model ─────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    conds: list[dict],
    data_dir: Path,
    run_dir: Path,
    use_cache: bool = True,
) -> dict:
    info = get_model_info(model_name)
    print(f"\n  Model : {config.MODELS.get(model_name, model_name)}")

    X, midi_labels, source_labels = collect_embeddings(
        model_name, conds, data_dir, use_cache=use_cache
    )
    print(f"  Embeddings shape : {X.shape}  "
          f"({len(set(source_labels))} sources × {len(set(midi_labels))} pitches)")

    # ── Run analyses ──────────────────────────────────────────────────────────
    print("  Running: linear probe …")
    probe = analysis_linear_probe(X, midi_labels)

    print("  Running: k-NN retrieval …")
    knn = analysis_knn(X, midi_labels, source_labels)

    print("  Running: cosine similarity vs. interval …")
    cosine_by_iv = analysis_cosine_vs_interval(X, midi_labels)

    print("  Running: octave periodicity …")
    octave = analysis_octave_periodicity(cosine_by_iv)

    print("  Running: PCA …")
    pca_stats = analysis_pca(X, midi_labels, source_labels)

    summary = {
        "n_embeddings":         X.shape[0],
        "embedding_dim":        X.shape[1],
        "n_sources":            len(set(source_labels)),
        "n_pitches":            len(set(midi_labels)),
        "linear_probe":         probe,
        "knn":                  knn,
        "cosine_by_interval":   cosine_by_iv,
        "octave_periodicity":   octave,
        "pca":                  pca_stats,
    }

    # ── Summary text ──────────────────────────────────────────────────────────
    n_classes = len(set(midi_labels))
    probe_acc  = probe.get("mean_accuracy")
    probe_str  = f"{probe_acc:.1%} ± {probe.get('std_accuracy', 0):.1%}" if probe_acc is not None else "N/A"

    summary_lines = [
        f"  Embeddings : {X.shape[0]}  ({len(set(source_labels))} sources × {n_classes} pitches)",
        f"  Embed dim  : {X.shape[1]}",
        "",
        f"  Linear probe ({probe.get('n_splits', 5)}-fold CV, LinearSVC, MIDI class):",
        f"    Accuracy     : {probe_str}",
        f"    Chance       : {probe.get('chance_accuracy', 1/n_classes):.1%}",
        f"    Lift×chance  : {probe.get('lift_over_chance', 0):.1f}×",
        "",
        "  k-NN (leave-one-out, cosine, k=1):",
    ]
    for cond_name, cond_key in [("overall", "overall"), ("within-source", "within_source"), ("cross-source", "cross_source")]:
        d = knn.get(cond_key, {})
        if d.get("exact_match") is not None:
            summary_lines.append(
                f"    {cond_name:14s}: {d['exact_match']:.1%} exact  "
                f"| ±1st: {d.get('within_1st', 0):.1%}"
                f"  | octave: {d.get('within_octave', 0):.1%}"
            )

    summary_lines += [
        "",
        "  Octave periodicity:",
        f"    same pitch  : {octave.get('sim_at_same_pitch', 'N/A')}",
        f"    +12 semitones: {octave.get('sim_at_12st', 'N/A')}",
        f"    mean octave intervals : {octave.get('mean_octave_sims', 'N/A')}",
        f"    mean non-octave intervals: {octave.get('mean_non_octave_sims', 'N/A')}",
        f"    octave lift : {octave.get('octave_lift', 'N/A')}",
        "",
        "  PCA variance:",
        f"    PC1: {pca_stats.get('variance_ratio_pc1', 'N/A')}  "
        f"PC2: {pca_stats.get('variance_ratio_pc2', 'N/A')}  "
        f"top-5: {pca_stats.get('variance_ratio_top5', 'N/A')}",
    ]

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {model_name}")
    for line in summary_lines:
        print(line)

    # ── Save plots ────────────────────────────────────────────────────────────
    _save_pca_plots(X, midi_labels, source_labels, run_dir, model_name)
    _save_cosine_heatmap(X, midi_labels, source_labels, run_dir, model_name)
    _save_cosine_vs_interval_plot(cosine_by_iv, run_dir, model_name)
    _save_knn_accuracy_plot(knn, run_dir, model_name)

    # ── Save JSON + TXT ───────────────────────────────────────────────────────
    stem = f"results_{_safe_stem(model_name)}"
    metadata = get_run_metadata(
        model_name=model_name, model_info=info,
        n_sources=len(set(source_labels)),
        sources=sorted(set(source_labels)),
        midi_range=[MIDI_MIN, MIDI_MAX],
        embedding_shape=list(X.shape),
        use_sklearn=_SKLEARN,
        use_umap=_UMAP,
    )
    payload = {"metadata": metadata, "summary": summary}
    (run_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    txt_lines = [
        f"Experiment : {EXP_NAME}",
        f"Model      : {model_name}",
    ] + [f"{k:10} : {v}" for k, v in metadata.items()] + ["", "SUMMARY", "=" * 50] + summary_lines
    (run_dir / f"{stem}.txt").write_text("\n".join(txt_lines) + "\n")
    print(f"\nResults saved → {run_dir}/{stem}.*")

    return {
        "probe_accuracy":         probe.get("mean_accuracy"),
        "probe_lift":             probe.get("lift_over_chance"),
        "knn_within_source":      knn.get("within_source", {}).get("exact_match"),
        "knn_cross_source":       knn.get("cross_source",  {}).get("exact_match"),
        "octave_lift":            octave.get("octave_lift"),
        "pca_variance_pc1":       pca_stats.get("variance_ratio_pc1"),
    }


# ── Entry points ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview",   action="store_true")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Re-query embeddings even if cache file exists")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help=f"Model slugs (default: all). Available: {list(config.MODELS)}")
    args, _ = parser.parse_known_args()
    return args


def preview() -> None:
    stimuli_root, sources = _discover_stimuli_dir()
    conds = build_conditions(stimuli_root, sources)
    print(f"Experiment   : {EXP_NAME}")
    print(f"Stimuli root : {stimuli_root}  ({len(sources)} sources)")
    print(f"Sources      : {sources}")
    print(f"MIDI range   : {MIDI_MIN}–{MIDI_MAX}  ({len(MIDI_PITCHES)} pitches)")
    print(f"Conditions   : {len(conds)}  (source × pitch pairs)")
    print(f"scikit-learn : {'available' if _SKLEARN else 'NOT installed (pip install scikit-learn)'}")
    print(f"matplotlib   : {'available' if _MPL     else 'NOT installed (pip install matplotlib)'}")
    print(f"umap-learn   : {'available' if _UMAP    else 'not installed (optional: pip install umap-learn)'}")
    print("\nRun without --preview to query the model(s).")


def run() -> dict:
    args = _parse_args()
    target_models = args.models or list(config.MODELS)
    stimuli_root, sources = _discover_stimuli_dir()
    conds    = build_conditions(stimuli_root, sources)
    data_dir = exp_data_dir(EXP_NAME)

    print(f"Experiment : {EXP_NAME}")
    print(f"Models     : {', '.join(target_models)}")
    print(f"Stimuli    : {len(conds)} (source × pitch pairs, {len(sources)} sources)")

    run_dir = make_run_dir(EXP_NAME)
    all_summaries: dict[str, dict] = {}
    for model_name in target_models:
        try:
            all_summaries[model_name] = run_one_model(
                model_name, conds, data_dir, run_dir,
                use_cache=not args.no_cache,
            )
        except (NotImplementedError, RuntimeError) as exc:
            print(f"\n  [SKIP] {model_name}: {exc}")
    save_comparison(run_dir, all_summaries, EXP_NAME)
    return extract_format_accuracies(run_dir, list(all_summaries.keys()))

if __name__ == "__main__":
    args = _parse_args()
    if args.preview:
        preview()
    else:
        run()
