"""Sweep LOCAL_CONCURRENCY for a local model server (default: AF-Next instruct).

Fires a fixed number of /analyze/upload requests at a range of concurrency
levels and reports throughput (req/s) and latency percentiles (p50/p95/p99).

Usage:
    python tools/bench_local_concurrency.py
    python tools/bench_local_concurrency.py --url http://localhost:8001 \
        --levels 1 2 3 4 6 8 --requests 12 --max-new-tokens 128
"""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def one_request(url: str, audio_path: Path, prompt: str, max_new_tokens: int,
                timeout: int) -> tuple[float, bool, str]:
    t0 = time.perf_counter()
    try:
        with audio_path.open("rb") as f:
            r = requests.post(
                f"{url}/analyze/upload",
                data={"prompt": prompt, "max_new_tokens": str(max_new_tokens)},
                files={"file": (audio_path.name, f, "audio/wav")},
                timeout=timeout,
            )
        ok = r.status_code == 200
        body = r.text[:200] if not ok else ""
    except Exception as exc:  # noqa: BLE001
        ok = False
        body = f"{type(exc).__name__}: {exc}"
    return time.perf_counter() - t0, ok, body


def warmup(url: str, audio_path: Path, prompt: str, max_new_tokens: int, timeout: int) -> None:
    print("[warmup] one request to prime CUDA graphs / caches…", flush=True)
    dt, ok, err = one_request(url, audio_path, prompt, max_new_tokens, timeout)
    print(f"[warmup] {dt:.2f}s ok={ok} {err}", flush=True)


def run_level(url: str, audio_paths: list[Path], prompt: str, max_new_tokens: int,
              concurrency: int, n_requests: int, timeout: int) -> dict:
    latencies: list[float] = []
    failures: list[str] = []
    wall_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(
                one_request,
                url,
                audio_paths[i % len(audio_paths)],
                prompt,
                max_new_tokens,
                timeout,
            )
            for i in range(n_requests)
        ]
        for fut in as_completed(futs):
            dt, ok, err = fut.result()
            if ok:
                latencies.append(dt)
            else:
                failures.append(err)
    wall = time.perf_counter() - wall_t0
    n_ok = len(latencies)
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_ok": n_ok,
        "n_fail": len(failures),
        "wall_s": wall,
        "throughput_rps": (n_ok / wall) if wall > 0 else 0.0,
        "p50_s": percentile(latencies, 0.50),
        "p95_s": percentile(latencies, 0.95),
        "p99_s": percentile(latencies, 0.99),
        "mean_s": statistics.fmean(latencies) if latencies else float("nan"),
        "max_s": max(latencies) if latencies else float("nan"),
        "first_failure": failures[0] if failures else "",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8001")
    p.add_argument(
        "--audio-dir",
        default="data/audio/pitchbench_a1_single_pitch_id",
        help="Directory of .wav files to cycle through",
    )
    p.add_argument("--prompt", default="What pitch is this? Answer with one note name.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--requests", type=int, default=12,
                   help="Number of requests per concurrency level")
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    audio_dir = Path(args.audio_dir)
    audio_paths = sorted(audio_dir.glob("*.wav"))
    if not audio_paths:
        raise SystemExit(f"No .wav files in {audio_dir}")
    audio_paths = audio_paths[:8]
    print(f"Using {len(audio_paths)} audio files from {audio_dir}")

    health = requests.get(f"{args.url}/health", timeout=5).json()
    print(f"Server: {health}")

    warmup(args.url, audio_paths[0], args.prompt, args.max_new_tokens, args.timeout)

    rows = []
    for c in args.levels:
        print(f"\n=== concurrency={c}, n_requests={args.requests} ===", flush=True)
        row = run_level(args.url, audio_paths, args.prompt, args.max_new_tokens,
                        c, args.requests, args.timeout)
        rows.append(row)
        print(
            f"  ok={row['n_ok']}/{row['n_requests']} fail={row['n_fail']} "
            f"wall={row['wall_s']:.2f}s  thr={row['throughput_rps']:.2f} req/s  "
            f"p50={row['p50_s']:.2f}s  p95={row['p95_s']:.2f}s  p99={row['p99_s']:.2f}s  "
            f"max={row['max_s']:.2f}s",
            flush=True,
        )
        if row["n_fail"]:
            print(f"  first failure: {row['first_failure']}", flush=True)

    print("\n\n===== SUMMARY =====")
    hdr = f"{'conc':>4} {'ok':>4} {'wall_s':>8} {'thr_rps':>8} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['concurrency']:>4} {r['n_ok']:>4} {r['wall_s']:>8.2f} "
            f"{r['throughput_rps']:>8.2f} {r['p50_s']:>7.2f} {r['p95_s']:>7.2f} "
            f"{r['p99_s']:>7.2f} {r['max_s']:>7.2f}"
        )

    best_thr = max(rows, key=lambda r: r["throughput_rps"])
    print(
        f"\nBest throughput: concurrency={best_thr['concurrency']} "
        f"@ {best_thr['throughput_rps']:.2f} req/s "
        f"(p99={best_thr['p99_s']:.2f}s)"
    )


if __name__ == "__main__":
    main()
