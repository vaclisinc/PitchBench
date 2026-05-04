"""
Launch all PitchBench model servers — one process per model, each on its own port.

Ports come from config.MODEL_URLS so there is no duplication.
Edit SERVERS below to choose which script backs each model slug.

Usage:
    python model/serve_all.py                    # start all 4 servers
    python model/serve_all.py --only audio_flamingo_3 audio_flamingo_next_instruct
    python model/serve_all.py --list             # list configs and exit

Each server's stdout/stderr is prefixed with its name in a distinct colour.
Press Ctrl+C to stop all servers cleanly.
"""

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

# ── Server definitions ────────────────────────────────────────────────────────
# Edit this list to change which script runs for each model slug.
# PORT is injected automatically from config.MODEL_URLS — do not set it here.
# Any key in `env` overrides the corresponding environment variable.

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import pitchbench.config as config

SERVERS: list[dict] = [
    {
        "slug":   "music_flamingo",
        "script": ROOT / "model" / "api_music_fl.py",
        "env":    {},
    },
    {
        "slug":   "audio_flamingo_next_instruct",
        "script": ROOT / "model" / "api_audio_fl_next.py",
        "env":    {"AF_NEXT_CHECKPOINT": "instruct"},
    },
    {
        "slug":   "audio_flamingo_next_think",
        "script": ROOT / "model" / "api_audio_fl_next.py",
        "env":    {"AF_NEXT_CHECKPOINT": "think"},
    },
    {
        "slug":   "audio_flamingo_next_captioner",
        "script": ROOT / "model" / "api_audio_fl_next.py",
        "env":    {"AF_NEXT_CHECKPOINT": "captioner"},
    },
    {
        "slug":   "qwen3_omni",
        "script": ROOT / "model" / "api_qwen3_omni.py",
        "env":    {},
    },
]

# ANSI colours for each server (cycles if more than 6)
_COLOURS = ["32", "33", "34", "35", "36", "96"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stream(proc, prefix: str, colour: str) -> None:
    """Read lines from proc.stdout and print them with a coloured prefix."""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        sys.stdout.write(f"\033[{colour}m[{prefix}]\033[0m {line}\n")
        sys.stdout.flush()


def _resolve_servers(only: list[str] | None) -> list[dict]:
    slugs  = {s["slug"] for s in SERVERS}
    target = set(only) if only else slugs

    unknown = target - slugs
    if unknown:
        sys.exit(f"Unknown model slug(s): {unknown}\nAvailable: {sorted(slugs)}")

    return [s for s in SERVERS if s["slug"] in target]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", metavar="SLUG",
        help="Start only these model servers (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print server configs and exit",
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'Slug':40s}  {'Port':6s}  Script")
        for s in SERVERS:
            port = config.MODEL_URLS.get(s["slug"], "???").rsplit(":", 1)[-1]
            print(f"  {s['slug']:38s}  {port:6s}  {Path(s['script']).name}  {s['env']}")
        return

    active = _resolve_servers(args.only)

    import subprocess
    procs: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []

    print(f"Starting {len(active)} model server(s)…\n")

    for idx, srv in enumerate(active):
        port_str = config.MODEL_URLS.get(srv["slug"], "http://localhost:8000").rsplit(":", 1)[-1]
        colour   = _COLOURS[idx % len(_COLOURS)]
        label    = srv["slug"].replace("audio_flamingo_", "af_")

        env = {**os.environ, "PORT": port_str, **srv["env"]}

        proc = subprocess.Popen(
            [sys.executable, str(srv["script"])],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
        procs.append(proc)

        t = threading.Thread(target=_stream, args=(proc, label, colour), daemon=True)
        t.start()
        threads.append(t)

        print(f"\033[{colour}m[{label}]\033[0m  PID {proc.pid}  port {port_str}  "
              f"→ {Path(srv['script']).name}  {srv['env']}")

    print()

    def _shutdown(signum=None, frame=None) -> None:
        print("\n\nStopping all servers…")
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # wait for any server to exit (abnormal) or forever (normal)
    reported: set[int] = set()
    while True:
        for p in procs:
            if p.poll() is not None and p.returncode != 0 and id(p) not in reported:
                reported.add(id(p))
                slug = active[procs.index(p)]["slug"]
                print(f"\n[serve_all] ⚠  {slug} exited with code {p.returncode}")
        if all(p.poll() is not None for p in procs):
            break
        threading.Event().wait(timeout=2)


if __name__ == "__main__":
    main()
