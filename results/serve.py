#!/usr/bin/env python3
"""PitchBench results viewer server.

Usage:
    python results/serve.py          # serves on http://localhost:7777
    python results/serve.py 8888     # custom port
"""
import json
import os
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

RESULTS_DIR = Path(__file__).parent
REPO_ROOT = RESULTS_DIR.parent


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def do_HEAD(self):
        self.do_GET()  # handle HEAD same as GET (for audio preload probes)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_file(RESULTS_DIR / "viewer.html", "text/html")
        elif path == "/api/structure":
            self._serve_json(self._build_structure())
        elif path == "/api/results":
            rel = query.get("file", [None])[0]
            if rel:
                full = REPO_ROOT / rel
                mime = {"json": "application/json", "png": "image/png",
                        "txt": "text/plain", "csv": "text/csv"}.get(full.suffix.lstrip("."), "")
                if mime and full.is_file() and _safe_path(full):
                    self._serve_file(full, mime)
                else:
                    self._error(404, "Not found")
            else:
                self._error(400, "Missing file param")
        elif path == "/audio":
            audio_path = query.get("path", [None])[0]
            if audio_path:
                full = Path(audio_path)
                if full.suffix in (".wav", ".mp3", ".flac", ".ogg") and full.is_file():
                    self._serve_file(full, "audio/wav")
                else:
                    self._error(404, "Audio not found")
            else:
                self._error(400, "Missing path param")
        else:
            self._error(404, "Not found")

    def _serve_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._error(500, str(e))

    def _serve_json(self, obj):
        data = json.dumps(obj, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, msg: str):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(msg.encode())

    def _build_structure(self):
        experiments = {}
        for exp_dir in sorted(RESULTS_DIR.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            runs = []
            for run_dir in sorted(exp_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                result_files = sorted(run_dir.glob("results_*.json"))
                if not result_files:
                    continue
                models = []
                for rf in result_files:
                    rel = str(rf.relative_to(REPO_ROOT))
                    model_name = rf.stem.replace("results_", "")
                    png_files = [
                        str(p.relative_to(REPO_ROOT))
                        for p in sorted(run_dir.glob("*.png"))
                    ]
                    models.append({
                        "model": model_name,
                        "file": rel,
                        "pngs": png_files,
                    })
                runs.append({"run": run_dir.name, "models": models})
            if runs:
                experiments[exp_dir.name] = runs
        return experiments


def _safe_path(p: Path) -> bool:
    try:
        p.resolve().relative_to(REPO_ROOT.resolve())
        return True
    except ValueError:
        return False


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7777
    server = ThreadedHTTPServer(("", port), Handler)
    print(f"PitchBench viewer → http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
