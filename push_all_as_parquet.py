"""Convert all PitchBench staging configs to Parquet and push to HF.

Replaces the audiofolder layout (WAV files + metadata.jsonl) with a pure-Parquet
layout.  This lets HF use its standard Parquet reader for every config — including
pitchbench_d7b_pitch_with_reference_split, which has two Audio columns (audio_1 for
the reference tone and audio_2 for the target tone) and therefore cannot be served by
the audiofolder builder.

d7b is skipped: its Parquet is already on HF.

Usage:
    python push_all_as_parquet.py                    # convert + push all
    python push_all_as_parquet.py --dry-run          # build parquets locally, no push
    python push_all_as_parquet.py --skip-generate    # skip local build, only push/delete
    python push_all_as_parquet.py --config pitchbench_a1_single_pitch_id
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitOperationAdd, CommitOperationDelete

_D7B = "pitchbench_d7b_pitch_with_reference_split"
_REPO = "pitchbench-authors/PitchBench"
_PARQUET_NAME = "test-00000-of-00001.parquet"

_AUDIO_TYPE = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])


# ──────────────────────────────────────── type helpers (PyArrow + HF meta) ──

def _arrow_type(v) -> pa.DataType:
    if isinstance(v, bool):
        return pa.bool_()
    if isinstance(v, int):
        return pa.int64()
    if isinstance(v, float):
        return pa.float64()
    if isinstance(v, list) and v:
        elem = v[0]
        if isinstance(elem, bool):
            return pa.list_(pa.bool_())
        if isinstance(elem, int):
            return pa.list_(pa.int64())
        if isinstance(elem, float):
            return pa.list_(pa.float64())
        return pa.list_(pa.string())
    if isinstance(v, list):
        return pa.list_(pa.string())
    return pa.string()


def _hf_type(v) -> dict:
    if isinstance(v, bool):
        return {"dtype": "bool", "_type": "Value"}
    if isinstance(v, int):
        return {"dtype": "int64", "_type": "Value"}
    if isinstance(v, float):
        return {"dtype": "float64", "_type": "Value"}
    if isinstance(v, list) and v:
        elem = v[0]
        inner = (
            {"dtype": "bool", "_type": "Value"} if isinstance(elem, bool)
            else {"dtype": "int64", "_type": "Value"} if isinstance(elem, int)
            else {"dtype": "float64", "_type": "Value"} if isinstance(elem, float)
            else {"dtype": "string", "_type": "Value"}
        )
        return {"feature": inner, "_type": "Sequence"}
    if isinstance(v, list):
        return {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"}
    return {"dtype": "string", "_type": "Value"}


# ─────────────────────────────────────────────────── local Parquet builder ──

def build_parquet(config_name: str, staging: Path) -> Path:
    """Load WAV + metadata.jsonl from staging; write a Parquet with HF Audio metadata."""
    config_dir = staging / config_name
    meta_path = config_dir / "metadata.jsonl"
    out_path = config_dir / _PARQUET_NAME

    rows_meta = [json.loads(line) for line in meta_path.open(encoding="utf-8")]

    audio_bytes_list: list[bytes] = []
    audio_paths_list: list[str] = []
    other_cols: dict[str, list] = {}

    for row in rows_meta:
        fname = row.pop("file_name")
        with open(config_dir / fname, "rb") as fh:
            audio_bytes_list.append(fh.read())
        audio_paths_list.append(fname)
        for k, v in row.items():
            other_cols.setdefault(k, []).append(v)

    # Audio struct column
    audio_array = pa.array(
        [{"bytes": b, "path": p} for b, p in zip(audio_bytes_list, audio_paths_list)],
        type=_AUDIO_TYPE,
    )

    # Other columns
    other_arrays: dict[str, pa.Array] = {}
    for k, vals in other_cols.items():
        v0 = vals[0] if vals else None
        try:
            other_arrays[k] = pa.array(vals, type=_arrow_type(v0))
        except Exception:
            other_arrays[k] = pa.array(
                [str(v) if v is not None else None for v in vals], type=pa.string()
            )

    # HF feature metadata
    features_meta: dict = {"audio": {"sampling_rate": 16000, "_type": "Audio"}}
    for k, vals in other_cols.items():
        features_meta[k] = _hf_type(vals[0] if vals else None)
    hf_meta = json.dumps({"info": {"features": features_meta}}).encode()

    # Build schema + table
    fields = [pa.field("audio", _AUDIO_TYPE)]
    for k, arr in other_arrays.items():
        fields.append(pa.field(k, arr.type))
    schema = pa.schema(fields, metadata={b"huggingface": hf_meta})
    table = pa.table(
        {"audio": audio_array, **other_arrays},
        schema=schema,
    )
    pq.write_table(table, str(out_path))
    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(table)} rows)")
    return out_path


# ──────────────────────────────────────────────────────────── HF push logic ──

def _repo_files_for_config(all_hf_files: list[str], config_name: str) -> list[str]:
    return [f for f in all_hf_files if f.startswith(f"{config_name}/")]


def push_config(
    config_name: str,
    staging: Path,
    all_hf_files: list[str],
    api: HfApi,
    dry_run: bool,
) -> None:
    parquet_path = staging / config_name / _PARQUET_NAME
    repo_parquet = f"{config_name}/{_PARQUET_NAME}"

    hf_config_files = _repo_files_for_config(all_hf_files, config_name)
    to_delete = [
        f for f in hf_config_files
        if f.endswith(".wav") or f.endswith(".jsonl")
    ]

    print(f"  Parquet: {parquet_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Deleting {len(to_delete)} files from HF")

    if dry_run:
        print("  [dry-run] skipping commit")
        return

    operations: list = [
        CommitOperationAdd(
            path_in_repo=repo_parquet,
            path_or_fileobj=str(parquet_path),
        )
    ]
    for f in to_delete:
        operations.append(CommitOperationDelete(path_in_repo=f))

    api.create_commit(
        repo_id=_REPO,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Convert {config_name} to Parquet",
    )
    print(f"  Committed to HF.")


# ──────────────────────────────────────────────────────────────────── main ──

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--staging", type=Path, default=Path("pitchbench_hf"))
    p.add_argument("--repo", default=_REPO)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip local Parquet generation (use existing files).")
    p.add_argument("--config", default=None,
                   help="Process only this config (default: all).")
    p.add_argument("--token", default=None)
    args = p.parse_args()

    staging = args.staging
    if not staging.exists():
        raise SystemExit(f"Staging dir {staging} not found.")

    # Discover all configs (non-d7b, have metadata.jsonl)
    all_configs = sorted(
        d.name for d in staging.iterdir()
        if d.is_dir()
        and d.name != _D7B
        and (d / "metadata.jsonl").exists()
    )
    if args.config:
        if args.config not in all_configs:
            raise SystemExit(f"Config {args.config!r} not found in staging.")
        all_configs = [args.config]

    print(f"Processing {len(all_configs)} configs.")

    # Step 1: Generate local Parquet files
    if not args.skip_generate:
        print("\n── Generating Parquet files ──")
        for cfg in all_configs:
            out = staging / cfg / _PARQUET_NAME
            if out.exists():
                print(f"[skip] {cfg} — parquet already exists ({out.stat().st_size/1e6:.1f} MB)")
                continue
            print(f"[build] {cfg}")
            build_parquet(cfg, staging)

    # Step 2: Upload Parquet + delete old files from HF
    print("\n── Uploading to HF ──")
    api = HfApi(token=args.token)
    all_hf_files = list(api.list_repo_files(args.repo, repo_type="dataset"))
    print(f"Fetched {len(all_hf_files)} HF repo files.")

    for i, cfg in enumerate(all_configs):
        parquet_path = staging / cfg / _PARQUET_NAME
        if not parquet_path.exists():
            print(f"[skip] {cfg} — parquet not found locally, skipping")
            continue

        print(f"\n[{i+1}/{len(all_configs)}] {cfg}")
        push_config(cfg, staging, all_hf_files, api, args.dry_run)

        # Avoid rate-limiting (128 commits/hour = 1 per ~28 s)
        if not args.dry_run and i < len(all_configs) - 1:
            time.sleep(2)

    # Step 3: Delete d7b WAV files (parquet already on HF; WAVs would re-trigger audiofolder)
    if not args.config:
        d7b_wavs = [f for f in all_hf_files if f.startswith(f"{_D7B}/") and f.endswith(".wav")]
        if d7b_wavs:
            print(f"\n── Deleting {len(d7b_wavs)} d7b WAV files ──")
            if args.dry_run:
                print(f"[dry-run] would delete {len(d7b_wavs)} d7b WAVs")
            else:
                ops = [CommitOperationDelete(path_in_repo=f) for f in d7b_wavs]
                api.create_commit(
                    repo_id=args.repo,
                    repo_type="dataset",
                    operations=ops,
                    commit_message="Remove d7b WAV files (audio embedded in Parquet)",
                )
                print(f"Deleted {len(d7b_wavs)} d7b WAV files.")

    # Step 4: Push updated README
    readme_path = staging / "README.md"
    if readme_path.exists():
        print(f"\n── Updating README ──")
        if args.dry_run:
            print("[dry-run] would push README.md")
        else:
            api.create_commit(
                repo_id=args.repo,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(
                        path_in_repo="README.md",
                        path_or_fileobj=str(readme_path),
                    )
                ],
                commit_message="Update README: switch configs to Parquet paths",
            )
            print("Pushed README.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
