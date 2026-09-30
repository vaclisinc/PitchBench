#!/usr/bin/env python3
"""Prepare and evaluate the DSP and Basic Pitch baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    required = {
        "experiment",
        "benchmark",
        "input_dataset",
        "runtime",
        "shared_decoder",
        "baselines",
        "outputs",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Config missing sections: {missing}")
    benchmark = payload["benchmark"]
    experiments = benchmark["experiments"]
    if len(experiments) != benchmark["expected_experiment_count"]:
        raise ValueError(
            "Experiment list length does not match expected_experiment_count"
        )
    if len(experiments) != len(set(experiments)):
        raise ValueError("Experiment list contains duplicates")
    return payload


def _formal_preflight() -> dict[str, str]:
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "Formal evaluation requires a clean PitchBench tree. Commit and push first.\n"
            + status
        )
    branch = _git("branch", "--show-current")
    if not branch:
        run_dir = os.environ.get("VACLAB_RUN_DIR")
        receipt_path = Path(run_dir) / "run.json" if run_dir else None
        if receipt_path is None or not receipt_path.is_file():
            raise RuntimeError("Detached evaluation requires a commit-pinned queue receipt")
        receipt = json.loads(receipt_path.read_text())
        head = _git("rev-parse", "HEAD")
        if not receipt.get("snapshot") or receipt.get("git_commit") != head:
            raise RuntimeError("Queue receipt does not match the evaluation snapshot")
        upstream = _git("branch", "-r", "--contains", head).strip()
        if not upstream:
            raise RuntimeError("Evaluation snapshot has not been pushed")
        return {"branch": "detached-queue-snapshot", "commit": head, "upstream": upstream}
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    head = _git("rev-parse", "HEAD")
    upstream_head = _git("rev-parse", upstream)
    if head != upstream_head:
        raise RuntimeError(
            f"Formal evaluation commit is not pushed: HEAD={head}, {upstream}={upstream_head}"
        )
    return {"branch": branch, "commit": head, "upstream": upstream}


def _prepare_runtime(config: dict[str, Any]) -> tuple[Path, Path]:
    runtime_root = Path(config["runtime"]["runtime_root"])
    persistent = REPO_ROOT / config["runtime"]["persistent_results_dir"]
    runtime_data = runtime_root / "data"
    runtime_data.mkdir(parents=True, exist_ok=True)
    persistent.mkdir(parents=True, exist_ok=True)
    asset_data_dir = Path(config["runtime"].get("asset_data_dir", REPO_ROOT / "data"))
    for name in ("preloaded", "soundfonts"):
        source = asset_data_dir / name
        target = runtime_data / name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"Unexpected runtime symlink target: {target}")
        elif target.exists():
            raise RuntimeError(f"Refusing to replace existing runtime path: {target}")
        else:
            target.symlink_to(source, target_is_directory=True)
    return runtime_root, persistent


def _json_value(value: Any) -> Any:
    """Convert Arrow/pandas values to deterministic JSON-compatible values."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _source_type(source: Any) -> str:
    return (
        "waveform"
        if str(source) in {"sine", "sawtooth", "square", "triangle"}
        else "instrument"
    )


def _condition_from_official_row(
    experiment_id: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the condition consumed by the existing task scorers.

    The official Hugging Face release stores the same condition metadata as
    generation, but renames outcome fields with a ``gt_`` prefix. This
    function performs only those documented inverse renames; no answer is
    passed to the acoustic baseline.
    """
    condition = {
        key: _json_value(value)
        for key, value in row.items()
        if key != "audio" and not key.startswith("prompt") and not key.startswith("gt_")
    }
    if "source" in condition:
        condition["source_type"] = _source_type(condition["source"])

    if "gt_midi" in row:
        condition["midi"] = int(row["gt_midi"])

    aliases = {
        "b2": {"query_time_s": "gt_query_time_s"},
        "b3": {"pos_ms": "gt_pos_ms"},
        "b4": {
            "target_onset_ms": "gt_target_onset_ms",
            "target_offset_ms": "gt_target_offset_ms",
        },
        "b5": {"onsets_ms": "gt_onsets_ms"},
        "c1": {"n": "gt_n"},
        "c2": {"interval_st": "gt_interval_st"},
        "c3": {"chord_quality_gt": "gt_quality"},
        "c4": {"midi_notes": "gt_midi_notes"},
        "d1": {"n": "gt_n"},
        "d2": {"answer_gt": "gt_answer"},
        "d3": {"pattern": "gt_pattern"},
        "d4": {"gt_seq": "gt_seq"},
        "d5": {"answer_gt": "gt_answer"},
        "d6": {"signed_st": "gt_signed_st"},
        "d8": {"midi_sequence": "gt_midi_sequence"},
        "f1": {"target_pitches": "gt_seq_midi"},
        "f2": {"target_pitches": "gt_seq_midi"},
    }
    for target, source in aliases.get(experiment_id, {}).items():
        condition[target] = _json_value(row[source])

    if experiment_id in {"f1", "f2"} and isinstance(condition.get("all_notes"), str):
        condition["all_notes"] = json.loads(condition["all_notes"])
    return condition


def _timing_ground_truth(experiment_id: str, condition: dict[str, Any]) -> list[float]:
    if experiment_id == "b3":
        onset = float(condition["pos_ms"])
        return [onset / 1000.0, (onset + float(condition["duration_ms"])) / 1000.0]
    if experiment_id == "b4":
        return [
            float(condition["target_onset_ms"]) / 1000.0,
            float(condition["target_offset_ms"]) / 1000.0,
        ]
    if experiment_id == "b5":
        duration = float(condition["duration_ms"])
        return [
            timestamp / 1000.0
            for onset in condition["onsets_ms"]
            for timestamp in (float(onset), float(onset) + duration)
        ]
    return []


def _official_shard_path(config: dict[str, Any], experiment: str) -> Path:
    dataset = config["input_dataset"]
    return Path(dataset["local_dir"]) / experiment / str(dataset["parquet_filename"])


def _official_shard_revision(config: dict[str, Any], experiment: str) -> str:
    dataset = config["input_dataset"]
    metadata_path = (
        Path(dataset["local_dir"])
        / ".cache"
        / "huggingface"
        / "download"
        / experiment
        / f"{dataset['parquet_filename']}.metadata"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Hugging Face download metadata is missing: {metadata_path}"
        )
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"Empty Hugging Face download metadata: {metadata_path}")
    return lines[0]


def _prepare_official_dataset(
    config: dict[str, Any],
    runtime_root: Path,
    counts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract paper audio and scorer metadata from the official HF Parquet."""
    generated_root = runtime_root / "data" / "generated"
    manifest: dict[str, Any] = {
        "repository": config["input_dataset"]["repository"],
        "revision": config["input_dataset"]["revision"],
        "split": config["input_dataset"]["split"],
        "subsets": {},
    }
    total_rows = 0
    total_audio_bytes = 0

    for experiment, expected in counts.items():
        shard = _official_shard_path(config, experiment)
        if not shard.exists():
            raise FileNotFoundError(f"Official dataset shard is missing: {shard}")
        actual_revision = _official_shard_revision(config, experiment)
        expected_revision = str(config["input_dataset"]["revision"])
        if actual_revision != expected_revision:
            raise RuntimeError(
                f"{experiment}: expected dataset revision {expected_revision}, "
                f"found {actual_revision}"
            )
        frame = pd.read_parquet(shard)
        if len(frame) != expected["sampled"]:
            raise RuntimeError(
                f"{experiment}: official shard has {len(frame)} rows; "
                f"expected {expected['sampled']}"
            )
        if config["input_dataset"]["embedded_audio_field"] not in frame:
            raise RuntimeError(
                f"{experiment}: official shard has no embedded audio field"
            )

        output_dir = generated_root / experiment
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        prepared_rows: list[dict[str, Any]] = []
        subset_audio_bytes = 0

        for row_index, raw_row in enumerate(frame.to_dict(orient="records")):
            row = {key: _json_value(value) for key, value in raw_row.items()}
            audio = row.pop(config["input_dataset"]["embedded_audio_field"])
            if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
                raise TypeError(
                    f"{experiment} row {row_index}: malformed embedded audio"
                )
            original_name = Path(str(audio.get("path") or "audio.wav")).name
            audio_path = audio_dir / f"{row_index:05d}_{original_name}"
            audio_bytes = audio["bytes"]
            if audio_path.exists():
                if (
                    config["input_dataset"]["verify_embedded_audio"]
                    and audio_path.read_bytes() != audio_bytes
                ):
                    raise RuntimeError(
                        f"Existing extracted audio differs: {audio_path}"
                    )
            else:
                audio_path.write_bytes(audio_bytes)
            subset_audio_bytes += len(audio_bytes)

            condition = _condition_from_official_row(
                expected["experiment_id"],
                row,
            )
            prepared = {
                **row,
                **condition,
                "audio_path": str(audio_path),
                "condition_json": json.dumps(condition, sort_keys=True),
            }
            prompt_renames = {
                "prompt_abc": "prompt_spn",
                "prompt_solfege": "prompt_doremi",
                "prompt_freq": "prompt_hz",
            }
            for source, target in prompt_renames.items():
                if source in prepared:
                    prepared[target] = prepared[source]
            if "prompt" in prepared:
                prepared["prompt_main"] = prepared["prompt"]
            gt_renames = {
                "gt_abc": "gt_spn",
                "gt_solfege": "gt_doremi",
                "gt_freq": "gt_hz",
            }
            for source, target in gt_renames.items():
                if source in prepared:
                    prepared[target] = prepared[source]
            gt_timestamps = _timing_ground_truth(
                expected["experiment_id"],
                condition,
            )
            if gt_timestamps:
                prepared["gt_timestamps_json"] = json.dumps(gt_timestamps)
            prepared_rows.append(prepared)

        prepared_frame = pd.DataFrame(prepared_rows)
        parquet_path = output_dir / "_questions.parquet"
        prepared_frame.to_parquet(parquet_path, index=False)
        total_rows += len(prepared_frame)
        total_audio_bytes += subset_audio_bytes
        subset_manifest: dict[str, Any] = {
            "source_parquet": str(shard),
            "rows": len(prepared_frame),
            "embedded_audio_bytes": subset_audio_bytes,
        }
        if config["input_dataset"]["write_shard_sha256"]:
            subset_manifest["parquet_sha256"] = hashlib.sha256(
                shard.read_bytes()
            ).hexdigest()
        manifest["subsets"][experiment] = subset_manifest
        print(
            f"  {experiment}: prepared {len(prepared_frame)} official rows "
            f"({subset_audio_bytes / (1024**2):.1f} MiB audio)",
            flush=True,
        )

    if total_rows != int(config["benchmark"]["expected_condition_count"]):
        raise RuntimeError(
            f"Prepared {total_rows} rows; expected "
            f"{config['benchmark']['expected_condition_count']}"
        )
    manifest["total_rows"] = total_rows
    manifest["embedded_audio_bytes"] = total_audio_bytes
    manifest_path = runtime_root / "official_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _environment(
    config: dict[str, Any], config_path: Path, runtime_root: Path
) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_entries = [str(REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    pythonpath = os.pathsep.join(pythonpath_entries)
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "PITCHBENCH_ROOT": str(runtime_root),
            "PITCHBENCH_BASELINE_CONFIG": str(config_path),
            "MPLCONFIGDIR": str(runtime_root / "matplotlib"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _validate_runtime_environment(config: dict[str, Any]) -> None:
    expected_python = str(config["runtime"]["python"])
    actual_python = ".".join(str(part) for part in sys.version_info[:3])
    if actual_python != expected_python:
        raise RuntimeError(
            f"Expected Python {expected_python}, running {actual_python}"
        )
    mismatches: list[str] = []
    for package, expected in config["runtime"]["packages"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "not-installed"
        if actual != str(expected):
            mismatches.append(f"{package}: expected {expected}, found {actual}")
    if mismatches:
        raise RuntimeError(
            "Runtime package versions do not match the formal config:\n"
            + "\n".join(mismatches)
        )


def _expected_counts(
    config: dict[str, Any],
    env: dict[str, str],
) -> dict[str, dict[str, Any]]:
    os.environ.update(
        {
            "PITCHBENCH_ROOT": env["PITCHBENCH_ROOT"],
            "PITCHBENCH_BASELINE_CONFIG": env["PITCHBENCH_BASELINE_CONFIG"],
        }
    )
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import pitchbench.config as pitchbench_config

    pitchbench_config.AUDIO_DIR = pitchbench_config.GENERATED_DIR
    from pitchbench.experiments.run import _id_to_name
    import pyarrow.parquet as pq

    counts: dict[str, dict[str, Any]] = {}
    for experiment_id in config["benchmark"]["experiments"]:
        name = _id_to_name(experiment_id)
        if name is None:
            raise RuntimeError(f"Unknown experiment ID in config: {experiment_id}")
        # The official shards already contain the frozen paper sample. Rebuilding
        # conditions here incorrectly depends on local synthesizers and music21,
        # even though evaluation uses the embedded audio and ground truth.
        revision = _official_shard_revision(config, name)
        if revision != str(config["input_dataset"]["revision"]):
            raise RuntimeError(f"{name}: unexpected dataset revision {revision}")
        count = pq.read_metadata(_official_shard_path(config, name)).num_rows
        if not count:
            raise RuntimeError(f"Empty official dataset shard: {name}")
        counts[name] = {
            "experiment_id": experiment_id,
            "sampled": count,
            "available": count,
            "sampling": {"source": "official frozen shard", "revision": revision},
        }
    total = sum(item["sampled"] for item in counts.values())
    if total != int(config["benchmark"]["expected_condition_count"]):
        raise RuntimeError(
            f"Expected {config['benchmark']['expected_condition_count']} conditions, got {total}"
        )
    return counts


def _validate_generated(
    runtime_root: Path,
    counts: dict[str, dict[str, Any]],
) -> None:
    failures: list[str] = []
    for experiment, expected in counts.items():
        path = runtime_root / "data" / "generated" / experiment / "_questions.parquet"
        if not path.exists():
            failures.append(f"{experiment}: missing {path}")
            continue
        actual = len(pd.read_parquet(path))
        if actual != expected["sampled"]:
            failures.append(
                f"{experiment}: expected {expected['sampled']} rows, found {actual}"
            )
    if failures:
        raise RuntimeError("Generated-data validation failed:\n" + "\n".join(failures))


def _validate_evaluation(
    run_dir: Path,
    counts: dict[str, dict[str, Any]],
) -> None:
    failures: list[str] = []
    for experiment, expected in counts.items():
        experiment_dir = run_dir / experiment
        files = sorted(experiment_dir.glob("results_*.json"))
        if len(files) != 1:
            failures.append(
                f"{experiment}: expected one results JSON, found {len(files)}"
            )
            continue
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        actual = len(payload.get("results", []))
        if actual != expected["sampled"]:
            failures.append(
                f"{experiment}: expected {expected['sampled']} results, found {actual}"
            )
    if failures:
        raise RuntimeError("Evaluation validation failed:\n" + "\n".join(failures))


def _package_versions() -> dict[str, str]:
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def _configured_output_labels(config: dict[str, Any]) -> list[str]:
    labels = [
        str(value)
        for key, value in config["outputs"].items()
        if key.endswith("_model_label")
    ]
    if not labels:
        raise ValueError("outputs must define at least one *_model_label")
    if len(labels) != len(set(labels)):
        raise ValueError("outputs contains duplicate model labels")
    return labels


def _copy_results(
    config: dict[str, Any],
    runtime_root: Path,
    persistent: Path,
) -> None:
    evaluation_root = runtime_root / "results" / "evaluation"
    run_name = config["outputs"]["run_name"]
    labels = _configured_output_labels(config)
    copied = 0
    for label in labels:
        source = evaluation_root / label / run_name
        if not source.exists():
            continue
        target = persistent / label / run_name
        shutil.copytree(source, target, dirs_exist_ok=True)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"No formal baseline results found under {evaluation_root}")


def _write_aggregate(
    config: dict[str, Any],
    runtime_root: Path,
    persistent: Path,
) -> Path | None:
    run_name = config["outputs"]["run_name"]
    frames: list[pd.DataFrame] = []
    for label in _configured_output_labels(config):
        source = (
            runtime_root
            / "results"
            / "evaluation"
            / label
            / run_name
            / "overall"
            / "aggregated_accuracies.csv"
        )
        if source.exists():
            frame = pd.read_csv(source)
            frame["model"] = label
            frames.append(frame)
    if not frames:
        return None
    aggregate = pd.concat(frames, ignore_index=True)
    output = persistent / "baseline_aggregate.csv"
    aggregate.to_csv(output, index=False)
    return output


def _write_receipt(
    *,
    config: dict[str, Any],
    config_path: Path,
    preflight: dict[str, str],
    counts: dict[str, dict[str, Any]],
    dataset_manifest: dict[str, Any] | None,
    persistent: Path,
    phases: list[str],
    started_at: str,
) -> Path:
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "git": preflight,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": config_sha,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "packages": _package_versions(),
        },
        "command": [sys.executable, *sys.argv],
        "benchmark": {
            **config["benchmark"],
            "resolved_counts": counts,
        },
        "input_dataset": {
            **config["input_dataset"],
            "manifest": dataset_manifest,
        },
        "baselines": config["baselines"],
        "phases": phases,
        "runtime_root": config["runtime"]["runtime_root"],
        "persistent_results": str(persistent.resolve()),
    }
    path = persistent / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--phase",
        choices=("prepare-data", "dsp", "basic-pitch", "all"),
        default="all",
    )
    parser.add_argument(
        "--skip-formal-preflight",
        action="store_true",
        help="Allowed only for local smoke testing; never use for the formal run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    started_at = datetime.now(UTC).isoformat()
    if args.skip_formal_preflight:
        preflight = {
            "branch": _git("branch", "--show-current") or "detached",
            "commit": _git("rev-parse", "HEAD"),
            "upstream": "SMOKE_TEST_ONLY",
        }
    else:
        preflight = _formal_preflight()

    runtime_root, persistent = _prepare_runtime(config)
    env = _environment(config, config_path, runtime_root)
    counts = _expected_counts(config, env)
    experiments = list(config["benchmark"]["experiments"])
    seed = str(config["benchmark"]["sample_seed"])
    run_name = config["outputs"]["run_name"]
    models: dict[str, tuple[str, str]] = {}
    if "dsp" in config["baselines"]:
        models["dsp"] = (
            config["baselines"]["dsp"]["model_name"],
            config["outputs"]["dsp_model_label"],
        )
    if "basic_pitch" in config["baselines"]:
        models["basic-pitch"] = (
            config["baselines"]["basic_pitch"]["model_name"],
            config["outputs"]["basic_pitch_model_label"],
        )
    phases = ["prepare-data", *models] if args.phase == "all" else [args.phase]
    unknown_phases = set(phases) - {"prepare-data", *models}
    if unknown_phases:
        raise ValueError(
            f"Requested phases are not configured: {sorted(unknown_phases)}"
        )
    if any(phase in phases for phase in models):
        _validate_runtime_environment(config)

    dataset_manifest: dict[str, Any] | None = None
    if "prepare-data" in phases:
        dataset_manifest = _prepare_official_dataset(
            config,
            runtime_root,
            counts,
        )
        _validate_generated(runtime_root, counts)
    else:
        manifest_path = runtime_root / "official_dataset_manifest.json"
        if manifest_path.exists():
            dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for phase, (model_name, model_label) in models.items():
        if phase not in phases:
            continue
        _validate_generated(runtime_root, counts)
        _run(
            [
                sys.executable,
                "-m",
                "pitchbench.experiments.run",
                "evaluate",
                *experiments,
                "--model",
                model_name,
                "--name",
                model_label,
                "--run-name",
                run_name,
                "--sample-seed",
                seed,
            ],
            env=env,
        )
        run_dir = runtime_root / "results" / "evaluation" / model_label / run_name
        _validate_evaluation(run_dir, counts)

    if any(phase in phases for phase in models):
        _copy_results(config, runtime_root, persistent)
        aggregate_path = _write_aggregate(config, runtime_root, persistent)
        if aggregate_path is not None:
            print(f"Aggregate: {aggregate_path}")
    receipt_path = _write_receipt(
        config=config,
        config_path=config_path,
        preflight=preflight,
        counts=counts,
        dataset_manifest=dataset_manifest,
        persistent=persistent,
        phases=phases,
        started_at=started_at,
    )
    print(f"Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
