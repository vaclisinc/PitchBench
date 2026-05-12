"""
Parquet I/O helpers for the generate/evaluate workflow.

Each experiment writes its generated dataset to::

    data/generated/<exp_name>/_questions.parquet
    data/generated/<exp_name>/_questions.csv

The schema mirrors the HuggingFace PitchBench dataset
(pitchbench-authors/PitchBench) but uses a local ``audio_path`` string
column in place of the embedded audio bytes, and adds a ``condition_json``
column containing a JSON dump of the full condition dict for use by the
evaluate step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pitchbench.config as config


def dataset_path(exp_name: str) -> Path:
    """Return the canonical Parquet path for ``exp_name``."""
    return config.GENERATED_DIR / exp_name / "_questions.parquet"


def append_dataset(path: Path, rows: list[dict[str, Any]]) -> int:
    """Append ``rows`` to ``path``, skipping duplicates by ``audio_path``.

    The dedup key is ``audio_path`` for standard experiments and
    ``audio_1_path`` for d7b (dual-audio).  Returns the number of rows
    actually written.
    """
    import pandas as pd

    if not rows:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)

    if path.exists():
        existing = pd.read_parquet(path)
        key = "audio_1_path" if "audio_1_path" in existing.columns else "audio_path"
        dedup_key = "audio_1_path" if "audio_1_path" in new_df.columns else "audio_path"
        existing_keys: set[str] = set(existing[key].astype(str).tolist())
        new_df = new_df[~new_df[dedup_key].astype(str).isin(existing_keys)]
        if new_df.empty:
            return 0
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_parquet(path, index=False)
    csv_path = path.with_name("_questions.csv")
    merged.to_csv(csv_path, index=False)
    return len(new_df)


def filter_rows_to_conditions(
    rows: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only the rows whose ``_condition`` matches a condition in *expected*.

    Used by ``evaluate_cat_*_from_parquet`` in analysis mode to restrict the
    full pre-generated parquet to the subset defined by the analysis preset
    (e.g. 7 pitches × 6 sources = 42 rows instead of 1159).

    Matching is done by JSON-serialising both sides with sorted keys so that
    numeric type differences (int vs int64) don't cause false misses.  Rows
    without a ``_condition`` key are passed through unchanged.
    """
    if not expected:
        return rows
    import json

    def _key(cond: dict) -> str:
        return json.dumps(
            {k: (int(v) if hasattr(v, "item") else v) for k, v in cond.items()},
            sort_keys=True,
        )

    expected_keys = {_key(c) for c in expected}

    result: list[dict[str, Any]] = []
    for row in rows:
        cond = row.get("_condition")
        if cond is None:
            result.append(row)
            continue
        if _key(cond) in expected_keys:
            result.append(row)
    return result


def read_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a generated dataset as a list of row dicts.

    Raises ``FileNotFoundError`` with a clear message if the file is missing
    (directing the user to run ``pitchbench generate`` first).
    """
    import pandas as pd

    if not path.exists():
        # Fall back to old filename for datasets generated before the rename.
        old_path = path.with_name("data.parquet")
        if old_path.exists():
            path = old_path
        else:
            exp = path.parent.name
            raise FileNotFoundError(
                f"Dataset not found for {exp!r}.\n"
                f"Run `pitchbench generate {exp}` first to create {path}."
            )
    df = pd.read_parquet(path)
    rows = df.where(df.notna(), other=None).to_dict(orient="records")
    # Parse condition_json back to dict where present.
    for row in rows:
        cj = row.get("condition_json")
        if isinstance(cj, str):
            try:
                row["_condition"] = json.loads(cj)
            except json.JSONDecodeError:
                row["_condition"] = {}
    return rows
