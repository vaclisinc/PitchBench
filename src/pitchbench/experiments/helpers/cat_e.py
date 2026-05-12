"""
Shared runner for category-E experiments (single-pitch ID under
robustness manipulations).

All five cat-E scripts (e1..e5) are 4-format pitch identification tasks
— exactly the same shape as cat-A — with one extra independent variable
applied to the audio:

* **e1** — loudness (dBFS variation)
* **e2** — audio effects (filters, distortion, reverb, chorus, …)
* **e3** — additive backgrounds at varying SNR
* **e4** — harmonic saturation (tanh soft-clipping levels)
* **e5** — time-stretching vs resampling (tempo ↔ pitch coupling)

Because the parsing/scoring pipeline is identical to cat-A
(``standard_pitch_record`` followed by the same Hz-column stripping for
the CSV), this module is just a re-export of :class:`CatASpec` and
:func:`run_cat_a_experiment` — under the names ``CatESpec`` and
``run_cat_e_experiment`` so each experiment script reads as cat-E
specific. No duplication.

Each script provides:

* ``build_conditions()`` — Cartesian product of (source, pitch, manipulation).
* ``wav_for(cond)`` — engine call that applies the manipulation.
* ``prompts_for(cond)`` (or ``prompt_prefix``) — usually a single fixed prefix.
* ``record_extras`` — the IV column(s) that should land on the per-stim record.
* ``SPEC = CatESpec(...)`` — frozen dataclass.

The auto-marginals in :mod:`results` then produce the
``by_<canonical_iv>.<value>.<format>`` rows automatically — no hand-built
``by_*`` summary blocks needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pitchbench.experiments.helpers.cat_a import (
    CatASpec as CatESpec,
    generate_cat_a_data,
    evaluate_cat_a_from_parquet,
    run_cat_a_experiment as run_cat_e_experiment,
    run_one_model,                         # noqa: F401  (re-export for tests)
    _strip_for_csv as _strip_for_csv,      # noqa: F401  (re-export)
)

__all__ = [
    "CatESpec",
    "run_cat_e_experiment",
    "run_one_model",
    "generate_cat_e_data",
    "evaluate_cat_e_from_parquet",
]


def generate_cat_e_data(  # type: ignore[name-defined]
    spec:        "CatESpec",
    *,
    sample_n:    int | None = None,
    sample_seed: int = 42,
) -> int:
    """Generate audio and write/append data.parquet for cat-E experiments.

    Delegates to :func:`generate_cat_a_data` since cat-E uses the identical
    four-format pitch scoring pipeline.
    """
    return generate_cat_a_data(spec, sample_n=sample_n, sample_seed=sample_seed)


def evaluate_cat_e_from_parquet(
    spec:        "CatESpec",  # type: ignore[name-defined]
    model_name:  str,
    run_dir:     Path,
    sample_info: dict[str, Any] | None = None,
) -> dict:
    """Evaluate ``model_name`` on the pre-generated dataset for a cat-E spec."""
    return evaluate_cat_a_from_parquet(spec, model_name, run_dir, sample_info)
