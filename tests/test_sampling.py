"""Tests for pitchbench.experiments.helpers.sampling."""

from __future__ import annotations

import pytest

from pitchbench.experiments.helpers.sampling import (
    sampling_meta,
    sampling_summary_lines,
    stratified_sample,
)


# ── stratified_sample ─────────────────────────────────────────────────────────

def _items(n: int, key: str = "group", groups: int = 3) -> list[dict]:
    """Return n items distributed round-robin across `groups` groups."""
    return [{"id": i, key: chr(ord("A") + (i % groups))} for i in range(n)]


class TestStratifiedSample:
    def test_returns_correct_count(self):
        items = _items(30)
        result = stratified_sample(items, 12, key_fn=lambda c: c["group"])
        assert len(result) == 12

    def test_balanced_across_strata(self):
        items = _items(30)
        result = stratified_sample(items, 12, key_fn=lambda c: c["group"])
        counts = {}
        for r in result:
            g = r["group"]
            counts[g] = counts.get(g, 0) + 1
        # With 3 groups and 12 samples: each gets 4
        assert all(v == 4 for v in counts.values())

    def test_seed_produces_same_result(self):
        items = _items(30)
        key = lambda c: c["group"]  # noqa: E731
        r1 = stratified_sample(items, 9, key_fn=key, seed=42)
        r2 = stratified_sample(items, 9, key_fn=key, seed=42)
        assert [x["id"] for x in r1] == [x["id"] for x in r2]

    def test_different_seeds_differ(self):
        items = _items(30)
        key = lambda c: c["group"]  # noqa: E731
        r1 = stratified_sample(items, 9, key_fn=key, seed=0)
        r2 = stratified_sample(items, 9, key_fn=key, seed=99)
        assert [x["id"] for x in r1] != [x["id"] for x in r2]

    def test_raises_if_stratum_too_small(self):
        # Only 1 item in group C, but quota > 1 per stratum
        items = [{"id": i, "g": "A"} for i in range(10)] + [{"id": 99, "g": "B"}]
        with pytest.raises((ValueError, Exception)):
            stratified_sample(items, 10, key_fn=lambda c: c["g"])

    def test_works_with_uneven_groups(self):
        # 10 A, 5 B — sample 6 (3 from each)
        items = [{"g": "A"} for _ in range(10)] + [{"g": "B"} for _ in range(5)]
        result = stratified_sample(items, 6, key_fn=lambda c: c["g"])
        assert len(result) == 6

    def test_full_population_when_n_equals_total(self):
        items = _items(9)
        result = stratified_sample(items, 9, key_fn=lambda c: c["group"])
        assert len(result) == 9


# ── sampling_meta ─────────────────────────────────────────────────────────────

class TestSamplingMeta:
    def test_returns_expected_keys(self):
        meta = sampling_meta(total=100, stratified_by="source", sample_n=20, sample_seed=42)
        assert "sample_n" in meta
        assert "sample_seed" in meta
        assert "total_available" in meta
        assert "stratified_by" in meta

    def test_values_match_inputs(self):
        meta = sampling_meta(total=50, stratified_by="duration", sample_n=10, sample_seed=7)
        assert meta["total_available"] == 50
        assert meta["sample_n"] == 10
        assert meta["sample_seed"] == 7
        assert meta["stratified_by"] == "duration"

    def test_none_sample_n_allowed(self):
        meta = sampling_meta(total=50, stratified_by="source", sample_n=None, sample_seed=0)
        assert meta["sample_n"] is None


# ── sampling_summary_lines ────────────────────────────────────────────────────

class TestSamplingSummaryLines:
    def test_returns_list_of_strings(self):
        meta = sampling_meta(total=100, stratified_by="source", sample_n=20, sample_seed=42)
        lines = sampling_summary_lines(meta)
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_empty_dict_returns_empty_or_list(self):
        lines = sampling_summary_lines({})
        assert isinstance(lines, list)

    def test_mentions_sample_count(self):
        meta = sampling_meta(total=100, stratified_by="source", sample_n=20, sample_seed=42)
        lines = sampling_summary_lines(meta)
        combined = " ".join(lines)
        assert "20" in combined

    def test_no_sampling_gives_minimal_output(self):
        meta = sampling_meta(total=100, stratified_by="source", sample_n=None, sample_seed=0)
        lines = sampling_summary_lines(meta)
        assert isinstance(lines, list)
