"""Tests for pitchbench.model.query — slug, routing, is_local_url."""

from __future__ import annotations

import pytest

from pitchbench.model.query import _is_local_url, model_slug
import pitchbench.config as config


# ── _is_local_url ─────────────────────────────────────────────────────────────

class TestIsLocalUrl:
    def test_http_localhost(self):
        assert _is_local_url("http://localhost:8001") is True

    def test_https_remote(self):
        assert _is_local_url("https://example.com") is True

    def test_openrouter_slug(self):
        assert _is_local_url("openrouter/google/gemini-2.5-flash") is False

    def test_dashscope_slug(self):
        assert _is_local_url("dashscope/qwen3-omni") is False

    def test_bare_name(self):
        assert _is_local_url("manual") is False


# ── model_slug ────────────────────────────────────────────────────────────────

class TestModelSlug:
    def test_http_url_returns_netloc(self):
        assert model_slug("http://localhost:8001") == "localhost_8001"

    def test_http_url_no_port(self):
        assert model_slug("http://localhost") == "localhost"

    def test_openrouter_slug(self):
        slug = model_slug("openrouter/google/gemini-2.5-flash")
        assert slug == "openrouter_google_gemini_2_5_flash"

    def test_dashscope_slug(self):
        slug = model_slug("dashscope/qwen3-omni-flash")
        assert slug == "dashscope_qwen3_omni_flash"

    def test_health_dict_with_checkpoint(self):
        info = {"model": "nvidia/audio-flamingo-next-hf", "checkpoint": "v2"}
        slug = model_slug(info)
        assert slug == "audio_flamingo_next_v2"

    def test_health_dict_strips_hf_suffix(self):
        info = {"model": "nvidia/music-flamingo-hf", "checkpoint": ""}
        slug = model_slug(info)
        assert slug == "music_flamingo"

    def test_health_dict_empty_falls_back(self):
        slug = model_slug({})
        assert slug == "unknown_model"

    def test_special_chars_sanitised(self):
        slug = model_slug("openrouter/openai/gpt-4o-audio-preview")
        assert "/" not in slug
        assert "-" not in slug
        assert slug.startswith("openrouter")


# ── concurrency_for ───────────────────────────────────────────────────────────

class TestConcurrencyFor:
    def test_openrouter(self):
        from pitchbench.config import concurrency_for
        result = concurrency_for("openrouter/google/gemini-2.5-flash")
        assert result == config.CONCURRENCY["openrouter"]

    def test_dashscope(self):
        from pitchbench.config import concurrency_for
        result = concurrency_for("dashscope/qwen3-omni")
        assert result == config.CONCURRENCY["dashscope"]

    def test_manual(self):
        from pitchbench.config import concurrency_for
        assert concurrency_for("manual") == 1

    def test_local_url(self):
        from pitchbench.config import concurrency_for
        result = concurrency_for("http://localhost:8001")
        assert result >= 1
