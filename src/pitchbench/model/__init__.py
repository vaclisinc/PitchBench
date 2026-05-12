"""PitchBench model package — ALM routing, cost tracking, and dispatch."""

from pitchbench.model import cost
from pitchbench.model.cost import record, get, all_totals, reset, parse_openrouter_usage
from pitchbench.model.dispatcher import dispatch
from pitchbench.model.query import (
    query_alm,
    query_alm_multi,
    query_four_formats,
    query_four_formats_multi,
    query_three_formats,
    get_model_info,
    model_slug,
)

__all__ = [
    "cost",
    "record", "get", "all_totals", "reset", "parse_openrouter_usage",
    "dispatch",
    "query_alm", "query_alm_multi",
    "query_four_formats", "query_four_formats_multi", "query_three_formats",
    "get_model_info", "model_slug",
]
