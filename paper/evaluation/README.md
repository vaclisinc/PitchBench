# Saved model answers

The compressed `results_*.json.gz` files contain ALM answers, stimulus metadata,
and inference provenance. They are the inputs to `pitchbench.analysis.replay`.
The `source_sha256` field identifies the full source record before compaction.

The 28-task benchmark scores and all model means are in
[Table 1](../figures-and-tables/table1.md), with
[full-precision evidence](../../results/table1-recomputed/README.md).
The optional split-reference answer files do not contribute to Table 1.

A1 analysis accepts these compressed answers directly:

```bash
python -m pitchbench.analysis.a1 --help
```
