# PitchBench commands

## Paper workflow

Table 1 uses **28 tasks and MIDI/SPN/Hz only**. Use `paper` for the paper
comparison. The `all` selector includes 32 tasks and additional formats; it is
for exploratory runs, not Table 1.

```bash
pitchbench generate paper
pitchbench evaluate paper --model openrouter/<provider>/<model>
```

Other backends use the same selector:

```bash
pitchbench evaluate paper --model dashscope/<model>
pitchbench evaluate paper --model http://localhost:8001 --name my-local-model
```

Evaluation reads the generated Parquet data. An official frozen-dataset
baseline reproduction uses the dedicated environment and preparation commands
in the [paper reproduction guide](paper/figures-and-tables/README.md).

## Rebuild or verify the committed paper table

These commands need no inference or dataset download:

```bash
PYTHONPATH=src python -m pitchbench.analysis.table1
PYTHONPATH=src python -m pitchbench.analysis.table1 --check
```

To reparse raw answers against the fixed official ground truth, use
`python -m pitchbench.analysis.replay` with the arguments in the reproduction
guide. It is the single paper replay implementation, including D8/F1/F2 F1 and
the equal-weight overall. The old `rescore_sequences` name forwards to it.

## Individual experiments and diagnostics

```bash
pitchbench --list
pitchbench generate a1
pitchbench evaluate a1 --model openrouter/<provider>/<model>
pitchbench evaluate a1 --model openrouter/<provider>/<model> --sample-n 20 --sample-seed 0
pitchbench analyze --preset q1 --model openrouter/<provider>/<model>
python -m pitchbench.analysis.run_analysis q1 --models <model1> <model2>
```

Individual/category runs and analysis presets do not constitute the complete
paper evaluation. Additional tasks and DoReMi remain available for diagnostics.

## Tests

```bash
uv run pytest
PYTHONPATH=src python -m pitchbench.analysis.table1 --check
```

For long evaluations in VacLab, use the workspace queue. Outside VacLab, run
from a clean checkout, keep results outside it, and record the evaluated commit.
