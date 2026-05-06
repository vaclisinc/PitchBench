#!/usr/bin/env python3
"""
Extract accuracies from analysis results across models and experiments.

Usage:
    python extract_analysis.py results/analysis > analysis_summary.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def extract_experiment_results(csv_path):
    """Extract accuracy breakdowns by variable from a single CSV.

    Returns {experiment: {variable: {value: {format: accuracy}}}}
    """
    results = defaultdict(lambda: defaultdict(dict))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row.get('experiment', '')
            metric = row.get('metric', '')

            # Skip total rows and non-breakdown metrics
            if not metric.startswith('by_') or metric.count('.') < 2:
                continue

            # Parse: by_<variable>.<value>.<format>
            parts = metric.split('.')
            if len(parts) < 3:
                continue

            variable = parts[0][3:]  # Remove 'by_' prefix
            value = '.'.join(parts[1:-1])  # Handle multi-part values
            fmt = parts[-1]

            try:
                accuracy = float(row.get('value', 0))
            except (ValueError, TypeError):
                accuracy = 0

            results[exp][variable][value] = accuracy

    return results


def main():
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path('results/analysis')

    if not root_dir.exists():
        print(f"Error: {root_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Collect all results by experiment, variable, and model
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Find all CSV files
    for csv_file in sorted(root_dir.rglob('accuracies_*.csv')):
        # Extract model name from path
        parts = csv_file.relative_to(root_dir).parts
        if len(parts) < 2:
            continue

        model = parts[0]
        exp_results = extract_experiment_results(csv_file)

        for exp, variables in exp_results.items():
            for variable, values in variables.items():
                for value, accuracy in values.items():
                    all_results[exp][variable][value][model] = accuracy

    # Get unique models (sorted)
    models = sorted(set(
        model for exp_vars in all_results.values()
        for var_vals in exp_vars.values()
        for val_models in var_vals.values()
        for model in val_models.keys()
    ))

    # Output CSV with results
    output = csv.writer(sys.stdout)

    # Header
    output.writerow(['experiment', 'variable', 'value'] + models)

    # Data
    for exp in sorted(all_results.keys()):
        variables = all_results[exp]
        for variable in sorted(variables.keys()):
            values = variables[variable]
            for value in sorted(values.keys(), key=lambda x: (
                # Try to sort numerically, fall back to string sort
                float(x) if x.replace('.', '').replace('-', '').isdigit() else float('inf'),
                x
            )):
                row = [exp, variable, value]
                for model in models:
                    accuracy = values[value].get(model, '')
                    if accuracy:
                        row.append(f"{accuracy:.4f}")
                    else:
                        row.append('')
                output.writerow(row)


if __name__ == '__main__':
    main()
