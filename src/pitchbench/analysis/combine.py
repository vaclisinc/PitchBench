"""Combine accuracies CSV files under direct underscore-prefixed folders.

Usage:
    python analysis/combine_accuracies.py /path/to/run_folder
    python analysis/combine_accuracies.py /path/to/run_folder --output merged.csv

The script only considers direct child folders of the supplied folder whose
names begin with ``_``. Inside each such folder, it only matches CSVs that
follow this exact structure:

    <root>/_.../<experiment>/run_.../accuracies_*.csv

It prepends ``model`` and ``experiment`` columns and writes one combined CSV at
the top level of the supplied folder. The model is derived from the filename
stem and the experiment is derived from the experiment directory in that path.
Source CSV files are left unchanged.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pitchbench.config as config

def combine_accuracies_csvs(
    run_dir: str | Path,
    output_name: str = "accuracies_combined.csv",
) -> Path | None:
    """Combine exact-match accuracies CSVs under direct ``_`` folders.

    Returns the output path, or ``None`` if no matching files are found.
    """
    run_path = Path(run_dir).expanduser().resolve()
    search_roots = sorted(
        path for path in run_path.iterdir()
        if path.is_dir() and path.name.startswith("_")
    )
    print(
        "Searching for accuracies_*.csv files with pattern "
        "<root>/_.../<experiment>/run_.../accuracies_*.csv "
        f"under {run_path}..."
    )
    csv_paths = sorted(
        csv_path
        for root in search_roots
        for csv_path in root.glob("*/run_*/accuracies_*.csv")
        if csv_path.is_file() and csv_path.name != output_name
    )
    if not csv_paths:
        return None

    output_path = run_path / output_name
    fieldnames: list[str] | None = None

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer: csv.DictWriter | None = None
        for csv_path in csv_paths:
            model_name = csv_path.stem.removeprefix("accuracies_")
            experiment_name = csv_path.parent.parent.name
            with csv_path.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                if reader.fieldnames is None:
                    continue
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(
                        out_f,
                        fieldnames=["model", "experiment", *fieldnames],
                    )
                    writer.writeheader()
                elif reader.fieldnames != fieldnames:
                    raise ValueError(
                        f"Mismatched header in {csv_path}: expected {fieldnames}, "
                        f"got {reader.fieldnames}"
                    )

                assert writer is not None
                print(
                    f"Processing {csv_path} with model={model_name}, "
                    f"experiment={experiment_name}... writing to {output_path}"
                )
                for row in reader:
                    writer.writerow(
                        {
                            "model": model_name,
                            "experiment": experiment_name,
                            **row,
                        }
                    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        help="Folder whose direct _* child folders will be searched for accuracies_*.csv",
        default=config.RESULTS_DIR,
    )
    parser.add_argument(
        "--output",
        default="accuracies_combined.csv",
        help="Output filename to write at the top level of run_dir",
    )
    args = parser.parse_args()
    
    print(f"Combining accuracies CSV files under {args.run_dir}...")
    output_path = combine_accuracies_csvs(args.run_dir, output_name=args.output)
    if output_path is None:
        print("No accuracies_*.csv files found.")
        return
    print(output_path)


if __name__ == "__main__":
    main()