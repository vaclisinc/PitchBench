"""Combine accuracies CSV files under a run folder.

Usage:
    python analysis/combine_accuracies.py /path/to/run_folder
    python analysis/combine_accuracies.py /path/to/run_folder --output merged.csv

The script recursively finds every ``accuracies_*.csv`` under the supplied
folder, prepends a ``model`` column derived from the filename stem, and writes
one combined CSV at the top level of the supplied folder. Source CSV files are
left unchanged.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def combine_accuracies_csvs(
    run_dir: str | Path,
    output_name: str = "accuracies_combined.csv",
) -> Path | None:
    """Combine all ``accuracies_*.csv`` files below ``run_dir`` into one CSV.

    Returns the output path, or ``None`` if no matching files are found.
    """
    run_path = Path(run_dir).expanduser().resolve()
    csv_paths = sorted(
        path for path in run_path.rglob("accuracies_*.csv")
        if path.is_file() and path.name != output_name
    )
    if not csv_paths:
        return None

    output_path = run_path / output_name
    fieldnames: list[str] | None = None

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer: csv.DictWriter | None = None
        for csv_path in csv_paths:
            model_name = csv_path.stem.removeprefix("accuracies_")
            with csv_path.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                if reader.fieldnames is None:
                    continue
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=["model", *fieldnames])
                    writer.writeheader()
                elif reader.fieldnames != fieldnames:
                    raise ValueError(
                        f"Mismatched header in {csv_path}: expected {fieldnames}, "
                        f"got {reader.fieldnames}"
                    )

                assert writer is not None
                for row in reader:
                    writer.writerow({"model": model_name, **row})

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Folder to search recursively for accuracies_*.csv")
    parser.add_argument(
        "--output",
        default="accuracies_combined.csv",
        help="Output filename to write at the top level of run_dir",
    )
    args = parser.parse_args()

    output_path = combine_accuracies_csvs(args.run_dir, output_name=args.output)
    if output_path is None:
        print("No accuracies_*.csv files found.")
        return
    print(output_path)


if __name__ == "__main__":
    main()