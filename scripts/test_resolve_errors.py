"""
Test script for resolving indexing errors.

Usage:
    python scripts/test_resolve_errors.py \
        --index-csv outputs/index_test/dataset_index.csv \
        --failed-csv outputs/index_test/failed_indexing.csv \
        --mixed-csv outputs/index_test/mixed_series.csv \
        --data-dist-csv outputs/data_distribution/dicom_metadata_per_volume.csv \
        --out-csv outputs/index_test/dataset_index_final.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.resolve_errors import resolve_index_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Test resolve errors workflow")
    parser.add_argument(
        "--index-csv",
        type=str,
        default="outputs/index_test/dataset_index.csv",
        help="Path to dataset_index.csv",
    )
    parser.add_argument(
        "--failed-csv",
        type=str,
        default="outputs/index_test/failed_indexing.csv",
        help="Path to failed_indexing.csv",
    )
    parser.add_argument(
        "--mixed-csv",
        type=str,
        default="outputs/index_test/mixed_series.csv",
        help="Path to mixed_series.csv",
    )
    parser.add_argument(
        "--data-dist-csv",
        type=str,
        default="outputs/data_distribution/dicom_metadata_per_volume.csv",
        help="Path to dicom_metadata_per_volume.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/index_test/dataset_index_final.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()
    resolve_index_errors(
        index_csv=Path(args.index_csv),
        failed_csv=Path(args.failed_csv),
        mixed_csv=Path(args.mixed_csv),
        data_dist_csv=Path(args.data_dist_csv),
        out_csv=Path(args.out_csv),
    )


if __name__ == "__main__":
    main()
