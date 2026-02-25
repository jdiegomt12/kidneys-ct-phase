"""
Script para testear performance del indexing optimizado.

Mide el tiempo que toma escanear los DICOMs con y sin cache.

Uso:
    python scripts/test_indexing_speed.py
    python scripts/test_indexing_speed.py --root-dir /path/to/data --force-rebuild
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time
from data_processing.dataset_index import build_index


def main():
    parser = argparse.ArgumentParser(
        description="Test indexing performance"
    )
    
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Root directory with DICOM data (default: prompt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/index_test",
        help="Output directory for test results"
    )
    
    args = parser.parse_args()
    
    if args.root_dir is None:
        from dataset_index import folder_dialog
        root_dir = folder_dialog()
    else:
        root_dir = Path(args.root_dir)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_index_csv = output_dir / "dataset_index.csv"
    out_failed_csv = output_dir / "failed_indexing.csv"
    
    print(f"Indexing: {root_dir}")
    
    # Run indexing
    start = time.time()
    df = build_index(
        root_dir,
        out_index_csv=out_index_csv,
        out_failed_csv=out_failed_csv,
    )
    elapsed = time.time() - start
    
    # Load failed cases
    import pandas as pd
    try:
        df_failed = pd.read_csv(out_failed_csv)
        n_failed = len(df_failed)
    except:
        n_failed = 0
    
    # Summary
    n_cases = df['case_id'].nunique() if len(df) > 0 else 0
    print(f"\nIndexing complete in {elapsed:.2f}s")
    print(f"Cases: {n_cases} successful, {n_failed} failed")
    print(f"Speed: {len(df) / elapsed:.1f} series/sec")
    print("="*60)


if __name__ == "__main__":
    main()
