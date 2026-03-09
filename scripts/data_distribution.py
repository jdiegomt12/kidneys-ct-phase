"""
Extract DICOM header metadata into a per-volume CSV and summary distributions.

Usage:
    python scripts/data_distribution.py \
        --index-csv outputs/index_test/dataset_index.csv \
        --output-dir outputs/data_distribution
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import SimpleITK as sitk


DICOM_TAGS: Dict[str, str] = {
    "manufacturer": "0008|0070",
    "model": "0008|1090",
    "modality": "0008|0060",
    "slice_thickness": "0018|0050",
    "kvp": "0018|0060",
    "data_collection_diameter": "0018|0090",
    "software_versions": "0018|1020",
    "reconstruction_diameter": "0018|1100",
    "table_height": "0018|1130",
    "exposure_time": "0018|1150",
    "xray_current": "0018|1151",
    "exposure": "0018|1152",
    "filter_type": "0018|1160",
    "generator_power": "0018|1162",
    "focal_spots": "0018|1190",
    "convolution_kernel": "0018|1210",
    "revolution_time": "0018|9305",
    "spiral_pitch_factor": "0018|9311",
    "energy_weight_factor": "0018|9353",
    "samples_per_pixel": "0028|0002",
    "rows": "0028|0010",
    "columns": "0028|0011",
    "pixel_spacing": "0028|0030",
    "window_center": "0028|1050",
    "window_width": "0028|1051",
    "rescale_intercept": "0028|1052",
    "rescale_slope": "0028|1053",
}


def is_dicom(filepath: str) -> bool:
    """Check if a file is a DICOM by reading its header."""
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def read_dicom_tags(filepath: str) -> Dict[str, str]:
    """Read DICOM tags from a single file."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(filepath)
    reader.ReadImageInformation()

    values: Dict[str, str] = {}
    for name, tag in DICOM_TAGS.items():
        if reader.HasMetaDataKey(tag):
            values[name] = reader.GetMetaData(tag)
        else:
            values[name] = ""
    return values


def _parse_suffix(case_id: str) -> Optional[int]:
    """Return numeric suffix if present (e.g., '134_2' -> 2)."""
    match = re.search(r"_(\d+)$", case_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _load_mixed_map(mixed_csv: Optional[Path]) -> Dict[Tuple[str, str, str], List[dict]]:
    """Load mixed series groups keyed by (case_id, phase, dicom_dir)."""
    if mixed_csv is None or not mixed_csv.exists():
        return {}

    df_mixed = pd.read_csv(mixed_csv)
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for _, row in df_mixed.iterrows():
        key = (str(row.get("case_id", "")), str(row.get("phase", "")), str(row.get("dicom_dir", "")))
        groups.setdefault(key, []).append(
            {
                "series_uid": str(row.get("series_uid", "")),
                "n_files": int(row.get("n_files", 0)),
                "files": str(row.get("files", "")),
            }
        )
    return groups


def _choose_mixed_file(
    groups: List[dict],
    dicom_dir: Path,
    phase: str,
) -> Optional[str]:
    """Choose a file from mixed groups based on phase rules."""
    if not groups:
        return None

    if phase == "venous":
        selected = max(groups, key=lambda g: g.get("n_files", 0))
    else:
        selected = min(groups, key=lambda g: g.get("n_files", 0))

    files_str = selected.get("files", "")
    if files_str:
        first_name = files_str.split(";")[0]
        return str(dicom_dir / first_name)
    return None


def _choose_any_dicom(dicom_dir: Path) -> Optional[str]:
    """Pick a single DICOM file from a directory."""
    try:
        filenames = sorted([f for f in os.listdir(dicom_dir) if os.path.isfile(dicom_dir / f)])
    except Exception:
        return None

    for fname in filenames:
        filepath = str(dicom_dir / fname)
        if is_dicom(filepath):
            return filepath
    return None


def build_volume_table(index_csv: Path, mixed_csv: Optional[Path]) -> pd.DataFrame:
    """Build per-volume metadata table using dataset_index.csv."""
    df_index = pd.read_csv(index_csv)
    mixed_map = _load_mixed_map(mixed_csv)
    rows: List[dict] = []

    for _, row in df_index.iterrows():
        case_id = str(row.get("case_id", ""))
        phase = str(row.get("phase", ""))
        dicom_dir = Path(str(row.get("dicom_dir", "")))
        note = str(row.get("note", ""))
        case_id_phase = f"{case_id}_{phase}" if phase else case_id

        suffix = _parse_suffix(case_id)
        if suffix is not None and suffix != 1:
            continue

        if not dicom_dir.exists():
            continue

        filepath = None
        if "MIXED_HEADERS" in note:
            key = (case_id, phase, str(dicom_dir))
            filepath = _choose_mixed_file(mixed_map.get(key, []), dicom_dir, phase)

        if filepath is None:
            filepath = _choose_any_dicom(dicom_dir)

        if filepath is None:
            continue

        try:
            print(f"Reading: {case_id_phase}")
            tags = read_dicom_tags(filepath)
        except Exception:
            continue

        record = {
            "case_id": case_id,
            "phase": phase,
            "case_id_phase": case_id_phase,
            "dicom_dir": str(dicom_dir),
            "filename": Path(filepath).name,
        }
        record.update(tags)
        rows.append(record)

    return pd.DataFrame(rows)


def write_value_counts(df: pd.DataFrame, output_dir: Path) -> None:
    """Write per-field value counts to CSV files."""
    counts_dir = output_dir / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)

    for field in DICOM_TAGS.keys():
        if field not in df.columns:
            continue
        series = df[field].fillna("")
        counts = series.value_counts(dropna=False)
        out = counts.reset_index()
        out.columns = [field, "count"]
        out.to_csv(counts_dir / f"{field}_counts.csv", index=False)


LINEAR_FIELDS = {
    "slice_thickness",
    "kvp",
    "data_collection_diameter",
    "reconstruction_diameter",
    "table_height",
    "exposure_time",
    "xray_current",
    "exposure",
    "generator_power",
    "revolution_time",
    "spiral_pitch_factor",
    "energy_weight_factor",
    "rows",
    "columns",
    "window_center",
    "window_width",
    "rescale_intercept",
    "rescale_slope",
}


def _first_float(value: str) -> Optional[float]:
    """Parse the first numeric value from a DICOM string."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.split("\\")[0]
    try:
        return float(text)
    except Exception:
        return None


def plot_bar_charts(df: pd.DataFrame, output_dir: Path, max_bars: int = 7) -> None:
    """Create bar charts with at most max_bars per field."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for field in DICOM_TAGS.keys():
        if field not in df.columns:
            continue

        series = df[field]
        values = series.apply(_first_float).dropna()
        numeric_ratio = 0.0
        if len(series) > 0:
            numeric_ratio = len(values) / len(series)

        is_numeric_like = field in LINEAR_FIELDS or numeric_ratio >= 0.8
        if is_numeric_like:
            if values.empty:
                continue
            unique_vals = values.nunique()
            if unique_vals > max_bars:
                bins = max_bars
                bins_series = pd.cut(values, bins=bins, include_lowest=True)
                counts = bins_series.value_counts().sort_index()
            else:
                counts = values.value_counts().sort_index()
        else:
            counts = series.fillna("(empty)").replace("", "(empty)").value_counts()
            if counts.empty:
                continue
            if len(counts) > max_bars:
                top = counts.iloc[: max_bars - 1]
                other = counts.iloc[max_bars - 1 :].sum()
                counts = pd.concat([top, pd.Series({"Other": other})])

        plt.figure(figsize=(9, 4))
        counts.plot(kind="bar")
        plt.title(field.replace("_", " "))
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{field}_bar.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DICOM header distribution data")
    parser.add_argument(
        "--index-csv",
        type=str,
        default="outputs/index_test/dataset_index.csv",
        help="Path to dataset_index.csv",
    )
    parser.add_argument(
        "--mixed-csv",
        type=str,
        default=None,
        help="Path to mixed_series.csv (default: sibling of index csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/data_distribution",
        help="Output directory for CSVs and plots",
    )

    args = parser.parse_args()
    index_csv = Path(args.index_csv)
    mixed_csv = Path(args.mixed_csv) if args.mixed_csv else index_csv.parent / "mixed_series.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    df = build_volume_table(index_csv, mixed_csv)

    out_csv = output_dir / "dicom_metadata_per_volume.csv"
    df.to_csv(out_csv, index=False)

    write_value_counts(df, output_dir)
    plot_bar_charts(df, output_dir)

    print(f"Wrote: {out_csv}")
    print(f"Counts in: {output_dir / 'counts'}")
    print(f"Plots in: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
