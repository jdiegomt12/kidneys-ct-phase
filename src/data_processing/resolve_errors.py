from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk


SIGNATURE_FIELDS = [
    "manufacturer",
    "model",
    "software_versions",
    "modality",
    "slice_thickness",
    "kvp",
    "reconstruction_diameter",
    "convolution_kernel",
    "pixel_spacing",
    "rows",
    "columns",
]

SIGNATURE_TAGS = {
    "manufacturer": "0008|0070",
    "model": "0008|1090",
    "software_versions": "0018|1020",
    "modality": "0008|0060",
    "slice_thickness": "0018|0050",
    "kvp": "0018|0060",
    "reconstruction_diameter": "0018|1100",
    "convolution_kernel": "0018|1210",
    "pixel_spacing": "0028|0030",
    "rows": "0028|0010",
    "columns": "0028|0011",
}


def _base_case_id(case_id: str) -> str:
    match = re.search(r"_(\d+)$", case_id)
    if match:
        return case_id[: -len(match.group(0))]
    return case_id


def _normalize_value(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _read_signature(filepath: str) -> Dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(filepath)
    reader.ReadImageInformation()

    signature: Dict[str, str] = {}
    for name, tag in SIGNATURE_TAGS.items():
        if reader.HasMetaDataKey(tag):
            signature[name] = _normalize_value(reader.GetMetaData(tag))
        else:
            signature[name] = ""
    return signature


def _load_data_distribution(path: Optional[Path]) -> Dict[Tuple[str, str], Dict[str, str]]:
    if path is None or not path.exists():
        return {}

    df = pd.read_csv(path)
    dist: Dict[Tuple[str, str], Dict[str, str]] = {}
    for _, row in df.iterrows():
        case_id = _base_case_id(str(row.get("case_id", "")))
        phase = str(row.get("phase", ""))
        signature = {k: _normalize_value(row.get(k, "")) for k in SIGNATURE_FIELDS}
        dist[(case_id, phase)] = signature
    return dist


def _expected_signature(
    case_id: str,
    phase: str,
    dist: Dict[Tuple[str, str], Dict[str, str]],
) -> Dict[str, str]:
    expected: Dict[str, str] = {}
    for other_phase in ("arterial", "venous", "late"):
        if other_phase == phase:
            continue
        sig = dist.get((case_id, other_phase))
        if not sig:
            continue
        for key, val in sig.items():
            if not val:
                continue
            if key not in expected:
                expected[key] = val
            elif expected[key] != val:
                expected[key] = ""
    return expected


def _signature_score(candidate: Dict[str, str], expected: Dict[str, str]) -> int:
    score = 0
    for key, expected_val in expected.items():
        if not expected_val:
            continue
        if candidate.get(key, "") == expected_val:
            score += 1
    return score


def _choose_any_file(dicom_dir: Path) -> Optional[str]:
    try:
        names = sorted([f for f in os.listdir(dicom_dir) if os.path.isfile(dicom_dir / f)])
    except Exception:
        return None
    for name in names:
        path = str(dicom_dir / name)
        if _is_dicom(path):
            return path
    return None


def _is_dicom(filepath: str) -> bool:
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def _load_mixed_map(mixed_csv: Path) -> Dict[Tuple[str, str, str], List[dict]]:
    if not mixed_csv.exists():
        return {}
    df = pd.read_csv(mixed_csv)
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for _, row in df.iterrows():
        key = (str(row.get("case_id", "")), str(row.get("phase", "")), str(row.get("dicom_dir", "")))
        files_str = str(row.get("files", ""))
        files = [f for f in files_str.split(";") if f]
        groups.setdefault(key, []).append(
            {
                "series_uid": str(row.get("series_uid", "")),
                "n_files": int(row.get("n_files", 0)),
                "files": files,
            }
        )
    return groups


def _load_volume_from_files(files: List[str]) -> Optional[np.ndarray]:
    if not files:
        return None
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)


def _load_volume_from_dir(dicom_dir: Path) -> Optional[np.ndarray]:
    try:
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir))
    except Exception:
        return None
    if not names:
        return None
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)


def _show_volume(volume: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if volume is None:
        return

    window_center = 60.0
    window_width = 400.0
    vmin = window_center - (window_width / 2.0)
    vmax = window_center + (window_width / 2.0)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    idx = volume.shape[0] // 2
    img = ax.imshow(volume[idx], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "slice", 0, volume.shape[0] - 1, valinit=idx, valstep=1)

    def update(val):
        slice_idx = int(slider.val)
        img.set_data(volume[slice_idx])
        img.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    slider.on_changed(update)


def _prompt_choice(options: List[str], prompt: str) -> Optional[int]:
    print(prompt)
    for idx, label in enumerate(options, 1):
        print(f"  {idx}: {label}")
    print("  0: skip")
    try:
        choice = int(input("Select option: ").strip())
    except Exception:
        return None
    if choice <= 0 or choice > len(options):
        return None
    return choice - 1


def resolve_index_errors(
    index_csv: Path,
    failed_csv: Path,
    mixed_csv: Path,
    data_dist_csv: Optional[Path],
    out_csv: Path,
) -> pd.DataFrame:
    df_index = pd.read_csv(index_csv)
    df_failed = pd.read_csv(failed_csv) if failed_csv.exists() else pd.DataFrame()
    dist = _load_data_distribution(data_dist_csv)
    mixed_map = _load_mixed_map(mixed_csv)

    final_rows: List[dict] = []
    unresolved: List[dict] = []

    df_index["base_case_id"] = df_index["case_id"].astype(str).apply(_base_case_id)
    grouped = df_index.groupby(["base_case_id", "phase"], dropna=False)

    for (base_case_id, phase), group in grouped:
        group = group.reset_index(drop=True)
        
        # Skip MIXED_HEADERS cases - they'll be processed in the mixed section
        if any("MIXED_HEADERS" in str(r.get("note", "")) for _, r in group.iterrows()):
            continue
        
        if len(group) == 1 and group.loc[0, "source"] == "auto":
            final_rows.append(group.loc[0].drop(labels=["base_case_id"]).to_dict())
            continue

        # Duplicate candidates for the same phase
        candidates = []
        for _, row in group.iterrows():
            dicom_dir = Path(str(row.get("dicom_dir", "")))
            filepath = _choose_any_file(dicom_dir)
            if not filepath:
                continue
            signature = _read_signature(filepath)
            candidates.append({
                "row": row,
                "filepath": filepath,
                "signature": signature,
            })

        expected = _expected_signature(base_case_id, phase, dist)
        scores = [_signature_score(c["signature"], expected) for c in candidates]

        auto_choice = None
        if scores:
            best = max(scores)
            if scores.count(best) == 1 and best > 0:
                auto_choice = scores.index(best)

        if auto_choice is not None:
            chosen = candidates[auto_choice]["row"].drop(labels=["base_case_id"]).to_dict()
            chosen["case_id"] = base_case_id
            final_rows.append(chosen)
            continue

        # Manual review
        import matplotlib.pyplot as plt

        vols = []
        labels = []
        for idx, cand in enumerate(candidates, 1):
            vol = _load_volume_from_dir(Path(str(cand["row"].get("dicom_dir", ""))))
            vols.append(vol)
            labels.append(f"candidate {idx}")

        # Reference phase
        ref_row = df_index[(df_index["base_case_id"] == base_case_id) & (df_index["phase"] != phase)]
        if not ref_row.empty:
            ref_dir = Path(str(ref_row.iloc[0].get("dicom_dir", "")))
            ref_vol = _load_volume_from_dir(ref_dir)
            if ref_vol is not None:
                vols.append(ref_vol)
                labels.append("reference")

        for vol, label in zip(vols, labels):
            _show_volume(vol, f"{base_case_id} {phase} - {label}")

        plt.show()
        choice = _prompt_choice(labels[: len(candidates)], f"Select {base_case_id} {phase}")
        if choice is None:
            unresolved.append({"case_id": base_case_id, "phase": phase, "reason": "manual_skip"})
            continue

        chosen = candidates[choice]["row"].drop(labels=["base_case_id"]).to_dict()
        chosen["case_id"] = base_case_id
        final_rows.append(chosen)

    # Mixed cases (manual only, with auto suggestion via metadata)
    processed_mixed = set()  # Track (case_id, phase, dicom_dir) to avoid duplicates
    
    for _, row in df_index.iterrows():
        note = str(row.get("note", ""))
        if "MIXED_HEADERS" not in note:
            continue

        base_case_id = _base_case_id(str(row.get("case_id", "")))
        phase = str(row.get("phase", ""))
        dicom_dir = Path(str(row.get("dicom_dir", "")))
        key = (str(row.get("case_id", "")), phase, str(dicom_dir))
        
        # Skip if already processed this mixed case
        if key in processed_mixed:
            continue
        processed_mixed.add(key)
        
        groups = mixed_map.get(key, [])
        if not groups:
            continue

        candidates = []
        for group in groups:
            files = [str(dicom_dir / f) for f in group["files"]]
            if not files:
                continue
            signature = _read_signature(files[0])
            candidates.append({
                "files": files,
                "n_files": group.get("n_files", 0),
                "signature": signature,
            })

        expected = _expected_signature(base_case_id, phase, dist)
        scores = [_signature_score(c["signature"], expected) for c in candidates]
        auto_choice = None
        if scores:
            best = max(scores)
            if scores.count(best) == 1 and best > 0:
                auto_choice = scores.index(best)

        import matplotlib.pyplot as plt

        labels = []
        for idx, cand in enumerate(candidates, 1):
            vol = _load_volume_from_files(cand["files"])
            labels.append(f"candidate {idx} ({cand['n_files']} slices)")
            _show_volume(vol, f"{base_case_id} {phase} - {labels[-1]}")

        ref_row = df_index[(df_index["base_case_id"] == base_case_id) & (df_index["phase"] != phase)]
        if not ref_row.empty:
            ref_dir = Path(str(ref_row.iloc[0].get("dicom_dir", "")))
            ref_vol = _load_volume_from_dir(ref_dir)
            if ref_vol is not None:
                labels.append("reference")
                _show_volume(ref_vol, f"{base_case_id} {phase} - reference")

        plt.show()
        if auto_choice is not None:
            print(f"Auto-suggested: {labels[auto_choice]}")

        choice = _prompt_choice(labels[: len(candidates)], f"Select mixed {base_case_id} {phase}")
        if choice is None:
            unresolved.append({"case_id": base_case_id, "phase": phase, "reason": "manual_skip_mixed"})
            continue

        # Update chosen row with correct series_uid from selected group
        selected_group = groups[choice]
        row_out = row.copy()
        row_out["case_id"] = base_case_id
        row_out["series_uid"] = selected_group["series_uid"]
        row_out["note"] = "MIXED_RESOLVED_MANUAL"
        final_rows.append(row_out.to_dict())

    # Handle failed cases
    for _, row in df_failed.iterrows():
        case_path = str(row.get("case_path", ""))
        case_id = str(row.get("case_id", ""))
        reason = str(row.get("reason", ""))
        if case_path and os.path.exists(case_path):
            os.startfile(case_path)
        print(f"Failed case {case_id}: {reason}")
        choice = input("Remove case? (y/n): ").strip().lower()
        if choice != "y":
            unresolved.append({"case_id": case_id, "phase": "", "reason": reason})

    df_final = pd.DataFrame(final_rows)
    
    # Remove duplicate (case_id, phase): keep resolved versions, drop MIXED_HEADERS originals
    df_final["base_case_id"] = df_final["base_case_id"].fillna("")
    
    cleaned_rows = []
    for (case_id, phase), group in df_final.groupby(["case_id", "phase"], sort=False):
        # Prioritize rows with base_case_id filled (manually resolved)
        resolved = group[group["base_case_id"] != ""]
        if not resolved.empty:
            cleaned_rows.append(resolved.iloc[0])
        else:
            # No resolved, take first row (avoid MIXED_HEADERS duplicates)
            non_mixed = group[~group["note"].astype(str).str.contains("MIXED_HEADERS", na=False)]
            if not non_mixed.empty:
                cleaned_rows.append(non_mixed.iloc[0])
            else:
                cleaned_rows.append(group.iloc[0])
    
    df_final = pd.DataFrame(cleaned_rows).reset_index(drop=True)
    df_final = df_final.sort_values(["case_id", "phase"]).reset_index(drop=True)
    df_final.to_csv(out_csv, index=False)
    
    print(f"\n✓ Resolved index saved: {len(final_rows)} → {len(df_final)} unique entries")
    print(f"  Output: {out_csv}")

    if unresolved:
        unresolved_csv = out_csv.parent / "unresolved_cases.csv"
        pd.DataFrame(unresolved).to_csv(unresolved_csv, index=False)
    
    return df_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve indexing errors")
    parser.add_argument("--index-csv", type=str, required=True)
    parser.add_argument("--failed-csv", type=str, required=True)
    parser.add_argument("--mixed-csv", type=str, required=True)
    parser.add_argument("--data-dist-csv", type=str, default=None)
    parser.add_argument("--out-csv", type=str, required=True)

    args = parser.parse_args()
    resolve_index_errors(
        index_csv=Path(args.index_csv),
        failed_csv=Path(args.failed_csv),
        mixed_csv=Path(args.mixed_csv),
        data_dist_csv=Path(args.data_dist_csv) if args.data_dist_csv else None,
        out_csv=Path(args.out_csv),
    )


if __name__ == "__main__":
    main()
