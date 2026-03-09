from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import gc
import json
import pandas as pd
import SimpleITK as sitk

from dataset_index import build_index, folder_dialog
from read_dicom import load_dicom, load_metadata
from kidney_features import kidneys_features
from extract_slices import extract_slices
from resolve_errors import resolve_index_errors


def _series_tag(case_id: str, phase: str) -> str:
    return f"{case_id}_{phase}"


def main(
    *,
    root_dir: Optional[Path] = None,
    outputs_root: Path = Path("odata/02_inter"),
    device: str = "cpu",
    fast_totalseg: bool = True,
    overwrite: bool = False,
) -> Dict[str, object]:
    """
    Pipeline: index → process each series.
    
    Processing order per case: VENOUS → ARTERIAL → LATE
    (venous cached + metadata alignment for arterial/late).
    """
    if root_dir is None:
        root_dir = folder_dialog()
        
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    index_root = outputs_root / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    out_index_csv = index_root / "dataset_index.csv"
    out_failed_csv = index_root / "failed_indexing.csv"
    
    features_csv = outputs_root / "kidney_features.csv"
    tensors_root = outputs_root / "tensors_15ch"
    tensors_root.mkdir(parents=True, exist_ok=True)
    segmentations_root = outputs_root / "segmentations"
    segmentations_root.mkdir(parents=True, exist_ok=True)
    failed_processing_csv = index_root / "failed_processing.csv"
    
    # Load or build index
    # 1) Build dataset index (auto)
    '''    
    print("[1/4] Building dataset index...")
    db = build_index(root_dir, out_failed_csv=out_failed_csv, out_index_csv=out_index_csv)
    print(f"      Found {len(db)} series across {db['case_id'].nunique() if len(db) > 0 else 0} cases\n")

    # 2) Resolve indexing errors (duplicates, mixed series, failed cases)
    print("[2/4] Resolving indexing errors...")
    mixed_csv = index_root / "mixed_series.csv"
    data_dist_csv = outputs_root / "data_distribution" / "dicom_metadata_per_volume.csv"
    out_final_csv = index_root / "dataset_index_final.csv"
    
    # Check if error resolution is needed
    has_errors = (out_failed_csv.exists() and pd.read_csv(out_failed_csv).shape[0] > 0) or \
                 (mixed_csv.exists() and pd.read_csv(mixed_csv).shape[0] > 0) or \
                 (db["note"].astype(str).str.contains("DUPLICATE|MIXED", na=False).any())
    
    if has_errors:
        print(f"      Errors detected, launching interactive resolution...")
        ds = resolve_index_errors(
            index_csv=out_index_csv,
            failed_csv=out_failed_csv,
            mixed_csv=mixed_csv,
            data_dist_csv=data_dist_csv,
            out_csv=out_final_csv
        )
        print(f"      ✓ Resolved: {len(ds)} clean entries\n")
    else:
        print(f"      No errors detected, skipping resolution\n")
        ds = db.copy()
        ds.to_csv(out_final_csv, index=False)
    '''    
    
    ds = pd.read_csv('outputs/index/dataset_index_final.csv', dtype={"case_id": str})

    # Pre-index dataset for O(1) lookup (instead of O(n) search)
    ds_indexed = {}
    for idx, row in ds.iterrows():
        key = (row["case_id"], row["phase"])
        ds_indexed[key] = row
    print(f"Pre-indexed {len(ds_indexed)} entries for fast lookup")
    
    # Load existing features if not overwriting
    print("\n[3/4] Processing series...")
    rows: List[dict] = []
    existing_df = None
    if features_csv.exists() and not overwrite:
        existing_df = pd.read_csv(features_csv)
    
    # Local cache variables (reset per case)
    current_case_id = None
    current_venous_kr = None  # kr_zyx from venous phase
    current_spacing_dict = None  # Metadata: spacing for all 3 phases
    current_origin_dict = None   # Metadata: origin for all 3 phases
    
    def _skip_if_processed(series_name: str) -> bool:
        if existing_df is None or "series_name" not in existing_df.columns:
            return False
        return (existing_df["series_name"] == series_name).any()
    
    def _append_row(csv_path: Path, row_dict: dict) -> None:
        df_one = pd.DataFrame([row_dict])
        write_header = not csv_path.exists()
        df_one.to_csv(csv_path, mode="a", header=write_header, index=False)
    
    # Process each series (using itertuples for 5-10x speed vs iterrows)
    for entry in ds.itertuples():
        series_name = _series_tag(entry.case_id, entry.phase)
        
        # Skip if already processed
        if _skip_if_processed(series_name) and not overwrite:
            print(f"      [{entry.Index+1}/{len(ds)}] Skipping {series_name}")
            continue
        
        print(f"      [{entry.Index+1}/{len(ds)}] {series_name}:", end=" ", flush=True)
        
        try:
            # Extract base case_id (already just number from _reorder_phases)
            case_id = entry.case_id  # "1" instead of "1_arterial"
            
            # If case changed, reset cache
            if current_case_id != case_id:
                current_case_id = case_id
                current_venous_kr = None
                current_spacing_dict = None
                current_origin_dict = None
            
            # Load current image
            vol, vol_sitk = load_dicom(
                entry.dicom_dir,
                series_uid=entry.series_uid,
                return_sitk_image=True
            )
            
            # Load metadata only ONCE per case (in venous phase) - WITHOUT loading full volumes
            if entry.phase == "venous":
                spacing_dict = {}
                origin_dict = {}
                
                for phase_name in ["arterial", "venous", "late"]:
                    # O(1) lookup instead of O(n) search
                    key = (case_id, phase_name)
                    if key in ds_indexed:
                        phase_entry = ds_indexed[key]
                        try:
                            # Read ONLY metadata (no pixel data) - saves ~2-3GB per phase
                            spacing_xyz, origin_xyz = load_metadata(
                                phase_entry["dicom_dir"],
                                phase_entry["series_uid"]
                            )
                            spacing_dict[phase_name] = spacing_xyz
                            origin_dict[phase_name] = origin_xyz
                        except Exception:
                            pass
                
                # Store metadata for later phases
                current_spacing_dict = spacing_dict
                current_origin_dict = origin_dict
            else:
                # Reuse metadata from venous phase
                spacing_dict = current_spacing_dict
                origin_dict = current_origin_dict
            
            # Extract kidney features (pass cached venous if available)
            # Returns tuple: (row_dict, kr_zyx)
            row, kr_zyx = kidneys_features(
                vol, vol_sitk,
                entry.dicom_dir,
                case_id=case_id,
                phase=entry.phase,
                device=device,
                fast=fast_totalseg,
                segmentations_root=segmentations_root,
                resample_factor=3.0,
                spacing_dict=spacing_dict,
                origin_dict=origin_dict,
                kr_venous_cached=current_venous_kr,
            )
            
            # If this is venous phase, save kr_zyx for next phases
            if entry.phase == "venous":
                current_venous_kr = kr_zyx
                if current_venous_kr is not None:
                    print(f"[CACHE] Stored venous segmentation for case {case_id}", end=" ", flush=True)
            
            row["series_name"] = series_name
            row["case_id"] = entry.case_id
            row["phase"] = entry.phase
            rows.append(row)
            
            if not overwrite:
                _append_row(features_csv, row)
            print("✓", end="", flush=True)
            
            # Extract tensor slices
            central = row.get("right_kidney_percentile_slices", [])
            if central and len(central) == 5:
                extract_slices(
                    vol,
                    entry.dicom_dir,
                    central_slices=list(central),
                    case_id=entry.case_id,
                    phase=entry.phase,
                    out_root=tensors_root,
                    level=60.0,
                    width=400.0,
                    save_individual_slices=False,
                )
                print("✓", flush=True)
            else:
                print("(no slices)", flush=True)
            
        except Exception as e:
            print(f"✗ {type(e).__name__}", flush=True)
            _append_row(failed_processing_csv, {
                "series_name": series_name,
                "case_id": entry.case_id,
                "phase": entry.phase,
                "dicom_dir": entry.dicom_dir,
                "error": type(e).__name__,
                "message": str(e)
            })
        finally:
            # Clean memory
            gc.collect()
    
    # Summary
    print("\n[4/4] Writing run summary...")
    if rows:
        if overwrite:
            df_new = pd.DataFrame(rows)
            df_new.to_csv(features_csv, index=False)
        print(f"      Processed {len(rows)} new series")
    else:
        print("      No new series processed")
    
    summary = {
        "root_dir": str(root_dir),
        "outputs_root": str(outputs_root),
        "num_series_processed": len(rows),
        "num_failed": failed_processing_csv.stat().st_size if failed_processing_csv.exists() else 0,
        "device": device,
        "fast_totalseg": fast_totalseg,
    }
    (outputs_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Complete ===\n")

    return summary


if __name__ == "__main__":    
    summary = main(
        root_dir=None,
        outputs_root=Path("data/02_inter"),
        device="gpu",
        fast_totalseg=True,
        overwrite=False,
    )
    print(json.dumps(summary, indent=2))

