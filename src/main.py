from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import gc
import json
import pandas as pd

from dataset_index import build_index, folder_dialog
from read_dicom import load_dicom
from kidney_features import kidneys_features
from extract_slices import extract_slices


def _series_tag(case_id: str, phase: str) -> str:
    return f"{case_id}_{phase}"


# Uso:
# inspect_bone_profile(mi_volumen_sitk)
def main(
    *,
    root_dir: Optional[Path] = None,
    outputs_root: Path = Path("outputs"),
    device: str = "cpu",
    fast_totalseg: bool = True,
    overwrite: bool = False,
) -> Dict[str, object]:
    """
    Returns a dict with summary + paths of artifacts.
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
    


    # 1) Build dataset index (auto)
    print("[1/3] Building dataset index...")
    ds = build_index(root_dir, out_failed_csv=out_failed_csv, out_index_csv=out_index_csv)
    print(f"      Found {len(ds)} series across {ds['case_id'].nunique() if len(ds) > 0 else 0} cases\n")

    # 2) Process each discovered series
    print("[2/3] Processing series...")
    rows: List[dict] = []
    failed_processing: List[dict] = []
    failed_processing_csv = index_root / "failed_processing.csv"
    checkpoint_path = index_root / "processing_checkpoint.txt"
    existing_df = None
    if features_csv.exists() and not overwrite:
        existing_df = pd.read_csv(features_csv)
        if "series_name" in existing_df.columns:
            existing_df = existing_df.drop_duplicates(["series_name"], keep="last")

    def _append_row(csv_path: Path, row_dict: dict) -> None:
        df_one = pd.DataFrame([row_dict])
        write_header = not csv_path.exists()
        df_one.to_csv(csv_path, mode="a", header=write_header, index=False)
        
    for idx, entry in ds.iterrows():
        series_name = _series_tag(entry["case_id"], entry["phase"])

        # a) features caching: skip if already present
        if existing_df is not None and "series_name" in existing_df.columns:
            if (existing_df["series_name"] == series_name).any() and not overwrite:
                print(f"      [{idx+1}/{len(ds)}] Skipping {series_name} (already processed)")
                continue

        print(f"      [{idx+1}/{len(ds)}] Processing {series_name}...", end=" ")
    
        try:
            vol, vol_sitk = load_dicom(entry["dicom_dir"], series_uid=entry["series_uid"], return_sitk_image=True)
            
            # compute features
            row = kidneys_features(
                vol, vol_sitk,
                entry["dicom_dir"],
                case_id=entry["case_id"],
                phase=entry["phase"],
                device=device,
                fast=fast_totalseg,
                keep_debug_dir=False,
                segmentations_root=segmentations_root,
                resample_factor=3.0  # Resample factor: multiply spacing by this factor
            )
            row["series_name"] = series_name  # enforce consistent naming
            row["case_id"] = entry["case_id"]
            row["phase"] = entry["phase"]
            rows.append(row)
            # save each row immediately to avoid losing progress
            if not overwrite:
                _append_row(features_csv, row)
            print("features processed", end=" ")

            # b) tensor caching: skip if exists
            tensor_dir = tensors_root / series_name
            tensor_path = tensor_dir / f"{series_name}_input_tensor.npy"
            if tensor_path.exists() and not overwrite:
                print("tensors cached")
                checkpoint_path.write_text(series_name)
                # Free memory
                del vol, vol_sitk, row
                gc.collect()
                continue

            central = row.get("right_kidney_percentile_slices", [])
            if not central or len(central) != 5:
                # cannot build 15ch input reliably
                print("(no valid slices)")
                checkpoint_path.write_text(series_name)
                # Free memory
                del vol, vol_sitk, row, central
                gc.collect()
                continue

            extract_slices(
                vol,
                entry["dicom_dir"],
                central_slices=list(central),
                case_id=entry["case_id"],
                phase=entry["phase"],
                out_root=tensors_root,
                level=60.0,
                width=400.0,
                save_individual_slices=False,
            )
            print("tensors processed")
            checkpoint_path.write_text(series_name)
            
            # Free memory explicitly
            del vol, vol_sitk, row
            if 'central' in locals():
                del central
            gc.collect()
        
        except MemoryError as e:
            print(f"FAILED (out of memory)")
            failed_row = {
                "series_name": series_name,
                "case_id": entry["case_id"],
                "phase": entry["phase"],
                "dicom_dir": entry["dicom_dir"],
                "reason": "memory_error",
                "error": str(e)
            }
            failed_processing.append(failed_row)
            _append_row(failed_processing_csv, failed_row)
            checkpoint_path.write_text(series_name)
            # Free any allocated memory
            if 'vol' in locals():
                del vol
            if 'vol_sitk' in locals():
                del vol_sitk
            if 'row' in locals():
                del row
            gc.collect()
            continue
        
        except Exception as e:
            print(f"FAILED ({type(e).__name__})")
            failed_row = {
                "series_name": series_name,
                "case_id": entry["case_id"],
                "phase": entry["phase"],
                "dicom_dir": entry["dicom_dir"],
                "reason": type(e).__name__,
                "error": str(e)
            }
            failed_processing.append(failed_row)
            _append_row(failed_processing_csv, failed_row)
            checkpoint_path.write_text(series_name)
            # Free any allocated memory
            if 'vol' in locals():
                del vol
            if 'vol_sitk' in locals():
                del vol_sitk
            if 'row' in locals():
                del row
            gc.collect()
            continue

    # append new rows (only needed if overwrite=True)
    if rows:
        if overwrite:
            df_new = pd.DataFrame(rows)
            df_new.to_csv(features_csv, index=False)
        print(f"\n      Saved {len(rows)} new entries to {features_csv.name}")
    else:
        print("\n      No new series to process")
    
    # Save failed processing log if any (only needed if overwrite=True)
    if failed_processing and overwrite:
        df_failed_proc = pd.DataFrame(failed_processing)
        df_failed_proc.to_csv(failed_processing_csv, index=False)
        print(f"      Saved {len(failed_processing)} failed cases to {failed_processing_csv.name}")

    # 3) Write a small run summary
    print("\n[3/3] Writing run summary...")
    summary = {
        "root_dir": str(root_dir),
        "outputs_root": str(outputs_root),
        "features_csv": str(features_csv),
        "failed_cases_csv": str(out_failed_csv),
        "num_index_entries": len(ds),
        "num_failed_cases": pd.read_csv(out_failed_csv).shape[0] if out_failed_csv.exists() else 0,
        "num_failed_processing": len(failed_processing),
        "processed_new_series": len(rows),
        "tensors_root": str(tensors_root),
        "device": device,
        "fast_totalseg": fast_totalseg,
        "overwrite": overwrite,
    }
    (outputs_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"      Summary saved to run_summary.json")
    print(f"\n=== Pipeline Complete ===")
    print(f"Processed: {summary['processed_new_series']} new series")
    print(f"Total indexed: {summary['num_index_entries']} series")
    print(f"Failed indexing: {summary['num_failed_cases']}")
    print(f"Failed processing: {summary['num_failed_processing']}\n")

    return summary

#%%
if __name__ == "__main__":    
    summary = main(
        root_dir=None,
        outputs_root=Path("outputs"),
        device="gpu",  # or "cpu"
        fast_totalseg=True,
        overwrite=False,
    )
    print(json.dumps(summary, indent=2))
# %%
