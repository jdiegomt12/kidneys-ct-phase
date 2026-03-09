#!/usr/bin/env python3
"""
Process kidneys using metadata-based alignment:
1. Segment kidney_right in VENOUS only
2. Project to arterial/late using DICOM metadata
3. Compute features for each phase
4. Extract slices for each phase
5. Generate validation plots
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_processing"))

from read_dicom import load_dicom
from kidney_features import kidneys_features, DEFAULT_PERCENTILES
from extract_slices import extract_slices
from segment_kidneys import segment_kidneys_metadata_aligned


class VolumeWrapper:
    """Minimal wrapper to mimic the vol object expected by kidney_features."""
    def __init__(self, hu_zyx: np.ndarray, spacing_xyz: Tuple[float, float, float], series_uid: str):
        self.hu_zyx = hu_zyx
        self.spacing_xyz = spacing_xyz
        self.series_uid = series_uid


def process_case_with_metadata_alignment(
    case_id: str,
    index_df: pd.DataFrame,
    output_dir: Path,
    features_output_dir: Path,
    device: str = "cpu",
) -> bool:
    """
    Process a single case with metadata-based alignment.
    
    Returns True if successful, False otherwise.
    """
    print("\n" + "="*70)
    print(f"METADATA-ALIGNED KIDNEYS - CASE {case_id}")
    print("="*70)
    
    phases = ["arterial", "venous", "late"]
    
    # Load volumes and metadata
    print("\n[1] LOADING VOLUMES & METADATA")
    vol_sitk_dict = {}
    spacing_dict = {}
    origin_dict = {}
    hu_dict = {}
    vol_objs = {}
    
    for phase in phases:
        print(f"  [{phase.upper()}] Loading...", end=" ", flush=True)
        try:
            case_entries = index_df[index_df["case_id"] == int(case_id)]
            entry = case_entries[case_entries["phase"] == phase]
            
            if entry.empty:
                print("NOT FOUND")
                return False
            
            dicom_dir = entry.iloc[0]["dicom_dir"]
            series_uid = entry.iloc[0]["series_uid"]
            
            hu_array, vol_sitk = load_dicom(dicom_dir, series_uid=series_uid, return_sitk_image=True)
            
            vol_sitk_dict[phase] = vol_sitk
            spacing_dict[phase] = vol_sitk.GetSpacing()
            origin_dict[phase] = vol_sitk.GetOrigin()
            hu_dict[phase] = hu_array
            vol_objs[phase] = VolumeWrapper(hu_array, spacing_dict[phase], series_uid)
            
            print(f"OK (shape={hu_array.shape})")
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    if "venous" not in vol_sitk_dict:
        print("\n[SKIP] Missing venous volume")
        return False
    
    # Compute features for each phase using metadata alignment
    print("\n[2] COMPUTING FEATURES WITH METADATA ALIGNMENT")
    features_rows = []
    
    for phase in phases:
        if phase not in vol_sitk_dict:
            continue
        
        print(f"  [{phase.upper()}]", end=" ", flush=True)
        
        try:
            case_entries = index_df[index_df["case_id"] == int(case_id)]
            entry = case_entries[case_entries["phase"] == phase]
            dicom_dir = entry.iloc[0]["dicom_dir"]
            
            # Compute features with metadata alignment
            features = kidneys_features(
                vol_objs[phase],
                vol_sitk_dict[phase],
                dicom_dir,
                case_id,
                phase,
                device=device,
                fast=True,
                percentiles=DEFAULT_PERCENTILES,
                use_metadata_alignment=True,  # Enable metadata alignment
                vol_sitk_dict=vol_sitk_dict,
                spacing_dict=spacing_dict,
                origin_dict=origin_dict,
            )
            
            features["phase"] = phase
            features_rows.append(features)
            
            kr_vol = features.get("kidney_right_volume_ml", np.nan)
            print(f"OK (kr_volume={kr_vol:.1f}ml)")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if not features_rows:
        print("[SKIP] No features computed")
        return False
    
    # Save features
    print("\n[3] SAVING FEATURES")
    features_df = pd.DataFrame(features_rows)
    features_csv = features_output_dir / f"{case_id}_features_aligned.csv"
    features_df.to_csv(features_csv, index=False)
    print(f"  Saved: {features_csv}")
    
    # Extract slices for each phase
    print("\n[4] EXTRACTING SLICES")
    for phase in phases:
        if phase not in vol_sitk_dict:
            continue
        
        print(f"  [{phase.upper()}]", end=" ", flush=True)
        
        try:
            # Get right_pct from features
            phase_features = [f for f in features_rows if f["phase"] == phase][0]
            right_pct = phase_features["right_kidney_percentile_slices"]
            
            if not right_pct:
                print("SKIP (no valid percentile slices)")
                continue
            
            case_entries = index_df[index_df["case_id"] == int(case_id)]
            entry = case_entries[case_entries["phase"] == phase]
            dicom_dir = entry.iloc[0]["dicom_dir"]
            
            tensor_path = extract_slices(
                vol_objs[phase],
                dicom_dir,
                central_slices=right_pct,
                case_id=case_id,
                phase=phase,
                out_root=features_output_dir / "tensors",
            )
            
            print(f"OK -> {tensor_path.name}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Generate validation plots
    print("\n[5] GENERATING VALIDATION PLOTS")
    try:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f"Case {case_id}: Kidney alignment via metadata\n(Green contour = venous kidney projected to each phase)", fontsize=12)
        
        # Re-compute masks for visualization
        masks_dict = segment_kidneys_metadata_aligned(
            case_id,
            vol_sitk_dict,
            spacing_dict,
            origin_dict,
            device=device,
            fast=True,
        )
        
        for row, phase in enumerate(phases):
            if phase not in vol_sitk_dict or phase not in masks_dict:
                continue
            
            vol_arr = sitk.GetArrayFromImage(vol_sitk_dict[phase])
            kr_zyx, kl_zyx = masks_dict[phase]
            phase_features = [f for f in features_rows if f["phase"] == phase][0]
            
            z_min = int(phase_features["kidney_right_z_min"])
            z_max = int(phase_features["kidney_right_z_max"])
            z_center = int((z_min + z_max) / 2)
            
            z_indices = [z_min, z_center, z_max]
            labels = ["Min", "Center", "Max"]
            
            for col, (z_idx, label) in enumerate(zip(z_indices, labels)):
                ax = axes[row, col]
                
                if z_idx < 0 or z_idx >= vol_arr.shape[0]:
                    ax.text(0.5, 0.5, "Out of range", ha="center", va="center", color="red")
                    ax.set_title(f"{phase.upper()} - {label}\nz={z_idx} (OUT OF RANGE)", color="red")
                    ax.set_facecolor('#ffeeee')
                    ax.set_axis_off()
                    continue
                
                # Show slice
                slice_img = vol_arr[z_idx]
                ax.imshow(slice_img, cmap="gray", vmin=-100, vmax=400)
                
                # Overlay kidney mask
                if kr_zyx[z_idx].sum() > 0:
                    ax.contour(kr_zyx[z_idx], levels=[0.5], colors="lime", linewidths=2, alpha=0.9)
                
                ax.set_title(f"{phase.upper()} - {label}\nz-idx={z_idx}")
                ax.set_axis_off()
        
        plt.tight_layout()
        plot_path = output_dir / f"{case_id}_alignment_validation.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"  WARNING: Could not generate plot: {e}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Process kidneys with metadata-based alignment")
    parser.add_argument("--case", required=True, help="Case ID or 'all'")
    parser.add_argument("--device", default="cpu", help="Device (cpu/gpu)")
    parser.add_argument("--output-dir", default="outputs/metadata_alignment_processed", help="Output directory for plots")
    parser.add_argument("--features-dir", default="outputs/metadata_alignment_processed", help="Features/tensors output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load index
    index_csv = Path("outputs_old/index/dataset_index_final.csv")
    if not index_csv.exists():
        print(f"ERROR: {index_csv} not found")
        return
    
    index_df = pd.read_csv(index_csv)
    
    case_arg = str(args.case).strip().lower()
    if case_arg == "all":
        case_ids = sorted(index_df["case_id"].dropna().astype(int).unique().tolist())
        print(f"Processing all cases: {len(case_ids)}")
    else:
        case_ids = [int(args.case)]
    
    ok_count = 0
    fail_count = 0
    
    for cid in case_ids:
        if process_case_with_metadata_alignment(str(cid), index_df, output_dir, features_dir, device=args.device):
            ok_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print(f"  Success: {ok_count}")
    print(f"  Failed/Skipped: {fail_count}")
    print(f"  Output dirs: {output_dir}, {features_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

