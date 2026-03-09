#!/usr/bin/env python3
"""
Quick test: Align kidney across phases using DICOM metadata only.

No segmentation, no registration - just metadata geometry.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.read_dicom import load_dicom


def compute_mask_stats(mask: np.ndarray, spacing_xyz: Tuple[float, float, float], origin_xyz: Tuple[float, float, float] = None) -> Dict:
    """Compute z-range stats for a mask."""
    if mask is None or mask.sum() == 0:
        return {
            "z_min_idx": None,
            "z_max_idx": None,
            "z_center_idx": None,
            "z_min_mm": None,
            "z_max_mm": None,
            "z_center_mm": None,
        }
    
    coords = np.where(mask > 0)
    z_vals = coords[0]
    z_min_idx = int(np.min(z_vals))
    z_max_idx = int(np.max(z_vals))
    z_center_idx = float(np.mean(z_vals))
    
    result = {
        "z_min_idx": z_min_idx,
        "z_max_idx": z_max_idx,
        "z_center_idx": z_center_idx,
    }
    
    if origin_xyz is not None:
        z_spacing = spacing_xyz[2]
        z_origin = origin_xyz[2]
        z_min_mm = z_origin + z_min_idx * z_spacing
        z_max_mm = z_origin + z_max_idx * z_spacing
        z_center_mm = z_origin + z_center_idx * z_spacing
        result.update({
            "z_min_mm": z_min_mm,
            "z_max_mm": z_max_mm,
            "z_center_mm": z_center_mm,
        })
    else:
        result.update({
            "z_min_mm": None,
            "z_max_mm": None,
            "z_center_mm": None,
        })
    
    return result


def physical_z_to_slice_index(z_mm: float, spacing: Tuple[float, float, float], origin: Tuple[float, float, float]) -> float:
    """Convert physical z coordinate (mm) to slice index."""
    z_spacing = spacing[2]
    z_origin = origin[2]
    if z_spacing == 0:
        return 0
    return (z_mm - z_origin) / z_spacing


def load_kidney_mask(case_id: str, phase: str, kidney: str = "right") -> Optional[np.ndarray]:
    """Load existing kidney segmentation."""
    seg_path = Path(f"outputs_old/segmentations/{case_id}/{case_id}_{phase}_kidney_{kidney}.nii.gz")
    if not seg_path.exists():
        return None
    
    kidney_img = nib.load(str(seg_path))
    kidney_xyz = (np.asanyarray(kidney_img.dataobj) > 0.5).astype(np.uint8)
    kidney_zyx = np.transpose(kidney_xyz, (2, 1, 0)).astype(np.uint8)
    return kidney_zyx


def process_case(case_id: str, phases, output_dir: Path, index_df: pd.DataFrame) -> bool:
    print("\n" + "="*70)
    print(f"METADATA ALIGNMENT TEST - CASE {case_id}")
    print("="*70 + "\n")

    # Load volumes and metadata
    print("[LOADING VOLUMES & METADATA]")
    volumes_raw = {}
    spacing_dict = {}
    origin_dict = {}

    for phase in phases:
        print(f"  [{phase.upper()}] Loading...", end=" ", flush=True)
        try:
            case_entries = index_df[index_df["case_id"] == int(case_id)]
            entry = case_entries[case_entries["phase"] == phase]

            if entry.empty:
                print("NOT FOUND")
                continue

            dicom_dir = entry.iloc[0]["dicom_dir"]
            series_uid = entry.iloc[0]["series_uid"]

            _, vol_sitk = load_dicom(dicom_dir, series_uid=series_uid, return_sitk_image=True)
            volumes_raw[phase] = vol_sitk
            spacing_dict[phase] = vol_sitk.GetSpacing()
            origin_dict[phase] = vol_sitk.GetOrigin()
            print(f"OK (Origin={origin_dict[phase]}, Spacing={spacing_dict[phase]})")
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    if "venous" not in volumes_raw:
        print("\n[SKIP] Missing venous volume")
        return False

    # Load kidney mask (VENOUS reference)
    print("\n[LOADING KIDNEY SEGMENTATION]")
    print("  [VENOUS] Loading kidney_right...", end=" ", flush=True)
    venous_kidney = load_kidney_mask(case_id, "venous", "right")
    if venous_kidney is None:
        print("NOT FOUND")
        print("[SKIP] No venous kidney mask")
        return False
    print("OK")

    # Get kidney z-range in venous (mm)
    venous_stats = compute_mask_stats(venous_kidney, spacing_dict["venous"], origin_dict["venous"])
    venous_z_min_mm = venous_stats["z_min_mm"]
    venous_z_max_mm = venous_stats["z_max_mm"]

    print("\n[REFERENCE: VENOUS KIDNEY]")
    print(f"  z_min: {venous_z_min_mm:.1f}mm (idx={venous_stats['z_min_idx']})")
    print(f"  z_max: {venous_z_max_mm:.1f}mm (idx={venous_stats['z_max_idx']})")

    # Convert to indices in other phases
    print("\n[METADATA PROJECTION]")
    projected_indices = {}
    for phase in phases:
        if phase not in volumes_raw:
            continue

        z_min_idx_phase = physical_z_to_slice_index(venous_z_min_mm, spacing_dict[phase], origin_dict[phase])
        z_max_idx_phase = physical_z_to_slice_index(venous_z_max_mm, spacing_dict[phase], origin_dict[phase])
        z_center_mm = (venous_z_min_mm + venous_z_max_mm) / 2
        z_center_idx_phase = physical_z_to_slice_index(z_center_mm, spacing_dict[phase], origin_dict[phase])

        projected_indices[phase] = {
            "z_min_idx": int(np.round(z_min_idx_phase)),
            "z_max_idx": int(np.round(z_max_idx_phase)),
            "z_center_idx": int(np.round(z_center_idx_phase)),
        }

        # Verify conversion by converting back to mm
        z_min_mm_check = origin_dict[phase][2] + projected_indices[phase]['z_min_idx'] * spacing_dict[phase][2]
        z_center_mm_check = origin_dict[phase][2] + projected_indices[phase]['z_center_idx'] * spacing_dict[phase][2]

        # Check if indices are in valid range
        vol_shape = sitk.GetArrayFromImage(volumes_raw[phase]).shape
        in_range_min = 0 <= projected_indices[phase]['z_min_idx'] < vol_shape[0]
        in_range_max = 0 <= projected_indices[phase]['z_max_idx'] < vol_shape[0]
        in_range_center = 0 <= projected_indices[phase]['z_center_idx'] < vol_shape[0]

        print(f"  [{phase.upper()}] venous mm [{venous_z_min_mm:.1f}, {venous_z_max_mm:.1f}] -> indices [{projected_indices[phase]['z_min_idx']}, {projected_indices[phase]['z_center_idx']}, {projected_indices[phase]['z_max_idx']}]")
        print(f"           verify: z_min_mm={z_min_mm_check:.1f} (delta={abs(z_min_mm_check-venous_z_min_mm):.1f}mm), z_center_mm={z_center_mm_check:.1f} (delta={abs(z_center_mm_check-z_center_mm):.1f}mm)")
        print(f"           valid range: [0, {vol_shape[0]-1}] | min={in_range_min}, center={in_range_center}, max={in_range_max}")

    # Plot: venous kidney contour on all phases at converted indices
    print("\n[GENERATING VISUALIZATION]")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"Case {case_id}: Metadata-based alignment test\n(Green contour = venous kidney | Other phases show CT slices at converted z-indices)", fontsize=12)

    for row, phase in enumerate(phases):
        if phase not in volumes_raw:
            continue

        vol = volumes_raw[phase]
        arr = sitk.GetArrayFromImage(vol)

        # Get venous kidney z-range in THIS phase's indices
        z_min_phase = projected_indices[phase]["z_min_idx"]
        z_max_phase = projected_indices[phase]["z_max_idx"]
        z_range_phase = z_max_phase - z_min_phase
        
        # Percentiles: 15, 50, 85
        z_p15_phase = int(z_min_phase + 0.15 * z_range_phase)
        z_p50_phase = int(z_min_phase + 0.50 * z_range_phase)
        z_p85_phase = int(z_min_phase + 0.85 * z_range_phase)

        z_indices = [z_p15_phase, z_p50_phase, z_p85_phase]
        labels = ["P15", "P50", "P85"]

        # Get corresponding venous slices
        venous_z_min = venous_stats["z_min_idx"]
        venous_z_max = venous_stats["z_max_idx"]
        venous_z_range = venous_z_max - venous_z_min
        
        # Percentiles: 15, 50, 85
        venous_z_p15 = int(venous_z_min + 0.15 * venous_z_range)
        venous_z_p50 = int(venous_z_min + 0.50 * venous_z_range)
        venous_z_p85 = int(venous_z_min + 0.85 * venous_z_range)
        venous_z_indices = [venous_z_p15, venous_z_p50, venous_z_p85]

        for col, (z_idx, venous_z_idx, label) in enumerate(zip(z_indices, venous_z_indices, labels)):
            ax = axes[row, col]

            # Check if index is in valid range BEFORE clamping
            original_z_idx = z_idx
            is_out_of_range = z_idx < 0 or z_idx >= arr.shape[0]

            if is_out_of_range:
                # Show a placeholder with clear message
                ax.text(0.5, 0.5, f"OUT OF RANGE\nRequested: z={original_z_idx}\nValid: [0, {arr.shape[0]-1}]",
                        ha="center", va="center", fontsize=10, color="red")
                ax.set_title(f"{phase.upper()} - {label}\nz-idx={original_z_idx} (OUT OF RANGE)", fontsize=9, color="red")
                ax.set_facecolor('#ffeeee')
                ax.set_axis_off()
                continue

            # Show CT slice from current phase
            slice_img = arr[z_idx]
            ax.imshow(slice_img, cmap="gray", vmin=0, vmax=255)

            # Overlay venous kidney contour ONLY in venous phase
            if phase == "venous":
                if 0 <= venous_z_idx < venous_kidney.shape[0]:
                    venous_mask_slice = venous_kidney[venous_z_idx]
                    if venous_mask_slice.sum() > 0:
                        ax.contour(venous_mask_slice, levels=[0.5], colors="lime", linewidths=2, alpha=0.9)

            # Title
            title = f"{phase.upper()} - {label}\n"
            if phase == "venous":
                title += f"z-idx={z_idx}"
            else:
                title += f"z-idx={z_idx} (converted from venous mm)"
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()

    plt.tight_layout()
    output_path = output_dir / f"{case_id}_alignment_metadata.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

    print("\n" + "="*70)
    print("DONE")
    print("="*70 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Quick metadata alignment test (no segmentation)")
    parser.add_argument("--case", required=True, help="Case ID or 'all'")
    parser.add_argument("--output-dir", default="outputs/metadata_alignment_test", help="Output directory")
    args = parser.parse_args()

    phases = ["arterial", "venous", "late"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_csv = Path("outputs/index/dataset_index_final.csv")
    if not index_csv.exists():
        print("ERROR: outputs/index/dataset_index_final.csv not found")
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
        if process_case(str(cid), phases, output_dir, index_df):
            ok_count += 1
        else:
            fail_count += 1

    print("\n" + "="*70)
    print("SUMMARY")
    print(f"  Success: {ok_count}")
    print(f"  Failed/Skipped: {fail_count}")
    print(f"  Output dir: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
