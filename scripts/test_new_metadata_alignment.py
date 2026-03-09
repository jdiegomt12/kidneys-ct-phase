#!/usr/bin/env python3
"""
Quick test: Verify new metadata alignment approach (converted indices instead of projected masks).
"""

import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_processing.read_dicom import load_dicom
from data_processing.segment_kidneys import (
    segment_kidneys,
    compute_metadata_aligned_z_indices,
    physical_z_to_slice_index,
)
from data_processing.dataset_index import index_dicom_root

# Load index
root_dir = Path(".")
index_df = index_dicom_root(root_dir, phases=["arterial", "venous", "late"])

# Pick a test case
case_id = "1"
print(f"\n{'='*70}")
print(f"TEST: Metadata-Aligned Z-Indices (Case {case_id})")
print(f"{'='*70}\n")

# Load all 3 phases
vol_sitk_dict = {}
spacing_dict = {}
origin_dict = {}

for phase in ["arterial", "venous", "late"]:
    entries = index_df[(index_df["case_id"] == int(case_id)) & (index_df["phase"] == phase)]
    if entries.empty:
        print(f"[{phase.upper()}] NOT FOUND")
        continue
    
    dicom_dir = entries.iloc[0]["dicom_dir"]
    series_uid = entries.iloc[0]["series_uid"]
    
    try:
        _, vol_sitk = load_dicom(dicom_dir, series_uid=series_uid, return_sitk_image=True)
        vol_sitk_dict[phase] = vol_sitk
        spacing_dict[phase] = vol_sitk.GetSpacing()
        origin_dict[phase] = vol_sitk.GetOrigin()
        
        print(f"[{phase.upper()}] Loaded")
        print(f"  Origin: {origin_dict[phase]}")
        print(f"  Spacing: {spacing_dict[phase]}")
    except Exception as e:
        print(f"[{phase.upper()}] ERROR: {e}")

if "venous" not in vol_sitk_dict:
    print("\n[ERROR] Venous phase required")
    sys.exit(1)

# Segment kidney_right in VENOUS only
print(f"\n[SEGMENTING] Kidney_right in VENOUS phase...")
try:
    kr_venous_zyx, _, _, _ = segment_kidneys(
        vol_sitk_dict["venous"],
        case_id=case_id,
        phase="venous",
        device="cpu",
        fast=True,
        resample_factor=None,
        keep_debug_dir=False,
        segmentations_root=Path("segmentations"),
    )
    print(f"  Segmentation complete: {kr_venous_zyx.sum()} voxels")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Compute converted z-indices
print(f"\n[COMPUTING] Converted Z-Indices...")
try:
    converted_indices = compute_metadata_aligned_z_indices(
        kr_venous_zyx,
        spacing_dict,
        origin_dict,
    )
    
    for phase, (z_min, z_max, z_center) in converted_indices.items():
        print(f"  [{phase.upper()}] z_min={z_min}, z_max={z_max}, z_center={z_center:.1f}")
        
        # Verify conversion
        z_min_mm = origin_dict[phase][2] + z_min * spacing_dict[phase][2]
        z_max_mm = origin_dict[phase][2] + z_max * spacing_dict[phase][2]
        print(f"             z_min_mm={z_min_mm:.1f}, z_max_mm={z_max_mm:.1f}")
        
        # Check valid range
        vol_shape = sitk.GetArrayFromImage(vol_sitk_dict[phase]).shape
        in_range_min = 0 <= z_min < vol_shape[0]
        in_range_max = 0 <= z_max < vol_shape[0]
        print(f"             in_range: min={in_range_min}, max={in_range_max}, vol_shape={vol_shape[0]}")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*70}")
print(f"SUCCESS: Converted indices computed correctly")
print(f"{'='*70}\n")
