"""
Segment kidney right in VENOUS phase only, then project mask to arterial/late 
using DICOM metadata geometry (origin, spacing, direction).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from segment_kidneys import segment_kidneys, xyz2zyx


def physical_z_to_slice_index(z_mm: float, spacing: Tuple[float, float, float], origin: Tuple[float, float, float]) -> float:
    """Convert physical z coordinate (mm) to slice index."""
    z_spacing = spacing[2]
    z_origin = origin[2]
    if z_spacing == 0:
        return 0
    return (z_mm - z_origin) / z_spacing


def project_mask_to_phase(
    source_mask_zyx: np.ndarray,
    source_spacing: Tuple[float, float, float],
    source_origin: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    target_origin: Tuple[float, float, float],
    target_vol_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Project a 3D mask from source coordinate space to target coordinate space.
    
    Strategy:
    - For each z-slice in source mask that has voxels:
      1. Get its physical z coordinate (mm)
      2. Convert to target phase's z-index
      3. For x,y: resample the mask slice to match target phase dimensions
      4. Place it in the target volume at the converted z-index
    """
    target_mask_zyx = np.zeros(target_vol_shape, dtype=np.uint8)
    
    # Get z-indices that have mask voxels in source
    z_indices_source = np.where(source_mask_zyx.sum(axis=(1, 2)) > 0)[0]
    
    if len(z_indices_source) == 0:
        return target_mask_zyx
    
    for z_src in z_indices_source:
        # Convert source z-index to physical mm
        z_mm = source_origin[2] + z_src * source_spacing[2]
        
        # Convert physical mm to target z-index
        z_target_idx = physical_z_to_slice_index(z_mm, target_spacing, target_origin)
        z_target_idx_int = int(np.round(z_target_idx))
        
        # Check bounds
        if not (0 <= z_target_idx_int < target_vol_shape[0]):
            continue
        
        # Get source slice and resample to target x,y dimensions
        source_slice = source_mask_zyx[z_src]
        if source_slice.sum() == 0:
            continue
        
        # Resample x,y if needed
        if source_slice.shape != (target_vol_shape[1], target_vol_shape[2]):
            # Compute scale factors (y, x)
            scale_y = target_vol_shape[1] / source_slice.shape[0]
            scale_x = target_vol_shape[2] / source_slice.shape[1]
            target_slice = ndimage.zoom(source_slice.astype(float), (scale_y, scale_x), order=1) > 0.5
            target_slice = target_slice.astype(np.uint8)
        else:
            target_slice = source_slice.astype(np.uint8)
        
        target_mask_zyx[z_target_idx_int] = target_slice
    
    return target_mask_zyx


def segment_and_align_kidneys_metadata(
    case_id: str,
    vol_sitk_dict: Dict[str, sitk.Image],  # {"arterial": img, "venous": img, "late": img}
    spacing_dict: Dict[str, Tuple[float, float, float]],
    origin_dict: Dict[str, Tuple[float, float, float]],
    *,
    device: str = "cpu",
    fast: bool = True,
    resample_factor: Optional[float] = None,
    keep_debug_dir: bool = False,
    segmentations_root: Optional[Path] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Segment kidney_right in VENOUS phase only, then project to arterial/late 
    using metadata-based coordinate transformation.
    
    Returns dict: {phase: (kr_projected_zyx, kl_dummy_zyx)} for each phase
    Note: left kidney is always empty (dummy) since we only segment right in venous.
    """
    
    if "venous" not in vol_sitk_dict:
        raise ValueError("venous phase required")
    
    # Segment ONLY on venous phase for kidney_right
    print("[METADATA ALIGNMENT] Segmenting kidney_right in VENOUS phase...")
    kr_venous_zyx, kl_venous_zyx, resample_factor_used, debug_dir = segment_kidneys(
        vol_sitk_dict["venous"],
        case_id=case_id,
        phase="venous",
        device=device,
        fast=fast,
        resample_factor=resample_factor,
        keep_debug_dir=keep_debug_dir,
        segmentations_root=segmentations_root,
    )
    
    # Project kidney_right to other phases
    print("[METADATA ALIGNMENT] Projecting kidney_right to arterial/late phases...")
    result = {}
    
    for phase in ["arterial", "venous", "late"]:
        if phase not in vol_sitk_dict:
            continue
        
        if phase == "venous":
            # Use original segmentation
            result[phase] = (kr_venous_zyx, kl_venous_zyx)
        else:
            # Project venous kidney_right to this phase
            vol_shape = sitk.GetArrayFromImage(vol_sitk_dict[phase]).shape
            
            kr_projected = project_mask_to_phase(
                kr_venous_zyx,
                source_spacing=spacing_dict["venous"],
                source_origin=origin_dict["venous"],
                target_spacing=spacing_dict[phase],
                target_origin=origin_dict[phase],
                target_vol_shape=vol_shape,
            )
            
            # Left kidney is always empty (we don't have it)
            kl_dummy = np.zeros_like(kr_projected)
            
            result[phase] = (kr_projected, kl_dummy)
            print(f"  [{phase.upper()}] projected mask with {kr_projected.sum()} voxels")
    
    return result
