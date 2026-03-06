from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage


def plot(image: sitk.Image, title: str = ""):
    """Plot a 2D slice of a 3D image."""
    img_array = sitk.GetArrayFromImage(image)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_array[img_array.shape[0] // 2], cmap="gray")
    plt.title(title)
    plt.show()

def plot3d(image: sitk.Image, title: str = ""):
    """Plot a 3D image using maximum intensity projection."""
    img_array = sitk.GetArrayFromImage(image)
    mip = np.max(img_array, axis=0)  # Max intensity projection along z-axis
    plt.figure(figsize=(5, 5))
    plt.imshow(mip, cmap="gray")
    plt.title(title)
    plt.show()


def totalseg_kidneys(
    input_nii: Path,
    out_dir: Path,
    *,
    device: str = "cpu",
    fast: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "TotalSegmentator",
        "-i", str(input_nii),
        "-o", str(out_dir),
        "--task", "total",
        "--roi_subset", "kidney_left", "kidney_right",
        "--output_type", "nifti",
        "--device", "gpu" if device.lower() in ["gpu", "cuda"] else "cpu",
    ]
    if fast:
        cmd.append("-f")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        # Check memory errors (it has been happening a lot, cause of using float64 instead of uint8)
        if "MemoryError" in proc.stderr or "_ArrayMemoryError" in proc.stderr:
            raise MemoryError(
                f"TotalSegmentator ran out of memory.\n"
            )
        raise RuntimeError(
            f"TotalSegmentator failed (exit {proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
        )


def xyz2zyx(mask_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(mask_xyz, (2, 1, 0)).astype(np.uint8)


def segment_kidneys(
    vol_sitk: sitk.Image,
    *,
    case_id: str = "",
    phase: str = "",
    device: str = "cpu",
    fast: bool = True,
    resample_factor: Optional[float] = None,
    keep_debug_dir: bool = False,
    segmentations_root: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[Path]]:
    tmp = Path(tempfile.mkdtemp())
    input_nii = tmp / "ct.nii.gz"
    out_dir = tmp / "totalseg_out"

    # Store original size before any modification
    original_spacing = vol_sitk.GetSpacing()
    original_size = vol_sitk.GetSize()
    actual_resample_factor = 1.0

    try:
        vol_sitk_for_seg = vol_sitk
        
        if resample_factor is not None and resample_factor != 1.0:
            actual_resample_factor = float(resample_factor)
            # Multiply original spacing by the factor
            new_spacing = tuple(float(s * resample_factor) for s in original_spacing)
            new_size = [
                int(np.ceil(original_size[i] / resample_factor))
                for i in range(3)
            ]
            vol_sitk_for_seg = sitk.Resample(
                vol_sitk,
                new_size,
                sitk.Transform(),
                sitk.sitkLinear,
                vol_sitk.GetOrigin(),
                new_spacing,
                vol_sitk.GetDirection(),
                0,
                sitk.sitkFloat32,
            )
        
        # Convert to int16 to save memory
        vol_sitk_for_seg = sitk.Cast(vol_sitk_for_seg, sitk.sitkInt16)
        sitk.WriteImage(vol_sitk_for_seg, str(input_nii))
        totalseg_kidneys(input_nii, out_dir, device=device, fast=fast)

        kl_path = out_dir / "kidney_left.nii.gz"
        kr_path = out_dir / "kidney_right.nii.gz"
        if not kl_path.exists() or not kr_path.exists():
            raise FileNotFoundError(f"Missing expected outputs: {kl_path} / {kr_path}")

        kl_img = nib.load(str(kl_path))
        kr_img = nib.load(str(kr_path))
        kl_xyz = (np.asanyarray(kl_img.dataobj) > 0.5)
        kr_xyz = (np.asanyarray(kr_img.dataobj) > 0.5)

        # If we resampled, expand masks back to original size by repeating voxels
        if actual_resample_factor != 1.0:
            factor = int(round(actual_resample_factor))
            # Repeat each voxel 'factor' times in each dimension (xyz format: x, y, z)
            kl_xyz = np.repeat(np.repeat(np.repeat(kl_xyz, factor, axis=0), factor, axis=1), factor, axis=2)
            kr_xyz = np.repeat(np.repeat(np.repeat(kr_xyz, factor, axis=0), factor, axis=1), factor, axis=2)
            
            # Crop to exact original size (in case of rounding errors)
            kl_xyz = kl_xyz[:original_size[0], :original_size[1], :original_size[2]]
            kr_xyz = kr_xyz[:original_size[0], :original_size[1], :original_size[2]]
            
            # Update nib images with expanded data in xyz format
            kl_img = nib.Nifti1Image(kl_xyz.astype(np.uint8), affine=np.eye(4))
            kr_img = nib.Nifti1Image(kr_xyz.astype(np.uint8), affine=np.eye(4))

        kl_zyx = xyz2zyx(kl_xyz)
        kr_zyx = xyz2zyx(kr_xyz)

        # Save segmentations if requested (now in original size)
        if segmentations_root is not None and case_id and phase:
            seg_case_dir = Path(segmentations_root) / case_id
            seg_case_dir.mkdir(parents=True, exist_ok=True)
            out_kl = seg_case_dir / f"{case_id}_{phase}_kidney_left.nii.gz"
            out_kr = seg_case_dir / f"{case_id}_{phase}_kidney_right.nii.gz"
            nib.save(kl_img, str(out_kl))
            nib.save(kr_img, str(out_kr))

        if keep_debug_dir:
            return kl_zyx, kr_zyx, actual_resample_factor, tmp

        shutil.rmtree(tmp, ignore_errors=True)
        return kl_zyx, kr_zyx, actual_resample_factor, None
    
    except (MemoryError, RuntimeError, FileNotFoundError):
        # Clean up temp directory on failure
        if not keep_debug_dir:
            shutil.rmtree(tmp, ignore_errors=True)
        raise


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
    Project a 3D mask from source coordinate space to target coordinate space using metadata.
    
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


def segment_kidneys_metadata_aligned(
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
    print("[SEGMENT] Segmenting kidney_right in VENOUS phase only...")
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
    print("[ALIGN] Projecting kidney_right to other phases via metadata...")
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
            n_voxels = kr_projected.sum()
            print(f"  [{phase.upper()}] projected mask with {n_voxels} voxels")
    
    if not keep_debug_dir and debug_dir:
        shutil.rmtree(debug_dir, ignore_errors=True)
    
    return result


def compute_metadata_aligned_z_indices(
    kr_venous_zyx: np.ndarray,
    spacing_dict: Dict[str, Tuple[float, float, float]],
    origin_dict: Dict[str, Tuple[float, float, float]],
) -> Dict[str, Tuple[int, int, float]]:
    """
    Compute z-indices for kidney right across phases using metadata alignment.
    
    Strategy (same as test_alignment_metadata.py):
    1. Find z_min, z_max of kidney_right in VENOUS (in array indices)
    2. Convert those to physical mm using VENOUS spacing/origin
    3. For each phase: convert those physical mm to that phase's z-indices
    
    Returns: Dict[phase] -> (z_min_idx, z_max_idx, z_center_idx_float)
    """
    result = {}
    
    # Validate that venous metadata is available
    if "venous" not in spacing_dict or "venous" not in origin_dict:
        return result
    
    # Get z-range of kidney_right in VENOUS (mm)
    z_indices_venous = np.where(kr_venous_zyx.sum(axis=(1, 2)) > 0)[0]
    if len(z_indices_venous) == 0:
        # Empty kidney, return empty indices
        return result
    
    z_min_venous_idx = int(np.min(z_indices_venous))
    z_max_venous_idx = int(np.max(z_indices_venous))
    
    # Convert to physical mm in VENOUS phase
    z_min_venous_mm = origin_dict["venous"][2] + z_min_venous_idx * spacing_dict["venous"][2]
    z_max_venous_mm = origin_dict["venous"][2] + z_max_venous_idx * spacing_dict["venous"][2]
    z_center_venous_mm = (z_min_venous_mm + z_max_venous_mm) / 2.0
    
    # Convert to indices in each phase
    for phase in ["arterial", "venous", "late"]:
        if phase not in spacing_dict or phase not in origin_dict:
            continue
        
        # Convert physical mm to this phase's z-indices
        z_min_idx = physical_z_to_slice_index(z_min_venous_mm, spacing_dict[phase], origin_dict[phase])
        z_max_idx = physical_z_to_slice_index(z_max_venous_mm, spacing_dict[phase], origin_dict[phase])
        z_center_idx = physical_z_to_slice_index(z_center_venous_mm, spacing_dict[phase], origin_dict[phase])
        
        result[phase] = (
            int(np.round(z_min_idx)),
            int(np.round(z_max_idx)),
            float(z_center_idx),  # Keep as float for interpolation
        )
    
    return result
