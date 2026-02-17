from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


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
