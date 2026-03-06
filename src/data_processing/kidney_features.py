from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union
import time

import numpy as np
import SimpleITK as sitk

from segment_kidneys import segment_kidneys, compute_metadata_aligned_z_indices

PathLike = Union[str, Path]

DEFAULT_PERCENTILES = [0.15, 0.30, 0.50, 0.70, 0.85] # Moein et al. 2026

def mask_stats(
    hu_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Dict[str, float]:
    if mask_zyx.dtype != np.uint8:
        mask_zyx = (mask_zyx > 0).astype(np.uint8)

    coords = np.argwhere(mask_zyx > 0)  # z,y,x
    sx, sy, sz = spacing_xyz
    voxel_vol = float(sx * sy * sz)

    if coords.size == 0:
        return {
            "present": 0,
            "z_min": np.nan, "z_max": np.nan,
            "y_min": np.nan, "y_max": np.nan,
            "x_min": np.nan, "x_max": np.nan,
            "z_center": np.nan, "y_center": np.nan, "x_center": np.nan,
            "sx_mm": float(sx), "sy_mm": float(sy), "sz_mm": float(sz),
            "voxel_volume_mm3": voxel_vol,
            "voxels": 0,
            "volume_mm3": 0.0,
            "volume_ml": 0.0,
            "hu_mean": np.nan, "hu_std": np.nan, "hu_median": np.nan,
            "hu_min": np.nan, "hu_max": np.nan,
            "hu_p05": np.nan, "hu_p95": np.nan,
        }

    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]

    z_min, z_max = int(z.min()), int(z.max())
    y_min, y_max = int(y.min()), int(y.max())
    x_min, x_max = int(x.min()), int(x.max())

    z_center = float((z_min + z_max) / 2.0)
    y_center = float((y_min + y_max) / 2.0)
    x_center = float((x_min + x_max) / 2.0)

    hu_vals = hu_zyx[mask_zyx > 0].astype(np.float32)

    nvox = int(coords.shape[0])
    vol_mm3 = float(nvox * voxel_vol)

    return {
        "present": 1,
        "z_min": z_min, "z_max": z_max,
        "y_min": y_min, "y_max": y_max,
        "x_min": x_min, "x_max": x_max,
        "z_center": z_center, "y_center": y_center, "x_center": x_center,
        "sx_mm": float(sx), "sy_mm": float(sy), "sz_mm": float(sz),
        "voxel_volume_mm3": voxel_vol,
        "voxels": nvox,
        "volume_mm3": vol_mm3,
        "volume_ml": float(vol_mm3 / 1000.0),
        "hu_mean": float(np.mean(hu_vals)),
        "hu_std": float(np.std(hu_vals)),
        "hu_median": float(np.median(hu_vals)),
        "hu_min": float(np.min(hu_vals)),
        "hu_max": float(np.max(hu_vals)),
        "hu_p05": float(np.percentile(hu_vals, 5)),
        "hu_p95": float(np.percentile(hu_vals, 95)),
    }


def _prefix(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def percentile_slices(z_min: int, z_max: int, percentiles: List[float]) -> List[int]:
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        return []
    z_min_i, z_max_i = int(z_min), int(z_max)
    if z_max_i <= z_min_i:
        return [z_min_i for _ in percentiles]

    height = z_max_i - z_min_i
    idxs = [int(round(z_min_i + p * height)) for p in percentiles]
    return [min(max(i, z_min_i), z_max_i) for i in idxs]


def xyz2zyx(mask_xyz: np.ndarray) -> np.ndarray:
    """Convert (x,y,z) indexed array to (z,y,x) indexed array."""
    return np.transpose(mask_xyz, (2, 1, 0)).astype(np.uint8)


def load_cached_segmentations(
    segmentations_root: Path,
    case_id: str,
    phase: str,
) -> Tuple[np.ndarray, np.ndarray, float] | None:
    """
    Try to load cached kidney segmentations from NIfTI files.
    Returns (kl_zyx, kr_zyx, resample_factor) if both files exist, else None.
    Cached segmentations are already in original size, so factor is 1.0.
    """
    import nibabel as nib
    
    seg_case_dir = Path(segmentations_root) / case_id
    kl_path = seg_case_dir / f"{case_id}_{phase}_kidney_left.nii.gz"
    kr_path = seg_case_dir / f"{case_id}_{phase}_kidney_right.nii.gz"
    
    if kl_path.exists() and kr_path.exists():
        try:
            kl_img = nib.load(str(kl_path))
            kr_img = nib.load(str(kr_path))
            
            kl_xyz = (np.asanyarray(kl_img.dataobj) > 0.5)
            kr_xyz = (np.asanyarray(kr_img.dataobj) > 0.5)
            
            kl_zyx = xyz2zyx(kl_xyz)
            kr_zyx = xyz2zyx(kr_xyz)
            
            return kl_zyx, kr_zyx, 1.0
        except Exception:
            # If loading fails, fall back to segmenting
            return None
    return None


def kidneys_features(
    vol, vol_sitk,
    dicom_dir: PathLike,
    case_id: str,
    phase: str,
    *,
    device: str = "cpu",
    fast: bool = True,
    percentiles: List[float] = DEFAULT_PERCENTILES,
    resample_factor: float | None = None,
    keep_debug_dir: bool = False,
    segmentations_root: Path | None = None,
    spacing_dict: dict | None = None,   # Only metadata: {phase: spacing_xyz}
    origin_dict: dict | None = None,    # Only metadata: {phase: origin_xyz}
    kr_venous_cached: np.ndarray | None = None,  # Cached venous kidney_right from previous phase
) -> Dict[str, object]:
    """
    Extract kidney features and compute percentile slices.
    
    For VENOUS phase: segments kidney_right with TotalSegmentator, caches result.
    For ARTERIAL/LATE phases: uses cached venous segmentation + metadata alignment.
    
    Args:
        spacing_dict: Optional dict of spacing for ALL phases (metadata only)
        origin_dict: Optional dict of origin coordinates for ALL phases (metadata only)
    """
    global _VENOUS_SEGMENTATION_CACHE
    
    start_time = time.perf_counter()
    dicom_dir = Path(dicom_dir)
    series_name = dicom_dir.name

    # Try to load cached segmentations from disk first
    kl_zyx = None
    kr_zyx = None
    kr_venous_zyx = None
    resample_factor_used = 1.0
    debug_dir = None
    
    if segmentations_root is not None:
        cached_segs = load_cached_segmentations(
            Path(segmentations_root), case_id, phase
        )
        if cached_segs is not None:
            kl_zyx, kr_zyx, resample_factor_used = cached_segs
    
    # If not found in disk cache, compute segmentation
    if kl_zyx is None or kr_zyx is None:
        if phase == "venous":
            # VENOUS: always segment with TotalSegmentator
            kl_zyx, kr_zyx, resample_factor_used, debug_dir = segment_kidneys(
                vol_sitk,
                case_id=case_id,
                phase="venous",
                device=device,
                fast=fast,
                resample_factor=resample_factor,
                keep_debug_dir=keep_debug_dir,
                segmentations_root=segmentations_root,
            )
            # kr_zyx will be returned in the row dict for use by next phases
            
        elif phase in ("arterial", "late"):
            # ARTERIAL/LATE: use metadata alignment with cached venous
            
            # Check if venous is cached (passed as parameter)
            if kr_venous_cached is not None:
                kr_venous_zyx = kr_venous_cached
                print(f"[META] Using cached venous kidney_right for case {case_id}", flush=True)
            else:
                # Venous not cached - error (shouldn't happen if phases ordered correctly)
                raise ValueError(
                    f"Cannot align phase {phase}: venous segmentation not cached. "
                    f"Ensure venous phase is processed before arterial/late."
                )
            
            # Project venous segmentation to current phase
            if spacing_dict is None or origin_dict is None:
                raise ValueError(f"Cannot align to {phase}: need spacing_dict and origin_dict")
            
            from segment_kidneys import project_mask_to_phase
            
            vol_array = sitk.GetArrayFromImage(vol_sitk)
            kr_zyx = project_mask_to_phase(
                kr_venous_zyx,
                source_spacing=spacing_dict["venous"],
                source_origin=origin_dict["venous"],
                target_spacing=spacing_dict[phase],
                target_origin=origin_dict[phase],
                target_vol_shape=vol_array.shape,
            )
            kl_zyx = np.zeros_like(kr_zyx)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    # Compute statistics
    left = mask_stats(vol.hu_zyx, kl_zyx, vol.spacing_xyz)
    right = mask_stats(vol.hu_zyx, kr_zyx, vol.spacing_xyz)

    # Percentile slices: use converted indices for non-venous phases if metadata alignment active
    right_pct = None
    if phase in ("arterial", "late") and spacing_dict is not None and kr_venous_zyx is not None:
        try:
            converted_indices = compute_metadata_aligned_z_indices(
                kr_venous_zyx,
                spacing_dict,
                origin_dict,
            )
            if phase in converted_indices:
                z_min_converted, z_max_converted, z_center_converted = converted_indices[phase]
                right_pct = percentile_slices(z_min_converted, z_max_converted, percentiles)
                print(f"  [INDICES] Using converted z-indices for {phase}: [{z_min_converted}, {z_max_converted}]", flush=True)
        except (KeyError, ValueError, IndexError, TypeError) as e:
            print(f"  [INDICES WARNING] Fallback to mask-based ({type(e).__name__})", flush=True)
    
    # Default: use mask-based indices
    if right_pct is None:
        right_pct = percentile_slices(right["z_min"], right["z_max"], percentiles)
        phase="venous",
        device=device,

    elapsed = time.perf_counter() - start_time
    
    row: Dict[str, object] = {
        "series_name": series_name,
        "dicom_dir": str(dicom_dir),
        "series_uid": vol.series_uid,
        "right_kidney_percentile_slices": right_pct,
        "processing_time_seconds": round(elapsed, 3),
        "resample_factor": resample_factor_used,
    }
    row.update(_prefix(left, "kidney_left_"))
    row.update(_prefix(right, "kidney_right_"))
    if keep_debug_dir:
        row["debug_dir"] = str(debug_dir) if debug_dir else ""

    return row, kr_zyx

