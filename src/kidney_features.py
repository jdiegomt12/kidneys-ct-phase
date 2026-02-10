from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from segment_kidneys import segment_kidneys

PathLike = Union[str, Path]


DEFAULT_PERCENTILES = [0.15, 0.30, 0.50, 0.70, 0.85] # Moein et al. 2026

def _compute_mask_stats(
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


def kidneys_features(
    vol, vol_sitk,
    dicom_dir: PathLike,
    *,
    device: str = "cpu",
    fast: bool = True,
    percentiles: List[float] = DEFAULT_PERCENTILES,
    keep_debug_dir: bool = False,
) -> Dict[str, object]:
    """
    Loads DICOM, runs TotalSegmentator kidneys, computes per-kidney stats,
    and adds right-kidney percentile slice indices for later slice extraction.
    """
    dicom_dir = Path(dicom_dir)
    series_name = dicom_dir.name

    kl_zyx, kr_zyx, debug_dir = segment_kidneys(
        vol_sitk, device=device, fast=fast, keep_debug_dir=keep_debug_dir
    )

    left = _compute_mask_stats(vol.hu_zyx, kl_zyx, vol.spacing_xyz)
    right = _compute_mask_stats(vol.hu_zyx, kr_zyx, vol.spacing_xyz)

    right_pct = percentile_slices(right["z_min"], right["z_max"], percentiles)

    row: Dict[str, object] = {
        "series_name": series_name,
        "dicom_dir": str(dicom_dir),
        "series_uid": vol.series_uid,
        "right_kidney_percentile_slices": right_pct,
    }
    row.update(_prefix(left, "kidney_left_"))
    row.update(_prefix(right, "kidney_right_"))
    if keep_debug_dir:
        row["debug_dir"] = str(debug_dir) if debug_dir else ""

    return row
