from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import nibabel as nib

# Import your loader from your code
from read_dicom import load_ct_dicom_series, choose_folder_dialog

PathLike = Union[str, Path]


def _save_hu_volume_as_nifti(
    hu_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    out_path: PathLike,
) -> None:
    """
    Save a (z,y,x) HU volume to NIfTI with an affine that encodes spacing.
    Orientation is not perfect here, but is usually OK for TotalSegmentator localization.
    If you need correct orientation, use SimpleITK to write NIfTI with direction/origin.
    """
    out_path = Path(out_path)

    # nibabel expects (x,y,z), so transpose
    vol_xyz = np.transpose(hu_zyx, (2, 1, 0))

    sx, sy, sz = spacing_xyz  # spacing in mm
    affine = np.array([
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    nii = nib.Nifti1Image(vol_xyz.astype(np.float32), affine)
    nib.save(nii, str(out_path))


def segment_kidneys_from_dicom_with_totalseg(
    dicom_folder: PathLike,
    *,
    series_uid: Optional[str] = None,
    device: str = "cuda",
    fast: bool = True,
    output_dir: Optional[PathLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs TotalSegmentator to segment left+right kidneys from a DICOM series folder.

    Returns
    -------
    kidney_mask_zyx : np.ndarray
        Binary mask of kidneys, shape (z,y,x).
    z_indices : np.ndarray
        Sorted slice indices where kidneys are present.
    """
    # 1) Load volume from DICOM (your existing function)
    vol = load_ct_dicom_series(dicom_folder, series_uid=series_uid)

    # 2) Create temp working directory
    workdir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)

    input_nii = workdir / "ct.nii.gz"
    out_nii_dir = workdir / "totalseg_out"
    out_nii_dir.mkdir(parents=True, exist_ok=True)

    # 3) Save to NIfTI for TotalSegmentator
    _save_hu_volume_as_nifti(vol.hu, vol.spacing_xyz, input_nii)

    # 4) Run TotalSegmentator (Python API)
    try:
        from totalsegmentator.python_api import totalsegmentator
    except Exception as e:
        raise ImportError(
            "Could not import TotalSegmentator Python API. "
            "Make sure 'TotalSegmentator' is installed in this environment."
        ) from e

    # 5) Run only kidneys to reduce memory
    kidney_left = out_nii_dir / "kidney_left.nii.gz"
    kidney_right = out_nii_dir / "kidney_right.nii.gz"

    try:
        totalsegmentator(
            str(input_nii),
            str(out_nii_dir),
            task="total",
            fast=fast,
            device=device,
            roi_subset=["kidney_left", "kidney_right"],
            output_type="nifti",
        )
    except Exception:
        raise RuntimeError(
            "Your TotalSegmentator version does not support roi_subset. "
            "Please upgrade TotalSegmentator or run the CLI with --roi_subset."
        )

    # 6) Load the two kidney masks
    kl_xyz = nib.load(str(kidney_left)).get_fdata() > 0.5
    kr_xyz = nib.load(str(kidney_right)).get_fdata() > 0.5

    kidney_mask_xyz = (kl_xyz | kr_xyz)
    kidney_mask = np.transpose(kidney_mask_xyz, (2, 1, 0)).astype(np.uint8)

    # 7) Compute z-slices where kidneys exist
    z_indices = np.where(kidney_mask.reshape(kidney_mask.shape[0], -1).any(axis=1))[0]

    return kidney_mask, z_indices




if __name__ == "__main__":
    # Example usage
    # Ejemplo mínimo: seleccionar carpeta y listar series DI
    try:
        folder = choose_folder_dialog()
    except FileNotFoundError:
        print("No se seleccionó ninguna carpeta. Saliendo.")
    else:
        print("Carpeta seleccionada:", folder)
    kidney_mask, z_slices = segment_kidneys_from_dicom_with_totalseg(
        folder,
        device="cpu",
        fast=True,
    )
    print(f"Kidney mask shape: {kidney_mask.shape}")
    print(f"Slices with kidneys: {z_slices}")