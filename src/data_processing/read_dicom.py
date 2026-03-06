from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

PathLike = Union[str, Path]


@dataclass
class DicomVolume:
    """Container for a CT volume loaded from a DICOM series."""
    hu_zyx: np.ndarray  # (z, y, x) float32
    spacing_xyz: Tuple[float, float, float]  # (x, y, z) in mm
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]  # 9 values (3x3)
    series_uid: str
    size_xyz: Tuple[int, int, int]  # (x, y, z)


def list_dicom(folder: PathLike) -> List[str]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {folder}")
    return list(series_ids)


def load_dicom(
    folder: PathLike,
    series_uid: Optional[str] = None,
    *,
    return_sitk_image: bool = False,
) -> Union[DicomVolume, Tuple[DicomVolume, sitk.Image]]:
    folder = Path(folder)
    series_ids = list_dicom(folder)

    if series_uid is None:
        if len(series_ids) != 1:
            raise ValueError(
                f"Multiple DICOM series found in {folder}. "
                f"Please specify 'series_uid'. Found: {len(series_ids)}"
            )
        series_uid = series_ids[0]
    else:
        if series_uid not in series_ids:
            raise ValueError(f"Requested series_uid not found. Requested={series_uid}")

    reader = sitk.ImageSeriesReader()
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series_uid)
    reader.SetFileNames(file_names)

    img = reader.Execute()
    arr_zyx = sitk.GetArrayFromImage(img).astype(np.float32)

    spacing_xyz = img.GetSpacing()
    origin_xyz = img.GetOrigin()
    direction = img.GetDirection()
    size_xyz = img.GetSize()

    vol = DicomVolume(
        hu_zyx=arr_zyx,
        spacing_xyz=(float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])),
        origin_xyz=(float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])),
        direction=tuple(float(v) for v in direction),
        series_uid=str(series_uid),
        size_xyz=(int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])),
    )

    if return_sitk_image:
        return vol, img
    return vol


def load_metadata(
    folder: PathLike,
    series_uid: Optional[str] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Load only DICOM metadata (spacing, origin) WITHOUT loading pixel data.
    
    Returns:
        (spacing_xyz, origin_xyz) tuples
    """
    folder = Path(folder)
    series_ids = list_dicom(folder)

    if series_uid is None:
        if len(series_ids) != 1:
            raise ValueError(
                f"Multiple DICOM series found in {folder}. "
                f"Please specify 'series_uid'. Found: {len(series_ids)}"
            )
        series_uid = series_ids[0]
    else:
        if series_uid not in series_ids:
            raise ValueError(f"Requested series_uid not found. Requested={series_uid}")

    # Use ImageFileReader on a single DICOM file to read only headers
    reader = sitk.ImageFileReader()
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series_uid)
    
    if not file_names:
        raise ValueError(f"No DICOM files found for series {series_uid}")
    
    # Read only the first file's metadata
    reader.SetFileName(file_names[0])
    reader.ReadImageInformation()
    
    spacing_xyz = reader.GetMetaData("0028|0030")  # Pixel Spacing
    origin_xyz = reader.GetOrigin()
    
    # Parse spacing from DICOM tag (format: "x\\y")
    spacing_parts = spacing_xyz.split("\\")
    if len(spacing_parts) == 2:
        sx, sy = float(spacing_parts[0]), float(spacing_parts[1])
    else:
        sx = sy = 1.0
    
    # Get slice thickness from metadata
    try:
        sz = float(reader.GetMetaData("0018|0088"))  # Slice Thickness
    except:
        sz = 1.0
    
    spacing_xyz = (sx, sy, sz)
    
    return spacing_xyz, origin_xyz

