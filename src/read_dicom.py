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

