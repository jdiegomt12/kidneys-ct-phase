from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk


PathLike = Union[str, Path]


@dataclass
class DicomVolume:
    """Container for a CT volume loaded from a DICOM series."""
    hu: np.ndarray  # shape: (z, y, x), dtype float32
    spacing_xyz: Tuple[float, float, float]  # (x, y, z) in mm
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]  # 9 values (3x3)
    series_uid: str
    size_xyz: Tuple[int, int, int]  # (x, y, z)


def list_dicom_series(folder: PathLike) -> List[str]:
    """
    Return all DICOM SeriesInstanceUIDs found inside 'folder'.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise ValueError(
            f"No DICOM series found in: {folder}\n"
            "Tip: Make sure the folder contains .dcm files (or DICOM files without extension)."
        )
    return list(series_ids)


def load_ct_dicom_series(
    folder: PathLike,
    series_uid: Optional[str] = None,
    *,
    return_sitk_image: bool = False
) -> Union[DicomVolume, Tuple[DicomVolume, sitk.Image]]:
    """
    Load a CT DICOM series from a folder.
    """
    folder = Path(folder)

    series_ids = list_dicom_series(folder)

    if series_uid is None:
        if len(series_ids) == 1:
            series_uid = series_ids[0]
        else:
            raise ValueError(
                f"Multiple DICOM series found in {folder}.\n"
                f"Please specify 'series_uid'. Options:\n- " + "\n- ".join(series_ids[:20]) +
                ("\n..." if len(series_ids) > 20 else "")
            )
    else:
        if series_uid not in series_ids:
            raise ValueError(
                f"Requested series_uid not found in folder.\n"
                f"Requested: {series_uid}\n"
                f"Available: {series_ids}"
            )

    reader = sitk.ImageSeriesReader()
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series_uid)
    reader.SetFileNames(file_names)

    # This helps in some messy datasets, but can be slower:
    # reader.MetaDataDictionaryArrayUpdateOn()
    # reader.LoadPrivateTagsOn()

    img = reader.Execute()  # sitk.Image, typically int16 values already in HU for CT

    # Convert to numpy in (z, y, x)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    # Metadata
    spacing_xyz = img.GetSpacing()      # (x, y, z)
    origin_xyz = img.GetOrigin()        # (x, y, z)
    direction = img.GetDirection()      # 9 values
    size_xyz = img.GetSize()            # (x, y, z)

    vol = DicomVolume(
        hu=arr,
        spacing_xyz=(float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])),
        origin_xyz=(float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])),
        direction=tuple(float(v) for v in direction),
        series_uid=str(series_uid),
        size_xyz=(int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])),
    )

    if return_sitk_image:
        return vol, img
    return vol


def window_hu(
    hu: np.ndarray,
    *,
    level: float = 60.0,
    width: float = 400.0,
    out_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Apply CT windowing and scale to [0,1] (or other range).
    """
    lo = level - width / 2.0
    hi = level + width / 2.0
    x = np.clip(hu, lo, hi)
    x = (x - lo) / (hi - lo)
    a, b = out_range
    return x * (b - a) + a


def get_axial_slice(hu_zyx: np.ndarray, z_index: int) -> np.ndarray:
    """
    Return one axial slice as (y, x).
    """
    if z_index < 0 or z_index >= hu_zyx.shape[0]:
        raise IndexError(f"z_index out of bounds: {z_index} for volume with z={hu_zyx.shape[0]}")
    return hu_zyx[z_index, :, :]


def choose_folder_dialog(initialdir: Optional[PathLike] = None) -> Path:
    """
    Open a native folder-selection dialog and return the chosen Path.

    Raises FileNotFoundError if the user cancels.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError("tkinter is required for folder dialog") from e

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(initialdir=str(initialdir) if initialdir else None)
    root.destroy()

    if not folder:
        raise FileNotFoundError("No folder selected")
    return Path(folder)


if __name__ == "__main__":
    # Ejemplo mínimo: seleccionar carpeta y listar series DICOM encontradas
    try:
        folder = choose_folder_dialog()
    except FileNotFoundError:
        print("No se seleccionó ninguna carpeta. Saliendo.")
    else:
        print("Carpeta seleccionada:", folder)
        try:
            series = list_dicom_series(folder)
            print("Series DICOM encontradas:")
            for i, s in enumerate(series):
                print(f"{i+1}. {s}")
        except Exception as e:
            print("Error al listar series DICOM:", e)
    vol = load_ct_dicom_series(folder, series_uid=series[0])
    print("Volumen cargado:")
    print(" - Tamaño (x,y,z):", vol.size_xyz)
    print(" - Espaciado (x,y,z):", vol.spacing_xyz)
    print(" - Origen (x,y,z):", vol.origin_xyz)
    print(" - Dirección:", vol.direction)
    print(" - HU min/max:", vol.hu.min(), "/", vol.hu.max())
    
    z = vol.size_xyz[2] // 2
    slice_hu = get_axial_slice(vol.hu, z)
    slice_win = window_hu(slice_hu, level=60, width=400)

    # Visualizar con matplotlib
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_win, cmap='gray')
    plt.title(f"Axial slice z={z} (windowed)")
    plt.colorbar(label='Normalized HU')
    plt.axis('off')
    plt.show()