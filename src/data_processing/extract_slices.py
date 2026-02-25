from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

PathLike = Union[str, Path]

# Moein et al. 2026
LEVEL = float(60.0)
WIDTH = float(400.0)

# Number of neighboring slices to include on each side of the central slice (total channels = 2*RADIUS + 1)
RADIUS = 1  


def prepro(
    hu_2d: np.ndarray,
    *,
    level: float = LEVEL,
    width: float = WIDTH,
) -> np.ndarray:
    lo = level - width / 2.0
    hi = level + width / 2.0
    x = np.clip(hu_2d, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)


def slice_indices(
    central_indices: List[int],
    *,
    z_max: int,
    neighbor_radius: int = RADIUS,
) -> List[int]:
    """
    For each central index, include [z-1, z, z+1]. Total 15 indices if 5 centrals.
    Keeps duplicates if they happen (rare but okay); model expects fixed channel count.
    """
    out: List[int] = []
    for c in central_indices:
        for dz in range(-neighbor_radius, neighbor_radius + 1):
            z = min(max(c + dz, 0), z_max)
            out.append(int(z))
    return out


def extract_slices(
    vol,
    dicom_dir: PathLike,
    *,
    central_slices: List[int],
    case_id: str,
    phase: str,
    out_root: PathLike,
    level: float = LEVEL,
    width: float = WIDTH,
    neighbor_radius: int = RADIUS,
    save_individual_slices: bool = False,
) -> Path:
    """
    Saves:
      <out_root>/<case_id>/<case_id>_<phase>.npy   shape=(15,H,W)
    Optionally saves individual slice npys inside same folder.
    """
    dicom_dir = Path(dicom_dir)
    out_root = Path(out_root)

    z_max = vol.hu_zyx.shape[0] - 1

    idxs = slice_indices(central_slices, z_max=z_max, neighbor_radius=neighbor_radius)

    case_out = out_root / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    channels: List[np.ndarray] = []
    for z in idxs:
        hu = vol.hu_zyx[z, :, :]
        img = prepro(hu, level=level, width=width)
        channels.append(img)

        if save_individual_slices:
            np.save(case_out / f"slice_{z:03d}.npy", img)

    tensor = np.stack(channels, axis=0).astype(np.float32)
    tensor_name = f"{case_id}_{phase}.npy"
    tensor_path = case_out / tensor_name
    np.save(tensor_path, tensor)
    return tensor_path
