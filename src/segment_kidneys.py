from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk


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
        cmd.append("--fast")

    proc = subprocess.run(cmd, capture_output=True, text=True)
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
    device: str = "cpu",
    fast: bool = True,
    keep_debug_dir: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Path]]:
    tmp = Path(tempfile.mkdtemp())
    input_nii = tmp / "ct.nii.gz"
    out_dir = tmp / "totalseg_out"

    try:
        sitk.WriteImage(vol_sitk, str(input_nii))
        totalseg_kidneys(input_nii, out_dir, device=device, fast=fast)

        kl_path = out_dir / "kidney_left.nii.gz"
        kr_path = out_dir / "kidney_right.nii.gz"
        if not kl_path.exists() or not kr_path.exists():
            raise FileNotFoundError(f"Missing expected outputs: {kl_path} / {kr_path}")

        kl_img = nib.load(str(kl_path))
        kr_img = nib.load(str(kr_path))
        kl_xyz = (np.asanyarray(kl_img.dataobj) > 0.5)
        kr_xyz = (np.asanyarray(kr_img.dataobj) > 0.5)

        kl_zyx = xyz2zyx(kl_xyz)
        kr_zyx = xyz2zyx(kr_xyz)

        if keep_debug_dir:
            return kl_zyx, kr_zyx, tmp

        shutil.rmtree(tmp, ignore_errors=True)
        return kl_zyx, kr_zyx, None
    
    except (MemoryError, RuntimeError, FileNotFoundError):
        # Clean up temp directory on failure
        if not keep_debug_dir:
            shutil.rmtree(tmp, ignore_errors=True)
        raise
