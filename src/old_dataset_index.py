from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import os
import random
import pandas as pd
import SimpleITK as sitk

PathLike = Union[str, Path]

PHASES = ("arterial", "venous", "late")
HEADER_SAMPLE_PERCENTILES = (15, 30, 45, 60, 75, 90)

PHASE_TOKENS = {
    "arterial": ["arterial", "artery", "aortic", "aorta", "art"],
    "venous": ["venous", "portal", "pv", "portovenous", "vena"],
    "late": ["late", "delayed", "delay", "excretory", "urographic"],
}


@dataclass(frozen=True)
class SeriesCandidate:
    dicom_dir: Path
    series_uid: str
    n_files: int
    score: int
    matched_phase: Optional[str]
    # Multi-criteria fields for disambiguation
    study_uid: str = ""
    acquisition_time: str = ""  # HHMMSS.ffffff format
    series_description: str = ""
    slice_thickness: float = 0.0
    num_slices: int = 0
    reconstruction_kernel: str = ""


def folder_dialog(initialdir: Optional[PathLike] = None) -> Path:
    """Native folder picker"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(initialdir=str(initialdir) if initialdir else None)
    root.destroy()

    if not folder:
        raise FileNotFoundError("No folder selected.")
    return Path(folder)


def is_numeric(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()


def contains_dicom(folder: Path) -> bool:
    """True if SimpleITK can find at least one DICOM series in folder."""
    try:
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
        return bool(ids)
    except Exception:
        return False


def series_folder(folder: Path) -> List[Tuple[str, int]]:
    """
    Return [(series_uid, n_files)] for all series found inside folder.
    """
    out: List[Tuple[str, int]] = []
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []
    for suid in ids:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), suid)
        out.append((str(suid), len(files)))
    return out


def phases(path: Path) -> Tuple[Optional[str], int]:
    """
    Assign a phase + score based on tokens in the directory name/path.
    Higher score = more confident. If no match -> (None, 0).
    """
    s = str(path).lower()
    best_phase = None
    best_score = 0

    for phase, toks in PHASE_TOKENS.items():
        score = 0
        for t in toks:
            if t in s:
                # stronger weight if token matches folder name directly
                score += 3 if t in path.name.lower() else 1
        if score > best_score:
            best_phase = phase
            best_score = score

    if best_score == 0:
        return None, 0
    return best_phase, best_score


def is_dicom(filepath: str) -> bool:
    """Check if a file is a DICOM by reading its header."""
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def extract_dicom_metadata(filepath: str) -> Dict[str, any]:
    """
    Extract multi-criteria metadata from a DICOM file.
    
    Returns:
        Dictionary with: study_uid, series_uid, acquisition_time, 
        series_description, slice_thickness, num_slices, reconstruction_kernel
    """
    metadata = {
        "study_uid": "",
        "series_uid": "",
        "acquisition_time": "",
        "series_description": "",
        "slice_thickness": 0.0,
        "num_slices": 0,
        "reconstruction_kernel": ""
    }
    
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(filepath)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # Extract metadata tags
        metadata["study_uid"] = reader.GetMetaData("0020|000d") if reader.HasMetaDataKey("0020|000d") else ""
        metadata["series_uid"] = reader.GetMetaData("0020|000e") if reader.HasMetaDataKey("0020|000e") else ""
        metadata["acquisition_time"] = reader.GetMetaData("0008|0032") if reader.HasMetaDataKey("0008|0032") else ""
        metadata["series_description"] = reader.GetMetaData("0008|103e") if reader.HasMetaDataKey("0008|103e") else ""
        metadata["reconstruction_kernel"] = reader.GetMetaData("0018|1210") if reader.HasMetaDataKey("0018|1210") else ""
        
        # Numeric metadata
        try:
            thickness = reader.GetMetaData("0018|0050") if reader.HasMetaDataKey("0018|0050") else "0"
            metadata["slice_thickness"] = float(thickness)
        except:
            metadata["slice_thickness"] = 0.0
        
        # Number of slices from image
        metadata["num_slices"] = reader.GetSize()[2] if len(reader.GetSize()) > 2 else 0
        
    except Exception:
        pass
    
    return metadata


def headers_mixed_in_folder(dicom_dir: Path) -> Tuple[bool, str]:
    """
    Sample a few files in a series folder and compare Study/Series UIDs.
    Returns (is_mixed, detail_string).
    """
    try:
        files = [f for f in os.listdir(dicom_dir) if os.path.isfile(dicom_dir / f)]
    except Exception:
        return False, "no_files"

    if not files:
        return False, "no_files"

    total = len(files)
    indices = []
    for pct in HEADER_SAMPLE_PERCENTILES:
        idx = int(round((pct / 100) * (total - 1)))
        idx = max(0, min(total - 1, idx))
        if idx not in indices:
            indices.append(idx)

    baseline = None
    mismatches = 0
    checked = 0

    for idx in indices:
        filepath = str(dicom_dir / files[idx])
        if not is_dicom(filepath):
            continue

        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(filepath)
            reader.ReadImageInformation()
            study_uid = reader.GetMetaData("0020|000d") if reader.HasMetaDataKey("0020|000d") else ""
            series_uid = reader.GetMetaData("0020|000e") if reader.HasMetaDataKey("0020|000e") else ""
        except Exception:
            continue

        signature = (study_uid, series_uid)
        if baseline is None:
            baseline = signature
        elif signature != baseline:
            mismatches += 1
        checked += 1

    if checked == 0:
        return False, "no_headers"

    return mismatches > 0, f"{mismatches}/{checked}"


def split_mixed_series(dicom_dir: Path) -> List[Dict[str, str]]:
    """
    Full scan of a folder to group files by SeriesInstanceUID.
    Returns list of dicts with series_uid, study_uid, n_files, and files list.
    """
    groups: Dict[str, Dict[str, any]] = {}

    try:
        files = [f for f in os.listdir(dicom_dir) if os.path.isfile(dicom_dir / f)]
    except Exception:
        return []

    for fname in files:
        filepath = str(dicom_dir / fname)
        if not is_dicom(filepath):
            continue

        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(filepath)
            reader.ReadImageInformation()
            series_uid = reader.GetMetaData("0020|000e") if reader.HasMetaDataKey("0020|000e") else ""
            study_uid = reader.GetMetaData("0020|000d") if reader.HasMetaDataKey("0020|000d") else ""
        except Exception:
            continue

        if series_uid not in groups:
            groups[series_uid] = {
                "series_uid": series_uid,
                "study_uid": study_uid,
                "n_files": 0,
                "files": [],
            }

        groups[series_uid]["n_files"] = str(int(groups[series_uid]["n_files"]) + 1)
        groups[series_uid]["files"].append(fname)

    # Convert file lists to semicolon-separated strings
    for series_uid in groups:
        groups[series_uid]["files"] = ";".join(sorted(groups[series_uid]["files"]))

    return list(groups.values())


def series_candidates(search_root: Path) -> List[SeriesCandidate]:
    """
    Fast DICOM series discovery: only read Series UID on first pass.

    OPTIMIZATION:
    - 2 file samples to verify DICOM
    - Only Series UID extraction (fast)
    - Count files in folder for quick validation (no DICOM reads)
    """
    candidates = []
    
    sitk.ProcessObject.SetGlobalWarningDisplay(False)

    for root, dirs, files in os.walk(search_root):
        n_files = len(files)
        
        # Skip if too few files (unlikely to be a DICOM series)
        if n_files < 50:
            continue
        
        # Sample 2 random files to verify DICOM
        sample_files = random.sample(files, min(2, n_files))
        dicom_count = 0
        first_dicom_file = None
        
        for filename in sample_files:
            filepath = os.path.join(root, filename)
            if is_dicom(filepath):
                dicom_count += 1
                if first_dicom_file is None:
                    first_dicom_file = filepath
        
        # If both samples are DICOM, assume entire folder is a DICOM series
        if dicom_count < 2 or first_dicom_file is None:
            continue
        
        # FAST PATH: Only extract Series UID (leave other metadata for later if needed)
        d = Path(root)
        phase, base_score = phases(d)
        
        series_uid = ""
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(first_dicom_file)
            reader.ReadImageInformation()
            series_uid = reader.GetMetaData("0020|000e") if reader.HasMetaDataKey("0020|000e") else ""
        except Exception:
            pass
        
        candidates.append(
            SeriesCandidate(
                dicom_dir=d,
                series_uid=series_uid,
                n_files=n_files,  # Quick file count - no DICOM reads
                score=base_score + min(n_files // 50, 5),
                matched_phase=phase,
                # Store filepath for deferred metadata extraction if needed
                study_uid=first_dicom_file,  # Temporary: store filepath
                acquisition_time="",
                series_description="",
                slice_thickness=0.0,
                num_slices=0,
                reconstruction_kernel="",
            )
        )
        
        # IMPORTANT: Don't search deeper in this branch
        # But os.walk will continue with sibling directories
        dirs[:] = []

    return candidates


def process_candidates(
    candidates: List[SeriesCandidate],
) -> Tuple[Optional[List[Dict[str, SeriesCandidate]]], str, bool]:
    """
    Process candidates using count-based logic:
    - If exactly 3 unique date folders found: assume correct, no validation needed
    - If 4+ date folders: create all valid combinations, mark for manual review
    
    Returns:
        (list_of_valid_triplets, status_message, needs_manual_review)
        - triplets is None if critical error
        - triplets is list of dicts (one per valid combination)
        - needs_manual_review flag indicates multiple combinations exist
    """
    # Group by phase
    phase_to = {p: [] for p in PHASES}
    for c in candidates:
        if c.matched_phase in PHASES:
            phase_to[c.matched_phase].append(c)
    
    # Check if all phases present
    missing = [p for p in PHASES if len(phase_to[p]) == 0]
    if missing:
        return None, f"missing_phases={missing}", False
    
    # Count total candidates (simpler approach: just count them directly)
    total_candidates = sum(len(phase_to[p]) for p in PHASES)
    
    # CASE 1: Exactly 3 candidates (1 per phase) → assume correct, no validation needed
    if total_candidates == 3:
        chosen = {p: phase_to[p][0] for p in PHASES}
        return [chosen], "ok_3_candidates", False
    
    # CASE 2: 4+ candidates → create all valid combinations, mark for review
    if total_candidates >= 4:
        from itertools import product
        
        all_triplets = []
        for combo in product(phase_to["arterial"], phase_to["venous"], phase_to["late"]):
            art, ven, lat = combo
            # Verify they're different series
            uids = {art.series_uid, ven.series_uid, lat.series_uid}
            if len(uids) == 3:
                all_triplets.append({
                    "arterial": art,
                    "venous": ven,
                    "late": lat,
                })
        
        if not all_triplets:
            return None, f"no_valid_triplets_from_{total_candidates}_candidates", False
        
        return all_triplets, f"{total_candidates}_candidates_multiple_combinations", True
    
    # CASE 3: Other counts → ambiguous, cannot proceed
    return None, f"ambiguous_candidate_count={total_candidates}", False


def build_index(
    root: PathLike,
    *,
    out_index_csv: PathLike,
    out_failed_csv: PathLike,
) -> pd.DataFrame:
    """
    Auto-discover DICOM dirs for each case and create/update the dataset index CSV.
    """
    root = Path(root)
    out_index_csv = Path(out_index_csv)
    out_failed_csv = Path(out_failed_csv)

    out_index_csv.parent.mkdir(parents=True, exist_ok=True)
    out_failed_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    failed_rows: List[dict] = []
    mixed_rows: List[dict] = []

    case_dirs = sorted([p for p in root.iterdir() if is_numeric(p)], key=lambda x: int(x.name))
    
    print(f"\n{'='*60}")
    print(f"{len(case_dirs)} cases found")
    print(f"{'='*60}")

    for idx, case_dir in enumerate(case_dirs, 1):
        case_id = case_dir.name

        # Search inside case/DICOM if exists
        dicom_root = case_dir / "DICOM"
        search_root = dicom_root if dicom_root.exists() else case_dir

        candidates = series_candidates(search_root)
        
        if not candidates:
            failed_rows.append(
                {
                    "case_id": case_id,
                    "case_path": str(case_dir),
                    "reason": "no_dicom_found",
                }
            )
            continue

        triplets, reason, needs_review = process_candidates(candidates)
        if triplets is None:
            failed_rows.append(
                {
                    "case_id": case_id,
                    "case_path": str(case_dir),
                    "reason": reason,
                }
            )
            continue
        
        # Group candidates by phase for independent processing
        phase_to_candidates = {p: [] for p in PHASES}
        for cand in candidates:
            if cand.matched_phase in PHASES:
                phase_to_candidates[cand.matched_phase].append(cand)
        
        # Detect duplicate file counts per phase
        phase_has_duplicates = {}
        for phase in PHASES:
            counts = [c.n_files for c in phase_to_candidates[phase]]
            phase_has_duplicates[phase] = len(counts) >= 2 and len(set(counts)) == 1

        # Pre-process mixed headers check (only once per case, before phase loop)
        mixed_series_cache = {}  # Cache for mixed series splits per dicom_dir
        for cand in candidates:
            if cand.matched_phase in ("arterial", "late") and cand.n_files > 350:
                is_mixed, detail = headers_mixed_in_folder(cand.dicom_dir)
                if is_mixed:
                    mixed_series_cache[str(cand.dicom_dir)] = split_mixed_series(cand.dicom_dir)
            elif cand.matched_phase == "venous" and cand.n_files > 750:
                is_mixed, detail = headers_mixed_in_folder(cand.dicom_dir)
                if is_mixed:
                    mixed_series_cache[str(cand.dicom_dir)] = split_mixed_series(cand.dicom_dir)

        # Process each phase independently: if 1 candidate, write once; if 2+, write each with suffix
        for phase in PHASES:
            phase_candidates = phase_to_candidates[phase]
            has_multiple = len(phase_candidates) > 1
            
            for cand_idx, c in enumerate(phase_candidates):
                # Apply suffix only if this phase has multiple candidates
                phase_suffix = f"_{cand_idx + 1}" if has_multiple else ""
                case_id_with_suffix = f"{case_id}{phase_suffix}"
                
                # Build phase-specific notes
                note = ""
                if has_multiple:
                    note = f"{len(phase_candidates)} candidates for {phase}"
                    
                    # Add duplicate file count if applicable
                    if phase_has_duplicates[phase]:
                        counts = [c.n_files for c in phase_candidates]
                        dup_note = f"DUPLICATE_{phase.upper()}({counts[0]})"
                        note = (note + " | " if note else "") + dup_note
                
                
                # Quick validation: check file count limits (no DICOM reads needed)
                validation_flags = []
                
                high_count = False
                if phase in ("arterial", "late"):
                    if c.n_files > 350:
                        validation_flags.append(f"HIGH_FILE_COUNT({c.n_files})")
                        high_count = True
                elif phase == "venous":
                    if c.n_files > 750:
                        validation_flags.append(f"HIGH_FILE_COUNT({c.n_files})")
                        high_count = True

                if high_count:
                    # Check cache first - only read headers if not already cached
                    dir_str = str(c.dicom_dir)
                    if dir_str in mixed_series_cache:
                        is_mixed = True
                        mixed_groups = mixed_series_cache[dir_str]
                    else:
                        is_mixed, detail = headers_mixed_in_folder(c.dicom_dir)
                        if is_mixed:
                            mixed_groups = split_mixed_series(c.dicom_dir)
                            mixed_series_cache[dir_str] = mixed_groups
                    
                    if is_mixed:
                        detail_str = f"({len(mixed_series_cache.get(dir_str, []))} groups)"
                        validation_flags.append(f"MIXED_HEADERS{detail_str}")

                        # Export mixed series groups
                        for group in mixed_series_cache[dir_str]:
                            mixed_rows.append(
                                {
                                    "case_id": case_id_with_suffix,
                                    "phase": phase,
                                    "dicom_dir": str(c.dicom_dir),
                                    "series_uid": group.get("series_uid", ""),
                                    "study_uid": group.get("study_uid", ""),
                                    "n_files": group.get("n_files", "0"),
                                    "files": group.get("files", ""),
                                    "note": "MIXED_HEADERS_FULL_SCAN",
                                }
                            )

                source_type = "auto_multiple" if has_multiple else "auto_mixed" if high_count else "auto"

                # Combine notes (removed DATE_MISMATCH check - likely anonimization artifact)
                combined_note = note
                if validation_flags:
                    combined_note = (combined_note + " | " if combined_note else "") + " ".join(validation_flags)
                
                rows.append(
                    {
                        "case_id": case_id_with_suffix,
                        "phase": phase,
                        "dicom_dir": str(c.dicom_dir),
                        "series_uid": c.series_uid,
                        "source": source_type,
                        "note": combined_note,
                    }
                )

    # Create DataFrames with defined columns if rows is empty
    if rows:
        df = pd.DataFrame(rows).sort_values(["case_id", "phase"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["case_id", "phase", "dicom_dir", "series_uid", "source", "note"])
    
    df.to_csv(out_index_csv, index=False)

    if failed_rows:
        df_failed = pd.DataFrame(failed_rows).sort_values(["case_id"]).reset_index(drop=True)
    else:
        df_failed = pd.DataFrame(columns=["case_id", "case_path", "reason"])
    
    df_failed.to_csv(out_failed_csv, index=False)

    # Write mixed series split results (if any)
    out_mixed_csv = out_index_csv.parent / "mixed_series.csv"
    if mixed_rows:
        df_mixed = pd.DataFrame(mixed_rows).sort_values(["case_id", "phase", "series_uid"]).reset_index(drop=True)
        df_mixed.to_csv(out_mixed_csv, index=False)
    else:
        pd.DataFrame(
            columns=["case_id", "phase", "dicom_dir", "series_uid", "study_uid", "n_files", "files", "note"]
        ).to_csv(out_mixed_csv, index=False)
    
    # Final summary
    n_success = len(df) // 3 if len(df) > 0 else 0
    n_failed = len(df_failed)
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful cases: {n_success}")
    print(f"Failed cases: {n_failed}")
    print(f"{'='*60}\n")

    return df


def failed_cases(
    failed_csv: PathLike,
    *,
    out_index_csv: PathLike,
) -> pd.DataFrame:
    """
    Opens tkinter dialogs to manually select dicom dirs for each failed case.
    Writes/updates the index CSV by appending manual rows.
    """
    failed_csv = Path(failed_csv)
    out_index_csv = Path(out_index_csv)

    if not failed_csv.exists():
        raise FileNotFoundError(f"Failed-cases CSV not found: {failed_csv}")

    df_failed = pd.read_csv(failed_csv)
    if df_failed.empty:
        return pd.read_csv(out_index_csv) if out_index_csv.exists() else pd.DataFrame()

    existing = pd.read_csv(out_index_csv) if out_index_csv.exists() else pd.DataFrame(
        columns=["case_id", "phase", "dicom_dir", "series_uid", "source", "note"]
    )

    manual_rows: List[dict] = []

    for _, r in df_failed.iterrows():
        case_id = str(r["case_id"])
        case_path = Path(r["case_path"])

        # Skip if already resolved
        if not existing.empty:
            sub = existing[existing["case_id"].astype(str) == case_id]
            if set(sub["phase"].tolist()) == set(PHASES):
                continue

        print(f"\nManual resolve case {case_id} ({case_path})")

        chosen_dirs: Dict[str, Path] = {}
        for phase in PHASES:
            while True:
                print(f"Select DICOM folder for phase: {phase}")
                p = folder_dialog(initialdir=case_path)
                if not contains_dicom(p):
                    print("Selected folder does not contain a readable DICOM series.")
                    continue
                chosen_dirs[phase] = p
                break

        for phase, folder in chosen_dirs.items():
            series_list = series_folder(folder)
            series_list = sorted(series_list, key=lambda x: x[1], reverse=True)
            series_uid = series_list[0][0]

            manual_rows.append(
                {
                    "case_id": case_id,
                    "phase": phase,
                    "dicom_dir": str(folder),
                    "series_uid": series_uid,
                    "source": "manual",
                    "note": "manual selection",
                }
            )

    out = pd.concat([existing, pd.DataFrame(manual_rows)], ignore_index=True)
    out = out.sort_values(["case_id", "phase"]).drop_duplicates(["case_id", "phase"], keep="last").reset_index(drop=True)
    out.to_csv(out_index_csv, index=False)
    return out


