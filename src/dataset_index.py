from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import SimpleITK as sitk

PathLike = Union[str, Path]

PHASES = ("arterial", "venous", "late")

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


def series_candidates(
    search_root: Path,
    *,
    skip_dirs: Tuple[str, ...] = ("segmenteringer",), # I assume that I won't always have the data already processed and segmented
) -> List[SeriesCandidate]:
    """
    Recursively find folders that contain DICOM series and return candidates.
    We do not assume .dcm extensions.
    """
    candidates: List[SeriesCandidate] = []
    skip_lower = {x.lower() for x in skip_dirs}

    prev_warning = sitk.ProcessObject.GetGlobalWarningDisplay()
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    try:
        for d in search_root.rglob("*"):
            if not d.is_dir():
                continue
            if d.name.startswith("."):
                continue
            if d.name.lower() in skip_lower:
                continue

            # quick reject
            if not contains_dicom(d):
                continue

            # for each series in that folder
            phase, base_score = phases(d)
            for suid, n_files in series_folder(d):
                candidates.append(
                    SeriesCandidate(
                        dicom_dir=d,
                        series_uid=suid,
                        n_files=n_files,
                        score=base_score + min(n_files // 50, 5),
                        matched_phase=phase,
                    )
                )
    finally:
        sitk.ProcessObject.SetGlobalWarningDisplay(prev_warning)
    return candidates


def best_candidate(
    candidates: List[SeriesCandidate],
) -> Tuple[Optional[Dict[str, SeriesCandidate]], str]:
    """
    Decide if we can uniquely assign one candidate to each of the 3 phases.
    If ambiguous or missing -> return (None, reason).
    """
    phase_to = {p: [] for p in PHASES}
    for c in candidates:
        if c.matched_phase in PHASES:
            phase_to[c.matched_phase].append(c)

    # Must have at least one candidate for each phase
    missing = [p for p in PHASES if len(phase_to[p]) == 0]
    if missing:
        return None, f"missing_phases={missing}"

    chosen: Dict[str, SeriesCandidate] = {}
    for p in PHASES:
        # sort by (score desc, n_files desc)
        ranked = sorted(phase_to[p], key=lambda x: (x.score, x.n_files), reverse=True)

        # send to manual
        if len(ranked) >= 2:
            top, second = ranked[0], ranked[1]
            if (top.score == second.score) and (abs(top.n_files - second.n_files) < 30):
                return None, f"ambiguous_phase={p} (tie-ish between {top.dicom_dir} and {second.dicom_dir})"

        chosen[p] = ranked[0]

    # If two phases point to the same (dir, series_uid) check
    used = {(v.dicom_dir, v.series_uid) for v in chosen.values()}
    if len(used) != 3:
        return None, "ambiguous_shared_series_across_phases"

    return chosen, "ok"


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

    case_dirs = sorted([p for p in root.iterdir() if is_numeric(p)], key=lambda x: int(x.name))

    for case_dir in case_dirs:
        case_id = case_dir.name

        # Search inside case/DICOM if exists
        dicom_root = case_dir / "DICOM"
        search_root = dicom_root if dicom_root.exists() else case_dir

        candidates = series_candidates(search_root)
        if not candidates:
            print(f"WARNING: No DICOM series found for case_id={case_id} (root={search_root})")

        chosen, reason = best_candidate(candidates)
        if chosen is None:
            failed_rows.append(
                {
                    "case_id": case_id,
                    "case_path": str(case_dir),
                    "reason": reason,
                }
            )
            continue

        for phase in PHASES:
            c = chosen[phase]
            rows.append(
                {
                    "case_id": case_id,
                    "phase": phase,
                    "dicom_dir": str(c.dicom_dir),
                    "series_uid": c.series_uid,
                    "source": "auto",
                    "note": "",
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
