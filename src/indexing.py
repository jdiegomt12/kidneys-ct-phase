from pathlib import Path
import pydicom
import pandas as pd

def index_dicoms(root: Path, max_files=None):
    rows = []
    files = list(root.rglob("*.dcm"))
    if max_files:
        files = files[:max_files]

    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            rows.append({
                "path": str(fp),
                "PatientID": getattr(ds, "PatientID", None),
                "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
                "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
                "SeriesDescription": getattr(ds, "SeriesDescription", None),
                "Modality": getattr(ds, "Modality", None),
                "ImageType": "\\".join(getattr(ds, "ImageType", [])) if hasattr(ds, "ImageType") else None,
                "BodyPartExamined": getattr(ds, "BodyPartExamined", None),
                "AcquisitionTime": getattr(ds, "AcquisitionTime", None),
                "SliceThickness": getattr(ds, "SliceThickness", None),
                "PixelSpacing": ",".join(map(str, getattr(ds, "PixelSpacing", []))) if hasattr(ds, "PixelSpacing") else None,
                "InstanceNumber": getattr(ds, "InstanceNumber", None),
            })
        except Exception as e:
            # keep going; log if needed
            continue

    df = pd.DataFrame(rows)
    return df

df = index_dicoms(Path("data_raw"))
df.to_csv("c:/Users/juand/Desktop/BIOMED DTU/4. THESIS/code/data_derived/dicom_index.csv", index=False)
print(df.head())
