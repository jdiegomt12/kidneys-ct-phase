"""
Analizar características de riñones para identificar segmentaciones problemáticas.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def analyze_kidney_features(features_csv: Path) -> None:
    """Analyze kidney features and suggest quality thresholds."""
    
    df = pd.read_csv(features_csv)
    
    print("\n" + "="*80)
    print("KIDNEY SEGMENTATION QUALITY ANALYSIS")
    print("="*80)
    
    # Check missing kidneys
    print("\n[1] MISSING KIDNEYS")
    print("-" * 80)
    left_missing = (df["kidney_left_present"] == 0).sum()
    right_missing = (df["kidney_right_present"] == 0).sum()
    print(f"  Left kidney missing: {left_missing}/{len(df)} ({100*left_missing/len(df):.1f}%)")
    print(f"  Right kidney missing: {right_missing}/{len(df)} ({100*right_missing/len(df):.1f}%)")
    
    if left_missing > 0:
        print(f"  Cases missing left kidney:")
        missing_left = df[df["kidney_left_present"] == 0][["series_name", "phase"]]
        for _, row in missing_left.iterrows():
            print(f"    - {row['series_name']} ({row['phase']})")
    
    if right_missing > 0:
        print(f"  Cases missing right kidney:")
        missing_right = df[df["kidney_right_present"] == 0][["series_name", "phase"]]
        for _, row in missing_right.iterrows():
            print(f"    - {row['series_name']} ({row['phase']})")
    
    # Filter present kidneys for remaining analysis
    df_left = df[df["kidney_left_present"] == 1].copy()
    df_right = df[df["kidney_right_present"] == 1].copy()
    
    # Compute derived features
    df_left["z_range"] = df_left["kidney_left_z_max"] - df_left["kidney_left_z_min"]
    df_left["hu_range"] = df_left["kidney_left_hu_max"] - df_left["kidney_left_hu_min"]
    
    df_right["z_range"] = df_right["kidney_right_z_max"] - df_right["kidney_right_z_min"]
    df_right["hu_range"] = df_right["kidney_right_hu_max"] - df_right["kidney_right_hu_min"]
    
    # Volume statistics
    print("\n[2] VOLUME STATISTICS (ml)")
    print("-" * 80)
    
    for side, df_side in [("LEFT", df_left), ("RIGHT", df_right)]:
        vol = df_side[f"kidney_{side.lower()}_volume_ml"]
        print(f"\n  {side} Kidney:")
        print(f"    Mean:  {vol.mean():.1f} ml")
        print(f"    Std:   {vol.std():.1f} ml")
        print(f"    Min:   {vol.min():.1f} ml")
        print(f"    Q1:    {vol.quantile(0.25):.1f} ml")
        print(f"    Q2:    {vol.median():.1f} ml")
        print(f"    Q3:    {vol.quantile(0.75):.1f} ml")
        print(f"    Max:   {vol.max():.1f} ml")
    
    # Z-range (slice thickness)
    print("\n[3] Z-RANGE (slice thickness differences)")
    print("-" * 80)
    
    for side, df_side in [("LEFT", df_left), ("RIGHT", df_right)]:
        z_range = df_side["z_range"]
        print(f"\n  {side} Kidney:")
        print(f"    Mean:  {z_range.mean():.1f} slices")
        print(f"    Std:   {z_range.std():.1f} slices")
        print(f"    Min:   {z_range.min():.0f} slices")
        print(f"    Max:   {z_range.max():.0f} slices")
        print(f"    Cases with z_range < 10: {(z_range < 10).sum()}")
    
    # HU statistics
    print("\n[4] HOUNSFIELD UNIT (HU) STATISTICS")
    print("-" * 80)
    
    for side, df_side in [("LEFT", df_left), ("RIGHT", df_right)]:
        prefix = f"kidney_{side.lower()}_"
        hu_mean = df_side[f"{prefix}hu_mean"]
        hu_std = df_side[f"{prefix}hu_std"]
        hu_range = df_side["hu_range"]
        hu_min = df_side[f"{prefix}hu_min"]
        hu_max = df_side[f"{prefix}hu_max"]
        
        print(f"\n  {side} Kidney HU Mean:")
        print(f"    Mean:  {hu_mean.mean():.1f} HU")
        print(f"    Std:   {hu_mean.std():.1f} HU")
        print(f"    Range: {hu_mean.min():.1f} to {hu_mean.max():.1f} HU")
        
        print(f"\n  {side} Kidney HU Std (tissue variability):")
        print(f"    Mean:  {hu_std.mean():.1f} HU")
        print(f"    Std:   {hu_std.std():.1f} HU")
        print(f"    Min:   {hu_std.min():.1f} HU (very homogeneous)")
        print(f"    Cases with HU_std < 20: {(hu_std < 20).sum()}")
        
        print(f"\n  {side} Kidney HU Range (max - min):")
        print(f"    Mean:  {hu_range.mean():.1f} HU")
        print(f"    Std:   {hu_range.std():.1f} HU")
        print(f"    Min:   {hu_range.min():.1f} HU")
        print(f"    Cases with HU_range < 100: {(hu_range < 100).sum()}")
        
        print(f"\n  {side} Kidney HU Min (negative = artifacts/air):")
        print(f"    Mean:  {hu_min.mean():.1f} HU")
        print(f"    Min:   {hu_min.min():.1f} HU")
        print(f"    Cases with HU_min < -50: {(hu_min < -50).sum()}")
        
        print(f"\n  {side} Kidney HU Max (high = vessel/other tissue):")
        print(f"    Mean:  {hu_max.mean():.1f} HU")
        print(f"    Max:   {hu_max.max():.1f} HU")
        print(f"    Cases with HU_max > 250: {(hu_max > 250).sum()}")
    
    # Identify problematic cases
    print("\n[5] QUALITY FLAGS & THRESHOLDS")
    print("-" * 80)
    
    problem_cases = []
    
    for idx, row in df.iterrows():
        issues = []
        series_name = row["series_name"]
        
        # Left kidney
        if row["kidney_left_present"] == 0:
            issues.append("LEFT_MISSING")
        else:
            z_range_l = row["kidney_left_z_max"] - row["kidney_left_z_min"]
            vol_l = row["kidney_left_volume_ml"]
            hu_std_l = row["kidney_left_hu_std"]
            hu_min_l = row["kidney_left_hu_min"]
            hu_max_l = row["kidney_left_hu_max"]
            hu_range_l = hu_max_l - hu_min_l
            
            if z_range_l < 10:
                issues.append(f"LEFT_THIN({int(z_range_l)})")
            if vol_l < 50:
                issues.append(f"LEFT_TINY({vol_l:.0f}ml)")
            if vol_l > 300:
                issues.append(f"LEFT_HUGE({vol_l:.0f}ml)")
            if hu_std_l < 20:
                issues.append(f"LEFT_HOMOG({hu_std_l:.0f})")
            if hu_range_l < 100:
                issues.append(f"LEFT_NARROW_HU({int(hu_range_l)})")
            if hu_min_l < -50:
                issues.append(f"LEFT_ARTIFACT({hu_min_l:.0f})")
            if hu_max_l > 250:
                issues.append(f"LEFT_VESSEL({hu_max_l:.0f})")
        
        # Right kidney
        if row["kidney_right_present"] == 0:
            issues.append("RIGHT_MISSING")
        else:
            z_range_r = row["kidney_right_z_max"] - row["kidney_right_z_min"]
            vol_r = row["kidney_right_volume_ml"]
            hu_std_r = row["kidney_right_hu_std"]
            hu_min_r = row["kidney_right_hu_min"]
            hu_max_r = row["kidney_right_hu_max"]
            hu_range_r = hu_max_r - hu_min_r
            
            if z_range_r < 10:
                issues.append(f"RIGHT_THIN({int(z_range_r)})")
            if vol_r < 50:
                issues.append(f"RIGHT_TINY({vol_r:.0f}ml)")
            if vol_r > 300:
                issues.append(f"RIGHT_HUGE({vol_r:.0f}ml)")
            if hu_std_r < 20:
                issues.append(f"RIGHT_HOMOG({hu_std_r:.0f})")
            if hu_range_r < 100:
                issues.append(f"RIGHT_NARROW_HU({int(hu_range_r)})")
            if hu_min_r < -50:
                issues.append(f"RIGHT_ARTIFACT({hu_min_r:.0f})")
            if hu_max_r > 250:
                issues.append(f"RIGHT_VESSEL({hu_max_r:.0f})")
        
        if issues:
            problem_cases.append({
                "series_name": series_name,
                "phase": row.get("phase", ""),
                "issues": ", ".join(issues)
            })
    
    if problem_cases:
        print(f"\n  Found {len(problem_cases)} problematic cases:")
        print()
        for case in sorted(problem_cases, key=lambda x: x["series_name"]):
            print(f"  {case['series_name']:20} {case['phase']:10} -> {case['issues']}")
    else:
        print("\n  No problematic cases detected!")
    
    # Suggested thresholds
    print("\n[6] SUGGESTED QUALITY THRESHOLDS")
    print("-" * 80)
    print("""
  Volume:
    ⚠️  < 50 ml      : Suspiciously small (possible undersegmentation)
    ⚠️  > 300 ml     : Suspiciously large (possible oversegmentation)
    ✓  50-300 ml    : Normal range
  
  Z-Range (slices):
    ⚠️  < 10 slices  : Too thin (could be thin slice acquisition or truncation)
    ✓  > 10 slices   : Acceptable thickness
  
  HU Std (tissue variability):
    ⚠️  < 20 HU      : Too homogeneous (possible partial volume / low quality)
    ⚠️  > 70 HU      : Too heterogeneous (possible artifact or cyst inclusion)
    ✓  20-70 HU     : Good variability
  
  HU Range (max - min):
    ⚠️  < 100 HU     : Too narrow (possible artifact or poor image contrast)
    ✓  > 100 HU     : Acceptable range
  
  HU Min:
    ⚠️  < -50 HU     : Contains noise/air (possible edge artifact)
    ✓  > -50 HU     : Good boundary definition
  
  HU Max:
    ⚠️  > 250 HU     : Contains vessels/fat (possible inclusion of non-kidney tissue)
    ✓  < 250 HU     : Likely kidney tissue only
  
  Negative/Positive Volume:
    ⚠️  Missing      : Segmentation failed completely
    ✓  Present      : Successful segmentation
    """)
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze kidney segmentation quality")
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs_old/kidney_features.csv",
        help="Path to kidney_features.csv"
    )
    
    args = parser.parse_args()
    csv_path = Path(args.csv)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    
    analyze_kidney_features(csv_path)
