"""
Script to check consistency of right kidney measurements across phases.
Verifies that volume and z_range are consistent within each case (case_id).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def check_kidney_consistency(csv_path, output_dir="outputs", volume_tolerance=None, z_range_tolerance=None):
    """
    Check consistency of right kidney metrics across phases.
    
    Args:
        csv_path: Path to kidney_features.csv
        output_dir: Directory to save results
        volume_tolerance: Tolerance for volume variation (ml). If None, uses statistical method.
        z_range_tolerance: Tolerance for z_range variation (slices). If None, uses statistical method.
    
    Returns:
        Tuple of (all_variations_df, outliers_df, statistics)
    """
    
    # Load data
    print(f"Loading kidney features from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={"case_id": str})
    
    print(f"Total records: {len(df)}")
    
    # Get unique case_ids
    case_ids = df["case_id"].unique()
    print(f"Unique cases: {len(case_ids)}")
    
    all_variations = []
    
    # Check each case
    for case_id in case_ids:
        case_data = df[df["case_id"] == case_id].copy()
        
        # Should have at most 3 rows (arterial, venous, late)
        phases = case_data["phase"].values
        
        # Check if we have right kidney data
        right_volumes = case_data["kidney_right_volume_ml"].values
        right_z_ranges = (case_data["kidney_right_z_max"] - case_data["kidney_right_z_min"]).values
        
        # Only check if we have multiple phases and all have right kidney data
        if len(case_data) > 1 and all(case_data["kidney_right_present"] == True):
            
            # Check volume consistency
            vol_min, vol_max = right_volumes.min(), right_volumes.max()
            vol_diff = vol_max - vol_min
            vol_mean = right_volumes.mean()
            vol_pct_diff = 100 * vol_diff / vol_mean if vol_mean > 0 else 0
            
            # Check z_range consistency
            z_min_range, z_max_range = right_z_ranges.min(), right_z_ranges.max()
            z_diff = z_max_range - z_min_range
            z_mean = right_z_ranges.mean()
            z_pct_diff = 100 * z_diff / z_mean if z_mean > 0 else 0
            
            all_variations.append({
                "case_id": case_id,
                "phases": ",".join(sorted(phases)),
                "num_phases": len(case_data),
                "right_vol_arterial": case_data[case_data["phase"] == "arterial"]["kidney_right_volume_ml"].values[0] if "arterial" in phases else None,
                "right_vol_venous": case_data[case_data["phase"] == "venous"]["kidney_right_volume_ml"].values[0] if "venous" in phases else None,
                "right_vol_late": case_data[case_data["phase"] == "late"]["kidney_right_volume_ml"].values[0] if "late" in phases else None,
                "vol_min": vol_min,
                "vol_max": vol_max,
                "vol_diff": vol_diff,
                "vol_mean": vol_mean,
                "vol_pct_diff": vol_pct_diff,
                "right_z_arterial": (case_data[case_data["phase"] == "arterial"]["kidney_right_z_max"].values[0] - case_data[case_data["phase"] == "arterial"]["kidney_right_z_min"].values[0]) if "arterial" in phases else None,
                "right_z_venous": (case_data[case_data["phase"] == "venous"]["kidney_right_z_max"].values[0] - case_data[case_data["phase"] == "venous"]["kidney_right_z_min"].values[0]) if "venous" in phases else None,
                "right_z_late": (case_data[case_data["phase"] == "late"]["kidney_right_z_max"].values[0] - case_data[case_data["phase"] == "late"]["kidney_right_z_min"].values[0]) if "late" in phases else None,
                "z_min": z_min_range,
                "z_max": z_max_range,
                "z_diff": z_diff,
                "z_mean": z_mean,
                "z_pct_diff": z_pct_diff,
            })
    
    # Create dataframe for all variations
    all_df = pd.DataFrame(all_variations)
    
    # Calculate statistics for automatic thresholding
    vol_diff_mean = all_df["vol_diff"].mean()
    vol_diff_std = all_df["vol_diff"].std()
    vol_diff_p95 = all_df["vol_diff"].quantile(0.95)
    
    z_diff_mean = all_df["z_diff"].mean()
    z_diff_std = all_df["z_diff"].std()
    z_diff_p95 = all_df["z_diff"].quantile(0.95)
    
    # Use provided tolerances or calculate from statistics (mean + 2*std)
    if volume_tolerance is None:
        volume_tolerance = vol_diff_mean + 2 * vol_diff_std
    
    if z_range_tolerance is None:
        z_range_tolerance = z_diff_mean + 2 * z_diff_std
    
    # Identify outliers
    outliers_mask = (all_df["vol_diff"] > volume_tolerance) | (all_df["z_diff"] > z_range_tolerance)
    outliers_df = all_df[outliers_mask].copy()
    
    # Add issue descriptions
    def get_issues(row):
        issues = []
        if row["vol_diff"] > volume_tolerance:
            issues.append(f"Volume diff: {row['vol_diff']:.1f}ml ({row['vol_pct_diff']:.1f}%)")
        if row["z_diff"] > z_range_tolerance:
            issues.append(f"Z-range diff: {row['z_diff']:.0f}slices ({row['z_pct_diff']:.1f}%)")
        return "; ".join(issues)
    
    outliers_df = outliers_df.copy()
    outliers_df["issues"] = outliers_df.apply(get_issues, axis=1)
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    
    all_path = Path(output_dir) / "kidney_consistency_all_variations.csv"
    all_df.to_csv(all_path, index=False)
    
    outliers_path = Path(output_dir) / "kidney_consistency_outliers.csv"
    outliers_df.to_csv(outliers_path, index=False)
    
    # Calculate statistics
    stats = {
        "total_cases": len(case_ids),
        "cases_analyzed": len(all_df),
        "outlier_cases": len(outliers_df),
        "outlier_percentage": 100 * len(outliers_df) / len(all_df),
        "vol_diff_mean": vol_diff_mean,
        "vol_diff_std": vol_diff_std,
        "vol_diff_p95": vol_diff_p95,
        "vol_tolerance_used": volume_tolerance,
        "z_diff_mean": z_diff_mean,
        "z_diff_std": z_diff_std,
        "z_diff_p95": z_diff_p95,
        "z_tolerance_used": z_range_tolerance,
    }
    
    print(f"\n{'='*80}")
    print(f"KIDNEY CONSISTENCY CHECK RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(f"  Total cases: {stats['total_cases']}")
    print(f"  Cases analyzed (with 3 phases): {stats['cases_analyzed']}")
    print(f"  Outlier cases: {stats['outlier_cases']}/{stats['cases_analyzed']} ({stats['outlier_percentage']:.1f}%)")
    
    print(f"\n📈 VOLUME VARIATION STATISTICS:")
    print(f"  Mean difference:   {vol_diff_mean:.1f} ml")
    print(f"  Std deviation:     {vol_diff_std:.1f} ml")
    print(f"  95th percentile:   {vol_diff_p95:.1f} ml")
    print(f"  Threshold used:    {volume_tolerance:.1f} ml (mean + 2*std)")
    
    print(f"\n📊 Z-RANGE VARIATION STATISTICS:")
    print(f"  Mean difference:   {z_diff_mean:.1f} slices")
    print(f"  Std deviation:     {z_diff_std:.1f} slices")
    print(f"  95th percentile:   {z_diff_p95:.1f} slices")
    print(f"  Threshold used:    {z_range_tolerance:.1f} slices (mean + 2*std)")
    
    if len(outliers_df) > 0:
        print(f"\n⚠️  OUTLIER CASES ({len(outliers_df)}):")
        display_cols = ["case_id", "phases", "vol_mean", "vol_diff", "vol_pct_diff", "z_mean", "z_diff", "z_pct_diff"]
        print(outliers_df[display_cols].to_string(index=False))
        
        print(f"\n✓ Outlier list saved to: {outliers_path}")
        print(f"✓ All variations saved to: {all_path}")
    else:
        print("\n✓ No outliers found! All cases have normal variation between phases.")
    
    print(f"{'='*80}\n")
    
    return all_df, outliers_df, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check kidney consistency across phases")
    parser.add_argument("--csv", type=str, default="outputs_old/kidney_features.csv",
                        help="Path to kidney_features.csv")
    parser.add_argument("--output", type=str, default="outputs_renew",
                        help="Output directory for results")
    parser.add_argument("--vol-tol", type=float, default=None,
                        help="Volume tolerance in ml (default: auto-calculated from statistics)")
    parser.add_argument("--z-tol", type=float, default=None,
                        help="Z-range tolerance in slices (default: auto-calculated from statistics)")
    
    args = parser.parse_args()
    
    all_df, outliers_df, stats = check_kidney_consistency(args.csv, args.output, args.vol_tol, args.z_tol)
