"""
Script to visualize problematic kidney cases identified by consistency check.
Shows side-by-side comparison of volumes across phases.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_consistency_outliers(outliers_csv, output_dir="outputs"):
    """
    Create visualizations for kidney consistency outliers.
    
    Args:
        outliers_csv: Path to kidney_consistency_outliers.csv
        output_dir: Directory to save visualizations
    """
    
    df = pd.read_csv(outliers_csv)
    
    print(f"Visualizing {len(df)} outlier cases...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    for idx, row in df.iterrows():
        case_id = row["case_id"]
        
        # Extract phase data
        phases = row["phases"].split(",")
        
        # Collect volume and z_range data
        vol_data = []
        z_data = []
        for phase in phases:
            vol_col = f"right_vol_{phase}"
            z_col = f"right_z_{phase}"
            
            if vol_col in row.index and pd.notna(row[vol_col]):
                vol_data.append({
                    "phase": phase.capitalize(),
                    "volume": row[vol_col],
                    "z_range": row[z_col]
                })
        
        if not vol_data:
            continue
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Case {case_id} - Right Kidney Inconsistency\n{row["issues"]}', 
                     fontsize=14, fontweight='bold')
        
        # Volume comparison
        phases_list = [d["phase"] for d in vol_data]
        volumes = [d["volume"] for d in vol_data]
        colors = ["#FF6B6B" if v == min(volumes) or v == max(volumes) else "#4ECDC4" 
                  for v in volumes]
        
        bars1 = ax1.bar(phases_list, volumes, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Volume (ml)', fontsize=11, fontweight='bold')
        ax1.set_title('Volume by Phase', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, vol in zip(bars1, volumes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{vol:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Z-range comparison
        z_ranges = [d["z_range"] for d in vol_data]
        bars2 = ax2.bar(phases_list, z_ranges, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Z-range (slices)', fontsize=11, fontweight='bold')
        ax2.set_title('Axial Extent (Z-range) by Phase', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, z_range in zip(bars2, z_ranges):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(z_range)}', ha='center', va='bottom', fontweight='bold')
        
        # Add summary text box
        summary_text = (
            f"Vol Min/Max: {row['vol_min']:.1f} / {row['vol_max']:.1f} ml\n"
            f"Vol Difference: {row['vol_diff']:.1f} ml ({row['vol_pct_diff']:.1f}%)\n"
            f"Z-range Min/Max: {int(row['z_min'])} / {int(row['z_max'])} slices\n"
            f"Z-range Difference: {int(row['z_diff'])} slices ({row['z_pct_diff']:.1f}%)"
        )
        fig.text(0.5, -0.05, summary_text, ha='center', fontsize=10, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        
        # Save figure
        output_path = Path(output_dir) / f"outlier_case_{case_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    print(f"\nVisualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize kidney consistency outliers")
    parser.add_argument("--csv", type=str, default="outputs/kidney_consistency_outliers.csv",
                        help="Path to kidney_consistency_outliers.csv")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    visualize_consistency_outliers(args.csv, args.output)
