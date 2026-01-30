#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Post-Processing and Visualization Script

This script performs two critical tasks:
1. Create the Master Dataset by merging simulation results with metadata
2. Generate paper-ready visualizations for Q1 analysis

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
# Input paths
SIMULATION_INPUT_PATH = Path(__file__).parent.parent / "data_processed" / "dwts_simulation_input.csv"
FAN_ESTIMATES_PATH = Path(__file__).parent.parent / "results" / "fan_vote_estimates.csv"

# Output paths
MASTER_DATASET_PATH = Path(__file__).parent.parent / "data_processed" / "dwts_full_simulation.csv"
IMAGES_DIR = Path(__file__).parent.parent / "images"

# Plot settings
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# Use a clean style
sns.set_style("whitegrid")
sns.set_palette("husl")


# =============================================================================
# Task 1: Create Master Dataset
# =============================================================================
def create_master_dataset() -> pd.DataFrame:
    """
    Merge simulation results with original metadata to create the master dataset.
    
    - Merges on [season, week, celebrity_name]
    - Fills NaN fan shares with uniform prior (1 / n_contestants)
    - Saves to data_processed/dwts_full_simulation.csv
    
    Returns:
        Master DataFrame with all columns
    """
    print("="*60)
    print("TASK 1: Creating Master Dataset")
    print("="*60)
    
    # Load datasets
    print(f"\n[INFO] Loading simulation input: {SIMULATION_INPUT_PATH}")
    sim_input = pd.read_csv(SIMULATION_INPUT_PATH)
    print(f"       Rows: {len(sim_input)}")
    
    print(f"[INFO] Loading fan estimates: {FAN_ESTIMATES_PATH}")
    fan_estimates = pd.read_csv(FAN_ESTIMATES_PATH)
    print(f"       Rows: {len(fan_estimates)}")
    
    # Merge on key columns
    merge_keys = ['season', 'week', 'celebrity_name']
    
    master_df = pd.merge(
        sim_input,
        fan_estimates[['season', 'week', 'celebrity_name', 
                       'estimated_fan_share', 'share_std', 'confidence', 'n_valid_sims']],
        on=merge_keys,
        how='left'
    )
    
    print(f"\n[INFO] Merged dataset rows: {len(master_df)}")
    
    # Fill NaN fan shares with uniform prior (1 / n_contestants)
    # For weeks with no elimination, we use this as the best estimate
    nan_mask = master_df['estimated_fan_share'].isna()
    if nan_mask.sum() > 0:
        print(f"[INFO] Filling {nan_mask.sum()} NaN fan shares with uniform prior...")
        
        # Calculate uniform share per group
        master_df.loc[nan_mask, 'estimated_fan_share'] = master_df.loc[nan_mask].apply(
            lambda row: 1.0 / row['n_contestants'], axis=1
        )
    
    # Add derived columns for analysis
    # Fan-Judge gap (positive = fan favorite, negative = judge favorite)
    master_df['fan_judge_gap'] = master_df['estimated_fan_share'] - master_df['judge_share']
    
    # Ensure output directory exists
    MASTER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save master dataset
    master_df.to_csv(MASTER_DATASET_PATH, index=False)
    print(f"\n✓ Master dataset saved to: {MASTER_DATASET_PATH}")
    print(f"  Total rows: {len(master_df)}")
    print(f"  Columns: {list(master_df.columns)}")
    
    # Quick validation
    print(f"\n[VALIDATION] Dataset summary:")
    print(f"  - Seasons: {master_df['season'].min()} - {master_df['season'].max()}")
    print(f"  - Unique contestants: {master_df['celebrity_name'].nunique()}")
    print(f"  - Mean fan share: {master_df['estimated_fan_share'].mean():.4f}")
    print(f"  - Mean confidence: {master_df['confidence'].mean():.4f}")
    
    return master_df


# =============================================================================
# Task 2: Generate Visualizations
# =============================================================================
def plot_bobby_bones_anomaly(df: pd.DataFrame) -> None:
    """
    Figure 1: The "Bobby Bones" Anomaly - Time Series
    
    Shows the huge gap between Judge and Fan perception for Bobby Bones (S27).
    Dual-axis line chart with confidence interval shading.
    
    Args:
        df: Master DataFrame
    """
    print("\n[PLOT] Generating Figure 1: Bobby Bones Anomaly...")
    
    # Filter for Bobby Bones, Season 27
    bobby_data = df[(df['season'] == 27) & 
                    (df['celebrity_name'].str.contains('Bobby Bones', case=False, na=False))]
    
    if len(bobby_data) == 0:
        # Try alternative name matching
        s27_contestants = df[df['season'] == 27]['celebrity_name'].unique()
        print(f"[WARNING] Bobby Bones not found. Season 27 contestants: {s27_contestants}")
        
        # Look for similar names
        for name in s27_contestants:
            if 'bobby' in name.lower() or 'bones' in name.lower():
                bobby_data = df[(df['season'] == 27) & (df['celebrity_name'] == name)]
                print(f"[INFO] Using contestant: {name}")
                break
    
    if len(bobby_data) == 0:
        print("[ERROR] Could not find Bobby Bones data. Skipping Figure 1.")
        return
    
    bobby_data = bobby_data.sort_values('week')
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Y-axis: Judge Share (Red)
    color1 = '#E74C3C'  # Red
    ax1.set_xlabel('Week', fontweight='bold')
    ax1.set_ylabel('Judge Share', color=color1, fontweight='bold')
    line1 = ax1.plot(bobby_data['week'], bobby_data['judge_share'], 
                     color=color1, marker='o', linewidth=2.5, markersize=8,
                     label='Judge Share')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(bobby_data['judge_share'].max() * 1.2, 0.25))
    
    # Right Y-axis: Fan Share (Blue)
    ax2 = ax1.twinx()
    color2 = '#3498DB'  # Blue
    ax2.set_ylabel('Estimated Fan Share', color=color2, fontweight='bold')
    line2 = ax2.plot(bobby_data['week'], bobby_data['estimated_fan_share'],
                     color=color2, marker='s', linewidth=2.5, markersize=8,
                     label='Estimated Fan Share')
    
    # Add confidence interval shading for fan share
    if 'share_std' in bobby_data.columns and bobby_data['share_std'].notna().any():
        fan_upper = bobby_data['estimated_fan_share'] + bobby_data['share_std'].fillna(0)
        fan_lower = bobby_data['estimated_fan_share'] - bobby_data['share_std'].fillna(0)
        fan_lower = fan_lower.clip(lower=0)
        ax2.fill_between(bobby_data['week'], fan_lower, fan_upper, 
                         alpha=0.2, color=color2, label='95% CI')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(bobby_data['estimated_fan_share'].max() * 1.2, 0.35))
    
    # Title and legend
    contestant_name = bobby_data['celebrity_name'].iloc[0]
    plt.title(f'The "{contestant_name}" Anomaly: Judge vs Fan Perception\n(Season 27 - Winner Despite Low Judge Scores)',
              fontweight='bold', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    
    # Add annotation for the gap
    max_gap_week = bobby_data.loc[bobby_data['fan_judge_gap'].idxmax(), 'week']
    ax1.axvline(x=max_gap_week, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('Max Gap', xy=(max_gap_week, ax1.get_ylim()[1]*0.9),
                 fontsize=9, ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save
    output_path = IMAGES_DIR / 'figure1_bobby_bones_anomaly.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_certainty_evolution(df: pd.DataFrame) -> None:
    """
    Figure 2: Certainty Evolution - How confidence changes across weeks
    
    Shows if model certainty increases or decreases as the season progresses.
    
    Args:
        df: Master DataFrame
    """
    print("\n[PLOT] Generating Figure 2: Certainty Evolution...")
    
    # Filter for rows with valid confidence values
    valid_df = df[df['confidence'].notna() & (df['confidence'] > 0)].copy()
    
    if len(valid_df) == 0:
        print("[ERROR] No valid confidence data. Skipping Figure 2.")
        return
    
    # Group by week and calculate statistics
    week_stats = valid_df.groupby('week').agg({
        'confidence': ['mean', 'std', 'count']
    }).reset_index()
    week_stats.columns = ['week', 'mean_confidence', 'std_confidence', 'count']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean confidence with error bars
    color = '#2ECC71'  # Green
    ax.plot(week_stats['week'], week_stats['mean_confidence'],
            color=color, marker='o', linewidth=2.5, markersize=8,
            label='Mean Confidence')
    
    # Add confidence interval
    ci_upper = week_stats['mean_confidence'] + week_stats['std_confidence']
    ci_lower = (week_stats['mean_confidence'] - week_stats['std_confidence']).clip(lower=0)
    ax.fill_between(week_stats['week'], ci_lower, ci_upper,
                    alpha=0.2, color=color, label='±1 Std Dev')
    
    # Add sample size annotation
    for _, row in week_stats.iterrows():
        ax.annotate(f"n={int(row['count'])}", 
                    xy=(row['week'], row['mean_confidence']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, alpha=0.7)
    
    # Styling
    ax.set_xlabel('Week Number', fontweight='bold')
    ax.set_ylabel('Average Model Confidence', fontweight='bold')
    ax.set_title('Model Certainty Evolution Across Season Weeks\n(Higher = More Deterministic Outcomes)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, min(week_stats['mean_confidence'].max() * 1.5, 1.0))
    ax.set_xlim(0.5, week_stats['week'].max() + 0.5)
    
    # Add trend line
    z = np.polyfit(week_stats['week'], week_stats['mean_confidence'], 1)
    p = np.poly1d(z)
    ax.plot(week_stats['week'], p(week_stats['week']), 
            '--', color='gray', alpha=0.7, label=f'Trend (slope={z[0]:.4f})')
    
    # Update legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    output_path = IMAGES_DIR / 'figure2_certainty_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_system_stability(df: pd.DataFrame) -> None:
    """
    Figure 3: System Stability - Compare rule systems by confidence
    
    Bar chart comparing which rule system produces more deterministic outcomes.
    
    Args:
        df: Master DataFrame
    """
    print("\n[PLOT] Generating Figure 3: System Stability Comparison...")
    
    # Filter for valid confidence
    valid_df = df[df['confidence'].notna() & (df['confidence'] > 0)].copy()
    
    if len(valid_df) == 0:
        print("[ERROR] No valid confidence data. Skipping Figure 3.")
        return
    
    # Group by rule system
    system_stats = valid_df.groupby('rule_system').agg({
        'confidence': ['mean', 'std', 'count']
    }).reset_index()
    system_stats.columns = ['rule_system', 'mean_confidence', 'std_confidence', 'count']
    
    # Order: Rank -> Percent -> Rank_With_Save (chronological)
    order = ['Rank', 'Percent', 'Rank_With_Save']
    system_stats['rule_system'] = pd.Categorical(system_stats['rule_system'], 
                                                  categories=order, ordered=True)
    system_stats = system_stats.sort_values('rule_system')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette
    colors = ['#E74C3C', '#3498DB', '#9B59B6']  # Red, Blue, Purple
    
    # Create bar chart
    bars = ax.bar(system_stats['rule_system'], system_stats['mean_confidence'],
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add error bars
    ax.errorbar(system_stats['rule_system'], system_stats['mean_confidence'],
                yerr=system_stats['std_confidence'], fmt='none', 
                color='black', capsize=8, capthick=2, linewidth=2)
    
    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, system_stats.itertuples())):
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add sample size below bar
        ax.annotate(f'(n={int(row.count)})',
                    xy=(bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, alpha=0.7)
    
    # Add era labels
    era_labels = ['S1-S2', 'S3-S27', 'S28-S34']
    for i, (bar, label) in enumerate(zip(bars, era_labels)):
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                    ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold', alpha=0.9)
    
    # Styling
    ax.set_xlabel('Rule System', fontweight='bold')
    ax.set_ylabel('Average Model Confidence', fontweight='bold')
    ax.set_title('Voting System Stability Comparison\n(Higher Confidence = More Predictable Eliminations)',
                 fontweight='bold', pad=20)
    ax.set_ylim(0, min(system_stats['mean_confidence'].max() * 1.4, 1.0))
    
    # Add horizontal reference line for mean
    overall_mean = valid_df['confidence'].mean()
    ax.axhline(y=overall_mean, color='gray', linestyle='--', alpha=0.7, 
               label=f'Overall Mean ({overall_mean:.3f})')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    output_path = IMAGES_DIR / 'figure3_system_stability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def generate_all_visualizations(df: pd.DataFrame) -> None:
    """
    Generate all paper visualizations.
    
    Args:
        df: Master DataFrame
    """
    print("\n" + "="*60)
    print("TASK 2: Generating Paper Visualizations")
    print("="*60)
    
    # Ensure images directory exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate each figure
    plot_bobby_bones_anomaly(df)
    plot_certainty_evolution(df)
    plot_system_stability(df)
    
    print(f"\n✓ All visualizations saved to: {IMAGES_DIR}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    """
    Main post-processing and visualization pipeline.
    """
    print("="*60)
    print("MCM 2026 Problem C - Post-Processing & Visualization")
    print("="*60)
    
    # Task 1: Create Master Dataset
    master_df = create_master_dataset()
    
    # Task 2: Generate Visualizations
    generate_all_visualizations(master_df)
    
    # Summary
    print("\n" + "="*60)
    print("POST-PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  1. Master Dataset: {MASTER_DATASET_PATH}")
    print(f"  2. Figure 1: {IMAGES_DIR / 'figure1_bobby_bones_anomaly.png'}")
    print(f"  3. Figure 2: {IMAGES_DIR / 'figure2_certainty_evolution.png'}")
    print(f"  4. Figure 3: {IMAGES_DIR / 'figure3_system_stability.png'}")
    
    return master_df


if __name__ == "__main__":
    main()
