#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Bayesian vs Basic Monte Carlo Comparison Visualization

This script generates visualizations comparing the Bayesian and Basic
Monte Carlo simulation approaches.

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BASIC_PATH = Path(__file__).parent.parent / "results" / "fan_vote_estimates.csv"
BAYESIAN_PATH = Path(__file__).parent.parent / "results" / "fan_vote_estimates_bayesian.csv"
IMAGES_DIR = Path(__file__).parent.parent / "images"

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")


def load_data():
    """Load both simulation results."""
    basic = pd.read_csv(BASIC_PATH)
    bayesian = pd.read_csv(BAYESIAN_PATH)
    return basic, bayesian


def plot_prior_strength_evolution(bayesian_df: pd.DataFrame):
    """
    Figure: Prior Strength (α_sum) Evolution Over Weeks
    
    Shows how confidence accumulates over time in the Bayesian model.
    """
    print("[PLOT] Generating Prior Strength Evolution...")
    
    # Group by week
    strength_by_week = bayesian_df.groupby('week').agg({
        'prior_strength': 'mean',
        'posterior_strength': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both prior and posterior strength
    ax.plot(strength_by_week['week'], strength_by_week['prior_strength'],
            'o-', color='#3498DB', linewidth=2.5, markersize=8, label='Prior α_sum')
    ax.plot(strength_by_week['week'], strength_by_week['posterior_strength'],
            's--', color='#E74C3C', linewidth=2.5, markersize=8, label='Posterior α_sum')
    
    # Add reference line for initial α
    n_contestants_avg = bayesian_df.groupby('week')['prior_strength'].count().mean()
    initial_alpha_sum = 0.8 * 12  # Approximate initial value
    ax.axhline(y=initial_alpha_sum, color='gray', linestyle=':', alpha=0.7,
               label=f'Initial (Week 1) ≈ {initial_alpha_sum:.1f}')
    
    ax.set_xlabel('Week Number', fontweight='bold')
    ax.set_ylabel('Dirichlet Concentration (α_sum)', fontweight='bold')
    ax.set_title('Bayesian Model: Confidence Accumulation Over Time\n(Higher α = More Concentrated Distribution)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper left')
    ax.set_xlim(0.5, strength_by_week['week'].max() + 0.5)
    
    # Add annotation
    ax.annotate('Evidence\nAccumulates',
                xy=(6, strength_by_week[strength_by_week['week']==6]['posterior_strength'].values[0]),
                xytext=(8, strength_by_week['posterior_strength'].max() * 0.7),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    
    output_path = IMAGES_DIR / 'figure_bayesian_alpha_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_estimate_stability(basic_df: pd.DataFrame, bayesian_df: pd.DataFrame):
    """
    Figure: Estimate Stability Comparison
    
    Shows week-to-week stability of fan share estimates.
    """
    print("[PLOT] Generating Estimate Stability Comparison...")
    
    # Calculate week-to-week changes for a sample contestant across seasons
    def calc_week_changes(df, label):
        changes = []
        for season in df['season'].unique():
            season_data = df[df['season'] == season].sort_values('week')
            for name in season_data['celebrity_name'].unique():
                contestant_data = season_data[season_data['celebrity_name'] == name]
                if len(contestant_data) > 1:
                    shares = contestant_data['estimated_fan_share'].values
                    week_changes = np.abs(np.diff(shares))
                    changes.extend(week_changes)
        return changes
    
    basic_changes = calc_week_changes(basic_df, 'Basic')
    bayesian_changes = calc_week_changes(bayesian_df, 'Bayesian')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Histogram comparison
    ax1 = axes[0]
    ax1.hist(basic_changes, bins=30, alpha=0.6, color='#3498DB', label='Basic MC', density=True)
    ax1.hist(bayesian_changes, bins=30, alpha=0.6, color='#E74C3C', label='Bayesian MC', density=True)
    ax1.axvline(np.mean(basic_changes), color='#3498DB', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(bayesian_changes), color='#E74C3C', linestyle='--', linewidth=2)
    ax1.set_xlabel('Week-to-Week |Δ Fan Share|', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Distribution of Week-to-Week Changes', fontweight='bold')
    ax1.legend()
    
    # Right: Box plot
    ax2 = axes[1]
    data_for_box = pd.DataFrame({
        'Change': basic_changes + bayesian_changes,
        'Method': ['Basic MC'] * len(basic_changes) + ['Bayesian MC'] * len(bayesian_changes)
    })
    sns.boxplot(data=data_for_box, x='Method', y='Change', ax=ax2, 
                palette=['#3498DB', '#E74C3C'])
    ax2.set_ylabel('Week-to-Week |Δ Fan Share|', fontweight='bold')
    ax2.set_title('Estimate Volatility Comparison', fontweight='bold')
    
    # Add statistics
    basic_mean = np.mean(basic_changes)
    bayesian_mean = np.mean(bayesian_changes)
    reduction = (basic_mean - bayesian_mean) / basic_mean * 100
    
    ax2.annotate(f'Mean: {basic_mean:.4f}', xy=(0, basic_mean), 
                 xytext=(0.3, basic_mean + 0.02), fontsize=9)
    ax2.annotate(f'Mean: {bayesian_mean:.4f}', xy=(1, bayesian_mean),
                 xytext=(0.7, bayesian_mean + 0.02), fontsize=9)
    
    plt.suptitle(f'Bayesian Model Reduces Volatility by {reduction:.1f}%', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = IMAGES_DIR / 'figure_bayesian_stability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_confidence_by_week(basic_df: pd.DataFrame, bayesian_df: pd.DataFrame):
    """
    Figure: Confidence Evolution by Week - Both Methods
    """
    print("[PLOT] Generating Confidence by Week Comparison...")
    
    # Filter valid data
    basic_valid = basic_df[basic_df['confidence'].notna() & (basic_df['confidence'] > 0)]
    bayesian_valid = bayesian_df[bayesian_df['confidence'].notna() & (bayesian_df['confidence'] > 0)]
    
    basic_by_week = basic_valid.groupby('week')['confidence'].agg(['mean', 'std']).reset_index()
    bayesian_by_week = bayesian_valid.groupby('week')['confidence'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both methods
    ax.errorbar(basic_by_week['week'], basic_by_week['mean'], 
                yerr=basic_by_week['std']*0.5, fmt='o-', 
                color='#3498DB', linewidth=2, markersize=8, capsize=4,
                label='Basic Monte Carlo')
    ax.errorbar(bayesian_by_week['week'], bayesian_by_week['mean'],
                yerr=bayesian_by_week['std']*0.5, fmt='s-',
                color='#E74C3C', linewidth=2, markersize=8, capsize=4,
                label='Bayesian Monte Carlo')
    
    ax.set_xlabel('Week Number', fontweight='bold')
    ax.set_ylabel('Mean Confidence', fontweight='bold')
    ax.set_title('Model Confidence by Week: Basic vs Bayesian',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    ax.set_xlim(0.5, max(basic_by_week['week'].max(), bayesian_by_week['week'].max()) + 0.5)
    ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    
    output_path = IMAGES_DIR / 'figure_bayesian_confidence_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_sample_season_trajectory(basic_df: pd.DataFrame, bayesian_df: pd.DataFrame, season: int = 10):
    """
    Figure: Sample Season Trajectory Comparison
    
    Shows how estimates evolve for contestants in a sample season.
    """
    print(f"[PLOT] Generating Season {season} Trajectory Comparison...")
    
    basic_season = basic_df[basic_df['season'] == season].copy()
    bayesian_season = bayesian_df[bayesian_df['season'] == season].copy()
    
    # Get finalists (contestants who lasted longest)
    final_week = basic_season['week'].max()
    finalists = basic_season[basic_season['week'] == final_week]['celebrity_name'].unique()[:3]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(finalists)))
    
    for i, name in enumerate(finalists):
        # Basic
        basic_data = basic_season[basic_season['celebrity_name'] == name].sort_values('week')
        axes[0].plot(basic_data['week'], basic_data['estimated_fan_share'],
                     'o-', color=colors[i], linewidth=2, markersize=6, label=name[:15])
        
        # Bayesian
        bayesian_data = bayesian_season[bayesian_season['celebrity_name'] == name].sort_values('week')
        axes[1].plot(bayesian_data['week'], bayesian_data['estimated_fan_share'],
                     'o-', color=colors[i], linewidth=2, markersize=6, label=name[:15])
    
    axes[0].set_title('Basic Monte Carlo', fontweight='bold')
    axes[0].set_xlabel('Week', fontweight='bold')
    axes[0].set_ylabel('Estimated Fan Share', fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_ylim(0, 0.5)
    
    axes[1].set_title('Bayesian Monte Carlo', fontweight='bold')
    axes[1].set_xlabel('Week', fontweight='bold')
    axes[1].set_ylabel('Estimated Fan Share', fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_ylim(0, 0.5)
    
    plt.suptitle(f'Season {season} Fan Share Trajectories: Basic vs Bayesian',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    output_path = IMAGES_DIR / f'figure_bayesian_season{season}_trajectory.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def generate_summary_table(basic_df: pd.DataFrame, bayesian_df: pd.DataFrame):
    """Generate and print summary comparison table."""
    print("\n" + "="*70)
    print("BAYESIAN vs BASIC MONTE CARLO: SUMMARY COMPARISON")
    print("="*70)
    
    basic_valid = basic_df[basic_df['confidence'].notna() & (basic_df['confidence'] > 0)]
    bayesian_valid = bayesian_df[bayesian_df['confidence'].notna() & (bayesian_df['confidence'] > 0)]
    
    # Calculate stability (week-to-week changes)
    def calc_volatility(df):
        changes = []
        for season in df['season'].unique():
            season_data = df[df['season'] == season].sort_values('week')
            for name in season_data['celebrity_name'].unique():
                contestant_data = season_data[season_data['celebrity_name'] == name]
                if len(contestant_data) > 1:
                    shares = contestant_data['estimated_fan_share'].values
                    changes.extend(np.abs(np.diff(shares)))
        return np.mean(changes) if changes else 0
    
    basic_vol = calc_volatility(basic_df)
    bayesian_vol = calc_volatility(bayesian_df)
    
    # Late-week confidence (weeks 6+)
    basic_late = basic_valid[basic_valid['week'] >= 6]['confidence'].mean()
    bayesian_late = bayesian_valid[bayesian_valid['week'] >= 6]['confidence'].mean()
    
    # Print table
    print(f"\n{'Metric':<35} {'Basic MC':>15} {'Bayesian MC':>15} {'Change':>12}")
    print("-" * 77)
    print(f"{'Mean Confidence':<35} {basic_valid['confidence'].mean():>15.4f} {bayesian_valid['confidence'].mean():>15.4f} {bayesian_valid['confidence'].mean() - basic_valid['confidence'].mean():>+12.4f}")
    print(f"{'Median Confidence':<35} {basic_valid['confidence'].median():>15.4f} {bayesian_valid['confidence'].median():>15.4f} {bayesian_valid['confidence'].median() - basic_valid['confidence'].median():>+12.4f}")
    print(f"{'Late-Week Confidence (Week≥6)':<35} {basic_late:>15.4f} {bayesian_late:>15.4f} {bayesian_late - basic_late:>+12.4f}")
    print(f"{'Estimate Volatility (|ΔShare|)':<35} {basic_vol:>15.4f} {bayesian_vol:>15.4f} {bayesian_vol - basic_vol:>+12.4f}")
    vol_reduction = (basic_vol - bayesian_vol) / basic_vol * 100
    print(f"{'Volatility Reduction':<35} {'-':>15} {'-':>15} {vol_reduction:>+11.1f}%")
    
    # Prior strength evolution
    if 'posterior_strength' in bayesian_df.columns:
        early_strength = bayesian_df[bayesian_df['week'] <= 2]['posterior_strength'].mean()
        late_strength = bayesian_df[bayesian_df['week'] >= 7]['posterior_strength'].mean()
        print(f"{'Prior Strength (Early, Week≤2)':<35} {'N/A':>15} {early_strength:>15.2f} {'-':>12}")
        print(f"{'Prior Strength (Late, Week≥7)':<35} {'N/A':>15} {late_strength:>15.2f} {'-':>12}")
        print(f"{'Strength Growth':<35} {'N/A':>15} {'-':>15} {(late_strength/early_strength - 1)*100:>+11.1f}%")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("""
1. TEMPORAL CONTINUITY: Bayesian model accumulates evidence over weeks,
   reflected in growing α_sum (prior strength).

2. ESTIMATE STABILITY: Bayesian estimates show lower week-to-week volatility,
   as prior constrains extreme jumps.

3. UNCERTAINTY QUANTIFICATION: α parameters explicitly encode confidence,
   enabling principled uncertainty estimates.

4. THEORETICAL SOUNDNESS: Unlike heuristic averaging, Bayesian updating
   has formal probabilistic foundations.
""")


def main():
    """Generate all comparison visualizations."""
    print("="*60)
    print("Bayesian vs Basic Monte Carlo - Comparison Visualizations")
    print("="*60)
    
    # Ensure output directory exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    basic_df, bayesian_df = load_data()
    print(f"Loaded Basic: {len(basic_df)} rows, Bayesian: {len(bayesian_df)} rows")
    
    # Generate visualizations
    plot_prior_strength_evolution(bayesian_df)
    plot_estimate_stability(basic_df, bayesian_df)
    plot_confidence_by_week(basic_df, bayesian_df)
    plot_sample_season_trajectory(basic_df, bayesian_df, season=10)
    
    # Print summary
    generate_summary_table(basic_df, bayesian_df)
    
    print(f"\n✓ All visualizations saved to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
