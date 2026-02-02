#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: System Diagnostics & Stress Testing
Advanced visualization for Q2 rule comparison analysis.

This module provides:
1. System Stress Test - Robustness analysis under vote chaos (Pareto injection)
2. Decision Boundary Map - Survival geography in (Judge Score, Fan Share) space

Author: MCM Team
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import entropy
from pathlib import Path
import warnings
from viz_config import *

warnings.filterwarnings('ignore')

# Judges' Save merit-based probability (from historical analysis)
MERIT_SAVE_PROB = 0.775

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "results" / "question1" / "full_simulation_bayesian.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots" / "question2"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Target contestants for stress test
TARGET_MEDIOCRITY = "Sean Spicer"  # Low Judge, need to be eliminated
TARGET_TALENT = "Ally Brooke"      # High Judge, need to be protected

# =============================================================================
# Part 1: System Stress Test (Pareto Chaos Injection)
# =============================================================================

def apply_rank_save_rule(judge_scores: np.ndarray, fan_shares: np.ndarray, 
                         names: list) -> str:
    """
    Apply System C (Rank + Judges' Save) rule to determine eliminated contestant.
    
    Args:
        judge_scores: Array of normalized judge scores
        fan_shares: Array of fan share percentages
        names: List of contestant names
        
    Returns:
        Name of eliminated contestant
    """
    n = len(names)
    
    # Compute ranks (1 = best)
    # Judge rank: higher score = better = lower rank number
    judge_ranks = pd.Series(judge_scores).rank(method='average', ascending=False).values
    # Fan rank: higher share = better = lower rank number
    fan_ranks = pd.Series(fan_shares).rank(method='average', ascending=False).values
    
    # Total rank (lower = better)
    total_ranks = judge_ranks + fan_ranks
    
    # Identify Bottom 2 (highest total ranks = worst)
    sorted_indices = np.argsort(-total_ranks)  # Descending by total rank
    bottom2_indices = sorted_indices[:2]
    
    # Judges' Save: Probabilistic based on higher judge score
    b2_scores = judge_scores[bottom2_indices]
    
    if b2_scores[0] > b2_scores[1]:
        # Person 0 is higher scorer
        if np.random.random() < MERIT_SAVE_PROB:
            eliminated_idx = bottom2_indices[1] # Save 0, eliminate 1
        else:
            eliminated_idx = bottom2_indices[0] # Irrational: save 1, eliminate 0
    elif b2_scores[1] > b2_scores[0]:
        # Person 1 is higher scorer
        if np.random.random() < MERIT_SAVE_PROB:
            eliminated_idx = bottom2_indices[0]
        else:
            eliminated_idx = bottom2_indices[1]
    else:
        # Tie: use fan rank as tiebreaker (worse fan rank loses)
        b2_fan_ranks = fan_ranks[bottom2_indices]
        if b2_fan_ranks[0] > b2_fan_ranks[1]:
            eliminated_idx = bottom2_indices[0]
        else:
            eliminated_idx = bottom2_indices[1]
    
    return names[eliminated_idx]


def run_stress_test(df: pd.DataFrame, season: int = 28, week: int = 6,
                    n_simulations: int = 500, lambda_range: tuple = (0.0, 0.5, 0.02)):
    """
    Run Pareto chaos injection stress test.
    
    Args:
        df: Full simulation dataframe
        season, week: Target week for analysis
        n_simulations: Number of Monte Carlo runs per lambda
        lambda_range: (min, max, step) for chaos weight
        
    Returns:
        Tuple of (lambda_values, entropy_values, mediocrity_survival, talent_survival)
    """
    print(f"[STRESS TEST] Analyzing Season {season} Week {week}...")
    
    # Filter data
    week_data = df[(df['season'] == season) & (df['week'] == week)].copy()
    
    if len(week_data) == 0:
        print(f"[ERROR] No data found for S{season} W{week}")
        return None
    
    names = week_data['celebrity_name'].tolist()
    judge_scores = week_data['normalized_score'].values
    real_shares = week_data['estimated_fan_share'].values
    
    # Check target contestants exist
    if TARGET_MEDIOCRITY not in names:
        print(f"[WARNING] {TARGET_MEDIOCRITY} not found, using lowest judge scorer")
        mediocrity_idx = np.argmin(judge_scores)
        mediocrity_name = names[mediocrity_idx]
    else:
        mediocrity_name = TARGET_MEDIOCRITY
        
    if TARGET_TALENT not in names:
        print(f"[WARNING] {TARGET_TALENT} not found, using highest judge scorer")
        talent_idx = np.argmax(judge_scores)
        talent_name = names[talent_idx]
    else:
        talent_name = TARGET_TALENT
    
    print(f"  Mediocrity target: {mediocrity_name}")
    print(f"  Talent target: {talent_name}")
    
    # Lambda sweep
    lambda_min, lambda_max, lambda_step = lambda_range
    lambda_values = np.arange(lambda_min, lambda_max + lambda_step, lambda_step)
    
    entropy_values = []
    mediocrity_survival = []
    talent_survival = []
    
    for lam in lambda_values:
        med_survived = 0
        tal_survived = 0
        entropies = []
        
        for _ in range(n_simulations):
            # Generate Pareto noise (shape=2 for heavy tail)
            noise = np.random.pareto(2.0, size=len(names)) + 1  # Shift to avoid zero
            noise = noise / noise.sum()  # Normalize to sum to 1
            
            # Mix real shares with noise
            simulated_shares = (1 - lam) * real_shares + lam * noise
            simulated_shares = simulated_shares / simulated_shares.sum()  # Renormalize
            
            # Calculate entropy
            h = entropy(simulated_shares)
            entropies.append(h)
            
            # Apply System C rule
            eliminated = apply_rank_save_rule(judge_scores, simulated_shares, names)
            
            # Track survival
            if eliminated != mediocrity_name:
                med_survived += 1
            if eliminated != talent_name:
                tal_survived += 1
        
        entropy_values.append(np.mean(entropies))
        mediocrity_survival.append(med_survived / n_simulations)
        talent_survival.append(tal_survived / n_simulations)
    
    return (np.array(lambda_values), np.array(entropy_values), 
            np.array(mediocrity_survival), np.array(talent_survival),
            mediocrity_name, talent_name)


def plot_stress_test(results, save_path: Path):
    """
    Generate stress test visualization.
    """
    lambda_vals, entropy_vals, med_surv, tal_surv, med_name, tal_name = results
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Main survival curves using Morandi colors
    # Talent (Protection) - Blue
    ax.plot(entropy_vals, tal_surv, color=MORANDI_ACCENT[1], linewidth=3, marker='s', markersize=6,
            label=f'{tal_name} Survival (Talent Protection)')
    
    # Mediocrity (Risk) - Strong Orange (#D95F02)
    ax.plot(entropy_vals, med_surv, color=MORANDI_ACCENT[3], linewidth=3, marker='o', markersize=6,
            label=f'{med_name} Survival (Mediocrity Risk)')
    
    # Identify robustness zone (talent safe >90%, mediocrity <40%)
    robust_mask = (tal_surv > 0.90) & (med_surv < 0.40)
    if robust_mask.any():
        # Find first and last robust points
        robust_indices = np.where(robust_mask)[0]
        robust_start = entropy_vals[robust_indices[0]]
        robust_end = entropy_vals[robust_indices[-1]]
        # Using Mint Green from config
        ax.axvspan(robust_start, robust_end, alpha=0.3, color=MORANDI_COLORS[6], 
                   label=f'Robustness Zone (H ∈ [{robust_start:.2f}, {robust_end:.2f}])')
    
    # Critical threshold line - Gray
    ax.axhline(y=0.5, linestyle='--', color=MORANDI_COLORS[5], alpha=0.7, linewidth=1.5)
    ax.text(entropy_vals[-1] * 0.95, 0.52, '50% Threshold', fontsize=10, ha='right', color=MORANDI_COLORS[5])
    
    # Real-world entropy annotation - Muted Lavender
    real_entropy = entropy_vals[0]
    ax.axvline(x=real_entropy, linestyle=':', color=MORANDI_COLORS[2], alpha=0.8, linewidth=2)
    ax.text(real_entropy + 0.01, 0.1, f'Real Data\nH={real_entropy:.2f}', 
            fontsize=10, color=MORANDI_COLORS[2], va='bottom', fontweight='bold')
    
    # Labels and formatting
    ax.set_xlabel('Vote Distribution Entropy (H) - Higher = More Chaotic', fontsize=13)
    ax.set_ylabel('Survival Probability', fontsize=13)
    ax.set_title('System C Stress Test: Robustness Under Vote Chaos\n'
                 f'S28 W6 | Pareto Noise Injection | {len(lambda_vals)*500:,} Simulations', 
                 fontsize=15, pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(entropy_vals.min() - 0.05, entropy_vals.max() + 0.05)
    ax.legend(fontsize=11, loc='center right')
    ax.grid(True, alpha=0.3, linewidth=0.8)
    
    # Add interpretation box
    interpretation = (
        "Interpretation:\n"
        "• Blue line staying HIGH = System protects talent\n"
        "• Red line staying LOW = System eliminates mediocrity\n"
        "• Green zone = Robust operating range"
    )
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved stress test plot: {save_path}")


# =============================================================================
# Part 2: Real-Data Decision Boundary Map
# =============================================================================

def apply_rank_rule(judge_score: float, fan_share: float, 
                    pool_scores: np.ndarray, pool_shares: np.ndarray) -> bool:
    """Check if virtual contestant is eliminated under Rank system."""
    # Combine virtual with pool
    all_scores = np.append(pool_scores, judge_score)
    all_shares = np.append(pool_shares, fan_share)
    
    # Compute ranks
    judge_ranks = pd.Series(all_scores).rank(method='average', ascending=False).values
    fan_ranks = pd.Series(all_shares).rank(method='average', ascending=False).values
    total_ranks = judge_ranks + fan_ranks
    
    # Virtual contestant is last element
    virtual_rank = total_ranks[-1]
    max_rank = total_ranks.max()
    
    # Eliminated if worst total rank
    return virtual_rank == max_rank


def apply_percent_rule(judge_score: float, fan_share: float,
                       pool_scores: np.ndarray, pool_shares: np.ndarray) -> bool:
    """Check if virtual contestant is eliminated under Percent system."""
    # Combine and normalize
    all_scores = np.append(pool_scores, judge_score)
    all_shares = np.append(pool_shares, fan_share)
    
    # Normalize shares to sum to 1
    all_shares = all_shares / all_shares.sum()
    
    # Convert scores to shares (normalize)
    all_score_shares = all_scores / all_scores.sum()
    
    # Total share
    total_shares = all_score_shares + all_shares
    
    # Virtual contestant is last element
    virtual_total = total_shares[-1]
    min_total = total_shares.min()
    
    # Eliminated if lowest total share
    return virtual_total == min_total


def apply_save_rule(judge_score: float, fan_share: float,
                    pool_scores: np.ndarray, pool_shares: np.ndarray) -> bool:
    """Check if virtual contestant is eliminated under Rank+Save system."""
    # Combine virtual with pool
    all_scores = np.append(pool_scores, judge_score)
    all_shares = np.append(pool_shares, fan_share)
    n = len(all_scores)
    virtual_idx = n - 1
    
    # Compute ranks
    judge_ranks = pd.Series(all_scores).rank(method='average', ascending=False).values
    fan_ranks = pd.Series(all_shares).rank(method='average', ascending=False).values
    total_ranks = judge_ranks + fan_ranks
    
    # Identify Bottom 2
    sorted_indices = np.argsort(-total_ranks)  # Descending
    bottom2_indices = sorted_indices[:2]
    
    # Check if virtual is in Bottom 2
    if virtual_idx not in bottom2_indices:
        return False  # Not eliminated if not in Bottom 2
    
    # In Bottom 2: compare judge scores
    other_idx = bottom2_indices[0] if bottom2_indices[1] == virtual_idx else bottom2_indices[1]
    
    # Probabilistic Judges' Save: 77.5% merit-based
    # If virtual has lower score, usually eliminated (but 22.5% chance saved)
    if all_scores[virtual_idx] < all_scores[other_idx]:
        return np.random.random() < MERIT_SAVE_PROB  # Prob of ELIMINATION
    elif all_scores[virtual_idx] > all_scores[other_idx]:
        return np.random.random() > MERIT_SAVE_PROB  # Prob of ELIMINATION (irrational)
    else:
        # Tie: worse fan rank loses
        return fan_ranks[virtual_idx] > fan_ranks[other_idx]


def get_survival_probability(system_type: str, judge_score: float, fan_share: float,
                             pool_scores: np.ndarray, pool_shares: np.ndarray) -> float:
    """Calculate the mathematical probability of survival (0.0 to 1.0)."""
    if system_type == 'rank':
        return 0.0 if apply_rank_rule(judge_score, fan_share, pool_scores, pool_shares) else 1.0
    elif system_type == 'percent':
        return 0.0 if apply_percent_rule(judge_score, fan_share, pool_scores, pool_shares) else 1.0
    elif system_type == 'save':
        # Combined virtual with pool
        all_scores = np.append(pool_scores, judge_score)
        all_shares = np.append(pool_shares, fan_share)
        n = len(all_scores)
        virtual_idx = n - 1
        
        # Compute ranks
        judge_ranks = pd.Series(all_scores).rank(method='average', ascending=False).values
        fan_ranks = pd.Series(all_shares).rank(method='average', ascending=False).values
        total_ranks = judge_ranks + fan_ranks
        
        # Identify Bottom 2
        sorted_indices = np.argsort(-total_ranks)  # Descending
        bottom2_indices = sorted_indices[:2]
        
        # Check if virtual is in Bottom 2
        if virtual_idx not in bottom2_indices:
            return 1.0  # Safe
        
        # In Bottom 2: compare judge scores
        other_idx = bottom2_indices[0] if bottom2_indices[1] == virtual_idx else bottom2_indices[1]
        
        if all_scores[virtual_idx] < all_scores[other_idx]:
            # Virtual has lower score, survival prob is low
            return 1.0 - MERIT_SAVE_PROB
        elif all_scores[virtual_idx] > all_scores[other_idx]:
            # Virtual has higher score, survival prob is high
            return MERIT_SAVE_PROB
        else:
            # Tie: survival depends on fan rank
            return 1.0 if fan_ranks[virtual_idx] < fan_ranks[other_idx] else 0.0
    return 0.0


def compute_decision_boundary(df: pd.DataFrame, season: int = 28, week: int = 6,
                              score_range: tuple = (15, 30), share_range: tuple = (0.0, 0.20),
                              resolution: int = 80):
    """
    Compute decision boundary maps for all three systems.
    
    Args:
        df: Full simulation dataframe
        season, week: Target week
        score_range: (min, max) judge score for virtual contestant
        share_range: (min, max) fan share for virtual contestant
        resolution: Grid resolution (NxN)
        
    Returns:
        Tuple of (X, Y, Z_rank, Z_percent, Z_save, real_contestants)
    """
    print(f"[DECISION BOUNDARY] Computing for S{season} W{week}...")
    
    # Filter data
    week_data = df[(df['season'] == season) & (df['week'] == week)].copy()
    
    if len(week_data) == 0:
        print(f"[ERROR] No data found for S{season} W{week}")
        return None
    
    names = week_data['celebrity_name'].tolist()
    pool_scores = week_data['normalized_score'].values
    pool_shares = week_data['estimated_fan_share'].values
    
    # Store real contestant info for overlay
    real_contestants = list(zip(names, pool_scores, pool_shares))
    
    print(f"  Real contestants: {len(names)}")
    print(f"  Score range: {score_range}")
    print(f"  Share range: {share_range}")
    print(f"  Resolution: {resolution}x{resolution}")
    
    # Create grid
    scores = np.linspace(score_range[0], score_range[1], resolution)
    shares = np.linspace(share_range[0], share_range[1], resolution)
    X, Y = np.meshgrid(scores, shares)
    
    Z_rank = np.zeros((resolution, resolution))
    Z_percent = np.zeros((resolution, resolution))
    Z_save = np.zeros((resolution, resolution))
    
    # Grid search - Using probabilistic mapping
    for i in range(resolution):
        for j in range(resolution):
            score = X[j, i]
            share = Y[j, i]
            
            # Normalize score to 0-1 range (like real data)
            norm_score = score / 30.0
            
            Z_rank[j, i] = get_survival_probability('rank', norm_score, share, pool_scores, pool_shares)
            Z_percent[j, i] = get_survival_probability('percent', norm_score, share, pool_scores, pool_shares)
            Z_save[j, i] = get_survival_probability('save', norm_score, share, pool_scores, pool_shares)
    
    print(f"  Grid computed: {resolution**2} points")
    
    return (X, Y, Z_rank, Z_percent, Z_save, real_contestants)


def plot_decision_boundary(results, save_path: Path):
    """
    Generate 3-panel decision boundary map.
    """
    X, Y, Z_rank, Z_percent, Z_save, real_contestants = results
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    systems = [
        ('System A: Rank', Z_rank, MORANDI_COLORS[0]),
        ('System B: Percent', Z_percent, MORANDI_COLORS[1]),
        ('System C: Rank+Save', Z_save, MORANDI_COLORS[3])
    ]
    
    # Create customized colormap with varying alpha and colors
    # Z=0.0 (Strong Orange/Dead) -> System A/B elimination
    # Z=0.4 (Cherry Blossom Pink/Risky) -> System C high-risk zone
    from matplotlib.colors import to_rgba
    orange_opaque = to_rgba(MORANDI_ACCENT[3], 0.95)
    pink_trans = to_rgba(MORANDI_COLORS[1], 0.7)  # Cherry Blossom Pink
    white_trans = to_rgba('#FFFFFF', 1.0)
    blue_opaque = to_rgba(MORANDI_COLORS[0], 0.85)

    cmap_smooth = LinearSegmentedColormap.from_list('Survival', [
        (0.0, orange_opaque),  # 0.0 - Strong Orange
        (0.4, pink_trans),     # 0.4 - Cherry Blossom Pink
        (0.5, white_trans),    # 0.5 - Neutral midpoint
        (1.0, blue_opaque)     # 1.0 - Sky Blue
    ])
    
    # Binary map for System A/B
    cmap_binary = ListedColormap([orange_opaque, blue_opaque])
    
    for ax, (title, Z, color) in zip(axes, systems):
        # Determine plot style based on whether data is probabilistic
        is_probabilistic = not np.all(np.logical_or(Z == 0, Z == 1))
        
        if is_probabilistic:
            # Use smooth gradient for probabilistic systems (e.g., System C)
            # Remove global alpha to use the RGBA alpha from colormap
            contour = ax.contourf(X, Y * 100, Z, levels=np.linspace(0, 1, 21), 
                                  cmap=cmap_smooth)
            # Add a dashed line for the 50% "Expected" boundary
            ax.contour(X, Y * 100, Z, levels=[0.5], colors='black', 
                      linewidths=1.5, linestyles='--')
        else:
            # Use binary colors for deterministic systems (System A & B)
            contour = ax.contourf(X, Y * 100, Z, levels=[-0.5, 0.5, 1.5], 
                                  cmap=cmap_binary)
            # Add solid contour lines
            ax.contour(X, Y * 100, Z, levels=[0.5], colors='black', linewidths=2)
        
        # Overlay real contestants
        for name, score, share in real_contestants:
            # Convert normalized score back to 0-30 scale for plotting
            plot_score = score * 30
            plot_share = share * 100
            
            # Highlight key contestants
            if 'Spicer' in name:
                marker = 'X'
                ms = 200
                color_pt = 'darkred'
                zorder = 10
            elif 'Ally' in name:
                marker = '*'
                ms = 300
                color_pt = 'blue'
                zorder = 10
            else:
                marker = 'o'
                ms = 80
                color_pt = 'white'
                zorder = 5
            
            ax.scatter(plot_score, plot_share, marker=marker, s=ms, 
                      c=color_pt, edgecolors='black', linewidth=1.5, zorder=zorder)
            
            # Label key contestants
            if 'Spicer' in name or 'Ally' in name:
                ax.annotate(name.split()[-1], (plot_score, plot_share),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Judge Score (out of 30)', fontsize=12)
        ax.set_ylabel('Fan Share (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Calculate safe zone percentage
        safe_pct = Z.mean() * 100
        ax.text(0.95, 0.05, f'Safe Area: {safe_pct:.1f}%', 
                transform=ax.transAxes, fontsize=10, ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create legend
    safe_patch = mpatches.Patch(color=MORANDI_COLORS[0], alpha=0.85, label='Safe Zone')
    elim_patch = mpatches.Patch(color=MORANDI_ACCENT[3], alpha=0.85, label='Elimination Zone')
    boundary_line = plt.Line2D([0], [0], color='black', linewidth=2, label='Survival Boundary')
    prob_line = plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='50% Prob Boundary')
    
    spicer_marker = plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred',
                                markersize=12, label='Sean Spicer (Mediocrity)')
    ally_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
                              markersize=15, label='Ally Brooke (Talent)')
    
    fig.legend(handles=[safe_patch, elim_patch, boundary_line, prob_line, spicer_marker, ally_marker],
               loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('S28 Week 6: The Geometry of Survival\n'
                 'Where Would a Virtual Contestant Land?', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved decision boundary plot: {save_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_diagnostics():
    """Run all diagnostic analyses."""
    print("="*70)
    print("MCM 2026 Problem C - System Diagnostics")
    print("="*70)
    
    # Load data
    print(f"\n[INFO] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} records")
    
    # Part 1: Stress Test
    print("\n" + "-"*50)
    print("PART 1: System Stress Test (Chaos Injection)")
    print("-"*50)
    
    stress_results = run_stress_test(df, season=28, week=6, n_simulations=500)
    if stress_results:
        plot_stress_test(stress_results, PLOTS_DIR / "q2_system_stress_test.png")
        
        # Print summary
        _, entropy_vals, med_surv, tal_surv, med_name, tal_name = stress_results
        print(f"\n[SUMMARY] At real entropy (H={entropy_vals[0]:.2f}):")
        print(f"  - {tal_name} survival: {tal_surv[0]:.1%}")
        print(f"  - {med_name} survival: {med_surv[0]:.1%}")
    
    # Part 2: Decision Boundary
    print("\n" + "-"*50)
    print("PART 2: Decision Boundary Map")
    print("-"*50)
    
    boundary_results = compute_decision_boundary(df, season=28, week=6, resolution=80)
    if boundary_results:
        plot_decision_boundary(boundary_results, PLOTS_DIR / "q2_real_decision_boundary.png")
        
        # Print summary
        X, Y, Z_rank, Z_percent, Z_save, real = boundary_results
        print(f"\n[SUMMARY] Elimination Zone Coverage:")
        print(f"  - Rank System: {(1-Z_rank.mean())*100:.1f}%")
        print(f"  - Percent System: {(1-Z_percent.mean())*100:.1f}%")
        print(f"  - Rank+Save: {(1-Z_save.mean())*100:.1f}%")
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - {PLOTS_DIR / 'q2_system_stress_test.png'}")
    print(f"  - {PLOTS_DIR / 'q2_real_decision_boundary.png'}")


if __name__ == "__main__":
    run_diagnostics()
