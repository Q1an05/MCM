#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Three-Stage Bayesian Diagnostic Analysis
Posterior Predictive Check for Chaos Parameter Optimization

Phase 1: Pure Rationality Baseline (λ=0) - Collect Survival Deficits
Phase 2: Distribution Identification - Visualize Skill-Bias Pattern
Phase 3: Parameter Optimization - Grid Search for Optimal λ

Author: MCM Team
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "data_processed" / "dwts_simulation_input.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
N_SIMULATIONS = 5000
RANDOM_SEED = 42

# Bayesian parameters (from existing model)
INITIAL_ALPHA = 1.0
LEARNING_RATE = 0.4
EVIDENCE_BOOST = 5.0
MIN_ALPHA = 0.1
SKILL_IMPACT_FACTOR = 0.3

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class WeekState:
    """State information passed between weeks for momentum tracking"""
    alpha: np.ndarray
    contestant_names: List[str]
    prev_fan_shares: Optional[np.ndarray] = None  # For momentum calculation
    
    @property
    def n_contestants(self) -> int:
        return len(self.contestant_names)


# =============================================================================
# Core Simulation Functions
# =============================================================================
def shares_to_ranks(shares: np.ndarray) -> np.ndarray:
    """Convert shares to ranks (1 = highest share = best)."""
    order = np.argsort(-shares, axis=1)
    ranks = np.argsort(order, axis=1) + 1
    return ranks.astype(float)


def generate_pure_rational_shares(alpha: np.ndarray, n_sims: int) -> np.ndarray:
    """
    Generate fan shares using PURE Dirichlet (no chaos component).
    This is the λ=0 baseline for Phase 1.
    """
    return np.random.dirichlet(alpha, size=n_sims)


def generate_mixed_shares(alpha: np.ndarray, n_sims: int, 
                          chaos_weight: float, use_pareto: bool = True) -> np.ndarray:
    """
    Generate fan shares using Mixture Model with Pareto chaos.
    
    Model: (1-λ) * Dirichlet(α_skill) + λ * Dirichlet(α_chaos_pareto)
    """
    n_chaos = int(n_sims * chaos_weight)
    n_skill = n_sims - n_chaos
    
    # 1. Rational/Skill Component
    samples_skill = np.random.dirichlet(alpha, size=n_skill)
    
    if n_chaos == 0:
        return samples_skill
    
    # 2. Chaos Component with Pareto weighting (biased toward underdogs)
    if use_pareto:
        # Pareto weights: higher weight for lower-alpha contestants (underdogs)
        # Invert alpha to give more weight to weaker performers
        alpha_inverted = 1.0 / (alpha + 0.1)  # Add small value to avoid div by zero
        pareto_noise = np.random.pareto(a=2.0, size=len(alpha)) + 1
        chaos_alpha = alpha_inverted * pareto_noise
        chaos_alpha = np.maximum(chaos_alpha, MIN_ALPHA)
    else:
        # Uniform chaos (original approach)
        chaos_alpha = np.ones_like(alpha)
    
    samples_chaos = np.random.dirichlet(chaos_alpha, size=n_chaos)
    
    return np.concatenate([samples_skill, samples_chaos], axis=0)


def apply_rank_rule(judge_ranks: np.ndarray, fan_shares: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANK elimination rule (Seasons 1-2).
    Returns: (eliminated_idx, total_scores)
    """
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks
    eliminated_idx = np.argmax(total_scores + fan_ranks * 0.001, axis=1)
    return eliminated_idx, total_scores


def apply_percent_rule(judge_shares: np.ndarray, fan_shares: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PERCENT elimination rule (Seasons 3-27).
    Returns: (eliminated_idx, total_scores)
    """
    total_scores = judge_shares + fan_shares
    eliminated_idx = np.argmin(total_scores, axis=1)
    return eliminated_idx, total_scores


def apply_rank_with_save_rule(judge_ranks: np.ndarray, judge_scores: np.ndarray,
                               fan_shares: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANK_WITH_SAVE elimination rule (Seasons 28+).
    Bottom 2 go to judges' save; lower judge score is eliminated.
    Returns: (eliminated_idx, total_scores)
    """
    n_sims, n_contestants = fan_shares.shape
    
    if n_contestants < 2:
        return np.zeros(n_sims, dtype=int), fan_shares
    
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks + fan_ranks * 1e-6
    
    if n_contestants == 2:
        if judge_scores[0] < judge_scores[1]:
            return np.zeros(n_sims, dtype=int), total_scores
        elif judge_scores[1] < judge_scores[0]:
            return np.ones(n_sims, dtype=int), total_scores
        else:
            return np.argmax(fan_ranks, axis=1), total_scores
    
    sorted_indices = np.argsort(-total_scores, axis=1)
    bottom2_first = sorted_indices[:, 0]
    bottom2_second = sorted_indices[:, 1]
    
    judge_scores_first = judge_scores[bottom2_first]
    judge_scores_second = judge_scores[bottom2_second]
    
    eliminated_idx = np.where(
        judge_scores_first < judge_scores_second,
        bottom2_first,
        np.where(judge_scores_second < judge_scores_first, bottom2_second, bottom2_first)
    )
    
    return eliminated_idx, total_scores


# =============================================================================
# Phase 1: Pure Rationality Baseline (Survival Deficit Collection)
# =============================================================================
def compute_rational_score(week_data: pd.DataFrame, prev_shares: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute "Rational Score" = Judge Performance + Momentum
    """
    judge_scores = week_data['normalized_score'].values.astype(float)
    
    # Momentum: difference from previous week's estimated performance
    if prev_shares is not None and len(prev_shares) == len(judge_scores):
        momentum = prev_shares - np.mean(prev_shares)
    else:
        momentum = np.zeros_like(judge_scores)
    
    # Combine: Judge score is primary, momentum is secondary
    rational_score = judge_scores + 0.2 * momentum
    return rational_score


def compute_survival_deficit(week_data: pd.DataFrame, 
                              predicted_scores: np.ndarray,
                              rule_system: str) -> pd.DataFrame:
    """
    Compute Survival Deficit Gap for each surviving contestant.
    
    Gap > 0 means the contestant was predicted to be eliminated but survived.
    """
    n_contestants = len(week_data)
    contestants = week_data['celebrity_name'].values
    is_eliminated = week_data['is_eliminated'].values
    
    # Z-score of rational scores
    mu = np.mean(predicted_scores)
    sigma = np.std(predicted_scores, ddof=1) if np.std(predicted_scores) > 1e-6 else 1.0
    z_scores = (predicted_scores - mu) / sigma
    
    records = []
    
    if rule_system in ['Rank', 'Rank_With_Save']:
        # Rank-based: lower score = higher rank = more likely eliminated
        predicted_ranks = np.argsort(np.argsort(-predicted_scores)) + 1  # 1 = best
        
        if rule_system == 'Rank_With_Save':
            # S28+: Bottom 2 at risk, so safe threshold = 3rd from bottom
            safe_threshold_rank = n_contestants - 2
        else:
            # S1-2: Last place eliminated
            safe_threshold_rank = n_contestants - 1
        
        for i, (name, eliminated, z, rank) in enumerate(zip(contestants, is_eliminated, z_scores, predicted_ranks)):
            if eliminated:
                continue  # Skip actually eliminated contestants
            
            # Gap = how far below the safe threshold (positive = predicted elimination)
            gap = rank - safe_threshold_rank
            predicted_elimination = (rank >= safe_threshold_rank)
            
            records.append({
                'contestant': name,
                'Rational_Score_Z': z,
                'Predicted_Rank': rank,
                'Safe_Threshold': safe_threshold_rank,
                'Predicted_Elimination': predicted_elimination,
                'Survival_Deficit_Gap': max(0, gap),  # Only positive gaps
                'Era': 'Rank' if rule_system == 'Rank' else 'Rank_With_Save'
            })
    
    else:  # Percent Era (S3-27)
        # Percent-based: higher score = safer
        # Find the actual safe threshold (lowest surviving contestant's score)
        surviving_scores = predicted_scores[~is_eliminated]
        if len(surviving_scores) > 0:
            safe_threshold_score = np.min(surviving_scores)
        else:
            safe_threshold_score = np.min(predicted_scores)
        
        for i, (name, eliminated, z, score) in enumerate(zip(contestants, is_eliminated, z_scores, predicted_scores)):
            if eliminated:
                continue
            
            # Gap = how much below the safe threshold (positive = predicted elimination)
            gap = safe_threshold_score - score
            predicted_elimination = (score < safe_threshold_score)
            
            records.append({
                'contestant': name,
                'Rational_Score_Z': z,
                'Predicted_Score': score,
                'Safe_Threshold': safe_threshold_score,
                'Predicted_Elimination': predicted_elimination,
                'Survival_Deficit_Gap': max(0, gap),
                'Era': 'Percent'
            })
    
    return pd.DataFrame(records)


def run_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: Run Pure Rationality baseline (λ=0) and collect Survival Deficits.
    
    Returns DataFrame with columns:
    - Season, Week, Contestant
    - Rational_Score_Z, Predicted_Elimination, Survival_Deficit_Gap, Era
    """
    print("\n" + "="*70)
    print("   PHASE 1: Pure Rationality Baseline (λ=0)")
    print("="*70)
    
    np.random.seed(RANDOM_SEED)
    
    all_diagnostics = []
    seasons = sorted(df['season'].unique())
    
    for season in tqdm(seasons, desc="Processing seasons"):
        season_data = df[df['season'] == season].copy()
        rule_system = season_data['rule_system'].iloc[0]
        weeks = sorted(season_data['week'].unique())
        
        prev_shares = None
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()
            week_data = week_data.sort_values('celebrity_name').reset_index(drop=True)
            
            # Skip non-elimination weeks
            if not week_data['is_eliminated'].any():
                continue
            
            # Compute rational score with momentum
            rational_scores = compute_rational_score(week_data, prev_shares)
            
            # Compute survival deficits
            week_diag = compute_survival_deficit(week_data, rational_scores, rule_system)
            
            if len(week_diag) > 0:
                week_diag['season'] = season
                week_diag['week'] = week
                all_diagnostics.append(week_diag)
            
            # Update momentum for next week (use judge scores as proxy)
            prev_shares = week_data['normalized_score'].values
    
    df_diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    
    # Summary
    anomalies = df_diagnostics[df_diagnostics['Survival_Deficit_Gap'] > 0]
    print(f"\n   Total survivor records: {len(df_diagnostics)}")
    print(f"   Anomalies (Gap > 0): {len(anomalies)} ({100*len(anomalies)/len(df_diagnostics):.1f}%)")
    print(f"   By Era: {anomalies.groupby('Era').size().to_dict()}")
    
    return df_diagnostics


# =============================================================================
# Phase 2: Distribution Identification (Visualization)
# =============================================================================
def plot_skill_bias_scatter(df_diagnostics: pd.DataFrame) -> None:
    """
    Phase 2: Generate Skill-Bias Scatter Plot.
    
    X-axis: Rational_Score_Z (lower = weaker performer)
    Y-axis: Normalized_Gap (how much help they needed)
    
    Expected pattern: "L-shape" or "Trumpet" - high gaps for low Z-scores
    """
    print("\n" + "="*70)
    print("   PHASE 2: Distribution Identification (Visualization)")
    print("="*70)
    
    # Filter anomalies only
    anomalies = df_diagnostics[df_diagnostics['Survival_Deficit_Gap'] > 0].copy()
    
    if len(anomalies) == 0:
        print("   [WARNING] No anomalies found. Cannot generate plot.")
        return
    
    # Check if 'Era' column exists
    if 'Era' not in anomalies.columns:
        print("   [ERROR] 'Era' column not found in anomalies DataFrame.")
        print(f"   Available columns: {anomalies.columns.tolist()}")
        return
    
    # Normalize gaps across eras (Min-Max scaling within each era)
    def normalize_gap(group):
        gap_min = group['Survival_Deficit_Gap'].min()
        gap_max = group['Survival_Deficit_Gap'].max()
        if gap_max - gap_min > 1e-6:
            group['Normalized_Gap'] = (group['Survival_Deficit_Gap'] - gap_min) / (gap_max - gap_min)
        else:
            group['Normalized_Gap'] = 0.5
        return group
    
    # Apply normalization by era
    try:
        anomalies = anomalies.groupby('Era', group_keys=False).apply(normalize_gap).reset_index(drop=True)
    except Exception as e:
        print(f"   [ERROR] Failed to normalize gaps by era: {e}")
        # Fallback: normalize globally
        gap_min = anomalies['Survival_Deficit_Gap'].min()
        gap_max = anomalies['Survival_Deficit_Gap'].max()
        if gap_max - gap_min > 1e-6:
            anomalies['Normalized_Gap'] = (anomalies['Survival_Deficit_Gap'] - gap_min) / (gap_max - gap_min)
        else:
            anomalies['Normalized_Gap'] = 0.5
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by era
    era_colors = {'Rank': '#e74c3c', 'Percent': '#3498db', 'Rank_With_Save': '#2ecc71'}
    
    # Check again if 'Era' column exists after processing
    if 'Era' not in anomalies.columns:
        # Plot all data as single group
        ax.scatter(anomalies['Rational_Score_Z'], anomalies['Normalized_Gap'],
                   alpha=0.6, s=80, label=f'All Anomalies (n={len(anomalies)})',
                   color='#95a5a6', edgecolors='white', linewidth=0.5)
    else:
        for era in anomalies['Era'].unique():
            era_data = anomalies[anomalies['Era'] == era]
            ax.scatter(era_data['Rational_Score_Z'], era_data['Normalized_Gap'],
                       alpha=0.6, s=80, label=f'{era} Era (n={len(era_data)})',
                       color=era_colors.get(era, '#95a5a6'), edgecolors='white', linewidth=0.5)
    
    # Add trend line (LOWESS or polynomial)
    from numpy.polynomial import polynomial as P
    x = anomalies['Rational_Score_Z'].values
    y = anomalies['Normalized_Gap'].values
    
    # Fit polynomial for trend visualization
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    
    try:
        # Fit 2nd degree polynomial
        coefs = np.polyfit(x_sorted, y_sorted, 2)
        poly = np.poly1d(coefs)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_smooth, poly(x_smooth), 'k--', linewidth=2, alpha=0.7, label='Trend (Polynomial)')
    except:
        pass
    
    ax.set_xlabel('Rational Score (Z-score)\n← Weaker Performers | Stronger Performers →', fontsize=12)
    ax.set_ylabel('Normalized Survival Gap\n(How much "help" they needed)', fontsize=12)
    ax.set_title('Phase 2: Skill-Bias Scatter Plot\n"L-shape" confirms Pareto chaos distribution', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Underdog Zone\n(High chaos needed)', 
                xy=(anomalies['Rational_Score_Z'].min() + 0.5, 0.8),
                fontsize=10, ha='center', style='italic', color='#7f8c8d')
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / "skill_bias_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {plot_path}")
    
    # Statistical summary
    correlation = anomalies['Rational_Score_Z'].corr(anomalies['Normalized_Gap'])
    print(f"   Correlation (Z vs Gap): {correlation:.3f}")
    print(f"   → {'Negative correlation confirms Underdog Effect!' if correlation < -0.1 else 'Weak/no correlation.'}")


# =============================================================================
# Phase 3: Parameter Optimization (Grid Search)
# =============================================================================
def simulate_week_with_lambda(week_data: pd.DataFrame, 
                               chaos_weight: float,
                               prev_alpha: np.ndarray,
                               use_pareto: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single week with given chaos weight.
    Returns: (simulated_eliminated_idx array, updated_alpha)
    """
    n_contestants = len(week_data)
    contestant_names = week_data['celebrity_name'].tolist()
    rule_system = week_data['rule_system'].iloc[0]
    
    # Build alpha from judge scores
    judge_scores = week_data['normalized_score'].values.astype(float)
    
    if len(judge_scores) > 1:
        mu = np.mean(judge_scores)
        sigma = np.std(judge_scores, ddof=1) if np.std(judge_scores) > 1e-6 else 1.0
        z_scores = (judge_scores - mu) / sigma
        skill_multipliers = 1.0 + SKILL_IMPACT_FACTOR * z_scores
        skill_multipliers = np.clip(skill_multipliers, 0.5, 2.0)
        alpha = prev_alpha * skill_multipliers if prev_alpha is not None else INITIAL_ALPHA * skill_multipliers
    else:
        alpha = prev_alpha if prev_alpha is not None else np.full(n_contestants, INITIAL_ALPHA)
    
    alpha = np.maximum(alpha, MIN_ALPHA)
    
    # Generate fan shares with chaos
    fan_shares = generate_mixed_shares(alpha, N_SIMULATIONS, chaos_weight, use_pareto)
    
    # Apply elimination rules
    judge_shares = week_data['judge_share'].values
    judge_ranks = week_data['judge_rank'].values
    judge_scores_norm = week_data['normalized_score'].values
    
    if rule_system == 'Rank':
        eliminated_idx, _ = apply_rank_rule(judge_ranks, fan_shares)
    elif rule_system == 'Percent':
        eliminated_idx, _ = apply_percent_rule(judge_shares, fan_shares)
    else:  # Rank_With_Save
        eliminated_idx, _ = apply_rank_with_save_rule(judge_ranks, judge_scores_norm, fan_shares)
    
    # Update alpha for next week
    valid_shares = fan_shares.mean(axis=0)
    new_alpha = alpha + LEARNING_RATE * valid_shares * EVIDENCE_BOOST
    new_alpha = np.maximum(new_alpha, MIN_ALPHA)
    
    return eliminated_idx, new_alpha


def compute_historical_recall(df: pd.DataFrame, chaos_weight: float, 
                               n_runs: int = 100, use_pareto: bool = True) -> float:
    """
    Compute Historical Recall Rate: % of actual eliminations correctly predicted.
    
    For each (season, week), check if the model predicts the actual eliminated contestant.
    """
    np.random.seed(RANDOM_SEED)
    
    correct_predictions = 0
    total_eliminations = 0
    
    seasons = sorted(df['season'].unique())
    
    for season in seasons:
        season_data = df[df['season'] == season].copy()
        weeks = sorted(season_data['week'].unique())
        
        prev_alpha = None
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()
            week_data = week_data.sort_values('celebrity_name').reset_index(drop=True)
            
            # Skip non-elimination weeks
            if not week_data['is_eliminated'].any():
                prev_alpha = np.full(len(week_data), INITIAL_ALPHA) if prev_alpha is None else prev_alpha
                continue
            
            actual_eliminated_mask = week_data['is_eliminated'].values
            actual_eliminated_idx = np.where(actual_eliminated_mask)[0][0]
            
            # Run simulation
            if prev_alpha is None or len(prev_alpha) != len(week_data):
                prev_alpha = np.full(len(week_data), INITIAL_ALPHA)
            
            simulated_eliminated, new_alpha = simulate_week_with_lambda(
                week_data, chaos_weight, prev_alpha, use_pareto
            )
            
            # Check if majority of simulations predict actual elimination
            prediction_counts = np.bincount(simulated_eliminated, minlength=len(week_data))
            predicted_eliminated_idx = np.argmax(prediction_counts)
            
            if predicted_eliminated_idx == actual_eliminated_idx:
                correct_predictions += 1
            
            total_eliminations += 1
            
            # Update alpha for surviving contestants
            surviving_mask = ~actual_eliminated_mask
            if surviving_mask.sum() > 0:
                prev_alpha = new_alpha[surviving_mask]
            else:
                prev_alpha = None
    
    recall_rate = correct_predictions / total_eliminations if total_eliminations > 0 else 0
    return recall_rate


def optimize_lambda(df: pd.DataFrame, 
                    lambda_range: Tuple[float, float] = (0.0, 0.15),
                    lambda_step: float = 0.005,
                    n_runs: int = 100) -> Tuple[float, pd.DataFrame]:
    """
    Phase 3: Grid search to find optimal chaos weight λ.
    
    Returns: (optimal_lambda, results_df)
    """
    print("\n" + "="*70)
    print("   PHASE 3: Parameter Optimization (Grid Search)")
    print("="*70)
    
    lambda_values = np.arange(lambda_range[0], lambda_range[1] + lambda_step, lambda_step)
    results = []
    
    print(f"   Testing λ from {lambda_range[0]:.3f} to {lambda_range[1]:.3f} (step={lambda_step})")
    print(f"   Total configurations: {len(lambda_values)}")
    
    for lam in tqdm(lambda_values, desc="Grid Search"):
        recall = compute_historical_recall(df, lam, n_runs, use_pareto=True)
        results.append({'lambda': lam, 'recall_rate': recall})
    
    results_df = pd.DataFrame(results)
    
    # Find optimal lambda
    optimal_idx = results_df['recall_rate'].idxmax()
    optimal_lambda = results_df.loc[optimal_idx, 'lambda']
    optimal_recall = results_df.loc[optimal_idx, 'recall_rate']
    
    print(f"\n   ╔══════════════════════════════════════╗")
    print(f"   ║  OPTIMAL LAMBDA: {optimal_lambda:.3f}                 ║")
    print(f"   ║  Historical Recall: {optimal_recall:.1%}            ║")
    print(f"   ╚══════════════════════════════════════╝")
    
    # Generate optimization curve plot
    plot_optimization_curve(results_df, optimal_lambda, optimal_recall)
    
    return optimal_lambda, results_df


def plot_optimization_curve(results_df: pd.DataFrame, optimal_lambda: float, optimal_recall: float) -> None:
    """
    Generate Plot 2: Lambda Optimization Curve.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results_df['lambda'], results_df['recall_rate'], 
            'o-', linewidth=2, markersize=6, color='#3498db', label='Historical Recall Rate')
    
    # Mark optimal point
    ax.axvline(x=optimal_lambda, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
    ax.scatter([optimal_lambda], [optimal_recall], s=200, color='#e74c3c', zorder=5, 
               edgecolors='white', linewidth=2, label=f'Optimal λ = {optimal_lambda:.3f}')
    
    ax.set_xlabel('Chaos Weight (λ)', fontsize=12)
    ax.set_ylabel('Historical Recall Rate', fontsize=12)
    ax.set_title('Phase 3: Lambda Optimization Curve\nMaximizing Overlap with Actual Elimination History', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add annotation
    ax.annotate(f'Peak: λ={optimal_lambda:.3f}\nRecall={optimal_recall:.1%}',
                xy=(optimal_lambda, optimal_recall),
                xytext=(optimal_lambda + 0.02, optimal_recall - 0.05),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d'))
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / "lambda_optimization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {plot_path}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("="*70)
    print("   MCM 2026 Problem C: Three-Stage Bayesian Diagnostic Analysis")
    print("="*70)
    
    # Load data
    print(f"\n[INFO] Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"   Rows: {len(df)}, Seasons: {df['season'].nunique()}")
    
    # Phase 1: Pure Rationality Baseline
    df_diagnostics = run_diagnostics(df)
    
    # Phase 2: Distribution Visualization
    plot_skill_bias_scatter(df_diagnostics)
    
    # Phase 3: Lambda Optimization
    optimal_lambda, optimization_results = optimize_lambda(df)
    
    # Final Summary
    print("\n" + "="*70)
    print("   ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n   📊 Plots saved to: {PLOTS_DIR}")
    print(f"   📈 Optimal Chaos Weight (λ): {optimal_lambda:.3f}")
    print(f"\n   Use this value in monte_carlo_bayesian_dirichlet.py:")
    print(f"      CHAOS_FACTOR = {optimal_lambda:.3f}")
    print("="*70)
    
    return optimal_lambda, df_diagnostics, optimization_results


if __name__ == "__main__":
    main()
