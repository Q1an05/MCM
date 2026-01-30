#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Bayesian Monte Carlo Simulation - Enhanced Fan Vote Estimation

This script enhances the basic Monte Carlo simulation with Bayesian updating,
introducing temporal continuity in fan vote estimation.

Key Improvements over Basic Monte Carlo:
1. Temporal Momentum: Uses previous week's posterior as current week's prior
2. Uncertainty Propagation: Tracks Dirichlet α parameters across weeks
3. Evidence Accumulation: Confidence naturally increases over time
4. Soft Priors: Reduces "unexplainable" cases through flexible priors

Mathematical Framework:
- Prior:     V_t | α_{t-1} ~ Dirichlet(α_{t-1})
- Update:    α_t = (1-η) · α_{t-1} + η · Evidence_t
- Evidence:  Mean of valid simulations from filtering step

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
INPUT_PATH = Path(__file__).parent.parent / "data_processed" / "dwts_simulation_input.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "fan_vote_estimates_bayesian.csv"
COMPARISON_PATH = Path(__file__).parent.parent / "results" / "bayesian_vs_basic_comparison.csv"

# Simulation parameters
N_SIMULATIONS = 10000
INITIAL_ALPHA = 0.8      # Initial Dirichlet concentration (weak prior)
LEARNING_RATE = 0.4      # η: How much new evidence updates the prior (increased)
MIN_ALPHA = 0.1          # Minimum α to prevent degenerate distributions
EVIDENCE_BOOST = 5.0     # Scale factor for evidence to strengthen posterior

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class WeekState:
    """State information passed between weeks"""
    alpha: np.ndarray           # Dirichlet concentration parameters
    contestant_names: List[str] # Ordered list of contestant names
    
    @property
    def alpha_sum(self) -> float:
        """Total concentration (measure of confidence)"""
        return self.alpha.sum()
    
    @property
    def n_contestants(self) -> int:
        return len(self.contestant_names)


# =============================================================================
# Core Bayesian Functions
# =============================================================================
def initialize_prior(n_contestants: int, alpha_0: float = INITIAL_ALPHA) -> np.ndarray:
    """
    Initialize weak Dirichlet prior for first week of a season.
    
    Args:
        n_contestants: Number of contestants
        alpha_0: Initial concentration parameter
        
    Returns:
        Array of α parameters
    """
    return np.full(n_contestants, alpha_0)


def align_prior_to_current(
    prev_state: WeekState,
    current_names: List[str]
) -> np.ndarray:
    """
    Align previous week's posterior to current week's contestants.
    
    Handles dimension changes when contestants are eliminated.
    
    Args:
        prev_state: Previous week's state
        current_names: Current week's contestant names
        
    Returns:
        Aligned α vector for current week
    """
    aligned_alpha = []
    
    for name in current_names:
        if name in prev_state.contestant_names:
            idx = prev_state.contestant_names.index(name)
            aligned_alpha.append(prev_state.alpha[idx])
        else:
            # New contestant (shouldn't happen in DWTS, but handle gracefully)
            aligned_alpha.append(INITIAL_ALPHA)
    
    aligned_alpha = np.array(aligned_alpha)
    
    # Renormalize to maintain relative proportions
    # (Optional: could also just use raw values)
    total = aligned_alpha.sum()
    if total > 0:
        # Scale to maintain similar total concentration
        scale_factor = prev_state.alpha_sum / total if len(current_names) < prev_state.n_contestants else 1.0
        aligned_alpha = aligned_alpha * min(scale_factor, 1.5)  # Cap scaling
    
    return np.maximum(aligned_alpha, MIN_ALPHA)


def update_prior_with_evidence(
    current_alpha: np.ndarray,
    evidence: np.ndarray,
    learning_rate: float = LEARNING_RATE
) -> np.ndarray:
    """
    Bayesian update of Dirichlet parameters with observed evidence.
    
    Update rule: α_new = α_old + η · evidence_scaled
    
    This is an accumulation model - evidence ADDS to the prior,
    increasing confidence over time.
    
    Args:
        current_alpha: Current α parameters
        evidence: Observed fan share distribution (from valid simulations)
        learning_rate: η, weight given to new evidence
        
    Returns:
        Updated α parameters
    """
    # Scale evidence to be meaningful update
    # Use EVIDENCE_BOOST to control how much each week's evidence adds
    evidence_scaled = evidence * EVIDENCE_BOOST
    
    # Accumulation update: α_new = α_old + η * evidence
    # This naturally increases α over time (confidence grows)
    new_alpha = current_alpha + learning_rate * evidence_scaled
    
    # Ensure minimum values
    new_alpha = np.maximum(new_alpha, MIN_ALPHA)
    
    return new_alpha


# =============================================================================
# Simulation Functions (Reused from basic version with modifications)
# =============================================================================
def generate_fan_shares_from_prior(
    alpha: np.ndarray,
    n_sims: int = N_SIMULATIONS
) -> np.ndarray:
    """
    Generate fan share samples from Dirichlet distribution with given α.
    
    Key insight: Higher α values = more concentrated distribution around the mean.
    This means as we accumulate evidence, samples become more "focused" on 
    what we believe the true fan shares are.
    
    Args:
        alpha: Dirichlet concentration parameters
        n_sims: Number of simulation trials
        
    Returns:
        Array of shape (n_sims, n_contestants)
    """
    # Normalize alpha to control variance while preserving mean
    # Higher alpha_sum = lower variance = more confident samples
    return np.random.dirichlet(alpha, size=n_sims)


def shares_to_ranks(shares: np.ndarray) -> np.ndarray:
    """Convert share values to ranks (1 = highest share = best)."""
    order = np.argsort(-shares, axis=1)
    ranks = np.argsort(order, axis=1) + 1
    return ranks.astype(float)


def apply_rank_rule(judge_ranks: np.ndarray, fan_shares: np.ndarray) -> np.ndarray:
    """Apply RANK elimination rule (Seasons 1-2)."""
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks
    tie_breaker = fan_ranks * 0.001
    composite_score = total_scores + tie_breaker
    return np.argmax(composite_score, axis=1)


def apply_percent_rule(judge_shares: np.ndarray, fan_shares: np.ndarray) -> np.ndarray:
    """Apply PERCENT elimination rule (Seasons 3-27)."""
    total_scores = judge_shares + fan_shares
    return np.argmin(total_scores, axis=1)


def apply_rank_with_save_rule(
    judge_ranks: np.ndarray,
    judge_scores: np.ndarray,
    fan_shares: np.ndarray
) -> np.ndarray:
    """Apply RANK_WITH_SAVE elimination rule (Seasons 28+)."""
    n_sims, n_contestants = fan_shares.shape
    
    if n_contestants < 2:
        return np.zeros(n_sims, dtype=int)
    
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks
    noise = fan_ranks * 1e-6
    total_scores_noisy = total_scores + noise
    
    if n_contestants == 2:
        if judge_scores[0] < judge_scores[1]:
            return np.zeros(n_sims, dtype=int)
        elif judge_scores[1] < judge_scores[0]:
            return np.ones(n_sims, dtype=int)
        else:
            return np.argmax(fan_ranks, axis=1)
    
    sorted_indices = np.argsort(-total_scores_noisy, axis=1)
    bottom2_first = sorted_indices[:, 0]
    bottom2_second = sorted_indices[:, 1]
    
    judge_scores_first = judge_scores[bottom2_first]
    judge_scores_second = judge_scores[bottom2_second]
    
    eliminated_idx = np.where(
        judge_scores_first < judge_scores_second,
        bottom2_first,
        np.where(
            judge_scores_second < judge_scores_first,
            bottom2_second,
            bottom2_first
        )
    )
    
    return eliminated_idx


# =============================================================================
# Bayesian Simulation Engine
# =============================================================================
def simulate_week_bayesian(
    week_data: pd.DataFrame,
    prior_alpha: np.ndarray,
    n_sims: int = N_SIMULATIONS
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Run Bayesian Monte Carlo simulation for a single week.
    
    Args:
        week_data: DataFrame with all contestants for one (season, week)
        prior_alpha: Dirichlet α parameters (prior from previous week)
        n_sims: Number of simulation trials
        
    Returns:
        Tuple of (result_df, posterior_alpha, metrics_dict)
    """
    season = week_data['season'].iloc[0]
    week = week_data['week'].iloc[0]
    rule_system = week_data['rule_system'].iloc[0]
    
    contestants = week_data['celebrity_name'].values
    n_contestants = len(contestants)
    
    # Get actual elimination info
    actual_eliminated_mask = week_data['is_eliminated'].values
    has_elimination = actual_eliminated_mask.any()
    
    # Extract judge data
    judge_shares = week_data['judge_share'].values.astype(float)
    judge_ranks = week_data['judge_rank'].values.astype(float)
    judge_scores = week_data['normalized_score'].values.astype(float)
    
    # Record prior strength
    prior_strength = prior_alpha.sum()
    
    # Generate fan share samples from current prior
    fan_shares = generate_fan_shares_from_prior(prior_alpha, n_sims)
    
    # Metrics to track
    metrics = {
        'prior_alpha_sum': prior_strength,
        'prior_alpha_mean': prior_alpha.mean(),
    }
    
    if not has_elimination:
        # No elimination this week - return prior mean, no update
        prior_mean = prior_alpha / prior_alpha.sum()
        
        result = week_data[['season', 'week', 'celebrity_name']].copy()
        result['estimated_fan_share'] = prior_mean
        result['share_std'] = np.nan
        result['confidence'] = np.nan
        result['n_valid_sims'] = 0
        result['prior_strength'] = prior_strength
        result['posterior_strength'] = prior_strength  # No update
        
        metrics['n_valid'] = 0
        metrics['confidence'] = np.nan
        
        return result, prior_alpha, metrics  # Return unchanged prior
    
    # Find actual eliminated contestant
    actual_eliminated_idx = np.where(actual_eliminated_mask)[0][0]
    
    # Apply elimination rule
    if rule_system == "Rank":
        simulated_eliminated = apply_rank_rule(judge_ranks, fan_shares)
    elif rule_system == "Percent":
        simulated_eliminated = apply_percent_rule(judge_shares, fan_shares)
    elif rule_system == "Rank_With_Save":
        simulated_eliminated = apply_rank_with_save_rule(
            judge_ranks, judge_scores, fan_shares
        )
    else:
        simulated_eliminated = apply_percent_rule(judge_shares, fan_shares)
    
    # Filter: Keep simulations that match actual outcome
    valid_mask = (simulated_eliminated == actual_eliminated_idx)
    n_valid = valid_mask.sum()
    confidence = n_valid / n_sims
    
    metrics['n_valid'] = n_valid
    metrics['confidence'] = confidence
    
    if n_valid == 0:
        # No valid simulations - use prior mean, but still update slightly
        # This "softens" the prior for future weeks
        prior_mean = prior_alpha / prior_alpha.sum()
        
        result = week_data[['season', 'week', 'celebrity_name']].copy()
        result['estimated_fan_share'] = prior_mean
        result['share_std'] = np.nan
        result['confidence'] = 0.0
        result['n_valid_sims'] = 0
        result['prior_strength'] = prior_strength
        result['posterior_strength'] = prior_strength * 0.9  # Slight decay
        
        # Decay prior slightly when we can't explain the outcome
        posterior_alpha = prior_alpha * 0.9
        posterior_alpha = np.maximum(posterior_alpha, MIN_ALPHA)
        
        return result, posterior_alpha, metrics
    
    # Aggregate valid simulations
    valid_fan_shares = fan_shares[valid_mask]
    estimated_shares = valid_fan_shares.mean(axis=0)
    share_stds = valid_fan_shares.std(axis=0)
    
    # Bayesian update: use evidence to update prior
    posterior_alpha = update_prior_with_evidence(
        prior_alpha, 
        estimated_shares,
        learning_rate=LEARNING_RATE
    )
    
    posterior_strength = posterior_alpha.sum()
    metrics['posterior_alpha_sum'] = posterior_strength
    
    # Build result DataFrame
    result = week_data[['season', 'week', 'celebrity_name']].copy()
    result['estimated_fan_share'] = estimated_shares
    result['share_std'] = share_stds
    result['confidence'] = confidence
    result['n_valid_sims'] = n_valid
    result['prior_strength'] = prior_strength
    result['posterior_strength'] = posterior_strength
    
    return result, posterior_alpha, metrics


def run_bayesian_simulation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Bayesian Monte Carlo simulation for all seasons.
    
    Key difference from basic: Process seasons sequentially,
    passing posterior from week t as prior for week t+1.
    
    Args:
        df: Input DataFrame from data processing
        
    Returns:
        Tuple of (results_df, metrics_df)
    """
    print(f"\n{'='*60}")
    print("BAYESIAN MONTE CARLO SIMULATION")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - N_SIMULATIONS: {N_SIMULATIONS:,}")
    print(f"  - INITIAL_ALPHA: {INITIAL_ALPHA}")
    print(f"  - LEARNING_RATE: {LEARNING_RATE}")
    print(f"  - Random Seed: {RANDOM_SEED}")
    
    np.random.seed(RANDOM_SEED)
    
    results = []
    all_metrics = []
    
    seasons = sorted(df['season'].unique())
    
    for season in tqdm(seasons, desc="Processing Seasons"):
        season_data = df[df['season'] == season].copy()
        weeks = sorted(season_data['week'].unique())
        
        # Initialize state for this season
        current_state = None
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week].reset_index(drop=True)
            current_names = week_data['celebrity_name'].tolist()
            n_contestants = len(current_names)
            
            # Determine prior for this week
            if current_state is None:
                # First week: use uninformative prior
                prior_alpha = initialize_prior(n_contestants)
            else:
                # Subsequent weeks: align previous posterior
                prior_alpha = align_prior_to_current(current_state, current_names)
            
            # Run simulation
            week_result, posterior_alpha, metrics = simulate_week_bayesian(
                week_data, prior_alpha
            )
            
            results.append(week_result)
            
            # Record metrics
            metrics['season'] = season
            metrics['week'] = week
            metrics['n_contestants'] = n_contestants
            all_metrics.append(metrics)
            
            # Update state for next week
            current_state = WeekState(
                alpha=posterior_alpha,
                contestant_names=current_names
            )
    
    results_df = pd.concat(results, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    
    return results_df, metrics_df


# =============================================================================
# Comparison with Basic Monte Carlo
# =============================================================================
def load_basic_results() -> Optional[pd.DataFrame]:
    """Load results from basic Monte Carlo simulation."""
    basic_path = Path(__file__).parent.parent / "results" / "fan_vote_estimates.csv"
    if basic_path.exists():
        return pd.read_csv(basic_path)
    return None


def compare_methods(bayesian_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Bayesian vs Basic Monte Carlo results.
    
    Args:
        bayesian_df: Results from Bayesian simulation
        basic_df: Results from basic simulation
        
    Returns:
        Comparison DataFrame
    """
    print(f"\n{'-'*60}")
    print("COMPARISON: Bayesian vs Basic Monte Carlo")
    print(f"{'-'*60}")
    
    # Merge on key columns
    merged = pd.merge(
        basic_df[['season', 'week', 'celebrity_name', 'estimated_fan_share', 
                  'confidence', 'n_valid_sims']],
        bayesian_df[['season', 'week', 'celebrity_name', 'estimated_fan_share',
                     'confidence', 'n_valid_sims', 'prior_strength', 'posterior_strength']],
        on=['season', 'week', 'celebrity_name'],
        suffixes=('_basic', '_bayesian')
    )
    
    # Calculate differences
    merged['fan_share_diff'] = merged['estimated_fan_share_bayesian'] - merged['estimated_fan_share_basic']
    merged['confidence_diff'] = merged['confidence_bayesian'] - merged['confidence_basic']
    
    # Summary statistics
    valid_basic = basic_df[basic_df['confidence'].notna() & (basic_df['confidence'] > 0)]
    valid_bayesian = bayesian_df[bayesian_df['confidence'].notna() & (bayesian_df['confidence'] > 0)]
    
    print(f"\n📊 Overall Statistics:")
    print(f"{'Metric':<30} {'Basic':>15} {'Bayesian':>15} {'Δ':>10}")
    print("-" * 70)
    
    basic_conf = valid_basic['confidence'].mean()
    bayes_conf = valid_bayesian['confidence'].mean()
    print(f"{'Mean Confidence':<30} {basic_conf:>15.4f} {bayes_conf:>15.4f} {bayes_conf-basic_conf:>+10.4f}")
    
    basic_median = valid_basic['confidence'].median()
    bayes_median = valid_bayesian['confidence'].median()
    print(f"{'Median Confidence':<30} {basic_median:>15.4f} {bayes_median:>15.4f} {bayes_median-basic_median:>+10.4f}")
    
    basic_zero = (basic_df['n_valid_sims'] == 0).sum()
    bayes_zero = (bayesian_df['n_valid_sims'] == 0).sum()
    # Exclude non-elimination weeks for fair comparison
    basic_zero_elim = ((basic_df['n_valid_sims'] == 0) & (basic_df['confidence'].notna())).sum()
    bayes_zero_elim = ((bayesian_df['n_valid_sims'] == 0) & (bayesian_df['confidence'].notna())).sum()
    print(f"{'Zero-Valid Cases (w/ elim)':<30} {basic_zero_elim:>15} {bayes_zero_elim:>15} {bayes_zero_elim-basic_zero_elim:>+10}")
    
    # Confidence by week (to show temporal improvement)
    print(f"\n📈 Confidence by Week (showing temporal improvement):")
    print(f"{'Week':<10} {'Basic':>15} {'Bayesian':>15} {'Improvement':>15}")
    print("-" * 55)
    
    for week in sorted(merged['week'].unique())[:8]:  # First 8 weeks
        week_data = merged[merged['week'] == week]
        basic_week = week_data['confidence_basic'].mean()
        bayes_week = week_data['confidence_bayesian'].mean()
        if pd.notna(basic_week) and pd.notna(bayes_week):
            pct_improve = ((bayes_week - basic_week) / basic_week * 100) if basic_week > 0 else 0
            print(f"{week:<10} {basic_week:>15.4f} {bayes_week:>15.4f} {pct_improve:>+14.1f}%")
    
    return merged


# =============================================================================
# Analysis & Output
# =============================================================================
def analyze_bayesian_results(df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Analyze and print Bayesian simulation results."""
    print(f"\n{'-'*60}")
    print("BAYESIAN SIMULATION ANALYSIS")
    print(f"{'-'*60}")
    
    print(f"\nTotal rows: {len(df)}")
    
    # Confidence analysis
    valid = df[df['confidence'].notna() & (df['confidence'] > 0)]
    if len(valid) > 0:
        print(f"\n📊 Confidence Statistics:")
        print(f"  - Mean: {valid['confidence'].mean():.4f}")
        print(f"  - Median: {valid['confidence'].median():.4f}")
        print(f"  - Std: {valid['confidence'].std():.4f}")
        print(f"  - Min: {valid['confidence'].min():.4f}")
        print(f"  - Max: {valid['confidence'].max():.4f}")
    
    # Prior/Posterior strength evolution
    if 'posterior_strength' in df.columns:
        print(f"\n📈 Prior Strength Evolution (α_sum):")
        strength_by_week = df.groupby('week')['posterior_strength'].mean()
        for week in [1, 3, 5, 7, 9]:
            if week in strength_by_week.index:
                print(f"  - Week {week}: {strength_by_week[week]:.2f}")
    
    # Zero-valid cases
    zero_valid = df[(df['n_valid_sims'] == 0) & (df['confidence'].notna())]
    if len(zero_valid) > 0:
        print(f"\n⚠️  Unexplainable cases: {len(zero_valid)} contestant-weeks")
        cases = zero_valid.groupby(['season', 'week']).first()
        for (s, w), _ in cases.head(3).iterrows():
            print(f"    - Season {s}, Week {w}")


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save Bayesian simulation results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_cols = [
        'season', 'week', 'celebrity_name',
        'estimated_fan_share', 'share_std', 'confidence', 'n_valid_sims',
        'prior_strength', 'posterior_strength'
    ]
    
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    output_df = df[output_cols].copy()
    
    # Round for readability
    for col in ['estimated_fan_share', 'share_std', 'confidence']:
        if col in output_df.columns:
            output_df[col] = output_df[col].round(6)
    
    output_df.to_csv(output_path, index=False)
    print(f"\n✓ Bayesian results saved to: {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    """Main Bayesian Monte Carlo simulation pipeline."""
    print("="*60)
    print("MCM 2026 Problem C - Bayesian Monte Carlo Simulation")
    print("="*60)
    print("\n🎯 Optimization: Temporal Bayesian Updating")
    print("   - Prior from week t-1 informs week t")
    print("   - Confidence accumulates over time")
    print("   - Soft priors reduce unexplainable cases")
    
    # Load data
    print(f"\n[INFO] Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded {len(df)} records")
    
    # Run Bayesian simulation
    bayesian_results, metrics_df = run_bayesian_simulation(df)
    
    # Analyze results
    analyze_bayesian_results(bayesian_results, metrics_df)
    
    # Compare with basic Monte Carlo
    basic_results = load_basic_results()
    if basic_results is not None:
        comparison_df = compare_methods(bayesian_results, basic_results)
        comparison_df.to_csv(COMPARISON_PATH, index=False)
        print(f"\n✓ Comparison saved to: {COMPARISON_PATH}")
    else:
        print("\n[INFO] Basic results not found, skipping comparison")
    
    # Save results
    save_results(bayesian_results, OUTPUT_PATH)
    
    print(f"\n{'='*60}")
    print("Bayesian Simulation Complete!")
    print("="*60)
    
    return bayesian_results, metrics_df


if __name__ == "__main__":
    main()
