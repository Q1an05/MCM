#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Create Full Bayesian Simulation Dataset

Merges the original processed data with Bayesian Monte Carlo results
to create a complete dataset for analysis.

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
SIM_INPUT_PATH = BASE_DIR / "data_processed" / "dwts_simulation_input.csv"
BAYESIAN_RESULTS_PATH = BASE_DIR / "results" / "fan_vote_estimates_bayesian.csv"
OUTPUT_PATH = BASE_DIR / "data_processed" / "dwts_full_simulation_bayesian.csv"


def main():
    """Create the complete Bayesian simulation dataset."""
    print("="*60)
    print("Creating Full Bayesian Simulation Dataset")
    print("="*60)
    
    # Load data
    print(f"\n[INFO] Loading simulation input: {SIM_INPUT_PATH}")
    sim_input = pd.read_csv(SIM_INPUT_PATH)
    print(f"       Rows: {len(sim_input)}")
    
    print(f"[INFO] Loading Bayesian results: {BAYESIAN_RESULTS_PATH}")
    bayesian_results = pd.read_csv(BAYESIAN_RESULTS_PATH)
    print(f"       Rows: {len(bayesian_results)}")
    
    # Merge on key columns
    merge_keys = ['season', 'week', 'celebrity_name']
    
    master_df = pd.merge(
        sim_input,
        bayesian_results[['season', 'week', 'celebrity_name', 
                          'estimated_fan_share', 'share_std', 'confidence', 
                          'n_valid_sims', 'prior_strength', 'posterior_strength']],
        on=merge_keys,
        how='left'
    )
    
    print(f"\n[INFO] Merged dataset rows: {len(master_df)}")
    
    # Fill NaN fan shares with uniform prior (for non-elimination weeks)
    nan_mask = master_df['estimated_fan_share'].isna()
    if nan_mask.sum() > 0:
        print(f"[INFO] Filling {nan_mask.sum()} NaN fan shares with uniform prior...")
        master_df.loc[nan_mask, 'estimated_fan_share'] = master_df.loc[nan_mask].apply(
            lambda row: 1.0 / row['n_contestants'], axis=1
        )
    
    # Calculate derived columns
    # 1. Fan-Judge gap (positive = fan favorite, negative = judge favorite)
    master_df['fan_judge_gap'] = master_df['estimated_fan_share'] - master_df['judge_share']
    
    # 2. Evidence ratio (posterior/prior strength - shows how much this week updated beliefs)
    master_df['evidence_ratio'] = master_df['posterior_strength'] / master_df['prior_strength']
    master_df['evidence_ratio'] = master_df['evidence_ratio'].fillna(1.0)
    
    # 3. Normalized confidence (relative to week's average)
    week_avg_conf = master_df.groupby(['season', 'week'])['confidence'].transform('mean')
    master_df['relative_confidence'] = master_df['confidence'] / week_avg_conf
    master_df['relative_confidence'] = master_df['relative_confidence'].fillna(1.0)
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n{'='*60}")
    print("OUTPUT SUMMARY")
    print(f"{'='*60}")
    print(f"\n✓ Saved to: {OUTPUT_PATH}")
    print(f"  Total rows: {len(master_df)}")
    print(f"  Total columns: {len(master_df.columns)}")
    
    # Print column list
    print(f"\nColumn List ({len(master_df.columns)} columns):")
    print("-" * 40)
    
    categories = {
        "Identifiers": ['season', 'week', 'celebrity_name'],
        "Contestant Info": ['exit_week', 'result_type', 'placement'],
        "Judge Scores": ['n_judges', 'raw_score_sum', 'max_possible_points', 
                         'normalized_score', 'judge_share', 'judge_rank'],
        "Competition State": ['n_contestants', 'is_eliminated', 'rule_system'],
        "Bayesian Estimates": ['estimated_fan_share', 'share_std', 'confidence', 
                               'n_valid_sims', 'prior_strength', 'posterior_strength'],
        "Derived Metrics": ['fan_judge_gap', 'evidence_ratio', 'relative_confidence']
    }
    
    for category, cols in categories.items():
        existing_cols = [c for c in cols if c in master_df.columns]
        if existing_cols:
            print(f"\n  {category}:")
            for col in existing_cols:
                dtype = master_df[col].dtype
                print(f"    - {col} ({dtype})")
    
    # Data quality check
    print(f"\n{'='*60}")
    print("DATA QUALITY CHECK")
    print(f"{'='*60}")
    
    print(f"\nSeasons: {master_df['season'].min()} - {master_df['season'].max()}")
    print(f"Unique contestants: {master_df['celebrity_name'].nunique()}")
    
    # Confidence stats
    valid_conf = master_df[master_df['confidence'].notna() & (master_df['confidence'] > 0)]
    print(f"\nConfidence (valid rows only, n={len(valid_conf)}):")
    print(f"  Mean: {valid_conf['confidence'].mean():.4f}")
    print(f"  Median: {valid_conf['confidence'].median():.4f}")
    
    # Prior strength evolution
    print(f"\nPrior Strength by Week (α_sum):")
    strength_by_week = master_df.groupby('week')['prior_strength'].mean()
    for week in [1, 3, 5, 7, 9]:
        if week in strength_by_week.index:
            print(f"  Week {week}: {strength_by_week[week]:.2f}")
    
    # Sample data
    print(f"\n{'='*60}")
    print("SAMPLE DATA (First 5 rows)")
    print(f"{'='*60}")
    
    display_cols = ['season', 'week', 'celebrity_name', 'judge_share', 
                    'estimated_fan_share', 'fan_judge_gap', 'confidence',
                    'prior_strength', 'posterior_strength']
    print(master_df[display_cols].head().to_string(index=False))
    
    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    
    return master_df


if __name__ == "__main__":
    main()
