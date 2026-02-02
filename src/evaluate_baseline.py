#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Baseline Evaluation Script

Calculates Upset Rate and Kendall's Tau for the HISTORICAL rule systems
to provide a benchmark for the New System Design.

Author: MCM Team
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "results" / "full_simulation_bayesian.csv"
OUTPUT_PATH = BASE_DIR / "results" / "system_design" / "baseline_metrics.csv"

# =============================================================================
# Helper Functions
# =============================================================================
def calculate_historical_metrics(df):
    """
    Reconstruct historical scoring and measure fairness/upset rate.
    """
    upset_count = 0
    total_eliminations = 0
    kendall_taus = []
    
    # Iterate through season-weeks
    for (season, week), group in df.groupby(['season', 'week']):
        if len(group) < 2: continue
        
        # 0. Identify Rule System
        # Assuming rule is consistent within week
        rule = group['rule_system'].iloc[0]
        
        # 1. Judge Data
        s_judge = group['raw_score_sum']
        
        # 2. Reconstruct Fan Data (from simulation mean)
        # Note: In reality, fan votes were secret, but the elimination was based on them.
        # We use our 'estimated_fan_share' which generates the result closest to reality.
        p_fan = group['estimated_fan_share']
        
        # 3. Reconstruct Historical Score
        if rule == 'Rank':
            # Rank Rule: Lower is better
            # Judge Rank (High score = Low Rank 1)
            r_judge = s_judge.rank(ascending=False, method='min')
            # Fan Rank (High share = Low Rank 1)
            r_fan = p_fan.rank(ascending=False, method='min')
            
            total_score = r_judge + r_fan
            # For rank rule, lowest val is best. 
            # To make it comparable for Tau (where higher is better), we invert it?
            # Or just correlate Judge Score vs Total Rank (inverted).
            system_metric = -total_score # Higher is better
            
        elif rule == 'Percent':
            # Percent Rule: Higher is better
            if s_judge.sum() == 0:
                p_judge = s_judge * 0
            else:
                p_judge = s_judge / s_judge.sum() * 100
            
            p_fan_pct = p_fan * 100
            total_score = p_judge + p_fan_pct
            system_metric = total_score
            
        elif rule == 'RelVote': # Judges Save era (Rank based)
             # S28+ uses Rank + Judges Save.
             # Underlying score is Rank.
             r_judge = s_judge.rank(ascending=False, method='min')
             r_fan = p_fan.rank(ascending=False, method='min')
             total_score = r_judge + r_fan
             system_metric = -total_score
        else:
             continue

        # 4. Identify Actual Elimination (using recorded exit)
        # In history, who actually went home?
        # The 'is_eliminated' column marks this.
        eliminated_mask = group['is_eliminated']
        if eliminated_mask.sum() == 0:
            # Maybe a non-elimination week, skip counts
            pass
        else:
            eliminated_names = group[eliminated_mask]['celebrity_name'].tolist()
            
            # 5. Judge Bottom Set (Bt)
            min_judge_raw = s_judge.min()
            bottom_set = group[group['raw_score_sum'] <= min_judge_raw + 1e-9]['celebrity_name'].tolist()
            
            # Check for Upset (Any eliminated person NOT in bottom set)
            # If multiple people eliminated, if ANY is outside bottom set -> Upset?
            # Or if ALL are outside? Usually 1 person.
            # Definition: "If the eliminated contestant Et is NOT in Bt"
            for name in eliminated_names:
                total_eliminations += 1
                if name not in bottom_set:
                    upset_count += 1
        
        # 6. Kendall's Tau
        # Correlation between Judge Raw Score and System Metric (Result)
        # Both should be "Higher is Better"
        tau, _ = stats.kendalltau(s_judge, system_metric)
        if not np.isnan(tau):
            kendall_taus.append(tau)

    ru = upset_count / total_eliminations if total_eliminations > 0 else 0.0
    avg_tau = np.mean(kendall_taus) if kendall_taus else 0.0
    
    return ru, avg_tau

def main():
    print("[INFO] calculating Baseline Metrics...")
    df = pd.read_csv(INPUT_PATH)
    ru, tau = calculate_historical_metrics(df)
    
    print("="*40)
    print(" BASELINE (HISTORICAL) METRICS")
    print("="*40)
    print(f"Historical Upset Rate: {ru:.2%}")
    print(f"Historical Fairness:   {tau:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
