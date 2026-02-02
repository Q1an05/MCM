#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Task 4: System Design - Dynamic-Threshold Percent Model (DTPM)

This script implements the optimization for the new scoring system:
1. Dynamic Weighting (Linearly decreasing judge weight)
2. Performance Gated Multiplier (Fan vote penalty for low judge scores)
3. Optimization via Pareto Frontier (Fairness vs. Entertainment)

Author: MCM Team
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "results" / "full_simulation_bayesian.csv"
OUTPUT_DIR = BASE_DIR / "results" / "system_design"
PLOT_DIR = BASE_DIR / "results" / "plots" / "system_design"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Grid Search Ranges
W_START_RANGE = np.arange(0.5, 0.95, 0.05)
W_END_RANGE = np.arange(0.3, 0.65, 0.05)
BETA_RANGE = np.arange(0.4, 1.05, 0.1)

# Target Ideal Upset Rate (The "Golden Mean")
TARGET_UPSET_RATE = 0.15

# =============================================================================
# Core Logic: Dynamic-Threshold Percent Model
# =============================================================================
class DTPM_Evaluator:
    def __init__(self, df):
        self.df = df.copy()
        # Pre-calculate weeks per season for dynamic weighting
        self.season_max_weeks = self.df.groupby('season')['week'].max().to_dict()
        
        # Filter for weeks where eliminations theoretically happen (rank <= n_contestants)
        # We only really care about weeks where we have scores.
        
    def calculate_metrics(self, w_start, w_end, beta):
        """
        Calculate Kendall's Tau (Fairness) and Upset Rate (Entertainment)
        for a specific set of parameters.
        """
        upset_count = 0
        total_eliminations = 0
        kendall_taus = []
        
        # Iterate through each season-week
        # Using groupby is safer to handle isolated contexts
        for (season, week), group in self.df.groupby(['season', 'week']):
            # Skip if only 1 contestant (Finale winner calculation not needed for elim)
            if len(group) < 2:
                continue
                
            # 1. Setup Data
            # Raw Judge Score
            s_judge = group['raw_score_sum']
            # Simulated Fan Share (Use the estimated mean)
            p_fan = group['estimated_fan_share']
            
            # Normalize Judge Scores to Percentage
            if s_judge.sum() == 0:
                p_judge = s_judge * 0 # Handle zero sum
            else:
                p_judge = s_judge / s_judge.sum()
            
            # 2. Dynamic Weight w(t)
            # T = Max weeks in this season
            T = self.season_max_weeks.get(season, 10)
            if T <= 1: T = 10 # Fallback
            
            # Linear decay
            # week is 1-based index
            progress = np.clip((week - 1) / (T - 1), 0, 1)
            w_t = w_start - (w_start - w_end) * progress
            
            # 3. Performance Gated Multiplier (Beta)
            mean_judge = s_judge.mean()
            # Vectorized gamma calculation
            # If score < mean, multiplier = beta, else 1.0
            gamma = np.where(s_judge < mean_judge, beta, 1.0)
            
            # 4. Total Score Calculation
            # Formula: w_t * P_judge + (1 - w_t) * (P_fan * gamma)
            # Note: We don't re-normalize fan votes after gamma, to act as a penalty
            scores_new = w_t * p_judge + (1 - w_t) * (p_fan * gamma)
            
            # 5. Determine Hypothetical Elimination (Lowest Score)
            # Find the index of the minimum score
            eliminated_idx = scores_new.idxmin()
            eliminated_contestant = group.loc[eliminated_idx, 'celebrity_name']
            
            # 6. Judge's Bottom Set (Bt)
            min_judge_raw = s_judge.min()
            # Allow for small floating point diffs if any, but raw scores are ints usually
            bottom_set = group[group['raw_score_sum'] <= min_judge_raw + 1e-9]['celebrity_name'].values
            
            # 7. Check for Upset
            # Upset = The eliminated person was NOT in the judge's bottom set
            is_upset = eliminated_contestant not in bottom_set
            
            if is_upset:
                upset_count += 1
            total_eliminations += 1
            
            # 8. Kendall's Tau (Fairness)
            # Correlation between Judge Raw Score and New System Score
            # Higher correlation = New system respects judge ranking more
            tau, _ = stats.kendalltau(s_judge, scores_new)
            if not np.isnan(tau):
                kendall_taus.append(tau)
        
        # Aggregated Metrics
        ru = upset_count / total_eliminations if total_eliminations > 0 else 0.0
        avg_tau = np.mean(kendall_taus) if kendall_taus else 0.0
        
        return {
            'w_start': w_start,
            'w_end': w_end,
            'beta': beta,
            'upset_rate': ru,
            'kendall_tau': avg_tau
        }

# =============================================================================
# Optimization & Plotting
# =============================================================================
def find_pareto_frontier(results_df):
    """
    Find points on the Pareto Frontier.
    Objectives: 
    1. Minimize |Upset Rate - Target| (X-axis in plot logic) -> Closer to target is better
    2. Maximize Kendall's Tau (Y-axis) -> Higher is better
    
    We define 'Cost' as distance from target upset rate.
    We define 'Value' as Kendall's Tau.
    A point P(cost, val) dominates Q(cost, val) if P.cost <= Q.cost AND P.val >= Q.val
    """
    
    # Calculate objectives
    results_df['upset_diff'] = np.abs(results_df['upset_rate'] - TARGET_UPSET_RATE)
    
    pareto_indices = []
    
    # Sort by upset_diff asc, then tau desc to optimization
    sorted_df = results_df.sort_values(by=['upset_diff', 'kendall_tau'], ascending=[True, False])
    
    current_max_tau = -np.inf
    
    # We scan. Since we sorted by cost (ascending), we just need to keep track of the best Value seen so far.
    # Actually, standard Pareto: For a given cost, we want max value.
    # But strictly, Pareto frontier implies trade-off.
    # As cost increases (getting further from ideal upset rate), does Tau increase?
    # Wait, usually Tau (Fairness) contradicts Upset Rate (Entertainment).
    # High Upset Rate -> Low Tau. Low Upset Rate -> High Tau.
    # Our target is Upset Rate = 0.15.
    # If Upset Rate = 0.0 (Pure Judge), Tau is 1.0. Distance is 0.15.
    # If Upset Rate = 0.15, Tau might be 0.7. Distance is 0.0.
    # So yes, there is a trade-off between "Hitting the Target Upset Rate" and "Maximizing Tau"?
    # Actually, not necessarily. 0.15 is *less* fair than 0.0.
    # So maximizing Tau pushes us towards Upset Rate 0.0.
    # Minimizing distance pushes us towards Upset Rate 0.15.
    # These are conflicting objectives.
    
    # Let's extract points that are non-dominated.
    # Candidate Set
    candidates = sorted_df[['upset_diff', 'kendall_tau']].drop_duplicates()
    
    pareto_mask = []
    for idx, row in results_df.iterrows():
        # Is there any other point that has SMALLER OR EQUAL diff AND LARGER OR EQUAL tau?
        # And strictly better in at least one?
        this_diff = row['upset_diff']
        this_tau = row['kendall_tau']
        
        # Vectorized check
        dominated = ((results_df['upset_diff'] <= this_diff) & 
                     (results_df['kendall_tau'] >= this_tau) & 
                     ((results_df['upset_diff'] < this_diff) | (results_df['kendall_tau'] > this_tau))).any()
        
        pareto_mask.append(not dominated)
        
    return results_df[pareto_mask]

def plot_pareto(all_results, pareto_results):
    plt.figure(figsize=(12, 8))
    
    # Scatter all points
    plt.scatter(all_results['upset_rate'], all_results['kendall_tau'], 
                c='lightgray', alpha=0.5, label='Simulation Scenarios')
    
    # Highlight Pareto Frontier
    # We plot Upset Rate on X, Tau on Y to show the curve naturally
    pareto_sorted = pareto_results.sort_values(by='upset_rate')
    plt.plot(pareto_sorted['upset_rate'], pareto_sorted['kendall_tau'], 
             c='red', linewidth=2, linestyle='--', label='Pareto Frontier')
    plt.scatter(pareto_sorted['upset_rate'], pareto_sorted['kendall_tau'], 
                c='red', s=50, zorder=5)
    
    # Highlight Golden Zone
    plt.axvspan(0.10, 0.20, color='gold', alpha=0.2, label='Golden Zone (10%-20% Upset)')
    
    # Find Best Point (Closest to 0.15 within Frontier)
    best_point = pareto_results.loc[pareto_results['upset_diff'].idxmin()]
    plt.scatter(best_point['upset_rate'], best_point['kendall_tau'], 
                c='blue', s=150, marker='*', zorder=10, 
                label=f'Optimal: Beta={best_point["beta"]:.1f}, W_s={best_point["w_start"]:.2f}')
    
    plt.xlabel('Upset Rate (Entertainment Factor)')
    plt.ylabel("Kendall's Tau (Fairness Factor)")
    plt.title('System Design: Trade-off between Fairness and Entertainment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = PLOT_DIR / "system_optimization_pareto.png"
    plt.savefig(plot_path)
    print(f"[INFO] Plot saved to {plot_path}")

# =============================================================================
# Main
# =============================================================================
def main():
    print("[INFO] Loading simulation data...")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded {len(df)} records.")
    
    evaluator = DTPM_Evaluator(df)
    
    print("[INFO] Starting Grid Search...")
    results = []
    
    # Grid Search
    total_iterations = len(W_START_RANGE) * len(W_END_RANGE) * len(BETA_RANGE)
    with tqdm(total=total_iterations) as pbar:
        for w_s in W_START_RANGE:
            for w_e in W_END_RANGE:
                if w_e > w_s: continue # Constraint: Start weight >= End weight
                
                for beta in BETA_RANGE:
                    res = evaluator.calculate_metrics(w_s, w_e, beta)
                    results.append(res)
                    pbar.update(1)
    
    results_df = pd.DataFrame(results)
    
    # Save raw results
    results_df.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)
    
    # Analysis
    print("[INFO] Calculating Pareto Frontier...")
    pareto_df = find_pareto_frontier(results_df)
    pareto_df.to_csv(OUTPUT_DIR / "pareto_frontier_points.csv", index=False)
    
    # Find Recommended Parameter Set
    # Filter for Golden Interval first
    golden_df = results_df[(results_df['upset_rate'] >= 0.10) & (results_df['upset_rate'] <= 0.20)]
    if not golden_df.empty:
        # Maximize Fairness within Golden Zone
        best_scenario = golden_df.loc[golden_df['kendall_tau'].idxmax()]
    else:
        # Fallback to closest to target upset
        best_scenario = results_df.loc[results_df['upset_diff'].idxmin()]
        
    print("\n" + "="*50)
    print(" RECOMMENDED SYSTEM CONFIGURATION (DTPM)")
    print("="*50)
    print(f"Start Weight (Judge): {best_scenario['w_start']:.2f}")
    print(f"End Weight (Judge):   {best_scenario['w_end']:.2f}")
    print(f"Penalty Beta:         {best_scenario['beta']:.2f}")
    print("-" * 30)
    print(f"Expected Upset Rate:  {best_scenario['upset_rate']:.2%}")
    print(f"Expected Fairness:    {best_scenario['kendall_tau']:.4f}")
    print("="*50)
    
    # Plot
    plot_pareto(results_df, pareto_df)

if __name__ == "__main__":
    main()
