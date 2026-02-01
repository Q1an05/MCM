#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Judges' Save Rationality Analysis
Analyze historical data (Seasons 28-34) to determine how often judges actually
save the contestant with the higher judge score in the Bottom 2.

Author: MCM Team
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
INPUT_JUDGE_PATH = BASE_DIR / "data_processed" / "dwts_simulation_input.csv"
INPUT_FAN_PATH = BASE_DIR / "results" / "question1" / "full_simulation_bayesian.csv"

def load_data():
    """Load and merge judge and fan data for Save Era (S28+)."""
    # Load raw judge data (contains actual eliminations)
    judge_df = pd.read_csv(INPUT_JUDGE_PATH)
    
    # Load fan estimates
    fan_df = pd.read_csv(INPUT_FAN_PATH)
    fan_subset = fan_df[['season', 'week', 'celebrity_name', 'estimated_fan_share']]
    
    # Merge
    merged = pd.merge(
        judge_df,
        fan_subset,
        on=['season', 'week', 'celebrity_name'],
        how='inner'
    )
    
    # Filter for Rank + Save Era (Seasons 28-34)
    save_era = merged[(merged['season'] >= 28) & (merged['season'] <= 34)].copy()
    
    return save_era

def compute_ranks(df_week):
    """Compute Total Rank to identify Bottom 2."""
    # Fan Rank (Average method for ties)
    fan_shares = df_week['estimated_fan_share'].values
    fan_ranks = pd.Series(fan_shares).rank(method='average', ascending=False).values
    
    # Judge Rank
    judge_ranks = df_week['judge_rank'].values
    
    # Total Rank
    total_ranks = judge_ranks + fan_ranks
    return total_ranks

def analyze_save_decisions(df):
    """
    Analyze every week with an elimination to see if the higher judge scorer was saved.
    """
    print(f"[INFO] Analyzing Judges' Save decisions (Seasons 28-34)...")
    
    grouped = df.groupby(['season', 'week'])
    
    stats = {
        'total_cases': 0,           # Matches where we successfully identified B2
        'higher_score_saved': 0,    # Survivor had strictly higher score
        'lower_score_saved': 0,     # Survivor had strictly lower score
        'equal_score_saved': 0,     # Scores were tied
        'shock_elimination': 0,     # Actual loser wasn't in our predicted B2
        'details': []
    }
    
    for (season, week), group in grouped:
        # Check if anyone was actually eliminated
        eliminated_rows = group[group['is_eliminated'] == 1]
        if len(eliminated_rows) == 0:
            continue
            
        # If multiple eliminations (double elimination night), logic is complex.
        # We focus on single/standard eliminations or treat each pairwise.
        # For simplicity, we look at the standard case: Bottom 2 -> 1 goes home.
        
        # Calculate Ranks to find Simulated Bottom 2
        group['total_rank'] = compute_ranks(group)
        
        # Sort by total rank descending (Worst to Best)
        # Use fan rank as secondary sort for ties (matches show logic)
        group['fan_rank_temp'] = pd.Series(group['estimated_fan_share'].values).rank(ascending=False).values
        
        sorted_group = group.sort_values(
            by=['total_rank', 'fan_rank_temp'], 
            ascending=[False, False]
        ).reset_index(drop=True)
        
        # Our Predicted Bottom 2
        predicted_b2 = sorted_group.iloc[:2]
        b2_names = predicted_b2['celebrity_name'].tolist()
        
        # Identify the Actual Loser
        actual_loser_name = eliminated_rows.iloc[0]['celebrity_name']
        
        # Check if Actual Loser is in our Predicted Bottom 2
        if actual_loser_name not in b2_names:
            stats['shock_elimination'] += 1
            # "Shock" implies our fan estimates might be slightly off, 
            # or the public vote was very different.
            continue
            
        # Identify the Survivor (The other person in B2)
        survivor_row = predicted_b2[predicted_b2['celebrity_name'] != actual_loser_name].iloc[0]
        loser_row = predicted_b2[predicted_b2['celebrity_name'] == actual_loser_name].iloc[0]
        
        survivor_score = survivor_row['normalized_score']
        loser_score = loser_row['normalized_score']
        
        stats['total_cases'] += 1
        
        decision_type = "UNKNOWN"
        if survivor_score > loser_score:
            stats['higher_score_saved'] += 1
            decision_type = "Rational (Higher Score Saved)"
        elif survivor_score < loser_score:
            stats['lower_score_saved'] += 1
            decision_type = "Irrational (Lower Score Saved)"
        else:
            stats['equal_score_saved'] += 1
            decision_type = "Tie (Equal Scores)"
            
        stats['details'].append({
            'season': season,
            'week': week,
            'loser': actual_loser_name,
            'survivor': survivor_row['celebrity_name'],
            'loser_score': f"{loser_score:.2f}",
            'survivor_score': f"{survivor_score:.2f}",
            'decision': decision_type
        })
        
    return stats

def main():
    df = load_data()
    stats = analyze_save_decisions(df)
    
    print("\n" + "="*60)
    print("RESULTS: Judges' Save Rationality Analysis (S28-S34)")
    print("="*60)
    
    valid_cases = stats['total_cases']
    rational = stats['higher_score_saved']
    irrational = stats['lower_score_saved']
    ties = stats['equal_score_saved']
    
    if valid_cases > 0:
        rational_rate = rational / valid_cases
        irrational_rate = irrational / valid_cases
        tie_rate = ties / valid_cases
        
        # Meritocracy adherence: (Rational / (Rational + Irrational)) ignoring ties
        # meaningful_decisions = rational + irrational
        # adjusted_merit_rate = rational / meaningful_decisions if meaningful_decisions > 0 else 0
        
        print(f"Total Analyzed Bottom 2 Scenarios: {valid_cases}")
        print(f"Predicted B2 Mismatch (Shock Eliminations): {stats['shock_elimination']}")
        print("-" * 40)
        print(f"High Score Saved (Rational):   {rational:2d} ({rational_rate:.1%})")
        print(f"Low Score Saved (Irrational):  {irrational:2d} ({irrational_rate:.1%})")
        print(f"Equal Scores (Tie):            {ties:2d} ({tie_rate:.1%})")
        print("-" * 40)
        
        print("\n[CONCLUSION]")
        if rational > irrational:
            print(f"The judges largely follow the scores. The model assumption is strong.")
            print(f"Recommended Model Update: Keep deterministic or use p={rational_rate:.2f} for 'Save High Score'.")
        else:
            print(f"The judges are unpredictable or prefer low scorers (underdogs).")
            print(f"Model assumption of 'Merit Save' might be flawed.")
            
        print("\n[DETAILED DECISIONS (Sample)]")
        # Print a few Irrational cases if any
        irrational_cases = [d for d in stats['details'] if 'Irrational' in d['decision']]
        for case in irrational_cases[:5]:
            print(f"S{case['season']} W{case['week']}: Saved {case['survivor']} ({case['survivor_score']}) over {case['loser']} ({case['loser_score']}) -> {case['decision']}")
            
    else:
        print("Not enough valid Bottom 2 reconstructions found to draw conclusions.")
        print("Check if fan estimates align with historical eliminations.")

if __name__ == "__main__":
    main()
