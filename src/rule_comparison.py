#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Rule Comparison & Counterfactual Analysis
The Multiverse Simulator: Compare 3 elimination rules against historical data.

Core Question: "Does one method favor fan votes more?"
"What if rules were different?"

Universes:
- A: Rank System (Classic) - Seasons 1-2
- B: Percent System (Modern) - Seasons 3-27
- C: Rank + Judges' Save (Hybrid) - Seasons 28+

Author: MCM Team
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path
import warnings
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
INPUT_JUDGE_PATH = BASE_DIR / "data_processed" / "dwts_simulation_input.csv"
INPUT_FAN_PATH = BASE_DIR / "results" / "question1" / "full_simulation_bayesian.csv"
OUTPUT_PATH = BASE_DIR / "results" / "question2" / "counterfactual_outcomes.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots" / "question2"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Era definitions for analysis
ERA_MAP = {
    "Rank": list(range(1, 3)),           # Seasons 1-2
    "Percent": list(range(3, 28)),       # Seasons 3-27
    "Rank_With_Save": list(range(28, 35)) # Seasons 28-34
}

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Data Loading & Preparation
# =============================================================================
def load_and_merge_data() -> pd.DataFrame:
    """
    Load judge data and fan estimates, inner join on [season, week, celebrity_name].
    
    Returns:
        Merged DataFrame with all necessary columns.
    """
    print("[INFO] Loading judge data...")
    judge_df = pd.read_csv(INPUT_JUDGE_PATH)
    
    print("[INFO] Loading fan estimates...")
    fan_df = pd.read_csv(INPUT_FAN_PATH)
    
    # Select relevant columns from fan estimates
    fan_cols = ['season', 'week', 'celebrity_name', 'estimated_fan_share', 'share_std']
    fan_subset = fan_df[fan_cols].copy()
    
    # Merge
    merged = pd.merge(
        judge_df,
        fan_subset,
        on=['season', 'week', 'celebrity_name'],
        how='inner'
    )
    
    # Ensure fan share is valid (non-negative, not NaN)
    merged = merged[merged['estimated_fan_share'].notna() & (merged['estimated_fan_share'] >= 0)]
    
    print(f"[INFO] Merged dataset shape: {merged.shape}")
    print(f"[INFO] Seasons: {merged['season'].nunique()}, Weeks: {merged['week'].nunique()}")
    
    return merged


def compute_fan_rank(fan_shares: np.ndarray) -> np.ndarray:
    """
    Compute fan ranks from fan shares using method='average' (ties get average rank).
    
    Args:
        fan_shares: Array of fan shares (higher = better)
        
    Returns:
        Array of ranks (1 = best, higher = worse)
    """
    # Use pandas rank for tie handling
    s = pd.Series(fan_shares)
    # Rank ascending=False: highest share gets rank 1
    ranks = s.rank(method='average', ascending=False).values
    return ranks


def simulate_week_rank(df_week: pd.DataFrame) -> Tuple[str, Dict]:
    """
    System A: Rank System (Classic)
    
    Logic:
    1. Compute fan ranks from estimated_fan_share (method='average')
    2. Total_Rank = Judge_Rank + Fan_Rank
    3. Loser = contestant with MAX Total_Rank (worst)
    4. Tie-breaker: If Total_Rank tied, contestant with worse Fan Rank loses.
    
    Returns:
        loser_name, metadata dict
    """
    if len(df_week) < 2:
        return None, {"error": "Not enough contestants"}
    
    # Compute fan ranks
    fan_shares = df_week['estimated_fan_share'].values
    fan_ranks = compute_fan_rank(fan_shares)
    
    judge_ranks = df_week['judge_rank'].values
    total_ranks = judge_ranks + fan_ranks
    
    # Find contestant(s) with worst total rank
    worst_total = total_ranks.max()
    worst_mask = total_ranks == worst_total
    
    if worst_mask.sum() == 1:
        # Single worst contestant
        loser_idx = np.where(worst_mask)[0][0]
    else:
        # Tie: pick the one with worse fan rank (higher rank number)
        tied_indices = np.where(worst_mask)[0]
        tied_fan_ranks = fan_ranks[tied_indices]
        # Higher fan rank number = worse
        worst_fan = tied_fan_ranks.max()
        # If still tied (unlikely), pick first
        loser_rel_idx = np.where(tied_fan_ranks == worst_fan)[0][0]
        loser_idx = tied_indices[loser_rel_idx]
    
    loser_name = df_week.iloc[loser_idx]['celebrity_name']
    
    metadata = {
        "total_ranks": total_ranks.tolist(),
        "fan_ranks": fan_ranks.tolist(),
        "judge_ranks": judge_ranks.tolist()
    }
    
    return loser_name, metadata


def simulate_week_percent(df_week: pd.DataFrame) -> Tuple[str, Dict]:
    """
    System B: Percent System (Modern)
    
    Logic:
    1. Total_Share = Judge_Share + Estimated_Fan_Share
    2. Loser = contestant with MIN Total_Share
    
    Returns:
        loser_name, metadata dict
    """
    if len(df_week) < 2:
        return None, {"error": "Not enough contestants"}
    
    judge_shares = df_week['judge_share'].values
    fan_shares = df_week['estimated_fan_share'].values
    total_shares = judge_shares + fan_shares
    
    loser_idx = total_shares.argmin()
    loser_name = df_week.iloc[loser_idx]['celebrity_name']
    
    metadata = {
        "total_shares": total_shares.tolist(),
        "judge_shares": judge_shares.tolist(),
        "fan_shares": fan_shares.tolist()
    }
    
    return loser_name, metadata


def simulate_week_rank_save(df_week: pd.DataFrame) -> Tuple[str, Dict]:
    """
    System C: Rank + Judges' Save (Hybrid)
    
    Logic:
    1. Compute Total_Rank (same as System A)
    2. Identify Bottom 2:
       - If >2 people tie at bottom, sort by Fan Rank (worse first) to pick specific Bottom 2
    3. The Save: Compare normalized_score (Judge Score) of Bottom 2
       - Higher Judge Score -> SAVED
       - Lower Judge Score -> ELIMINATED
       - If Judge Scores equal, contestant with worse Fan Rank goes home
    
    Returns:
        loser_name, metadata dict
    """
    if len(df_week) < 2:
        return None, {"error": "Not enough contestants"}
    
    # Step 1: Compute Total_Rank (same as System A)
    fan_shares = df_week['estimated_fan_share'].values
    fan_ranks = compute_fan_rank(fan_shares)
    judge_ranks = df_week['judge_rank'].values
    total_ranks = judge_ranks + fan_ranks
    
    # Step 2: Identify Bottom 2
    # Sort contestants by total_rank (descending: worst first)
    # Use fan_rank as secondary sort (worse fan rank first)
    df_sorted = df_week.copy()
    df_sorted['total_rank'] = total_ranks
    df_sorted['fan_rank'] = fan_ranks
    
    # Sort: total_rank descending, then fan_rank descending (higher rank number = worse)
    df_sorted = df_sorted.sort_values(
        ['total_rank', 'fan_rank'], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    # Bottom 2 are the first two rows
    bottom2 = df_sorted.iloc[:2]
    
    # Step 3: The Save
    judge_scores = bottom2['normalized_score'].values
    fan_ranks_b2 = bottom2['fan_rank'].values
    
    if judge_scores[0] > judge_scores[1]:
        # Contestant 0 has higher judge score -> saved, contestant 1 eliminated
        loser_idx_in_b2 = 1
    elif judge_scores[1] > judge_scores[0]:
        loser_idx_in_b2 = 0
    else:
        # Judge scores equal, compare fan ranks
        if fan_ranks_b2[0] > fan_ranks_b2[1]:
            # Contestant 0 has worse fan rank
            loser_idx_in_b2 = 0
        else:
            loser_idx_in_b2 = 1
    
    loser_name = bottom2.iloc[loser_idx_in_b2]['celebrity_name']
    
    metadata = {
        "total_ranks": total_ranks.tolist(),
        "fan_ranks": fan_ranks.tolist(),
        "judge_ranks": judge_ranks.tolist(),
        "bottom2_names": bottom2['celebrity_name'].tolist(),
        "bottom2_judge_scores": judge_scores.tolist(),
        "bottom2_fan_ranks": fan_ranks_b2.tolist()
    }
    
    return loser_name, metadata


# =============================================================================
# Simulation Loop with Enhanced Metrics
# =============================================================================
def run_counterfactual_simulation_with_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all three rule systems for each (season, week) group.
    Also compute per-contestant-week rankings for each system.
    
    Returns:
        results_df: weekly outcomes
        rank_df: per-contestant-week rankings for each system
    """
    print("[INFO] Running counterfactual simulation with metrics...")
    
    results = []
    rank_records = []
    
    # Group by season and week
    grouped = df.groupby(['season', 'week'])
    
    for (season, week), group in grouped:
        # Skip weeks with insufficient contestants
        if len(group) < 2:
            continue
        
        # Get actual eliminated contestant (if any)
        actual_loser = None
        if group['is_eliminated'].any():
            actual_loser = group[group['is_eliminated']]['celebrity_name'].iloc[0]
        
        # Determine rule system based on season
        if season <= 2:
            rule_system = "Rank"
        elif season <= 27:
            rule_system = "Percent"
        else:
            rule_system = "Rank_With_Save"
        
        # Simulate with each system
        rank_loser, rank_meta = simulate_week_rank(group)
        percent_loser, percent_meta = simulate_week_percent(group)
        save_loser, save_meta = simulate_week_rank_save(group)
        
        # Determine if any reversal occurred
        is_flipped = (rank_loser != percent_loser)
        
        results.append({
            'season': season,
            'week': week,
            'n_contestants': len(group),
            'rule_system': rule_system,
            'actual_loser': actual_loser,
            'rank_loser': rank_loser,
            'percent_loser': percent_loser,
            'save_loser': save_loser,
            'is_flipped': is_flipped
        })
        
        # Compute rankings for each contestant under each system
        # System A: Rank by total_rank (lower is better)
        fan_shares = group['estimated_fan_share'].values
        fan_ranks = compute_fan_rank(fan_shares)
        judge_ranks = group['judge_rank'].values
        total_ranks = judge_ranks + fan_ranks
        # Ranking: 1 = best (lowest total_rank)
        rank_rank = pd.Series(total_ranks).rank(method='average', ascending=True).values
        
        # System B: Rank by total_share (higher is better)
        judge_shares = group['judge_share'].values
        total_shares = judge_shares + fan_shares
        percent_rank = pd.Series(total_shares).rank(method='average', ascending=False).values
        
        # System C: Rank by total_rank (same as System A) but with save adjustment?
        # For ranking correlation, we use total_rank as well (save only affects elimination)
        save_rank = rank_rank.copy()
        
        # Record per contestant - use positional index
        # Reset group index to ensure alignment with arrays
        group_reset = group.reset_index(drop=True)
        for pos_idx, row in group_reset.iterrows():
            rank_records.append({
                'season': season,
                'week': week,
                'celebrity_name': row['celebrity_name'],
                'normalized_score': row['normalized_score'],
                'judge_share': row['judge_share'],
                'estimated_fan_share': row['estimated_fan_share'],
                'rank_rank': rank_rank[pos_idx],
                'percent_rank': percent_rank[pos_idx],
                'save_rank': save_rank[pos_idx],
                'is_eliminated': row['is_eliminated'],
                'is_bottom3_judge': False,  # will be computed later
                'rule_system': rule_system
            })
    
    results_df = pd.DataFrame(results)
    rank_df = pd.DataFrame(rank_records)
    
    # Determine bottom 3 judge scorers per week
    rank_df['is_bottom3_judge'] = False
    for (season, week), group in rank_df.groupby(['season', 'week']):
        # Sort by normalized_score ascending (lower = worse)
        sorted_indices = group['normalized_score'].argsort().values
        k = min(3, len(group))
        bottom_indices = sorted_indices[:k]
        # Get original indices in rank_df
        orig_indices = group.iloc[bottom_indices].index
        rank_df.loc[orig_indices, 'is_bottom3_judge'] = True
    
    print(f"[INFO] Simulated {len(results_df)} weeks")
    print(f"[INFO] Generated {len(rank_df)} contestant-week rankings")
    
    return results_df, rank_df


# =============================================================================
# Evaluation Metrics (Complete)
# =============================================================================
def compute_reversal_rate(results_df: pd.DataFrame) -> float:
    """Proportion of weeks where Rank result != Percent result."""
    if len(results_df) == 0:
        return 0.0
    return results_df['is_flipped'].mean()


def compute_fan_power_index(rank_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Spearman correlation between final ranking and fan share.
    
    For each system, compute correlation across all contestant-weeks.
    Also compute by era.
    
    Returns:
        Dict with FPI for each rule system and era
    """
    print("[INFO] Computing Fan Power Index...")
    
    # Overall FPI for each system
    systems = ['rank', 'percent', 'save']
    fpi = {}
    
    for sys in systems:
        rank_col = f'{sys}_rank'
        # Spearman correlation between rank and fan share (higher fan share should correlate with better rank (lower number))
        # Since rank: 1 = best, we expect negative correlation (higher fan share -> lower rank number)
        # Use absolute value for FPI magnitude
        corr, pval = spearmanr(rank_df[rank_col], rank_df['estimated_fan_share'])
        fpi[sys.capitalize()] = abs(corr) if not np.isnan(corr) else 0.0
    
    # FPI by era
    eras = ['Rank', 'Percent', 'Rank_With_Save']
    for era in eras:
        era_mask = rank_df['rule_system'] == era
        if era_mask.sum() > 0:
            for sys in systems:
                rank_col = f'{sys}_rank'
                corr, _ = spearmanr(rank_df.loc[era_mask, rank_col], 
                                    rank_df.loc[era_mask, 'estimated_fan_share'])
                key = f"{sys.capitalize()}_{era}"
                fpi[key] = abs(corr) if not np.isnan(corr) else 0.0
    
    return fpi


def compute_merit_safety_rate(results_df: pd.DataFrame, rank_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute % of times a "Bottom 3 Judge Scorer" survives under each system.
    
    For each system, check if the loser is among bottom 3 judge scorers.
    Survival rate = (bottom 3 contestants not eliminated) / total bottom 3 contestants.
    
    Returns:
        Dict with survival rates for each system
    """
    print("[INFO] Computing Merit Safety Rate...")
    
    # For each system, compute survival rate
    systems = ['rank', 'percent', 'save']
    safety_rates = {}
    
    for sys in systems:
        loser_col = f'{sys}_loser'
        # Filter weeks where we have a loser
        valid_weeks = results_df[results_df[loser_col].notna()].copy()
        
        bottom3_survived = 0
        bottom3_total = 0
        
        for _, row in valid_weeks.iterrows():
            season = row['season']
            week = row['week']
            loser = row[loser_col]
            
            # Get all contestants in this week that are bottom 3
            week_bottom3 = rank_df[
                (rank_df['season'] == season) & 
                (rank_df['week'] == week) & 
                (rank_df['is_bottom3_judge'])
            ]
            
            bottom3_total += len(week_bottom3)
            
            # Count how many bottom 3 contestants survived (i.e., not the loser)
            survived = week_bottom3[week_bottom3['celebrity_name'] != loser]
            bottom3_survived += len(survived)
        
        safety_rate = bottom3_survived / bottom3_total if bottom3_total > 0 else 0.0
        safety_rates[sys.capitalize()] = safety_rate
    
    return safety_rates


def compute_talent_elimination_rate(results_df: pd.DataFrame, rank_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算 Top 3 Judge Scorer (高技术选手) 的被淘汰率。
    
    这是 Mediocrity Survival Rate 的互补指标：
    - Talent Elimination Rate 越低，说明系统越保护高技术选手
    - 如果 System C 的 TER 最低，证明"评委拯救"有效保护了技术派
    
    Returns:
        Dict with elimination rates for each system
    """
    print("[INFO] Computing Talent Elimination Rate (Top 3 Judge Scorers)...")
    
    systems = ['rank', 'percent', 'save']
    elim_rates = {}
    
    for sys in systems:
        loser_col = f'{sys}_loser'
        valid_weeks = results_df[results_df[loser_col].notna()].copy()
        
        top3_eliminated = 0
        top3_total = 0
        
        for _, row in valid_weeks.iterrows():
            season = row['season']
            week = row['week']
            loser = row[loser_col]
            
            # Get this week's data
            week_data = rank_df[
                (rank_df['season'] == season) & 
                (rank_df['week'] == week)
            ].copy()
            
            if len(week_data) < 3:
                continue
            
            # Identify top 3 judge scorers (highest normalized_score)
            week_data = week_data.sort_values('normalized_score', ascending=False)
            top3_names = week_data.head(3)['celebrity_name'].tolist()
            
            top3_total += len(top3_names)
            
            # Check if loser is among top 3
            if loser in top3_names:
                top3_eliminated += 1
        
        elim_rate = top3_eliminated / top3_total if top3_total > 0 else 0.0
        elim_rates[sys.capitalize()] = elim_rate
    
    return elim_rates


def validate_system_c(results_df: pd.DataFrame) -> float:
    """
    Validate System C against actual history for Seasons 28-34.
    
    Returns:
        Accuracy rate (proportion of weeks where simulated loser == actual loser)
    """
    # Filter for seasons 28-34
    validation_df = results_df[
        (results_df['season'] >= 28) & 
        (results_df['season'] <= 34) &
        (results_df['actual_loser'].notna())
    ].copy()
    
    if len(validation_df) == 0:
        return 0.0
    
    # Compare save_loser vs actual_loser
    matches = validation_df['save_loser'] == validation_df['actual_loser']
    accuracy = matches.mean()
    
    print(f"[VALIDATION] System C accuracy (Seasons 28-34): {accuracy:.2%} ({matches.sum()}/{len(validation_df)})")
    
    return accuracy


# =============================================================================
# Visualization Functions (Complete)
# =============================================================================
def plot_fan_bias_comparison(fpi: Dict[str, float]):
    """
    Create bar chart of Fan Power Index by Era & Rule.
    """
    # Extract data for plotting
    # We'll show overall FPI for each system, and maybe by era
    systems = ['Rank', 'Percent', 'Save']
    overall_values = [fpi.get(sys, 0.0) for sys in systems]
    
    # Era-specific values
    era_values = {}
    for era in ['Rank', 'Percent', 'Rank_With_Save']:
        era_values[era] = [fpi.get(f'{sys}_{era}', 0.0) for sys in ['Rank', 'Percent', 'Save']]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Overall FPI
    ax1 = axes[0]
    x_pos = np.arange(len(systems))
    ax1.bar(x_pos, overall_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_xlabel('Rule System', fontsize=12)
    ax1.set_ylabel('Fan Power Index (|Spearman ρ|)', fontsize=12)
    ax1.set_title('Overall Fan Power Index\nHigher = More Fan Influence', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(systems)
    ax1.set_ylim(0, 1)
    # Add value labels
    for i, v in enumerate(overall_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Plot 2: FPI by Era
    ax2 = axes[1]
    width = 0.25
    x = np.arange(len(systems))
    for idx, (era, values) in enumerate(era_values.items()):
        offset = (idx - 1) * width
        ax2.bar(x + offset, values, width, label=era, alpha=0.8)
    
    ax2.set_xlabel('Rule System', fontsize=12)
    ax2.set_ylabel('Fan Power Index (|Spearman ρ|)', fontsize=12)
    ax2.set_title('Fan Power Index by Historical Era', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems)
    ax2.set_ylim(0, 1)
    ax2.legend(title='Era')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "q2_fan_bias_comparison.png", dpi=300)
    plt.close()
    
    print("[INFO] Saved fan bias comparison plot")


def plot_mediocrity_survival(survival_rates: Dict[str, float], talent_elim_rates: Dict[str, float]):
    """
    Create dual bar chart comparing:
    1. Mediocrity Survival Rate (平庸存活率) - 越低越公平
    2. Talent Elimination Rate (英才被杀率) - 越低越保护技术
    """
    systems = ['Rank', 'Percent', 'Save']
    surv_values = [survival_rates.get(sys, 0.0) for sys in systems]
    elim_values = [talent_elim_rates.get(sys, 0.0) for sys in systems]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mediocrity Survival Rate (lower = more meritocratic)
    ax1 = axes[0]
    bars1 = ax1.bar(systems, surv_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_title('Mediocrity Survival Rate\n(Bottom 3 Judge Scorers Survival Rate)\n↓ Lower = More Meritocratic', fontsize=12)
    ax1.set_ylabel('Survival Rate (%)', fontsize=11)
    ax1.set_xlabel('Rule System', fontsize=11)
    ax1.set_ylim(0, 1)
    for i, v in enumerate(surv_values):
        ax1.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=10)
    # Highlight the best (lowest)
    min_idx = np.argmin(surv_values)
    bars1[min_idx].set_edgecolor('gold')
    bars1[min_idx].set_linewidth(3)
    
    # Plot 2: Talent Elimination Rate (lower = better talent protection)
    ax2 = axes[1]
    bars2 = ax2.bar(systems, elim_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax2.set_title('Talent Elimination Rate\n(Top 3 Judge Scorers Elimination Rate)\n↓ Lower = Better Talent Protection', fontsize=12)
    ax2.set_ylabel('Elimination Rate (%)', fontsize=11)
    ax2.set_xlabel('Rule System', fontsize=11)
    ax2.set_ylim(0, max(elim_values) * 1.3 if max(elim_values) > 0 else 0.1)
    for i, v in enumerate(elim_values):
        ax2.text(i, v + 0.002, f'{v:.2%}', ha='center', fontsize=10)
    # Highlight the best (lowest)
    min_idx = np.argmin(elim_values)
    bars2[min_idx].set_edgecolor('gold')
    bars2[min_idx].set_linewidth(3)
    
    plt.suptitle('Merit-Based Evaluation: Which System Best Balances Talent vs Popularity?', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "q2_merit_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[INFO] Saved merit metrics plot (mediocrity survival + talent elimination)")


def analyze_celebrity_case(rank_df: pd.DataFrame, results_df: pd.DataFrame,
                           name: str, season: int) -> Dict:
    """
    通用名人案例分析框架：分析指定选手在三种规则下的命运。
    
    Args:
        rank_df: 选手每周排名数据
        results_df: 每周淘汰结果数据
        name: 选手名字（支持模糊匹配）
        season: 赛季号
    
    Returns:
        Dict containing:
        - weeks_data: 每周在三系统下的排名
        - elimination_weeks: {system: elimination_week} 各系统淘汰周次
        - bottom3_count: 评委分进入Bottom 3的次数
        - actual_finish: 实际最终排名
    """
    # Filter data for specified season and name
    celeb_data = rank_df[
        (rank_df['season'] == season) & 
        (rank_df['celebrity_name'].str.contains(name, case=False, na=False))
    ].copy()
    
    if celeb_data.empty:
        print(f"[WARNING] {name} not found in Season {season}")
        return None
    
    # Get weekly results for this season
    season_results = results_df[results_df['season'] == season].copy()
    
    # Find elimination week under each system
    elimination_weeks = {'rank': None, 'percent': None, 'save': None}
    for _, row in season_results.iterrows():
        for sys in ['rank', 'percent', 'save']:
            loser_col = f'{sys}_loser'
            if row[loser_col] and name.lower() in row[loser_col].lower():
                if elimination_weeks[sys] is None:
                    elimination_weeks[sys] = row['week']
    
    # Count bottom 3 judge appearances
    bottom3_count = celeb_data['is_bottom3_judge'].sum()
    total_weeks = len(celeb_data)
    
    # Get actual final position (approximation based on last week they appeared)
    actual_last_week = celeb_data['week'].max()
    
    # Extract weekly ranks
    weeks = sorted(celeb_data['week'].unique())
    weeks_data = []
    for week in weeks:
        week_data = celeb_data[celeb_data['week'] == week].iloc[0]
        weeks_data.append({
            'week': week,
            'rank_rank': week_data['rank_rank'],
            'percent_rank': week_data['percent_rank'],
            'save_rank': week_data['save_rank'],
            'is_bottom3_judge': week_data['is_bottom3_judge'],
            'normalized_score': week_data['normalized_score']
        })
    
    return {
        'name': name,
        'season': season,
        'weeks_data': weeks_data,
        'elimination_weeks': elimination_weeks,
        'bottom3_count': bottom3_count,
        'total_weeks': total_weeks,
        'actual_last_week': actual_last_week
    }


def plot_celebrity_case(case_data: Dict, save_path: Path = None):
    """
    为单个名人生成三系统命运对比图。
    """
    if case_data is None:
        return
    
    weeks_data = case_data['weeks_data']
    weeks = [d['week'] for d in weeks_data]
    rank_vals = [d['rank_rank'] for d in weeks_data]
    percent_vals = [d['percent_rank'] for d in weeks_data]
    save_vals = [d['save_rank'] for d in weeks_data]
    bottom3_weeks = [d['week'] for d in weeks_data if d['is_bottom3_judge']]
    
    # Calculate reasonable y-axis range
    all_ranks = rank_vals + percent_vals + save_vals
    min_rank = min(all_ranks)
    max_rank = max(all_ranks)
    rank_range = max_rank - min_rank
    y_padding = max(1, rank_range * 0.15)  # At least 1 rank padding
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.step(weeks, rank_vals, where='mid', label='System A (Rank)', linewidth=2.5, marker='o', markersize=6)
    ax.step(weeks, percent_vals, where='mid', label='System B (Percent)', linewidth=2.5, marker='s', markersize=6)
    ax.step(weeks, save_vals, where='mid', label='System C (Rank+Save)', linewidth=2.5, marker='^', markersize=6)
    
    # Mark bottom 3 judge weeks
    for w in bottom3_weeks:
        ax.axvline(x=w, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Mark elimination weeks
    elim = case_data['elimination_weeks']
    colors = {'rank': '#3498db', 'percent': '#2ecc71', 'save': '#e74c3c'}
    y_text_pos = min_rank - y_padding * 0.5
    for sys, week in elim.items():
        if week:
            ax.axvline(x=week, color=colors[sys], linestyle='--', linewidth=2, alpha=0.7)
            ax.text(week + 0.1, y_text_pos, f'{sys.upper()} elim', rotation=90, 
                    fontsize=9, color=colors[sys], va='bottom', ha='left')
    
    ax.set_xlabel('Week', fontsize=13)
    ax.set_ylabel('Virtual Rank (1 = Best)', fontsize=13)
    ax.set_title(f"{case_data['name']} (Season {case_data['season']}) - Virtual Rank Under Different Rules\n"
                 f"Bottom 3 Judge: {case_data['bottom3_count']}/{case_data['total_weeks']} weeks", fontsize=15, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.8)
    
    # Set y-axis limits with padding to avoid compression
    ax.set_ylim(max_rank + y_padding, min_rank - y_padding)
    
    plt.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved celebrity case plot: {save_path}")
    plt.close()


def analyze_all_celebrities(rank_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict:
    """
    分析题目中提到的4位争议名人，生成可视化和汇总报告。
    
    Returns:
        Dict with analysis results for all celebrities
    """
    print("\n" + "="*70)
    print("CELEBRITY CASE STUDY ANALYSIS")
    print("="*70)
    
    celebrities = [
        ('Jerry Rice', 2),      # S2: 评委分极低却获亚军
        ('Billy Ray Cyrus', 4), # S4: 评委分垫底却获第5名  
        ('Bristol Palin', 11),  # S11: 12次评委分最低却获第3名
        ('Bobby Bones', 27)     # S27: 评委分低却夺冠
    ]
    
    all_results = {}
    
    for name, season in celebrities:
        print(f"\n[CASE STUDY] Analyzing {name} (Season {season})...")
        case_data = analyze_celebrity_case(rank_df, results_df, name, season)
        
        if case_data:
            all_results[name] = case_data
            
            # Print summary
            elim = case_data['elimination_weeks']
            print(f"  - Bottom 3 Judge appearances: {case_data['bottom3_count']}/{case_data['total_weeks']} weeks")
            print(f"  - Elimination under Rank System: Week {elim['rank'] or 'Never (survived)'}")
            print(f"  - Elimination under Percent System: Week {elim['percent'] or 'Never (survived)'}")
            print(f"  - Elimination under Rank+Save: Week {elim['save'] or 'Never (survived)'}")
            
            # Generate individual plot
            safe_name = name.replace(' ', '_').lower()
            plot_path = PLOTS_DIR / f"q2_celebrity_{safe_name}_s{season}.png"
            plot_celebrity_case(case_data, plot_path)
        else:
            print(f"  [WARNING] Could not analyze {name}")
    
    # Generate combined comparison plot
    if all_results:
        plot_combined_celebrities(all_results)
    
    return all_results


def plot_combined_celebrities(all_results: Dict):
    """
    生成4位名人的综合对比图（2x2子图）。
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (name, case_data) in enumerate(all_results.items()):
        ax = axes[idx]
        weeks_data = case_data['weeks_data']
        weeks = [d['week'] for d in weeks_data]
        rank_vals = [d['rank_rank'] for d in weeks_data]
        percent_vals = [d['percent_rank'] for d in weeks_data]
        save_vals = [d['save_rank'] for d in weeks_data]
        
        ax.step(weeks, rank_vals, where='mid', label='Rank', linewidth=2, marker='o', markersize=4)
        ax.step(weeks, percent_vals, where='mid', label='Percent', linewidth=2, marker='s', markersize=4)
        ax.step(weeks, save_vals, where='mid', label='Rank+Save', linewidth=2, marker='^', markersize=4)
        
        # Mark elimination weeks
        elim = case_data['elimination_weeks']
        colors = {'rank': '#3498db', 'percent': '#2ecc71', 'save': '#e74c3c'}
        for sys, week in elim.items():
            if week:
                ax.axvline(x=week, color=colors[sys], linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Virtual Rank', fontsize=10)
        ax.set_title(f"{name} (S{case_data['season']})\nBottom 3: {case_data['bottom3_count']}/{case_data['total_weeks']} weeks", 
                     fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    
    plt.suptitle('Controversial Celebrity Case Studies: What If Rules Were Different?', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "q2_celebrity_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved combined celebrity plot")


def plot_bobby_bones_survival(rank_df: pd.DataFrame, results_df: pd.DataFrame):
    """
    Plot Bobby Bones' virtual rank across systems.
    
    Requirements:
    - Filter Season 27, Bobby Bones
    - Step plot of his "Virtual Rank" (Universe A vs B vs C) over time
    - Horizontal line at "Bottom 2" threshold
    - Highlight week he would be eliminated under System C
    """
    print("[INFO] Creating Bobby Bones survival plot...")
    
    # Filter data for Season 27, Bobby Bones
    bones_data = rank_df[
        (rank_df['season'] == 27) & 
        (rank_df['celebrity_name'].str.contains('Bobby Bones', case=False, na=False))
    ].copy()
    
    if bones_data.empty:
        print("[WARNING] Bobby Bones not found in Season 27")
        return
    
    # Get weekly results for Season 27
    season_results = results_df[results_df['season'] == 27].copy()
    
    # Determine when he would be eliminated under System C
    # Find first week where save_loser == Bobby Bones
    elimination_week = None
    for _, row in season_results.iterrows():
        if row['save_loser'] and 'Bobby Bones' in row['save_loser']:
            elimination_week = row['week']
            break
    
    weeks = bones_data['week'].unique()
    weeks.sort()
    
    # Extract his ranks for each system
    rank_vals = []
    percent_vals = []
    save_vals = []
    
    for week in weeks:
        week_data = bones_data[bones_data['week'] == week]
        if len(week_data) > 0:
            rank_vals.append(week_data['rank_rank'].iloc[0])
            percent_vals.append(week_data['percent_rank'].iloc[0])
            save_vals.append(week_data['save_rank'].iloc[0])
        else:
            rank_vals.append(np.nan)
            percent_vals.append(np.nan)
            save_vals.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Step plot
    ax.step(weeks, rank_vals, where='mid', label='System A (Rank)', linewidth=2, marker='o')
    ax.step(weeks, percent_vals, where='mid', label='System B (Percent)', linewidth=2, marker='s')
    ax.step(weeks, save_vals, where='mid', label='System C (Rank+Save)', linewidth=2, marker='^')
    
    # Bottom 2 threshold (rank > n_contestants - 1)
    # Need n_contestants per week
    for week in weeks:
        n_contestants = season_results[season_results['week'] == week]['n_contestants'].iloc[0]
        bottom2_threshold = n_contestants - 1  # If rank > this, in bottom 2
        ax.axhline(y=bottom2_threshold + 0.5, xmin=(week-1)/len(weeks), xmax=week/len(weeks), 
                   color='red', linestyle='--', alpha=0.3)
    
    # Highlight elimination week
    if elimination_week:
        ax.axvline(x=elimination_week, color='red', linestyle='-', alpha=0.7, linewidth=2)
        ax.text(elimination_week, max(save_vals) * 0.8, f'Eliminated Week {elimination_week}', 
                rotation=90, ha='right', va='top', fontsize=10, color='red')
    
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Virtual Rank (1 = Best)', fontsize=12)
    ax.set_title('Bobby Bones (Season 27) - Virtual Rank Under Different Rules', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis so rank 1 is at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "q2_bobby_bones_survival.png", dpi=300)
    plt.close()
    
    print(f"[INFO] Saved Bobby Bones survival plot (eliminated week: {elimination_week})")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("="*70)
    print("MCM 2026 Problem C - Rule Comparison (Multiverse Simulator)")
    print("="*70)
    
    # Step 1: Load and merge data
    df = load_and_merge_data()
    
    # Step 2: Run counterfactual simulation with metrics
    results_df, rank_df = run_counterfactual_simulation_with_metrics(df)
    
    # Step 3: Compute metrics
    reversal_rate = compute_reversal_rate(results_df)
    print(f"\n[METRICS] Reversal Rate (Rank vs Percent): {reversal_rate:.2%}")
    
    fpi = compute_fan_power_index(rank_df)
    mediocrity_rates = compute_merit_safety_rate(results_df, rank_df)  # Renamed for clarity
    talent_elim_rates = compute_talent_elimination_rate(results_df, rank_df)  # NEW
    validation_accuracy = validate_system_c(results_df)
    
    # Print metrics
    print("\n[METRICS] Fan Power Index (|Spearman ρ|):")
    for key, value in fpi.items():
        if '_' not in key:  # Overall metrics
            print(f"  - {key}: {value:.3f}")
    
    print("\n[METRICS] Mediocrity Survival Rate (Bottom 3 Judge Scorers - Lower = More Meritocratic):")
    for sys, rate in mediocrity_rates.items():
        print(f"  - {sys}: {rate:.2%}")
    
    print("\n[METRICS] Talent Elimination Rate (Top 3 Judge Scorers - Lower = Better Protection):")
    for sys, rate in talent_elim_rates.items():
        print(f"  - {sys}: {rate:.2%}")
    
    # Step 4: Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved counterfactual outcomes to: {OUTPUT_PATH}")
    
    # Save rank data for further analysis
    rank_output = BASE_DIR / "results" / "question2" / "counterfactual_rankings.csv"
    rank_df.to_csv(rank_output, index=False)
    print(f"[INFO] Saved contestant rankings to: {rank_output}")
    
    # Step 5: Generate visualizations
    plot_fan_bias_comparison(fpi)
    plot_mediocrity_survival(mediocrity_rates, talent_elim_rates)  # UPDATED
    plot_bobby_bones_survival(rank_df, results_df)
    
    # Step 6: Celebrity Case Studies (NEW - addresses all 4 controversial figures)
    celebrity_results = analyze_all_celebrities(rank_df, results_df)
    
    # Step 7: Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total weeks simulated: {len(results_df)}")
    print(f"Reversal Rate (Rank vs Percent): {reversal_rate:.2%}")
    print(f"System C Validation Accuracy (S28-34): {validation_accuracy:.2%}")
    print(f"\nCelebrity Case Studies Analyzed: {len(celebrity_results)}")
    for name, data in celebrity_results.items():
        print(f"  - {name} (S{data['season']}): Bottom 3 Judge {data['bottom3_count']}/{data['total_weeks']} weeks")
    print(f"\nOutput files:")
    print(f"  - Counterfactual outcomes: {OUTPUT_PATH}")
    print(f"  - Contestant rankings: {rank_output}")
    print(f"  - Fan bias plot: {PLOTS_DIR / 'q2_fan_bias_comparison.png'}")
    print(f"  - Merit metrics plot: {PLOTS_DIR / 'q2_merit_metrics.png'}")
    print(f"  - Bobby Bones plot: {PLOTS_DIR / 'q2_bobby_bones_survival.png'}")
    print(f"  - Celebrity combined plot: {PLOTS_DIR / 'q2_celebrity_combined.png'}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()