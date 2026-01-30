#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Bayesian-Dirichlet Monte Carlo Simulation
- Bayesian Updating with Industry-Based Asymmetric Dirichlet Prior

══════════════════════════════════════════════════════════════════════════════
Bayesian-Dirichlet Optimization: Industry-Informed Two-Class Dirichlet Prior
══════════════════════════════════════════════════════════════════════════════

核心问题：
---------
原始模型使用对称 Dirichlet 先验 Dir(α, α, ..., α)，假设所有选手的"粉丝爆发概率"
相同。然而，现实中不同行业的选手具有截然不同的粉丝基础：

- Sean Spicer (S28, Politician): 评委分数垫底，但政治粉丝投票极高，存活多周
- Reality Stars: 自带社交媒体粉丝群体，投票动员能力强
- 传统 Actor/Athlete: 粉丝基础相对普通

这种"起跑线不同"的现象导致对称先验无法解释某些"异常"淘汰/存活案例。

方案 A：两类先验策略
-------------------
将选手分为两类：

1. **"自带粉丝群体"类 (Built-in Fanbase)**:
   - Reality Star / TV Personality
   - Social Media Personality  
   - Politician
   - Radio Personality
   → 使用较高的 α_high = 1.2（更可能成为"大头"）

2. **"普通选手"类 (Regular Contestants)**:
   - Actor/Actress, Athlete, Singer, Model, etc.
   → 使用标准的 α_low = 0.8（稀疏解倾向）

数学直觉：
---------
- α < 1: 分布呈"U型"，倾向于极端值（少数人拿大头）
- α = 1: 均匀分布
- α > 1: 分布呈"倒U型"，倾向于集中（大家份额接近）

通过给"自带粉丝"选手更高的 α，使其在先验分布中更可能获得较高份额。

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
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = BASE_DIR / "data_raw" / "2026_MCM_Problem_C_Data.csv"
INPUT_PATH = BASE_DIR / "data_processed" / "dwts_simulation_input.csv"
OUTPUT_PATH = BASE_DIR / "results" / "full_simulation_bayesian.csv"

# Simulation parameters
N_SIMULATIONS = 10000
RANDOM_SEED = 42

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │           方案 A: 行业二分类 Dirichlet 先验参数                              │
# └─────────────────────────────────────────────────────────────────────────────┘

# "自带粉丝群体"行业列表
BUILT_IN_FANBASE_INDUSTRIES = {
    'TV Personality',
    'Social Media Personality', 
    'Social media personality',  # 数据中有大小写不一致
    'Politician',
    'Radio Personality',
    'Reality Star',              # 如果存在
}

# 两类选手的 α 值
ALPHA_FANBASE = 1.2    # "自带粉丝群体"选手：更可能获得高粉丝份额
ALPHA_REGULAR = 0.8    # "普通选手"：标准稀疏解先验

# 时间动态参数
LEARNING_RATE = 0.4
EVIDENCE_BOOST = 5.0
MIN_ALPHA = 0.1


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class WeekState:
    """State information passed between weeks"""
    alpha: np.ndarray
    contestant_names: List[str]
    contestant_industries: List[str]
    
    @property
    def alpha_sum(self) -> float:
        return self.alpha.sum()
    
    @property
    def n_contestants(self) -> int:
        return len(self.contestant_names)


# =============================================================================
# Load Industry Information
# =============================================================================
def load_industry_mapping() -> Dict[str, str]:
    """
    Load celebrity_industry mapping from raw data.
    
    Returns:
        Dict mapping celebrity_name -> celebrity_industry
    """
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    # Clean column names
    raw_df.columns = raw_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Create mapping
    industry_map = dict(zip(
        raw_df['celebrity_name'].str.strip(),
        raw_df['celebrity_industry'].str.strip()
    ))
    
    return industry_map


def classify_industry(industry: str) -> str:
    """
    Classify industry into 'fanbase' or 'regular'.
    
    Args:
        industry: Raw industry string
        
    Returns:
        'fanbase' or 'regular'
    """
    if pd.isna(industry):
        return 'regular'
    
    industry_clean = industry.strip()
    
    if industry_clean in BUILT_IN_FANBASE_INDUSTRIES:
        return 'fanbase'
    else:
        return 'regular'


def get_alpha_by_industry(industry: str) -> float:
    """
    Get Dirichlet α value based on industry classification.
    
    Args:
        industry: Raw industry string
        
    Returns:
        α value for Dirichlet distribution
    """
    classification = classify_industry(industry)
    
    if classification == 'fanbase':
        return ALPHA_FANBASE
    else:
        return ALPHA_REGULAR


# =============================================================================
# Core Bayesian Functions with Industry Prior
# =============================================================================
def initialize_industry_prior(
    contestant_names: List[str],
    industry_map: Dict[str, str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Initialize Dirichlet prior based on industry classification.
    
    Args:
        contestant_names: List of contestant names
        industry_map: Name -> Industry mapping
        
    Returns:
        (alpha_array, industry_list)
    """
    alpha_list = []
    industry_list = []
    
    for name in contestant_names:
        industry = industry_map.get(name, 'Unknown')
        alpha = get_alpha_by_industry(industry)
        
        alpha_list.append(alpha)
        industry_list.append(industry)
    
    return np.array(alpha_list), industry_list


def align_prior_to_current(
    prev_state: WeekState,
    current_names: List[str],
    industry_map: Dict[str, str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Align previous week's posterior to current week's contestants.
    
    Args:
        prev_state: Previous week's state
        current_names: Current week's contestant names
        industry_map: Name -> Industry mapping
        
    Returns:
        (aligned_alpha, industry_list)
    """
    aligned_alpha = []
    industry_list = []
    
    for name in current_names:
        industry = industry_map.get(name, 'Unknown')
        industry_list.append(industry)
        
        if name in prev_state.contestant_names:
            # Continuing contestant: inherit accumulated α
            idx = prev_state.contestant_names.index(name)
            aligned_alpha.append(prev_state.alpha[idx])
        else:
            # New contestant (rare)
            aligned_alpha.append(get_alpha_by_industry(industry))
    
    aligned_alpha = np.array(aligned_alpha)
    
    # Scale to maintain relative proportions after elimination
    remaining_ratio = len(current_names) / prev_state.n_contestants
    if remaining_ratio < 1:
        scale_factor = min(1.0 / remaining_ratio, 1.3)
        aligned_alpha = aligned_alpha * scale_factor
    
    aligned_alpha = np.maximum(aligned_alpha, MIN_ALPHA)
    
    return aligned_alpha, industry_list


def update_prior_with_evidence(
    current_alpha: np.ndarray,
    evidence: np.ndarray,
    learning_rate: float = LEARNING_RATE
) -> np.ndarray:
    """
    Bayesian update: accumulate evidence into prior.
    
    Update rule: α_new = α_old + η · evidence_scaled
    """
    evidence_scaled = evidence * EVIDENCE_BOOST
    new_alpha = current_alpha + learning_rate * evidence_scaled
    new_alpha = np.maximum(new_alpha, MIN_ALPHA)
    return new_alpha


# =============================================================================
# Simulation Functions
# =============================================================================
def generate_fan_shares(alpha: np.ndarray, n_sims: int = N_SIMULATIONS) -> np.ndarray:
    """Generate fan share samples from Dirichlet(α)."""
    return np.random.dirichlet(alpha, size=n_sims)


def shares_to_ranks(shares: np.ndarray) -> np.ndarray:
    """Convert shares to ranks (1 = highest)."""
    order = np.argsort(-shares, axis=1)
    ranks = np.argsort(order, axis=1) + 1
    return ranks.astype(float)


def apply_rank_rule(judge_ranks: np.ndarray, fan_shares: np.ndarray) -> np.ndarray:
    """RANK elimination rule (Seasons 1-2)."""
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks
    return np.argmax(total_scores + fan_ranks * 0.001, axis=1)


def apply_percent_rule(judge_shares: np.ndarray, fan_shares: np.ndarray) -> np.ndarray:
    """PERCENT elimination rule (Seasons 3-27)."""
    return np.argmin(judge_shares + fan_shares, axis=1)


def apply_rank_with_save_rule(
    judge_ranks: np.ndarray,
    judge_scores: np.ndarray,
    fan_shares: np.ndarray
) -> np.ndarray:
    """RANK_WITH_SAVE elimination rule (Seasons 28+)."""
    n_sims, n_contestants = fan_shares.shape
    
    if n_contestants < 2:
        return np.zeros(n_sims, dtype=int)
    
    fan_ranks = shares_to_ranks(fan_shares)
    total_scores = judge_ranks + fan_ranks + fan_ranks * 1e-6
    
    if n_contestants == 2:
        if judge_scores[0] < judge_scores[1]:
            return np.zeros(n_sims, dtype=int)
        elif judge_scores[1] < judge_scores[0]:
            return np.ones(n_sims, dtype=int)
        else:
            return np.argmax(fan_ranks, axis=1)
    
    sorted_indices = np.argsort(-total_scores, axis=1)
    bottom2_first = sorted_indices[:, 0]
    bottom2_second = sorted_indices[:, 1]
    
    judge_scores_first = judge_scores[bottom2_first]
    judge_scores_second = judge_scores[bottom2_second]
    
    return np.where(
        judge_scores_first < judge_scores_second,
        bottom2_first,
        np.where(judge_scores_second < judge_scores_first, bottom2_second, bottom2_first)
    )


def simulate_week_with_industry_prior(
    week_data: pd.DataFrame,
    prev_state: Optional[WeekState],
    rule_system: str,
    industry_map: Dict[str, str]
) -> Tuple[pd.DataFrame, Optional[WeekState]]:
    """
    Simulate single week with industry-based Dirichlet prior.
    """
    contestant_names = week_data['celebrity_name'].tolist()
    n_contestants = len(contestant_names)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Construct Industry-Based Prior (方案 A 核心)
    # ─────────────────────────────────────────────────────────────────────────
    if prev_state is None:
        # First week: use industry-based α
        alpha, industries = initialize_industry_prior(contestant_names, industry_map)
    else:
        # Subsequent weeks: inherit + align
        alpha, industries = align_prior_to_current(prev_state, contestant_names, industry_map)
    
    prior_strength = alpha.sum()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Generate Samples
    # ─────────────────────────────────────────────────────────────────────────
    fan_share_samples = generate_fan_shares(alpha, N_SIMULATIONS)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Apply Elimination Rules
    # ─────────────────────────────────────────────────────────────────────────
    judge_shares = week_data['judge_share'].values
    judge_ranks = week_data['judge_rank'].values
    judge_scores = week_data['normalized_score'].values
    
    if rule_system == 'Rank':
        simulated_eliminated = apply_rank_rule(
            np.tile(judge_ranks, (N_SIMULATIONS, 1)), fan_share_samples
        )
    elif rule_system == 'Percent':
        simulated_eliminated = apply_percent_rule(
            np.tile(judge_shares, (N_SIMULATIONS, 1)), fan_share_samples
        )
    else:  # Rank_With_Save
        simulated_eliminated = apply_rank_with_save_rule(
            np.tile(judge_ranks, (N_SIMULATIONS, 1)), judge_scores, fan_share_samples
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Filter Valid Simulations
    # ─────────────────────────────────────────────────────────────────────────
    actual_eliminated_mask = week_data['is_eliminated'].values
    
    if actual_eliminated_mask.sum() == 0:
        # Non-elimination week
        new_state = WeekState(alpha=alpha, contestant_names=contestant_names, 
                              contestant_industries=industries)
        results = week_data.copy()
        results['estimated_fan_share'] = alpha / alpha.sum()
        results['share_std'] = np.nan
        results['confidence'] = np.nan
        results['n_valid_sims'] = 0
        results['prior_strength'] = prior_strength
        results['posterior_strength'] = prior_strength
        results['industry_class'] = [classify_industry(i) for i in industries]
        return results, new_state
    
    actual_eliminated_idx = np.where(actual_eliminated_mask)[0][0]
    valid_mask = simulated_eliminated == actual_eliminated_idx
    n_valid = valid_mask.sum()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Estimate Fan Shares
    # ─────────────────────────────────────────────────────────────────────────
    if n_valid > 0:
        valid_samples = fan_share_samples[valid_mask]
        estimated_shares = valid_samples.mean(axis=0)
        share_stds = valid_samples.std(axis=0)
        confidence = n_valid / N_SIMULATIONS
        new_alpha = update_prior_with_evidence(alpha, estimated_shares)
    else:
        estimated_shares = alpha / alpha.sum()
        share_stds = np.full(n_contestants, np.nan)
        confidence = 0.0
        new_alpha = alpha
    
    posterior_strength = new_alpha.sum()
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Build Results
    # ─────────────────────────────────────────────────────────────────────────
    results = week_data.copy()
    results['estimated_fan_share'] = estimated_shares
    results['share_std'] = share_stds
    results['confidence'] = confidence
    results['n_valid_sims'] = n_valid
    results['prior_strength'] = prior_strength
    results['posterior_strength'] = posterior_strength
    results['industry_class'] = [classify_industry(i) for i in industries]
    
    new_state = WeekState(
        alpha=new_alpha,
        contestant_names=contestant_names,
        contestant_industries=industries
    )
    
    return results, new_state


def run_simulation_with_industry_prior(df: pd.DataFrame) -> pd.DataFrame:
    """Run full simulation with industry-based prior."""
    np.random.seed(RANDOM_SEED)
    
    # Load industry mapping
    print("[INFO] Loading industry mapping...")
    industry_map = load_industry_mapping()
    
    # Count classifications
    fanbase_count = sum(1 for i in industry_map.values() if classify_industry(i) == 'fanbase')
    regular_count = len(industry_map) - fanbase_count
    print(f"   'Fanbase' industries: {fanbase_count} contestants (alpha={ALPHA_FANBASE})")
    print(f"   'Regular' industries: {regular_count} contestants (alpha={ALPHA_REGULAR})")
    
    all_results = []
    seasons = df['season'].unique()
    
    for season in tqdm(seasons, desc="Processing seasons"):
        season_data = df[df['season'] == season].copy()
        weeks = sorted(season_data['week'].unique())
        rule_system = season_data['rule_system'].iloc[0]
        
        prev_state = None
        
        for week in weeks:
            week_data = season_data[season_data['week'] == week].copy()
            week_data = week_data.sort_values('celebrity_name').reset_index(drop=True)
            
            results, prev_state = simulate_week_with_industry_prior(
                week_data, prev_state, rule_system, industry_map
            )
            
            results['season'] = season
            results['week'] = week
            all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# Pressure Test: Sean Spicer Case
# =============================================================================
def pressure_test_sean_spicer(df: pd.DataFrame, results: pd.DataFrame):
    """
    Pressure test: Check if the model can now explain Sean Spicer's survival.
    
    Sean Spicer (S28) is the canonical "unexplainable" case:
    - Lowest judge scores
    - Survived many weeks due to political fanbase
    """
    print("\n" + "="*70)
    print("   PRESSURE TEST: Sean Spicer (S28)")
    print("="*70)
    
    # Find Sean Spicer's data
    spicer_results = results[results['celebrity_name'].str.contains('Spicer', case=False, na=False)]
    
    if len(spicer_results) == 0:
        print("   [WARNING] Sean Spicer not found in results")
        return
    
    print(f"\n   Sean Spicer's simulation results:")
    print("-" * 70)
    
    for _, row in spicer_results.iterrows():
        conf_val = row['confidence'] if pd.notna(row['confidence']) else 0
        print(f"   Week {row['week']:2d}: n_valid={row['n_valid_sims']:5d}, "
              f"conf={conf_val:.4f}, "
              f"class={row['industry_class']}, "
              f"fan_share={row['estimated_fan_share']:.4f}")
    
    # Summary
    valid_weeks = spicer_results[spicer_results['n_valid_sims'] > 0]
    total_weeks = len(spicer_results)
    explainable_weeks = len(valid_weeks)
    
    print(f"\n   Summary: {explainable_weeks}/{total_weeks} weeks explained (n_valid > 0)")
    
    if explainable_weeks > 0:
        print(f"   [OK] Industry-based prior IMPROVED explainability for Spicer")
    else:
        print(f"   [!!] Still cannot explain Spicer's survival")


# =============================================================================
# Main
# =============================================================================
def main():
    print("="*70)
    print("   MCM 2026 Problem C: Bayesian-Dirichlet Monte Carlo")
    print("   Industry-Based Two-Class Dirichlet Prior + Bayesian Updating")
    print("="*70)
    
    print(f"\n[CONFIG] Hyperparameters:")
    print(f"   alpha_fanbase (ALPHA_FANBASE) = {ALPHA_FANBASE}")
    print(f"   alpha_regular (ALPHA_REGULAR) = {ALPHA_REGULAR}")
    print(f"   eta (LEARNING_RATE)           = {LEARNING_RATE}")
    print(f"   N_SIMULATIONS                 = {N_SIMULATIONS}")
    
    print(f"\n[CONFIG] 'Built-in Fanbase' Industries:")
    for ind in sorted(BUILT_IN_FANBASE_INDUSTRIES):
        print(f"   - {ind}")
    
    # Load data
    print(f"\n[INFO] Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"   Rows: {len(df)}, Seasons: {df['season'].nunique()}")
    
    # Run simulation
    print(f"\n[INFO] Running simulation with industry-based prior...")
    results = run_simulation_with_industry_prior(df)
    
    # Results already contain all original columns
    final_results = results.copy()
    
    # Reorder columns: original columns first, then estimates
    estimate_cols = ['estimated_fan_share', 'share_std', 'confidence', 'n_valid_sims',
                     'prior_strength', 'posterior_strength', 'industry_class']
    other_cols = [c for c in final_results.columns if c not in estimate_cols]
    final_results = final_results[other_cols + estimate_cols]
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[INFO] Full simulation results saved to: {OUTPUT_PATH}")
    print(f"   Rows: {len(final_results)}, Columns: {len(final_results.columns)}")
    
    # Summary
    print("\n" + "="*70)
    print("   RESULTS SUMMARY")
    print("="*70)
    
    valid_results = final_results[final_results['n_valid_sims'] > 0]
    
    print(f"\n   Total contestant-weeks: {len(final_results)}")
    print(f"   Valid simulations: {len(valid_results)} ({100*len(valid_results)/len(final_results):.1f}%)")
    
    print(f"\n   Confidence by Industry Class:")
    for cls in ['fanbase', 'regular']:
        cls_data = valid_results[valid_results['industry_class'] == cls]
        if len(cls_data) > 0:
            print(f"      {cls:10s}: mean={cls_data['confidence'].mean():.4f}, "
                  f"n={len(cls_data)}")
    
    print(f"\n   Prior Strength Evolution (alpha_sum):")
    strength_by_week = final_results.groupby('week')['prior_strength'].mean()
    for week in [1, 3, 5, 7, 9]:
        if week in strength_by_week.index:
            print(f"      Week {week}: {strength_by_week[week]:.2f}")
    
    # Pressure test
    pressure_test_sean_spicer(df, final_results)
    
    print("\n" + "="*70)
    print("   Simulation Complete!")
    print("="*70)
    
    return final_results


if __name__ == "__main__":
    main()
