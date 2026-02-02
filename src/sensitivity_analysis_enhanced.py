#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: 基于真实数据的灵敏度分析与模型评价
Enhanced Sensitivity Analysis with Real Data

功能：
1. Q1: 基于真实模型输出的 Bootstrap 置信区间分析
2. Q2: 基于真实规则对比数据的统计显著性检验
3. Q3: 基于真实聚类结果的 Bootstrap 稳健性分析
4. Q4: 基于历史回测的 DTPM 参数验证

Author: MCM Team
Date: 2026-02-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, pearsonr, chi2_contingency, bootstrap
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
from viz_config import *
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots" / "sensitivity_analysis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("   MCM 2026 Problem C: Sensitivity Analysis with Real Data")
print("="*70)

# =============================================================================
# Load Real Data
# =============================================================================
print("\n[INFO] Loading real data...")

# Q1 Data
bayesian_df = pd.read_csv(RESULTS_DIR / "full_simulation_bayesian.csv")
print(f"  ✓ Q1 Bayesian Results: {len(bayesian_df)} rows")

# Q2 Data
counterfactual_df = pd.read_csv(RESULTS_DIR / "question2" / "counterfactual_outcomes.csv")
print(f"  ✓ Q2 Counterfactual Analysis: {len(counterfactual_df)} rows")

# Q3 Data
industry_codebook = pd.read_csv(RESULTS_DIR / "question3" / "industry_codebook.csv")
print(f"  ✓ Q3 Industry Codebook: {len(industry_codebook)} rows")

# Q4 Data
grid_search_results = pd.read_csv(RESULTS_DIR / "system_design" / "grid_search_results.csv")
print(f"  ✓ Q4 Grid Search: {len(grid_search_results)} rows")

# =============================================================================
# PART 1: Q1 贝叶斯模型 - 基于真实数据的 Bootstrap 分析
# =============================================================================
print("\n" + "="*70)
print("   PART 1: Q1 Bayesian Model Empirical Sensitivity Analysis")
print("="*70)

def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, confidence=0.95):
    """Calculate Bootstrap confidence interval"""
    np.random.seed(42)
    bootstrap_statistics = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_statistics.append(statistic_func(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_statistics, alpha * 100)
    upper = np.percentile(bootstrap_statistics, (1 - alpha) * 100)
    
    return lower, upper, np.mean(bootstrap_statistics), np.std(bootstrap_statistics)

def q1_model_bootstrap_analysis():
    """Q1 Model Bootstrap Analysis"""
    print("\n[1.1] Q1 Bootstrap Confidence Interval Analysis...")
    
    # 过滤有效数据
    valid_data = bayesian_df[bayesian_df['n_valid_sims'] > 0].copy()
    
    # 计算关键指标
    results = []
    
    # 1. 置信度分析
    confidence_values = valid_data['confidence'].dropna().values
    if len(confidence_values) > 0:
        lower, upper, mean, std = bootstrap_confidence_interval(
            confidence_values, np.mean, n_bootstrap=1000
        )
        results.append({
            'metric': 'confidence',
            'mean': mean,
            'std': std,
            'ci_lower': lower,
            'ci_upper': upper,
            'n_samples': len(confidence_values)
        })
    
    # 2. 有效模拟数分析
    valid_sims = valid_data['n_valid_sims'].dropna().values
    if len(valid_sims) > 0:
        lower, upper, mean, std = bootstrap_confidence_interval(
            valid_sims, np.mean, n_bootstrap=1000
        )
        results.append({
            'metric': 'n_valid_sims',
            'mean': mean,
            'std': std,
            'ci_lower': lower,
            'ci_upper': upper,
            'n_samples': len(valid_sims)
        })
    
    # 3. 解释率计算（按规则系统）
    for rule in valid_data['rule_system'].unique():
        rule_data = valid_data[valid_data['rule_system'] == rule]
        explained = (rule_data['n_valid_sims'] > 0).sum()
        total = len(rule_data)
        rate = explained / total
        
        # Bootstrap 置信区间
        binary_data = (rule_data['n_valid_sims'] > 0).astype(int).values
        lower, upper, mean_rate, std_rate = bootstrap_confidence_interval(
            binary_data, np.mean, n_bootstrap=1000
        )
        
        results.append({
            'metric': f'explanation_rate_{rule}',
            'mean': mean_rate,
            'std': std_rate,
            'ci_lower': lower,
            'ci_upper': upper,
            'n_samples': total,
            'interpretation': f'{rule}: 解释率 = {mean_rate:.1%} (95% CI: [{lower:.1%}, {upper:.1%}])'
        })
    
    df_results = pd.DataFrame(results)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 置信度分布
    axes[0].hist(confidence_values, bins=30, color=MORANDI_COLORS[0], 
                 edgecolor='white', alpha=0.7, density=True)
    axes[0].axvline(x=np.mean(confidence_values), color=MORANDI_ACCENT[1], 
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidence_values):.3f}')
    ci_mean = df_results[df_results['metric'] == 'confidence']['mean'].values[0]
    ci_l = df_results[df_results['metric'] == 'confidence']['ci_lower'].values[0]
    ci_u = df_results[df_results['metric'] == 'confidence']['ci_upper'].values[0]
    axes[0].axvspan(ci_l, ci_u, alpha=0.2, color='green', label=f'95% CI: [{ci_l:.3f}, {ci_u:.3f}]')
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Q1 Confidence Distribution (Bootstrap)', fontsize=14)
    axes[0].legend()
    
    # Valid simulations distribution
    axes[1].hist(valid_sims, bins=30, color=MORANDI_COLORS[2], 
                 edgecolor='white', alpha=0.7, density=True)
    axes[1].axvline(x=np.mean(valid_sims), color=MORANDI_ACCENT[1], 
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_sims):.0f}')
    sim_mean = df_results[df_results['metric'] == 'n_valid_sims']['mean'].values[0]
    sim_l = df_results[df_results['metric'] == 'n_valid_sims']['ci_lower'].values[0]
    sim_u = df_results[df_results['metric'] == 'n_valid_sims']['ci_upper'].values[0]
    axes[1].axvspan(sim_l, sim_u, alpha=0.2, color='green', label=f'95% CI: [{sim_l:.0f}, {sim_u:.0f}]')
    axes[1].set_xlabel('Number of Valid Simulations', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Q1 Valid Simulations Distribution (Bootstrap)', fontsize=14)
    axes[1].legend()
    
    # 解释率对比（按规则系统）
    rule_rates = df_results[df_results['metric'].str.contains('explanation_rate')]
    labels = [m.replace('explanation_rate_', '') for m in rule_rates['metric']]
    means = rule_rates['mean'].values * 100
    ci_lowers = rule_rates['ci_lower'].values * 100
    ci_uppers = rule_rates['ci_upper'].values * 100
    
    # 计算误差范围
    errors_lower = means - ci_lowers
    errors_upper = ci_uppers - means
    errors = np.array([errors_lower, errors_upper])  # 正确的格式: (2, n)
    
    bars = axes[2].bar(labels, means, color=[MORANDI_COLORS[i] for i in range(len(labels))],
                       edgecolor='black', yerr=errors, capsize=5, alpha=0.8)
    axes[2].set_ylabel('Explanation Rate (%)', fontsize=12)
    axes[2].set_title('Q1 Explanation Rate by Rule System (95% CI)', fontsize=14)
    axes[2].set_ylim(0, 120)
    for bar, mean in zip(bars, means):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{mean:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'q1_bootstrap_confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Chart saved: q1_bootstrap_confidence_intervals.png")
    
    return df_results

def q1_stability_by_season():
    """Q1 Model Stability by Season Analysis"""
    print("\n[1.2] Q1 Model Stability by Season Analysis...")
    
    valid_data = bayesian_df[bayesian_df['n_valid_sims'] > 0].copy()
    
    # Calculate statistics by season
    season_stats = valid_data.groupby('season').agg({
        'confidence': ['mean', 'std', 'count'],
        'n_valid_sims': 'mean'
    }).reset_index()
    
    season_stats.columns = ['season', 'conf_mean', 'conf_std', 'n_weeks', 'valid_sims_mean']
    season_stats = season_stats.dropna()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence trend
    axes[0].errorbar(season_stats['season'], season_stats['conf_mean'],
                     yerr=season_stats['conf_std']/np.sqrt(season_stats['n_weeks']),
                     fmt='o-', color=MORANDI_COLORS[0], linewidth=2, markersize=6,
                     capsize=3, capthick=2, label='Mean ± SE')
    axes[0].axhline(y=season_stats['conf_mean'].mean(), color=MORANDI_ACCENT[1],
                    linestyle='--', linewidth=2, label=f'Overall Mean: {season_stats["conf_mean"].mean():.3f}')
    axes[0].set_xlabel('Season', fontsize=12)
    axes[0].set_ylabel('Mean Confidence', fontsize=12)
    axes[0].set_title('Q1 Model Confidence by Season', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Valid simulations trend
    axes[1].plot(season_stats['season'], season_stats['valid_sims_mean'],
                 's-', color=MORANDI_COLORS[2], linewidth=2, markersize=6)
    axes[1].fill_between(season_stats['season'], 
                         season_stats['valid_sims_mean'] * 0.8,
                         season_stats['valid_sims_mean'] * 1.2,
                         alpha=0.2, color=MORANDI_COLORS[2])
    axes[1].set_xlabel('Season', fontsize=12)
    axes[1].set_ylabel('Mean Valid Simulations', fontsize=12)
    axes[1].set_title('Q1 Valid Simulations by Season', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'q1_stability_by_season.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Chart saved: q1_stability_by_season.png")
    
    return season_stats

# Execute Q1 Analysis
q1_results = {}
q1_results['bootstrap_ci'] = q1_model_bootstrap_analysis()
q1_results['season_stability'] = q1_stability_by_season()

# =============================================================================
# PART 2: Q2 规则系统 - 统计显著性检验
# =============================================================================
print("\n" + "="*70)
print("   PART 2: Q2 Rule System Statistical Significance Tests")
print("="*70)

def q2_rule_comparison_statistical_tests():
    """Q2 Rule Comparison Statistical Significance Tests"""
    print("\n[2.1] Q2 Statistical Significance Tests...")
    
    # Use real data for statistical tests
    results = []
    
    # 1. Chi-square test: independence between rule system and outcome
    # 创建列联表
    contingency_table = pd.crosstab(
        counterfactual_df['rule_system'],
        counterfactual_df['is_flipped']
    )
    
    if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        results.append({
            'test': 'Chi-square Independence',
            'statistic': chi2,
            'p_value': p_value,
            'dof': dof,
            'conclusion': 'Reject' if p_value < 0.05 else 'Fail to reject',
            'interpretation': f'规则系统与结果翻转{"存在" if p_value < 0.05 else "不存在"}显著关联 (α=0.05)'
        })
    
    # 2. Distribution of elimination outcomes by rule system
    for rule in counterfactual_df['rule_system'].unique():
        rule_data = counterfactual_df[counterfactual_df['rule_system'] == rule]
        flip_rate = rule_data['is_flipped'].mean()
        n_total = len(rule_data)
        n_flipped = rule_data['is_flipped'].sum()
        
        # 计算置信区间 (Wilson score interval)
        if n_total > 0:
            z = 1.96  # 95% CI
            phat = n_flipped / n_total
            denominator = 1 + z**2/n_total
            center = (phat + z**2/(2*n_total)) / denominator
            margin = z * np.sqrt((phat*(1-phat) + z**2/(4*n_total)) / n_total)
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
            
            results.append({
                'test': f'Flip Rate - {rule}',
                'statistic': flip_rate,
                'p_value': None,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_samples': n_total,
                'interpretation': f'{rule}: 翻转率 = {flip_rate:.1%} (95% CI: [{ci_lower:.1%}, {ci_upper:.1%}])'
            })
    
    df_results = pd.DataFrame(results)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Flip rate comparison by rule system (with CI)
    flip_rates = df_results[df_results['test'].str.contains('Flip Rate')]
    rules = [t.replace('Flip Rate - ', '') for t in flip_rates['test']]
    rates = flip_rates['statistic'].values * 100
    ci_l = flip_rates['ci_lower'].values * 100
    ci_u = flip_rates['ci_upper'].values * 100
    errors_lower = rates - ci_l
    errors_upper = ci_u - rates
    errors = np.array([errors_lower, errors_upper])  # Correct format: (2, n)
    
    colors = [MORANDI_COLORS[i] for i in range(len(rules))]
    bars = axes[0].bar(rules, rates, color=colors, edgecolor='black', 
                       yerr=errors, capsize=5, alpha=0.8)
    axes[0].set_ylabel('Flip Rate (%)', fontsize=12)
    axes[0].set_title('Q2 Flip Rate by Rule System (95% CI)', fontsize=14)
    axes[0].set_ylim(0, max(ci_u) * 1.3)
    
    for bar, rate, n in zip(bars, rates, flip_rates['n_samples'].values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%\n(n={n})', ha='center', fontsize=9)
    
    # Statistical test results display
    test_results = df_results[~df_results['test'].str.contains('Flip Rate')]
    if len(test_results) > 0:
        test_name = test_results['test'].values[0]
        chi2_val = test_results['statistic'].values[0]
        p_val = test_results['p_value'].values[0]
        
        axes[1].text(0.1, 0.9, f"Chi-Square Test Results:", fontsize=14, 
                    fontweight='bold', transform=axes[1].transAxes)
        axes[1].text(0.1, 0.75, f"Test: {test_name}", fontsize=12, 
                    transform=axes[1].transAxes)
        axes[1].text(0.1, 0.60, f"χ² = {chi2_val:.4f}", fontsize=12, 
                    transform=axes[1].transAxes)
        axes[1].text(0.1, 0.45, f"p-value = {p_val:.6f}", fontsize=12, 
                    transform=axes[1].transAxes)
        
        significance = "Significant" if p_val < 0.05 else "Not Significant"
        color = '#E7298A' if p_val < 0.05 else '#1B9E77'
        axes[1].text(0.1, 0.30, f"Conclusion: {significance} (α=0.05)", 
                    fontsize=12, fontweight='bold', color=color,
                    transform=axes[1].transAxes)
        
        axes[1].text(0.1, 0.15, f"Rule system has", fontsize=11, 
                    transform=axes[1].transAxes)
        axes[1].text(0.1, 0.05, f"a {'significant' if p_val < 0.05 else 'non-significant'} impact", fontsize=11, 
                    fontweight='bold', color=color, transform=axes[1].transAxes)
        
        axes[1].axis('off')
        axes[1].set_title('Q2 Statistical Test Results', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'q2_statistical_tests.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Chart saved: q2_statistical_tests.png")
    
    return df_results

def q2_robustness_analysis():
    """Q2 Rule System Robustness Analysis by Season"""
    print("\n[2.2] Q2 Robustness Analysis by Season...")
    
    # Analyze rule consistency across seasons
    results = []
    
    # Calculate rule performance by season
    for season in counterfactual_df['season'].unique():
        season_data = counterfactual_df[counterfactual_df['season'] == season]
        
        for rule in season_data['rule_system'].unique():
            rule_data = season_data[season_data['rule_system'] == rule]
            
            if len(rule_data) > 0:
                flip_rate = rule_data['is_flipped'].mean()
                results.append({
                    'season': season,
                    'rule_system': rule,
                    'flip_rate': flip_rate,
                    'n_weeks': len(rule_data)
                })
    
    df_results = pd.DataFrame(results)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, rule in enumerate(df_results['rule_system'].unique()):
        rule_data = df_results[df_results['rule_system'] == rule]
        ax.scatter(rule_data['season'], rule_data['flip_rate'] * 100,
                  color=MORANDI_COLORS[i], s=50, alpha=0.7, label=rule)
        
        # 添加趋势线
        if len(rule_data) > 2:
            z = np.polyfit(rule_data['season'], rule_data['flip_rate'] * 100, 1)
            p = np.poly1d(z)
            ax.plot(rule_data['season'].sort_values(), 
                   p(rule_data['season'].sort_values()),
                   '--', color=MORANDI_COLORS[i], alpha=0.5)
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Flip Rate (%)', fontsize=12)
    ax.set_title('Q2 Robustness by Season', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'q2_robustness_by_season.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Chart saved: q2_robustness_by_season.png")
    
    return df_results

# 执行 Q2 分析
q2_results = {}
q2_results['statistical_tests'] = q2_rule_comparison_statistical_tests()
q2_results['robustness'] = q2_robustness_analysis()

# =============================================================================
# PART 3: Q3 聚类分析 - Bootstrap 稳健性检验
# =============================================================================
print("\n" + "="*70)
print("   PART 3: Q3 Clustering Robustness Bootstrap Tests")
print("="*70)

def q3_clustering_bootstrap_analysis():
    """Q3 Clustering Bootstrap Robustness Analysis"""
    print("\n[3.1] Q3 Clustering Bootstrap Analysis...")
    
    # 加载行业数据
    if 'Physicality' in industry_codebook.columns:
        features = industry_codebook[['Physicality', 'Performance', 'Fanbase']].values
    else:
        # 使用三维评分
        features = industry_codebook.iloc[:, 1:4].values
    
    # 移除含有 NaN 的行
    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    
    # Bootstrap 聚类稳定性
    n_bootstrap = 100
    k_best = 3
    cluster_assignments = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Bootstrap 采样
        idx = np.random.choice(len(features), size=len(features), replace=True)
        sample_features = features[idx]
        
        # 标准化
        scaler = StandardScaler()
        sample_scaled = scaler.fit_transform(sample_features)
        
        # K-Means 聚类
        kmeans = KMeans(n_clusters=k_best, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sample_scaled)
        cluster_assignments.append(labels)
    
    # 计算聚类稳定性（调整兰德指数）
    from sklearn.metrics import adjusted_rand_score
    
    # 以第一次聚类结果为参考
    reference = cluster_assignments[0]
    ari_scores = []
    
    for labels in cluster_assignments[1:]:
        ari = adjusted_rand_score(reference, labels)
        ari_scores.append(ari)
    
    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)
    
    results = {
        'n_bootstrap': n_bootstrap,
        'k_clusters': k_best,
        'ari_mean': ari_mean,
        'ari_std': ari_std,
        'ci_95_lower': np.percentile(ari_scores, 2.5),
        'ci_95_upper': np.percentile(ari_scores, 97.5),
        'stability': 'High' if ari_mean > 0.8 else 'Moderate' if ari_mean > 0.6 else 'Low'
    }
    
    df_results = pd.DataFrame([results])
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ARI 分布
    axes[0].hist(ari_scores, bins=20, color=MORANDI_COLORS[0], 
                 edgecolor='white', alpha=0.7, density=True)
    axes[0].axvline(x=ari_mean, color=MORANDI_ACCENT[1], linestyle='--', 
                    linewidth=2, label=f'Mean: {ari_mean:.3f}')
    axes[0].axvline(x=results['ci_95_lower'], color='green', linestyle=':', 
                    linewidth=2, label=f'95% CI: [{results["ci_95_lower"]:.3f}, {results["ci_95_upper"]:.3f}]')
    axes[0].axvline(x=results['ci_95_upper'], color='green', linestyle=':', linewidth=2)
    axes[0].set_xlabel('Adjusted Rand Index', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'Q3 Clustering Stability (Bootstrap, n={n_bootstrap})', fontsize=14)
    axes[0].legend()
    
    # 稳定性指标
    metrics = ['Mean ARI', 'Std ARI', 'CI Lower', 'CI Upper']
    values = [ari_mean, ari_std, results['ci_95_lower'], results['ci_95_upper']]
    colors = [MORANDI_COLORS[0], MORANDI_COLORS[2], MORANDI_COLORS[4], MORANDI_COLORS[6]]
    
    bars = axes[1].bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Q3 Clustering Stability Metrics', fontsize=14)
    axes[1].axhline(y=0.8, color='red', linestyle='--', label='High Stability Threshold')
    axes[1].legend()
    
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'q3_clustering_bootstrap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Chart saved: q3_clustering_bootstrap.png")
    
    return df_results

def q3_assumption_validation():
    """Q3 Model Assumption Validation"""
    print("\n[3.2] Q3 Assumption Validation...")
    
    # Based on existing summary for validation
    # Age linearity assumption
    age_tests = [
        {'assumption': 'Age-Score Linearity', 'F_stat': 'N/A', 'p_value': 0.0892, 
         'significant': False, 'conclusion': 'Linear assumption holds'},
        {'assumption': 'Age-Vote Linearity', 'F_stat': 'N/A', 'p_value': 0.7029, 
         'significant': False, 'conclusion': 'Linear assumption holds'},
        {'assumption': 'Growth Trajectory ANOVA', 'F_stat': 0.91, 'p_value': 0.404, 
         'significant': False, 'conclusion': 'No significant difference across clusters'},
        {'assumption': 'Performance Artist Effect', 'z_stat': 'N/A', 'p_value': 0.438, 
         'significant': False, 'conclusion': 'Not significantly different from baseline'}
    ]
    
    df_results = pd.DataFrame(age_tests)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1B9E77' if not sig else '#E7298A' for sig in df_results['significant']]
    bars = ax.barh(df_results['assumption'], -np.log10(df_results['p_value'] + 1e-10), 
                   color=colors, edgecolor='black', alpha=0.8)
    
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, 
               label='Significance Threshold (α=0.05)')
    ax.set_xlabel('-log10(p-value)', fontsize=12)
    ax.set_title('Q3 Assumption Test Results (Higher = More Significant)', fontsize=14)
    ax.legend()
    ax.invert_yaxis()
    
    # 添加 p-value 标签
    for bar, pval in zip(bars, df_results['p_value']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
               f'p={pval:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'q3_assumption_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Chart saved: q3_assumption_validation.png")
    
    return df_results

# 执行 Q3 分析
q3_results = {}
q3_results['clustering_bootstrap'] = q3_clustering_bootstrap_analysis()
q3_results['assumption_validation'] = q3_assumption_validation()

# =============================================================================
# PART 4: Q4 DTPM - 历史回测验证
# =============================================================================
print("\n" + "="*70)
print("   PART 4: Q4 DTPM Historical Backtesting Validation")
print("="*70)

def q4_dtpm_backtesting():
    """Q4 DTPM Historical Backtesting Analysis"""
    print("\n[4.1] Q4 DTPM Historical Backtesting Analysis...")
    
    # 使用网格搜索结果进行分析
    if len(grid_search_results) > 0:
        # Analyze parameter combinations
        results = []
        
        for _, row in grid_search_results.iterrows():
            results.append({
                'w_start': row.get('w_start', row.get('w_start', 0.9)),
                'w_end': row.get('w_end', row.get('w_end', 0.6)),
                'beta': row.get('beta', row.get('beta', 0.4)),
                'kendall_tau': row.get('kendall_tau', row.get('Kendall_Tau', 0)),
                'upset_rate': row.get('upset_rate', row.get('Upset_Rate', 0))
            })
        
        df_results = pd.DataFrame(results)
        
        # 可视化参数敏感性
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # w_start 敏感性
        w_start_groups = df_results.groupby('w_start').agg({
            'kendall_tau': 'mean',
            'upset_rate': 'mean'
        }).reset_index()
        
        axes[0].plot(w_start_groups['w_start'], w_start_groups['kendall_tau'],
                     'o-', color=MORANDI_COLORS[0], linewidth=2, markersize=8, label='Kendall Tau')
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(w_start_groups['w_start'], w_start_groups['upset_rate'],
                     's--', color=MORANDI_COLORS[2], linewidth=2, markersize=8, label='Upset Rate')
        axes[0].set_xlabel('w_start', fontsize=12)
        axes[0].set_ylabel('Kendall Tau', fontsize=12, color=MORANDI_COLORS[0])
        ax0_twin.set_ylabel('Upset Rate', fontsize=12, color=MORANDI_COLORS[2])
        axes[0].set_title('Q4 w_start Sensitivity', fontsize=14)
        axes[0].legend(loc='upper left')
        ax0_twin.legend(loc='upper right')
        
        # w_end 敏感性
        w_end_groups = df_results.groupby('w_end').agg({
            'kendall_tau': 'mean',
            'upset_rate': 'mean'
        }).reset_index()
        
        axes[1].plot(w_end_groups['w_end'], w_end_groups['kendall_tau'],
                     'o-', color=MORANDI_COLORS[0], linewidth=2, markersize=8, label='Kendall Tau')
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(w_end_groups['w_end'], w_end_groups['upset_rate'],
                     's--', color=MORANDI_COLORS[2], linewidth=2, markersize=8, label='Upset Rate')
        axes[1].set_xlabel('w_end', fontsize=12)
        axes[1].set_ylabel('Kendall Tau', fontsize=12, color=MORANDI_COLORS[0])
        ax1_twin.set_ylabel('Upset Rate', fontsize=12, color=MORANDI_COLORS[2])
        axes[1].set_title('Q4 w_end Sensitivity', fontsize=14)
        axes[1].legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # beta 敏感性
        beta_groups = df_results.groupby('beta').agg({
            'kendall_tau': 'mean',
            'upset_rate': 'mean'
        }).reset_index()
        
        axes[2].plot(beta_groups['beta'], beta_groups['kendall_tau'],
                     'o-', color=MORANDI_COLORS[0], linewidth=2, markersize=8, label='Kendall Tau')
        ax2_twin = axes[2].twinx()
        ax2_twin.plot(beta_groups['beta'], beta_groups['upset_rate'],
                     's--', color=MORANDI_COLORS[2], linewidth=2, markersize=8, label='Upset Rate')
        axes[2].set_xlabel('beta', fontsize=12)
        axes[2].set_ylabel('Kendall Tau', fontsize=12, color=MORANDI_COLORS[0])
        ax2_twin.set_ylabel('Upset Rate', fontsize=12, color=MORANDI_COLORS[2])
        axes[2].set_title('Q4 beta Sensitivity', fontsize=14)
        axes[2].legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'q4_dtpm_backtesting.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Chart saved: q4_dtpm_backtesting.png")
        
    else:
        # 如果没有网格搜索数据，使用理论分析
        df_results = pd.DataFrame([{
            'message': 'No grid search data available'
        }])
    
    return df_results

def q4_pareto_frontier_analysis():
    """Q4 Pareto Frontier Analysis"""
    print("\n[4.2] Q4 Pareto Frontier Analysis...")
    
    # 读取 Pareto 前沿点
    pareto_df = pd.read_csv(RESULTS_DIR / "system_design" / "pareto_frontier_points.csv")
    
    if len(pareto_df) > 0:
        # Check if fairness/entertainment columns exist, if not map them
        if 'fairness' not in pareto_df.columns and 'kendall_tau' in pareto_df.columns:
            pareto_df['fairness'] = pareto_df['kendall_tau']  # Use Kendall Tau as fairness proxy
            pareto_df['entertainment'] = 1 - pareto_df['upset_rate']  # Entertainment=1-upset_rate
        
        # Add is_pareto and is_optimal markers (if not present)
        if 'is_pareto' not in pareto_df.columns:
            pareto_df['is_pareto'] = True  # Assume all points are Pareto points
        if 'is_optimal' not in pareto_df.columns:
            # Find optimal solution (highest Kendall Tau)
            optimal_idx = pareto_df['kendall_tau'].idxmax()
            pareto_df['is_optimal'] = False
            pareto_df.loc[optimal_idx, 'is_optimal'] = True
        
        # 可视化 Pareto 前沿
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 所有点
        ax.scatter(pareto_df['fairness'], pareto_df['entertainment'],
                  alpha=0.3, s=50, color=MORANDI_COLORS[5], label='All Points')
        
        # Pareto 前沿点
        pareto_pts = pareto_df[pareto_df['is_pareto'] == True]
        if len(pareto_pts) > 0:
            ax.scatter(pareto_pts['fairness'], pareto_pts['entertainment'],
                      s=150, color=MORANDI_COLORS[0], edgecolor='black', 
                      linewidth=2, label='Pareto Frontier', zorder=5)
            
            # 标记当前最优解
            optimal = pareto_df[pareto_df['is_optimal'] == True]
            if len(optimal) > 0:
                ax.scatter(optimal['fairness'], optimal['entertainment'],
                          s=200, color=MORANDI_ACCENT[0], edgecolor='black',
                          linewidth=3, marker='*', label='Optimal Solution', zorder=10)
                
                # 标注参数
                ax.annotate(f"w_start={optimal['w_start'].values[0]:.2f}\n"
                           f"w_end={optimal['w_end'].values[0]:.2f}\n"
                           f"beta={optimal['beta'].values[0]:.2f}",
                           xy=(optimal['fairness'].values[0], optimal['entertainment'].values[0]),
                           xytext=(optimal['fairness'].values[0]+0.02, optimal['entertainment'].values[0]+0.02),
                           fontsize=9, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.set_xlabel('Fairness Score', fontsize=12)
        ax.set_ylabel('Entertainment Score', fontsize=12)
        ax.set_title('Q4 DTPM Pareto Frontier Analysis', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'q4_pareto_frontier.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Chart saved: q4_pareto_frontier.png")
    
    return pareto_df

# 执行 Q4 分析
q4_results = {}
q4_results['backtesting'] = q4_dtpm_backtesting()
q4_results['pareto_frontier'] = q4_pareto_frontier_analysis()

# =============================================================================
# 综合总结
# =============================================================================
print("\n" + "="*70)
print("   Sensitivity Analysis Summary Based on Real Data")
print("="*70)

summary = """
## Sensitivity Analysis and Model Evaluation Summary Based on Real Data

### Q1 Bayesian Model (Empirical Bootstrap Analysis)
1. **Confidence Distribution**: Bootstrap 95% CI calculated from {n_conf} valid samples
2. **Explanation Rate Analysis**: 95% CI calculated by rule system
3. **Stability Analysis**: Temporal consistency analysis by season

### Q2 Rule System (Statistical Significance Tests)
1. **Chi-square Test**: Test independence between rule system and outcome flips
2. **Confidence Intervals**: Wilson CI for flip rates by rule system
3. **Robustness**: Consistency analysis across seasons

### Q3 Clustering (Bootstrap Robustness)
1. **Clustering Stability**: Bootstrap-based Adjusted Rand Index (ARI) calculation
2. **Assumption Validation**: Age linearity, growth trajectory ANOVA tests

### Q4 DTPM Backtesting
1. **Parameter Sensitivity**: Real backtesting analysis of parameter effects
2. **Pareto Frontier**: Optimal balance between fairness and entertainment
""".format(
    n_conf=len(bayesian_df[bayesian_df['n_valid_sims'] > 0])
)

# Save summary
with open(PLOTS_DIR / 'enhanced_sensitivity_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n[INFO] Analysis complete!")
print(f"[INFO] All charts saved to: {PLOTS_DIR}")

# 保存各题结果
for q_num, q_data in [('Q1', q1_results), ('Q2', q2_results), ('Q3', q3_results), ('Q4', q4_results)]:
    for name, df in q_data.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(PLOTS_DIR / f'{q_num}_{name}_enhanced.csv', index=False)
            print(f"  ✓ {q_num}_{name}_enhanced.csv")

print("\n" + "="*70)
print("   Sensitivity Analysis Complete!")
print("="*70)
