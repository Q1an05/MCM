#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Task 3 OPTIMIZED: Advanced Factor Analysis with K-Means Clustering & Interaction Effects
-----------------------------------------------------------------------------------------

Key Improvements over v1:
1. Industry Codebook: 3-dimensional scoring (Physicality, Performance, Fanbase)
2. K-Means Clustering: Data-driven industry classification (eliminates subjectivity)
3. Growth Trajectory Analysis: Individual growth slopes + ANOVA
4. Dual-Track LMM with Interactions: Class×Week interaction terms
5. Enhanced Visualizations: Growth trajectories, Forest plots

Author: MCM Team
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
from viz_config import *
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = BASE_DIR / "data_raw" / "2026_MCM_Problem_C_Data.csv"
BAYESIAN_RESULTS_PATH = BASE_DIR / "results" / "full_simulation_bayesian.csv"
OUTPUT_DIR = BASE_DIR / "results" / "plots" / "question3"
OUTPUT_DATA_DIR = BASE_DIR / "results" / "question3"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STEP 1: Industry Codebook (3-Dimensional Scoring)
# =============================================================================
# Each industry is scored on 3 dimensions (1-5 scale):
# - Physicality: Physical training/body control requirements
# - Performance: Stage presence, camera experience, emotional expression
# - Fanbase: Built-in fan loyalty, social media following, narrative potential

INDUSTRY_CODEBOOK = {
    # Format: 'industry': (Physicality, Performance, Fanbase)
    # Physicality: Physical training/body control (1-5)
    # Performance: Stage presence, camera experience (1-5)
    # Fanbase: Built-in fan loyalty, social media following (1-5)
    
    # ═══════════════════════════════════════════════════════════════
    # HIGH PHYSICALITY (Athletes & Physical Performers)
    # ═══════════════════════════════════════════════════════════════
    'Athlete': (5, 2, 3),               # 95人 - 运动员，体能强但表演弱
    'Racing Driver': (4, 2, 3),         # 4人 - 赛车手，需要反应和耐力
    'Fitness Instructor': (5, 3, 2),    # 1人 - 健身教练，高体能+一定表演
    'Military': (5, 1, 2),              # 1人 - 军人，高体能但无表演经验
    'Astronaut': (4, 2, 4),             # 1人 - 宇航员，高体能+高公众关注度
    
    # ═══════════════════════════════════════════════════════════════
    # HIGH PERFORMANCE (Entertainers & Artists)
    # ═══════════════════════════════════════════════════════════════
    'Actor/Actress': (2, 5, 3),         # 128人 - 演员，表演能力最强
    'Singer/Rapper': (3, 5, 4),         # 61人 - 歌手，表演强+粉丝多
    'Musician': (2, 4, 3),              # 1人 - 音乐家（器乐为主）
    'Comedian': (1, 5, 3),              # 12人 - 喜剧演员，纯表演型
    'Magician': (2, 5, 2),              # 1人 - 魔术师，表演技巧强
    'Producer': (1, 3, 2),              # 1人 - 制片人，幕后为主
    'Fashion Designer': (1, 4, 3),      # 1人 - 时装设计师，有审美和舞台感
    'Motivational Speaker': (1, 4, 3),  # 1人 - 励志演说家，舞台经验丰富
    
    # ═══════════════════════════════════════════════════════════════
    # HIGH FANBASE (Reality Stars & Social Media)
    # ═══════════════════════════════════════════════════════════════
    'TV Personality': (1, 3, 5),        # 67人 - 电视名人，流量最高
    'Social Media Personality': (1, 3, 5),  # 8人 - 网红，纯流量型
    'Social media personality': (1, 3, 5),  # 1人 - 同上（小写版本）
    
    # ═══════════════════════════════════════════════════════════════
    # MIXED / BALANCED
    # ═══════════════════════════════════════════════════════════════
    'Model': (3, 4, 3),                 # 17人 - 模特，体态+镜头感
    'Beauty Pagent': (3, 4, 2),         # 1人 - 选美冠军
    'News Anchor': (1, 4, 2),           # 3人 - 新闻主播，镜头经验
    'Journalist': (1, 3, 2),            # 1人 - 记者
    'Sports Broadcaster': (1, 3, 2),    # 2人 - 体育解说员
    'Radio Personality': (1, 3, 3),     # 4人 - 电台主持人
    
    # ═══════════════════════════════════════════════════════════════
    # LOW ON ALL / OUTSIDERS
    # ═══════════════════════════════════════════════════════════════
    'Politician': (1, 3, 1),            # 3人 - 政客，最低流量
    'Entrepreneur': (1, 2, 2),          # 4人 - 企业家
    'Conservationist': (2, 2, 3),       # 1人 - 环保主义者（如动物专家）
    'Con artist': (1, 4, 2),            # 1人 - 骗子（话术型，高表演）
    
    # ═══════════════════════════════════════════════════════════════
    # DEFAULT (Unknown)
    # ═══════════════════════════════════════════════════════════════
    'Other': (2, 3, 3),                 # 未知类型的默认值
}

# Cluster naming based on centroid characteristics
CLUSTER_NAMES = {
    0: 'Athletic Elite',      # High Physicality
    1: 'Performance Artist',  # High Performance
    2: 'Fan Favorite',        # High Fanbase
    3: 'Underdog'             # Low on all / Balanced low
}


def map_industry_to_scores(industry: str) -> tuple:
    """Map raw industry string to 3D scores using codebook."""
    if pd.isna(industry):
        return INDUSTRY_CODEBOOK['Other']
    
    industry = str(industry).strip()
    
    # Direct match
    if industry in INDUSTRY_CODEBOOK:
        return INDUSTRY_CODEBOOK[industry]
    
    # Fuzzy matching
    industry_lower = industry.lower()
    
    if 'actor' in industry_lower or 'actress' in industry_lower:
        return INDUSTRY_CODEBOOK['Actor/Actress']
    elif 'athlete' in industry_lower or 'nba' in industry_lower or 'nfl' in industry_lower or 'olympian' in industry_lower:
        return INDUSTRY_CODEBOOK['Athlete']
    elif 'singer' in industry_lower or 'rapper' in industry_lower or 'musician' in industry_lower:
        return INDUSTRY_CODEBOOK['Singer/Rapper']
    elif 'model' in industry_lower:
        return INDUSTRY_CODEBOOK['Model']
    elif 'reality' in industry_lower or 'bachelor' in industry_lower or 'housewife' in industry_lower:
        return INDUSTRY_CODEBOOK['TV Personality']
    elif 'tv' in industry_lower or 'personality' in industry_lower or 'host' in industry_lower:
        return INDUSTRY_CODEBOOK['TV Personality']
    elif 'news' in industry_lower or 'anchor' in industry_lower:
        return INDUSTRY_CODEBOOK['News Anchor']
    elif 'radio' in industry_lower:
        return INDUSTRY_CODEBOOK['Radio Personality']
    elif 'sport' in industry_lower and 'broadcast' in industry_lower:
        return INDUSTRY_CODEBOOK['Sports Broadcaster']
    elif 'racing' in industry_lower or 'driver' in industry_lower:
        return INDUSTRY_CODEBOOK['Racing Driver']
    elif 'comedian' in industry_lower or 'comic' in industry_lower:
        return INDUSTRY_CODEBOOK['Comedian']
    elif 'magician' in industry_lower:
        return INDUSTRY_CODEBOOK['Magician']
    elif 'politic' in industry_lower:
        return INDUSTRY_CODEBOOK['Politician']
    elif 'entrepreneur' in industry_lower or 'business' in industry_lower:
        return INDUSTRY_CODEBOOK['Entrepreneur']
    elif 'pageant' in industry_lower or 'beauty' in industry_lower:
        return INDUSTRY_CODEBOOK['Beauty Pagent']
    else:
        return INDUSTRY_CODEBOOK['Other']


def perform_kmeans_clustering(df: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
    """
    Perform K-Means clustering on industry scores.
    
    If n_clusters is None, automatically determine optimal K using:
    1. Elbow Method (SSE / Inertia)
    2. Silhouette Score
    
    Returns DataFrame with cluster assignments.
    """
    print("\n" + "="*60)
    print("STEP 1: K-MEANS INDUSTRY CLUSTERING")
    print("="*60)
    
    # Extract unique industries and their scores
    industries = df['celebrity_industry'].unique()
    
    industry_scores = []
    for ind in industries:
        phys, perf, fan = map_industry_to_scores(ind)
        industry_scores.append({
            'industry': ind,
            'physicality': phys,
            'performance': perf,
            'fanbase': fan
        })
    
    df_ind = pd.DataFrame(industry_scores)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_ind[['physicality', 'performance', 'fanbase']])
    
    # ========================================
    # AUTO-DETERMINE OPTIMAL K
    # ========================================
    if n_clusters is None:
        print("\n[INFO] Determining optimal K using Elbow Method & Silhouette Score...")
        
        k_range = range(2, min(8, len(df_ind)))  # Test K from 2 to 7 (or max possible)
        inertias = []
        silhouettes = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            sil = silhouette_score(X, km.labels_)
            silhouettes.append(sil)
            print(f"  K={k}: Inertia={km.inertia_:.2f}, Silhouette={sil:.4f}")
        
        # Find optimal K by silhouette score (higher is better)
        optimal_k_sil = list(k_range)[np.argmax(silhouettes)]
        
        # Elbow detection: find the "knee" point
        # Use second derivative approximation
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            optimal_k_elbow = list(k_range)[np.argmax(diffs2) + 1]
        else:
            optimal_k_elbow = optimal_k_sil
        
        print(f"\n[RESULT] Optimal K by Silhouette Score: {optimal_k_sil} (score={max(silhouettes):.4f})")
        print(f"[RESULT] Optimal K by Elbow Method: {optimal_k_elbow}")
        
        # Use silhouette-based K as it's more reliable for small datasets
        n_clusters = optimal_k_sil
        print(f"\n[DECISION] Using K={n_clusters} (based on Silhouette Score)")
        
        # Visualize K selection
        visualize_k_selection(k_range, inertias, silhouettes, n_clusters)
    
    # K-Means clustering with optimal K
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_ind['cluster'] = kmeans.fit_predict(X)
    
    # Analyze cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    print("\n[INFO] Cluster Centers (Original Scale):")
    print(f"{'Cluster':<20} {'Physicality':>12} {'Performance':>12} {'Fanbase':>10}")
    print("-"*60)
    
    # Assign meaningful names based on dominant dimension
    # Use a counter to handle duplicate names
    name_counts = {}
    cluster_mapping = {}
    
    for i, center in enumerate(centers):
        phys, perf, fan = center
        max_dim = np.argmax(center)
        
        if max_dim == 0:
            base_name = 'Athletic Elite'
        elif max_dim == 1:
            base_name = 'Performance Artist'
        elif max_dim == 2:
            base_name = 'Fan Favorite'
        else:
            base_name = 'Balanced'
        
        # Handle low-all case
        if np.max(center) < 3:
            base_name = 'Underdog'
        
        # Handle duplicate names by adding suffix
        if base_name in name_counts:
            name_counts[base_name] += 1
            # Use secondary dimension to differentiate
            dims = ['Physical', 'Artistic', 'Popular']
            sorted_dims = np.argsort(center)[::-1]  # Descending order
            secondary = dims[sorted_dims[1]]
            name = f'{base_name} ({secondary})'
        else:
            name_counts[base_name] = 1
            name = base_name
            
        cluster_mapping[i] = name
        print(f"{name:<25} {phys:>12.2f} {perf:>12.2f} {fan:>10.2f}")
    
    df_ind['cluster_name'] = df_ind['cluster'].map(cluster_mapping)
    
    # Create mapping dictionary
    industry_to_cluster = dict(zip(df_ind['industry'], df_ind['cluster_name']))
    
    # Apply to main dataframe
    df['industry_cluster'] = df['celebrity_industry'].map(industry_to_cluster)
    df['industry_cluster'] = df['industry_cluster'].fillna('Underdog')
    
    # Show distribution
    print("\n[INFO] Cluster Distribution:")
    print(df['industry_cluster'].value_counts())
    
    # Save codebook
    df_ind.to_csv(OUTPUT_DATA_DIR / 'industry_codebook.csv', index=False)
    print(f"\n[INFO] Codebook saved to {OUTPUT_DATA_DIR / 'industry_codebook.csv'}")
    
    # Visualize clustering
    visualize_clusters(df_ind, centers, scaler, n_clusters)
    
    return df


def visualize_k_selection(k_range, inertias, silhouettes, chosen_k):
    """Visualize the elbow method and silhouette scores for K selection."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Elbow Method
    ax1 = axes[0]
    ax1.plot(list(k_range), inertias, 'o-', linewidth=2.5, markersize=8, 
             color=MORANDI_ACCENT[1], markerfacecolor=MORANDI_ACCENT[1], 
             markeredgecolor='white', markeredgewidth=1.5)
    ax1.axvline(x=chosen_k, color=MORANDI_ACCENT[0], linestyle='--', linewidth=2, 
                label=f'Chosen K={chosen_k}', alpha=0.8)
    style_axes(ax1, 
               title='Elbow Method for Optimal K',
               xlabel='Number of Clusters (K)', 
               ylabel='Inertia (SSE)')
    ax1.legend(framealpha=0.9)
    
    # Plot 2: Silhouette Score
    ax2 = axes[1]
    bars = ax2.bar(list(k_range), silhouettes, 
                   color=MORANDI_COLORS[1], alpha=0.7, 
                   edgecolor='white', linewidth=1.2)
    
    # Highlight the best K
    best_idx = list(k_range).index(chosen_k)
    bars[best_idx].set_color(MORANDI_ACCENT[2])  # Mint green for best
    bars[best_idx].set_alpha(0.95)
    
    ax2.axhline(y=max(silhouettes), color=MORANDI_ACCENT[2], linestyle='--', alpha=0.7, linewidth=1.5)
    style_axes(ax2,
               title='Silhouette Score for Different K',
               xlabel='Number of Clusters (K)',
               ylabel='Silhouette Score')
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for i, (k, s) in enumerate(zip(k_range, silhouettes)):
        ax2.text(k, s + 0.02, f'{s:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    save_figure(fig, OUTPUT_DIR / 'q3_optimal_k_selection.png')
    print(f"[INFO] K selection visualization saved to {OUTPUT_DIR / 'q3_optimal_k_selection.png'}")


def visualize_clusters(df_ind: pd.DataFrame, centers: np.ndarray, scaler: StandardScaler, n_clusters: int):
    """Visualize the K-Means clustering results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use Morandi palette
    unique_clusters = df_ind['cluster_name'].unique()
    colors = {name: get_color(i) for i, name in enumerate(unique_clusters)}
    
    ax1 = axes[0]
    for cluster_name in unique_clusters:
        subset = df_ind[df_ind['cluster_name'] == cluster_name]
        ax1.scatter(subset['performance'], subset['fanbase'], 
                   c=colors.get(cluster_name, MORANDI_COLORS[0]), label=cluster_name, 
                   s=120, alpha=0.75, edgecolors='white', linewidth=1.5)
    
    style_axes(ax1,
               title=f'Industry Clustering (K={n_clusters}): Performance vs Fanbase',
               xlabel='Performance Score',
               ylabel='Fanbase Score')
    ax1.legend(framealpha=0.9)
    
    # Plot 2: Bar chart of cluster centers
    ax2 = axes[1]
    x = np.arange(3)
    width = 0.8 / n_clusters
    
    # Map cluster indices to names (sorted by cluster number for consistency)
    cluster_name_list = list(unique_clusters)
    
    for i, center in enumerate(centers):
        name = cluster_name_list[i] if i < len(cluster_name_list) else f'Cluster {i}'
        ax2.bar(x + i*width, center, width, label=name, 
                color=colors.get(name, get_color(i)), alpha=0.85, 
                edgecolor='white', linewidth=1)
    
    ax2.set_xticks(x + width * (n_clusters - 1) / 2)
    ax2.set_xticklabels(['Physicality', 'Performance', 'Fanbase'], fontweight='bold')
    style_axes(ax2,
               title=f'Cluster Profiles (K={n_clusters})',
               ylabel='Score (1-5)')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(0, 5.5)
    
    save_figure(fig, OUTPUT_DIR / 'q3_industry_clustering.png')
    print(f"[INFO] Clustering visualization saved to {OUTPUT_DIR / 'q3_industry_clustering.png'}")


# =============================================================================
# STEP 2: Growth Trajectory Analysis
# =============================================================================
def compute_growth_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute individual growth slopes for contestants who survived ≥4 weeks.
    """
    print("\n" + "="*60)
    print("STEP 2: GROWTH TRAJECTORY ANALYSIS")
    print("="*60)
    
    # Group by contestant (season + celebrity_name)
    growth_data = []
    
    for (season, name), group in df.groupby(['season', 'celebrity_name']):
        weeks_survived = group['week'].nunique()
        
        if weeks_survived >= 4:
            # Simple linear regression: Score ~ Week
            X = group['week'].values
            y = group['normalized_score'].values
            
            # Fit regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            
            growth_data.append({
                'season': season,
                'celebrity_name': name,
                'weeks_survived': weeks_survived,
                'growth_slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'industry_cluster': group['industry_cluster'].iloc[0],
                'age': group['age'].iloc[0] if 'age' in group.columns else np.nan
            })
    
    df_growth = pd.DataFrame(growth_data)
    print(f"[INFO] Computed growth slopes for {len(df_growth)} contestants (≥4 weeks)")
    
    # Summary by cluster
    print("\n[INFO] Average Growth Slope by Cluster:")
    summary = df_growth.groupby('industry_cluster')['growth_slope'].agg(['mean', 'std', 'count'])
    print(summary.round(4))
    
    # ANOVA test
    print("\n[INFO] One-way ANOVA: Growth Slope ~ Industry Cluster")
    clusters = df_growth['industry_cluster'].unique()
    groups = [df_growth[df_growth['industry_cluster'] == c]['growth_slope'].values for c in clusters]
    
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("  ✓ SIGNIFICANT: Industry cluster significantly affects growth rate!")
    else:
        print("  ✗ Not significant at α=0.05")
    
    # Save results
    df_growth.to_csv(OUTPUT_DATA_DIR / 'growth_trajectories.csv', index=False)
    
    # Visualize
    visualize_growth_trajectories(df, df_growth)
    
    return df_growth


def visualize_growth_trajectories(df: pd.DataFrame, df_growth: pd.DataFrame):
    """Visualize growth trajectories by cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'Athletic Elite': '#3498db', 'Performance Artist': '#e74c3c', 
              'Fan Favorite': '#2ecc71', 'Underdog': '#95a5a6'}
    
    # Plot 1: Average score trajectory by cluster
    ax1 = axes[0]
    
    for cluster in df['industry_cluster'].unique():
        subset = df[df['industry_cluster'] == cluster]
        weekly_avg = subset.groupby('week')['normalized_score'].mean()
        
        # Fit trend line
        X = weekly_avg.index.values
        y = weekly_avg.values
        z = np.polyfit(X, y, 1)
        p = np.poly1d(z)
        
        ax1.scatter(X, y, c=colors.get(cluster, '#333'), alpha=0.5, s=50)
        ax1.plot(X, p(X), c=colors.get(cluster, '#333'), linewidth=2.5, label=f'{cluster} (slope={z[0]:.3f})')
    
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('Normalized Judge Score', fontsize=12)
    ax1.set_title('Score Trajectories by Industry Cluster', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of growth slopes
    ax2 = axes[1]
    
    cluster_order = ['Athletic Elite', 'Performance Artist', 'Fan Favorite', 'Underdog']
    cluster_order = [c for c in cluster_order if c in df_growth['industry_cluster'].unique()]
    
    bp = ax2.boxplot([df_growth[df_growth['industry_cluster'] == c]['growth_slope'].values 
                      for c in cluster_order],
                     labels=cluster_order, patch_artist=True)
    
    for patch, cluster in zip(bp['boxes'], cluster_order):
        patch.set_facecolor(colors.get(cluster, '#333'))
        patch.set_alpha(0.7)
    
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Growth Slope (Δ Score / Week)', fontsize=12)
    ax2.set_title('Individual Growth Rates by Cluster', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'q3_growth_trajectories.png', dpi=300)
    print(f"[INFO] Growth trajectories saved to {OUTPUT_DIR / 'q3_growth_trajectories.png'}")


# =============================================================================
# STEP 2.5: Age Linearity Verification
# =============================================================================
def verify_age_linearity(df: pd.DataFrame) -> dict:
    """
    Compare linear vs quadratic age effects to validate linearity assumption.
    
    Tests whether age has a linear or non-linear (quadratic) relationship
    with judge scores and fan share.
    """
    print("\n" + "="*60)
    print("STEP 2.5: AGE LINEARITY VERIFICATION")
    print("="*60)
    
    # Create age squared term
    df['age_sq'] = df['age_std'] ** 2
    
    results = {}
    
    # Model A: Judge Score
    print("\n[TEST A] Judge Score ~ Age (Linear vs Quadratic)")
    
    # Linear model
    model_linear_skill = smf.mixedlm(
        "normalized_score ~ age_std",
        data=df,
        groups=df['ballroom_partner']
    ).fit(reml=True)
    
    # Quadratic model
    model_quad_skill = smf.mixedlm(
        "normalized_score ~ age_std + age_sq",
        data=df,
        groups=df['ballroom_partner']
    ).fit(reml=True)
    
    # Extract p-value for quadratic term
    p_quad_skill = model_quad_skill.pvalues.get('age_sq', 1.0)
    
    print(f"  Linear model Log-Likelihood: {model_linear_skill.llf:.2f}")
    print(f"  Quadratic model Log-Likelihood: {model_quad_skill.llf:.2f}")
    print(f"  Quadratic term (age²) coefficient: {model_quad_skill.fe_params.get('age_sq', 0):.4f}")
    print(f"  Quadratic term P-value: {p_quad_skill:.4f}")
    
    if p_quad_skill < 0.05:
        print("  → Quadratic term SIGNIFICANT: Consider non-linear age effect")
        results['skill_model'] = 'quadratic'
    else:
        print("  → Quadratic term NOT significant: Linear model is appropriate ✓")
        results['skill_model'] = 'linear'
    
    # Model B: Fan Share
    print("\n[TEST B] Fan Share ~ Age (Linear vs Quadratic)")
    
    # Linear model
    model_linear_pop = smf.mixedlm(
        "log_rfs ~ age_std",
        data=df,
        groups=df['ballroom_partner']
    ).fit(reml=True)
    
    # Quadratic model
    model_quad_pop = smf.mixedlm(
        "log_rfs ~ age_std + age_sq",
        data=df,
        groups=df['ballroom_partner']
    ).fit(reml=True)
    
    p_quad_pop = model_quad_pop.pvalues.get('age_sq', 1.0)
    
    print(f"  Linear model Log-Likelihood: {model_linear_pop.llf:.2f}")
    print(f"  Quadratic model Log-Likelihood: {model_quad_pop.llf:.2f}")
    print(f"  Quadratic term (age²) coefficient: {model_quad_pop.fe_params.get('age_sq', 0):.4f}")
    print(f"  Quadratic term P-value: {p_quad_pop:.4f}")
    
    if p_quad_pop < 0.05:
        print("  → Quadratic term SIGNIFICANT: Consider non-linear age effect")
        results['popularity_model'] = 'quadratic'
    else:
        print("  → Quadratic term NOT significant: Linear model is appropriate ✓")
        results['popularity_model'] = 'linear'
    
    # Save comparison results
    comparison_df = pd.DataFrame({
        'Model': ['Technical (Judge Score)', 'Popularity (Fan Share)'],
        'Linear_LogLik': [model_linear_skill.llf, model_linear_pop.llf],
        'Quadratic_LogLik': [model_quad_skill.llf, model_quad_pop.llf],
        'Age_Sq_Coef': [model_quad_skill.fe_params.get('age_sq', 0), 
                        model_quad_pop.fe_params.get('age_sq', 0)],
        'Age_Sq_Pvalue': [p_quad_skill, p_quad_pop],
        'Recommended': [results['skill_model'], results['popularity_model']]
    })
    comparison_df.to_csv(OUTPUT_DATA_DIR / 'age_linearity_comparison.csv', index=False)
    print(f"\n[INFO] Comparison saved to {OUTPUT_DATA_DIR / 'age_linearity_comparison.csv'}")
    
    # Visualization
    visualize_age_effects(df, results)
    
    return results


def visualize_age_effects(df: pd.DataFrame, linearity_results: dict):
    """Visualize age effects with linear vs quadratic fits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Aggregate by age for cleaner visualization
    age_skill = df.groupby('age')['normalized_score'].agg(['mean', 'std', 'count']).reset_index()
    age_pop = df.groupby('age')['log_rfs'].agg(['mean', 'std', 'count']).reset_index()
    
    # Filter for sufficient sample size
    age_skill = age_skill[age_skill['count'] >= 5]
    age_pop = age_pop[age_pop['count'] >= 5]
    
    # Plot 1: Age vs Judge Score
    ax1 = axes[0]
    ax1.scatter(age_skill['age'], age_skill['mean'], s=age_skill['count']*3, 
                alpha=0.6, c='steelblue', edgecolors='navy')
    
    # Fit lines
    if len(age_skill) > 2:
        z_linear = np.polyfit(age_skill['age'], age_skill['mean'], 1)
        z_quad = np.polyfit(age_skill['age'], age_skill['mean'], 2)
        x_line = np.linspace(age_skill['age'].min(), age_skill['age'].max(), 100)
        
        ax1.plot(x_line, np.polyval(z_linear, x_line), 'b-', linewidth=2, label='Linear Fit')
        ax1.plot(x_line, np.polyval(z_quad, x_line), 'r--', linewidth=2, label='Quadratic Fit')
    
    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel('Mean Normalized Score', fontsize=12)
    ax1.set_title(f'Age vs Technical Skill\n(Recommended: {linearity_results.get("skill_model", "linear")})', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Age vs Fan Share
    ax2 = axes[1]
    ax2.scatter(age_pop['age'], age_pop['mean'], s=age_pop['count']*3, 
                alpha=0.6, c='coral', edgecolors='darkred')
    
    if len(age_pop) > 2:
        z_linear = np.polyfit(age_pop['age'], age_pop['mean'], 1)
        z_quad = np.polyfit(age_pop['age'], age_pop['mean'], 2)
        x_line = np.linspace(age_pop['age'].min(), age_pop['age'].max(), 100)
        
        ax2.plot(x_line, np.polyval(z_linear, x_line), 'b-', linewidth=2, label='Linear Fit')
        ax2.plot(x_line, np.polyval(z_quad, x_line), 'r--', linewidth=2, label='Quadratic Fit')
    
    ax2.set_xlabel('Age', fontsize=12)
    ax2.set_ylabel('Mean Log(Relative Fan Share)', fontsize=12)
    ax2.set_title(f'Age vs Popularity\n(Recommended: {linearity_results.get("popularity_model", "linear")})', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'q3_age_linearity_test.png', dpi=300)
    print(f"[INFO] Age linearity visualization saved to {OUTPUT_DIR / 'q3_age_linearity_test.png'}")


# =============================================================================
# STEP 2.6: Blind Voting Effect Analysis
# =============================================================================
def analyze_blind_voting_effect(df: pd.DataFrame) -> dict:
    """
    Analyze the "Blind Voting" effect on the decoupling between judge scores and fan votes.
    
    Background:
    -----------
    "Blind Voting" refers to the phenomenon where West Coast viewers voted BEFORE
    seeing the live performance (due to time zone delays). This was a structural
    issue in DWTS until live voting was introduced.
    
    Proxy Variable Construction:
    ----------------------------
    Since raw data doesn't contain explicit West Coast timezone flags, we use:
    - **D_Blind = 1**: Seasons 28, 29, 30 (Live Vote era, but West Coast still has 3-hour delay)
      - These seasons introduced live voting but time zone issues persist
    - **D_Blind = 0**: All other seasons (S1-S27: delayed broadcast voting, S31+: if any)
    
    Model Formula (Popularity Track):
    Y_Fan = α₀ + α₁·Score + α₂·(Score × D_Blind) + α₃·Rep + α₄·(Rep × D_Blind) + Σαₖ·Class + v_Partner
    
    Key Interactions:
    - Score × D_Blind: Tests if current performance matters less during blind voting
    - Rep × D_Blind: Tests if reputation (prior impression) substitutes for current performance
    """
    print("\n" + "="*60)
    print("STEP 2.6: BLIND VOTING EFFECT ANALYSIS")
    print("="*60)
    
    df = df.copy()
    
    # =========================================
    # NEW DEFINITION: D_Blind for S28-S30 (Live Vote Era with Time Zone Issues)
    # =========================================
    # Rationale: S28-S30 introduced live voting, but West Coast viewers
    # still vote during the show (3-hour delay), potentially before seeing performances.
    # This is the "residual blind voting" period.
    
    df['D_Blind'] = df['season'].apply(lambda s: 1 if s in [28, 29, 30] else 0)
    df['blind_label'] = df['D_Blind'].map({1: 'Blind Era (S28-S30)', 0: 'Non-Blind Era'})
    
    print("\n[INFO] Blind Voting Definition (D_Blind):")
    print("  - D_Blind = 1: Seasons 28, 29, 30 (Live Vote with West Coast time delay)")
    print("  - D_Blind = 0: Other seasons (S1-S27 delayed broadcast, S31+ if any)")
    print(f"\n[INFO] Sample Distribution:")
    print(df.groupby('blind_label').size())
    
    # Check if we have reputation variable
    if 'reputation' not in df.columns:
        print("[WARNING] 'reputation' column not found, creating from lagged scores...")
        # Create reputation as lagged cumulative average score
        df = df.sort_values(['season', 'celebrity_name', 'week'])
        df['reputation'] = df.groupby(['season', 'celebrity_name'])['normalized_score'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['reputation'] = df['reputation'].fillna(df['normalized_score'])
    
    # Calculate correlation by era
    results = {}
    
    for era in [0, 1]:
        subset = df[df['D_Blind'] == era]
        if len(subset) > 10:
            corr_score = subset['normalized_score'].corr(subset['log_rfs'])
            corr_rep = subset['reputation'].corr(subset['log_rfs'])
            results[era] = {
                'n': len(subset),
                'corr_score': corr_score,
                'corr_rep': corr_rep,
                'label': 'Blind Era (S28-S30)' if era == 1 else 'Non-Blind Era'
            }
    
    print("\n[INFO] Score-Vote Correlation by Era:")
    for era, data in results.items():
        print(f"  {data['label']}:")
        print(f"    Score → FanVote: r = {data['corr_score']:.4f}")
        print(f"    Reputation → FanVote: r = {data['corr_rep']:.4f}")
        print(f"    (n={data['n']})")
    
    # =========================================
    # FULL INTERACTION MODEL
    # =========================================
    # Y_Fan = α₀ + α₁·Score + α₂·(Score×D_Blind) + α₃·Rep + α₄·(Rep×D_Blind) + Cluster + (1|Partner)
    print("\n[INFO] Fitting Full Interaction Model:")
    print("  log(FanShare) ~ Score + Score×D_Blind + Reputation + Reputation×D_Blind + Cluster + (1|Partner)")
    
    try:
        model_full = smf.mixedlm(
            "log_rfs ~ normalized_score * D_Blind + reputation * D_Blind + C(industry_cluster) + age_std",
            data=df,
            groups=df['ballroom_partner']
        ).fit(reml=True)
        
        print("\n" + "="*60)
        print("[MODEL] Blind Voting Interaction Model Results")
        print("="*60)
        print(model_full.summary())
        
        # Extract key coefficients
        score_main = model_full.fe_params.get('normalized_score', 0)
        score_blind = model_full.fe_params.get('normalized_score:D_Blind', 0)
        rep_main = model_full.fe_params.get('reputation', 0)
        rep_blind = model_full.fe_params.get('reputation:D_Blind', 0)
        
        score_blind_pval = model_full.pvalues.get('normalized_score:D_Blind', 1.0)
        rep_blind_pval = model_full.pvalues.get('reputation:D_Blind', 1.0)
        
        print("\n[RESULTS] Key Interaction Effects:")
        print(f"  α₁ (Score main effect): {score_main:.4f}")
        print(f"  α₂ (Score × D_Blind): {score_blind:.4f} (P={score_blind_pval:.4f})")
        print(f"  α₃ (Reputation main effect): {rep_main:.4f}")
        print(f"  α₄ (Reputation × D_Blind): {rep_blind:.4f} (P={rep_blind_pval:.4f})")
        
        # Interpretation
        print("\n[INTERPRETATION]:")
        
        if score_blind_pval < 0.05:
            if score_blind < 0:
                print("  ✓ Score × D_Blind NEGATIVE & SIGNIFICANT:")
                print("    → During blind voting, current performance matters LESS")
                print("    → Evidence of 'blind voting decoupling'")
            else:
                print("  ✓ Score × D_Blind POSITIVE & SIGNIFICANT:")
                print("    → During blind voting, current performance matters MORE (unexpected)")
        else:
            print("  ✗ Score × D_Blind not significant: No evidence of score decoupling")
        
        if rep_blind_pval < 0.05:
            if rep_blind > 0:
                print("  ✓ Reputation × D_Blind POSITIVE & SIGNIFICANT:")
                print("    → During blind voting, prior reputation matters MORE")
                print("    → Evidence of 'reputation substitution effect'")
            else:
                print("  ✓ Reputation × D_Blind NEGATIVE & SIGNIFICANT:")
                print("    → During blind voting, reputation matters LESS (unexpected)")
        else:
            print("  ✗ Reputation × D_Blind not significant: No reputation substitution effect")
        
        results['model'] = model_full
        results['score_blind_coef'] = score_blind
        results['score_blind_pval'] = score_blind_pval
        results['rep_blind_coef'] = rep_blind
        results['rep_blind_pval'] = rep_blind_pval
        results['score_main'] = score_main
        results['rep_main'] = rep_main
        
    except Exception as e:
        print(f"  [WARNING] Full interaction model failed: {e}")
        import traceback
        traceback.print_exc()
        results['model'] = None
    
    # Visualize
    visualize_blind_voting_effect(df, results)
    
    return results


def visualize_blind_voting_effect(df: pd.DataFrame, results: dict):
    """
    Visualize the blind voting effect: Score vs Fan Vote by Era.
    
    Creates a scatter plot with two regression lines showing how
    the relationship between judge scores and fan votes differs
    between blind and non-blind voting eras.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors
    colors = {'Blind Era (S28-S30)': '#e74c3c', 'Non-Blind Era': '#3498db'}
    
    # ===== Plot 1: Score vs Fan Vote =====
    ax1 = axes[0]
    for label in df['blind_label'].unique():
        subset = df[df['blind_label'] == label]
        ax1.scatter(subset['normalized_score'], subset['log_rfs'], 
                   c=colors.get(label, '#333'), alpha=0.3, s=15, label=None)
    
    # Regression lines
    for label in df['blind_label'].unique():
        subset = df[df['blind_label'] == label]
        
        if len(subset) > 10:
            z = np.polyfit(subset['normalized_score'], subset['log_rfs'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['normalized_score'].min(), 
                                 subset['normalized_score'].max(), 100)
            
            style = '--' if 'Blind' in label else '-'
            corr = subset['normalized_score'].corr(subset['log_rfs'])
            
            ax1.plot(x_line, p(x_line), style, color=colors.get(label, '#333'), 
                    linewidth=3, label=f"{label}\n(slope={z[0]:.2f}, r={corr:.3f})")
    
    ax1.set_xlabel('Judge Score (Normalized)', fontsize=12)
    ax1.set_ylabel('Log(Relative Fan Share)', fontsize=12)
    ax1.set_title('Score vs Fan Vote by Era\n(Blind Voting Decoupling Test)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Reputation vs Fan Vote =====
    ax2 = axes[1]
    for label in df['blind_label'].unique():
        subset = df[df['blind_label'] == label]
        ax2.scatter(subset['reputation'], subset['log_rfs'], 
                   c=colors.get(label, '#333'), alpha=0.3, s=15, label=None)
    
    for label in df['blind_label'].unique():
        subset = df[df['blind_label'] == label]
        
        if len(subset) > 10:
            z = np.polyfit(subset['reputation'], subset['log_rfs'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['reputation'].min(), 
                                 subset['reputation'].max(), 100)
            
            style = '--' if 'Blind' in label else '-'
            corr = subset['reputation'].corr(subset['log_rfs'])
            
            ax2.plot(x_line, p(x_line), style, color=colors.get(label, '#333'), 
                    linewidth=3, label=f"{label}\n(slope={z[0]:.2f}, r={corr:.3f})")
    
    ax2.set_xlabel('Reputation (Prior Avg Score)', fontsize=12)
    ax2.set_ylabel('Log(Relative Fan Share)', fontsize=12)
    ax2.set_title('Reputation vs Fan Vote by Era\n(Reputation Substitution Test)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add model results annotation
    if 'score_blind_pval' in results:
        annotation = (
            f"Interaction Effects:\n"
            f"Score×Blind: {results.get('score_blind_coef', 0):.3f} "
            f"(P={results.get('score_blind_pval', 1):.3f})\n"
            f"Rep×Blind: {results.get('rep_blind_coef', 0):.3f} "
            f"(P={results.get('rep_blind_pval', 1):.3f})"
        )
        fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT_DIR / 'q3_blind_voting_effect.png', dpi=300)
    print(f"[INFO] Blind voting effect visualization saved to {OUTPUT_DIR / 'q3_blind_voting_effect.png'}")
    
    # Save results to CSV
    blind_summary = pd.DataFrame([
        {'Era': 'Non-Blind (S1-S27, S31+)', 
         'Score_Corr': results.get(0, {}).get('corr_score', np.nan),
         'Rep_Corr': results.get(0, {}).get('corr_rep', np.nan),
         'N': results.get(0, {}).get('n', 0)},
        {'Era': 'Blind (S28-S30)', 
         'Score_Corr': results.get(1, {}).get('corr_score', np.nan),
         'Rep_Corr': results.get(1, {}).get('corr_rep', np.nan),
         'N': results.get(1, {}).get('n', 0)},
    ])
    blind_summary.to_csv(OUTPUT_DATA_DIR / 'blind_voting_analysis.csv', index=False)
    print(f"[INFO] Blind voting analysis saved to {OUTPUT_DATA_DIR / 'blind_voting_analysis.csv'}")


# =============================================================================
# STEP 3: Data Loading & Preparation
# =============================================================================
def load_and_merge_data() -> pd.DataFrame:
    """Load and merge datasets."""
    print("\n" + "="*60)
    print("DATA LOADING & PREPARATION")
    print("="*60)
    
    # Load Bayesian results
    df_bayes = pd.read_csv(BAYESIAN_RESULTS_PATH)
    
    # Calculate Relative Fan Share
    df_bayes['avg_share_benchmark'] = 1.0 / df_bayes['n_contestants']
    df_bayes['relative_fan_share'] = df_bayes['estimated_fan_share'] / df_bayes['avg_share_benchmark']
    df_bayes['log_rfs'] = np.log(df_bayes['relative_fan_share'] + 0.001)
    
    # Load raw metadata
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_raw.columns = [col.strip().lower().replace(' ', '_').replace('/', '_') for col in df_raw.columns]
    
    cols_to_keep = ['celebrity_name', 'ballroom_partner', 'celebrity_industry', 
                    'celebrity_age_during_season', 'season']
    df_meta = df_raw[cols_to_keep].drop_duplicates(subset=['celebrity_name', 'season'])
    
    # Merge
    df = pd.merge(df_bayes, df_meta, on=['season', 'celebrity_name'], how='left')
    
    # Clean features
    df['age'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
    df['ballroom_partner'] = df['ballroom_partner'].astype(str).str.strip()
    
    # Standardize age
    df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
    
    # Calculate cumulative reputation (average score up to previous week)
    df = df.sort_values(['season', 'celebrity_name', 'week'])
    df['cumulative_score'] = df.groupby(['season', 'celebrity_name'])['normalized_score'].expanding().mean().reset_index(level=[0,1], drop=True)
    df['reputation'] = df.groupby(['season', 'celebrity_name'])['cumulative_score'].shift(1)
    df['reputation'] = df['reputation'].fillna(df['normalized_score'])  # First week uses current score
    
    # Remove rows with missing values
    df = df.dropna(subset=['age', 'normalized_score', 'log_rfs', 'ballroom_partner'])
    
    print(f"[INFO] Final dataset: {len(df)} records")
    
    return df


# =============================================================================
# STEP 4: Dual-Track LMM with Interactions
# =============================================================================
def run_advanced_models(df: pd.DataFrame):
    """
    Run advanced dual-track models with interaction terms.
    
    Model A (Technical): Score ~ Week + Cluster + Cluster×Week + Age + (1|Partner)
    Model B (Popularity): Fan ~ Score + Reputation + Cluster + (1|Partner)
    """
    print("\n" + "="*60)
    print("STEP 4: DUAL-TRACK LMM WITH INTERACTIONS")
    print("="*60)
    
    # Ensure categorical
    df['industry_cluster'] = df['industry_cluster'].astype('category')
    
    # -----------------------------------------------------------------
    # MODEL A: Technical Track with Growth Interactions
    # -----------------------------------------------------------------
    print("\n[MODEL A] Technical Track: Judge Score with Cluster×Week Interaction")
    
    formula_skill = """normalized_score ~ week + C(industry_cluster) + 
                       C(industry_cluster):week + age_std"""
    
    try:
        model_skill = smf.mixedlm(formula_skill, df, groups=df["ballroom_partner"])
        res_skill = model_skill.fit(method='powell')
        print(res_skill.summary())
    except Exception as e:
        print(f"[WARNING] Model A failed with interaction: {e}")
        print("[INFO] Falling back to simpler model...")
        formula_skill = "normalized_score ~ week + C(industry_cluster) + age_std"
        model_skill = smf.mixedlm(formula_skill, df, groups=df["ballroom_partner"])
        res_skill = model_skill.fit()
        print(res_skill.summary())
    
    # -----------------------------------------------------------------
    # MODEL B: Popularity Track with Reputation
    # -----------------------------------------------------------------
    print("\n[MODEL B] Popularity Track: Fan Share with Reputation Effect")
    
    formula_pop = """log_rfs ~ normalized_score + reputation + 
                     C(industry_cluster) + age_std"""
    
    try:
        model_pop = smf.mixedlm(formula_pop, df, groups=df["ballroom_partner"])
        res_pop = model_pop.fit()
        print(res_pop.summary())
    except Exception as e:
        print(f"[WARNING] Model B failed: {e}")
        formula_pop = "log_rfs ~ normalized_score + C(industry_cluster) + age_std"
        model_pop = smf.mixedlm(formula_pop, df, groups=df["ballroom_partner"])
        res_pop = model_pop.fit()
        print(res_pop.summary())
    
    return res_skill, res_pop


# =============================================================================
# STEP 5: Enhanced Visualizations
# =============================================================================
def visualize_forest_plot(res_skill, res_pop):
    """Create a forest plot comparing coefficients across models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract parameters (excluding intercept and group variance)
    skill_params = res_skill.params.drop(['Intercept', 'Group Var'], errors='ignore')
    pop_params = res_pop.params.drop(['Intercept', 'Group Var'], errors='ignore')
    
    # Find common variables
    common_vars = list(set(skill_params.index) & set(pop_params.index))
    
    if len(common_vars) == 0:
        print("[WARNING] No common variables for forest plot")
        return
    
    # Prepare data
    y_positions = np.arange(len(common_vars))
    
    skill_vals = [skill_params.get(v, 0) for v in common_vars]
    pop_vals = [pop_params.get(v, 0) for v in common_vars]
    
    # Normalize for comparison (Z-score within each model)
    skill_norm = (np.array(skill_vals) - np.mean(skill_vals)) / (np.std(skill_vals) + 0.001)
    pop_norm = (np.array(pop_vals) - np.mean(pop_vals)) / (np.std(pop_vals) + 0.001)
    
    # Plot
    ax.scatter(skill_norm, y_positions - 0.15, c='#3498db', s=150, label='Judge Model', marker='s', zorder=3)
    ax.scatter(pop_norm, y_positions + 0.15, c='#e74c3c', s=150, label='Fan Model', marker='o', zorder=3)
    
    # Connect pairs
    for i in range(len(common_vars)):
        ax.plot([skill_norm[i], pop_norm[i]], [y_positions[i]-0.15, y_positions[i]+0.15], 
                'gray', alpha=0.5, linewidth=1)
    
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([v.replace('C(industry_cluster)[T.', '').replace(']', '') for v in common_vars])
    ax.set_xlabel('Normalized Coefficient (Z-Score)', fontsize=12)
    ax.set_title('Forest Plot: Judge vs Fan Model Coefficients', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'q3_forest_plot.png', dpi=300)
    print(f"[INFO] Forest plot saved to {OUTPUT_DIR / 'q3_forest_plot.png'}")


def visualize_partner_effects(res_skill, res_pop, df):
    """Extract and visualize partner random effects."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    re_skill = res_skill.random_effects
    re_pop = res_pop.random_effects
    
    partners = list(re_skill.keys())
    
    data = []
    for p in partners:
        skill_boost = re_skill[p]['Group']
        pop_boost = re_pop[p]['Group']
        count = df[df['ballroom_partner'] == p]['celebrity_name'].nunique()
        
        if count >= 3:
            data.append({
                'Partner': p,
                'Skill_Boost': skill_boost,
                'Popularity_Boost': pop_boost,
                'Count': count
            })
    
    df_effects = pd.DataFrame(data)
    
    # Scatter plot
    scatter = ax.scatter(df_effects['Skill_Boost'], df_effects['Popularity_Boost'],
                        s=df_effects['Count']*30, alpha=0.7, c=df_effects['Skill_Boost'],
                        cmap='viridis', edgecolors='white', linewidth=1)
    
    # Label outliers
    for _, row in df_effects.iterrows():
        if abs(row['Skill_Boost']) > 0.03 or abs(row['Popularity_Boost']) > 0.15:
            ax.annotate(row['Partner'], (row['Skill_Boost'], row['Popularity_Boost']),
                       fontsize=9, alpha=0.8)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Technical Boost (Judge Score Effect)', fontsize=12)
    ax.set_ylabel('Popularity Boost (Log Fan Share Effect)', fontsize=12)
    ax.set_title('The "Kingmaker" Analysis: Partner Effects', fontsize=14, fontweight='bold')
    
    # Quadrant labels
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(xlim[1]*0.7, ylim[1]*0.8, 'KINGMAKERS\n(High Both)', ha='center', color='green', fontweight='bold')
    ax.text(xlim[0]*0.7, ylim[1]*0.8, 'CULT FAVORITES\n(Low Skill, High Fans)', ha='center', color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'q3_partner_effects_v2.png', dpi=300)
    print(f"[INFO] Partner effects saved to {OUTPUT_DIR / 'q3_partner_effects_v2.png'}")
    
    return df_effects


def generate_summary_report(df, df_growth, res_skill, res_pop, df_effects, blind_results=None):
    """Generate comprehensive summary report."""
    report_path = OUTPUT_DATA_DIR / 'q3_advanced_summary.md'
    
    with open(report_path, 'w') as f:
        f.write("# Task 3 Advanced Analysis: Key Findings\n\n")
        
        f.write("## 1. Industry Clustering (K-Means)\n")
        f.write("Industries were classified into clusters based on 3 dimensions:\n")
        f.write("- **Physicality**: Physical training requirements\n")
        f.write("- **Performance**: Stage presence and emotional expression\n")
        f.write("- **Fanbase**: Built-in fan loyalty\n\n")
        
        cluster_dist = df['industry_cluster'].value_counts()
        f.write("### Cluster Distribution:\n")
        for cluster, count in cluster_dist.items():
            f.write(f"- **{cluster}**: {count} observations ({count/len(df)*100:.1f}%)\n")
        
        f.write("\n## 2. Growth Trajectory Analysis\n")
        f.write("### Average Growth Slope by Cluster:\n")
        growth_summary = df_growth.groupby('industry_cluster')['growth_slope'].agg(['mean', 'count'])
        for cluster, row in growth_summary.iterrows():
            f.write(f"- **{cluster}**: {row['mean']:.4f} (n={int(row['count'])})\n")
        
        f.write("\n## 3. Top Professional Partners (Kingmaker Effect)\n")
        f.write("### Top Technical Instructors:\n")
        for _, row in df_effects.nlargest(3, 'Skill_Boost').iterrows():
            f.write(f"- **{row['Partner']}**: +{row['Skill_Boost']:.3f} skill boost\n")
        
        f.write("\n### Top Fan Magnets:\n")
        for _, row in df_effects.nlargest(3, 'Popularity_Boost').iterrows():
            f.write(f"- **{row['Partner']}**: +{row['Popularity_Boost']:.3f} popularity boost\n")
        
        f.write("\n## 4. Model Coefficients\n")
        f.write("### Technical Model (Judge Score):\n")
        for param, val in res_skill.params.items():
            if 'Intercept' not in param and 'Var' not in param:
                pval = res_skill.pvalues.get(param, 1.0)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                f.write(f"- {param}: {val:.4f} {sig}\n")
        
        f.write("\n### Popularity Model (Fan Share):\n")
        for param, val in res_pop.params.items():
            if 'Intercept' not in param and 'Var' not in param:
                pval = res_pop.pvalues.get(param, 1.0)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                f.write(f"- {param}: {val:.4f} {sig}\n")
        
        # Add blind voting analysis
        if blind_results:
            f.write("\n## 5. Blind Voting Effect Analysis\n")
            f.write("**Definition**: Seasons 28-30 (Live Vote era with time zone delay) as 'Blind Voting' proxy.\n\n")
            
            if 0 in blind_results and 1 in blind_results:
                f.write("### Score-Vote Correlation by Era:\n")
                corr_0 = blind_results[0].get('corr_score', 0)
                corr_1 = blind_results[1].get('corr_score', 0)
                n_0 = blind_results[0].get('n', 0)
                n_1 = blind_results[1].get('n', 0)
                
                f.write(f"- **Non-Blind Era**: r = {corr_0:.4f} (n={n_0})\n")
                f.write(f"- **Blind Era (S28-S30)**: r = {corr_1:.4f} (n={n_1})\n\n")
                
                if corr_0 != 0 and corr_1 != 0:
                    pct_weaker = (1 - corr_1 / corr_0) * 100
                    f.write(f"**Result**: Blind Era correlation is {abs(pct_weaker):.1f}% {'weaker' if pct_weaker > 0 else 'stronger'}\n\n")
            
            if 'score_blind_pval' in blind_results:
                sig = "✓" if blind_results.get('score_blind_pval', 1) < 0.05 else "✗"
                f.write(f"### Interaction Test (Score × BlindEra):\n")
                f.write(f"- Coefficient: {blind_results.get('score_blind_coef', 0):.4f}\n")
                f.write(f"- P-value: {blind_results.get('score_blind_pval', 1):.4f} {sig}\n\n")
                
                sig_rep = "✓" if blind_results.get('rep_blind_pval', 1) < 0.05 else "✗"
                f.write(f"### Interaction Test (Reputation × BlindEra):\n")
                f.write(f"- Coefficient: {blind_results.get('rep_blind_coef', 0):.4f}\n")
                f.write(f"- P-value: {blind_results.get('rep_blind_pval', 1):.4f} {sig_rep}\n")
    
    print(f"[INFO] Summary report saved to {report_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("TASK 3 OPTIMIZED: ADVANCED FACTOR ANALYSIS")
    print("="*60)
    
    # Step 1: Load data
    df = load_and_merge_data()
    
    # Step 2: K-Means clustering (n_clusters=None triggers auto-selection)
    df = perform_kmeans_clustering(df, n_clusters=None)
    
    # Step 3: Growth trajectory analysis
    df_growth = compute_growth_trajectories(df)
    
    # Step 4: Age linearity verification
    linearity_results = verify_age_linearity(df)
    
    # Step 5: Blind voting effect analysis
    blind_results = analyze_blind_voting_effect(df)
    
    # Step 6: Advanced LMM models
    res_skill, res_pop = run_advanced_models(df)
    
    # Step 7: Visualizations
    visualize_forest_plot(res_skill, res_pop)
    df_effects = visualize_partner_effects(res_skill, res_pop, df)
    
    # Step 8: Generate report
    generate_summary_report(df, df_growth, res_skill, res_pop, df_effects, blind_results)
    
    print("\n" + "="*60)
    print("[SUCCESS] Task 3 Advanced Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
