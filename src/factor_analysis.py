#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Task 3: Factor Analysis & Dual-Track Attribution Model
------------------------------------------------------
This script implements a "Dual-Track Attribution Model" using Linear Mixed-Effects Models (LMM)
to analyze the drivers of success in DWTS.

It separates success into two distinct dependent variables:
1. Y_skill (Technical): Normalized Judge Scores
2. Y_pop (Popularity): Relative Fan Share (derived from Task 1 Bayesian Estimates)

It quantifies the impact of:
- Demographics (Age, Industry) as Fixed Effects
- Professional Partners (The "Kingmaker" Effect) as Random Effects

Author: MCM Team
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

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
# Data Preparation
# =============================================================================
def load_and_merge_data() -> pd.DataFrame:
    """
    Load Bayesian results (Y variables) and Raw Data (X features), 
    merging them into a single analytical dataset.
    """
    print("[INFO] Loading datasets...")
    
    # 1. Load Bayesian Estimates (The "Truth" about popularity)
    df_bayes = pd.read_csv(BAYESIAN_RESULTS_PATH)
    
    # Calculate Relative Fan Share (RFS)
    # RFS = Share / (1/N_contestants)
    # If RFS > 1.0, the contestant is more popular than average.
    df_bayes['avg_share_benchmark'] = 1.0 / df_bayes['n_contestants']
    df_bayes['relative_fan_share'] = df_bayes['estimated_fan_share'] / df_bayes['avg_share_benchmark']
    
    # Log-transform RFS for better normality in regression (RFS is strictly positive)
    # Adding a small epsilon just in case, though shares shouldn't be 0
    df_bayes['log_rfs'] = np.log(df_bayes['relative_fan_share'] + 0.001)
    
    # 2. Load Raw Metadata (Demographics)
    df_raw = pd.read_csv(RAW_DATA_PATH)
    
    # Normalize column names for raw data
    df_raw.columns = [col.strip().lower().replace(' ', '_').replace('/', '_') for col in df_raw.columns]
    
    # We need: celebrity_name, ballroom_partner, celebrity_industry, celebrity_age_during_season
    cols_to_keep = ['celebrity_name', 'ballroom_partner', 'celebrity_industry', 'celebrity_age_during_season', 'season']
    
    # Remove duplicates in raw data (one row per celebrity per season)
    df_meta = df_raw[cols_to_keep].drop_duplicates(subset=['celebrity_name', 'season'])
    
    # 3. Merge
    # Note: 'celebrity_name' in bayes is already cleaned/consistent with the pipeline.
    # We hope raw data matches.
    print(f"[INFO] Merging {len(df_bayes)} weekly records with metadata...")
    df_merged = pd.merge(
        df_bayes, 
        df_meta, 
        on=['season', 'celebrity_name'], 
        how='left'
    )
    
    # Check match rate
    missing_meta = df_merged['ballroom_partner'].isna().mean()
    print(f"[INFO] Missing metadata rate: {missing_meta:.2%}")
    
    if missing_meta > 0.1:
        print("[WARNING] High missing metadata. Check name matching.")
        
    return df_merged


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and categorize feature columns.
    """
    df = df.copy()
    
    # 1. Clean Industry
    # Simplify into top categories: Actor, Athlete, Singer, Reality, TV Host, Other
    def simplify_industry(ind):
        if pd.isna(ind): return "Other"
        ind = str(ind).lower()
        if 'actor' in ind or 'actress' in ind: return 'Actor'
        if 'athlete' in ind or 'nba' in ind or 'nfl' in ind or 'olympian' in ind or 'football' in ind: return 'Athlete'
        if 'singer' in ind or 'musician' in ind or 'rapper' in ind or 'pop star' in ind: return 'Singer'
        if 'reality' in ind or 'bachelor' in ind or 'housewife' in ind: return 'Reality Star'
        if 'host' in ind or 'presenter' in ind: return 'TV Host'
        if 'model' in ind: return 'Model'
        return 'Other'

    df['industry_group'] = df['celebrity_industry'].apply(simplify_industry)
    
    # 2. Clean Age
    # Ensure numeric
    df['age'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
    df = df.dropna(subset=['age', 'normalized_score', 'log_rfs', 'ballroom_partner'])
    
    # 3. Standardize Age (Z-score) for better regression coefficients
    df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
    
    # 4. Clean Partner
    df['ballroom_partner'] = df['ballroom_partner'].astype(str).str.strip()
    
    print(f"[INFO] Final dataset size for analysis: {len(df)} records")
    print(f"[INFO] Industry distribution:\n{df['industry_group'].value_counts()}")
    
    return df


# =============================================================================
# Modeling: Linear Mixed Effects
# =============================================================================
def run_dual_track_models(df: pd.DataFrame):
    """
    Run two LMMs:
    1. Skill Model: Y = Judge Score
    2. Pop Model: Y = Fan Share (Log RFS)
    """
    print("\n" + "="*50)
    print("RUNNING DUAL-TRACK ATTRIBUTION MODELS (LMM)")
    print("="*50)
    
    # -------------------------------------------------------------
    # MODEL A: The Technical Method (Judge Sensitivity)
    # -------------------------------------------------------------
    # Formula: Score ~ Age + Industry + (1 | Partner)
    # We use '1 | Partner' to treat Partners as random intercepts
    
    formula_skill = "normalized_score ~ age_std + C(industry_group)"
    
    print(f"[INFO] Fitting Model A (Skill)... Formula: {formula_skill} + (1|Partner)")
    model_skill = smf.mixedlm(formula_skill, df, groups=df["ballroom_partner"])
    res_skill = model_skill.fit()
    
    print(res_skill.summary())
    
    # -------------------------------------------------------------
    # MODEL B: The Popularity Method (Fan Sensitivity)
    # -------------------------------------------------------------
    
    formula_pop = "log_rfs ~ age_std + C(industry_group)"
    
    print(f"[INFO] Fitting Model B (Popularity)... Formula: {formula_pop} + (1|Partner)")
    model_pop = smf.mixedlm(formula_pop, df, groups=df["ballroom_partner"])
    res_pop = model_pop.fit()
    
    print(res_pop.summary())
    
    return res_skill, res_pop


def compare_age_linearity(df: pd.DataFrame):
    """
    Compare Linear vs Quadratic models for Age effect using AIC/BIC.
    """
    print("\n" + "="*50)
    print("ANALYZING AGE LINEARITY (Linear vs Quadratic)")
    print("="*50)
    
    results = []
    
    for target, name in [('normalized_score', 'Skill'), ('log_rfs', 'Popularity')]:
        # 1. Linear Model
        formula_lin = f"{target} ~ age_std + C(industry_group)"
        model_lin = smf.mixedlm(formula_lin, df, groups=df["ballroom_partner"])
        res_lin = model_lin.fit()
        
        # 2. Quadratic Model (Quadratic term: I(age_std**2))
        formula_quad = f"{target} ~ age_std + np.power(age_std, 2) + C(industry_group)"
        model_quad = smf.mixedlm(formula_quad, df, groups=df["ballroom_partner"])
        res_quad = model_quad.fit()
        
        # Capture metrics
        aic_lin, bic_lin = res_lin.aic, res_lin.bic
        aic_quad, bic_quad = res_quad.aic, res_quad.bic
        
        # P-value for quadratic term
        p_quad = res_quad.pvalues.get('np.power(age_std, 2)', 1.0)
        
        results.append({
            'Target': name,
            'Model': 'Linear',
            'AIC': aic_lin,
            'BIC': bic_lin,
            'P_Quad': None
        })
        results.append({
            'Target': name,
            'Model': 'Quadratic',
            'AIC': aic_quad,
            'BIC': bic_quad,
            'P_Quad': p_quad
        })
        
        print(f"\n[{name}] Comparison:")
        print(f"  Linear    -> AIC: {aic_lin:.2f}, BIC: {bic_lin:.2f}")
        print(f"  Quadratic -> AIC: {aic_quad:.2f}, BIC: {bic_quad:.2f}")
        print(f"  Quadratic term P-value: {p_quad:.4f}")
        
    df_comp = pd.DataFrame(results)
    df_comp.to_csv(OUTPUT_DATA_DIR / "age_linearity_comparison.csv", index=False)
    
    return df_comp


# =============================================================================
# Analysis & Visualizations
# =============================================================================
def visualize_partner_effects(res_skill, res_pop, df):
    """
    Extract random effects (Partner quality) and plot Skill vs Popularity boost.
    """
    # Extract Random Effects (conditional modes)
    re_skill = res_skill.random_effects
    re_pop = res_pop.random_effects
    
    partners = list(re_skill.keys())
    
    data = []
    for p in partners:
        # random_effects returns a dict of Series (usually 'Group' term)
        skill_boost = re_skill[p]['Group']
        pop_boost = re_pop[p]['Group']
        count = df[df['ballroom_partner'] == p]['celebrity_name'].nunique()
        
        if count >= 3: # Only include partners with significant history
            data.append({
                'Partner': p,
                'Skill_Boost': skill_boost,
                'Popularity_Boost': pop_boost,
                'Contestant_Count': count
            })
            
    df_effects = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Scatter plot
    ax = sns.scatterplot(
        data=df_effects,
        x='Skill_Boost',
        y='Popularity_Boost',
        size='Contestant_Count',
        sizes=(50, 400),
        alpha=0.7,
        hue='Skill_Boost',
        palette="viridis",
        legend=False
    )
    
    # Add labels for top outliers
    for _, row in df_effects.iterrows():
        # Label validation: Only label significant outliers
        if abs(row['Skill_Boost']) > 0.03 or abs(row['Popularity_Boost']) > 0.1:
            plt.text(
                row['Skill_Boost']+0.002, 
                row['Popularity_Boost'], 
                row['Partner'], 
                fontsize=9,
                alpha=0.8
            )

    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("The 'Kingmaker' Analysis: Partner Impact on Skill vs. Popularity", fontsize=16, fontweight='bold')
    plt.xlabel("Technical Boost (Judge Score Effect)", fontsize=12)
    plt.ylabel("Popularity Boost (Fan Share Effect, Log Scale)", fontsize=12)
    
    # Quadrant Annotation
    plt.text(df_effects['Skill_Boost'].max(), df_effects['Popularity_Boost'].max(), "THE KINGMAKERS\n(High Skill, High Fans)", ha='right', va='top', color='green', fontweight='bold')
    plt.text(df_effects['Skill_Boost'].min(), df_effects['Popularity_Boost'].max(), "CULT FAVORITES\n(Low Skill, High Fans)", ha='left', va='top', color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q3_partner_effects.png", dpi=300)
    print(f"[INFO] Partner analysis saved to {OUTPUT_DIR / 'q3_partner_effects.png'}")
    
    return df_effects

def visualize_industry_radar(res_skill, res_pop):
    """
    Compare industry coefficients using a Radar/Bar chart.
    """
    # Extract Fixed Effects
    params_skill = res_skill.params
    params_pop = res_pop.params
    
    # Filter for Industry coefficients
    inds = [idx for idx in params_skill.index if 'industry_group' in idx]
    
    data = []
    for idx in inds:
        # format: C(industry_group)[T.Athlete]
        label = idx.split('[T.')[1].replace(']', '')
        
        # Skill effect (scaled for visibility if needed, but scores are 0-1)
        # Pop effect is log scale.
        # We need to normalize them to compare "Magnitude of Bias"
        
        data.append({
            'Industry': label,
            'Technical_Bias': params_skill[idx],
            'Popularity_Bias': params_pop[idx]
        })
        
    df_ind = pd.DataFrame(data)
    
    # Melting for seaborn barplot
    df_melt = df_ind.melt(id_vars='Industry', var_name='Metric', value_name='Coefficient')
    
    plt.figure(figsize=(12, 6))
    
    sns.barplot(data=df_melt, x='Industry', y='Coefficient', hue='Metric', palette=['#3498db', '#e74c3c'])
    
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Industry Bias: Technical Ability vs. Fan Popularity", fontsize=14, fontweight='bold')
    plt.ylabel("Model Coefficient (Effect Size)", fontsize=12)
    plt.xlabel("Industry (Baseline: Actor)", fontsize=12)
    
    plt.legend(title='Dimension')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q3_industry_bias.png", dpi=300)
    print(f"[INFO] Industry analysis saved to {OUTPUT_DIR / 'q3_industry_bias.png'}")


def visualize_age_trends(df):
    """
    Visualize Age effect on Normalized Score vs Fan Share.
    Show both Linear and Quadratic fits.
    """
    # Double Y-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # 1. Technical Score (Left Axis)
    color1 = '#3498db' # Blue
    ax1.set_xlabel('Celebrity Age', fontsize=12)
    ax1.set_ylabel('Judge Score (Normalized)', color=color1, fontsize=12)
    
    # Linear Fit for Skill
    sns.regplot(data=df, x='age', y='normalized_score', ax=ax1, 
                scatter=False, color=color1, line_kws={'linestyle': '--', 'alpha': 0.5}, label='Skill (Linear)')
    # Quadratic Fit for Skill
    sns.regplot(data=df, x='age', y='normalized_score', ax=ax1, 
                order=2, scatter_kws={'alpha': 0.1, 's': 10}, color=color1, label='Skill (Quadratic)')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 2. Popularity (Right Axis)
    ax2 = ax1.twinx()
    color2 = '#e74c3c' # Red
    ax2.set_ylabel('Relative Fan Share', color=color2, fontsize=12)
    
    # Linear Fit for Pop
    sns.regplot(data=df, x='age', y='relative_fan_share', ax=ax2, 
                scatter=False, color=color2, line_kws={'linestyle': '--', 'alpha': 0.5}, label='Pop (Linear)')
    # Quadratic Fit for Pop
    sns.regplot(data=df, x='age', y='relative_fan_share', ax=ax2, 
                order=2, scatter=False, color=color2, label='Pop (Quadratic)')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Age Paradox: Linear vs. Quadratic Trends (Judge vs. Fans)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "q3_age_trends_comparison.png", dpi=300)
    print(f"[INFO] Age trend comparison saved to {OUTPUT_DIR / 'q3_age_trends_comparison.png'}")


def generate_memo_summary(res_skill, res_pop, df_effects, df_comp):
    """
    Generate a text summary for the Memo.
    """
    summary_path = OUTPUT_DATA_DIR / "q3_model_summary.md"
    
    with open(summary_path, 'w') as f:
        f.write("# Task 3: Factor Analysis Key Findings\n\n")
        
        f.write("## 1. The 'Partner Effect' (Kingmaker Analysis)\n")
        top_pro_skill = df_effects.sort_values('Skill_Boost', ascending=False).head(3)
        top_pro_pop = df_effects.sort_values('Popularity_Boost', ascending=False).head(3)
        
        f.write("### Top Technical Instructors (Judge Bonus):\n")
        for _, row in top_pro_skill.iterrows():
            f.write(f"- **{row['Partner']}**: +{row['Skill_Boost']:.3f} score boost\n")
            
        f.write("\n### Top Fan Favorites (Vote Magnets):\n")
        for _, row in top_pro_pop.iterrows():
            f.write(f"- **{row['Partner']}**: +{row['Popularity_Boost']:.3f} log-share boost\n")
            
        f.write("\n## 2. Demographic Biases\n")
        f.write("### Age Effect (Linear View):\n")
        f.write(f"- **Skill**: {res_skill.params['age_std']:.4f} (per SD of age)\n")
        f.write(f"- **Popularity**: {res_pop.params['age_std']:.4f} (per SD of age)\n")
        
        f.write("\n### Age Non-linearity Analysis:\n")
        for target in ['Skill', 'Popularity']:
            comp = df_comp[df_comp['Target'] == target]
            lin = comp[comp['Model'] == 'Linear'].iloc[0]
            quad = comp[comp['Model'] == 'Quadratic'].iloc[0]
            
            p_val = quad['P_Quad']
            is_better = quad['AIC'] < lin['AIC']
            
            f.write(f"#### {target} Linearity Test:\n")
            f.write(f"- Linear AIC: {lin['AIC']:.2f} | Quadratic AIC: {quad['AIC']:.2f}\n")
            f.write(f"- Quadratic term p-value: {p_val:.4f}\n")
            f.write(f"- **Conclusion**: {'Quadratic' if (is_better and p_val < 0.05) else 'Linear'} effect is more appropriate.\n")

        f.write("\n### Industry Effect (vs Actor):\n")
        # List significant industry coefficients
        sk_p = res_skill.pvalues
        for idx, val in res_skill.params.items():
            if 'industry' in idx and sk_p[idx] < 0.05:
                f.write(f"- **Skill ({idx})**: {val:.3f}\n")
                
        pop_p = res_pop.pvalues
        for idx, val in res_pop.params.items():
            if 'industry' in idx and pop_p[idx] < 0.05:
                f.write(f"- **Pop ({idx})**: {val:.3f}\n")

    print(f"[INFO] Memo attributes summary saved to {summary_path}")

# =============================================================================
# Main
# =============================================================================
def main():
    # 1. Load Data
    df = load_and_merge_data()
    
    # 2. Preprocess
    df_clean = clean_features(df)
    
    # 3. Linearity Analysis (NEW)
    df_comp = compare_age_linearity(df_clean)
    
    # 4. Run Models
    res_skill, res_pop = run_dual_track_models(df_clean)
    
    # 5. Visualize
    df_effects = visualize_partner_effects(res_skill, res_pop, df_clean)
    visualize_industry_radar(res_skill, res_pop)
    visualize_age_trends(df_clean)
    
    # 6. Report
    generate_memo_summary(res_skill, res_pop, df_effects, df_comp)
    
    print("[SUCCESS] Task 3 Analysis Complete.")

if __name__ == "__main__":
    main()
