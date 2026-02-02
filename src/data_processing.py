#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Data Processing Script - Wide to Long Format Transformation

This script transforms the raw DWTS data from Wide Format to a clean Long Format
dataset that serves as the Ground Truth for Monte Carlo simulation.

Author: MCM Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Configuration
# =============================================================================
RAW_DATA_PATH = Path(__file__).parent.parent / "data_raw" / "2026_MCM_Problem_C_Data.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data_processed" / "dwts_simulation_input.csv"
PLOTS_DIR = Path(__file__).parent.parent / "results" / "plots"

# Safe high number to mark finalists as active throughout the season
FINALIST_EXIT_WEEK = 99

# Rule system mapping based on Problem Description
RULE_SYSTEM_MAP = {
    "Rank": list(range(1, 3)),           # Seasons 1-2
    "Percent": list(range(3, 28)),        # Seasons 3-27
    "Rank_With_Save": list(range(28, 35)) # Seasons 28-34
}


# =============================================================================
# Data Loading & Cleaning
# =============================================================================
def load_and_clean_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw CSV data and perform initial cleaning.
    
    - Standardize column names to snake_case
    - Convert 'N/A' strings to np.nan
    - Ensure season is integer type
    - Ensure all score columns are float type
    
    Args:
        filepath: Path to the raw CSV file
        
    Returns:
        Cleaned DataFrame
    """
    print(f"[INFO] Loading data from: {filepath}")
    
    # Load CSV, treating 'N/A' as missing values
    df = pd.read_csv(filepath, na_values=['N/A', 'n/a', ''])
    
    # Standardize column names to snake_case
    def to_snake_case(name: str) -> str:
        # Replace spaces with underscores, convert to lowercase
        name = name.strip().lower()
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'[^\w]', '_', name)
        return name
    
    df.columns = [to_snake_case(col) for col in df.columns]
    
    # Ensure season is integer
    df['season'] = df['season'].astype(int)
    
    # Identify all score columns and convert to float
    score_cols = [col for col in df.columns if 'score' in col]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"[INFO] Loaded {len(df)} contestants across Seasons {df['season'].min()}-{df['season'].max()}")
    print(f"[INFO] Found {len(score_cols)} score columns")
    
    return df


# =============================================================================
# Ground Truth Parsing - Exit Week Extraction
# =============================================================================
def parse_exit_week_from_results(results_text: str) -> tuple:
    """
    Parse the results column text to extract exit week and result type.
    
    Rules (case-insensitive):
    - "Eliminated Week X" -> exit_week = X, type = "eliminated"
    - "1st/2nd/3rd Place" or Finalist indicators -> exit_week = 99, type = "finalist"
    - "Withdrew" -> exit_week = -1 (placeholder, will be computed from scores), type = "withdrew"
    
    Args:
        results_text: Text from the results column
        
    Returns:
        Tuple of (exit_week, result_type)
    """
    if pd.isna(results_text):
        return (1, "unknown")
    
    text = str(results_text).strip().lower()
    
    # Pattern 1: Eliminated Week X
    eliminated_match = re.search(r'eliminated\s+week\s+(\d+)', text, re.IGNORECASE)
    if eliminated_match:
        return (int(eliminated_match.group(1)), "eliminated")
    
    # Pattern 2: Finalist indicators (1st, 2nd, 3rd, Winner, Runner Up, Finalist)
    finalist_patterns = [
        r'1st\s*place', r'2nd\s*place', r'3rd\s*place',
        r'winner', r'runner\s*up', r'finalist',
        r'\d+(st|nd|rd|th)\s*place'  # Generic placement pattern for top finishers
    ]
    for pattern in finalist_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # For top 3-5 placements, treat as finalist (active all season)
            return (FINALIST_EXIT_WEEK, "finalist")
    
    # Pattern 3: Withdrew
    if 'withdrew' in text:
        return (-1, "withdrew")  # Placeholder, will compute from scores
    
    # Default: unknown, treat conservatively
    return (1, "unknown")


def find_last_valid_score_week(row: pd.Series, score_pattern: str = r'week(\d+)_judge\d+_score') -> int:
    """
    Find the last week where a contestant has a valid score (>0).
    Used for "Withdrew" cases.
    
    Args:
        row: A single row from the DataFrame
        score_pattern: Regex pattern to identify score columns
        
    Returns:
        The last week number with valid scores
    """
    last_week = 1  # Default to week 1 if no valid scores found
    
    for col in row.index:
        match = re.match(score_pattern, col)
        if match:
            week_num = int(match.group(1))
            score_value = row[col]
            # Check if score is valid (not NaN and > 0)
            if pd.notna(score_value) and score_value > 0:
                last_week = max(last_week, week_num)
    
    return last_week


def compute_exit_week_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create exit_week and result_type columns for the entire DataFrame.
    
    For "Withdrew" cases, dynamically compute exit_week from the last valid score week.
    
    Args:
        df: DataFrame with results column
        
    Returns:
        DataFrame with exit_week and result_type columns added
    """
    print("[INFO] Parsing exit weeks from results column...")
    
    # Parse results for each row
    parsed_results = df['results'].apply(parse_exit_week_from_results)
    df['exit_week'] = parsed_results.apply(lambda x: x[0])
    df['result_type'] = parsed_results.apply(lambda x: x[1])
    
    # Handle "Withdrew" cases - find last valid score week
    withdrew_mask = df['result_type'] == 'withdrew'
    if withdrew_mask.sum() > 0:
        print(f"[INFO] Found {withdrew_mask.sum()} 'Withdrew' cases, computing exit week from scores...")
        for idx in df[withdrew_mask].index:
            df.loc[idx, 'exit_week'] = find_last_valid_score_week(df.loc[idx])
    
    # Log summary
    result_counts = df['result_type'].value_counts()
    print(f"[INFO] Result type distribution:")
    for rtype, count in result_counts.items():
        print(f"       - {rtype}: {count}")
    
    return df


# =============================================================================
# Wide-to-Long Transformation
# =============================================================================
def melt_scores_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform wide format data to long format.
    
    Creates one row per (season, week, celebrity_name) with:
    - raw_score_sum: Sum of all valid judge scores
    - max_possible_points: Count of valid judges * 10 (row-wise calculation)
    - normalized_score: raw_score_sum / max_possible_points (0.0 to 1.0 scale)
    
    Args:
        df: Wide format DataFrame with score columns
        
    Returns:
        Long format DataFrame
    """
    print("[INFO] Transforming data from Wide to Long format...")
    
    # Identify score columns and their structure
    score_cols = [col for col in df.columns if re.match(r'week\d+_judge\d+_score', col)]
    
    # Extract unique weeks
    weeks = sorted(set(int(re.match(r'week(\d+)', col).group(1)) for col in score_cols))
    print(f"[INFO] Found data for weeks: {min(weeks)}-{max(weeks)}")
    
    # Base columns to keep
    base_cols = ['season', 'celebrity_name', 'celebrity_industry', 'exit_week', 'result_type', 'placement']
    
    long_records = []
    
    for _, row in df.iterrows():
        season = row['season']
        celeb_name = row['celebrity_name']
        celeb_industry = row.get('celebrity_industry', 'Unknown')
        exit_week = row['exit_week']
        result_type = row['result_type']
        placement = row.get('placement', np.nan)
        
        for week in weeks:
            # Skip weeks after contestant's exit
            if week > exit_week:
                continue
            
            # Collect all judge scores for this week
            week_score_cols = [col for col in score_cols if col.startswith(f'week{week}_')]
            
            # Extract valid (non-NaN) scores
            valid_scores = []
            for col in week_score_cols:
                score = row[col]
                if pd.notna(score):
                    valid_scores.append(float(score))
            
            # Calculate row-wise max possible points based on valid judges
            n_valid_judges = len(valid_scores)
            raw_score_sum = sum(valid_scores) if valid_scores else 0.0
            max_possible_points = n_valid_judges * 10.0
            
            # Skip rows with no valid scores (contestant didn't dance that week)
            if n_valid_judges == 0 or raw_score_sum == 0:
                continue
            
            # Calculate normalized score (0.0 to 1.0 scale)
            normalized_score = raw_score_sum / max_possible_points if max_possible_points > 0 else 0.0
            
            long_records.append({
                'season': season,
                'week': week,
                'celebrity_name': celeb_name,
                'celebrity_industry': celeb_industry,
                'exit_week': exit_week,
                'result_type': result_type,
                'placement': placement,
                'n_judges': n_valid_judges,
                'raw_score_sum': raw_score_sum,
                'max_possible_points': max_possible_points,
                'normalized_score': normalized_score
            })
    
    long_df = pd.DataFrame(long_records)
    print(f"[INFO] Created {len(long_df)} long-format records")
    
    return long_df


# =============================================================================
# Feature Engineering
# =============================================================================
def get_rule_system(season: int) -> str:
    """
    Determine the rule system based on season number.
    
    Args:
        season: Season number
        
    Returns:
        Rule system string: "Rank", "Percent", or "Rank_With_Save"
    """
    if season in RULE_SYSTEM_MAP["Rank"]:
        return "Rank"
    elif season in RULE_SYSTEM_MAP["Percent"]:
        return "Percent"
    elif season in RULE_SYSTEM_MAP["Rank_With_Save"]:
        return "Rank_With_Save"
    else:
        return "Unknown"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simulation-ready features for the Monte Carlo model.
    
    Features computed:
    1. judge_share: normalized_score / sum(normalized_scores) within [season, week]
    2. judge_rank: Rank by normalized_score (descending, 1=best), using average for ties
    3. n_contestants: Count of active contestants in that week
    4. is_eliminated: True if week == exit_week and not finalist/withdrew
    5. rule_system: Era tag based on season number
    
    Args:
        df: Long format DataFrame
        
    Returns:
        DataFrame with computed features
    """
    print("[INFO] Computing simulation features...")
    
    df = df.copy()
    
    # 1. Calculate n_contestants per (season, week)
    contestant_counts = df.groupby(['season', 'week'])['celebrity_name'].transform('count')
    df['n_contestants'] = contestant_counts.astype(int)
    
    # 2. Calculate judge_share: proportion of normalized_score within each (season, week)
    score_sum_by_week = df.groupby(['season', 'week'])['normalized_score'].transform('sum')
    df['judge_share'] = df['normalized_score'] / score_sum_by_week
    
    # 3. Calculate judge_rank: rank by normalized_score (descending), method='average' for ties
    df['judge_rank'] = df.groupby(['season', 'week'])['normalized_score'].rank(
        method='average', ascending=False
    )
    
    # 4. Calculate is_eliminated
    # True ONLY if: week == exit_week AND result_type is "eliminated"
    # For "withdrew" cases, is_eliminated is always False (they left voluntarily)
    df['is_eliminated'] = (
        (df['week'] == df['exit_week']) & 
        (df['result_type'] == 'eliminated')
    )
    
    # 5. Assign rule_system based on season
    df['rule_system'] = df['season'].apply(get_rule_system)
    
    # Log summary
    print(f"[INFO] Feature computation complete:")
    print(f"       - Unique seasons: {df['season'].nunique()}")
    print(f"       - Total (season, week) groups: {df.groupby(['season', 'week']).ngroups}")
    print(f"       - Elimination events: {df['is_eliminated'].sum()}")
    
    return df


# =============================================================================
# Validation & Output
# =============================================================================
def validate_and_save(df: pd.DataFrame, output_path: Path) -> None:
    """
    Validate the processed data and save to CSV.
    
    Validations:
    1. Check that judge_share sums to ~1.0 for each (season, week)
    2. Check for any NaN values in critical columns
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the output CSV
    """
    print("\n" + "="*60)
    print("VALIDATION & OUTPUT")
    print("="*60)
    
    # Validation 1: Check judge_share sums
    share_sums = df.groupby(['season', 'week'])['judge_share'].sum()
    
    print("\n[VALIDATION] Judge share sums per (season, week):")
    print(share_sums.describe())
    
    # Check if all sums are approximately 1.0
    tolerance = 0.001
    invalid_sums = share_sums[(share_sums < 1.0 - tolerance) | (share_sums > 1.0 + tolerance)]
    if len(invalid_sums) > 0:
        print(f"[WARNING] Found {len(invalid_sums)} (season, week) groups with invalid share sums!")
        print(invalid_sums.head())
    else:
        print("[✓] All judge_share groups sum to ~1.0")
    
    # Validation 2: Check for NaN in critical columns
    critical_cols = ['season', 'week', 'celebrity_name', 'normalized_score', 
                     'judge_share', 'judge_rank', 'is_eliminated', 'rule_system']
    for col in critical_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"[WARNING] Column '{col}' has {nan_count} NaN values")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n[✓] Saved {len(df)} rows to: {output_path}")
    
    # Print summary statistics
    print("\n" + "-"*60)
    print("SUMMARY STATISTICS")
    print("-"*60)
    print(f"Total rows processed: {len(df)}")
    print(f"Seasons covered: {df['season'].min()} - {df['season'].max()}")
    print(f"Unique contestants: {df['celebrity_name'].nunique()}")
    print(f"\nRule system distribution:")
    print(df['rule_system'].value_counts().sort_index())
    
    print(f"\nResult type distribution:")
    print(df['result_type'].value_counts())


def display_sample_data(df: pd.DataFrame) -> None:
    """
    Display sample rows from different eras for visual inspection.
    
    Args:
        df: Processed DataFrame
    """
    print("\n" + "-"*60)
    print("SAMPLE DATA")
    print("-"*60)
    
    # Display columns for readability
    display_cols = ['season', 'week', 'celebrity_name', 'normalized_score', 
                    'judge_share', 'judge_rank', 'n_contestants', 'is_eliminated', 'rule_system']
    
    # Sample from Season 1 (Rank Era)
    s1_data = df[df['season'] == 1].head(5)
    if len(s1_data) > 0:
        print("\n[SAMPLE] Season 1 (Rank Era):")
        print(s1_data[display_cols].to_string(index=False))
    
    # Sample from Season 10 (Percent Era)
    s10_data = df[df['season'] == 10].head(5)
    if len(s10_data) > 0:
        print("\n[SAMPLE] Season 10 (Percent Era):")
        print(s10_data[display_cols].to_string(index=False))
    
    # Sample from Season 30 (Rank_With_Save Era)
    s30_data = df[df['season'] == 30].head(5)
    if len(s30_data) > 0:
        print("\n[SAMPLE] Season 30 (Rank_With_Save Era):")
        print(s30_data[display_cols].to_string(index=False))


# =============================================================================
# Diagnostic Visualizations
# =============================================================================
def generate_diagnostic_plots(df: pd.DataFrame) -> None:
    """
    Generate diagnostic plots as shown in the requirements:
    1. Avg Total Judge Score Heatmap (Season x Week)
    2. Number of Contestants per Season
    3. Season Length (Max Week with Competition)
    
    Args:
        df: Processed Long Format DataFrame
    """
    print("\n" + "="*60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("="*60)
    
    try:
        # Try to use unified viz config if available
        import viz_config
        viz_config.setup_academic_style()
        palette = viz_config.MORANDI_COLORS
        accent_blue = viz_config.MORANDI_ACCENT[1]
        accent_pink = viz_config.MORANDI_ACCENT[0]
        # Use Morandi sequential blue for the heatmap
        cmap_heatmap = 'morandi_seq_blue'
    except ImportError:
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("muted")
        accent_blue = "steelblue"
        accent_pink = "crimson"
        cmap_heatmap = 'viridis'

    # Set up the figure with 3 subplots (stacked vertically)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.4)
    
    # --- Plot 1: Heatmap of Avg Total Judge Score ---
    # Pivot: index=Season, columns=Week, values=Avg raw_score_sum
    heatmap_data = df.groupby(['season', 'week'])['raw_score_sum'].mean().unstack()
    
    # Re-index seasons to ensure they are in order and no gaps
    all_seasons = sorted(df['season'].unique())
    heatmap_data = heatmap_data.reindex(all_seasons)
    
    sns.heatmap(heatmap_data, ax=axes[0], cmap=cmap_heatmap, 
                cbar_kws={'label': 'Avg Total Judge Score'},
                linewidths=0.5, linecolor='white')
    
    axes[0].set_title('Avg Total Judge Score Heatmap (Season x Week)', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Week', fontsize=12)
    axes[0].set_ylabel('Season', fontsize=12)
    
    # --- Plot 2: Number of Contestants per Season ---
    contestants_per_season = df.groupby('season')['celebrity_name'].nunique()
    
    axes[1].plot(contestants_per_season.index, contestants_per_season.values, 
                 marker='o', markersize=6, linestyle='-', linewidth=2, color=accent_blue)
    axes[1].fill_between(contestants_per_season.index, contestants_per_season.values, color=accent_blue, alpha=0.1)
    
    axes[1].set_title('Number of Contestants per Season', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Season', fontsize=12)
    axes[1].set_ylabel('Contestants', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_xticks(range(0, int(df['season'].max()) + 5, 5))
    
    # --- Plot 3: Season Length (Max Week) ---
    weeks_per_season = df.groupby('season')['week'].max()
    
    axes[2].plot(weeks_per_season.index, weeks_per_season.values, 
                 marker='s', markersize=6, linestyle='-', linewidth=2, color=accent_pink)
    axes[2].fill_between(weeks_per_season.index, weeks_per_season.values, color=accent_pink, alpha=0.1)
    
    axes[2].set_title('Season Length (Max Week with Competition)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Season', fontsize=12)
    axes[2].set_ylabel('Weeks', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].set_xticks(range(0, int(df['season'].max()) + 5, 5))
    
    # Save the figure
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "data_diagnostics_summary.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Diagnostic plots saved to: {save_path}")

    # --- Plot 4: Top 15 Celebrity Industries (Count) ---
    plt.figure(figsize=(10, 8))
    
    # Get unique contestants first to avoid double counting across weeks
    unique_contestants = df.drop_duplicates(subset=['celebrity_name'])
    industry_counts = unique_contestants['celebrity_industry'].value_counts().head(15)
    
    # Create a gradient from light blue to medium blue (avoiding black-ish colors)
    # Using a subset of the Blues palette or a custom Morandi gradient
    n_bars = len(industry_counts)
    try:
        # Generate colors from the Morandi sequential blue cmap
        # We sample from 0.2 to 0.7 to keep it in the "light to medium" blue range
        cmap = plt.get_cmap('morandi_seq_blue')
        colors = [cmap(i) for i in np.linspace(0.3, 0.8, n_bars)]
    except:
        colors = sns.color_palette("Blues", n_colors=n_bars)
    
    # Reverse so the largest bars at the top are the darkest (but mid-range blue)
    colors = colors[::-1]
    
    bars = sns.barplot(x=industry_counts.values, y=industry_counts.index, 
                       palette=colors, alpha=0.85, hue=industry_counts.index, legend=False)
    
    # Add data labels to each bar
    for i, v in enumerate(industry_counts.values):
        plt.text(v + 0.5, i, str(int(v)), color='#333333', va='center', fontweight='bold', fontsize=10)
    
    plt.title('Top 15 Celebrity Industries (Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Industry', fontsize=12)
    plt.xlim(0, industry_counts.values.max() * 1.1)  # Give some space for labels
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    industry_save_path = PLOTS_DIR / "top_15_industries.png"
    plt.savefig(industry_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Industry plot saved to: {industry_save_path}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    """
    Main data processing pipeline.
    
    Steps:
    1. Load and clean raw data
    2. Parse exit weeks from results column
    3. Transform from wide to long format
    4. Compute simulation features
    5. Validate and save output
    """
    print("="*60)
    print("MCM 2026 Problem C - Data Processing Pipeline")
    print("="*60 + "\n")
    
    # Step 1: Load and clean data
    df = load_and_clean_data(RAW_DATA_PATH)
    
    # Step 2: Parse exit weeks
    df = compute_exit_week_column(df)
    
    # Step 3: Transform to long format
    long_df = melt_scores_to_long(df)
    
    # Step 4: Compute features
    final_df = compute_features(long_df)
    
    # Step 5: Validate and save
    validate_and_save(final_df, OUTPUT_PATH)
    
    # Step 6: Generate diagnostic plots
    generate_diagnostic_plots(final_df)
    
    # Display samples
    display_sample_data(final_df)
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    
    return final_df


if __name__ == "__main__":
    main()
