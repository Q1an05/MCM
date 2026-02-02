#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Task 4: Case Study Analysis

Analyzing the impact of the new DTPM system on specific controversial contestants:
- Jerry Rice (Season 2)
- Billy Ray Cyrus (Season 4)
- Bristol Palin (Season 11)
- Bobby Bones (Season 27)

We simulate week-by-week:
If the new system ranks them last (eliminated), we declare their "New Exit Week".

Author: MCM Team
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "results" / "full_simulation_bayesian.csv"
OUTPUT_PATH = BASE_DIR / "results" / "system_design" / "case_study_report.md"
PLOT_DIR = BASE_DIR / "results" / "plots" / "system_design"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# The "Golden Configuration" from previous optimization
# Start Weight (Judge): 0.90
# End Weight (Judge):   0.60
# Penalty Beta:         0.40
PARAMS = {
    'w_start': 0.90,
    'w_end': 0.60,
    'beta': 0.40
}

TARGETS = [
    {'name': 'Jerry Rice', 'season': 2, 'actual_result': 'Runner-Up (2nd)'},
    {'name': 'Billy Ray Cyrus', 'season': 4, 'actual_result': '5th Place'},
    {'name': 'Bristol Palin', 'season': 11, 'actual_result': '3rd Place'},
    {'name': 'Bobby Bones', 'season': 27, 'actual_result': 'Winner (1st)'}
]

# =============================================================================
# Analysis Logic
# =============================================================================
def analyze_case_study(df, targets, params):
    results = []
    
    # Pre-calculate season lengths for dynamic weighting
    season_max_weeks = df.groupby('season')['week'].max().to_dict()

    for target in targets:
        name = target['name']
        season = target['season']
        actual_res = target['actual_result']
        
        print(f"[INFO] Analyzing {name} (Season {season})...")
        
        # Get all data for this season
        season_df = df[df['season'] == season].copy()
        
        # Get maximum weeks for this season (T)
        T = season_max_weeks.get(season, 10)
        
        new_exit_week = "Survived to Final"
        elimination_reason = "N/A"
        
        # Iterate through weeks 1 to T
        # We check every week they actually competed.
        # Note: If they survive week 1 in new system, we assume they proceed to week 2.
        # We proceed until we find a week where they would be eliminated.
        
        weeks_competed = sorted(season_df[season_df['celebrity_name'] == name]['week'].unique())
        
        for week in weeks_competed:
            week_data = season_df[season_df['week'] == week].copy()
            
            # Skip if only 1 contestant (Winner week) - usually week T
            if len(week_data) < 2:
                continue
                
            # --- DTPM Calculation ---
            
            # 1. Normalize Judge Scores (Raw -> Percent)
            s_judge = week_data['raw_score_sum']
            if s_judge.sum() == 0:
                p_judge = s_judge * 0
            else:
                p_judge = s_judge / s_judge.sum()
                
            # 2. Fan Shares (Simulated)
            p_fan = week_data['estimated_fan_share']
            if p_fan.sum() == 0: # Should not happen, but safety
                p_fan = p_fan * 0 
            else:
                p_fan = p_fan / p_fan.sum()
            
            # 3. Dynamic Weight w(t)
            progress = np.clip((week - 1) / (T - 1) if T > 1 else 1, 0, 1)
            w_t = params['w_start'] - (params['w_start'] - params['w_end']) * progress
            
            # 4. Performance Gated Multiplier
            mean_judge = s_judge.mean()
            gamma = np.where(s_judge < mean_judge, params['beta'], 1.0)
            
            # 5. Total Score
            # DTPM Score = w * P_judge + (1-w) * (P_fan * gamma)
            # Scores are percentages (decimals). Higher is better.
            total_score = w_t * p_judge + (1 - w_t) * (p_fan * gamma)
            
            # --- Check Impact ---
            
            # Assign scores back to temp dataframe to find rank
            week_data['dtpm_score'] = total_score
            
            # Rank descending (High score = Rank 1)
            week_data['rank_dtpm'] = week_data['dtpm_score'].rank(ascending=False, method='min')
            
            # Find the target's rank and score
            target_row = week_data[week_data['celebrity_name'] == name]
            if target_row.empty:
                continue # Should not happen based on loop
                
            target_rank = target_row['rank_dtpm'].iloc[0]
            target_judge_score = target_row['raw_score_sum'].iloc[0]
            target_avg = mean_judge
            
            # Determination: Would they be eliminated?
            # Criteria: Last Place in Total Score
            num_contestants = len(week_data)
            
            # Logic: In standard weeks, 1 person leaves. So Rank >= num_contestants is out.
            # (Handling ties: if rank is same as num_contestants, they are at risk.
            #  If multiple people are last, it's a tie, but usually one goes. We count as elimination risk.)
            
            if target_rank == num_contestants:
                new_exit_week = f"Week {week}"
                new_exit_week_num = week
                # Analyze why
                is_below_avg = target_judge_score < target_avg
                penalty_text = "Yes (x0.4)" if is_below_avg else "No"
                elimination_reason = f"Ranked Last ({int(target_rank)}/{num_contestants}). Penalty Applied: {penalty_text}"
                break
        
        # If survived loop without breaking
        if new_exit_week == "Survived to Final":
            new_exit_week_num = T

        # Determine Actual Exit Week Number for Plotting
        actual_week_record = season_df[season_df['celebrity_name'] == name]['exit_week'].iloc[0]
        if actual_week_record == 99:
            actual_exit_week_num = T
        else:
            actual_exit_week_num = actual_week_record

        results.append({
            'Contestant': name,
            'Season': season,
            'Actual Outcome': actual_res,
            'New Outcome': new_exit_week,
            'Details': elimination_reason,
            'Actual Week Num': int(actual_exit_week_num),
            'New Week Num': int(new_exit_week_num)
        })
        
    return pd.DataFrame(results)

def plot_case_study_comparison(results_df):
    plt.figure(figsize=(12, 7))
    
    # Create plot data
    plot_data = results_df.copy()
    plot_data['Label'] = plot_data['Contestant'] + " (S" + plot_data['Season'].astype(str) + ")"
    
    # Plot formatting
    y_pos = np.arange(len(plot_data))
    
    ax = plt.gca()
    
    # Colors
    color_actual = '#FF9999' # Red-ish
    color_new = '#66B2FF'    # Blue-ish
    
    # Draw lines connecting the points (if different)
    for i, row in plot_data.iterrows():
        actual = row['Actual Week Num']
        new = row['New Week Num']
        
        # Draw line
        ax.plot([actual, new], [i, i], color='gray', linestyle='--', zorder=1, alpha=0.7)
        
        # Plot points
        # Actual
        ax.scatter(actual, i, color=color_actual, s=250, label='Actual Historical Exit' if i == 0 else "", zorder=3, edgecolors='black', linewidth=1.5)
        # New
        ax.scatter(new, i, color=color_new, s=250, label='New DTPM Exit' if i == 0 else "", zorder=3, edgecolors='black', linewidth=1.5)
        
        # Add labels - adjusted for clearer spacing
        # Top label (New Result) - Move higher up (more negative relative to i)
        ax.text(new, i - 0.25, f"Result: Week {new}", 
                ha='center', va='bottom', fontsize=10, color='darkblue', fontweight='bold')
        
        # Bottom label (Actual Result) - Move lower down (more positive relative to i)
        ax.text(actual, i + 0.25, f"Actual: {row['Actual Outcome']}\n(Week {actual})", 
                ha='center', va='top', fontsize=10, color='darkred', weight='bold')

        # Add "Saved" or "Cut" annotation - Center on line
        diff = new - actual
        mid = (actual + new) / 2
        
        # For small differences (like Jerry Rice, diff=1), shift text slightly to avoid overlap
        if abs(diff) <= 1.5:
            # Shift annotation slightly above the line to clear the red dot's label below
            text_y = i - 0.05
        else:
            text_y = i

        if diff < 0:
            ax.text(mid, text_y, f"{abs(diff)} Weeks Earlier", ha='center', va='center', 
                    fontsize=9, color='red', weight='bold', backgroundcolor='white',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        elif diff > 0:
             ax.text(mid, text_y, f"+{diff} Weeks", ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data['Label'], fontsize=12, fontweight='bold')
    
    # Invert Y axis so index 0 is at top
    ax.invert_yaxis()
    
    # Set Limits with generous padding
    # Y-axis: -0.8 (top) to len-0.2 (bottom) 
    # Since inverted: -0.8 is "above" 0.0. 
    ax.set_ylim(len(plot_data) - 0.2, -0.8)
    
    # X-axis: 0 to 12 to fit text
    ax.set_xlim(0, 12)
    
    ax.set_xlabel('Competition Week', fontsize=12)
    ax.set_title('Case Study: Impact of DTPM System on Controversial Contestants', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
    # Grid
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Save
    plot_path = PLOT_DIR / "case_study_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"[INFO] Plot saved to {plot_path}")

def generate_report(results_df):
    markdown = "# Case Study: DTPM System Impact on Controversial Contestants\n\n"
    markdown += "This analysis simulates the **Dynamic-Threshold Percent Model (DTPM)** retrospectively on famous controversial cases.\n"
    markdown += f"**System Parameters**: Start Weight={PARAMS['w_start']}, End Weight={PARAMS['w_end']}, Penalty Beta={PARAMS['beta']}\n\n"
    
    markdown += "## Summary Table\n\n"
    # Manual markdown table generation to avoid tabulate dependency
    headers = results_df.columns.tolist()
    markdown += "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in results_df.iterrows():
        markdown += "| " + " | ".join(str(x) for x in row.values) + " |\n"
    
    markdown += "\n\n## Detailed Analysis\n\n"
    
    for _, row in results_df.iterrows():
        markdown += f"### {row['Contestant']} (Season {row['Season']})\n"
        markdown += f"- **Historical Result**: {row['Actual Outcome']}\n"
        markdown += f"- **DTPM Result**: {row['New Outcome']}\n"
        markdown += f"- **Why**: {row['Details']}\n\n"
        
        if "Winner" in row['Actual Outcome'] and "Week" in row['New Outcome']:
            markdown += "> **IMPACT**: The new system successfully prevented a controversial win, protecting the integrity of the competition.\n\n"
        elif "Week" in row['New Outcome']:
             markdown += "> **IMPACT**: Significant correction of historical longevity.\n\n"
    
    print(markdown)
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(markdown)
    print(f"[INFO] Report saved to {OUTPUT_PATH}")

# =============================================================================
# Main
# =============================================================================
def main():
    df = pd.read_csv(INPUT_PATH)
    results_df = analyze_case_study(df, TARGETS, PARAMS)
    
    # Plotting
    plot_case_study_comparison(results_df)
    
    generate_report(results_df)

if __name__ == "__main__":
    main()
