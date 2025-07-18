import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, Optional
import glob
from config.paths import paths

def load_all_data(base_path: Path) -> pd.DataFrame:
    """Load all CSV files from base_path and combine them into a single DataFrame"""
    all_data = []
    
    # Find all CSV files matching the pattern
    pattern = str(base_path / "processed_*_trajectories.csv")
    csv_files = glob.glob(pattern)
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Extract rat and day from filename
            filename = Path(file_path).stem
            parts = filename.split('_')
            if len(parts) >= 3:
                rat = parts[1]
                day = parts[2]
                df['rat'] = rat
                df['day'] = day
                df['file_source'] = filename
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        raise ValueError(f"No CSV files found in {base_path}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the data"""
    # Convert boolean columns
    df['Correct'] = df['Correct'].astype(bool)
    df['VTE'] = df['VTE'].astype(bool)
    
    # Ensure Average_Speed is numeric
    df['Average_Speed'] = pd.to_numeric(df['Average_Speed'], errors='coerce')
    
    # Remove rows with missing essential data
    df = df.dropna(subset=['Average_Speed', 'Correct', 'VTE'])
    
    return df

def calculate_per_rat_speed_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speed threshold for each rat individually"""
    df_with_thresholds = df.copy()
    
    # Calculate speed threshold for each rat
    rat_thresholds = df.groupby('rat')['Average_Speed'].agg(['mean', 'std']).reset_index()
    rat_thresholds['speed_threshold'] = rat_thresholds['mean'] - rat_thresholds['std']
    
    # Merge back with original data
    df_with_thresholds = df_with_thresholds.merge(rat_thresholds[['rat', 'speed_threshold']], on='rat')
    
    # Create High_Speed column based on per-rat thresholds
    df_with_thresholds['High_Speed'] = df_with_thresholds['Average_Speed'] > df_with_thresholds['speed_threshold']
    
    return df_with_thresholds

def analyze_vte_speed_correctness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze VTE rates by speed (high/low) and correctness with SEM"""
    
    # Calculate per-rat thresholds
    df_with_thresholds = calculate_per_rat_speed_thresholds(df)
    
    # Group by rat, speed, and correctness to get per-rat VTE rates
    per_rat_results = df_with_thresholds.groupby(['rat', 'High_Speed', 'Correct']).agg({
        'VTE': ['count', 'sum', 'mean']
    }).round(4)
    
    per_rat_results.columns = ['Total_Trials', 'VTE_Count', 'VTE_Rate']
    per_rat_results = per_rat_results.reset_index()
    
    # Calculate mean and SEM across rats
    summary_results = per_rat_results.groupby(['High_Speed', 'Correct']).agg({
        'VTE_Rate': ['mean', 'std', 'count']
    }).round(4)
    
    summary_results.columns = ['VTE_Rate_Mean', 'VTE_Rate_Std', 'Rat_Count']
    summary_results['VTE_Rate_SEM'] = summary_results['VTE_Rate_Std'] / np.sqrt(summary_results['Rat_Count'])
    summary_results = summary_results.reset_index()
    
    return summary_results, df_with_thresholds

def analyze_vte_after_condition(df: pd.DataFrame, same_trial_type: bool = False, 
                               num_trials_after: int = 4) -> pd.DataFrame:
    """Analyze VTE rates in high speed incorrect trials and the following trials with SEM"""
    
    # Calculate per-rat thresholds
    df_with_thresholds = calculate_per_rat_speed_thresholds(df)
    
    results = []
    
    # Group by rat and day to analyze within sessions
    for (rat, day), group in df_with_thresholds.groupby(['rat', 'day']):
        group = group.sort_values('Unnamed: 0').reset_index(drop=True)
        
        # Find high speed incorrect trials
        high_speed_incorrect_indices = group[
            (group['High_Speed'] == True) & (group['Correct'] == False)
        ].index.tolist()
        
        for base_idx in high_speed_incorrect_indices:
            base_trial = group.iloc[base_idx]
            
            # Include the high speed incorrect trial itself (offset 0) and the next num_trials_after trials
            for offset in range(0, num_trials_after + 1):
                trial_idx = base_idx + offset
                
                if trial_idx < len(group):
                    current_trial = group.iloc[trial_idx]
                    
                    # Skip if we want same trial type but they're different (except for offset 0)
                    if same_trial_type and offset > 0 and current_trial['Trial Type'] != base_trial['Trial Type']:
                        continue
                    
                    results.append({
                        'rat': rat,
                        'day': day,
                        'base_trial_idx': base_idx,
                        'offset': offset,
                        'next_trial_vte': current_trial['VTE'],
                        'next_trial_type': current_trial['Trial Type'],
                        'base_trial_type': base_trial['Trial Type']
                    })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return pd.DataFrame(columns=['offset', 'VTE_Rate_Mean', 'VTE_Rate_SEM', 'Trial_Count'])
    
    # Calculate per-rat VTE rates by offset
    per_rat_summary = results_df.groupby(['rat', 'offset']).agg({
        'next_trial_vte': ['count', 'sum', 'mean']
    }).round(4)
    
    per_rat_summary.columns = ['Trial_Count', 'VTE_Count', 'VTE_Rate']
    per_rat_summary = per_rat_summary.reset_index()
    
    # Calculate mean and SEM across rats
    final_summary = per_rat_summary.groupby('offset').agg({
        'VTE_Rate': ['mean', 'std', 'count'],
        'Trial_Count': 'sum'
    }).round(4)
    
    final_summary.columns = ['VTE_Rate_Mean', 'VTE_Rate_Std', 'Rat_Count', 'Total_Trials']
    final_summary['VTE_Rate_SEM'] = final_summary['VTE_Rate_Std'] / np.sqrt(final_summary['Rat_Count'])
    final_summary = final_summary.reset_index()
    
    return final_summary

def analyze_vte_comparison(df: pd.DataFrame, condition1: str, condition2: str,
                          num_trials_after: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare VTE rates in base trials and following trials between two different conditions with SEM"""
    
    # Calculate per-rat thresholds
    df_with_thresholds = calculate_per_rat_speed_thresholds(df)
    
    def get_condition_trials(group, condition):
        if condition == 'high_speed_incorrect':
            return (group['High_Speed'] == True) & (group['Correct'] == False)
        elif condition == 'low_speed_incorrect':
            return (group['High_Speed'] == False) & (group['Correct'] == False)
        elif condition == 'high_speed_correct':
            return (group['High_Speed'] == True) & (group['Correct'] == True)
        else:
            raise ValueError(f"Unknown condition: {condition}")
    
    results1 = []
    results2 = []
    
    # Group by rat and day
    for (rat, day), group in df_with_thresholds.groupby(['rat', 'day']):
        group = group.sort_values('Unnamed: 0').reset_index(drop=True)
        
        # Find trials matching each condition
        condition1_indices = group[get_condition_trials(group, condition1)].index.tolist()
        condition2_indices = group[get_condition_trials(group, condition2)].index.tolist()
        
        # Analyze condition 1
        for base_idx in condition1_indices:
            base_trial = group.iloc[base_idx]
            for offset in range(0, num_trials_after + 1):
                trial_idx = base_idx + offset
                if trial_idx < len(group):
                    current_trial = group.iloc[trial_idx]
                    results1.append({
                        'rat': rat,
                        'day': day,
                        'offset': offset,
                        'next_trial_vte': current_trial['VTE'],
                        'condition': condition1
                    })
        
        # Analyze condition 2
        for base_idx in condition2_indices:
            base_trial = group.iloc[base_idx]
            for offset in range(0, num_trials_after + 1):
                trial_idx = base_idx + offset
                if trial_idx < len(group):
                    current_trial = group.iloc[trial_idx]
                    results2.append({
                        'rat': rat,
                        'day': day,
                        'offset': offset,
                        'next_trial_vte': current_trial['VTE'],
                        'condition': condition2
                    })
    
    # Combine results
    all_results = pd.DataFrame(results1 + results2)
    
    if len(all_results) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate per-rat VTE rates by condition and offset
    per_rat_summary = all_results.groupby(['rat', 'condition', 'offset']).agg({
        'next_trial_vte': ['count', 'sum', 'mean']
    }).round(4)
    
    per_rat_summary.columns = ['Trial_Count', 'VTE_Count', 'VTE_Rate']
    per_rat_summary = per_rat_summary.reset_index()
    
    # Calculate mean and SEM across rats
    final_summary = per_rat_summary.groupby(['condition', 'offset']).agg({
        'VTE_Rate': ['mean', 'std', 'count'],
        'Trial_Count': 'sum'
    }).round(4)
    
    final_summary.columns = ['VTE_Rate_Mean', 'VTE_Rate_Std', 'Rat_Count', 'Total_Trials']
    final_summary['VTE_Rate_SEM'] = final_summary['VTE_Rate_Std'] / np.sqrt(final_summary['Rat_Count'])
    final_summary = final_summary.reset_index()
    
    return final_summary, all_results

def create_plots(df: pd.DataFrame):
    """Create all five plots with SEM bars"""
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('VTE Analysis (Per-Rat Speed Thresholds)', fontsize=16)
    
    # Plot 1: VTE rate by speed and correctness
    speed_results, df_with_thresholds = analyze_vte_speed_correctness(df)
    
    ax1 = axes[0, 0]
    speed_results['Speed_Label'] = speed_results['High_Speed'].map({True: 'High Speed', False: 'Low Speed'})
    speed_results['Correctness_Label'] = speed_results['Correct'].map({True: 'Correct', False: 'Incorrect'})
    
    # Create grouped bar plot with error bars
    x_pos = np.arange(2)  # High Speed, Low Speed
    width = 0.35
    
    correct_data = speed_results[speed_results['Correct'] == True]
    incorrect_data = speed_results[speed_results['Correct'] == False]
    
    # Sort by High_Speed for consistent ordering
    correct_data = correct_data.sort_values('High_Speed')
    incorrect_data = incorrect_data.sort_values('High_Speed')
    
    ax1.bar(x_pos - width/2, correct_data['VTE_Rate_Mean'], width, 
            yerr=correct_data['VTE_Rate_SEM'], capsize=5, label='Correct', alpha=0.8)
    ax1.bar(x_pos + width/2, incorrect_data['VTE_Rate_Mean'], width, 
            yerr=incorrect_data['VTE_Rate_SEM'], capsize=5, label='Incorrect', alpha=0.8)
    
    ax1.set_title('VTE Rate by Speed and Correctness\n(Per-Rat Speed Thresholds)')
    ax1.set_ylabel('VTE Rate')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Low Speed', 'High Speed'])
    ax1.legend(title='Trial Outcome')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: VTE rate after high speed incorrect trials (all trial types)
    ax2 = axes[0, 1]
    after_high_speed_incorrect = analyze_vte_after_condition(df, same_trial_type=False)
    
    if len(after_high_speed_incorrect) > 0:
        ax2.errorbar(after_high_speed_incorrect['offset'], after_high_speed_incorrect['VTE_Rate_Mean'], 
                    yerr=after_high_speed_incorrect['VTE_Rate_SEM'], 
                    marker='o', linewidth=2, markersize=8, capsize=5)
        ax2.set_title('VTE Rate: High Speed Incorrect Trial and Following Trials')
        ax2.set_xlabel('Trial Position (0 = High Speed Incorrect Trial)')
        ax2.set_ylabel('VTE Rate')
        ax2.set_xticks(range(0, 5))
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('VTE Rate: High Speed Incorrect Trial and Following Trials')
    
    # Plot 3: VTE rate after high speed incorrect trials (same trial type only)
    ax3 = axes[0, 2]
    after_high_speed_incorrect_same_type = analyze_vte_after_condition(df, same_trial_type=True)
    
    if len(after_high_speed_incorrect_same_type) > 0:
        ax3.errorbar(after_high_speed_incorrect_same_type['offset'], after_high_speed_incorrect_same_type['VTE_Rate_Mean'], 
                    yerr=after_high_speed_incorrect_same_type['VTE_Rate_SEM'], 
                    marker='o', linewidth=2, markersize=8, color='green', capsize=5)
        ax3.set_title('VTE Rate: High Speed Incorrect Trial and Following Trials\n(Same Trial Type Only)')
        ax3.set_xlabel('Trial Position (0 = High Speed Incorrect Trial)')
        ax3.set_ylabel('VTE Rate')
        ax3.set_xticks(range(0, 5))
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('VTE Rate: High Speed Incorrect Trial and Following Trials\n(Same Trial Type Only)')
    
    # Plot 4: Compare high speed incorrect vs low speed incorrect
    ax4 = axes[1, 0]
    comparison_speed, _ = analyze_vte_comparison(df, 'high_speed_incorrect', 'low_speed_incorrect')
    
    if len(comparison_speed) > 0:
        for condition in comparison_speed['condition'].unique():
            condition_data = comparison_speed[comparison_speed['condition'] == condition]
            label = 'High Speed Incorrect' if condition == 'high_speed_incorrect' else 'Low Speed Incorrect'
            ax4.errorbar(condition_data['offset'], condition_data['VTE_Rate_Mean'], 
                        yerr=condition_data['VTE_Rate_SEM'], 
                        marker='o', linewidth=2, markersize=8, label=label, capsize=5)
        
        ax4.set_title('VTE Rate Comparison:\nHigh vs Low Speed Incorrect Trials')
        ax4.set_xlabel('Trial Position (0 = Base Trial)')
        ax4.set_ylabel('VTE Rate')
        ax4.set_xticks(range(0, 5))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('VTE Rate Comparison:\nHigh vs Low Speed Incorrect Trials')
    
    # Plot 5: Compare high speed incorrect vs high speed correct
    ax5 = axes[1, 1]
    comparison_correct, _ = analyze_vte_comparison(df, 'high_speed_incorrect', 'high_speed_correct')
    
    if len(comparison_correct) > 0:
        for condition in comparison_correct['condition'].unique():
            condition_data = comparison_correct[comparison_correct['condition'] == condition]
            label = 'High Speed Incorrect' if condition == 'high_speed_incorrect' else 'High Speed Correct'
            color = 'red' if condition == 'high_speed_incorrect' else 'blue'
            ax5.errorbar(condition_data['offset'], condition_data['VTE_Rate_Mean'], 
                        yerr=condition_data['VTE_Rate_SEM'], 
                        marker='o', linewidth=2, markersize=8, label=label, color=color, capsize=5)
        
        ax5.set_title('VTE Rate Comparison:\nHigh Speed Incorrect vs Correct Trials')
        ax5.set_xlabel('Trial Position (0 = Base Trial)')
        ax5.set_ylabel('VTE Rate')
        ax5.set_xticks(range(0, 5))
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('VTE Rate Comparison:\nHigh Speed Incorrect vs Correct Trials')
    
    # Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics with per-rat thresholds
    mean_threshold = df_with_thresholds.groupby('rat')['speed_threshold'].first().mean()
    std_threshold = df_with_thresholds.groupby('rat')['speed_threshold'].first().std()
    
    high_speed_incorrect_count = len(df_with_thresholds[(df_with_thresholds['High_Speed'] == True) & (df_with_thresholds['Correct'] == False)])
    low_speed_incorrect_count = len(df_with_thresholds[(df_with_thresholds['High_Speed'] == False) & (df_with_thresholds['Correct'] == False)])
    high_speed_correct_count = len(df_with_thresholds[(df_with_thresholds['High_Speed'] == True) & (df_with_thresholds['Correct'] == True)])
    
    summary_text = f"""
    Summary Statistics:
    
    Total trials: {len(df)}
    Total rats: {df['rat'].nunique()}
    Total days: {df.groupby(['rat', 'day']).ngroups}
    
    Overall VTE rate: {df['VTE'].mean():.3f}
    
    Per-rat speed thresholds:
    Mean threshold: {mean_threshold:.2f}
    Std threshold: {std_threshold:.2f}
    
    Global speed stats:
    Mean speed: {df['Average_Speed'].mean():.2f}
    Std speed: {df['Average_Speed'].std():.2f}
    
    Trial Counts:
    High Speed Incorrect: {high_speed_incorrect_count}
    Low Speed Incorrect: {low_speed_incorrect_count}
    High Speed Correct: {high_speed_correct_count}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return (speed_results, after_high_speed_incorrect, after_high_speed_incorrect_same_type, 
            comparison_speed, comparison_correct)

def main_analysis(base_path: Union[str, Path]):
    """Main function to run the complete analysis"""
    
    base_path = Path(base_path)
    
    print(f"Loading data from {base_path}")
    df = load_all_data(base_path)
    print(f"Loaded {len(df)} total trials")
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    print(f"After preprocessing: {len(df)} trials")
    
    print("Creating plots...")
    results = create_plots(df)
    
    print("\nResults Summary:")
    print("\n1. VTE Rate by Speed and Correctness:")
    print(results[0])
    
    print("\n2. VTE Rate in High Speed Incorrect Trials and Following Trials:")
    print(results[1])
    
    print("\n3. VTE Rate in High Speed Incorrect Trials and Following Trials (Same Trial Type):")
    print(results[2])
    
    print("\n4. Speed Comparison (High vs Low Speed Incorrect):")
    print(results[3])
    
    print("\n5. Correctness Comparison (High Speed Incorrect vs Correct):")
    print(results[4])
    
    return df, results

# Example usage:
if __name__ == "__main__":
    # Set your base path here
    base_path = paths.central
    
    # Run analysis with per-rat speed thresholds
    df, results = main_analysis(base_path)