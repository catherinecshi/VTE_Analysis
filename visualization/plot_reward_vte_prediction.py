import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import stats

def load_vte_data(base_path, excluded_rats=None):
    """
    Load VTE uncertainty data for all rats, excluding specified rats and folders
    with 'Example', 'aggregated', or 'analysis' in the name.
    
    Args:
        base_path: Path to processed_data/new_model_data/
        excluded_rats: List of rat names to exclude
    
    Returns:
        Dictionary with rat names as keys and DataFrames as values
    """
    if excluded_rats is None:
        excluded_rats = []
    
    data = {}
    base_path = Path(base_path)
    
    for rat_dir in base_path.iterdir():
        if not rat_dir.is_dir():
            continue
            
        rat_name = rat_dir.name
        
        # Skip folders with excluded keywords
        if any(keyword.lower() in rat_name.lower() for keyword in ['example', 'aggregated', 'analysis']):
            continue
            
        # Skip explicitly excluded rats
        if rat_name in excluded_rats:
            continue
            
        vte_file = rat_dir / 'vte_uncertainty.csv'
        if vte_file.exists():
            try:
                df = pd.read_csv(vte_file)
                data[rat_name] = df
                print(f"Loaded data for {rat_name}: {len(df)} trials")
            except Exception as e:
                print(f"Error loading data for {rat_name}: {e}")
    
    return data

def calculate_trial_type_agreement(data):
    """
    Calculate the percentage of agreement between vte_occurred and inverse of reward
    for each trial type, averaged across all rats.
    
    Args:
        data: Dictionary of rat DataFrames
    
    Returns:
        Dictionary with trial types as keys and agreement percentages as values
    """
    trial_type_agreements = {}
    
    for rat_name, df in data.items():
        # Create inverse of reward (0 becomes 1, 1 becomes 0)
        df['inverse_reward'] = 1 - df['reward']
        
        # Calculate agreement for each trial (1 if same, 0 if different)
        df['agreement'] = (df['vte_occurred'] == df['inverse_reward']).astype(int)
        
        # Group by trial type (pair column) and calculate agreement percentage
        rat_agreements = df.groupby('pair')['agreement'].mean() * 100
        
        for trial_type, agreement in rat_agreements.items():
            if trial_type not in trial_type_agreements:
                trial_type_agreements[trial_type] = []
            trial_type_agreements[trial_type].append(agreement)
    
    # Average across all rats for each trial type
    avg_agreements = {trial_type: np.mean(agreements) 
                     for trial_type, agreements in trial_type_agreements.items()}
    
    return avg_agreements

def calculate_trial_type_vte_agreement(data):
    """
    Calculate the percentage of agreement between vte_occurred and inverse of reward
    for each trial type, but only for trials where both rat and inverse reward show VTE (both = 1).
    
    Args:
        data: Dictionary of rat DataFrames
    
    Returns:
        Dictionary with trial types as keys and VTE agreement percentages as values
    """
    trial_type_vte_agreements = {}
    
    for rat_name, df in data.items():
        # Create inverse of reward (0 becomes 1, 1 becomes 0)
        df['inverse_reward'] = 1 - df['reward']
        
        # Filter to only trials where both rat and inverse reward show VTE
        vte_trials = df[(df['vte_occurred'] == 1) & (df['inverse_reward'] == 1)]
        
        if len(vte_trials) == 0:
            continue
        
        # Group by trial type and count occurrences
        vte_counts = vte_trials.groupby('pair').size()
        total_counts = df.groupby('pair').size()
        
        # Calculate percentage of trials with concurrent VTE for each trial type
        for trial_type in total_counts.index:
            vte_count = vte_counts.get(trial_type, 0)
            total_count = total_counts[trial_type]
            vte_agreement_pct = (vte_count / total_count) * 100
            
            if trial_type not in trial_type_vte_agreements:
                trial_type_vte_agreements[trial_type] = []
            trial_type_vte_agreements[trial_type].append(vte_agreement_pct)
    
    # Average across all rats for each trial type
    avg_vte_agreements = {trial_type: np.mean(agreements) 
                         for trial_type, agreements in trial_type_vte_agreements.items()}
    
    return avg_vte_agreements

def calculate_vte_percentages(data):
    """
    Calculate VTE percentages for each rat-day combination.
    
    Args:
        data: Dictionary of rat DataFrames
    
    Returns:
        DataFrame with columns: rat, day, vte_occurred_pct, inverse_reward_vte_pct
    """
    results = []
    
    for rat_name, df in data.items():
        # Create inverse of reward (0 becomes 1, 1 becomes 0)
        df['inverse_reward'] = 1 - df['reward']
        
        # Group by day and calculate percentages
        day_stats = df.groupby('day').agg({
            'vte_occurred': lambda x: (x == 1).mean() * 100,
            'inverse_reward': lambda x: (x == 1).mean() * 100
        }).reset_index()
        
        day_stats['rat'] = rat_name
        results.append(day_stats)
    
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.rename(columns={
            'vte_occurred': 'vte_occurred_pct',
            'inverse_reward': 'inverse_reward_vte_pct'
        }, inplace=True)
        return combined_df
    else:
        return pd.DataFrame()

def plot_trial_type_agreement(trial_agreements, save_path=None):
    """
    Create bar plot of trial type agreement percentages.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    trial_types = list(trial_agreements.keys())
    agreements = list(trial_agreements.values())
    
    bars = ax.bar(trial_types, agreements)
    ax.set_xlabel('Trial Type')
    ax.set_ylabel('Agreement Percentage (%)')
    ax.set_title('Inverse Reward-Rat VTE Agreement by Trial Type')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, agreement in zip(bars, agreements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{agreement:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_trial_type_vte_agreement(trial_vte_agreements, save_path=None):
    """
    Create bar plot of trial type concurrent VTE percentages.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    trial_types = list(trial_vte_agreements.keys())
    vte_agreements = list(trial_vte_agreements.values())
    
    bars = ax.bar(trial_types, vte_agreements)
    ax.set_xlabel('Trial Type')
    ax.set_ylabel('Concurrent VTE Percentage (%)')
    ax.set_title('Percentage of Trials with Both Inverse Reward and Rat VTE by Trial Type')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, vte_agreement in zip(bars, vte_agreements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{vte_agreement:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_vte_scatter(vte_percentages, save_path=None):
    """
    Create scatter plot of VTE percentages.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(vte_percentages['vte_occurred_pct'], 
                        vte_percentages['inverse_reward_vte_pct'],
                        alpha=0.6, s=50)
    
    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(vte_percentages['vte_occurred_pct'], 
                                         vte_percentages['inverse_reward_vte_pct'])
    
    ax.set_xlabel('Rat VTE Percentage (%)')
    ax.set_ylabel('Inverse Reward VTE Percentage (%)')
    ax.set_title(f'Inverse Reward vs Rat VTE Percentages by Day\nr = {correlation:.3f}, p = {p_value:.4f}')
    
    # Add diagonal line for perfect agreement
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect Agreement')
    ax.legend()
    
    # Set equal aspect ratio and limits
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_vte_scatter_log(vte_percentages, save_path=None):
    """
    Create scatter plot of VTE percentages with log scale on both axes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Filter out zero values for log scale
    non_zero_data = vte_percentages[
        (vte_percentages['vte_occurred_pct'] > 0) & 
        (vte_percentages['inverse_reward_vte_pct'] > 0)
    ]
    
    if len(non_zero_data) == 0:
        print("No data points with non-zero values for log scale plot")
        return
    
    scatter = ax.scatter(non_zero_data['vte_occurred_pct'], 
                        non_zero_data['inverse_reward_vte_pct'],
                        alpha=0.6, s=50)
    
    # Calculate Pearson correlation for non-zero data
    correlation, p_value = stats.pearsonr(non_zero_data['vte_occurred_pct'], 
                                         non_zero_data['inverse_reward_vte_pct'])
    
    ax.set_xlabel('Rat VTE Percentage (%) - Log Scale')
    ax.set_ylabel('Inverse Reward VTE Percentage (%) - Log Scale')
    ax.set_title(f'Inverse Reward vs Rat VTE Percentages by Day (Log Scale)\nr = {correlation:.3f}, p = {p_value:.4f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add diagonal line for perfect agreement
    min_val = min(non_zero_data['vte_occurred_pct'].min(), non_zero_data['inverse_reward_vte_pct'].min())
    max_val = max(non_zero_data['vte_occurred_pct'].max(), non_zero_data['inverse_reward_vte_pct'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Agreement')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    base_path = "processed_data/new_model_data"
    
    # List of rats to exclude (modify as needed)
    excluded_rats = [
        # Add rat names here to exclude them
        'BP06', 'BP08', 'BP09', 'BP10', 'BP13'
    ]
    
    # Load data
    print("Loading VTE data...")
    data = load_vte_data(base_path, excluded_rats)
    
    if not data:
        print("No data loaded. Check file paths and exclusions.")
        return
    
    print(f"Loaded data for {len(data)} rats: {list(data.keys())}")
    
    # Calculate trial type agreements
    print("\nCalculating trial type agreements...")
    trial_agreements = calculate_trial_type_agreement(data)
    print(f"Found {len(trial_agreements)} trial types: {list(trial_agreements.keys())}")
    
    # Calculate trial type VTE agreements (concurrent VTE only)
    print("\nCalculating trial type VTE agreements...")
    trial_vte_agreements = calculate_trial_type_vte_agreement(data)
    print(f"Found {len(trial_vte_agreements)} trial types with VTE data: {list(trial_vte_agreements.keys())}")
    
    # Calculate VTE percentages
    print("\nCalculating VTE percentages...")
    vte_percentages = calculate_vte_percentages(data)
    print(f"Generated {len(vte_percentages)} rat-day combinations")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Plot 1: Trial type agreement
    plot_trial_type_agreement(trial_agreements, 'reward_trial_type_agreement.png')
    
    # Plot 2: Trial type concurrent VTE
    plot_trial_type_vte_agreement(trial_vte_agreements, 'reward_trial_type_vte_agreement.png')
    
    # Plot 3: VTE percentage scatter
    plot_vte_scatter(vte_percentages, 'reward_vte_percentage_scatter.png')
    
    # Plot 4: VTE percentage scatter with log scale
    plot_vte_scatter_log(vte_percentages, 'reward_vte_percentage_scatter_log.png')
    
    print("\nPlots saved as 'reward_trial_type_agreement.png', 'reward_trial_type_vte_agreement.png', 'reward_vte_percentage_scatter.png', and 'reward_vte_percentage_scatter_log.png'")

if __name__ == "__main__":
    main()