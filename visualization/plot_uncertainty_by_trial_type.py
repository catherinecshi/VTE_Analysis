import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration - easily exclude specific rats by adding them to this list
EXCLUDED_RATS = ["BP06", "BP07", "BP08", "BP09", "BP10", "BP13"]  # e.g., ['BP06', 'TH405'] to exclude those rats

# Base directory
BASE_DIR = "/Users/catpillow/Documents/VTE_Analysis/processed_data/new_model_data"

def find_valid_rat_directories():
    """
    Find valid rat directories (BP* and TH*) excluding Example/analysis/aggregated folders.
    
    Returns:
        list: List of valid rat directory paths
    """
    base_path = Path(BASE_DIR)
    valid_dirs = []
    
    # Get all directories in the base path
    for item in base_path.iterdir():
        if item.is_dir():
            dir_name = item.name
            
            # Check if it's a rat directory (starts with BP or TH)
            if dir_name.startswith(('BP', 'TH')):
                # Exclude directories containing specific keywords
                exclude_keywords = ['Example', 'analysis', 'aggregated']
                if not any(keyword in dir_name for keyword in exclude_keywords):
                    # Check if not in excluded rats list
                    if dir_name not in EXCLUDED_RATS:
                        # Check if vte_uncertainty.csv exists
                        csv_path = item / 'vte_uncertainty.csv'
                        if csv_path.exists():
                            valid_dirs.append(str(item))
                        else:
                            print(f"Warning: {dir_name} missing vte_uncertainty.csv file")
                    else:
                        print(f"Excluding rat {dir_name} (in EXCLUDED_RATS list)")
    
    print(f"Found {len(valid_dirs)} valid rat directories")
    return valid_dirs

def load_rat_data(rat_dirs):
    """
    Load vte_uncertainty.csv data from all valid rat directories.
    
    Args:
        rat_dirs (list): List of rat directory paths
        
    Returns:
        dict: Dictionary with rat_id as key and DataFrame as value
    """
    rat_data = {}
    
    for rat_dir in rat_dirs:
        rat_id = os.path.basename(rat_dir)
        csv_path = os.path.join(rat_dir, 'vte_uncertainty.csv')
        
        try:
            df = pd.read_csv(csv_path)
            # Create trial type column by combining stim1 and stim2
            df['trial_type'] = df['stim1'].astype(str) + '-' + df['stim2'].astype(str)
            rat_data[rat_id] = df
            print(f"Loaded {len(df)} trials for rat {rat_id}")
        except Exception as e:
            print(f"Error loading data for rat {rat_id}: {e}")
    
    return rat_data

def calculate_per_rat_averages(rat_data):
    """
    Calculate average uncertainty values per trial type for each rat.
    
    Args:
        rat_data (dict): Dictionary with rat_id as key and DataFrame as value
        
    Returns:
        dict: Dictionary with rat_id as key and averaged DataFrame as value
    """
    uncertainty_columns = ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']
    rat_averages = {}
    
    for rat_id, df in rat_data.items():
        # Group by trial type and calculate means
        rat_avg = df.groupby('trial_type')[uncertainty_columns].mean().reset_index()
        rat_avg['rat_id'] = rat_id
        rat_averages[rat_id] = rat_avg
        print(f"Calculated averages for {len(rat_avg)} trial types in rat {rat_id}")
    
    return rat_averages

def aggregate_across_rats(rat_averages):
    """
    Aggregate uncertainty values across rats, calculating means and SEM for each trial type.
    
    Args:
        rat_averages (dict): Dictionary with rat_id as key and averaged DataFrame as value
        
    Returns:
        dict: Dictionary with uncertainty type as key and aggregated DataFrame as value
    """
    uncertainty_columns = ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']
    
    # Combine all rat data
    combined_data = pd.concat(rat_averages.values(), ignore_index=True)
    
    # Initialize results dictionary
    aggregated_results = {}
    
    for uncertainty_col in uncertainty_columns:
        # Group by trial type and calculate statistics
        stats = combined_data.groupby('trial_type')[uncertainty_col].agg([
            'mean', 'std', 'count', 'sem'
        ]).reset_index()
        
        # Rename columns for clarity
        stats.columns = ['trial_type', 'mean_uncertainty', 'std_uncertainty', 'n_rats', 'sem_uncertainty']
        
        # Sort trial types for consistent plotting
        stats = stats.sort_values('trial_type')
        
        aggregated_results[uncertainty_col] = stats
        print(f"Aggregated {uncertainty_col} across {len(stats)} trial types")
    
    return aggregated_results

def calculate_per_rat_averages_by_stimulus(rat_data, group_by_column):
    """
    Calculate average uncertainty values per stimulus value for each rat.
    
    Args:
        rat_data (dict): Dictionary with rat_id as key and DataFrame as value
        group_by_column (str): Column to group by ('chosen', 'unchosen', 'stim1', 'stim2')
        
    Returns:
        dict: Dictionary with rat_id as key and averaged DataFrame as value
    """
    uncertainty_columns = ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']
    rat_averages = {}
    
    for rat_id, df in rat_data.items():
        # Group by the specified stimulus column and calculate means
        rat_avg = df.groupby(group_by_column)[uncertainty_columns].mean().reset_index()
        rat_avg['rat_id'] = rat_id
        rat_averages[rat_id] = rat_avg
        print(f"Calculated averages for {len(rat_avg)} {group_by_column} values in rat {rat_id}")
    
    return rat_averages

def aggregate_across_rats_by_stimulus(rat_averages, group_by_column):
    """
    Aggregate uncertainty values across rats for stimulus-based grouping.
    
    Args:
        rat_averages (dict): Dictionary with rat_id as key and averaged DataFrame as value
        group_by_column (str): Column that was grouped by
        
    Returns:
        dict: Dictionary with uncertainty type as key and aggregated DataFrame as value
    """
    uncertainty_columns = ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']
    
    # Combine all rat data
    combined_data = pd.concat(rat_averages.values(), ignore_index=True)
    
    # Initialize results dictionary
    aggregated_results = {}
    
    for uncertainty_col in uncertainty_columns:
        # Group by stimulus value and calculate statistics
        stats = combined_data.groupby(group_by_column)[uncertainty_col].agg([
            'mean', 'std', 'count', 'sem'
        ]).reset_index()
        
        # Rename columns for clarity
        stats.columns = [group_by_column, 'mean_uncertainty', 'std_uncertainty', 'n_rats', 'sem_uncertainty']
        
        # Sort stimulus values for consistent plotting
        stats = stats.sort_values(group_by_column)
        
        aggregated_results[uncertainty_col] = stats
        print(f"Aggregated {uncertainty_col} by {group_by_column} across {len(stats)} stimulus values")
    
    return aggregated_results

def create_uncertainty_plots(aggregated_results, rat_averages):
    """
    Create three separate plots for each uncertainty type.
    
    Args:
        aggregated_results (dict): Dictionary with uncertainty type as key and aggregated DataFrame as value
        rat_averages (dict): Dictionary with rat_id as key and averaged DataFrame as value
    """
    uncertainty_labels = {
        'stim1_uncertainty': 'Stimulus 1 Uncertainty',
        'stim2_uncertainty': 'Stimulus 2 Uncertainty', 
        'pair_roc_uncertainty': 'Pair ROC Uncertainty'
    }
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (uncertainty_col, label) in enumerate(uncertainty_labels.items()):
        ax = axes[i]
        data = aggregated_results[uncertainty_col]
        
        # Create bar plot with error bars
        bars = ax.bar(
            data['trial_type'], 
            data['mean_uncertainty'],
            yerr=data['sem_uncertainty'],
            capsize=5,
            alpha=0.7,
            color=plt.cm.viridis(i/3)
        )
        
        # Customize plot
        ax.set_xlabel('Trial Type', fontsize=12)
        ax.set_ylabel('Mean Uncertainty', fontsize=12)
        ax.set_title(f'{label}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample size annotation
        max_y = ax.get_ylim()[1]
        for j, (_, row) in enumerate(data.iterrows()):
            ax.text(j, max_y * 0.95, f'n={int(row["n_rats"])}', 
                   ha='center', va='top', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('uncertainty_by_trial_type_all.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual plots for each uncertainty type
    for uncertainty_col, label in uncertainty_labels.items():
        plt.figure(figsize=(10, 6))
        data = aggregated_results[uncertainty_col]
        
        bars = plt.bar(
            data['trial_type'], 
            data['mean_uncertainty'],
            yerr=data['sem_uncertainty'],
            capsize=8,
            alpha=0.8,
            color=plt.cm.viridis(0.5),
            edgecolor='black',
            linewidth=0.5
        )
        
        plt.xlabel('Trial Type', fontsize=14)
        plt.ylabel('Mean Uncertainty', fontsize=14) 
        plt.title(f'{label} by Trial Type', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add sample size annotations
        max_y = plt.gca().get_ylim()[1]
        for i, (_, row) in enumerate(data.iterrows()):
            plt.text(i, max_y * 0.95, f'n={int(row["n_rats"])}', 
                    ha='center', va='top', fontsize=10, alpha=0.8)
        
        # Add statistical info text box
        n_trial_types = len(data)
        total_rats = len(set(pd.concat([df['rat_id'] for df in rat_averages.values()])))
        info_text = f'Trial Types: {n_trial_types}\nTotal Rats: {total_rats}'
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual plots
        filename = f'uncertainty_by_trial_type_{uncertainty_col}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot: {filename}")

def create_uncertainty_plots_by_stimulus(aggregated_results, rat_averages, group_by_column):
    """
    Create three separate plots for each uncertainty type, grouped by stimulus.
    
    Args:
        aggregated_results (dict): Dictionary with uncertainty type as key and aggregated DataFrame as value
        rat_averages (dict): Dictionary with rat_id as key and averaged DataFrame as value
        group_by_column (str): Column that was grouped by ('chosen', 'unchosen', 'stim1', 'stim2')
    """
    uncertainty_labels = {
        'stim1_uncertainty': 'Stimulus 1 Uncertainty',
        'stim2_uncertainty': 'Stimulus 2 Uncertainty', 
        'pair_roc_uncertainty': 'Pair ROC Uncertainty'
    }
    
    group_labels = {
        'chosen': 'Chosen Stimulus',
        'unchosen': 'Unchosen Stimulus',
        'stim1': 'Stimulus 1',
        'stim2': 'Stimulus 2'
    }
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (uncertainty_col, label) in enumerate(uncertainty_labels.items()):
        ax = axes[i]
        data = aggregated_results[uncertainty_col]
        
        # Create bar plot with error bars
        bars = ax.bar(
            data[group_by_column], 
            data['mean_uncertainty'],
            yerr=data['sem_uncertainty'],
            capsize=5,
            alpha=0.7,
            color=plt.cm.viridis(i/3)
        )
        
        # Customize plot
        ax.set_xlabel(f'{group_labels[group_by_column]} Value', fontsize=12)
        ax.set_ylabel('Mean Uncertainty', fontsize=12)
        ax.set_title(f'{label} by {group_labels[group_by_column]}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample size annotation
        max_y = ax.get_ylim()[1]
        for j, (_, row) in enumerate(data.iterrows()):
            ax.text(j, max_y * 0.95, f'n={int(row["n_rats"])}', 
                   ha='center', va='top', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'uncertainty_by_{group_by_column}_all.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual plots for each uncertainty type
    for uncertainty_col, label in uncertainty_labels.items():
        plt.figure(figsize=(10, 6))
        data = aggregated_results[uncertainty_col]
        
        bars = plt.bar(
            data[group_by_column], 
            data['mean_uncertainty'],
            yerr=data['sem_uncertainty'],
            capsize=8,
            alpha=0.8,
            color=plt.cm.viridis(0.5),
            edgecolor='black',
            linewidth=0.5
        )
        
        plt.xlabel(f'{group_labels[group_by_column]} Value', fontsize=14)
        plt.ylabel('Mean Uncertainty', fontsize=14) 
        plt.title(f'{label} by {group_labels[group_by_column]}', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add sample size annotations
        max_y = plt.gca().get_ylim()[1]
        for i, (_, row) in enumerate(data.iterrows()):
            plt.text(i, max_y * 0.95, f'n={int(row["n_rats"])}', 
                    ha='center', va='top', fontsize=10, alpha=0.8)
        
        # Add statistical info text box
        n_values = len(data)
        total_rats = len(set(pd.concat([df['rat_id'] for df in rat_averages.values()])))
        info_text = f'{group_labels[group_by_column]} Values: {n_values}\nTotal Rats: {total_rats}'
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual plots
        filename = f'uncertainty_by_{group_by_column}_{uncertainty_col}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot: {filename}")

def main():
    """
    Main function to run the uncertainty by trial type analysis.
    """
    print("Starting uncertainty by trial type analysis...")
    print(f"Excluded rats: {EXCLUDED_RATS if EXCLUDED_RATS else 'None'}")
    
    # Step 1: Find valid rat directories
    rat_dirs = find_valid_rat_directories()
    if not rat_dirs:
        print("No valid rat directories found. Check the base directory and file structure.")
        return
    
    # Step 2: Load data from all rats
    rat_data = load_rat_data(rat_dirs)
    if not rat_data:
        print("No data could be loaded. Check file formats and paths.")
        return
    
    # Step 3: Calculate per-rat averages by trial type
    rat_averages = calculate_per_rat_averages(rat_data)
    
    # Step 4: Aggregate across rats
    aggregated_results = aggregate_across_rats(rat_averages)
    
    # Step 5: Create plots for trial type analysis
    print("\n=== TRIAL TYPE ANALYSIS ===")
    create_uncertainty_plots(aggregated_results, rat_averages)
    
    # Print summary statistics for trial type analysis
    print("\nTrial Type Analysis Summary:")
    for uncertainty_col in ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']:
        data = aggregated_results[uncertainty_col]
        print(f"\n{uncertainty_col}:")
        print(f"  Trial types analyzed: {len(data)}")
        print(f"  Trial types: {', '.join(data['trial_type'].values)}")
        print(f"  Uncertainty range: {data['mean_uncertainty'].min():.4f} - {data['mean_uncertainty'].max():.4f}")
    
    # Step 6: Run stimulus-based analyses
    stimulus_analyses = ['chosen', 'unchosen', 'stim1', 'stim2']
    
    for stimulus_col in stimulus_analyses:
        print(f"\n=== {stimulus_col.upper()} STIMULUS ANALYSIS ===")
        
        # Calculate per-rat averages by stimulus
        stimulus_rat_averages = calculate_per_rat_averages_by_stimulus(rat_data, stimulus_col)
        
        # Aggregate across rats
        stimulus_aggregated_results = aggregate_across_rats_by_stimulus(stimulus_rat_averages, stimulus_col)
        
        # Create plots
        create_uncertainty_plots_by_stimulus(stimulus_aggregated_results, stimulus_rat_averages, stimulus_col)
        
        # Print summary statistics
        print(f"\n{stimulus_col.capitalize()} Stimulus Analysis Summary:")
        for uncertainty_col in ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']:
            data = stimulus_aggregated_results[uncertainty_col]
            stimulus_values = data[stimulus_col].astype(str).values
            print(f"\n{uncertainty_col}:")
            print(f"  {stimulus_col.capitalize()} values analyzed: {len(data)}")
            print(f"  {stimulus_col.capitalize()} values: {', '.join(stimulus_values)}")
            print(f"  Uncertainty range: {data['mean_uncertainty'].min():.4f} - {data['mean_uncertainty'].max():.4f}")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("Generated plots:")
    print("- Trial type analysis: uncertainty_by_trial_type_*.png")
    print("- Chosen stimulus analysis: uncertainty_by_chosen_*.png") 
    print("- Unchosen stimulus analysis: uncertainty_by_unchosen_*.png")
    print("- Stim1 analysis: uncertainty_by_stim1_*.png")
    print("- Stim2 analysis: uncertainty_by_stim2_*.png")
    print("="*50)

if __name__ == "__main__":
    main()