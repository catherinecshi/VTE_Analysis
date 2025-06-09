import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os
from models import helper

def load_data(file_path):
    """
    Load the CSV data, handling both space-delimited and comma-delimited formats.
    """
    try:
        # Try reading as space-separated values first (as shown in the sample)
        df = pd.read_csv(file_path, delim_whitespace=True)
        
        # If that doesn't parse correctly, try standard CSV format
        if len(df.columns) <= 1:
            df = pd.read_csv(file_path)
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['trial_num', 'stim1', 'stim2', 'chosen', 'unchosen', 
                       'vte_occurred', 'stim1_uncertainty', 'stim2_uncertainty', 
                       'pair_relational_uncertainty', 'pair_roc_uncertainty', 'reward']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    return df

def create_continuous_trials(df):
    """
    Create a continuous trial numbering across all days.
    """
    # Convert day to string if needed for consistent sorting
    df['day'] = df['day'].astype(str)
    
    # Sort by day and trial_num to ensure correct order
    df = df.sort_values(['day', 'trial_num'])
    
    # Create a list of unique days in sorted order
    days = sorted(df['day'].unique())
    
    # Initialize a dictionary to store maximum trial number for each day
    max_trials = {}
    cumulative_max = 0
    
    # Calculate cumulative trial offsets for each day
    for i, day in enumerate(days):
        if i > 0:
            prev_day = days[i-1]
            max_trials[day] = max_trials[prev_day] + df[df['day'] == prev_day]['trial_num'].max()
        else:
            max_trials[day] = 0
    
    # Create a new column for continuous trial numbering
    df['continuous_trial'] = df.apply(
        lambda row: row['trial_num'] + max_trials[row['day']], 
        axis=1
    )
    
    return df, days, max_trials

def analyze_uncertainties(df):
    """
    Analyze and visualize the different types of uncertainties with continuous trial numbering.
    """
    # Create continuous trial numbering
    df, days, day_offsets = create_continuous_trials(df)
    
    # Define uncertainty columns
    uncertainty_columns = [
        'stim1_uncertainty', 
        'stim2_uncertainty', 
        'pair_relational_uncertainty', 
        'pair_roc_uncertainty'
    ]
    
    # Get continuous trial boundaries for each day (for plotting vertical lines)
    day_boundaries = [day_offsets[day] for day in days[1:]]
    
    # 1. Plot stimulus uncertainties (generally similar scale)
    plt.figure(figsize=(15, 6))
    plt.plot(df['continuous_trial'], df['stim1_uncertainty'], 
            marker='o', linestyle='-', markersize=4, alpha=0.7, label='Stim 1 Uncertainty')
    plt.plot(df['continuous_trial'], df['stim2_uncertainty'], 
            marker='x', linestyle='--', markersize=4, alpha=0.7, label='Stim 2 Uncertainty')
    
    # Add vertical lines to mark day boundaries
    for boundary in day_boundaries:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Add day labels
    for i, day in enumerate(days):
        if i < len(days) - 1:
            mid_point = (day_offsets[day] + day_offsets[days[i+1]]) / 2
        else:
            mid_point = (day_offsets[day] + df['continuous_trial'].max()) / 2
        plt.text(mid_point, plt.ylim()[1] * 0.95, f'Day {day}', 
                horizontalalignment='center', verticalalignment='top')
    
    plt.title('Stimulus Uncertainties Across All Trials')
    plt.xlabel('Continuous Trial Number')
    plt.ylabel('Uncertainty Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('stimulus_uncertainties_continuous.png')
    plt.show()
    
    # 2. Plot pair uncertainties (generally similar scale)
    plt.figure(figsize=(15, 6))
    plt.plot(df['continuous_trial'], df['pair_relational_uncertainty'], 
            marker='o', linestyle='-', markersize=4, alpha=0.7, label='Pair Relational Uncertainty')
    plt.plot(df['continuous_trial'], df['pair_roc_uncertainty'], 
            marker='x', linestyle='--', markersize=4, alpha=0.7, label='Pair ROC Uncertainty')
    
    # Add vertical lines to mark day boundaries
    for boundary in day_boundaries:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Add day labels
    for i, day in enumerate(days):
        if i < len(days) - 1:
            mid_point = (day_offsets[day] + day_offsets[days[i+1]]) / 2
        else:
            mid_point = (day_offsets[day] + df['continuous_trial'].max()) / 2
        plt.text(mid_point, plt.ylim()[1] * 0.95, f'Day {day}', 
                horizontalalignment='center', verticalalignment='top')
    
    plt.title('Pair Uncertainties Across All Trials')
    plt.xlabel('Continuous Trial Number')
    plt.ylabel('Uncertainty Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pair_uncertainties_continuous.png')
    plt.show()
    
    # 3. Create separate plots for each uncertainty type (different scales)
    for uncertainty in uncertainty_columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df['continuous_trial'], df[uncertainty], 
                marker='o', markersize=4, alpha=0.7, label=uncertainty.replace('_', ' ').title())
        
        # Add vertical lines to mark day boundaries
        for boundary in day_boundaries:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Add day labels
        for i, day in enumerate(days):
            if i < len(days) - 1:
                mid_point = (day_offsets[day] + day_offsets[days[i+1]]) / 2
            else:
                mid_point = (day_offsets[day] + df['continuous_trial'].max()) / 2
            plt.text(mid_point, plt.ylim()[1] * 0.95, f'Day {day}', 
                    horizontalalignment='center', verticalalignment='top')
        
        plt.title(f'{uncertainty.replace("_", " ").title()} Across All Trials')
        plt.xlabel('Continuous Trial Number')
        plt.ylabel('Uncertainty Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{uncertainty}_continuous.png')
        plt.show()
    
    # 4. Create normalized plot to compare all uncertainties on the same scale
    df_norm = df.copy()
    
    # Normalize each uncertainty column to 0-1 scale
    for col in uncertainty_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val > min_val:  # Avoid division by zero
            df_norm[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[f'{col}_norm'] = 0
    
    # Plot all normalized uncertainties on the same graph
    plt.figure(figsize=(15, 6))
    
    for col in uncertainty_columns:
        plt.plot(df_norm['continuous_trial'], df_norm[f'{col}_norm'], 
                marker='o', markersize=4, alpha=0.7, label=col.replace('_', ' ').title())
    
    # Add vertical lines to mark day boundaries
    for boundary in day_boundaries:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Add day labels
    for i, day in enumerate(days):
        if i < len(days) - 1:
            mid_point = (day_offsets[day] + day_offsets[days[i+1]]) / 2
        else:
            mid_point = (day_offsets[day] + df['continuous_trial'].max()) / 2
        plt.text(mid_point, 0.95, f'Day {day}', 
                horizontalalignment='center', verticalalignment='top')
    
    plt.title('All Normalized Uncertainties Across Trials')
    plt.xlabel('Continuous Trial Number')
    plt.ylabel('Normalized Uncertainty (0-1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_normalized_uncertainties_continuous.png')
    plt.show()
    
    # 5. Create a correlation plot between uncertainty measures and trial progression
    df['trial_bin'] = pd.cut(df['continuous_trial'], bins=10, labels=False)
    
    # Calculate mean uncertainties for each bin
    uncertainty_by_progress = df.groupby('trial_bin')[uncertainty_columns].mean()
    
    # Plot the progression
    plt.figure(figsize=(14, 8))
    for col in uncertainty_columns:
        plt.plot(uncertainty_by_progress.index, uncertainty_by_progress[col], 
                marker='o', linewidth=2, label=col.replace('_', ' ').title())
    
    plt.title('Progression of Uncertainties Throughout Experiment')
    plt.xlabel('Experiment Progress (Binned Trials)')
    plt.ylabel('Average Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('uncertainty_progression.png')
    plt.show()
    
    # 6. Create heatmap for visual comparison
    plt.figure(figsize=(12, 8))
    corr = df[uncertainty_columns + ['continuous_trial']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlations Between Uncertainty Measures and Trial Progression')
    plt.tight_layout()
    plt.savefig('uncertainty_correlations.png')
    plt.show()

def main(file_path='experiment_data.csv'):
    # Load the data
    df = load_data(file_path)
    
    if df is not None:
        # Analyze uncertainties
        analyze_uncertainties(df)
        print("Analysis complete. All plots have been saved as PNG files.")
    else:
        print("Could not analyze data. Please check the file format.")

if __name__ == "__main__":
    file_path = os.path.join(helper.BASE_PATH, "processed_data", "new_model_data", "TH510", "vte_uncertainty.csv")
    main(file_path)