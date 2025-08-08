import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import glob

def load_vte_data(exclude_rats=None, base_path="/Users/catpillow/Documents/VTE_Analysis/processed_data/VTE_values"):
    """Load all zIdPhi CSV files from the processed data directory.
    
    Args:
        exclude_rats (list, optional): List of rat IDs to exclude from analysis
        base_path (str): Path to the VTE values directory
    """
    if exclude_rats is None:
        exclude_rats = []
    
    pattern = os.path.join(base_path, "*", "*", "zIdPhi_day_*.csv")
    files = glob.glob(pattern)
    
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Extract rat and day from file path
            parts = file.split(os.sep)
            rat = parts[-3]
            day = parts[-2]
            
            # Skip if rat is in exclude list
            if rat in exclude_rats:
                print(f"Excluding rat {rat}")
                continue
            
            if day == "Day1":
                continue
                
            df['Rat'] = rat
            df['Day_Label'] = day
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        raise ValueError("No CSV files found. Check the data path.")
    
    return pd.concat(all_data, ignore_index=True)

def plot_zlength_by_trial_type(exclude_rats=None, save_path=None):
    """Plot average zLength values by trial type with SEM error bars.
    
    Args:
        exclude_rats (list, optional): List of rat IDs to exclude from analysis
        save_path (str, optional): Path to save the plot
    """
    # Load data
    data = load_vte_data(exclude_rats=exclude_rats)
    
    # Calculate mean and SEM for each trial type
    stats_df = data.groupby('Trial_Type')['zLength'].agg([
        'mean', 
        'count',
        lambda x: stats.sem(x, nan_policy='omit')
    ]).reset_index()
    stats_df.columns = ['Trial_Type', 'mean_zLength', 'count', 'sem_zLength']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar plot with error bars
    bars = ax.bar(stats_df['Trial_Type'], stats_df['mean_zLength'], 
                  yerr=stats_df['sem_zLength'], capsize=5, alpha=0.8, 
                  color=['#1f77b4', '#ff7f0e'])
    
    # Customize plot
    ax.set_xlabel('Trial Type', fontsize=12)
    ax.set_ylabel('Average zLength', fontsize=12)
    ax.set_title('Average zLength by Trial Type', fontsize=14, fontweight='bold')
    
    # Add sample sizes as text below x-axis labels
    for i, (bar, count) in enumerate(zip(bars, stats_df['count'])):
        ax.text(bar.get_x() + bar.get_width()/2., -0.15, 
                f'n={count}', ha='center', va='top', fontsize=9,
                transform=ax.get_xaxis_transform())
    
    # Format x-axis
    ax.set_xticks(stats_df['Trial_Type'])
    ax.set_xticklabels([f'Type {int(t)}' for t in stats_df['Trial_Type']])
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Print summary statistics
    print("Summary Statistics:")
    print("=" * 40)
    for _, row in stats_df.iterrows():
        print(f"Trial Type {int(row['Trial_Type'])}:")
        print(f"  Mean zLength: {row['mean_zLength']:.3f} ± {row['sem_zLength']:.3f}")
        print(f"  Sample size: {row['count']}")
        print()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    # Create the plot
    # exclude specific rats from analysis
    plot_zlength_by_trial_type(exclude_rats=['BP06', 'BP07', 'BP08', 'BP09', 'BP10'], save_path="zlength_by_trial_type.png")
    #plot_zlength_by_trial_type(save_path="zlength_by_trial_type.png")