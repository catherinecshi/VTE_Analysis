import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path

from src import helper

base_path = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison")

def load_and_process_data():
    """
    Load all ti_comparison.csv files from the specified directory structure and process them.
    
    The function finds all CSV files, extracts the rat_id from the path, calculates match rates,
    and then aggregates statistics by model across all files.
    
    Parameters:
    - base_path: Base directory containing all the model comparison data
    
    Returns:
    - model_stats: DataFrame with statistics grouped by model
    - combined_df: Combined DataFrame with all processed data
    """
    # List to store DataFrames from each file
    all_dfs = []
    
    # Find all ti_comparison.csv files
    pattern = os.path.join(base_path, '**', 'ti_comparison.csv')
    file_paths = glob.glob(pattern, recursive=True)
    
    if not file_paths:
        raise FileNotFoundError(f"No matching CSV files found in {base_path}")
    
    print(f"Found {len(file_paths)} CSV files to process")
    
    # Process each file
    for file_path in file_paths:
        # Extract the rat_id from the path
        rat_id = Path(file_path).parent.name
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add the rat_id as a column
        df['rat_id'] = rat_id
        
        # Calculate match rate as a percentage (1 - error)
        # The error column already represents the difference between model_accuracy and actual_accuracy
        df['match_rate'] = (1 - abs(df['error'])) * 100
        
        all_dfs.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Group by model and calculate average match rate and SEM
    model_stats = combined_df.groupby('model').agg(
        avg_match_rate=('match_rate', 'mean'),
        sem_match_rate=('match_rate', lambda x: x.sem()),  # Calculate SEM across files
        count=('match_rate', 'count')
    ).reset_index()
    
    return model_stats, combined_df

def plot_model_match_rates(model_stats, combined_df, figsize=(12, 8)):
    """
    Plot the average match rates with SEM error bars.
    
    Creates a bar plot showing the average match rate for each model, with error bars
    representing the standard error of the mean.
    
    Parameters:
        - model_stats: DataFrame with model statistics (must contain columns: 'model', 
                      'avg_match_rate', 'sem_match_rate', 'count')
        - output_path: Path to save the plot (if None, plot is just displayed)
        - figsize: Figure size as a tuple (width, height)
    """
    # Sort by match rate for better visualization
    model_order = model_stats.sort_values('avg_match_rate', ascending=False)['model'].tolist()
    
    # Create figure with specified size
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    # Create bar plot without error bars first
    boxplot = sns.boxplot(
        x='model', 
        y='match_rate',  # This should be the raw match rate column with individual measurements
        data=combined_df,
        palette="viridis",
        order=model_order  # Use sorted order from model_stats
    )
    
    # Add individual data points as a stripplot
    sns.stripplot(
        x='model', 
        y='match_rate',  # Same column as above
        data=combined_df,
        color='black',
        alpha=0.5,
        jitter=True,
        order=model_order  # Use same order as boxplot
    )

    # Add error bars manually for better control over appearance
    for idx in range(len(model_stats)):
        boxplot.errorbar(
            x=idx,  # This is the correct x-position (0, 1, 2, ...)
            y=model_stats.iloc[idx]['avg_match_rate'],
            yerr=model_stats.iloc[idx]['sem_match_rate'],
            color='black',
            capsize=5,
            fmt='none'  # This prevents adding markers
        )
    
    # Add perfect match reference line at 100%
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot appearance
    plt.title("Transitive Inference Match Rates", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Match Rate (%)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for i, model in enumerate(model_order):
        count = len(combined_df[combined_df['model'] == model])
        plt.text(
            i,  # x position (0, 1, 2, ...)
            plt.ylim()[0] + 2,  # y position with small offset from bottom
            f"n={count}",
            ha='center',
            fontsize=9,
            color='black'
        )
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot if output_path is provided
    output_path = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison", "TI_match_rate_2.png")
    plt.savefig(output_path, dpi=300)
    
    return boxplot

def main():
    """
    Main function to execute the data processing and plotting.
    
    Parameters:
    - base_path: Base directory for the data files
    - output_path: Path to save the main plot
    """
    try:
        # Load and process data
        model_stats, combined_df = load_and_process_data()
        
        print("Model statistics:")
        print(model_stats)
        
        # Plot the main results
        plot_model_match_rates(model_stats, combined_df)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()