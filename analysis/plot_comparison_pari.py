import os
import json
import matplotlib.pyplot as plt
import numpy as np

from config.paths import paths
from visualization import betasort_plots

def plot_and_save(plot_func, output_path, filename_prefix, *args, **kwargs):
    """
    Execute a plotting function and save the result
    
    Parameters:
        - plot_func: Function that creates a plot
        - output_path: Directory to save the plot
        - filename_prefix: Prefix for the filename
        - *args, **kwargs: Additional arguments for the plotting function
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Define the save path
    filepath = os.path.join(output_path, f"{filename_prefix}_{plot_func.__name__}.png")
    
    # Set save parameter in kwargs
    kwargs['save'] = filepath
    
    # Call plotting function with save parameter
    result = plot_func(*args, **kwargs)
    
    print(f"Saved plot to {filepath}")
    
    return result

def main():
    # Define paths
    save_path = paths.betasort_data
    aggregated_data_path = os.path.join(save_path, "aggregated_adjacent_pair_analysis.json")
    aggregated_plots_dir = os.path.join(save_path, "aggregated_plots")
    
    # Check if the aggregated data file exists
    if not os.path.exists(aggregated_data_path):
        print(f"Error: Aggregated data file not found at {aggregated_data_path}")
        print("Please run the main analysis script first to generate the aggregated data.")
        return
    
    # Load the aggregated data
    print(f"Loading aggregated data from {aggregated_data_path}")
    with open(aggregated_data_path, 'r') as f:
        aggregated_data = json.load(f)
    
    # Extract the data needed for plotting
    pair_names = aggregated_data['pair_names']
    rat_rates = aggregated_data['rat_rates']
    post_model_rates = aggregated_data['post_model_rates']
    rat_counts = aggregated_data['rat_counts']
    total_rats = aggregated_data['total_rats']
    
    print(f"Loaded data for {len(pair_names)} pairs across {total_rats} rats")
    print(f"Pairs: {pair_names}")
    
    # Generate the post-model vs rat comparison plot
    print("Generating post-model vs rat comparison plot...")
    
    plot_and_save(
        betasort_plots.plot_post_model_vs_rat_comparison,
        aggregated_plots_dir,
        "all_rats_post_model_vs_rat_comparison",
        pair_names,
        rat_rates,
        post_model_rates,
        rat_counts,
        total_rats=total_rats
    )
    
    print("Plot generation complete!")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    correlation = np.corrcoef(rat_rates, post_model_rates)[0, 1]
    mse = np.mean((np.array(rat_rates) - np.array(post_model_rates)) ** 2)
    mae = np.mean(np.abs(np.array(rat_rates) - np.array(post_model_rates)))
    
    print(f"Correlation between rat and post-model performance: {correlation:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    
    # Show pair-by-pair comparison
    print("\n=== PAIR-BY-PAIR COMPARISON ===")
    for i, pair in enumerate(pair_names):
        print(f"{pair}: Rat={rat_rates[i]:.3f}, Model={post_model_rates[i]:.3f}, "
              f"Diff={abs(rat_rates[i] - post_model_rates[i]):.3f}, n={rat_counts[i]}")

if __name__ == "__main__":
    main()