import os
import pandas as pd
import matplotlib.pyplot as plt

from config.paths import paths
from analysis import betasort_analysis
from models import betasort_test
from visualization import betasort_plots

def save_plot(fig, output_dir, filename, dpi=300, close_fig=True):
    """
    Save a matplotlib figure to a file
    
    Parameters:
        - fig: matplotlib figure to save
        - output_dir: directory to save the figure
        - filename: name of the file
        - dpi: resolution of the image
        - close_fig: whether to close the figure after saving
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {filepath}")
    
    if close_fig:
        plt.close(fig)

# Create wrapper functions for the plotting functions to save figures
def plot_and_save(plot_func, model, output_path, filename_prefix, *args, **kwargs):
    """
    Execute a plotting function and save the result
    
    Parameters:
        - plot_func: Function that creates a plot
        - model: Model to plot
        - output_path: Directory to save the plot
        - filename_prefix: Prefix for the filename
        - *args, **kwargs: Additional arguments for the plotting function
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Call plotting function
    plot_func(model, *args, **kwargs)
    
    # Save figure
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, f"{filename_prefix}_{plot_func.__name__}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filepath}")
    plt.close()


data_path = paths.preprocessed_data_model
save_path = paths.betasort_data

for rat in os.listdir(data_path):
    if rat != "TH510":
        continue
    
    rat_path = os.path.join(data_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                continue

            file_path = os.path.join(root, file)
            file_csv = pd.read_csv(file_path)
            
            #best_tau, best_xi, best_threshold, best_performance = betasort_analysis.diff_evolution(file_csv, rat)
            #print(best_tau)
            #print(best_xi)
            #print(best_threshold)
            #print(best_performance)
            
            # check how well it matches up with the transitive inference results
            model, all_models, match_rates = betasort_analysis.compare_model_to_one_rat(file_csv, rat, tau=0.006, xi=0.99, threshold=0.85, test=True)
            
            pair_labels = ['AB', 'BC', 'CD', 'DE']
            fig, ax = betasort_plots.plot_ROC_uncertainty_across_days(all_models, mode='detailed', pair_labels=pair_labels, figsize=(12, 8), show_markers=False)
            
            print(match_rates)
            results = betasort_analysis.check_transitive_inference(model)
            print(results)
            
            # check positions over all days
            betasort_plots.plot_positions_across_days(all_models)
            betasort_plots.plot_uncertainty_across_days(all_models)
            
            # also check with binomial test
            #binomial_results = betasort_analysis.binomial_analysis_by_session(file_csv, rat)
            #print(binomial_results)
            
            betasort_plots.plot_stimulus_uncertainty(model)
            betasort_plots.plot_ROC_uncertainty(model)
            betasort_plots.plot_positions(model)
            betasort_plots.plot_beta_distributions(model)
            betasort_plots.plot_boundaries_history(model)
            