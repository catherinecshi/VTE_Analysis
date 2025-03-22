import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import helper
from src import betasort

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


data_path = os.path.join(helper.BASE_PATH, "processed_data", "data_for_model")
save_path = os.path.join(helper.BASE_PATH, "processed_data", "new_model_data")

for rat in os.listdir(data_path):
    if "TH510" not in rat:
        continue

    rat_path = os.path.join(data_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                continue

            file_path = os.path.join(root, file)
            file_csv = pd.read_csv(file_path)
            
            # analyze the data sequentially
            """
            final_model, all_day_models = betasort.analyze_one_rat(file_csv, rat)
            
            # analyze and plot for each day
            for day, model in all_day_models.items():
                print(f"Day {day}")
                print(f"    Number of stimuli: {model.n_stimuli}")
                
                betasort.plot_positions(model)
                betasort.plot_uncertainty(model)
                betasort.plot_beta_distributions(model)
                betasort.plot_boundaries_history(model)
            """
            
            # check the heatmaps
            best_xi, best_tau, best_threshold, best_performance, param_performances, results_df, summary_df = betasort.find_optimal_threshold(file_csv, rat)
            print(f"Best parameters: xi={best_xi:.3f}, tau={best_tau:.3f}, avg_performance={best_performance:.3f}")

            # save dataframes
            save_path_2 = os.path.join(save_path, rat)
            results_df_path = os.path.join(save_path, rat, "results.csv")
            summary_df_path = os.path.join(save_path, rat, "summary.csv")
            
            if os.path.exists(os.path.join(save_path, rat)):
                results_df.to_csv(results_df_path)
                summary_df.to_csv(summary_df_path)
            else:
                os.mkdir(os.path.join(save_path, rat))
                results_df.to_csv(results_df_path)
                summary_df.to_csv(summary_df_path)
            
            # sort days
            unique_days = sorted(file_csv['Day'].unique())
            
            # Plot overall heatmap
            fig, ax = betasort.parameter_performance_heatmap_with_threshold(
                param_performances,
                title=f'Overall Performance (Threshold = {best_threshold:.3f})',
                fixed_param=2,  # Fix threshold
                fixed_value=best_threshold
            )
            save_plot(fig, save_path_2, f"{rat}_xi_tau_heatmap_threshold{best_threshold:.3f}.png")
            
             # Plot overall xi-threshold heatmap with fixed best tau
            fig, ax = betasort.parameter_performance_heatmap_with_threshold(
                param_performances,
                title=f'Overall Performance (Tau = {best_tau:.3f})',
                fixed_param=1,  # Fix tau
                fixed_value=best_tau
            )
            save_plot(fig, save_path_2, f"{rat}_xi_threshold_heatmap_tau{best_tau:.3f}.png")
            
            # Plot overall tau-threshold heatmap with fixed best xi
            fig, ax = betasort.parameter_performance_heatmap_with_threshold(
                param_performances,
                title=f'Overall Performance (Xi = {best_xi:.3f})',
                fixed_param=0,  # Fix xi
                fixed_value=best_xi
            )
            save_plot(fig, save_path_2, f"{rat}_tau_threshold_heatmap_xi{best_xi:.3f}.png")
            
            avg_param_perf = {params: np.mean(day_rates) for params, day_rates in param_performances.items()}
            print("Plotting overall parameter performance heatmap...")
            betasort.parameter_performance_heatmap(avg_param_perf)

            # Plot individual day heatmaps
            print("Plotting day-by-day parameter performance heatmaps...")
            for day_idx, day in enumerate(unique_days):
                # Create a heatmap for this specific day
                day_param_perf = {params: day_rates[day_idx] for params, day_rates in param_performances.items()}
                betasort.parameter_performance_heatmap(day_param_perf)

            # Run analysis with best parameters
            final_model, all_day_models, _ = betasort.compare_model_to_one_rat(
                file_csv, rat, tau=best_tau, xi=best_xi, threshold=best_threshold
            )

            # Plot results for each day
            for day, model in all_day_models.items():
                day_plots_dir = os.path.join(save_path_2, f"day_{day}")
                os.makedirs(day_plots_dir, exist_ok=True)
                print(f"Plotting results for Day {day}...")
                
                # Get stimulus labels for this day
                day_data = file_csv[file_csv['Day'] == day]
                all_indices = np.unique(np.concatenate([day_data['first'].values, day_data['second'].values]))
                index_to_letter = {idx: chr(65 + i) for i, idx in enumerate(sorted(all_indices))}
                stimulus_labels = [f"Stimulus {index_to_letter.get(i, i)}" for i in range(model.n_stimuli)]
                
                # Generate and save all plots for this day
                plot_and_save(
                    betasort.plot_positions, 
                    all_day_models[day], 
                    day_plots_dir, 
                    f"{rat}_day{day}", 
                    stimulus_labels
                )
                
                plot_and_save(
                    betasort.plot_uncertainty, 
                    all_day_models[day], 
                    day_plots_dir, 
                    f"{rat}_day{day}", 
                    stimulus_labels
                )
                
                plot_and_save(
                    betasort.plot_beta_distributions, 
                    all_day_models[day], 
                    day_plots_dir, 
                    f"{rat}_day{day}", 
                    stimulus_labels=stimulus_labels
                )
                
                plot_and_save(
                    betasort.plot_boundaries_history, 
                    all_day_models[day], 
                    day_plots_dir, 
                    f"{rat}_day{day}", 
                    stimulus_labels
                )