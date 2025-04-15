import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import helper
from src import betasort

# Create wrapper functions for the plotting functions to save figures
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

data_path = os.path.join(helper.BASE_PATH, "processed_data", "data_for_model")
save_path = os.path.join(helper.BASE_PATH, "processed_data", "new_model_data")

for rat in os.listdir(data_path):
    if "BP06" in rat or "BP08" in rat:
        continue
    
    rat_path = os.path.join(data_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                continue

            file_path = os.path.join(root, file)
            file_csv = pd.read_csv(file_path)
            
            try:
                # analyze the data sequentially
                all_results = betasort.analyze_betasort_comprehensive(file_csv, rat)
            except Exception as e:
                print(rat, file_path)
                print(e)
                continue
            
            # check with transitive inference
            all_models = all_results["all_models"]
            final_day = max(all_models.keys())
            ti_result = betasort.check_transitive_inference(all_models[final_day])
            ti_result_serializable = {f"{k[0]},{k[1]}": v for k, v in ti_result.items()}
            ti_result_json = json.dumps(ti_result_serializable)
            
            # check uncertainty vs vte correspondence
            pair_results = betasort.analyze_correlations(all_results["pair_vte_df"])
            
            # save results
            results = {"rat": rat,
                          "best_xi": all_results["best_xi"],
                          "best_tau": all_results["best_tau"],
                          "best_threshold": all_results["best_threshold"],
                          "best_performance": all_results["best_performance"],
                          "session_regression": all_results["session_predictions_regression"],
                          "session_binomial_test": all_results["session_results_binomial"],
                          "TI_Result": ti_result_json
                          }
            results_df = pd.DataFrame([results])
            pair_vte_df = pd.DataFrame(all_results["pair_vte_df"])
            
            results_dir = os.path.join(save_path, rat)
            results_path = os.path.join(save_path, rat, "results.csv")
            vte_path = os.path.join(save_path, rat, "vte_uncertainty.csv")
            
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            results_df.to_csv(results_path, index=False)
            pair_vte_df.to_csv(vte_path, index=False)
            
            uncertainty_vte_path = os.path.join(save_path, rat, "uncertainty_vte.json")
            with open(uncertainty_vte_path, 'w') as f:
                json.dump(pair_results, f, indent=2)
            
            # Plot results for each day
            for day, model in all_models.items():
                day_plots_dir = os.path.join(results_dir, f"day_{day}")
                os.makedirs(day_plots_dir, exist_ok=True)
                print(f"Plotting results for Day {day}...")
                
                # Get stimulus labels for this day
                day_data = file_csv[file_csv['Day'] == day]
                all_indices = np.unique(np.concatenate([day_data['first'].values, day_data['second'].values]))
                index_to_letter = {idx: chr(65 + i) for i, idx in enumerate(sorted(all_indices))}
                stimulus_labels = [f"Stimulus {index_to_letter.get(i, i)}" for i in range(model.n_stimuli)]
                
                # Generate and save all plots for this day
                plot_and_save(
                    betasort.plot_positions_across_days, 
                    day_plots_dir,           # output path
                    f"{rat}_day{day}_positions_across_days",  # filename prefix
                    all_models,              # first argument (all_models)
                    mode='detailed',         # additional kwargs
                    stimulus_labels=stimulus_labels
                )

                plot_and_save(
                    betasort.plot_uncertainty_across_days, 
                    day_plots_dir,           # output path
                    f"{rat}_day{day}_uncertainty_across_days",  # filename prefix
                    all_models,              # first argument (all_models)
                    uncertainty_type='ROC',  # additional kwargs
                    mode='detailed',
                    stimulus_labels=stimulus_labels
                )
                
                plot_and_save(
                    betasort.plot_uncertainty_across_days, 
                    day_plots_dir,           # output path
                    f"{rat}_day{day}_uncertainty_across_days",  # filename prefix
                    all_models,              # first argument (all_models)
                    uncertainty_type='stimulus',  # additional kwargs
                    mode='detailed',
                    stimulus_labels=stimulus_labels
                )
                
                plot_and_save(
                    betasort.plot_beta_distributions,
                    day_plots_dir,
                    f"{rat}_day{day}_beta",
                    all_models[day],
                    stimulus_labels=stimulus_labels
                )
                
                plot_and_save(
                    betasort.plot_boundaries_history,
                    day_plots_dir,
                    f"{rat}_day{day}_boundaries",
                    all_models[day],
                    stimulus_labels=stimulus_labels
                )
            
            