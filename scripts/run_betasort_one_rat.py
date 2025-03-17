import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import helper
from src import betasort

data_path = os.path.join(helper.BASE_PATH, "processed_data", "data_for_model")

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
            best_xi, best_tau, best_performance, param_performances = betasort.find_optimal_parameters_for_rat(file_csv, rat)
            print(f"Best parameters: xi={best_xi:.3f}, tau={best_tau:.3f}, avg_performance={best_performance:.3f}")

            unique_days = sorted(file_csv['Day'].unique())
            
            # Plot overall heatmap
            avg_param_perf = {params: np.mean(day_rates) for params, day_rates in param_performances.items()}
            print("Plotting overall parameter performance heatmap...")
            betasort.parameter_performance_heatmap(avg_param_perf)

            # Plot individual day heatmaps
            print("Plotting day-by-day parameter performance heatmaps...")
            for day_idx, day in enumerate(unique_days):
                # Create a heatmap for this specific day
                day_param_perf = {params: day_rates[day_idx] for params, day_rates in param_performances.items()}
                plt.figure(figsize=(10, 8))
                betasort.parameter_performance_heatmap(day_param_perf)
                plt.title(f"Parameter Performance for Day {day}")
                plt.tight_layout()
                plt.show()

            # Run analysis with best parameters
            final_model, all_day_models = betasort.analyze_one_rat(
                file_csv, rat, tau=best_tau, xi=best_xi
            )

            # Plot results for each day
            for day, model in all_day_models.items():
                print(f"Plotting results for Day {day}...")
                
                # Get stimulus labels for this day
                day_data = file_csv[file_csv['Day'] == day]
                all_indices = np.unique(np.concatenate([day_data['first'].values, day_data['second'].values]))
                index_to_letter = {idx: chr(65 + i) for i, idx in enumerate(sorted(all_indices))}
                stimulus_labels = [f"Stimulus {index_to_letter.get(i, i)}" for i in range(model.n_stimuli)]
                
                # Plot model results
                plt.figure(figsize=(12, 8))
                betasort.plot_positions(model, stimulus_labels)
                plt.title(f"Stimulus Positions for Day {day}")
                plt.tight_layout()
                plt.show()
                
                plt.figure(figsize=(12, 8))
                betasort.plot_uncertainty(model, stimulus_labels)
                plt.title(f"Uncertainty for Day {day}")
                plt.tight_layout()
                plt.show()
                
                plt.figure(figsize=(12, 8))
                betasort.plot_beta_distributions(model, stimulus_labels=stimulus_labels)
                plt.title(f"Beta Distributions for Day {day}")
                plt.tight_layout()
                plt.show()