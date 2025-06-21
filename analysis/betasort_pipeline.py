import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from config.paths import paths
from analysis import betasort_analysis
from models import betasort
from visualization import betasort_plots

# pylint: disable=broad-exception-caught

def analyze_betasort_comprehensive(all_data_df, rat, n_simulations=100, use_diff_evolution=True,
                                  xi=None, tau=None, threshold=None):
    """
    Comprehensive analysis function for Betasort model that:
    1. Finds optimal parameters (optionally using differential evolution)
    2. Runs a model with optimal parameters
    3. Performs binomial tests, t-tests, and VTE uncertainty analysis
    
    Parameters:
    - all_data_df: DataFrame containing rat choice data
    - rat: String identifier for the rat
    - n_simulations: Number of simulations for choice testing
    - use_diff_evolution: Whether to find optimal parameters using differential evolution
    - xi, tau, threshold: Optional parameter values to use if not using differential evolution
    
    Returns:
    - Dictionary containing:
        - all_models: Dictionary of models for each day
        - pair_vte_df: DataFrame with paired VTE and uncertainty values
        - best_xi: Optimal xi parameter
        - best_tau: Optimal tau parameter
        - best_threshold: Optimal threshold parameter
        - best_performance: Performance with optimal parameters
        - session_results_ttest: T-test results by session
        - overall_results_ttest: Overall t-test results
        - session_results_binomial: Binomial test results by session
    """
    results = {}
    
    # Step 1: Find optimal parameters if requested
    if use_diff_evolution:
        print("Finding optimal parameters using differential evolution...")
        best_xi, best_tau, best_threshold, best_performance = betasort_analysis.diff_evolution(all_data_df, rat)
    else:
        # Use provided parameters
        best_xi = xi if xi is not None else 0.95  # Default values
        best_tau = tau if tau is not None else 0.05
        best_threshold = threshold if threshold is not None else 0.8
    
    # Store parameter values
    results['best_xi'] = best_xi
    results['best_tau'] = best_tau
    results['best_threshold'] = best_threshold
    
    print(f"Using parameters: xi={best_xi}, tau={best_tau}, threshold={best_threshold}")
    
    # Step 2: Initialize storage for all analyses
    all_models = {}
    global_U, global_L, global_R, global_N = {}, {}, {}, {}
    
    # VTE data collection
    pair_vte_data = []
    
    # Binomial analysis storage
    session_results_binomial = {}
    
    # T-test analysis storage
    session_results_regression = []
    
    # Match rates for performance calculation
    all_match_rates = []
    
    # Process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        print(f"Processing day {day}...")
        
        # Extract relevant data for this day
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        rewards = day_data["correct"].values
        
        # Handle optional VTE column
        if 'VTE' in day_data.columns:
            vtes = day_data["VTE"].values
        else:
            vtes = np.zeros_like(chosen_idx)  # Default to no VTEs if not provided
            
        # Handle optional ID column
        if 'ID' in day_data.columns:
            traj_nums = day_data["ID"].values
        else:
            traj_nums = np.arange(len(chosen_idx))  # Default to sequential IDs
        
        # Identify which stimuli are present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1  # +1 because of 0-indexing
        
        # Initialize a model for this day
        model = betasort.Betasort(n_stimuli, rat, day, tau=best_tau, xi=best_xi)
        
        # Transfer state from previous days
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
        
        # Reset histories
        model.uncertainty_history = [model.get_all_stimulus_uncertainties()]
        model.ROC_uncertainty_history = [model.get_all_ROC_uncertainties()]
        model.position_history = [model.get_all_positions()]
        model.U_history = [model.U.copy()]
        model.L_history = [model.L.copy()]
        
        # Data for analyses
        day_matches = []
        model_correct_rates = []
        rat_correct_rates = []
        
        # Process trials for this day
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            vte = vtes[t]
            traj_num = traj_nums[t]
            
            # VTE Analysis: Get uncertainties before updates
            stim1_uncertainty = model.get_uncertainty_stimulus(min(chosen, unchosen))
            stim2_uncertainty = model.get_uncertainty_stimulus(max(chosen, unchosen))
            pair_roc_uncertainty = model.get_uncertainty_relation_ROC(min(chosen, unchosen), max(chosen, unchosen))
            
            # Store VTE data
            vte_occurred = 1 if vte else 0
            pair_vte_data.append({
                'day': day,
                'trial_num': traj_num,
                'stim1': min(chosen, unchosen),
                'stim2': max(chosen, unchosen),
                'chosen': chosen,
                'unchosen': unchosen,
                'vte_occurred': vte_occurred,
                'stim1_uncertainty': stim1_uncertainty,
                'stim2_uncertainty': stim2_uncertainty,
                'pair_roc_uncertainty': pair_roc_uncertainty,
                'reward': reward
            })
            
            # Simulate model choices (for all analyses)
            model_choices = np.zeros(n_simulations)
            model_correct = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen, unchosen])
                model_choices[sim] = model_choice
                # Correct = choosing the lower-valued stimulus (as per original code)
                model_correct[sim] = 1 if model_choice == min(chosen, unchosen) else 0
            
            # Calculate match rate (model choice matches rat's choice)
            model_match_rate = np.mean(model_choices == chosen)
            day_matches.append(model_match_rate)
            
            # Store correct rates for binomial analysis
            model_correct_rates.append(np.mean(model_correct))
            rat_correct_rates.append(1 if chosen < unchosen else 0)
            
            # Update model based on actual choice
            model.update(chosen, unchosen, reward, model_match_rate, threshold=best_threshold)
        
        # Store the model for this day
        all_models[day] = model
        
        # Update global states for the next day
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
        
        # Calculate day-specific match rate for performance
        day_match_rate = np.mean(day_matches)
        all_match_rates.append(day_match_rate)
        
        # Binomial analysis for this day
        n_rat_correct = int(np.sum(rat_correct_rates))
        n_trials = len(rat_correct_rates)
        model_correct_rate = sum(model_correct_rates) / len(model_correct_rates)
        p_value_binomial = stats.binomtest(n_rat_correct, n_trials, p=model_correct_rate)
        
        session_results_binomial[day] = {
            'matches': n_rat_correct,
            'trials': n_trials,
            'match_rate': n_rat_correct/n_trials if n_trials > 0 else 0,
            'p_value': p_value_binomial,
            'model_rate': model_correct_rate,
            'significant': p_value_binomial.pvalue < 0.05
        }
        
        # calculate logistic regression
        X = np.array(model_correct_rates).reshape(-1, 1) # model probabilities as features
        Y = np.array(rat_correct_rates)
        
        # Check if there's more than one class before fitting
        unique_classes = np.unique(Y)
        if len(unique_classes) > 1:
            regression_model = LogisticRegression()
            regression_model.fit(X, Y)
            
            # get model accuracy for logistic regression
            predictions = regression_model.predict(X)
            accuracy = accuracy_score(Y, predictions)
        else:
            # Handle single-class case
            # If all predictions are the same, the accuracy is either 0% or 100%
            accuracy = float(unique_classes[0])  # If all Y are 1, accuracy is 1.0; if all Y are 0, accuracy is 0.0
            
        session_results_regression.append(accuracy)
    
    # Calculate overall performance
    best_performance = np.mean(all_match_rates)
    results['best_performance'] = best_performance
    
    # Convert VTE data to DataFrame
    pair_vte_df = pd.DataFrame(pair_vte_data)
    if len(pair_vte_df) > 0:  # Add pair column for VTE analysis
        pair_vte_df['pair'] = pair_vte_df.apply(lambda row: f"{row['stim1']}-{row['stim2']}", axis=1)
    
    # Store all results
    results['all_models'] = all_models
    results['pair_vte_df'] = pair_vte_df
    results['session_results_binomial'] = session_results_binomial
    results['session_predictions_regression'] = session_results_regression
    
    return results

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

data_path = paths.preprocessed_data_model
save_path = paths.betasort_data

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
            
            try:
                # analyze the data sequentially
                all_results = analyze_betasort_comprehensive(file_csv, rat)
            except Exception as e:
                print(rat, file_path)
                print(e)
                continue
            
            # check with transitive inference
            all_models = all_results["all_models"]
            final_day = max(all_models.keys())
            ti_result = betasort_analysis.check_transitive_inference(all_models[final_day])
            ti_result_serializable = {f"{k[0]},{k[1]}": v for k, v in ti_result.items()}
            ti_result_json = json.dumps(ti_result_serializable)
            
            # check uncertainty vs vte correspondence
            pair_results = betasort_analysis.analyze_correlations(all_results["pair_vte_df"])
            
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
                    betasort_plots.plot_positions_across_days, 
                    day_plots_dir,           # output path
                    f"{rat}_day{day}_positions_across_days",  # filename prefix
                    all_models,              # first argument (all_models)
                    mode='detailed',         # additional kwargs
                    stimulus_labels=stimulus_labels
                )

                plot_and_save(
                    betasort_plots.plot_uncertainty_across_days, 
                    day_plots_dir,           # output path
                    f"{rat}_day{day}_uncertainty_across_days",  # filename prefix
                    all_models,              # first argument (all_models)
                    uncertainty_type='ROC',  # additional kwargs
                    mode='detailed',
                    stimulus_labels=stimulus_labels
                )
                
                plot_and_save(
                    betasort_plots.plot_uncertainty_across_days, 
                    day_plots_dir,           # output path
                    f"{rat}_day{day}_uncertainty_across_days",  # filename prefix
                    all_models,              # first argument (all_models)
                    uncertainty_type='stimulus',  # additional kwargs
                    mode='detailed',
                    stimulus_labels=stimulus_labels
                )
                
                plot_and_save(
                    betasort_plots.plot_beta_distributions,
                    day_plots_dir,
                    f"{rat}_day{day}_beta",
                    all_models[day],
                    stimulus_labels=stimulus_labels
                )
                
                plot_and_save(
                    betasort_plots.plot_boundaries_history,
                    day_plots_dir,
                    f"{rat}_day{day}_boundaries",
                    all_models[day],
                    stimulus_labels=stimulus_labels
                )
            
            