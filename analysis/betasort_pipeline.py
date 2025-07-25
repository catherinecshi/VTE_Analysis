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
from models import betasort_test
from visualization import betasort_plots

# pylint: disable=broad-exception-caught, consinder-using-enumerate

def analyze_betasort_comprehensive(all_data_df, rat, n_simulations=100, use_diff_evolution=True,
                                  xi=None, tau=None, threshold=None, test=False):
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
    
    pair_vte_data = [] # VTE data collection
    session_results_binomial = {} # Binomial analysis storage
    session_results_regression = [] # T-test analysis storage
    all_match_rates = [] # Match rates for performance calculation
    
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
            print(f"WHY ARE THERE NO VTES FOR {day} FOR {rat}")
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
        if test:
            model = betasort_test.Betasort(n_stimuli, rat, day, tau=best_tau, xi=best_xi)
        else:
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
        
        # pre update model predictions for trial types
        adjacent_pairs = []
        for i in range(n_stimuli - 1):
            if i in present_stimuli and (i + 1) in present_stimuli:
                adjacent_pairs.append ((i, i + 1))
        
        pre_update_model_data = {}
        for pair in adjacent_pairs:
            stim1, stim2 = pair
            model_choices = []
            for sim in range(n_simulations):
                if test:
                    model_choice = model.choose(stim1, stim2, False)
                else:
                    model_choice = model.choose([stim1, stim2])
                model_choices.append(model_choice)
            
            # correct rates for the model
            correct_rate = np.mean([1 if choice == stim1 else 0 for choice in model_choices])
            pre_update_model_data[pair] = {
                'model_correct_rate': correct_rate,
                'pair_name' : f'{stim1}-{stim2}'
            }
            
        # store real rat performance
        actual_rat_data = {pair: {'rewards': [], 'choice': []} for pair in adjacent_pairs}
        
        # Process trials for this day
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            vte = vtes[t]
            traj_num = traj_nums[t]
            
            # store rat performance
            current_pair = (min(chosen, unchosen), max(chosen, unchosen))
            if current_pair in adjacent_pairs:
                actual_rat_data[current_pair]['rewards'].append(reward)
            
            # VTE Analysis: Get uncertainties before updates
            stim1_uncertainty = model.get_uncertainty_stimulus(min(chosen, unchosen))
            stim2_uncertainty = model.get_uncertainty_stimulus(max(chosen, unchosen))
            pair_roc_uncertainty = model.get_uncertainty_ROC(min(chosen, unchosen), max(chosen, unchosen))
            
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
                if test:
                    model_choice = model.choose(chosen, unchosen, vte)
                else:
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
        
        # post update storage of model predictions
        post_update_model_data = {}
        for pair in adjacent_pairs:
            stim1, stim2 = pair
            model_choices = []
            for sim in range(n_simulations):
                if test:
                    model_choice = model.choose(stim1, stim2, False)
                else:
                    model_choice = model.choose([stim1, stim2])
                model_choices.append(model_choice)
            
            # correct rate for model
            correct_rate = np.mean([1 if choice == stim1 else 0 for choice in model_choices])
            post_update_model_data[pair] = {
                'model_correct_rate': correct_rate,
                'pair_name': f"{stim1}-{stim2}"
            }
        
        # centralize session analysis data
        actual_performance = {}
        for pair in adjacent_pairs:
            if len(actual_rat_data[pair]['rewards']) > 0:
                # Rat correct rate is percentage of rewards = 1
                rat_correct_rate = np.mean(actual_rat_data[pair]['rewards'])
                actual_performance[pair] = {
                    'rat_correct_rate': rat_correct_rate,
                    'n_trials': len(actual_rat_data[pair]['rewards']),
                    'pair_name': f"{pair[0]}-{pair[1]}"
                }
            else:
                actual_performance[pair] = {
                    'rat_correct_rate': 0,
                    'n_trials': 0,
                    'pair_name': f"{pair[0]}-{pair[1]}"
                }

        adjacent_pair_data = {
            'day': day,
            'adjacent_pairs': adjacent_pairs,
            'pre_update_model': pre_update_model_data,
            'post_update_model': post_update_model_data,
            'actual_rat_performance': actual_performance
        }
        
        if 'adjacent_pair_analysis' not in results:
            results['adjacent_pair_analysis'] = {}
        results['adjacent_pair_analysis'][day] = adjacent_pair_data
        
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


# === AGGREGATE ADJACENT PAIR ANALYSIS ACROSS ALL RATS ===
print("Aggregating adjacent pair analysis across all rats...")

# Storage for all rats' data
all_rats_adjacent_data = {}

# Read back the saved data for each rat to get adjacent pair analysis
for rat in os.listdir(data_path):
    rat_results_path = os.path.join(save_path, rat, "results.csv")
    if not os.path.exists(rat_results_path):
        continue
    
    if "BP09" in rat:
        print(rat)
        continue
        
    # Load the rat's data and re-run analysis to get adjacent pair data
    rat_path = os.path.join(data_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                continue

            file_path = os.path.join(root, file)
            file_csv = pd.read_csv(file_path)
            
            try:
                # Re-run analysis to get adjacent pair data
                all_results = analyze_betasort_comprehensive(file_csv, rat, use_diff_evolution=False, test=True)
                
                if 'adjacent_pair_analysis' in all_results:
                    # Find the last day for this rat
                    last_day = max(all_results['adjacent_pair_analysis'].keys())
                    last_day_data = all_results['adjacent_pair_analysis'][last_day]
                    
                    all_rats_adjacent_data[rat] = {
                        'last_day': last_day,
                        'data': last_day_data
                    }
                    print(f"Collected adjacent pair data for {rat}, last day: {last_day}")
                    
            except Exception as e:
                print(f"Error processing {rat}: {e}")
                continue

# === AGGREGATE PERFORMANCE ACROSS RATS ===
# Find all unique pairs across all rats
all_pairs = set()
for rat_data in all_rats_adjacent_data.values():
    all_pairs.update(rat_data['data']['adjacent_pairs'])

# Convert to sorted list for consistent ordering
all_pairs = sorted(list(all_pairs))
pair_names = [f"{p[0]}-{p[1]}" for p in all_pairs]

print(f"Found pairs across all rats: {pair_names}")

# Initialize storage for averaged data
aggregated_data = {
    'pair_names': pair_names,
    'rat_rates': [],
    'pre_model_rates': [],
    'post_model_rates': [],
    'rat_counts': [],  # Number of rats that had data for each pair
}

# Calculate averages for each pair
for pair in all_pairs:
    rat_rates_for_pair = []
    pre_model_rates_for_pair = []
    post_model_rates_for_pair = []
    
    for rat, rat_info in all_rats_adjacent_data.items():
        rat_data = rat_info['data']
        
        if pair in rat_data['adjacent_pairs']:
            # Get rat performance
            if pair in rat_data['actual_rat_performance']:
                rat_rates_for_pair.append(rat_data['actual_rat_performance'][pair]['rat_correct_rate'])
            
            # Get pre-update model performance
            if pair in rat_data['pre_update_model']:
                pre_model_rates_for_pair.append(rat_data['pre_update_model'][pair]['model_correct_rate'])
            
            # Get post-update model performance
            if pair in rat_data['post_update_model']:
                post_model_rates_for_pair.append(rat_data['post_update_model'][pair]['model_correct_rate'])
    
    # Calculate averages
    aggregated_data['rat_rates'].append(np.mean(rat_rates_for_pair) if rat_rates_for_pair else 0)
    aggregated_data['pre_model_rates'].append(np.mean(pre_model_rates_for_pair) if pre_model_rates_for_pair else 0)
    aggregated_data['post_model_rates'].append(np.mean(post_model_rates_for_pair) if post_model_rates_for_pair else 0)
    aggregated_data['rat_counts'].append(len(rat_rates_for_pair))
    
    print(f"Pair {pair[0]}-{pair[1]}: {len(rat_rates_for_pair)} rats, "
          f"Rat avg: {np.mean(rat_rates_for_pair):.3f}, "
          f"Pre-model avg: {np.mean(pre_model_rates_for_pair):.3f}, "
          f"Post-model avg: {np.mean(post_model_rates_for_pair):.3f}")

# === SAVE AGGREGATED DATA ===
aggregated_results_path = os.path.join(save_path, "aggregated_adjacent_pair_analysis.json")
with open(aggregated_results_path, 'w') as f:
    json.dump({
        'pair_names': aggregated_data['pair_names'],
        'rat_rates': aggregated_data['rat_rates'],
        'pre_model_rates': aggregated_data['pre_model_rates'],
        'post_model_rates': aggregated_data['post_model_rates'],
        'rat_counts': aggregated_data['rat_counts'],
        'rats_included': list(all_rats_adjacent_data.keys()),
        'total_rats': len(all_rats_adjacent_data)
    }, f, indent=2)

print(f"Saved aggregated data to {aggregated_results_path}")

# === GENERATE AGGREGATED PLOT ===
aggregated_plots_dir = os.path.join(save_path, "aggregated_plots")
os.makedirs(aggregated_plots_dir, exist_ok=True)

# Create the aggregated comparison plot
plot_and_save(
    betasort_plots.plot_aggregated_adjacent_pair_comparison,
    aggregated_plots_dir,
    "all_rats_adjacent_pair_comparison",
    aggregated_data['pair_names'],
    aggregated_data['rat_rates'],
    aggregated_data['pre_model_rates'],
    aggregated_data['post_model_rates'],
    aggregated_data['rat_counts'],
    total_rats=len(all_rats_adjacent_data)
)

print("Generated aggregated adjacent pair comparison plot")

# === GENERATE POST-MODEL VS RAT ONLY PLOT ===
# Create the simplified comparison plot (post-model vs rat only)
plot_and_save(
    betasort_plots.plot_post_model_vs_rat_comparison,
    aggregated_plots_dir,
    "all_rats_post_model_vs_rat_comparison",
    aggregated_data['pair_names'],
    aggregated_data['rat_rates'],
    aggregated_data['post_model_rates'],
    aggregated_data['rat_counts'],
    total_rats=len(all_rats_adjacent_data)
)

print("Generated post-model vs rat comparison plot")

# --- AGGREGATE TRANSITIVE INFERENCE REAL RESULTS ACROSS RATS ---
all_rats_ti_real = {}

for rat in os.listdir(data_path):
    if "BP09" in rat:
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
                all_results = analyze_betasort_comprehensive(file_csv, rat, use_diff_evolution=False, test=True)
            except Exception as e:
                print(rat, file_path)
                print(e)
                continue
            
            # check with transitive inference
            all_models = all_results["all_models"]
            final_day = max(all_models.keys())
            ti_result = betasort_analysis.check_transitive_inference(all_models[final_day], test=True)
            ti_result_serializable = {f"{k[0]},{k[1]}": v for k, v in ti_result.items()}
            ti_result_json = json.dumps(ti_result_serializable)
            
            # check with real transitive inference
            real_ti_data_path = os.path.join(data_path, "inferenceTesting", f"{rat}.csv")
            real_ti_data = pd.read_csv(real_ti_data_path)
            ti_result_real = betasort_analysis.check_transitive_inference_real(all_models[final_day], real_ti_data, test=True)
            
            
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
            
            print(results["best_performance"], results["rat"])
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
            
            # Store ti_result_real for this rat
            all_rats_ti_real[rat] = ti_result_real
            
            """
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

                day_pair_data = all_results['adjacent_pair_analysis'][day]
                
                # Prepare data for plotting
                pairs = day_pair_data['adjacent_pairs']
                pair_names = [f"{p[0]}-{p[1]}" for p in pairs]
                
                rat_rates = [day_pair_data['actual_rat_performance'][p]['rat_correct_rate'] for p in pairs]
                pre_model_rates = [day_pair_data['pre_update_model'][p]['model_correct_rate'] for p in pairs]
                post_model_rates = [day_pair_data['post_update_model'][p]['model_correct_rate'] for p in pairs]
                
                # Create comparison plot
                plot_and_save(
                    betasort_plots.plot_adjacent_pair_comparison,
                    day_plots_dir,
                    f"{rat}_day{day}_adjacent_pair_comparison",
                    pair_names,
                    rat_rates,
                    pre_model_rates,
                    post_model_rates,
                    day=day
                )
                """
            
# Aggregate by trial type
from collections import defaultdict
import matplotlib.pyplot as plt

type_to_model = defaultdict(list)
type_to_rat = defaultdict(list)
type_to_n = defaultdict(list)

for rat, ti_dict in all_rats_ti_real.items():
    for pair, (model_pct, rat_pct, n) in ti_dict.items():
        type_to_model[pair].append(model_pct)
        type_to_rat[pair].append(rat_pct)
        type_to_n[pair].append(n)

# Prepare data for plotting
trial_types = sorted(type_to_model.keys())
model_means = [np.mean(type_to_model[pair]) for pair in trial_types]
rat_means = [np.mean(type_to_rat[pair]) for pair in trial_types]
model_sems = [np.std(type_to_model[pair], ddof=1)/np.sqrt(len(type_to_model[pair])) for pair in trial_types]
rat_sems = [np.std(type_to_rat[pair], ddof=1)/np.sqrt(len(type_to_rat[pair])) for pair in trial_types]
labels = [f"{pair[0]}-{pair[1]}" for pair in trial_types]

# Save aggregated data as CSV
agg_df = pd.DataFrame({
    'trial_type': labels,
    'model_mean': model_means,
    'model_sem': model_sems,
    'rat_mean': rat_means,
    'rat_sem': rat_sems,
    'n_rats': [len(type_to_model[pair]) for pair in trial_types]
})
agg_csv_path = os.path.join(save_path, 'aggregated_ti_real_results.csv')
agg_df.to_csv(agg_csv_path, index=False)
print(f"Saved aggregated TI real results to {agg_csv_path}")

# Plot
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - width/2, rat_means, width, yerr=rat_sems, label='Rat', color='blue', alpha=0.7, capsize=5)
rects2 = ax.bar(x + width/2, model_means, width, yerr=model_sems, label='Model', color='green', alpha=0.7, capsize=5)

ax.set_xlabel('Trial Type', fontsize=14)
ax.set_ylabel('Percent Correct', fontsize=14)
ax.set_title('Transitive Inference: Model vs Rat (Averaged Across Rats)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.05)

# Add value labels
for rects, means in zip([rects1, rects2], [rat_means, model_means]):
    for rect, mean in zip(rects, means):
        height = rect.get_height()
        ax.annotate(f'{mean:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plot_path = os.path.join(save_path, 'aggregated_ti_real_plot.png')
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Saved aggregated TI real plot to {plot_path}")
            