import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from config.paths import paths
from analysis import betasort_analysis
from models import betasort_test
from models import betasort
from utilities import conversion_utils

def analyze_betasort_for_adjacent_pairs(all_data_df, rat, xi=0.95, tau=0.05, threshold=0.8, model_type='test', n_simulations=100):
    """
    Simplified analysis function that only runs the model to get adjacent pair results for the last day
    
    Parameters:
    - model_type: 'test' for betasort_test, 'regular' for betasort
    """
    # Initialize storage for all models
    all_models = {}
    global_U, global_L, global_R, global_N = {}, {}, {}, {}
    
    # Process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        print(f"Processing day {day} for {rat} (model: {model_type})...")
        
        # Extract relevant data for this day
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        rewards = day_data["correct"].values
        
        # Handle optional VTE column
        if 'VTE' in day_data.columns:
            vtes = day_data["VTE"].values
        else:
            vtes = np.zeros_like(chosen_idx)  # Default to no VTEs if not provided
        
        # Identify which stimuli are present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1  # +1 because of 0-indexing
        
        # Initialize a model for this day based on model_type
        if model_type == 'test':
            model = betasort_test.Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        else:  # regular betasort
            model = betasort.Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
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
        
        # Process trials for this day
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            vte = vtes[t]
            
            model_choices = np.zeros(n_simulations)
            model_correct = np.zeros(n_simulations)
            for sim in range(n_simulations):
                if model_type == "test":
                    model_choice = model.choose(chosen, unchosen, vte)
                else:
                    model_choice = model.choose(chosen, unchosen, vte)
                model_choices[sim] = model_choice
                model_correct[sim] = 1 if model_choice == min(chosen, unchosen) else 0
            
            # Calculate match rate (model choice matches rat's choice)
            model_match_rate = np.mean(model_choices == chosen)
            
            # Update model based on actual choice
            model.update(chosen, unchosen, reward, model_match_rate, threshold=0.8)
        
        # Store the model for this day
        all_models[day] = model
        
        # Update global states for the next day
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
    
    return all_models

def get_adjacent_pair_performance(all_data_df, model, model_type='test', n_simulations=100):
    """
    Get model performance on adjacent pairs for the last day
    
    Parameters:
    - all_data_df: DataFrame containing all training data
    - model: Trained model from the last day
    - model_type: 'test' for betasort_test, 'regular' for betasort
    """
    results = {}
    
    # Get the last day's data
    last_day = max(all_data_df['Day'].unique())
    last_day_data = all_data_df[all_data_df['Day'] == last_day]
    
    # Get all stimuli present on the last day
    chosen_idx = last_day_data["first"].values
    unchosen_idx = last_day_data["second"].values
    present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
    n_stimuli = max(present_stimuli) + 1
    
    # Find adjacent pairs (consecutive stimuli)
    adjacent_pairs = []
    for i in range(n_stimuli - 1):
        if i in present_stimuli and (i + 1) in present_stimuli:
            adjacent_pairs.append((i, i + 1))
    
    # Test model performance on each adjacent pair
    for pair in adjacent_pairs:
        stim1, stim2 = pair
        model_choices = np.zeros(n_simulations)
        
        for sim in range(n_simulations):
            if model_type == 'test':
                model_choice = model.choose(stim1, stim2, False)
            else:
                model_choice = model.choose(stim1, stim2, False)
            model_choices[sim] = model_choice
        
        # Calculate performance as choosing the lower-valued stimulus (stim1)
        model_performance = np.mean(model_choices == stim1)
        results[pair] = model_performance
    
    return results

def get_rat_adjacent_pair_performance(all_data_df):
    """
    Get rat performance on adjacent pairs for the last day
    """
    results = {}
    
    # Get the last day's data
    last_day = max(all_data_df['Day'].unique())
    last_day_data = all_data_df[all_data_df['Day'] == last_day]
    
    # Get all stimuli present on the last day
    chosen_idx = last_day_data["first"].values
    unchosen_idx = last_day_data["second"].values
    rewards = last_day_data["correct"].values
    present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
    n_stimuli = max(present_stimuli) + 1
    
    # Find adjacent pairs
    adjacent_pairs = []
    for i in range(n_stimuli - 1):
        if i in present_stimuli and (i + 1) in present_stimuli:
            adjacent_pairs.append((i, i + 1))
    
    # Calculate rat performance on each adjacent pair
    for pair in adjacent_pairs:
        stim1, stim2 = pair
        
        # Find trials where this pair was presented
        pair_trials = []
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            
            if (chosen == stim1 and unchosen == stim2) or (chosen == stim2 and unchosen == stim1):
                # Check if rat chose the lower-valued stimulus (stim1)
                correct_choice = 1 if chosen == stim1 else 0
                pair_trials.append(correct_choice)
        
        # Calculate performance as percentage of correct choices
        if len(pair_trials) > 0:
            results[pair] = np.mean(pair_trials)
        else:
            results[pair] = 0.0
    
    return results

def plot_adjacent_pair_differences(trial_type_labels, test_minus_rat, regular_minus_rat, save_path=None):
    """Create bar plot showing differences between models and rat performance (matching TI plot format)"""
    plt.figure(figsize=(15, 8))
    x_pos = np.arange(len(trial_type_labels))
    width = 0.35
    
    bars1 = plt.bar(x_pos - width/2, test_minus_rat, width, label='Uncertainty Model', color='steelblue')
    bars2 = plt.bar(x_pos + width/2, regular_minus_rat, width, label='Regular Model', color='forestgreen')
    
    # Add horizontal line at y=0 to show no difference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    plt.xlabel('Trial Types', fontsize=20)
    plt.ylabel('Performance Difference', fontsize=20)
    plt.title('Model Performance Differences before Testing Compared to Rats', fontsize=24)
    plt.xticks(x_pos, trial_type_labels, rotation=45 if len(trial_type_labels) > 8 else 0, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Set y-limits to accommodate all values with some padding
    all_values = list(test_minus_rat) + list(regular_minus_rat)
    y_min = min(all_values) - 0.05
    y_max = max(all_values) + 0.05
    plt.ylim(y_min, y_max)
    
    # Add value labels on bars
    for i, (test_rat_diff, regular_rat_diff) in enumerate(zip(test_minus_rat, regular_minus_rat)):
        # Adjust label position based on whether value is positive or negative
        test_rat_offset = 0.01 if test_rat_diff >= 0 else -0.02
        regular_rat_offset = 0.01 if regular_rat_diff >= 0 else -0.02
        
        va_test_rat = 'bottom' if test_rat_diff >= 0 else 'top'
        va_regular_rat = 'bottom' if regular_rat_diff >= 0 else 'top'
        
        plt.text(i - width/2, test_rat_diff + test_rat_offset, f'{test_rat_diff:.3f}', 
                ha='center', va=va_test_rat, fontsize=10)
        plt.text(i + width/2, regular_rat_diff + regular_rat_offset, f'{regular_rat_diff:.3f}', 
                ha='center', va=va_regular_rat, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Difference plot saved to {save_path}")
    
    plt.show()

def plot_adjacent_pair_comparison(trial_type_labels, model_test_averages, model_regular_averages, rat_averages, save_path=None):
    """Create bar plot comparing both models vs rat performance (matching TI plot format)"""
    plt.figure(figsize=(15, 8))
    x_pos = np.arange(len(trial_type_labels))
    width = 0.25
    
    bars1 = plt.bar(x_pos - width, model_test_averages, width, label='Model (Uncertainty)', color='skyblue')
    bars2 = plt.bar(x_pos, model_regular_averages, width, label='Model (Regular)', color='lightgreen')
    bars3 = plt.bar(x_pos + width, rat_averages, width, label='Rat', color='lightcoral')
    
    plt.xlabel('Trial Types', fontsize=20)
    plt.ylabel('Performance/Accuracy', fontsize=20)
    plt.title('Adjacent Pair Performance (Last Day): Both Models vs Rat Performance', fontsize=24)
    plt.xticks(x_pos, trial_type_labels, rotation=45 if len(trial_type_labels) > 8 else 0, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (test_val, regular_val, rat_val) in enumerate(zip(model_test_averages, model_regular_averages, rat_averages)):
        plt.text(i - width, test_val + 0.02, f'{test_val:.3f}', 
                ha='center', va='bottom', fontsize=10)
        plt.text(i, regular_val + 0.02, f'{regular_val:.3f}', 
                ha='center', va='bottom', fontsize=10)
        plt.text(i + width, rat_val + 0.02, f'{rat_val:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def number_to_letter(num):
    """Convert number to letter (0->A, 1->B, etc.)"""
    return chr(ord('A') + num)

def main():
    data_path = paths.preprocessed_data_model
    
    # Storage for all rats' adjacent pair results
    all_model_test_results = {}
    all_model_regular_results = {}
    all_rat_results = {}
    
    print("Starting Adjacent Pair Analysis (Last Day)...")
    print("=" * 50)
    
    # Process each rat
    for rat in os.listdir(data_path):
        if "BP09" in rat:  # Skip BP09 as in original code
            continue
        
        print(f"\nProcessing rat: {rat}")
        
        rat_path = os.path.join(data_path, rat)
        processed_rat = False
        
        for root, _, files in os.walk(rat_path):
            for file in files:
                if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                    continue

                file_path = os.path.join(root, file)
                try:
                    file_csv = pd.read_csv(file_path)
                    
                    # Run analysis for betasort_test model
                    print(f"  - Running betasort_test analysis...")
                    all_models_test = analyze_betasort_for_adjacent_pairs(file_csv, rat, model_type='test')
                    final_day_test = max(all_models_test.keys())
                    adjacent_result_test = get_adjacent_pair_performance(file_csv, all_models_test[final_day_test], model_type='test')
                    all_model_test_results[rat] = adjacent_result_test
                    
                    # Run analysis for regular betasort model
                    print(f"  - Running betasort (regular) analysis...")
                    all_models_regular = analyze_betasort_for_adjacent_pairs(file_csv, rat, model_type='regular')
                    final_day_regular = max(all_models_regular.keys())
                    adjacent_result_regular = get_adjacent_pair_performance(file_csv, all_models_regular[final_day_regular], model_type='regular')
                    all_model_regular_results[rat] = adjacent_result_regular
                    
                    # Get rat adjacent pair performance
                    rat_adjacent_results = get_rat_adjacent_pair_performance(file_csv)
                    all_rat_results[rat] = rat_adjacent_results
                    
                    print(f"  - Model (test) adjacent pair results: {len(adjacent_result_test)} pairs")
                    print(f"  - Model (regular) adjacent pair results: {len(adjacent_result_regular)} pairs")
                    print(f"  - Rat adjacent pair results: {len(rat_adjacent_results)} pairs")
                    processed_rat = True
                    break  # Only process one file per rat
                    
                except Exception as e:
                    print(f"  - Error processing {rat}: {e}")
                    continue
        
        if not processed_rat:
            print(f"  - Could not process {rat}")
    
    print("\n" + "=" * 50)
    print("Analysis Summary:")
    print(f"Rats with model (test) data: {len(all_model_test_results)}")
    print(f"Rats with model (regular) data: {len(all_model_regular_results)}")
    print(f"Rats with adjacent pair data: {len(all_rat_results)}")
    
    # Find common rats (those with all three types of data)
    common_rats = set(all_model_test_results.keys()) & set(all_model_regular_results.keys()) & set(all_rat_results.keys())
    print(f"Rats with all three data types: {len(common_rats)}")
    
    if len(common_rats) == 0:
        print("Error: No rats have all three data types. Cannot create comparison.")
        return
    
    # Aggregate results across rats
    all_pairs = set()
    for rat in common_rats:
        all_pairs.update(all_model_test_results[rat].keys())
        all_pairs.update(all_model_regular_results[rat].keys())
        all_pairs.update(all_rat_results[rat].keys())
    
    all_pairs = sorted(list(all_pairs))
    print(f"Total pairs found: {all_pairs}")
    
    # Calculate averages for pairs that have data from all three sources
    model_test_averages = []
    model_regular_averages = []
    rat_averages = []
    pair_data = []  # Store data with pair info for sorting
    
    for pair in all_pairs:
        # Get rats that have this pair in all three data sources
        rats_with_data = [rat for rat in common_rats 
                         if pair in all_model_test_results[rat] 
                         and pair in all_model_regular_results[rat]
                         and pair in all_rat_results[rat]]
        
        if len(rats_with_data) > 0:  # Only include if we have data from all sources
            test_values = [all_model_test_results[rat][pair] for rat in rats_with_data]
            regular_values = [all_model_regular_results[rat][pair] for rat in rats_with_data]
            rat_values = [all_rat_results[rat][pair] for rat in rats_with_data]
            
            # Convert numbers to letters for pair label
            pair_label = f"{number_to_letter(pair[0])}{number_to_letter(pair[1])}"
            
            pair_data.append({
                'label': pair_label,
                'test_avg': np.mean(test_values),
                'regular_avg': np.mean(regular_values),
                'rat_avg': np.mean(rat_values),
                'original_pair': pair
            })
            
            print(f"Pair {pair_label}: {len(rats_with_data)} rats, "
                  f"Test avg: {np.mean(test_values):.3f}, "
                  f"Regular avg: {np.mean(regular_values):.3f}, "
                  f"Rat avg: {np.mean(rat_values):.3f}")
    
    # Sort pairs alphabetically
    pair_data.sort(key=lambda x: x['label'])
    
    # Extract the ordered data
    trial_type_labels = [data['label'] for data in pair_data]
    model_test_averages = [data['test_avg'] for data in pair_data]
    model_regular_averages = [data['regular_avg'] for data in pair_data]
    rat_averages = [data['rat_avg'] for data in pair_data]
    
    if len(trial_type_labels) == 0:
        print("Error: No pairs have data from all sources.")
        return
    
    # Create output directory for plots
    output_dir = os.path.join(paths.betasort_data, "adjacent_pair_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison plot
    save_path = os.path.join(output_dir, "adjacent_pair_both_models_vs_rat_comparison.png")
    plot_adjacent_pair_comparison(trial_type_labels, model_test_averages, model_regular_averages, rat_averages, save_path)
    
    # create differences plot
    test_minus_rat_diff = np.array(model_test_averages) - np.array(rat_averages)
    regular_minus_rat_diff = np.array(model_regular_averages) - np.array(rat_averages)
    
    diff_save_path = os.path.join(output_dir, "model_differences.png")
    plot_adjacent_pair_differences(trial_type_labels, test_minus_rat_diff, regular_minus_rat_diff, diff_save_path)
    
    
    # Save detailed results
    results_df = pd.DataFrame({
        'pair_type': trial_type_labels,
        'model_test_performance': model_test_averages,
        'model_regular_performance': model_regular_averages,
        'rat_performance': rat_averages,
        'test_minus_rat': np.array(model_test_averages) - np.array(rat_averages),
        'regular_minus_rat': np.array(model_regular_averages) - np.array(rat_averages),
        'test_minus_regular': np.array(model_test_averages) - np.array(model_regular_averages)
    })
    
    results_path = os.path.join(output_dir, "adjacent_pair_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print(f"Mean model (test) performance: {np.mean(model_test_averages):.3f}")
    print(f"Mean model (regular) performance: {np.mean(model_regular_averages):.3f}")
    print(f"Mean rat performance: {np.mean(rat_averages):.3f}")
    print(f"Mean difference (test - rat): {np.mean(np.array(model_test_averages) - np.array(rat_averages)):.3f}")
    print(f"Mean difference (regular - rat): {np.mean(np.array(model_regular_averages) - np.array(rat_averages)):.3f}")
    print(f"Mean difference (test - regular): {np.mean(np.array(model_test_averages) - np.array(model_regular_averages)):.3f}")
    print(f"Correlation (test vs rat): {np.corrcoef(model_test_averages, rat_averages)[0,1]:.3f}")
    print(f"Correlation (regular vs rat): {np.corrcoef(model_regular_averages, rat_averages)[0,1]:.3f}")
    print(f"Correlation (test vs regular): {np.corrcoef(model_test_averages, model_regular_averages)[0,1]:.3f}")

if __name__ == "__main__":
    main()