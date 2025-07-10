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

def analyze_betasort_for_ti(all_data_df, rat, xi=0.95, tau=0.05, threshold=0.8, model_type='test', n_simulations=100):
    """
    Simplified analysis function that only runs the model to get transitive inference results
    
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

def check_transitive_inference(model, model_type='test', n_simulations=100):
    """
    Updated method that adapts to the number of stimuli in the model
    
    Parameters:
    - model_type: 'test' for betasort_test, 'regular' for betasort
    """
    results = {}
    
    # Get the number of stimuli from the model (adaptive to model size)
    n_stimuli = model.n_stimuli
    
    for chosen_idx in range(0, n_stimuli): 
        for other_idx in range(chosen_idx + 1, n_stimuli):
            if chosen_idx == other_idx:
                continue
            
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                if model_type == 'test':
                    model_choice = model.choose_ti(chosen_idx, other_idx, vte=False)
                else:  # regular betasort
                    model_choice = model.choose(chosen_idx, other_idx, vte=False)
                model_choices[sim] = model_choice
            
            # see how well the model matches up with real choices
            model_match_rate = np.mean(model_choices == chosen_idx)
            
            results[(chosen_idx, other_idx)] = model_match_rate
            
    return results

def load_rat_ti_data(rat):
    """Load transitive inference data for a specific rat"""
    ti_path = os.path.join(paths.preprocessed_data_model, rat, "ti_results.csv")
    if os.path.exists(ti_path):
        return pd.read_csv(ti_path)
    else:
        print(f"TI data not found for {rat}: {ti_path}")
        return None

def plot_ti_comparison(trial_type_labels, model_test_averages, model_regular_averages, rat_averages, save_path=None):
    """Create bar plot comparing both models vs rat performance"""
    plt.figure(figsize=(15, 8))
    x_pos = np.arange(len(trial_type_labels))
    width = 0.25
    
    bars1 = plt.bar(x_pos - width, model_test_averages, width, label='Model (Uncertainty)', color='skyblue')
    bars2 = plt.bar(x_pos, model_regular_averages, width, label='Model (Regular)', color='lightgreen')
    bars3 = plt.bar(x_pos + width, rat_averages, width, label='Rat', color='lightcoral')
    
    plt.xlabel('Trial Types', fontsize=20)  # Increased from 15
    plt.ylabel('Performance/Accuracy', fontsize=20)  # Increased from 15
    plt.title('Transitive Inference: Both Models vs Rat Performance', fontsize=24)  # Increased from 20
    plt.xticks(x_pos, trial_type_labels, rotation=45 if len(trial_type_labels) > 8 else 0, fontsize=16)  # Added fontsize
    plt.yticks(fontsize=16)  # Added fontsize for y-axis
    plt.legend(loc='lower right', fontsize=16)  # Increased legend fontsize
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Add value labels on bars with bigger font
    for i, (test_val, regular_val, rat_val) in enumerate(zip(model_test_averages, model_regular_averages, rat_averages)):
        plt.text(i - width, test_val + 0.02, f'{test_val:.3f}', 
                ha='center', va='bottom', fontsize=10)  # Increased from 7
        plt.text(i, regular_val + 0.02, f'{regular_val:.3f}', 
                ha='center', va='bottom', fontsize=10)  # Increased from 7
        plt.text(i + width, rat_val + 0.02, f'{rat_val:.3f}', 
                ha='center', va='bottom', fontsize=10)  # Increased from 7
    
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
    
    # Storage for all rats' TI results
    all_model_test_results = {}
    all_model_regular_results = {}
    all_rat_results = {}
    
    print("Starting Transitive Inference Analysis...")
    print("=" * 50)
    
    # Process each rat
    for rat in os.listdir(data_path):
        if "TH510" not in rat:  # Skip BP09 as in original code
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
                    all_models_test = analyze_betasort_for_ti(file_csv, rat, model_type='test')
                    final_day_test = max(all_models_test.keys())
                    ti_result_test = check_transitive_inference(all_models_test[final_day_test], model_type='test')
                    all_model_test_results[rat] = ti_result_test
                    
                    # Run analysis for regular betasort model
                    print(f"  - Running betasort (regular) analysis...")
                    all_models_regular = analyze_betasort_for_ti(file_csv, rat, model_type='regular')
                    final_day_regular = max(all_models_regular.keys())
                    ti_result_regular = check_transitive_inference(all_models_regular[final_day_regular], model_type='regular')
                    all_model_regular_results[rat] = ti_result_regular
                    
                    # Load rat TI data
                    rat_ti_data = load_rat_ti_data(rat)
                    if rat_ti_data is not None:
                        # Convert to same format as model results
                        rat_results = {}
                        for _, row in rat_ti_data.iterrows():
                            stim1, stim2, accuracy = int(row['stim1']), int(row['stim2']), row['accuracy']
                            rat_results[(stim1, stim2)] = accuracy
                        all_rat_results[rat] = rat_results
                        print(f"  - Loaded TI data: {len(rat_results)} trial types")
                    else:
                        print(f"  - No TI data found for {rat}")
                    
                    print(f"  - Model (test) TI results: {len(ti_result_test)} trial types")
                    print(f"  - Model (regular) TI results: {len(ti_result_regular)} trial types")
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
    print(f"Rats with TI data: {len(all_rat_results)}")
    
    # Find common rats (those with all three types of data)
    common_rats = set(all_model_test_results.keys()) & set(all_model_regular_results.keys()) & set(all_rat_results.keys())
    common_rats = set(all_model_test_results.keys()) & set(all_rat_results.keys())
    print(f"Rats with all three data types: {len(common_rats)}")
    
    if len(common_rats) == 0:
        print("Error: No rats have all three data types. Cannot create comparison.")
        return
    
    # Aggregate results across rats
    all_trial_types = set()
    for rat in common_rats:
        all_trial_types.update(all_model_test_results[rat].keys())
        #all_trial_types.update(all_model_regular_results[rat].keys())
        all_trial_types.update(all_rat_results[rat].keys())
    
    all_trial_types = sorted(list(all_trial_types))
    print(f"Total trial types found: {all_trial_types}")
    
    # Calculate averages for trial types that have data from all three sources
    model_test_averages = []
    model_regular_averages = []
    rat_averages = []
    trial_type_data = []  # Store data with trial type info for sorting
    
    for trial_type in all_trial_types:
        # Get rats that have this trial type in all three data sources
        rats_with_data = [rat for rat in common_rats 
                         if trial_type in all_model_test_results[rat] 
                         and trial_type in all_rat_results[rat]]
        
        if len(rats_with_data) > 0:  # Only include if we have data from all sources
            test_values = [all_model_test_results[rat][trial_type] for rat in rats_with_data]
            regular_values = [all_model_regular_results[rat][trial_type] for rat in rats_with_data]
            rat_values = [all_rat_results[rat][trial_type] for rat in rats_with_data]
            
            # Convert numbers to letters for trial type label
            trial_type_label = f"{number_to_letter(trial_type[0])}{number_to_letter(trial_type[1])}"
            
            # Skip certain trial types (converted to letter format)
            if trial_type_label in ["AC", "AD", "BE", "CE"]:
                continue
            
            trial_type_data.append({
                'label': trial_type_label,
                'test_avg': np.mean(test_values),
                'regular_avg': np.mean(regular_values),
                'rat_avg': np.mean(rat_values),
                'original_type': trial_type
            })
            
            print(f"Trial type {trial_type_label}: {len(rats_with_data)} rats, "
                  f"Test avg: {np.mean(test_values):.3f}, "
                  f"Rat avg: {np.mean(rat_values):.3f}")
    
    # Define the desired order
    desired_order = ["AB", "BC", "CD", "DE", "BD", "AE"]
    
    # Sort trial_type_data according to desired order
    ordered_data = []
    for desired_label in desired_order:
        for data in trial_type_data:
            if data['label'] == desired_label:
                ordered_data.append(data)
                break
    
    # Extract the ordered data
    trial_type_labels = [data['label'] for data in ordered_data]
    model_test_averages = [data['test_avg'] for data in ordered_data]
    model_regular_averages = [data['regular_avg'] for data in ordered_data]
    rat_averages = [data['rat_avg'] for data in ordered_data]
    
    #if len(trial_type_labels) == 0:
        #print("Error: No trial types have data from all sources.")
        #return
    
    # Create output directory for plots
    output_dir = os.path.join(paths.betasort_data, "transitive_inference_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison plot
    save_path = os.path.join(output_dir, "ti_both_models_vs_rat_comparison.png")
    plot_ti_comparison(trial_type_labels, model_test_averages, model_regular_averages, rat_averages, save_path)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'trial_type': trial_type_labels,
        'model_test_performance': model_test_averages,
        'model_regular_performance': model_regular_averages,
        'rat_performance': rat_averages,
        'test_minus_rat': np.array(model_test_averages) - np.array(rat_averages),
        'regular_minus_rat': np.array(model_regular_averages) - np.array(rat_averages),
        'test_minus_regular': np.array(model_test_averages) - np.array(model_regular_averages)
    })
    
    results_path = os.path.join(output_dir, "ti_comparison_results.csv")
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