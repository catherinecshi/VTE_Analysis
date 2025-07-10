import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.paths import paths
from analysis import betasort_analysis
from models import betasort_test
from visualization import betasort_plots

def extract_daily_performance_data(all_data_df, rat, n_simulations=100, 
                                  xi=0.95, tau=0.05, threshold=0.8):
    """
    Extract day-by-day performance for both rat and model
    Returns lists of daily performance scores for comparison
    """
    # Initialize storage
    daily_rat_performance = []
    daily_model_performance = []
    
    # Initialize global state tracking
    global_U, global_L, global_R, global_N = {}, {}, {}, {}
    
    # Process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        print(f"Processing day {day} for {rat}...")
        
        # Extract data for this day
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        rewards = day_data["correct"].values
        
        # Identify stimuli present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        # Initialize model for this day
        model = betasort_test.Betasort(n_stimuli, rat, day, tau=tau, xi=xi)
        
        # Transfer state from previous days
        for stim_idx in range(n_stimuli):
            if stim_idx in global_U:
                model.U[stim_idx] = global_U[stim_idx]
                model.L[stim_idx] = global_L[stim_idx]
                model.R[stim_idx] = global_R[stim_idx]
                model.N[stim_idx] = global_N[stim_idx]
        
        # Find adjacent pairs for this day
        adjacent_pairs = []
        for i in range(n_stimuli - 1):
            if i in present_stimuli and (i + 1) in present_stimuli:
                adjacent_pairs.append((i, i + 1))
        
        # Calculate PRE-UPDATE model performance on adjacent pairs
        pre_update_model_rates = []
        for pair in adjacent_pairs:
            stim1, stim2 = pair
            model_choices = []
            for sim in range(n_simulations):
                model_choice = model.choose(stim1, stim2, False)
                model_choices.append(model_choice)
            
            # Model "correct" rate (choosing lower-valued stimulus)
            correct_rate = np.mean([1 if choice == stim1 else 0 for choice in model_choices])
            pre_update_model_rates.append(correct_rate)
        
        # Calculate rat performance on adjacent pairs for this day
        rat_performance_pairs = []
        for pair in adjacent_pairs:
            stim1, stim2 = pair
            # Find all trials with this pair
            pair_trials = []
            for t in range(len(chosen_idx)):
                chosen, unchosen = chosen_idx[t], unchosen_idx[t]
                if (min(chosen, unchosen), max(chosen, unchosen)) == pair:
                    pair_trials.append(rewards[t])
            
            if pair_trials:
                # Rat's reward rate on this pair
                pair_performance = np.mean(pair_trials)
                rat_performance_pairs.append(pair_performance)
        
        # Process all trials for this day (for model updating)
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            reward = rewards[t]
            
            # Update model based on actual choice and reward
            model_match_rate = 0.5  # Placeholder for match rate calculation
            model.update(chosen, unchosen, reward, model_match_rate, threshold=threshold)
        
        # Calculate POST-UPDATE model performance on adjacent pairs
        post_update_model_rates = []
        for pair in adjacent_pairs:
            stim1, stim2 = pair
            model_choices = []
            for sim in range(n_simulations):
                model_choice = model.choose(stim1, stim2, False)
                model_choices.append(model_choice)
            
            correct_rate = np.mean([1 if choice == stim1 else 0 for choice in model_choices])
            post_update_model_rates.append(correct_rate)
        
        # Store daily performance (using post-update model performance)
        if post_update_model_rates and rat_performance_pairs:
            daily_model_performance.append(np.mean(post_update_model_rates))
            daily_rat_performance.append(np.mean(rat_performance_pairs))
        else:
            # Fallback: use overall day performance if no adjacent pairs
            daily_rat_performance.append(np.mean(rewards))
            daily_model_performance.append(0.5)  # Placeholder
        
        # Update global states for next day
        for stim_idx in range(n_stimuli):
            global_U[stim_idx] = model.U[stim_idx]
            global_L[stim_idx] = model.L[stim_idx]
            global_R[stim_idx] = model.R[stim_idx]
            global_N[stim_idx] = model.N[stim_idx]
    
    return daily_rat_performance, daily_model_performance

def normalize_by_training_progress(all_rats_data, n_points=20):
    """
    Method 2: Normalize both rat and model performance by training progress (0-100%)
    Returns DataFrame with normalized data using Standard Error of Mean
    """
    proportions = np.linspace(0, 1, n_points)
    normalized_data = []
    
    for prop in proportions:
        rat_performances = []
        model_performances = []
        
        for rat_name, data in all_rats_data.items():
            rat_perf = data['rat_performance']
            model_perf = data['model_performance']
            n_days = len(rat_perf)
            
            if n_days > 1:
                # Find corresponding day index for this proportion
                day_index = int(prop * (n_days - 1))
                rat_performances.append(rat_perf[day_index])
                model_performances.append(model_perf[day_index])
        
        if rat_performances and model_performances:
            # Calculate means
            rat_mean = np.mean(rat_performances)
            model_mean = np.mean(model_performances)
            
            # Calculate Standard Error of Mean (SEM)
            rat_sem = np.std(rat_performances) / np.sqrt(len(rat_performances))
            model_sem = np.std(model_performances) / np.sqrt(len(model_performances))
            
            normalized_data.append({
                'training_progress': prop * 100,
                'rat_mean': rat_mean,
                'rat_sem': rat_sem,
                'model_mean': model_mean,
                'model_sem': model_sem,
                'n_rats': len(rat_performances)
            })
    
    return pd.DataFrame(normalized_data)

def plot_rat_vs_model_normalized(normalized_df, save_path=None):
    """
    Create line plot comparing rat vs model performance with Method 2 normalization
    """
    plt.figure(figsize=(12, 8))
    
    # Plot rat performance
    plt.errorbar(normalized_df['training_progress'], 
                normalized_df['rat_mean'],
                yerr=normalized_df['rat_sem'],
                label='Rat Performance', 
                color='#2E86AB', 
                linewidth=3, 
                marker='o', 
                markersize=6,
                capsize=5,
                capthick=2)
    
    # Plot model performance  
    plt.errorbar(normalized_df['training_progress'], 
                normalized_df['model_mean'],
                yerr=normalized_df['model_sem'],
                label='Model Performance', 
                color='#A23B72', 
                linewidth=3, 
                marker='s', 
                markersize=6,
                capsize=5,
                capthick=2,
                linestyle='--')
    
    plt.xlabel('Training Progress (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Performance', fontsize=14, fontweight='bold')
    plt.title('Rat vs Model Performance: Normalized by Training Progress', fontsize=16, fontweight='bold')
    
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add sample size annotation
    n_rats = normalized_df['n_rats'].iloc[0]
    plt.text(0.02, 0.98, f'N = {n_rats} rats', transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'rat_vs_model_normalized.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return plt

def analyze_all_rats_normalized_performance(data_path, save_path, use_existing_params=True):
    """
    Main function to extract and analyze normalized performance for all rats
    """
    all_rats_data = {}
    
    # Process each rat
    for rat in os.listdir(data_path):
        if "BP06" in rat or "BP07" in rat:  # Skip these rats as in original code
            continue
            
        rat_path = os.path.join(data_path, rat)
        for root, _, files in os.walk(rat_path):
            for file in files:
                if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                    continue

                file_path = os.path.join(root, file)
                file_csv = pd.read_csv(file_path)
                
                try:
                    print(f"\nProcessing {rat}...")
                    
                    # Get optimal parameters if available
                    if use_existing_params:
                        rat_results_path = os.path.join(save_path, rat, "results.csv")
                        if os.path.exists(rat_results_path):
                            results_df = pd.read_csv(rat_results_path)
                            xi = results_df['best_xi'].iloc[0]
                            tau = results_df['best_tau'].iloc[0] 
                            threshold = results_df['best_threshold'].iloc[0]
                        else:
                            xi, tau, threshold = 0.95, 0.05, 0.8  # Defaults
                    else:
                        xi, tau, threshold = 0.95, 0.05, 0.8  # Defaults
                    
                    # Extract daily performance data
                    rat_performance, model_performance = extract_daily_performance_data(
                        file_csv, rat, xi=xi, tau=tau, threshold=threshold)
                    
                    if len(rat_performance) > 0 and len(model_performance) > 0:
                        all_rats_data[rat] = {
                            'rat_performance': rat_performance,
                            'model_performance': model_performance,
                            'n_days': len(rat_performance)
                        }
                        print(f"  Extracted {len(rat_performance)} days of data")
                    
                except Exception as e:
                    print(f"Error processing {rat}: {e}")
                    continue
    
    print(f"\nSuccessfully processed {len(all_rats_data)} rats")
    
    if len(all_rats_data) == 0:
        print("No data extracted! Check file paths and processing.")
        return None, None
    
    # Apply Method 2 normalization
    print("Applying Method 2 normalization...")
    normalized_df = normalize_by_training_progress(all_rats_data, n_points=20)
    
    # Create plot
    print("Creating comparison plot...")
    plot_rat_vs_model_normalized(normalized_df, save_path)
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NORMALIZED PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    early_rat = normalized_df[normalized_df['training_progress'] <= 33]['rat_mean'].mean()
    early_model = normalized_df[normalized_df['training_progress'] <= 33]['model_mean'].mean()
    
    late_rat = normalized_df[normalized_df['training_progress'] >= 67]['rat_mean'].mean()
    late_model = normalized_df[normalized_df['training_progress'] >= 67]['model_mean'].mean()
    
    print(f"Early Training (0-33%):  Rat={early_rat:.3f}, Model={early_model:.3f}")
    print(f"Late Training (67-100%): Rat={late_rat:.3f}, Model={late_model:.3f}")
    print(f"Overall Correlation: {np.corrcoef(normalized_df['rat_mean'], normalized_df['model_mean'])[0,1]:.3f}")
    
    # Save results
    output_file = os.path.join(save_path, 'normalized_performance_comparison.csv')
    normalized_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return normalized_df, all_rats_data

# Run the analysis
if __name__ == "__main__":
    data_path = paths.preprocessed_data_model
    save_path = paths.betasort_data
    
    normalized_results, individual_data = analyze_all_rats_normalized_performance(
        data_path, save_path, use_existing_params=True)