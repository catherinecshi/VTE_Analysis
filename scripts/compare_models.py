import os
import json
import glob
import time
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy.stats import pointbiserialr

from src import helper
from src import bayesian_learner
from src import neural_network
from src import rw_generalization
from src import temporal_difference_learning
from src import value_transfer

def compare_model_to_data(all_data_df, rat, model_class, model_params, n_simulations=100):
    """
    Compare any model to rat data
    
    Parameters:
    - all_data_df: DataFrame with rat data
    - rat: rat identifier
    - model_class: class of the model to use
    - model_params: dictionary of parameters for the model
    - n_simulations: number of simulations per trial
    
    Returns:
    - final_model: trained model after all days
    - all_models: dictionary of models for each day
    - match_rates: list of match rates for each day
    """
    all_models = {}
    match_rates = []
    
    # Process each day separately
    for day, day_data in all_data_df.groupby('Day'):
        # Extract data
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        
        # Identify stimuli present on this day
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        n_stimuli = max(present_stimuli) + 1
        
        # Initialize model
        model = model_class(n_stimuli=n_stimuli, **model_params)
        
        # Process trials for this day
        day_matches = []
        
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            
            # Run simulations to get choice probability
            model_choices = np.zeros(n_simulations)
            for sim in range(n_simulations):
                model_choice = model.choose([chosen, unchosen])
                model_choices[sim] = model_choice
            
            # Calculate match rate
            model_match_rate = np.mean(model_choices == chosen)
            day_matches.append(model_match_rate)
            
            # Determine reward (as in your original code)
            reward = 1 if chosen < unchosen else 0
            
            # Update model
            model.update(chosen, unchosen, reward)
        
        # Store day's match rate and model
        match_rate = np.mean(day_matches)
        match_rates.append(match_rate)
        all_models[day] = model
    
    final_day = max(all_models.keys())
    return all_models[final_day], all_models, match_rates

def save_model_performance(models_dict, output_dir="model_comparison", prefix=""):
    """
    Save performance metrics for multiple models to CSV files
    
    Parameters:
        - models_dict: Dictionary with model name as key and dict of results as value
        - output_dir: Directory to save CSV files
        - prefix: Optional prefix for filenames
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract overall performance
    overall_performance = {
        "model": [],
        "mean_match_rate": [],
        "std_error": []
    }
    
    # Daily performance dataframe
    daily_performance = []
    
    for model_name, results in models_dict.items():
        # Add to overall performance
        overall_performance["model"].append(model_name)
        overall_performance["mean_match_rate"].append(results["mean_match_rate"])
        
        # Calculate standard error of the mean
        daily_rates = results["daily_match_rates"]
        sem = np.std(daily_rates) / np.sqrt(len(daily_rates))
        overall_performance["std_error"].append(sem)
        
        # Add to daily performance
        for day_idx, rate in enumerate(daily_rates):
            daily_performance.append({
                "model": model_name,
                "day": day_idx + 1,
                "match_rate": rate
            })
    
    # Convert to DataFrames
    overall_df = pd.DataFrame(overall_performance)
    daily_df = pd.DataFrame(daily_performance)
    
    # Save to CSV
    overall_df.to_csv(os.path.join(output_dir, f"{prefix}overall_performance.csv"), index=False)
    daily_df.to_csv(os.path.join(output_dir, f"{prefix}daily_performance.csv"), index=False)
    
    return overall_df, daily_df

def plot_model_comparison(overall_df, output_dir="model_comparison", filename="model_comparison.png"):
    """
    Create bar plot comparing model performance with SEM error bars
    
    Parameters:
        - overall_df: DataFrame with model performance metrics
        - output_dir: Directory to save the plot
        - filename: Filename for the plot
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Sort models by performance
    sorted_df = overall_df.sort_values("mean_match_rate", ascending=False)
    
    # Create bar plot with error bars
    bar_plot = sns.barplot(
        x="model", 
        y="mean_match_rate", 
        data=sorted_df,
        palette="viridis"
    )
    
    # Add error bars manually
    for i, bar in enumerate(bar_plot.patches):
        bar_height = bar.get_height()
        error = sorted_df.iloc[i]["std_error"]
        bar_plot.errorbar(
            x=i, 
            y=bar_height,
            yerr=error,
            color='black',
            capsize=5,
            capthick=1,
            elinewidth=1
        )
    
    # Add chance level line
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot
    plt.title("Model Comparison: Rat Choice Prediction Accuracy", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Mean Match Rate", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.4, 1.0)  # Adjust as needed
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def check_transitive_inference_across_models(models_dict, n_simulations=100):
    """
    Check transitive inference performance for all possible stimulus pairs across models
    
    Parameters:
        - models_dict: Dictionary with model name as key and model as value
        - n_simulations: Number of simulations for each pair
    
    Returns:
        - results_dict: Dictionary with model name as key and TI results as value
    """
    
    results_dict = {}
    
    for model_name, model in models_dict.items():
        # Get number of stimuli from model
        n_stimuli = model.n_stimuli if hasattr(model, "n_stimuli") else 5  # Default to 5 if not specified
        
        # Check TI performance for all possible pairs
        model_results = {}
        
        for chosen_idx in range(0, n_stimuli-1):
            for other_idx in range(chosen_idx + 1, n_stimuli):
                # Skip same element
                if chosen_idx == other_idx:
                    continue
                
                # Simulate choices
                model_choices = np.zeros(n_simulations)
                for sim in range(n_simulations):
                    model_choice = model.choose([chosen_idx, other_idx])
                    model_choices[sim] = model_choice
                
                # Calculate match rate (how often model chooses the lower-valued stimulus)
                model_match_rate = np.mean(model_choices == chosen_idx)
                
                # Store results
                model_results[(chosen_idx, other_idx)] = model_match_rate
        
        results_dict[model_name] = model_results
    
    return results_dict

def compare_ti_with_actual(ti_results_dict, actual_ti_df, output_dir="model_comparison", prefix=""):
    """
    Compare model TI performance with actual TI performance and save results
    
    Parameters:
        - ti_results_dict: Dictionary with model TI performance
        - actual_ti_df: DataFrame with actual TI performance
        - output_dir: Directory to save results
        - prefix: Optional prefix for filenames
    
    Returns:
        - comparison_df: DataFrame with comparison results
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize comparison data
    comparison_data = []
    
    # Get all unique stimulus pairs
    all_pairs = set()
    for model_results in ti_results_dict.values():
        all_pairs.update(model_results.keys())
    
    # For each model and pair, compare with actual performance
    for model_name, model_results in ti_results_dict.items():
        model_errors = []
        
        for pair in all_pairs:
            if pair in model_results:
                # Get model performance for this pair
                model_perf = model_results[pair]
                
                # Get actual performance from DataFrame
                actual_row = actual_ti_df[(actual_ti_df['stim1'] == pair[0]) & 
                                         (actual_ti_df['stim2'] == pair[1])]
                
                if len(actual_row) > 0:
                    actual_perf = actual_row['accuracy'].values[0]
                    
                    # Calculate absolute error
                    error = abs(model_perf - actual_perf)
                    model_errors.append(error)
                    
                    # Add to comparison data
                    comparison_data.append({
                        'model': model_name,
                        'stim1': pair[0],
                        'stim2': pair[1],
                        'model_accuracy': model_perf,
                        'actual_accuracy': actual_perf,
                        'error': error
                    })
        
        # Add summary row
        comparison_data.append({
            'model': model_name,
            'stim1': -1,  # Indicator for summary row
            'stim2': -1,
            'model_accuracy': np.nan,
            'actual_accuracy': np.nan,
            'error': np.mean(model_errors),  # Mean error across all pairs
            'error_sem': np.std(model_errors) / np.sqrt(len(model_errors))  # SEM
        })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_df.to_csv(os.path.join(output_dir, f"{prefix}ti_comparison.csv"), index=False)
    
    # Create summary DataFrame for plotting
    summary_df = comparison_df[comparison_df['stim1'] == -1][['model', 'error', 'error_sem']]
    summary_df = summary_df.sort_values('error')  # Sort by error (ascending)
    
    return comparison_df, summary_df

def plot_ti_comparison(summary_df, output_dir="model_comparison", filename="ti_comparison.png"):
    """
    Create bar plot comparing model TI performance with SEM error bars
    
    Parameters:
        - summary_df: DataFrame with model TI performance summary
        - output_dir: Directory to save the plot
        - filename: Filename for the plot
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if necessary columns exist for plotting
    if 'error' not in summary_df.columns or 'error_sem' not in summary_df.columns:
        print("Cannot create TI comparison plot: missing error metrics")
        return
    
    # Set style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create bar plot WITHOUT error bars (we'll add them manually)
    ax = plt.subplot(111)
    bar_plot = sns.barplot(
        x="model", 
        y="error", 
        data=summary_df,
        errorbar=None,  # Don't use built-in error bars
        palette="viridis",
        ax=ax
    )
    
    # Add error bars manually
    for i, row in summary_df.reset_index(drop=True).iterrows():
        ax.errorbar(
            x=i, 
            y=row["error"],
            yerr=row["error_sem"],
            color='black',
            capsize=5,
            fmt='none'  # This prevents adding markers
        )
    
    # Customize plot
    plt.title("Model Comparison: Transitive Inference Performance", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Mean Absolute Error", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def compute_vte_uncertainty_correlations(all_data_df, models_dict, output_dir="model_comparison", prefix=""):
    """
    Compute correlations between VTE behavior and model uncertainty for each model
    
    Parameters:
        - all_data_df: DataFrame with rat data including VTE
        - models_dict: Dictionary with model name as key and dict of day models as value
        - output_dir: Directory to save results
        - prefix: Optional prefix for filenames
    
    Returns:
        - correlation_df: DataFrame with correlation results
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data for combined DataFrame
    combined_data = []
    
    # Process each day
    for day, day_data in all_data_df.groupby('Day'):
        # Check if VTE column exists
        if 'VTE' not in day_data.columns:
            print(f"Warning: No VTE data found for day {day}. Skipping correlation analysis.")
            continue
        
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        vtes = day_data["VTE"].values
        
        # For each trial
        for t in range(len(chosen_idx)):
            chosen = chosen_idx[t]
            unchosen = unchosen_idx[t]
            vte = 1 if vtes[t] else 0
            
            # Base trial data
            trial_data = {
                "day": day,
                "trial": t,
                "stim1": min(chosen, unchosen),
                "stim2": max(chosen, unchosen),
                "pair": f"{min(chosen, unchosen)}-{max(chosen, unchosen)}",
                "vte_occurred": vte
            }
            
            # Get uncertainty from each model
            for model_name, day_models in models_dict.items():
                if day in day_models:
                    model = day_models[day]
                    uncertainty = model.get_uncertainty(min(chosen, unchosen), max(chosen, unchosen))
                    trial_data[f"{model_name}_uncertainty"] = uncertainty
            
            combined_data.append(trial_data)
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Save combined data
    combined_df.to_csv(os.path.join(output_dir, f"{prefix}vte_uncertainty_data.csv"), index=False)
    
    # Calculate correlations for each model
    correlation_results = []
    
    for model_name in models_dict.keys():
        uncertainty_col = f"{model_name}_uncertainty"
        
        if uncertainty_col in combined_df.columns:
            # Overall correlation
            r, p = pointbiserialr(combined_df["vte_occurred"], combined_df[uncertainty_col])
            correlation_results.append({
                "model": model_name,
                "scope": "overall",
                "pair": "all",
                "correlation": r,
                "p_value": p,
                "significant": p < 0.05
            })
            
            # Correlation by pair
            for pair, pair_data in combined_df.groupby("pair"):
                if len(pair_data) >= 5 and pair_data["vte_occurred"].nunique() > 1:
                    r, p = pointbiserialr(pair_data["vte_occurred"], pair_data[uncertainty_col])
                    correlation_results.append({
                        "model": model_name,
                        "scope": "pair",
                        "pair": pair,
                        "correlation": r,
                        "p_value": p,
                        "significant": p < 0.05
                    })
    
    # Convert to DataFrame
    correlation_df = pd.DataFrame(correlation_results)
    
    # Save correlations
    correlation_df.to_csv(os.path.join(output_dir, f"{prefix}vte_uncertainty_correlations.csv"), index=False)
    
    # Create summary for plotting
    summary_df = correlation_df[correlation_df["scope"] == "overall"].sort_values("correlation", ascending=False)
    
    return correlation_df, summary_df, combined_df

def plot_vte_uncertainty_correlations(summary_df, output_dir="model_comparison", filename="vte_correlation_comparison.png"):
    """
    Create bar plot comparing VTE-uncertainty correlations across models
    
    Parameters:
        - summary_df: DataFrame with VTE-uncertainty correlation summary
        - output_dir: Directory to save the plot
        - filename: Filename for the plot
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create bar plot with error bars (bootstrapped from correlation_df)
    bar_plot = sns.barplot(
        x="model", 
        y="correlation", 
        data=summary_df,
        palette="viridis"
    )

    # Add error bars manually
    for i, row in summary_df.iterrows():
        error = row.get("correlation_sem", 0.05)  # Use default if not available
        bar_plot.errorbar(
            x=i, 
            y=row["correlation"],
            yerr=error,
            color='black',
            capsize=5,
            fmt='none'
        )
    
    # Add significance markers
    for i, is_sig in enumerate(summary_df["significant"]):
        if is_sig:
            bar_height = summary_df.iloc[i]["correlation"]
            bar_plot.text(
                i, 
                bar_height + 0.02,
                "*",
                ha='center',
                va='bottom',
                fontsize=16
            )
    
    # Customize plot
    plt.title("Model Comparison: VTE-Uncertainty Correlations", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Correlation (r)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_value_history(model_dict, filename, output_dir, attribute="value_history"):
    """
    Plot value history across days for a model
    
    Parameters:
        - model_dict: Dictionary with day as key and model as value
        - filename: Base filename for the plot
        - output_dir: Directory to save plots
        - attribute: Model attribute to plot
    """
    # Plot for each day
    for day, model in model_dict.items():
        if hasattr(model, attribute):
            plt.figure(figsize=(12, 8))
            
            # Get history array
            history = getattr(model, attribute)
            
            # If history is a list of arrays, convert to array
            if isinstance(history, list) and isinstance(history[0], np.ndarray):
                history = np.array(history)
            
            # Plot each stimulus
            for i in range(model.n_stimuli):
                plt.plot(history[:, i], label=f"Stimulus {i}")
                
            plt.xlabel("Trial")
            plt.ylabel(attribute.replace("_history", "").title())
            plt.title(f"{attribute.replace('_history', '').title()} Evolution (Day {day})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"{filename}_day{day}.png"), dpi=300)
            plt.close()
    
    # Create a combined plot across all days
    plt.figure(figsize=(15, 10))

    # Different colors for different days
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_dict)))

    for idx, (day, model) in enumerate(sorted(model_dict.items())):
        if hasattr(model, attribute):
            history = getattr(model, attribute)
            
            # If history is a list of arrays, convert to array
            if isinstance(history, list) and isinstance(history[0], np.ndarray):
                history = np.array(history)
            
            # Get final values for this day
            final_values = history[-1]
            
            # Add to plot with day-specific color
            for i in range(model.n_stimuli):
                plt.plot(idx, final_values[i], 'o', color=colors[idx], 
                        markersize=10, label=f"Day {day}, Stim {i}" if i == 0 else "")
                
                # Add connecting lines between days
                if idx > 0:
                    prev_day = sorted(model_dict.keys())[idx-1]
                    prev_model = model_dict[prev_day]
                    
                    # Only try to connect if this stimulus exists in the previous day
                    if i < prev_model.n_stimuli:  # Check if stimulus exists in previous day
                        prev_history = getattr(prev_model, attribute)
                        
                        if isinstance(prev_history, list) and isinstance(prev_history[0], np.ndarray):
                            prev_history = np.array(prev_history)
                        
                        prev_final = prev_history[-1][i]
                        
                        plt.plot([idx-1, idx], [prev_final, final_values[i]], '-', 
                                color=plt.cm.tab10(i), alpha=0.5)
    
    plt.xlabel("Day")
    plt.ylabel(attribute.replace("_history", "").title())
    plt.title(f"{attribute.replace('_history', '').title()} Evolution Across Days")
    plt.xticks(range(len(model_dict)), [f"Day {d}" for d in sorted(model_dict.keys())])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{filename}_across_days.png"), dpi=300)
    plt.close()

def plot_prediction_history(model_dict, filename, output_dir):
    """
    Plot prediction history for neural network model
    
    Parameters:
        - model_dict: Dictionary with day as key and model as value
        - filename: Base filename for the plot
        - output_dir: Directory to save plots
    """
    
    # Collect all predictions across days
    all_predictions = []
    day_boundaries = [0]  # Trial indices where days change
    
    for day, model in sorted(model_dict.items()):
        if hasattr(model, 'prediction_history'):
            all_predictions.extend(model.prediction_history)
            day_boundaries.append(len(all_predictions))
    
    # Plot predictions over all days
    plt.figure(figsize=(15, 8))
    
    # Plot predictions
    plt.plot(all_predictions, 'b-', alpha=0.7)
    
    # Add day boundaries
    for boundary in day_boundaries[1:-1]:  # Skip first and last
        plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.5)
    
    # Add day labels
    days = sorted(model_dict.keys())
    for i in range(len(days)):
        if i < len(day_boundaries) - 1:
            mid_point = (day_boundaries[i] + day_boundaries[i+1]) // 2
            plt.text(mid_point, 0.05, f"Day {days[i]}", ha='center')
    
    plt.xlabel("Trial")
    plt.ylabel("Prediction")
    plt.title("Neural Network Prediction History Across Days")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    plt.close()

def plot_value_matrix(model_dict, filename, output_dir):
    """
    Plot value transfer matrix for the final model
    
    Parameters:
        - model_dict: Dictionary with day as key and model as value
        - filename: Base filename for the plot
        - output_dir: Directory to save plots
    """
    
    # Get the final day model
    final_day = max(model_dict.keys())
    final_model = model_dict[final_day]
    
    # Plot value matrix
    plt.figure(figsize=(10, 8))
    
    if hasattr(final_model, 'V'):
        # Create heatmap
        sns.heatmap(final_model.V, annot=True, cmap='viridis', 
                    xticklabels=range(final_model.n_stimuli), 
                    yticklabels=range(final_model.n_stimuli))
        
        plt.xlabel("Stimulus")
        plt.ylabel("Stimulus")
        plt.title(f"Value Transfer Matrix (Final Day {final_day})")
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
        plt.close()

def plot_uncertainty_across_days(model_dict, filename, output_dir):
    # Get all days sorted
    days = sorted(model_dict.keys())
    
    # Collect all unique stimulus pairs across all days
    all_pairs = set()
    for day, model in model_dict.items():
        n_stimuli = model.n_stimuli
        for i in range(n_stimuli-1):
            for j in range(i+1, n_stimuli):
                all_pairs.add((i, j))
    
    # Convert to sorted list for consistent ordering
    pairs = sorted(list(all_pairs))
    
    # Initialize uncertainty data with NaN values (for pairs that don't exist on some days)
    uncertainty_data = np.full((len(days), len(pairs)), np.nan)
    
    # Collect uncertainty across days for each pair
    for day_idx, day in enumerate(days):
        model = model_dict[day]
        
        for pair_idx, (stim1, stim2) in enumerate(pairs):
            # Only calculate uncertainty if both stimuli exist in this model
            if stim1 < model.n_stimuli and stim2 < model.n_stimuli:
                try:
                    uncertainty = model.get_uncertainty(stim1, stim2)
                    uncertainty_data[day_idx, pair_idx] = uncertainty
                except Exception as e:
                    print(f"Error calculating uncertainty for pair {stim1}-{stim2} on day {day}: {e}")
    
    # For plotting, use masked arrays to handle NaN values
    # ... rest of the function remains the same but use masked_data for the heatmap
    masked_data = np.ma.masked_invalid(uncertainty_data.T)
    plt.imshow(masked_data, aspect='auto', cmap='viridis')
    
    # Plot uncertainty evolution across days
    plt.figure(figsize=(15, 10))
    
    for pair_idx, (stim1, stim2) in enumerate(pairs):
        plt.plot(days, uncertainty_data[:, pair_idx], 'o-', 
                 label=f"Pair {stim1}-{stim2}", linewidth=2)
    
    plt.xlabel("Day")
    plt.ylabel("Uncertainty")
    plt.title("Stimulus Pair Uncertainty Evolution Across Days")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    plt.close()
    
    # Plot heatmap of uncertainty across days and pairs
    plt.figure(figsize=(12, 8))
    
    plt.imshow(uncertainty_data.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Uncertainty')
    plt.xlabel("Day")
    plt.ylabel("Stimulus Pair")
    plt.title("Uncertainty Heatmap Across Days and Stimulus Pairs")
    plt.xticks(range(len(days)), days)
    plt.yticks(range(len(pairs)), [f"{s1}-{s2}" for s1, s2 in pairs])
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{filename}_heatmap.png"), dpi=300)
    plt.close()

def compare_ti_performance(betasort_ti_results, new_model_ti_results, actual_ti_df=None, output_dir="model_comparison"):
    """
    Compare transitive inference performance between Betasort and new models
    
    Parameters:
        - betasort_ti_results: TI results from Betasort
        - new_model_ti_results: Dictionary of TI results from new models
        - actual_ti_df: Optional DataFrame with actual TI performance
        - output_dir: Directory to save results
    
    Returns:
        - comparison_df: DataFrame with comparison results
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all TI results
    all_ti_results = {"Betasort": betasort_ti_results}
    all_ti_results.update(new_model_ti_results)
    
    # Create a DataFrame for direct comparison
    comparison_data = []
    
    # Get all unique stimulus pairs
    all_pairs = set()
    for model_results in all_ti_results.values():
        all_pairs.update(model_results.keys())
    
    # For each pair, get predictions from all models
    for pair in sorted(all_pairs):
        row_data = {
            "stim1": pair[0],
            "stim2": pair[1],
            "pair": f"{pair[0]}-{pair[1]}"
        }
        
        # Add actual accuracy if available
        if actual_ti_df is not None:
            actual_row = actual_ti_df[(actual_ti_df['stim1'] == pair[0]) & 
                                      (actual_ti_df['stim2'] == pair[1])]
            if len(actual_row) > 0:
                row_data["actual_accuracy"] = actual_row['accuracy'].values[0]
        
        # Add model predictions
        for model_name, model_results in all_ti_results.items():
            if pair in model_results:
                row_data[f"{model_name}_accuracy"] = model_results[pair]
        
        comparison_data.append(row_data)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(output_dir, "ti_comparison_detail.csv"), index=False)
    
    # Calculate model error compared to actual performance
    if actual_ti_df is not None and "actual_accuracy" in comparison_df.columns:
        summary_data = []
        
        for model_name in all_ti_results.keys():
            model_col = f"{model_name}_accuracy"
            if model_col in comparison_df.columns:
                # Calculate absolute error for each pair
                errors = abs(comparison_df[model_col] - comparison_df["actual_accuracy"])
                
                # Calculate mean error and SEM
                mean_error = errors.mean()
                error_sem = errors.std() / np.sqrt(len(errors))
                
                summary_data.append({
                    "model": model_name,
                    "error": mean_error,
                    "error_sem": error_sem
                })
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("error")  # Sort by error (ascending)
        
        # Save summary
        summary_df.to_csv(os.path.join(output_dir, "ti_comparison_summary.csv"), index=False)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Create bar plot
        bar_plot = sns.barplot(
            x="model", 
            y="error", 
            data=summary_df,
            yerr=summary_df["error_sem"],
            capsize=0.2,
            palette="viridis"
        )
        
        # Customize plot
        plt.title("Model Comparison: Transitive Inference Performance", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Mean Absolute Error", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bar_plot.patches):
            bar_value = summary_df.iloc[i]["error"]
            bar_plot.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar_value:.3f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ti_comparison.png"), dpi=300)
        plt.close()
        
        return comparison_df, summary_df
    
    return comparison_df

def combine_results_across_rats(comparison_output_path):
    """
    Combine model comparison results across all rats
    
    Parameters:
        - comparison_output_path: Base directory containing results for all rats
        
    Returns:
        - combined_df: DataFrame with combined results
    """
    # Find all overall performance CSV files
    performance_files = glob.glob(os.path.join(comparison_output_path, "*", "overall_performance.csv"))
    
    # Initialize list to store DataFrames
    all_performance_dfs = []
    
    # Process each file
    for file_path in performance_files:
        # Extract rat ID from path
        rat = os.path.basename(os.path.dirname(file_path))
        
        # Load performance data
        try:
            df = pd.read_csv(file_path)
            df['rat'] = rat  # Add rat identifier
            all_performance_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Combine all DataFrames
    if all_performance_dfs:
        combined_df = pd.concat(all_performance_dfs, ignore_index=True)
        
        # Save combined results
        combined_df.to_csv(os.path.join(comparison_output_path, "combined_performance.csv"), index=False)
        
        return combined_df
    else:
        print("No performance files found")
        return None

def plot_combined_results(combined_df, output_path):
    """
    Create visualizations of combined results across rats
    
    Parameters:
        - combined_df: DataFrame with combined results
        - output_path: Directory to save plots
    """
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Calculate aggregated statistics by model
    model_stats = combined_df.groupby('model').agg({
        'mean_match_rate': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten MultiIndex columns
    model_stats.columns = ['model', 'mean_rate', 'std_rate', 'count']
    
    # Calculate standard error
    model_stats['sem_rate'] = model_stats['std_rate'] / np.sqrt(model_stats['count'])
    
    # Sort by mean rate
    model_stats = model_stats.sort_values('mean_rate', ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create a copy of the dataframe with percentages
    model_stats_pct = model_stats.copy()
    model_stats_pct['mean_rate'] = model_stats_pct['mean_rate'] * 100
    model_stats_pct['sem_rate'] = model_stats_pct['sem_rate'] * 100
    
    # Create bar plot without error bars first
    bar_plot = sns.barplot(
        x='model', 
        y='mean_rate', 
        data=model_stats_pct,
        palette="viridis"
    )

    # Add error bars manually
    for i, row in model_stats_pct.iterrows():
        bar_plot.errorbar(
            x=i, 
            y=row['mean_rate'],
            yerr=row['sem_rate'],
            color='black',
            capsize=5,
            fmt='none'  # This prevents adding markers
        )
    
    # Add chance level line (50%)
    plt.axhline(y=50, color='k', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot
    plt.title("Model Comparison Across All Rats", fontsize=24)
    plt.xlabel("Model", fontsize=20)
    plt.ylabel("Mean Match Rate (%)", fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(40, 100)  # Adjust as needed (equivalent to 0.4 to 1.0 in proportions)
    
    # Add value labels on bars (as percentages)
    for i, bar in enumerate(bar_plot.patches):
        bar_value = model_stats_pct.iloc[i]["mean_rate"]
        bar_plot.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,  # Adjusted for percentage scale
            f"{bar_value:.1f}%",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "combined_performance.png"), dpi=300)
    plt.close()
    
    # Create a copy of combined_df with percentages
    combined_df_pct = combined_df.copy()
    combined_df_pct['mean_match_rate'] = combined_df_pct['mean_match_rate'] * 100
    
    # Create boxplot to show distribution across rats
    plt.figure(figsize=(12, 8))
    
    boxplot = sns.boxplot(
        x='model', 
        y='mean_match_rate', 
        data=combined_df_pct,
        palette="viridis_r",
        order=model_stats['model']  # Use same order as bar plot
    )
    
    # Add individual points
    sns.stripplot(
        x='model', 
        y='mean_match_rate', 
        data=combined_df_pct,
        color='black',
        alpha=0.5,
        jitter=True,
        order=model_stats['model']  # Use same order as bar plot
    )
    
    # Add chance level line (50%)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot
    plt.title("Model Performance Distribution Across Rats", fontsize=24)
    plt.xlabel("Model", fontsize=20)
    plt.ylabel("Match Rate (%)", fontsize=20)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "performance_distribution.png"), dpi=300)
    plt.close()
    
    # Create heatmap showing performance by model and rat
    # First convert to percentages
    pivot_df = combined_df.pivot(index='rat', columns='model', values='mean_match_rate') * 100
    
    plt.figure(figsize=(14, 10))
    
    # Sort models by average performance
    model_order = model_stats['model'].tolist()
    pivot_df = pivot_df[model_order]
    
    # Create heatmap
    ax = plt.subplot(111)
    heatmap = sns.heatmap(
        pivot_df,
        annot=False,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={'label': 'Match Rate (%)'},
        ax=ax,
        annot_kws={'size': 20}
    )
    
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    plt.title("Performance by Model and Rat", fontsize=24)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "performance_heatmap.png"), dpi=300)
    plt.close()

def aggregate_vte_correlations(comparison_output_path):
    """
    Aggregate VTE-uncertainty correlation results across rats
    
    Parameters:
        - comparison_output_path: Base directory containing results for all rats
        
    Returns:
        - combined_df: DataFrame with combined correlation results
    """
    # Find all VTE correlation CSV files
    correlation_files = glob.glob(os.path.join(comparison_output_path, "*", "vte_uncertainty_correlations.csv"))
    
    # Initialize list to store DataFrames
    all_correlation_dfs = []
    
    # Process each file
    for file_path in correlation_files:
        # Extract rat ID from path
        rat = os.path.basename(os.path.dirname(file_path))
        
        # Load correlation data
        try:
            df = pd.read_csv(file_path)
            df['rat'] = rat  # Add rat identifier
            all_correlation_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Combine all DataFrames
    if all_correlation_dfs:
        combined_df = pd.concat(all_correlation_dfs, ignore_index=True)
        
        # Save combined results
        combined_df.to_csv(os.path.join(comparison_output_path, "combined_vte_correlations.csv"), index=False)
        
        return combined_df
    else:
        print("No VTE correlation files found")
        return None

def plot_combined_vte_correlations(combined_df, output_path):
    """
    Create visualizations of combined VTE correlation results
    
    Parameters:
        - combined_df: DataFrame with combined VTE correlation results
        - output_path: Directory to save plots
    """
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Filter for overall correlations
    overall_df = combined_df[combined_df['scope'] == 'overall']
    
    # Calculate aggregated statistics by model
    model_stats = overall_df.groupby('model').agg({
        'correlation': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten MultiIndex columns
    model_stats.columns = ['model', 'mean_corr', 'std_corr', 'count']
    
    # Calculate standard error
    model_stats['sem_corr'] = model_stats['std_corr'] / np.sqrt(model_stats['count'])
    
    # Sort by absolute mean correlation (descending)
    model_stats['abs_corr'] = abs(model_stats['mean_corr'])
    model_stats = model_stats.sort_values('abs_corr', ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create bar plot without error bars
    bar_plot = sns.barplot(
        x='model', 
        y='mean_corr', 
        data=model_stats,
        palette="viridis"
    )

    # Add error bars manually
    for i, row in model_stats.iterrows():
        bar_plot.errorbar(
            x=i, 
            y=row['mean_corr'],
            yerr=row['sem_corr'],
            color='black',
            capsize=5,
            fmt='none'
        )
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    
    # Customize plot
    plt.title("VTE-Uncertainty Correlation Across All Rats", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Mean Correlation (r)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bar_plot.patches):
        bar_value = model_stats.iloc[i]["mean_corr"]
        bar_plot.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01 if bar_value >= 0 else bar.get_height() - 0.05,
            f"{bar_value:.3f}",
            ha='center',
            va='bottom' if bar_value >= 0 else 'top',
            fontsize=10
        )
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "combined_vte_correlation.png"), dpi=300)
    plt.close()
    
    # Create boxplot to show distribution across rats
    plt.figure(figsize=(12, 8))
    
    boxplot = sns.boxplot(
        x='model', 
        y='correlation', 
        data=overall_df,
        palette="viridis",
        order=model_stats['model']  # Use same order as bar plot
    )
    
    # Add individual points
    sns.stripplot(
        x='model', 
        y='correlation', 
        data=overall_df,
        color='black',
        alpha=0.5,
        jitter=True,
        order=model_stats['model']  # Use same order as bar plot
    )
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    
    # Customize plot
    plt.title("VTE-Uncertainty Correlation Distribution Across Rats", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Correlation (r)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "vte_correlation_distribution.png"), dpi=300)
    plt.close()

def plot_model_parameters(model_dict, model_name, output_dir="model_plots"):
    """
    Plot model-specific parameters based on model type
    
    Parameters:
        - model_dict: Dictionary with day as key and model as value
        - model_name: Name of the model
        - output_dir: Directory to save plots
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model type from first model instance
    first_day = min(model_dict.keys())
    model = model_dict[first_day]
    model_type = type(model).__name__
    
    # Plot based on model type
    if model_type == "RWGeneralization":
        # Plot value history
        plot_value_history(model_dict, f"{model_name}_value_history", output_dir)
        
    elif model_type == "BayesianLearner":
        # Plot mean history
        plot_value_history(model_dict, f"{model_name}_mean_history", output_dir, attribute="mean_history")
        
        # Plot variance history
        plot_value_history(model_dict, f"{model_name}_variance_history", output_dir, attribute="variance_history")
        
    elif model_type == "ValueTransferModel":
        # Plot final value matrix
        plot_value_matrix(model_dict, f"{model_name}_value_matrix", output_dir)
        
    elif model_type == "TDLambdaModel":
        # Plot value history
        plot_value_history(model_dict, f"{model_name}_value_history", output_dir)
        
        # Plot eligibility trace history
        plot_value_history(model_dict, f"{model_name}_e_trace_history", output_dir, attribute="e_trace_history")
        
    elif model_type == "BayesianTDModel":
        # Plot value history
        plot_value_history(model_dict, f"{model_name}_value_history", output_dir)
        
        # Plot uncertainty history
        plot_value_history(model_dict, f"{model_name}_uncertainty_history", output_dir, attribute="uncertainty_history")
        
    elif model_type == "NeuralNetworkModel":
        # Plot prediction history
        plot_prediction_history(model_dict, f"{model_name}_prediction_history", output_dir)
        
    elif model_type == "Betasort":
        # Use existing plotting functions
        for day, model in model_dict.items():
            plt.figure(figsize=(12, 8))
            
            # Plot positions
            positions = np.array(model.position_history)
            for i in range(model.n_stimuli):
                plt.plot(positions[:, i], label=f"Stimulus {i}")
                
            plt.xlabel("Trial")
            plt.ylabel("Estimated Position")
            plt.title(f"Estimated Stimulus Positions (Day {day})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"{model_name}_positions_day{day}.png"), dpi=300)
            plt.close()
            
            # Plot uncertainty
            plt.figure(figsize=(12, 8))
            
            uncertainty = np.array(model.uncertainty_history)
            for i in range(model.n_stimuli):
                plt.plot(uncertainty[:, i], label=f"Stimulus {i}")
                
            plt.xlabel("Trial")
            plt.ylabel("Uncertainty")
            plt.title(f"Stimulus Uncertainty (Day {day})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"{model_name}_uncertainty_day{day}.png"), dpi=300)
            plt.close()
    
    # Plot uncertainty across days for all models
    plot_uncertainty_across_days(model_dict, f"{model_name}_uncertainty_across_days", output_dir)

def compute_vte_uncertainty_correlations_with_existing(all_data_df, betasort_vte_df, new_model_day_models, output_dir="model_comparison"):
    """
    Compute correlations between VTE behavior and model uncertainty for each model, 
    using existing Betasort VTE data
    
    Parameters:
        - all_data_df: DataFrame with rat data including VTE
        - betasort_vte_df: Existing VTE-uncertainty DataFrame from Betasort
        - new_model_day_models: Dictionary with model name as key and dict of day models as value
        - output_dir: Directory to save results
    
    Returns:
        - correlation_df: DataFrame with correlation results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make sure the betasort_vte_df has the necessary columns
    required_columns = ["day", "trial_num", "stim1", "stim2", "vte_occurred"]
    for col in required_columns:
        if col not in betasort_vte_df.columns:
            print(f"Warning: Missing column '{col}' in betasort_vte_df")
            return None, None, None
    
    # Create a copy of betasort_vte_df to add new model uncertainties
    combined_df = betasort_vte_df.copy()
    
    # Ensure 'pair' column exists
    if 'pair' not in combined_df.columns:
        combined_df['pair'] = combined_df.apply(lambda row: f"{row['stim1']}-{row['stim2']}", axis=1)
    
    # Rename Betasort uncertainty columns if they exist
    uncertainty_columns = ["stim1_uncertainty", "stim2_uncertainty", 
                         "pair_relational_uncertainty", "pair_roc_uncertainty"]
    
    for col in uncertainty_columns:
        if col in combined_df.columns:
            combined_df[f"Betasort_{col}"] = combined_df[col]
    
    # Add uncertainties from new models
    for model_name, day_models in new_model_day_models.items():
        # Initialize uncertainty columns
        combined_df[f"{model_name}_uncertainty"] = np.nan
        
        # For each day, calculate uncertainties for that day's trials
        for day, model in day_models.items():
            day_mask = combined_df['day'] == day
            
            # Skip if no trials for this day
            if not day_mask.any():
                continue
            
            # For each trial on this day, calculate uncertainty
            day_trials = combined_df[day_mask]
            
            for idx, row in day_trials.iterrows():
                try:
                    uncertainty = model.get_uncertainty(row['stim1'], row['stim2'])
                    combined_df.loc[idx, f"{model_name}_uncertainty"] = uncertainty
                except Exception as e:
                    print(f"Error calculating uncertainty for {model_name}, day {day}, trial {row['trial_num']}: {e}")
    
    # Save combined data
    combined_df.to_csv(os.path.join(output_dir, "vte_uncertainty_data.csv"), index=False)
    
    # Calculate correlations for each model
    correlation_results = []
    
    # First, add Betasort correlations
    for col in uncertainty_columns:
        betasort_col = f"Betasort_{col}"
        if betasort_col in combined_df.columns:
            # Overall correlation
            r, p = pointbiserialr(combined_df["vte_occurred"], combined_df[betasort_col])
            correlation_results.append({
                "model": "Betasort",
                "uncertainty_type": col,
                "scope": "overall",
                "pair": "all",
                "correlation": r,
                "p_value": p,
                "significant": p < 0.05
            })
    
    # Add correlations for new models
    for model_name in new_model_day_models.keys():
        uncertainty_col = f"{model_name}_uncertainty"
        
        if uncertainty_col in combined_df.columns:
            # Overall correlation
            r, p = pointbiserialr(combined_df["vte_occurred"], combined_df[uncertainty_col])
            correlation_results.append({
                "model": model_name,
                "uncertainty_type": "model",
                "scope": "overall",
                "pair": "all",
                "correlation": r,
                "p_value": p,
                "significant": p < 0.05
            })
            
            # Correlation by pair
            for pair, pair_data in combined_df.groupby("pair"):
                if len(pair_data) >= 5 and pair_data["vte_occurred"].nunique() > 1:
                    r, p = pointbiserialr(pair_data["vte_occurred"], pair_data[uncertainty_col])
                    correlation_results.append({
                        "model": model_name,
                        "uncertainty_type": "model",
                        "scope": "pair",
                        "pair": pair,
                        "correlation": r,
                        "p_value": p,
                        "significant": p < 0.05
                    })
    
    # Convert to DataFrame
    correlation_df = pd.DataFrame(correlation_results)
    
    # Save correlations
    correlation_df.to_csv(os.path.join(output_dir, "vte_uncertainty_correlations.csv"), index=False)
    
    # Create summary for plotting (only overall correlations)
    summary_df = correlation_df[correlation_df["scope"] == "overall"]
    
    # Find the best correlation for each model (in case there are multiple uncertainty types)
    best_by_model = []
    for model, model_data in summary_df.groupby("model"):
        best_row = model_data.loc[model_data["correlation"].abs().idxmax()]
        best_by_model.append(best_row)
    
    summary_df = pd.DataFrame(best_by_model)
    summary_df = summary_df.sort_values("correlation", ascending=False)
    
    return correlation_df, summary_df, combined_df

def run_comparison_with_existing_betasort(all_data_df, betasort_results, ti_data, rat, output_dir="model_comparison"):
    """
    Run model comparison using existing Betasort results and new models,
    maintaining model state across days
    
    Parameters:
        - all_data_df: DataFrame with rat choice data
        - betasort_results: Results from previous Betasort analysis
        - ti_data: data for TI in format {'stim1': stim 1, 'stim2': stim 2, 'accuracy': performance}
        - rat: Rat identifier
        - output_dir: Directory to save results and plots
    
    Returns:
        - results_dict: Dictionary with all comparison results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define alternative models to compare
    models_to_compare = [
        ("RW-Generalization", rw_generalization.RWGeneralization, {"alpha": 0.1, "beta": 0.1, "generalization_factor": 0.5}),
        ("Bayesian", bayesian_learner.BayesianLearner, {"prior_mean": 0.5, "prior_var": 1.0}),
        ("ValueTransfer", value_transfer.ValueTransferModel, {"alpha": 0.1, "transfer_rate": 0.3, "temp": 1.0}),
        ("TD-Lambda", temporal_difference_learning.TDLambdaModel, {"alpha": 0.1, "gamma": 0.9, "lambda_": 0.6, "temp": 1.0}),
        ("NeuralNetwork", neural_network.NeuralNetworkModel, {"learning_rate": 0.01, "hidden_size": 10, "dropout_rate": 0.2})
    ]
    
    # Master results dictionary (initialize with Betasort results)
    results_dict = {
        "performance": {
            "Betasort": {
                "mean_match_rate": betasort_results.get("best_performance", 0.5),
                "daily_match_rates": []  # We'll derive this from session data if available
            }
        },
        "ti_results": {
            "Betasort": {}  # We'll reconstruct this from saved TI results
        },
        "day_models": {},
        "vte_correlations": {}
    }
    
    # Extract session performance data if available for Betasort
    if "session_predictions_regression" in betasort_results:
        results_dict["performance"]["Betasort"]["daily_match_rates"] = betasort_results["session_predictions_regression"]
    
    # Parse TI results if available
    if "TI_Result" in betasort_results:
        if isinstance(betasort_results["TI_Result"], str):
            ti_result_dict = json.loads(betasort_results["TI_Result"])
            # Convert string keys back to tuples
            for key_str, value in ti_result_dict.items():
                key_parts = key_str.split(",")
                key_tuple = (int(key_parts[0]), int(key_parts[1]))
                results_dict["ti_results"]["Betasort"][key_tuple] = value
    
    # Get the days in chronological order
    days = sorted(all_data_df['Day'].unique())
    
    # Find the maximum number of stimuli across all days
    max_stimuli = 0
    for day in days:
        day_data = all_data_df[all_data_df['Day'] == day]
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        day_max_stimuli = max(present_stimuli) + 1
        max_stimuli = max(max_stimuli, day_max_stimuli)
    
    # Timer for tracking progress
    total_models = len(models_to_compare)
    
    # Run each alternative model
    for model_idx, (model_name, model_class, params) in enumerate(models_to_compare):
        print(f"Running model {model_idx+1}/{total_models}: {model_name}")
        start_time = time.time()
        
        # Initialize model once with maximum number of stimuli
        model = model_class(n_stimuli=max_stimuli, **params)
        
        # Track day models and performance for this model
        day_models = {}
        performance_results = {
            "daily_match_rates": [],
            "mean_match_rate": 0
        }
        
        # Process each day in chronological order
        for day in days:
            day_data = all_data_df[all_data_df['Day'] == day]
            # Extract data
            chosen_idx = day_data["first"].values
            unchosen_idx = day_data["second"].values
            
            # Process trials
            day_matches = []
            
            for t in range(len(chosen_idx)):
                chosen = chosen_idx[t]
                unchosen = unchosen_idx[t]
                
                # Simulate model choice multiple times
                n_simulations = 100
                model_choices = np.zeros(n_simulations)
                
                for sim in range(n_simulations):
                    model_choice = model.choose([chosen, unchosen])
                    model_choices[sim] = model_choice
                
                # Calculate match rate
                model_match_rate = np.mean(model_choices == chosen)
                day_matches.append(model_match_rate)
                
                # Determine reward
                reward = 1 if chosen < unchosen else 0
                
                # Update model
                model.update(chosen, unchosen, reward)
            
            # Store day's match rate
            day_match_rate = np.mean(day_matches)
            performance_results["daily_match_rates"].append(day_match_rate)
            
            # Store a deep copy of the model state at the end of this day
            day_models[day] = copy.deepcopy(model)  # Create a deep copy of the model
        
        # Calculate overall performance
        performance_results["mean_match_rate"] = np.mean(performance_results["daily_match_rates"])
        
        # Store results for this model
        results_dict["performance"][model_name] = performance_results
        results_dict["day_models"][model_name] = day_models
        
        # Get transitive inference results using the final model
        final_day = max(days)
        final_model = day_models[final_day]
        
        # Use generic function for TI assessment
        ti_results = {}
        n_stimuli = final_model.n_stimuli
        
        for chosen_idx in range(0, n_stimuli-1):
            for other_idx in range(chosen_idx + 1, n_stimuli):
                if chosen_idx == other_idx:
                    continue
                
                n_simulations = 100
                model_choices = np.zeros(n_simulations)
                for sim in range(n_simulations):
                    model_choice = final_model.choose([chosen_idx, other_idx])
                    model_choices[sim] = model_choice
                
                model_match_rate = np.mean(model_choices == chosen_idx)
                ti_results[(chosen_idx, other_idx)] = model_match_rate
        
        results_dict["ti_results"][model_name] = ti_results
        
        # Plot model parameters
        plot_model_parameters(day_models, model_name, os.path.join(output_dir, "parameter_plots"))
        
        # Print timing information
        elapsed_time = time.time() - start_time
        print(f"Completed {model_name} in {elapsed_time:.2f} seconds")
        print(f"Mean match rate: {performance_results['mean_match_rate']:.4f}")
    
    # Save performance results
    overall_df, daily_df = save_model_performance(results_dict["performance"], output_dir=output_dir)
    
    # Plot performance comparison
    plot_model_comparison(overall_df, output_dir=output_dir)
    
    # Compare transitive inference results
    if results_dict["ti_results"]["Betasort"]:
        comparison_df, summary_df = compare_ti_with_actual(
            results_dict["ti_results"],
            ti_data,
            output_dir=output_dir
        )
        
        # Plot TI comparison
        plot_ti_comparison(summary_df, output_dir=output_dir)
    
    # Analyze VTE-uncertainty correlations if pair_vte_df exists
    if "pair_vte_df" in betasort_results and len(betasort_results["pair_vte_df"]) > 0:
        correlation_df, summary_df, combined_df = compute_vte_uncertainty_correlations_with_existing(
            all_data_df,
            betasort_results["pair_vte_df"],
            results_dict["day_models"],
            output_dir=output_dir
        )
        
        # Plot VTE correlations
        plot_vte_uncertainty_correlations(summary_df, output_dir=output_dir)
        
        # Store correlation results
        results_dict["vte_correlations"] = {
            "correlation_df": correlation_df,
            "summary_df": summary_df,
            "combined_df": combined_df
        }
    
    return results_dict

def compare_models_with_betasort(betasort_results_path, all_data_df_path, ti_path, output_dir="model_comparison"):
    """
    Load existing Betasort results and compare with new models
    
    Parameters:
        - betasort_results_path: Path to saved Betasort results CSV
        - all_data_df_path: Path to the original data CSV
        - output_dir: Directory to save comparison results
    
    Returns:
        - results_dict: Dictionary with all comparison results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Betasort results
    betasort_df = pd.read_csv(betasort_results_path)
    
    # Load VTE uncertainty data if it exists
    vte_path = os.path.join(os.path.dirname(betasort_results_path), "vte_uncertainty.csv")
    pair_vte_df = pd.read_csv(vte_path) if os.path.exists(vte_path) else None
    
    # Load uncertainty VTE correlation if it exists
    uncertainty_vte_path = os.path.join(os.path.dirname(betasort_results_path), "uncertainty_vte.json")
    uncertainty_vte_results = None
    if os.path.exists(uncertainty_vte_path):
        with open(uncertainty_vte_path, 'r') as f:
            uncertainty_vte_results = json.load(f)
    
    # Extract betasort results (assuming only one row in the DataFrame)
    betasort_results = betasort_df.iloc[0].to_dict()
    
    # Add VTE data to results
    if pair_vte_df is not None:
        betasort_results["pair_vte_df"] = pair_vte_df
    
    # Add uncertainty VTE correlation
    if uncertainty_vte_results is not None:
        betasort_results["uncertainty_vte_results"] = uncertainty_vte_results
    
    # Load original data
    all_data_df = pd.read_csv(all_data_df_path)
    ti_data = pd.read_csv(ti_path)
    
    # Extract rat ID from results or filename
    rat = betasort_results.get("rat", os.path.basename(os.path.dirname(betasort_results_path)))
    
    # Run comparison
    results = run_comparison_with_existing_betasort(all_data_df, betasort_results, ti_data, rat, output_dir)
    
    return results

def run_full_analysis():
    # Base paths
    data_path = os.path.join(helper.BASE_PATH, "processed_data", "data_for_model")
    betasort_results_path = os.path.join(helper.BASE_PATH, "processed_data", "new_model_data")
    comparison_output_path = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison")
    
    # Define which rats to analyze
    rats_to_analyze = [rat for rat in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, rat)) 
                      and "BP06" not in rat and "BP08" not in rat]
    
    # Start timing
    start_time = time.time()
    
    """
    # Process each rat
    for rat in rats_to_analyze:
        print(f"\nProcessing rat: {rat}")
        
        # Find the relevant data files
        rat_data_path = os.path.join(data_path, rat)
        betasort_rat_path = os.path.join(betasort_results_path, rat)
        
        if not os.path.exists(betasort_rat_path):
            print(f"No Betasort results found for {rat}")
            continue
        
        # Find all CSV files with all_days in the name
        data_files = []
        for root, _, files in os.walk(rat_data_path):
            for file in files:
                if file.endswith(".csv") and "all_days" in file and ".DS_Store" not in file and "zIdPhi" not in file:
                    data_files.append(os.path.join(root, file))
        
        # Check if betasort results exist
        betasort_results_file = os.path.join(betasort_rat_path, "results.csv")
        if not os.path.exists(betasort_results_file):
            print(f"No Betasort results found for {rat}")
            continue
        
        # get ti results from real rats
        ti_path = os.path.join(rat_data_path, "ti_results.csv")
        
        # Find the corresponding data file
        if len(data_files) == 0:
            print(f"No data files found for {rat}")
            continue
        
        # Use the first data file found (or ideally match based on some criteria)
        data_file = data_files[0]
        
        # Create output directory for this rat
        rat_output_dir = os.path.join(comparison_output_path, rat)
        os.makedirs(rat_output_dir, exist_ok=True)
        
        # Run the comparison
        rat_start_time = time.time()
        print(f"Comparing models for {rat} using data from {os.path.basename(data_file)}")
        
        try:
            results = compare_models_with_betasort(betasort_results_file, data_file, ti_path, rat_output_dir)
            
            # Print summary of results
            models = list(results["performance"].keys())
            models_sorted = sorted(models, key=lambda x: results["performance"][x]["mean_match_rate"], reverse=True)
            
            print("\nPerformance Summary:")
            for model in models_sorted:
                mean_rate = results["performance"][model]["mean_match_rate"]
                print(f"{model}: {mean_rate:.4f}")
                
            rat_elapsed_time = time.time() - rat_start_time
            print(f"Completed analysis for {rat} in {rat_elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"Error analyzing {rat}: {e}")
    """
    
    # Combine results across rats
    print("\nAggregating results across all rats...")
    combined_performance = combine_results_across_rats(comparison_output_path)
    combined_vte = aggregate_vte_correlations(comparison_output_path)
    
    # Create summary visualizations
    if combined_performance is not None:
        plot_combined_results(combined_performance, comparison_output_path)
        print("Created combined performance visualizations")
    
    if combined_vte is not None:
        plot_combined_vte_correlations(combined_vte, comparison_output_path)
        print("Created combined VTE correlation visualizations")
    
    # Print total time
    total_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {comparison_output_path}")

def optimize_model_parameters(model_class, all_data_df, parameter_bounds, max_iter=10, popsize=15, 
                             strategy='best1bin', n_simulations=100, verbose=True):
    """
    Optimize model parameters using differential evolution
    
    Parameters:
        - model_class: The model class to optimize
        - all_data_df: DataFrame with rat data
        - parameter_bounds: Dictionary mapping parameter names to (min, max) tuples
        - max_iter: Maximum number of generations
        - popsize: Population size for DE
        - strategy: DE strategy
        - n_simulations: Number of simulations per trial for model evaluation
        - verbose: Whether to print progress information
        
    Returns:
        - optimal_params: Dictionary of optimized parameters
        - best_score: Best score (negative match rate) achieved
    """
    # Extract parameter names and bounds in the order expected by the model
    param_names = list(parameter_bounds.keys())
    bounds = [parameter_bounds[param] for param in param_names]
    
    # Find the maximum number of stimuli across all days
    max_stimuli = 0
    for day, day_data in all_data_df.groupby('Day'):
        chosen_idx = day_data["first"].values
        unchosen_idx = day_data["second"].values
        present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
        day_max_stimuli = max(present_stimuli) + 1
        max_stimuli = max(max_stimuli, day_max_stimuli)
    
    # Define objective function for minimization (negative match rate)
    def objective_function(params):
        # Convert parameters to dictionary for model initialization
        param_dict = {name: params[i] for i, name in enumerate(param_names)}
        
        # Initialize model with these parameters
        model = model_class(n_stimuli=max_stimuli, **param_dict)
        
        # Track performance across all days
        all_matches = []
        
        # Process each day
        for day, day_data in all_data_df.groupby('Day'):
            # Extract data
            chosen_idx = day_data["first"].values
            unchosen_idx = day_data["second"].values
            
            # Process trials
            day_matches = []
            
            for t in range(len(chosen_idx)):
                chosen = chosen_idx[t]
                unchosen = unchosen_idx[t]
                
                # Simulate model choice
                model_choices = np.zeros(n_simulations)
                for sim in range(n_simulations):
                    model_choice = model.choose([chosen, unchosen])
                    model_choices[sim] = model_choice
                
                # Calculate match rate
                model_match_rate = np.mean(model_choices == chosen)
                day_matches.append(model_match_rate)
                
                # Determine reward
                reward = 1 if chosen < unchosen else 0
                
                # Update model
                model.update(chosen, unchosen, reward)
            
            # Add day's matches to overall matches
            all_matches.extend(day_matches)
        
        # Calculate overall match rate
        mean_match_rate = np.mean(all_matches)
        
        # Return negative match rate (for minimization)
        return -mean_match_rate
    
    # Run differential evolution
    if verbose:
        print(f"Starting optimization for {model_class.__name__}")
        print(f"Parameter space: {parameter_bounds}")
    
    start_time = time.time()
    
    # Define callback function for progress tracking
    current_best = [float('inf')]  # Use list to allow modification inside callback
    
    def callback(xk, convergence):
        current_score = objective_function(xk)
        if current_score < current_best[0]:
            current_best[0] = current_score
            if verbose:
                param_dict = {name: xk[i] for i, name in enumerate(param_names)}
                print(f"New best: {-current_score:.4f} match rate with parameters: {param_dict}")
        return False
    
    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        maxiter=max_iter,
        popsize=popsize,
        strategy=strategy,
        callback=callback if verbose else None,
        disp=verbose
    )
    
    # Convert optimized parameters to dictionary
    optimal_params = {name: result.x[i] for i, name in enumerate(param_names)}
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best match rate: {-result.fun:.4f}")
        print(f"Optimal parameters: {optimal_params}")
    
    return optimal_params, -result.fun

def define_parameter_bounds():
    """
    Define parameter bounds for each model type
    
    Returns:
        - bounds_dict: Dictionary mapping model class to parameter bounds dict
    """
    bounds_dict = {
        rw_generalization.RWGeneralization: {
            "alpha": (0.01, 0.5),          # Learning rate
            "beta": (0.01, 5.0),           # Temperature parameter
            "generalization_factor": (0.0, 1.0)  # Generalization factor
        },
        
        bayesian_learner.BayesianLearner: {
            "prior_mean": (0.0, 1.0),      # Prior mean
            "prior_var": (0.1, 5.0)        # Prior variance
        },
        
        value_transfer.ValueTransferModel: {
            "alpha": (0.01, 0.5),          # Learning rate
            "transfer_rate": (0.0, 1.0),   # Transfer rate
            "temp": (0.1, 5.0)             # Temperature parameter
        },
        
        temporal_difference_learning.TDLambdaModel: {
            "alpha": (0.01, 0.5),          # Learning rate
            "gamma": (0.5, 0.99),          # Discount factor
            "lambda_": (0.0, 1.0),         # Eligibility trace decay
            "temp": (0.1, 5.0)             # Temperature parameter
        },
        
        neural_network.NeuralNetworkModel: {
            "learning_rate": (0.001, 0.1),  # Learning rate
            "hidden_size": (5, 30),         # Hidden layer size (integers)
            "dropout_rate": (0.0, 0.5)      # Dropout rate
        }
    }
    
    return bounds_dict

def run_optimized_comparison(all_data_df, betasort_results, ti_data, rat, output_dir="model_comparison", 
                           optimization_iterations=10):
    """
    Run model comparison with optimized parameters
    
    Parameters:
        - all_data_df: DataFrame with rat choice data
        - betasort_results: Results from previous Betasort analysis
        - ti_data: data for TI in format {'stim1': stim 1, 'stim2': stim 2, 'accuracy': performance}
        - rat: Rat identifier
        - output_dir: Directory to save results and plots
        - optimization_iterations: Number of DE iterations for optimization
    
    Returns:
        - results_dict: Dictionary with all comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    models_to_optimize = [
        #("RW-Generalization", rw_generalization.RWGeneralization),
        ("Bayesian", bayesian_learner.BayesianLearner),
        ("ValueTransfer", value_transfer.ValueTransferModel),
        ("TD-Lambda", temporal_difference_learning.TDLambdaModel),
        ("NeuralNetwork", neural_network.NeuralNetworkModel)
    ]
    
    # Get parameter bounds for each model
    parameter_bounds = define_parameter_bounds()
    
    # Master results dictionary (initialize with Betasort results)
    results_dict = {
        "performance": {
            "Betasort": {
                "mean_match_rate": betasort_results.get("best_performance", 0.5),
                "daily_match_rates": []  # We'll derive this from session data if available
            }
        },
        "ti_results": {
            "Betasort": {}  # We'll reconstruct this from saved TI results
        },
        "day_models": {},
        "vte_correlations": {},
        "optimized_parameters": {}  # Store optimized parameters
    }
    
    # Extract session performance data if available for Betasort
    if "session_predictions_regression" in betasort_results:
        results_dict["performance"]["Betasort"]["daily_match_rates"] = betasort_results["session_predictions_regression"]
    
    # Parse TI results if available
    if "TI_Result" in betasort_results:
        if isinstance(betasort_results["TI_Result"], str):
            ti_result_dict = json.loads(betasort_results["TI_Result"])
            # Convert string keys back to tuples
            for key_str, value in ti_result_dict.items():
                key_parts = key_str.split(",")
                key_tuple = (int(key_parts[0]), int(key_parts[1]))
                results_dict["ti_results"]["Betasort"][key_tuple] = value
    
    # Create optimization directory for storing parameter optimization results
    optimization_dir = os.path.join(output_dir, "optimization_results")
    os.makedirs(optimization_dir, exist_ok=True)
    
    # Run optimization and evaluation for each model
    total_models = len(models_to_optimize)
    
    for model_idx, (model_name, model_class) in enumerate(models_to_optimize):
        print(f"\nModel {model_idx+1}/{total_models}: Optimizing {model_name}")
        
        # Check if we have cached optimization results
        cache_file = os.path.join(optimization_dir, f"{model_name}_params.json")
        
        if os.path.exists(cache_file):
            # Load cached parameters
            print(f"Loading cached parameters for {model_name}")
            with open(cache_file, 'r') as f:
                optimal_params = json.load(f)
            best_score = None  # We don't have this information from the cache
        else:
            # Optimize parameters
            bounds = parameter_bounds[model_class]
            optimal_params, best_score = optimize_model_parameters(
                model_class,
                all_data_df,
                bounds,
                max_iter=optimization_iterations,
                popsize=15,
                strategy='best1bin',
                verbose=True
            )
            
            # Save optimized parameters
            with open(cache_file, 'w') as f:
                json.dump(optimal_params, f, indent=2)
        
        # Store optimized parameters
        results_dict["optimized_parameters"][model_name] = optimal_params
        
        # Now evaluate the model with optimized parameters
        print(f"Evaluating {model_name} with optimized parameters")
        start_time = time.time()
        
        # Special handling for NeuralNetwork - convert hidden_size to int
        if model_class == neural_network.NeuralNetworkModel:
            optimal_params["hidden_size"] = int(optimal_params["hidden_size"])
        
        # Initialize model with optimized parameters
        model = model_class(n_stimuli=max_stimuli, **optimal_params)
        
        # Use your existing evaluation code here...
        # [The rest of the code follows your original pattern but uses the optimized models]
        
        # Track day models and performance
        day_models = {}
        performance_results = {
            "daily_match_rates": [],
            "mean_match_rate": 0
        }
        
        # Process each day in chronological order
        for day in sorted(all_data_df['Day'].unique()):
            day_data = all_data_df[all_data_df['Day'] == day]
            chosen_idx = day_data["first"].values
            unchosen_idx = day_data["second"].values
            
            # Process trials
            day_matches = []
            
            for t in range(len(chosen_idx)):
                chosen = chosen_idx[t]
                unchosen = unchosen_idx[t]
                
                # Simulate model choice multiple times
                n_simulations = 100
                model_choices = np.zeros(n_simulations)
                
                for sim in range(n_simulations):
                    model_choice = model.choose([chosen, unchosen])
                    model_choices[sim] = model_choice
                
                # Calculate match rate
                model_match_rate = np.mean(model_choices == chosen)
                day_matches.append(model_match_rate)
                
                # Determine reward
                reward = 1 if chosen < unchosen else 0
                
                # Update model
                model.update(chosen, unchosen, reward)
            
            # Store day's match rate
            day_match_rate = np.mean(day_matches)
            performance_results["daily_match_rates"].append(day_match_rate)
            
            # Store a deep copy of the model state at the end of this day
            day_models[day] = copy.deepcopy(model)
        
        # Calculate overall performance
        performance_results["mean_match_rate"] = np.mean(performance_results["daily_match_rates"])
        
        # Store results for this model
        results_dict["performance"][model_name] = performance_results
        results_dict["day_models"][model_name] = day_models
        
        # Get transitive inference results using the final model
        final_day = max(all_data_df['Day'])
        final_model = day_models[final_day]
        
        # Use generic function for TI assessment
        ti_results = {}
        n_stimuli = final_model.n_stimuli
        
        # Call your existing plotting and evaluation functions here
        
        # Print timing information
        elapsed_time = time.time() - start_time
        print(f"Completed {model_name} evaluation in {elapsed_time:.2f} seconds")
        print(f"Mean match rate: {performance_results['mean_match_rate']:.4f}")
    
    # Call your existing evaluation functions with the optimized models
    
    return results_dict

def run_optimized_analysis(optimization_iterations=10):
    """
    Run full analysis with parameter optimization
    
    Parameters:
        - optimization_iterations: Number of iterations for DE optimization
    """
    # Base paths
    data_path = os.path.join(helper.BASE_PATH, "processed_data", "data_for_model")
    betasort_results_path = os.path.join(helper.BASE_PATH, "processed_data", "new_model_data")
    comparison_output_path = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison_optimized")
    
    # Define which rats to analyze
    rats_to_analyze = [rat for rat in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, rat)) 
                      and "BP06" not in rat and "BP08" not in rat]
    
    # Start timing
    start_time = time.time()
    
    # Process each rat
    for rat in rats_to_analyze:
        print(f"\nProcessing rat: {rat}")
        
        # Find the relevant data files
        rat_data_path = os.path.join(data_path, rat)
        betasort_rat_path = os.path.join(betasort_results_path, rat)
        
        if not os.path.exists(betasort_rat_path):
            print(f"No Betasort results found for {rat}")
            continue
        
        # Find all CSV files with all_days in the name
        data_files = []
        for root, _, files in os.walk(rat_data_path):
            for file in files:
                if file.endswith(".csv") and "all_days" in file and ".DS_Store" not in file and "zIdPhi" not in file:
                    data_files.append(os.path.join(root, file))
        
        # Check if betasort results exist
        betasort_results_file = os.path.join(betasort_rat_path, "results.csv")
        if not os.path.exists(betasort_results_file):
            print(f"No Betasort results found for {rat}")
            continue
        
        # get ti results from real rats
        ti_path = os.path.join(rat_data_path, "ti_results.csv")
        
        # Find the corresponding data file
        if len(data_files) == 0:
            print(f"No data files found for {rat}")
            continue
        
        # Use the first data file found
        data_file = data_files[0]
        
        # Create output directory for this rat
        rat_output_dir = os.path.join(comparison_output_path, rat)
        
        # Run comparison with optimized parameters
        try:
            # Load data and betasort results
            betasort_df = pd.read_csv(betasort_results_file)
            all_data_df = pd.read_csv(data_file)
            ti_data = pd.read_csv(ti_path)
            
            # Extract betasort results
            betasort_results = betasort_df.iloc[0].to_dict()
            
            # Add VTE data if available
            vte_path = os.path.join(os.path.dirname(betasort_results_file), "vte_uncertainty.csv")
            if os.path.exists(vte_path):
                betasort_results["pair_vte_df"] = pd.read_csv(vte_path)
            
            # Run optimized comparison
            results = run_optimized_comparison(
                all_data_df, 
                betasort_results, 
                ti_data, 
                rat,
                rat_output_dir,
                optimization_iterations=optimization_iterations
            )
            
            # Print summary
            models = list(results["performance"].keys())
            models_sorted = sorted(models, key=lambda x: results["performance"][x]["mean_match_rate"], reverse=True)
            
            print("\nPerformance Summary:")
            for model in models_sorted:
                mean_rate = results["performance"][model]["mean_match_rate"]
                print(f"{model}: {mean_rate:.4f}")
                
        except Exception as e:
            print(f"Error analyzing {rat}: {e}")
    
    # Combine results across rats
    print("\nAggregating results across all rats...")
    combined_performance = combine_results_across_rats(comparison_output_path)
    
    if combined_performance is not None:
        plot_combined_results(combined_performance, comparison_output_path)
        print("Created combined performance visualizations")
    
    # Print total time
    total_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    run_full_analysis()
    #run_optimized_analysis()