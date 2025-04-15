import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pointbiserialr

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
        yerr=sorted_df["std_error"],
        capsize=0.2,
        palette="viridis"
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
    
    # Set style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create bar plot with error bars
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
        yerr=summary_df.get("correlation_sem", 0.05),  # Use SEM if available
        capsize=0.2,
        palette="viridis"
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
    """
    Plot uncertainty for each stimulus pair across days
    
    Parameters:
        - model_dict: Dictionary with day as key and model as value
        - filename: Base filename for the plot
        - output_dir: Directory to save plots
    """
    
    # Get all days and stimulus pairs
    days = sorted(model_dict.keys())
    
    # Get number of stimuli from first model
    first_model = model_dict[days[0]]
    n_stimuli = first_model.n_stimuli
    
    # Create all possible pairs
    pairs = []
    for i in range(n_stimuli-1):
        for j in range(i+1, n_stimuli):
            pairs.append((i, j))
    
    # Initialize uncertainty data
    uncertainty_data = np.zeros((len(days), len(pairs)))
    
    # Collect uncertainty across days for each pair
    for day_idx, day in enumerate(days):
        model = model_dict[day]
        
        for pair_idx, (stim1, stim2) in enumerate(pairs):
            uncertainty = model.get_uncertainty(stim1, stim2)
            uncertainty_data[day_idx, pair_idx] = uncertainty
    
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

def run_comparison_with_existing_betasort(all_data_df, betasort_results, rat, output_dir="model_comparison"):
    """
    Run model comparison using existing Betasort results and new models
    
    Parameters:
        - all_data_df: DataFrame with rat choice data
        - betasort_results: Results from previous Betasort analysis
        - rat: Rat identifier
        - output_dir: Directory to save results and plots
    
    Returns:
        - results_dict: Dictionary with all comparison results
    """
    import os
    import numpy as np
    import pandas as pd
    import time
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define alternative models to compare
    models_to_compare = [
        ("RW-Generalization", RWGeneralization, {"alpha": 0.1, "beta": 0.1, "generalization_factor": 0.5}),
        ("Bayesian", BayesianLearner, {"prior_mean": 0.5, "prior_var": 1.0}),
        ("ValueTransfer", ValueTransferModel, {"alpha": 0.1, "transfer_rate": 0.3, "temp": 1.0}),
        ("TD-Lambda", TDLambdaModel, {"alpha": 0.1, "gamma": 0.9, "lambda_": 0.6, "temp": 1.0}),
        ("Bayesian-TD", BayesianTDModel, {"alpha": 0.1, "uncertainty_weight": 0.5, "prior_var": 1.0}),
        ("NeuralNetwork", NeuralNetworkModel, {"learning_rate": 0.01, "hidden_size": 10, "dropout_rate": 0.2})
    ]
    
    # Master results dictionary (initialize with Betasort results)
    results_dict = {
        "performance": {
            "Betasort": {
                "mean_match_rate": betasort_results["best_performance"],
                "daily_match_rates": []  # We'll derive this from session data if available
            }
        },
        "ti_results": {
            "Betasort": {}  # We'll reconstruct this from saved TI results
        },
        "day_models": {
            "Betasort": betasort_results["all_models"]
        },
        "vte_correlations": {}
    }
    
    # Extract session performance data if available for Betasort
    if "session_predictions_regression" in betasort_results:
        results_dict["performance"]["Betasort"]["daily_match_rates"] = betasort_results["session_predictions_regression"]
    
    # Parse TI results if available
    if "TI_Result" in betasort_results:
        import json
        if isinstance(betasort_results["TI_Result"], str):
            ti_result_dict = json.loads(betasort_results["TI_Result"])
            # Convert string keys back to tuples
            for key_str, value in ti_result_dict.items():
                key_parts = key_str.split(",")
                key_tuple = (int(key_parts[0]), int(key_parts[1]))
                results_dict["ti_results"]["Betasort"][key_tuple] = value
    
    # Timer for tracking progress
    total_models = len(models_to_compare)
    
    # Run each alternative model
    for model_idx, (model_name, model_class, params) in enumerate(models_to_compare):
        print(f"Running model {model_idx+1}/{total_models}: {model_name}")
        start_time = time.time()
        
        # Track day models and performance for this model
        day_models = {}
        performance_results = {
            "daily_match_rates": [],
            "mean_match_rate": 0
        }
        
        # Process each day
        for day, day_data in all_data_df.groupby('Day'):
            # Extract data
            chosen_idx = day_data["first"].values
            unchosen_idx = day_data["second"].values
            
            # Identify stimuli present
            present_stimuli = set(np.concatenate([chosen_idx, unchosen_idx]))
            n_stimuli = max(present_stimuli) + 1
            
            # Initialize model
            model = model_class(n_stimuli=n_stimuli, **params)
            
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
            
            # Store day's match rate and model
            day_match_rate = np.mean(day_matches)
            performance_results["daily_match_rates"].append(day_match_rate)
            day_models[day] = model
        
        # Calculate overall performance
        performance_results["mean_match_rate"] = np.mean(performance_results["daily_match_rates"])
        
        # Store results for this model
        results_dict["performance"][model_name] = performance_results
        results_dict["day_models"][model_name] = day_models
        
        # Get transitive inference results
        final_day = max(day_models.keys())
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
    if "ti_df" in betasort_results:
        actual_ti_df = betasort_results["ti_df"]
        comparison_df, summary_df = compare_ti_with_actual(
            results_dict["ti_results"],
            actual_ti_df,
            output_dir=output_dir
        )
        
        # Plot TI comparison
        plot_ti_comparison(summary_df, output_dir=output_dir)
    
    # Analyze VTE-uncertainty correlations if VTE data exists
    if "pair_vte_df" in betasort_results and len(betasort_results["pair_vte_df"]) > 0:
        # Extract VTE column from the original data
        vte_data = all_data_df['VTE'] if 'VTE' in all_data_df.columns else None
        
        if vte_data is not None:
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

# Define models to compare
models_to_compare = [
    #("RW-Generalization", rw_generalization.RWGeneralization, {"alpha": 0.1, "beta": 0.1, "generalization_factor": 0.5}),
    #("Bayesian", bayesian_learner.BayesianLearner, {"prior_mean": 0.5, "prior_var": 1.0}),
    ("ValueTransfer", value_transfer.ValueTransferModel, {"alpha": 0.1, "transfer_rate": 0.3, "temp": 1.0}),
    ("TD-Lambda", temporal_difference_learning.TDLambdaModel, {"alpha": 0.1, "gamma": 0.9, "lambda_": 0.6, "temp": 1.0}),
    ("NeuralNetwork", neural_network.NeuralNetworkModel, {"learning_rate": 0.01, "hidden_size": 10, "dropout_rate": 0.2})
]

# Run comparison
results = {}


for rat in os.listdir(data_path):
    rat_path = os.path.join(data_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if ".DS_Store" in file or "zIdPhi" in file or "all_days" not in file:
                continue

            file_path = os.path.join(root, file)
            file_csv = pd.read_csv(file_path)
            
            try:
                # analyze the data sequentially
                for model_name, model_class, params in models_to_compare:
                    print(f"Testing {model_name}...")
                    
                    _, _, match_rates = compare_model_to_data(file_csv, rat, model_class, params)
                    
                    results[model_name] = {
                        "mean_match_rate": np.mean(match_rates),
                        "daily_match_rates": match_rates
                    }
            except Exception as e:
                print(rat, file_path)
                print(e)
                continue

