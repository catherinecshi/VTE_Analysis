import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src import helper
from src import statistics

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
    
    # Create bar plot without error bars first
    bar_plot = sns.barplot(
        x='model', 
        y='mean_rate', 
        data=model_stats,
        palette="viridis"
    )

    # Add error bars manually
    for i, row in model_stats.iterrows():
        bar_plot.errorbar(
            x=i, 
            y=row['mean_rate'],
            yerr=row['sem_rate'],
            color='black',
            capsize=5,
            fmt='none'  # This prevents adding markers
        )
    
    # Add chance level line
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot
    plt.title("Model Comparison Across All Rats", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Mean Match Rate", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.4, 1.0)  # Adjust as needed
    
    # Add value labels on bars
    for i, bar in enumerate(bar_plot.patches):
        bar_value = model_stats.iloc[i]["mean_rate"]
        bar_plot.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{bar_value:.3f}",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "combined_performance.png"), dpi=300)
    plt.close()
    
    # Create boxplot to show distribution across rats
    plt.figure(figsize=(12, 8))
    
    boxplot = sns.boxplot(
        x='model', 
        y='mean_match_rate', 
        data=combined_df,
        palette="viridis",
        order=model_stats['model']  # Use same order as bar plot
    )
    
    # Add individual points
    sns.stripplot(
        x='model', 
        y='mean_match_rate', 
        data=combined_df,
        color='black',
        alpha=0.5,
        jitter=True,
        order=model_stats['model']  # Use same order as bar plot
    )
    
    # Add chance level line
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot
    plt.title("Model Performance Distribution Across Rats", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Match Rate", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "performance_distribution.png"), dpi=300)
    plt.close()
    
    # Create heatmap showing performance by model and rat
    pivot_df = combined_df.pivot(index='rat', columns='model', values='mean_match_rate')
    
    plt.figure(figsize=(14, 10))
    
    # Sort models by average performance
    model_order = model_stats['model'].tolist()
    pivot_df = pivot_df[model_order]
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={'label': 'Match Rate'}
    )
    
    plt.title("Performance by Model and Rat", fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "performance_heatmap.png"), dpi=300)
    plt.close()
    
    return model_stats

def run_statistical_tests(combined_df, output_path):
    """
    Run statistical tests:
    1. 1-sample t-test against chance level (0.5) for each model
    2. Paired t-tests comparing Betasort to each other model
    
    Parameters:
        - combined_df: DataFrame with combined results
        - output_path: Directory to save test results
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize results dictionary
    stat_results = {
        'chance_level_tests': [],
        'betasort_comparison_tests': []
    }
    
    # Get all unique models
    models = combined_df['model'].unique()
    
    # 1. Run 1-sample t-tests against chance level (0.5)
    print("\n=== T-Tests Against Chance Level (0.5) ===")
    for model in models:
        model_data = combined_df[combined_df['model'] == model]['mean_match_rate'].values
        t_stat, p_value = stats.ttest_1samp(model_data, 0.5)
        significant = p_value < 0.05
        
        result = {
            'model': model,
            'mean': np.mean(model_data),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'n_samples': len(model_data)
        }
        
        stat_results['chance_level_tests'].append(result)
        
        print(f"Model: {model}")
        print(f"  Mean: {np.mean(model_data):.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant vs. chance (0.5): {'Yes' if significant else 'No'}")
        print(f"  n = {len(model_data)}")
        print()
    
    # Convert to DataFrame and save
    chance_df = pd.DataFrame(stat_results['chance_level_tests'])
    chance_df.to_csv(os.path.join(output_path, "chance_level_tests.csv"), index=False)
    
    # 2. Run paired t-tests comparing Betasort to each other model
    print("\n=== Paired T-Tests: Betasort vs. Other Models ===")
    
    # Check if 'Betasort' is one of the models
    if 'Betasort' not in models:
        print("Error: Betasort model not found in data")
        return stat_results
    
    # Get unique rats
    rats = combined_df['rat'].unique()
    
    # Create a pivot table with rats as rows and models as columns
    pivot_df = combined_df.pivot(index='rat', columns='model', values='mean_match_rate')
    
    # Run paired t-tests
    for model in models:
        if model == 'Betasort':
            continue
        
        # Get paired data (removing any NaN values)
        valid_pairs = ~(pd.isna(pivot_df['Betasort']) | pd.isna(pivot_df[model]))
        betasort_data = pivot_df.loc[valid_pairs, 'Betasort'].values
        model_data = pivot_df.loc[valid_pairs, model].values
        
        # Run paired t-test
        t_stat, p_value = stats.ttest_rel(betasort_data, model_data)
        significant = p_value < 0.05
        
        result = {
            'model': model,
            'betasort_mean': np.mean(betasort_data),
            'model_mean': np.mean(model_data),
            'mean_diff': np.mean(betasort_data - model_data),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'n_samples': len(betasort_data)
        }
        
        stat_results['betasort_comparison_tests'].append(result)
        
        print(f"Betasort vs. {model}")
        print(f"  Betasort Mean: {np.mean(betasort_data):.4f}")
        print(f"  {model} Mean: {np.mean(model_data):.4f}")
        print(f"  Mean Difference: {np.mean(betasort_data - model_data):.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if significant else 'No'}")
        print(f"  n = {len(betasort_data)}")
        print()
    
    # Convert to DataFrame and save
    comparison_df = pd.DataFrame(stat_results['betasort_comparison_tests'])
    comparison_df.to_csv(os.path.join(output_path, "betasort_comparison_tests.csv"), index=False)
    
    # Create a bar plot of p-values for Betasort comparison
    plt.figure(figsize=(10, 6))
    comparison_df = comparison_df.sort_values('p_value')
    bars = plt.bar(comparison_df['model'], comparison_df['p_value'], color='skyblue')
    
    # Add significance threshold line
    plt.axhline(y=0.05, color='r', linestyle='--', label='p = 0.05')
    
    # Highlight significant results
    for i, bar in enumerate(bars):
        if comparison_df.iloc[i]['significant']:
            bar.set_color('green')
    
    plt.title('P-values: Betasort vs. Other Models')
    plt.xlabel('Model')
    plt.ylabel('p-value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(output_path, "betasort_pvalues.png"), dpi=300)
    plt.close()
    
    return stat_results

def model_anova_with_posthoc(combined_df, output_path=None):
    """
    Perform one-way ANOVA comparing all models, followed by post-hoc pairwise
    comparisons with Bonferroni correction.
    
    Parameters:
        - combined_df: DataFrame with columns 'model' and 'mean_match_rate'
        - output_path: Optional directory to save results
    
    Returns:
        - dict containing ANOVA results and post-hoc test results
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create output directory if specified
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    # Get all unique models
    models = combined_df['model'].unique()
    n_models = len(models)
    
    # Extract data for each model
    model_data = {}
    for model in models:
        model_data[model] = combined_df[combined_df['model'] == model]['mean_match_rate'].values
    
    # Prepare data for ANOVA (list of arrays)
    anova_data = [model_data[model] for model in models]
    
    # calculate observations for reporting
    obs_per_group = [len(data) for data in anova_data]
    total_obs = sum(obs_per_group)
    df_between = n_models - 1
    df_within = total_obs - n_models
    
    # Run one-way ANOVA
    f_stat, p_value = stats.f_oneway(*anova_data)
    
    anova_results = {
        "f_stat": f_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
    
    print("\n=== One-way ANOVA Results ===")
    print(f"F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_value:.8f}")
    print(f"Total observations: {total_obs}")
    print("Observations per group:")
    for i, model in enumerate(models):
        print(f"  {model}: {obs_per_group[i]} observations")
    print(f"Significant difference across models: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Initialize post-hoc results
    posthoc_results = []
    
    # If ANOVA significant, perform post-hoc tests
    if p_value < 0.05:
        print("\n=== Post-hoc Pairwise Comparisons (Bonferroni corrected) ===")
        
        # Number of comparisons for Bonferroni correction
        n_comparisons = (n_models * (n_models - 1)) // 2
        alpha_bonferroni = 0.05 / n_comparisons
        
        print(f"Bonferroni-corrected alpha: {alpha_bonferroni:.8f} (for {n_comparisons} comparisons)")
        
        # Perform all pairwise comparisons
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Avoid duplicate comparisons
                    # Get data for both models
                    data1 = model_data[model1]
                    data2 = model_data[model2]
                    
                    # Run paired t-test if same number of samples (e.g., same rats)
                    # Otherwise run independent t-test
                    if len(data1) == len(data2) and combined_df['rat'].nunique() == len(data1):
                        # Create pivot table
                        pivot_df = combined_df.pivot(index='rat', columns='model', values='mean_match_rate')
                        # Get paired data (removing any NaN values)
                        valid_pairs = ~(pd.isna(pivot_df[model1]) | pd.isna(pivot_df[model2]))
                        data1_paired = pivot_df.loc[valid_pairs, model1].values
                        data2_paired = pivot_df.loc[valid_pairs, model2].values
                        
                        t_stat, p_value = stats.ttest_rel(data1_paired, data2_paired)
                        test_type = "paired"
                        n_samples = len(data1_paired)
                    else:
                        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                        test_type = "independent"
                        n_samples = min(len(data1), len(data2))
                    
                    # Apply Bonferroni correction
                    significant = p_value < alpha_bonferroni
                    
                    # Store results
                    result = {
                        'model1': model1,
                        'model2': model2,
                        'model1_mean': np.mean(data1),
                        'model2_mean': np.mean(data2),
                        'mean_diff': np.mean(data1) - np.mean(data2),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'bonferroni_significant': significant,
                        'uncorrected_significant': p_value < 0.05,
                        'test_type': test_type,
                        'n_samples': n_samples
                    }
                    
                    posthoc_results.append(result)
                    
                    print(f"{model1} vs. {model2}:")
                    print(f"  {model1} Mean: {np.mean(data1):.4f}")
                    print(f"  {model2} Mean: {np.mean(data2):.4f}")
                    print(f"  Mean Difference: {np.mean(data1) - np.mean(data2):.4f}")
                    print(f"  t-statistic: {t_stat:.4f}")
                    print(f"  p-value: {p_value:.8f}")
                    print(f"  Significant after Bonferroni correction: {'Yes' if significant else 'No'}")
                    print(f"  Test type: {test_type}, n = {n_samples}")
                    print()
    
    # Convert post-hoc results to DataFrame
    if posthoc_results:
        posthoc_df = pd.DataFrame(posthoc_results)
        
        if output_path:
            # Save results to CSV
            posthoc_df.to_csv(os.path.join(output_path, "posthoc_comparisons.csv"), index=False)
            
            # Create visualization of p-values
            plt.figure(figsize=(12, 8))
            
            # Sort by p-value
            posthoc_df_sorted = posthoc_df.sort_values('p_value')
            
            # Create comparison labels
            comparison_labels = [f"{row['model1']} vs. {row['model2']}" for _, row in posthoc_df_sorted.iterrows()]
            
            # Create bar plot of p-values
            bars = plt.bar(comparison_labels, posthoc_df_sorted['p_value'], color='skyblue')
            
            # Add uncorrected alpha threshold
            plt.axhline(y=0.05, color='orange', linestyle='--', label='α = 0.05')
            
            # Add Bonferroni-corrected alpha threshold
            plt.axhline(y=alpha_bonferroni, color='r', linestyle='--', 
                       label=f'Bonferroni α = {alpha_bonferroni:.8f}')
            
            # Highlight significant results
            for i, bar in enumerate(bars):
                if posthoc_df_sorted.iloc[i]['bonferroni_significant']:
                    bar.set_color('green')
                elif posthoc_df_sorted.iloc[i]['uncorrected_significant']:
                    bar.set_color('yellow')
            
            plt.title('Post-hoc Pairwise Comparison P-values')
            plt.xlabel('Model Comparison')
            plt.ylabel('p-value')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.legend()
            
            # Add log scale for p-values for better visualization
            plt.yscale('log')
            
            # Save plot
            plt.savefig(os.path.join(output_path, "posthoc_pvalues.png"), dpi=300)
            plt.close()
            
            # Create matrix visualization of p-values
            plt.figure(figsize=(10, 8))
            
            # Create empty matrix for p-values
            p_matrix = np.ones((n_models, n_models))
            sig_matrix = np.zeros((n_models, n_models))
            
            # Fill in p-values
            for result in posthoc_results:
                i = np.where(models == result['model1'])[0][0]
                j = np.where(models == result['model2'])[0][0]
                p_matrix[i, j] = result['p_value']
                p_matrix[j, i] = result['p_value']  # Mirror
                
                sig_matrix[i, j] = int(result['bonferroni_significant'])
                sig_matrix[j, i] = int(result['bonferroni_significant'])  # Mirror
            
            # Set diagonal to NaN for better visualization
            np.fill_diagonal(p_matrix, np.nan)
            
            # Create heatmap
            mask = np.isnan(p_matrix)
            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(p_matrix, annot=True, fmt=".4g", mask=mask,
                           xticklabels=models, yticklabels=models,
                           cmap="YlOrRd_r", vmin=0, vmax=0.05)
            
            # Add asterisks for significant results
            for i in range(n_models):
                for j in range(n_models):
                    if i != j and sig_matrix[i, j] == 1:
                        ax.text(j + 0.5, i + 0.85, '*', 
                               horizontalalignment='center', 
                               verticalalignment='center',
                               color='white', fontweight='bold', fontsize=15)
            
            plt.title('Pairwise Comparison P-values')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_path, "pvalue_matrix.png"), dpi=300)
            plt.close()
    
    return {
        "anova_results": anova_results,
        "posthoc_results": posthoc_results if posthoc_results else None
    }

def main():
    """Main function to run the analysis without recomputing all models"""
    # Define the path to the model comparison results
    # Modify this path based on your directory structure
    comparison_output_path = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison")
    
    # Check if combined results already exist
    combined_file = os.path.join(comparison_output_path, "combined_performance.csv")
    if os.path.exists(combined_file):
        print(f"Loading existing combined results from {combined_file}")
        combined_df = pd.read_csv(combined_file)
    else:
        # Find all overall performance CSV files
        performance_files = glob.glob(os.path.join(comparison_output_path, "*", "overall_performance.csv"))
        
        if not performance_files:
            print(f"No performance files found in {comparison_output_path}")
            return
            
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
            combined_df.to_csv(combined_file, index=False)
        else:
            print("No performance data could be loaded. Exiting.")
            return
    
    print(f"Found data for {combined_df['rat'].nunique()} rats and {combined_df['model'].nunique()} models")
    
    # Plot combined results
    print("Creating plots...")
    model_stats = plot_combined_results(combined_df, comparison_output_path)
    print(model_stats)
    
    # Run statistical tests
    print("Running statistical tests...")
    stat_results = run_statistical_tests(combined_df, comparison_output_path)
    print(stat_results)
    
    anova_results = model_anova_with_posthoc(combined_df, comparison_output_path)
    print(anova_results["anova_results"])
    
    print(f"\nAnalysis complete! Results saved to {comparison_output_path}")

if __name__ == "__main__":
    main()