import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path
from scipy import stats

from config.paths import paths

base_path = paths.model_comparison

def load_and_process_data():
    """
    Load all ti_comparison.csv files from the specified directory structure and process them.
    
    Returns:
    - model_stats: DataFrame with statistics grouped by model
    - rat_level_df: DataFrame with mean match rates for each rat for each model
    - combined_df: Combined DataFrame with all processed data
    """
    # List to store DataFrames from each file
    all_dfs = []
    
    # Find all ti_comparison.csv files
    pattern = os.path.join(base_path, '**', 'ti_comparison.csv')
    file_paths = glob.glob(pattern, recursive=True)
    
    if not file_paths:
        raise FileNotFoundError(f"No matching CSV files found in {base_path}")
    
    print(f"Found {len(file_paths)} CSV files to process")
    
    # Process each file
    for file_path in file_paths:
        # Extract the rat_id from the path
        rat_id = Path(file_path).parent.name
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add the rat_id as a column
        df['rat_id'] = rat_id
        
        # Calculate match rate as a percentage (1 - error)
        df['match_rate'] = (1 - abs(df['error'])) * 100
        
        all_dfs.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Create rat-level DataFrame (average match rate per rat per model)
    rat_level_df = combined_df.groupby(['rat_id', 'model'])['match_rate'].mean().reset_index()
    
    # Group by model and calculate statistics across rats
    model_stats = rat_level_df.groupby('model').agg(
        avg_match_rate=('match_rate', 'mean'),
        sem_match_rate=('match_rate', 'sem'),  # SEM across rats
        count=('match_rate', 'count')  # Number of rats
    ).reset_index()
    
    return model_stats, rat_level_df, combined_df

def run_statistical_tests(combined_df, output_path=None):
    """
    Run statistical tests on the TI match rate data:
    1. 1-sample t-test against chance level (50%) for each model
    2. One-way ANOVA to test for differences between models
    3. Post-hoc pairwise comparisons with Bonferroni correction
    4. Paired t-tests comparing Betasort to each other model
    
    Parameters:
        - combined_df: DataFrame with combined results
        - output_path: Directory to save test results (optional)
    
    Returns:
        - Dictionary containing statistical test results
    """
    # Initialize results dictionary
    stat_results = {
        'chance_level_tests': [],
        'anova_results': {},
        'posthoc_results': [],
        'betasort_comparison_tests': []
    }
    
    # Get all unique models
    models = combined_df['model'].unique()
    n_models = len(models)
    
    # Create pivot table for rat-level data (average match rate per rat per model)
    # This gives us the data at the right level for statistical testing
    rat_pivot = combined_df.groupby(['rat_id', 'model'])['match_rate'].mean().reset_index()
    rat_pivot = rat_pivot.pivot(index='rat_id', columns='model', values='match_rate')
    
    # 1. Run 1-sample t-tests against chance level (50%)
    print("\n=== T-Tests Against Chance Level (50%) ===")
    for model in models:
        model_data = rat_pivot[model].dropna().values
        t_stat, p_value = stats.ttest_1samp(model_data, 50)  # Testing against 50%
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
        print(f"  Significant vs. chance (50%): {'Yes' if significant else 'No'}")
        print(f"  n = {len(model_data)}")
        print()
    
    # 2. Run one-way ANOVA to test for differences between models
    print("\n=== One-way ANOVA Comparing All Models ===")
    
    # Extract data for each model
    model_data = {}
    anova_data = []
    
    for model in models:
        data = rat_pivot[model].dropna().values
        model_data[model] = data
        anova_data.append(data)
    
    # Calculate observation counts
    obs_per_group = [len(data) for data in anova_data]
    total_obs = sum(obs_per_group)
    df_between = n_models - 1
    df_within = total_obs - n_models
    
    # Run one-way ANOVA
    try:
        f_stat, p_value = stats.f_oneway(*anova_data)
        
        anova_significant = p_value < 0.05
        
        # Store ANOVA results
        stat_results['anova_results'] = {
            'f_stat': f_stat,
            'p_value': p_value,
            'significant': anova_significant,
            'n_models': n_models,
            'total_observations': total_obs,
            'observations_per_group': obs_per_group,
            'df_between': df_between,
            'df_within': df_within
        }
        
        # Print ANOVA results
        print(f"F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_value:.8f}")
        print(f"Total observations: {total_obs}")
        print("Observations per group:")
        for i, model in enumerate(models):
            print(f"  {model}: {obs_per_group[i]} observations")
        print(f"Significant difference across models: {'Yes' if anova_significant else 'No'}")
        
        # 3. If ANOVA is significant, perform post-hoc tests with Bonferroni correction
        if anova_significant:
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
                        
                        # Check if data1 and data2 are from the same rats (paired)
                        is_paired = len(data1) == len(data2)
                        
                        if is_paired:
                            # For paired data, we need to ensure we're comparing the same rats
                            mask1 = ~pd.isna(rat_pivot[model1])
                            mask2 = ~pd.isna(rat_pivot[model2])
                            common_mask = mask1 & mask2
                            
                            if common_mask.sum() > 0:
                                paired_data1 = rat_pivot.loc[common_mask, model1].values
                                paired_data2 = rat_pivot.loc[common_mask, model2].values
                                t_stat, p_value = stats.ttest_rel(paired_data1, paired_data2)
                                test_type = "paired"
                                n_samples = len(paired_data1)
                            else:
                                # Fall back to independent test if no common rats
                                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                                test_type = "independent"
                                n_samples = min(len(data1), len(data2))
                        else:
                            # For independent samples
                            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                            test_type = "independent"
                            n_samples = min(len(data1), len(data2))
                        
                        # Apply Bonferroni correction
                        bonferroni_significant = p_value < alpha_bonferroni
                        uncorrected_significant = p_value < 0.05
                        
                        # Store result
                        result = {
                            'model1': model1,
                            'model2': model2,
                            'model1_mean': np.mean(data1),
                            'model2_mean': np.mean(data2),
                            'mean_diff': np.mean(data1) - np.mean(data2),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'bonferroni_significant': bonferroni_significant,
                            'uncorrected_significant': uncorrected_significant,
                            'test_type': test_type,
                            'n_samples': n_samples
                        }
                        
                        stat_results['posthoc_results'].append(result)
                        
                        # Print result
                        print(f"{model1} vs. {model2}:")
                        print(f"  {model1} Mean: {np.mean(data1):.4f}")
                        print(f"  {model2} Mean: {np.mean(data2):.4f}")
                        print(f"  Mean Difference: {np.mean(data1) - np.mean(data2):.4f}")
                        print(f"  t-statistic: {t_stat:.4f}")
                        print(f"  p-value: {p_value:.8f}")
                        print(f"  Significant after Bonferroni correction: {'Yes' if bonferroni_significant else 'No'}")
                        print(f"  Test type: {test_type}, n = {n_samples}")
                        print()
    
    except Exception as e:
        print(f"Error in ANOVA: {e}")
        stat_results['anova_results'] = {
            'error': str(e)
        }
    
    # 4. Run paired t-tests comparing Betasort to each other model
    print("\n=== Paired T-Tests: Betasort vs. Other Models ===")
    
    # Check if 'Betasort' is one of the models
    if 'Betasort' not in models:
        print("Error: Betasort model not found in data")
        return stat_results
    
    # Run paired t-tests
    for model in models:
        if model == 'Betasort':
            continue
        
        # Get paired data (removing any NaN values)
        valid_pairs = ~(pd.isna(rat_pivot['Betasort']) | pd.isna(rat_pivot[model]))
        betasort_data = rat_pivot.loc[valid_pairs, 'Betasort'].values
        model_data = rat_pivot.loc[valid_pairs, model].values
        
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
    
    # Save results to CSV if output_path is provided
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to DataFrames and save
        chance_df = pd.DataFrame(stat_results['chance_level_tests'])
        chance_df.to_csv(os.path.join(output_path, "ti_chance_level_tests.csv"), index=False)
        
        # Save ANOVA results
        anova_df = pd.DataFrame([stat_results['anova_results']])
        anova_df.to_csv(os.path.join(output_path, "ti_anova_results.csv"), index=False)
        
        # Save post-hoc results if they exist
        if stat_results['posthoc_results']:
            posthoc_df = pd.DataFrame(stat_results['posthoc_results'])
            posthoc_df.to_csv(os.path.join(output_path, "ti_posthoc_results.csv"), index=False)
        
        # Save Betasort comparison results
        comparison_df = pd.DataFrame(stat_results['betasort_comparison_tests'])
        comparison_df.to_csv(os.path.join(output_path, "ti_betasort_comparison_tests.csv"), index=False)
        
        # Create visualization of post-hoc p-values if they exist
        if stat_results['posthoc_results']:
            # Create p-value bar chart
            plt.figure(figsize=(12, 8))
            
            # Sort by p-value
            posthoc_df = pd.DataFrame(stat_results['posthoc_results'])
            posthoc_df_sorted = posthoc_df.sort_values('p_value')
            
            # Create comparison labels
            comparison_labels = [f"{row['model1']} vs. {row['model2']}" 
                                for _, row in posthoc_df_sorted.iterrows()]
            
            # Create bar plot of p-values
            bars = plt.bar(comparison_labels, posthoc_df_sorted['p_value'], color='skyblue')
            
            # Add uncorrected alpha threshold
            plt.axhline(y=0.05, color='orange', linestyle='--', label='α = 0.05')
            
            # Add Bonferroni-corrected alpha threshold
            alpha_bonferroni = 0.05 / ((n_models * (n_models - 1)) // 2)
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
            plt.savefig(os.path.join(output_path, "ti_posthoc_pvalues.png"), dpi=300)
            plt.close()
            
            # Create matrix visualization of p-values
            p_matrix = np.ones((n_models, n_models))
            sig_matrix = np.zeros((n_models, n_models))
            
            # Fill in p-values
            for result in stat_results['posthoc_results']:
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
            plt.savefig(os.path.join(output_path, "ti_pvalue_matrix.png"), dpi=300)
            plt.close()
    
    return stat_results

def plot_model_match_rates(model_stats, rat_level_df, combined_df, stat_results=None, figsize=(12, 8)):
    """
    Plot the average match rates with SEM error bars.
    
    Parameters:
        - model_stats: DataFrame with model statistics
        - rat_level_df: DataFrame with rat-level averages
        - combined_df: Combined DataFrame with all processed data
        - stat_results: Results from statistical tests (optional)
        - figsize: Figure size as a tuple (width, height)
    """
    # Sort by match rate for better visualization
    model_order = model_stats.sort_values('avg_match_rate', ascending=False)['model'].tolist()
    
    # Create figure with specified size
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    # Create boxplot using rat-level data
    boxplot = sns.boxplot(
        x='model', 
        y='match_rate',
        data=rat_level_df,
        palette="viridis_r",
        order=model_order
    )
    
    # Add individual data points as a stripplot (one point per rat)
    sns.stripplot(
        x='model', 
        y='match_rate',
        data=rat_level_df, 
        color='black',
        alpha=0.5,
        jitter=True,
        order=model_order
    )
    
    # Add chance level reference line at 50%
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot appearance
    plt.title("Transitive Inference Match Rates", fontsize=30)
    plt.xlabel("Model", fontsize=24)
    plt.ylabel("Match Rate (%)", fontsize=24)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    
    plt.tight_layout()
    
    # Save plot
    output_path = base_path / "TI_match_rate_with_stats.png"
    plt.savefig(output_path, dpi=300)
    
    return boxplot

def main():
    """
    Main function to execute the data processing, statistical testing, and plotting.
    """
    try:
        # Load and process data
        model_stats, rat_level_df, combined_df = load_and_process_data()
        
        print("Model statistics:")
        print(model_stats)
        
        # Run statistical tests
        stat_results = run_statistical_tests(rat_level_df, base_path)
        
        # Plot the results with statistical markers
        plot_model_match_rates(model_stats, rat_level_df, combined_df, stat_results)
        
        print(f"\nAnalysis complete! Results saved to {base_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()