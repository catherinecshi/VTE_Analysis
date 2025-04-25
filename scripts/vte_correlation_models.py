import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
from pathlib import Path

from src import helper

# Set the style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
sns.set_context("paper", font_scale=1.2)

# Function to read and prepare the data
def load_data(file_path):
    """
    Load and prepare the VTE correlations data
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert p_value to numeric if needed
    df['p_value'] = pd.to_numeric(df['p_value'], errors='coerce')
    
    # Convert correlation to numeric if needed
    df['correlation'] = pd.to_numeric(df['correlation'], errors='coerce')
    
    # Convert significant to boolean if needed
    if df['significant'].dtype == 'object':
        df['significant'] = df['significant'].map({'TRUE': True, 'FALSE': False})
    
    return df

def run_statistical_tests(df, output_path=None):
    """
    Run statistical tests on VTE correlation data:
    1. 1-sample t-test against 0 (no correlation) for each model using all data
    2. Independent t-tests comparing Betasort to each other model with Bonferroni correction
    3. One-way ANOVA comparing all models
    4. T-tests for each uncertainty type against zero
    5. Pairwise t-tests between uncertainty types with Bonferroni correction
    6. One-way ANOVA comparing all uncertainty types
    
    Parameters:
        - df: DataFrame with correlation data
        - output_path: Directory to save test results (optional)
    
    Returns:
        - Dictionary containing statistical test results
    """
    # Initialize results dictionary
    stat_results = {
        'zero_correlation_tests': [],
        'betasort_comparison_tests': [],
        'betasort_uncertainty_tests': [],
        'model_anova': None,
        'uncertainty_pairwise_tests': [],
        'uncertainty_anova': None
    }
    
    # Filter for overall scope data if scope column exists
    if 'scope' in df.columns:
        df_overall = df[df['scope'] == 'overall']
    else:
        df_overall = df
    
    # Get all unique models
    models = df_overall['model'].unique()
    
    # 1. Run 1-sample t-tests against 0 (no correlation) for each model (NO Bonferroni correction)
    print("\n=== T-Tests Against Zero Correlation ===")
    for model in models:
        # Get all correlation values for this model
        model_data = df_overall[df_overall['model'] == model]['correlation'].values
        
        # Run one-sample t-test against zero
        t_stat, p_value = stats.ttest_1samp(model_data, 0)
        significant = p_value < 0.05  # Standard alpha, no correction
        
        result = {
            'model': model,
            'mean': np.mean(model_data),
            'std': np.std(model_data),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'n_samples': len(model_data)
        }
        
        stat_results['zero_correlation_tests'].append(result)
        
        print(f"Model: {model}")
        print(f"  Mean correlation: {np.mean(model_data):.4f}")
        print(f"  Standard deviation: {np.std(model_data):.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant vs. zero: {'Yes' if significant else 'No'}")
        print(f"  n = {len(model_data)}")
        print()
    
    # 2. Compare Betasort to each other model using independent t-test WITH Bonferroni correction
    print("\n=== Independent T-Tests: Betasort vs. Other Models (Bonferroni-corrected) ===")
    
    # Check if 'Betasort' is one of the models
    if 'Betasort' in models:
        betasort_data = df_overall[df_overall['model'] == 'Betasort']['correlation'].values
        
        # Calculate the number of comparisons for Bonferroni correction
        alpha = 0.05  # Original significance level
        num_comparisons = len(models) - 1  # Number of models excluding Betasort
        bonferroni_alpha = alpha / num_comparisons  # Bonferroni-corrected alpha
        
        print(f"  Using Bonferroni-corrected alpha = {bonferroni_alpha:.5f} for {num_comparisons} comparisons")
        
        for model in models:
            if model == 'Betasort':
                continue
            
            # Get all correlation values for this model
            model_data = df_overall[df_overall['model'] == model]['correlation'].values
            
            # Run independent t-test (not assuming equal variances)
            t_stat, p_value = stats.ttest_ind(betasort_data, model_data, equal_var=False)
            significant = p_value < bonferroni_alpha  # Apply Bonferroni correction
            
            result = {
                'model': model,
                'betasort_mean': np.mean(betasort_data),
                'model_mean': np.mean(model_data),
                'mean_diff': np.mean(betasort_data) - np.mean(model_data),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': significant,
                'n_betasort': len(betasort_data),
                'n_model': len(model_data),
                'corrected_alpha': bonferroni_alpha
            }
            
            stat_results['betasort_comparison_tests'].append(result)
            
            print(f"Betasort vs. {model}")
            print(f"  Betasort Mean: {np.mean(betasort_data):.4f}")
            print(f"  {model} Mean: {np.mean(model_data):.4f}")
            print(f"  Mean Difference: {np.mean(betasort_data) - np.mean(model_data):.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant difference (Bonferroni-corrected): {'Yes' if significant else 'No'}")
            print(f"  n(Betasort) = {len(betasort_data)}, n({model}) = {len(model_data)}")
            print()
    
    # 3. NEW: One-way ANOVA comparing all models
    print("\n=== One-way ANOVA: Comparing All Models ===")
    model_groups = []
    model_names = []
    
    for model in models:
        model_data = df_overall[df_overall['model'] == model]['correlation'].values
        model_groups.append(model_data)
        model_names.append(model)
    
    # Run one-way ANOVA
    f_stat, p_value = stats.f_oneway(*model_groups)
    significant = p_value < 0.05
    
    anova_result = {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': significant,
        'models': model_names,
        'n_models': len(models)
    }
    
    stat_results['model_anova'] = anova_result
    
    print(f"One-way ANOVA comparing all models:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant difference between models: {'Yes' if significant else 'No'}")
    print(f"  Models compared: {', '.join(model_names)}")
    print()
    
    # 4. Run t-tests for each uncertainty type within the Betasort model against zero (NO Bonferroni)
    print("\n=== T-Tests for Betasort Uncertainty Types Against Zero ===")
    
    # Filter for Betasort model data
    betasort_df = df_overall[df_overall['model'] == 'Betasort']
    
    # Get all unique uncertainty types
    uncertainty_types = betasort_df['uncertainty_type'].unique()
    
    for uncertainty_type in uncertainty_types:
        # Get all correlation values for this uncertainty type
        type_data = betasort_df[betasort_df['uncertainty_type'] == uncertainty_type]['correlation'].values
        
        # Skip if we don't have enough data
        if len(type_data) < 2:
            print(f"Uncertainty type '{uncertainty_type}' has insufficient data (n={len(type_data)})")
            continue
        
        # Run one-sample t-test against zero
        t_stat, p_value = stats.ttest_1samp(type_data, 0)
        significant = p_value < 0.05  # Standard alpha
        
        result = {
            'uncertainty_type': uncertainty_type,
            'mean': np.mean(type_data),
            'std': np.std(type_data),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'n_samples': len(type_data)
        }
        
        stat_results['betasort_uncertainty_tests'].append(result)
        
        print(f"Uncertainty type: {uncertainty_type}")
        print(f"  Mean correlation: {np.mean(type_data):.4f}")
        print(f"  Standard deviation: {np.std(type_data):.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant vs. zero: {'Yes' if significant else 'No'}")
        print(f"  n = {len(type_data)}")
        print()
    
    # 5. NEW: Pairwise t-tests between uncertainty types WITH Bonferroni correction
    print("\n=== Pairwise T-Tests Between Uncertainty Types (Bonferroni-corrected) ===")
    
    # Calculate number of pairwise comparisons for Bonferroni correction
    num_uncertainty_types = len(uncertainty_types)
    num_uncertainty_comparisons = (num_uncertainty_types * (num_uncertainty_types - 1)) // 2  # n(n-1)/2 pairs
    
    if num_uncertainty_comparisons > 0:
        uncertainty_alpha = 0.05 / num_uncertainty_comparisons  # Bonferroni-corrected alpha
        
        print(f"  Using Bonferroni-corrected alpha = {uncertainty_alpha:.5f} for {num_uncertainty_comparisons} comparisons")
        
        # Perform all pairwise comparisons
        for i in range(len(uncertainty_types)):
            for j in range(i+1, len(uncertainty_types)):
                type1 = uncertainty_types[i]
                type2 = uncertainty_types[j]
                
                # Get data for both types
                type1_data = betasort_df[betasort_df['uncertainty_type'] == type1]['correlation'].values
                type2_data = betasort_df[betasort_df['uncertainty_type'] == type2]['correlation'].values
                
                # Skip if either doesn't have enough data
                if len(type1_data) < 2 or len(type2_data) < 2:
                    print(f"Skipping comparison {type1} vs {type2} due to insufficient data")
                    continue
                
                # Run independent t-test
                t_stat, p_value = stats.ttest_ind(type1_data, type2_data, equal_var=False)
                significant = p_value < uncertainty_alpha  # Apply Bonferroni correction
                
                result = {
                    'type1': type1,
                    'type2': type2,
                    'type1_mean': np.mean(type1_data),
                    'type2_mean': np.mean(type2_data),
                    'mean_diff': np.mean(type1_data) - np.mean(type2_data),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significant,
                    'n_type1': len(type1_data),
                    'n_type2': len(type2_data),
                    'corrected_alpha': uncertainty_alpha
                }
                
                stat_results['uncertainty_pairwise_tests'].append(result)
                
                print(f"{type1} vs. {type2}")
                print(f"  {type1} Mean: {np.mean(type1_data):.4f}")
                print(f"  {type2} Mean: {np.mean(type2_data):.4f}")
                print(f"  Mean Difference: {np.mean(type1_data) - np.mean(type2_data):.4f}")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Significant difference (Bonferroni-corrected): {'Yes' if significant else 'No'}")
                print(f"  n({type1}) = {len(type1_data)}, n({type2}) = {len(type2_data)}")
                print()
    else:
        print("  Not enough uncertainty types for pairwise comparison")
    
    # 6. NEW: One-way ANOVA comparing all uncertainty types
    print("\n=== One-way ANOVA: Comparing All Uncertainty Types ===")
    uncertainty_groups = []
    uncertainty_names = []
    
    for uncertainty_type in uncertainty_types:
        type_data = betasort_df[betasort_df['uncertainty_type'] == uncertainty_type]['correlation'].values
        if len(type_data) >= 2:  # Need at least 2 samples for each group
            uncertainty_groups.append(type_data)
            uncertainty_names.append(uncertainty_type)
    
    if len(uncertainty_groups) > 1:  # Need at least 2 groups for ANOVA
        # Run one-way ANOVA
        f_stat, p_value = stats.f_oneway(*uncertainty_groups)
        significant = p_value < 0.05
        
        anova_result = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': significant,
            'uncertainty_types': uncertainty_names,
            'n_types': len(uncertainty_names)
        }
        
        stat_results['uncertainty_anova'] = anova_result
        
        print(f"One-way ANOVA comparing all uncertainty types:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant difference between uncertainty types: {'Yes' if significant else 'No'}")
        print(f"  Uncertainty types compared: {', '.join(uncertainty_names)}")
        
        # Calculate total number of observations
        total_n = sum(len(group) for group in model_groups)
        # Calculate degrees of freedom
        df_between = len(model_groups) - 1
        df_within = total_n - len(model_groups)
        df_total = total_n - 1

        print(f"  Total observations: {total_n}")
        print(f"  Degrees of freedom (between groups): {df_between}")
        print(f"  Degrees of freedom (within groups): {df_within}")
        print(f"  Degrees of freedom (total): {df_total}")
    else:
        print("  Not enough uncertainty types with sufficient data for ANOVA")
    
    # Save results to CSV if output_path is provided
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to DataFrames and save
        zero_corr_df = pd.DataFrame(stat_results['zero_correlation_tests'])
        zero_corr_df.to_csv(os.path.join(output_path, "vte_zero_correlation_tests.csv"), index=False)
        
        if len(stat_results['betasort_comparison_tests']) > 0:
            comparison_df = pd.DataFrame(stat_results['betasort_comparison_tests'])
            comparison_df.to_csv(os.path.join(output_path, "vte_betasort_comparison_tests.csv"), index=False)
            
        if len(stat_results['betasort_uncertainty_tests']) > 0:
            uncertainty_df = pd.DataFrame(stat_results['betasort_uncertainty_tests'])
            uncertainty_df.to_csv(os.path.join(output_path, "vte_betasort_uncertainty_tests.csv"), index=False)
        
        if len(stat_results['uncertainty_pairwise_tests']) > 0:
            uncertainty_pairwise_df = pd.DataFrame(stat_results['uncertainty_pairwise_tests'])
            uncertainty_pairwise_df.to_csv(os.path.join(output_path, "vte_uncertainty_pairwise_tests.csv"), index=False)
        
        if stat_results['model_anova']:
            model_anova_df = pd.DataFrame([stat_results['model_anova']])
            model_anova_df.to_csv(os.path.join(output_path, "vte_model_anova.csv"), index=False)
            
        if stat_results['uncertainty_anova']:
            uncertainty_anova_df = pd.DataFrame([stat_results['uncertainty_anova']])
            uncertainty_anova_df.to_csv(os.path.join(output_path, "vte_uncertainty_anova.csv"), index=False)
    
    return stat_results

# 2. Function to visualize Betasort model correlations by uncertainty type
def plot_betasort_uncertainty_types(df, stat_results=None, save_path=None):
    """
    Create a bar plot showing Betasort correlations across different uncertainty types
    with statistical significance markers
    """
    # Filter data for Betasort model only
    betasort_df = df[df['model'] == 'Betasort']
    
    # Calculate mean correlation by uncertainty type
    uncertainty_corr = betasort_df.groupby('uncertainty_type')['correlation'].mean().reset_index()
    
    # Sort by correlation value
    uncertainty_corr = uncertainty_corr.sort_values('correlation', ascending=False)
    
    # Create the figure
    plt.figure(figsize=(12, 6))
    
    # Create the bar plot
    ax = sns.barplot(x='uncertainty_type', y='correlation', data=uncertainty_corr, 
                    palette='viridis', errorbar=None)
    
    # Add standard error bars
    uncertainty_sem = betasort_df.groupby('uncertainty_type')['correlation'].sem().reset_index()
    for i, uncertainty_type in enumerate(uncertainty_corr['uncertainty_type']):
        sem = uncertainty_sem[uncertainty_sem['uncertainty_type'] == uncertainty_type]['correlation'].values[0]
        mean = uncertainty_corr[uncertainty_corr['uncertainty_type'] == uncertainty_type]['correlation'].values[0]
        ax.errorbar(i, mean, yerr=sem, fmt='none', color='black', capsize=5)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add significance markers if we have statistical results
    if stat_results and 'betasort_uncertainty_tests' in stat_results:
        uncertainty_tests = pd.DataFrame(stat_results['betasort_uncertainty_tests'])
        
        for i, uncertainty_type in enumerate(uncertainty_corr['uncertainty_type']):
            if uncertainty_type in uncertainty_tests['uncertainty_type'].values:
                test_result = uncertainty_tests[uncertainty_tests['uncertainty_type'] == uncertainty_type].iloc[0]
                if test_result['significant']:
                    # Add star for significance vs zero
                    plt.text(
                        i,
                        uncertainty_corr[uncertainty_corr['uncertainty_type'] == uncertainty_type]['correlation'].values[0] + 0.01,
                        '*',
                        ha='center',
                        va='center',
                        fontsize=16,
                        color='black'
                    )
    
    # Enhance the plot
    plt.title('Betasort: Correlation Between Uncertainty Types and VTE Behavior', fontsize=16)
    plt.xlabel('Uncertainty Type', fontsize=14)
    plt.ylabel('Average Correlation', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003 if bar.get_height() > 0 else bar.get_height() - 0.01,
            f'{bar.get_height():.4f}',
            ha='center',
            va='bottom' if bar.get_height() > 0 else 'top',
            fontsize=10
        )
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='r', linestyle='--', label='Zero Correlation'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=10, label='p < 0.05 vs. Zero')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return ax

# 1. Function to create a plot comparing correlations across different models
def plot_model_comparison(df, stat_results=None, save_path=None):
    """
    Create a bar plot comparing average correlations across different models
    """
    # Calculate mean correlation by model
    model_corr = df.groupby('model')['correlation'].mean().reset_index()
    
    # Sort models by average correlation
    model_corr = model_corr.sort_values('correlation', ascending=False)
    
    # Create the figure
    plt.figure(figsize=(12, 6))
    
    # Create the bar plot
    ax = sns.barplot(x='model', y='correlation', data=model_corr, 
                    palette='viridis', errorbar=None)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add error bars showing standard error
    model_sem = df.groupby('model')['correlation'].sem().reset_index()
    for i, model in enumerate(model_corr['model']):
        sem = model_sem[model_sem['model'] == model]['correlation'].values[0]
        mean = model_corr[model_corr['model'] == model]['correlation'].values[0]
        ax.errorbar(i, mean, yerr=sem, fmt='none', color='black', capsize=5)
    
    # Add significance markers if we have statistical results
    if stat_results and 'zero_correlation_tests' in stat_results:
        zero_tests = pd.DataFrame(stat_results['zero_correlation_tests'])
        
        for i, model in enumerate(model_corr['model']):
            if model in zero_tests['model'].values:
                test_result = zero_tests[zero_tests['model'] == model].iloc[0]
                if 'significant' in test_result and test_result['significant']:
                    # Add star for significance vs zero
                    plt.text(
                        i,
                        model_corr[model_corr['model'] == model]['correlation'].values[0] + 0.01,
                        '*',
                        ha='center',
                        va='center',
                        fontsize=16,
                        color='black'
                    )
        
        # Add significance markers for Betasort comparison if available
        if 'betasort_comparison_tests' in stat_results and stat_results['betasort_comparison_tests']:
            betasort_tests = pd.DataFrame(stat_results['betasort_comparison_tests'])
            model_order = model_corr['model'].tolist()
            betasort_idx = model_order.index('Betasort') if 'Betasort' in model_order else None
            
            if betasort_idx is not None:
                for i, model in enumerate(model_order):
                    if model == 'Betasort':
                        continue
                    
                    if model in betasort_tests['model'].values:
                        test_result = betasort_tests[betasort_tests['model'] == model].iloc[0]
                        if test_result['significant']:
                            # Draw a line connecting Betasort and this model
                            y_level = max(model_corr['correlation']) + 0.02
                            plt.plot([betasort_idx, i], [y_level, y_level], 'k-', linewidth=1.5)
                            plt.text(
                                (betasort_idx + i) / 2,
                                y_level + 0.01,
                                '+',
                                ha='center',
                                va='bottom',
                                fontsize=14,
                                color='black'
                            )
    
    # Enhance the plot
    plt.title('Average Correlation by Model', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Average Correlation', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003 if bar.get_height() > 0 else bar.get_height() - 0.01,
            f'{bar.get_height():.4f}',
            ha='center',
            va='bottom' if bar.get_height() > 0 else 'top',
            fontsize=10
        )
    
    # Add legend with significance markers
    if stat_results:
        legend_elements = [
            plt.Line2D([0], [0], color='r', linestyle='--', label='Zero Correlation'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=10, label='p < 0.05 vs. Zero'),
            plt.Line2D([0], [0], marker='+', color='w', markerfacecolor='k', markersize=10, label='p < 0.05 vs. Betasort')
        ]
        plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return ax

# 2. Function to visualize Betasort model correlations by uncertainty type
def plot_betasort_uncertainty_types(df, stat_results=None, save_path=None):
    """
    Create a bar plot showing Betasort correlations across different uncertainty types
    """
    # Filter data for Betasort model only
    betasort_df = df[df['model'] == 'Betasort']
    
    # Calculate mean correlation by uncertainty type
    uncertainty_corr = betasort_df.groupby('uncertainty_type')['correlation'].mean().reset_index()
    
    # Sort by correlation value
    uncertainty_corr = uncertainty_corr.sort_values('correlation', ascending=False)
    
    # Create the figure
    plt.figure(figsize=(12, 6))
    
    # Create the bar plot
    ax = sns.barplot(x='uncertainty_type', y='correlation', data=uncertainty_corr, 
                    palette='viridis', errorbar=None)
    
    # Add standard error bars
    uncertainty_sem = betasort_df.groupby('uncertainty_type')['correlation'].sem().reset_index()
    for i, uncertainty_type in enumerate(uncertainty_corr['uncertainty_type']):
        sem = uncertainty_sem[uncertainty_sem['uncertainty_type'] == uncertainty_type]['correlation'].values[0]
        mean = uncertainty_corr[uncertainty_corr['uncertainty_type'] == uncertainty_type]['correlation'].values[0]
        ax.errorbar(i, mean, yerr=sem, fmt='none', color='black', capsize=5)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add significance stars for each uncertainty type if p-value < 0.05
    for i, uncertainty_type in enumerate(uncertainty_corr['uncertainty_type']):
        type_data = betasort_df[betasort_df['uncertainty_type'] == uncertainty_type]
        significant = type_data['significant'].mean() > 0.5  # If more than half are significant
        
        if significant:
            plt.text(
                i,
                uncertainty_corr[uncertainty_corr['uncertainty_type'] == uncertainty_type]['correlation'].values[0] + 0.01,
                '*',
                ha='center',
                va='center',
                fontsize=16,
                color='black'
            )
    
    # Enhance the plot
    plt.title('Betasort: Average Correlation by Uncertainty Type', fontsize=16)
    plt.xlabel('Uncertainty Type', fontsize=14)
    plt.ylabel('Average Correlation', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003 if bar.get_height() > 0 else bar.get_height() - 0.01,
            f'{bar.get_height():.4f}',
            ha='center',
            va='bottom' if bar.get_height() > 0 else 'top',
            fontsize=10
        )
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='r', linestyle='--', label='Zero Correlation'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=10, label='p < 0.05')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return ax

# 3. Function to create a scatter plot of correlation vs p-value for Betasort
def plot_betasort_significance(df, save_path=None):
    """
    Create a scatter plot of correlation vs p-value for Betasort model
    """
    # Filter data for Betasort model only
    betasort_df = df[df['model'] == 'Betasort']
    
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Create the scatter plot with colors based on significance
    ax = sns.scatterplot(
        x='correlation', 
        y='p_value',
        hue='significant',
        style='uncertainty_type',
        s=100,  # point size
        alpha=0.7,
        data=betasort_df
    )
    
    # Add a horizontal line at p=0.05 for significance reference
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    
    # Add a vertical line at correlation=0 for reference
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Enhance the plot
    plt.title('Betasort: Correlation vs. P-value', fontsize=16)
    plt.xlabel('Correlation', fontsize=14)
    plt.ylabel('P-value', fontsize=14)
    
    # Set y-axis to log scale for better visualization of p-values
    plt.yscale('log')
    
    # Add grid for easier reading
    plt.grid(True, alpha=0.3)
    
    # Adjust legend
    plt.legend(title='Significance', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return ax

# 4. Function to create a heatmap of correlations by model and uncertainty type
def plot_correlation_heatmap(df, save_path=None):
    """
    Create a heatmap showing average correlations by model and uncertainty type
    """
    # Create a pivot table of average correlations
    heatmap_data = df.pivot_table(
        values='correlation',
        index='model',
        columns='uncertainty_type',
        aggfunc='mean'
    )
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,  # Show values in cells
        fmt=".4f",   # Format for the annotations
        cmap="coolwarm",  # Colormap (red for negative, blue for positive)
        center=0,    # Center the colormap at 0
        linewidths=.5,
        cbar_kws={"shrink": 0.8, "label": "Average Correlation"}
    )
    
    # Create a significance mask for the heatmap
    # Calculate a significance matrix with the same shape as heatmap_data
    sig_pivot = df.pivot_table(
        values='significant',
        index='model',
        columns='uncertainty_type',
        aggfunc=lambda x: np.mean(x) > 0.5  # True if more than half are significant
    )
    
    # Add stars to significant cells
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if i < sig_pivot.shape[0] and j < sig_pivot.shape[1]:
                if sig_pivot.iloc[i, j]:
                    ax.text(j + 0.5, i + 0.85, '*', ha='center', va='center', color='black', fontsize=14)
    
    # Enhance the plot
    plt.title('Average Correlation by Model and Uncertainty Type', fontsize=16)
    plt.ylabel('Model', fontsize=14)
    plt.xlabel('Uncertainty Type', fontsize=14)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return ax

# 5. Function to create box plots of correlations by model
def plot_correlation_boxplots(df, stat_results=None, save_path=None):
    """
    Create box plots showing the distribution of correlations by model
    """
    # Sort models by mean correlation for consistent ordering
    model_order = df.groupby('model')['correlation'].mean().sort_values(ascending=False).index.tolist()
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create the box plot
    ax = sns.boxplot(
        x='model',
        y='correlation',
        data=df,
        palette='viridis',
        order=model_order
    )
    
    # Add a swarm plot for individual data points
    sns.swarmplot(
        x='model',
        y='correlation',
        data=df,
        color='black',
        alpha=0.5,
        size=4,
        order=model_order
    )
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add significance markers if we have statistical results
    if stat_results and 'zero_correlation_tests' in stat_results:
        zero_tests = pd.DataFrame(stat_results['zero_correlation_tests'])
        
        for i, model in enumerate(model_order):
            if model in zero_tests['model'].values:
                test_result = zero_tests[zero_tests['model'] == model].iloc[0]
                if 'significant' in test_result and test_result['significant']:
                    # Add star for significance vs zero
                    y_pos = df[df['model'] == model]['correlation'].max() + 0.05
                    plt.text(
                        i,
                        y_pos,
                        '*',
                        ha='center',
                        va='center',
                        fontsize=16,
                        color='black'
                    )
        
        # Add significance markers for Betasort comparison
        if 'betasort_comparison_tests' in stat_results and stat_results['betasort_comparison_tests']:
            betasort_tests = pd.DataFrame(stat_results['betasort_comparison_tests'])
            betasort_idx = model_order.index('Betasort') if 'Betasort' in model_order else None
            
            if betasort_idx is not None:
                for i, model in enumerate(model_order):
                    if model == 'Betasort':
                        continue
                    
                    if model in betasort_tests['model'].values:
                        test_result = betasort_tests[betasort_tests['model'] == model].iloc[0]
                        if test_result['significant']:
                            # Draw a line connecting Betasort and this model
                            y_level = df['correlation'].max() + 0.1
                            plt.plot([betasort_idx, i], [y_level, y_level], 'k-', linewidth=1.5)
                            plt.text(
                                (betasort_idx + i) / 2,
                                y_level + 0.05,
                                '+',
                                ha='center',
                                va='bottom',
                                fontsize=14,
                                color='black'
                            )
    
    # Enhance the plot
    plt.title('Distribution of Correlations by Model', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Correlation', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for easier reading
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add legend with significance markers
    if stat_results:
        legend_elements = [
            plt.Line2D([0], [0], color='r', linestyle='--', label='Zero Correlation'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=10, label='p < 0.05 vs. Zero'),
            plt.Line2D([0], [0], marker='+', color='w', markerfacecolor='k', markersize=10, label='p < 0.05 vs. Betasort')
        ]
        plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return ax

# Main execution function
def main(file_path):
    """
    Main function to generate all plots and run statistical analyses
    """
    # Load the data
    df = load_data(file_path)
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Models: {', '.join(df['model'].unique())}")
    print(f"Uncertainty types: {', '.join(df['uncertainty_type'].unique())}")
    
    # Create output directory for plots if it doesn't exist
    output_dir = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison", 'correlation_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run statistical tests
    stat_results = run_statistical_tests(df, output_dir)
    
    # Generate and save all plots with statistical markers
    plot_model_comparison(df, stat_results, os.path.join(output_dir, '1_model_comparison_with_stats.png'))
    plot_betasort_uncertainty_types(df, stat_results, os.path.join(output_dir, '2_betasort_uncertainty_with_stats.png'))
    plot_betasort_significance(df, os.path.join(output_dir, '3_betasort_significance.png'))
    plot_correlation_heatmap(df, os.path.join(output_dir, '4_correlation_heatmap_with_stats.png'))
    plot_correlation_boxplots(df, stat_results, os.path.join(output_dir, '5_correlation_boxplots_with_stats.png'))
    
    print(f"All plots and statistical analyses have been generated and saved to the '{output_dir}' directory")

# Run the script
if __name__ == "__main__":
    # Replace this with your actual file path
    file_path = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison", "combined_vte_correlations.csv")
    main(file_path)