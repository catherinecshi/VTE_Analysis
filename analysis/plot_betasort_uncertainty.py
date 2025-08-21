"""
Standalone script to plot Betasort uncertainty correlations with VTE behavior.

This script extracts VTE correlation data from betasort_overall_pipeline results
and generates the plot_betasort_uncertainty_types visualization without needing
to run the full compare_models.py pipeline.

Usage:
    python analysis/plot_betasort_uncertainty.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, ttest_1samp

from config.paths import paths

# Set the style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
sns.set_context("paper", font_scale=1.2)


def extract_vte_correlations_from_betasort_results(save_path=None):
    """
    Extract VTE correlation data from saved betasort pipeline results
    
    Parameters:
    -----------
    save_path : str, optional
        Path to betasort results. Uses paths.betasort_data if None
    
    Returns:
    --------
    pd.DataFrame : DataFrame with correlation data for each rat and uncertainty type
    """
    if save_path is None:
        save_path = paths.betasort_data
    
    print(f"Extracting VTE correlations from: {save_path}")
    
    # Storage for all correlation data
    all_correlations = []
    
    # Get all rat directories
    rat_dirs = [d for d in os.listdir(save_path) 
               if os.path.isdir(os.path.join(save_path, d))
               and d != "aggregated_plots"]
    
    print(f"Found {len(rat_dirs)} rat directories: {rat_dirs}")
    
    for rat_name in rat_dirs:
        rat_dir = os.path.join(save_path, rat_name)
        vte_file = os.path.join(rat_dir, "vte_uncertainty.csv")
        
        if not os.path.exists(vte_file):
            print(f"  Warning: No VTE file found for {rat_name}")
            continue
        
        try:
            # Load VTE data
            vte_df = pd.read_csv(vte_file)
            
            if len(vte_df) == 0:
                print(f"  Warning: Empty VTE file for {rat_name}")
                continue
            
            print(f"  Processing {rat_name}: {len(vte_df)} trials")
            
            # Calculate correlations for different uncertainty types
            uncertainty_types = ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']
            
            for uncertainty_type in uncertainty_types:
                if uncertainty_type not in vte_df.columns:
                    print(f"    Warning: {uncertainty_type} not found in {rat_name}")
                    continue
                
                # Calculate point-biserial correlation
                uncertainty_values = vte_df[uncertainty_type].values
                vte_values = vte_df['vte_occurred'].values
                
                # Remove any NaN values
                mask = ~(np.isnan(uncertainty_values) | np.isnan(vte_values))
                uncertainty_clean = uncertainty_values[mask]
                vte_clean = vte_values[mask]
                
                if len(uncertainty_clean) < 3:  # Need at least 3 points for correlation
                    print(f"    Warning: Insufficient data for {uncertainty_type} in {rat_name}")
                    continue
                
                # Calculate correlation
                correlation, p_value = pointbiserialr(vte_clean, uncertainty_clean)
                significant = p_value < 0.05
                
                # Store result
                all_correlations.append({
                    'rat': rat_name,
                    'model': 'Betasort',  # For compatibility with plotting function
                    'uncertainty_type': uncertainty_type,
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': significant,
                    'n_trials': len(uncertainty_clean),
                    'scope': 'overall'  # For compatibility
                })
                
                print(f"    {uncertainty_type}: r={correlation:.4f}, p={p_value:.4f}, n={len(uncertainty_clean)}")
        
        except Exception as e:
            print(f"  Error processing {rat_name}: {e}")
            continue
    
    # Convert to DataFrame
    correlations_df = pd.DataFrame(all_correlations)
    
    if len(correlations_df) == 0:
        raise ValueError("No correlation data could be extracted from the results")
    
    print(f"\nExtracted {len(correlations_df)} correlation measurements")
    print(f"Uncertainty types: {correlations_df['uncertainty_type'].unique()}")
    print(f"Rats: {correlations_df['rat'].unique()}")
    
    return correlations_df


def run_betasort_uncertainty_statistical_tests(df):
    """
    Run statistical tests on Betasort uncertainty correlation data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Betasort correlation data
    
    Returns:
    --------
    dict : Statistical test results
    """
    print("\n=== Statistical Tests for Betasort Uncertainty Types ===")
    
    stat_results = {
        'betasort_uncertainty_tests': []
    }
    
    # Get unique uncertainty types
    uncertainty_types = df['uncertainty_type'].unique()
    
    # Test each uncertainty type against zero correlation
    for uncertainty_type in uncertainty_types:
        type_data = df[df['uncertainty_type'] == uncertainty_type]['correlation'].values
        
        if len(type_data) < 2:
            print(f"Insufficient data for {uncertainty_type} (n={len(type_data)})")
            continue
        
        # Run one-sample t-test against zero
        t_stat, p_value = ttest_1samp(type_data, 0)
        significant = p_value < 0.05
        
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
    
    return stat_results


def plot_betasort_uncertainty_types(df, stat_results=None, save_path=None):
    """
    Create a bar plot showing Betasort correlations across different uncertainty types
    with statistical significance markers
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with correlation data
    stat_results : dict, optional
        Statistical test results
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.axes.Axes : The plot axes
    """
    # Filter data for Betasort model only
    betasort_df = df[df['model'] == 'Betasort']
    
    # Calculate mean correlation by uncertainty type
    uncertainty_corr = betasort_df.groupby('uncertainty_type')['correlation'].mean().reset_index()
    
    # Clean up uncertainty type names for better display
    uncertainty_corr['uncertainty_type_clean'] = uncertainty_corr['uncertainty_type'].map({
        'stim1_uncertainty': 'Stimulus 1\nUncertainty',
        'stim2_uncertainty': 'Stimulus 2\nUncertainty', 
        'pair_roc_uncertainty': 'Pairwise ROC\nUncertainty'
    })
    
    # Sort by correlation value
    uncertainty_corr = uncertainty_corr.sort_values('correlation', ascending=False)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create the bar plot
    ax = sns.barplot(x='uncertainty_type_clean', y='correlation', data=uncertainty_corr, 
                    palette='viridis', errorbar=None)
    
    # Add standard error bars
    uncertainty_sem = betasort_df.groupby('uncertainty_type')['correlation'].sem().reset_index()
    uncertainty_sem['uncertainty_type_clean'] = uncertainty_sem['uncertainty_type'].map({
        'stim1_uncertainty': 'Stimulus 1\nUncertainty',
        'stim2_uncertainty': 'Stimulus 2\nUncertainty', 
        'pair_roc_uncertainty': 'Pairwise ROC\nUncertainty'
    })
    
    for i, uncertainty_type in enumerate(uncertainty_corr['uncertainty_type_clean']):
        # Get original uncertainty type name for lookup
        orig_type = uncertainty_corr.iloc[i]['uncertainty_type']
        sem = uncertainty_sem[uncertainty_sem['uncertainty_type'] == orig_type]['correlation'].values[0]
        mean = uncertainty_corr.iloc[i]['correlation']
        ax.errorbar(i, mean, yerr=sem, fmt='none', color='black', capsize=5, linewidth=2)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add significance markers if we have statistical results
    if stat_results and 'betasort_uncertainty_tests' in stat_results:
        uncertainty_tests = pd.DataFrame(stat_results['betasort_uncertainty_tests'])
        
        for i, row in uncertainty_corr.iterrows():
            uncertainty_type = row['uncertainty_type']
            if uncertainty_type in uncertainty_tests['uncertainty_type'].values:
                test_result = uncertainty_tests[uncertainty_tests['uncertainty_type'] == uncertainty_type].iloc[0]
                if test_result['significant']:
                    # Add star for significance vs zero
                    plt.text(
                        i,
                        row['correlation'] + 0.015,
                        '*',
                        ha='center',
                        va='center',
                        fontsize=20,
                        color='black',
                        weight='bold'
                    )
    
    # Enhance the plot
    plt.title('Betasort: Correlation Between Uncertainty Types and VTE Behavior', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Uncertainty Type', fontsize=16, fontweight='bold')
    plt.ylabel('Average Correlation with VTE', fontsize=16, fontweight='bold')
    
    # Style the axes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim(-0.1, max(uncertainty_corr['correlation']) + 0.05)
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005 if height > 0 else height - 0.015,
            f'{height:.4f}',
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=12,
            fontweight='bold'
        )
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Zero Correlation'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k', 
                  markersize=15, label='p < 0.05 vs. Zero', linestyle='None')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=12)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return ax


def main(save_path=None, output_dir=None):
    """
    Main function to extract data and generate the uncertainty types plot
    
    Parameters:
    -----------
    save_path : str, optional
        Path to betasort results directory
    output_dir : str, optional
        Directory to save plots and results
    """
    print("=" * 60)
    print("BETASORT UNCERTAINTY CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Set default paths
    if save_path is None:
        save_path = paths.betasort_data
    if output_dir is None:
        output_dir = os.path.join(save_path, "uncertainty_analysis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract VTE correlation data from betasort results
        print("Step 1: Extracting VTE correlation data...")
        correlations_df = extract_vte_correlations_from_betasort_results(save_path)
        
        # Save the extracted correlations
        correlations_file = os.path.join(output_dir, "betasort_vte_correlations.csv")
        correlations_df.to_csv(correlations_file, index=False)
        print(f"Saved correlation data to: {correlations_file}")
        
        # Run statistical tests
        print("\nStep 2: Running statistical tests...")
        stat_results = run_betasort_uncertainty_statistical_tests(correlations_df)
        
        # Save statistical results
        if stat_results['betasort_uncertainty_tests']:
            stats_df = pd.DataFrame(stat_results['betasort_uncertainty_tests'])
            stats_file = os.path.join(output_dir, "betasort_uncertainty_statistics.csv")
            stats_df.to_csv(stats_file, index=False)
            print(f"Saved statistical results to: {stats_file}")
        
        # Generate the plot
        print("\nStep 3: Generating uncertainty types plot...")
        plot_path = os.path.join(output_dir, "betasort_uncertainty_types.png")
        plot_betasort_uncertainty_types(correlations_df, stat_results, plot_path)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)
        
        return correlations_df, stat_results
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Run the analysis
    correlations_df, stat_results = main()
    
    # Optional: Show the plot if running interactively
    if correlations_df is not None:
        plt.show()