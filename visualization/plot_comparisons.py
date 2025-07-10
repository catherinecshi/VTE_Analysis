import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config.paths import paths

def load_and_plot_individual_results(comparison_output_path, exclude_models=["ValueTransfer"]):
    """
    Load and plot results for individual rats, excluding specified models
    
    Parameters:
        - comparison_output_path: Base directory containing results for all rats
        - exclude_models: List of model names to exclude from plots
    """
    
    # Find all rat directories
    rat_dirs = [d for d in os.listdir(comparison_output_path) 
                if os.path.isdir(os.path.join(comparison_output_path, d)) 
                and d not in ['combined_results', 'optimization_results']]
    
    for rat in rat_dirs:
        rat_dir = os.path.join(comparison_output_path, rat)
        print(f"Processing plots for {rat}")
        
        # 1. Plot overall performance comparison
        overall_perf_file = os.path.join(rat_dir, "overall_performance.csv")
        if os.path.exists(overall_perf_file):
            df = pd.read_csv(overall_perf_file)
            # Filter out excluded models
            df_filtered = df[~df['model'].isin(exclude_models)]
            
            if len(df_filtered) > 0:
                plot_model_comparison_filtered(df_filtered, rat_dir, f"{rat}_model_comparison_filtered.png")
        
        # 2. Plot TI comparison
        ti_comparison_file = os.path.join(rat_dir, "ti_comparison.csv")
        if os.path.exists(ti_comparison_file):
            df = pd.read_csv(ti_comparison_file)
            # Create summary for plotting (filter out summary rows and excluded models)
            summary_df = df[(df['stim1'] == -1) & (~df['model'].isin(exclude_models))]
            
            if len(summary_df) > 0 and 'error' in summary_df.columns:
                summary_df = summary_df.sort_values('error')
                plot_ti_comparison_filtered(summary_df, rat_dir, f"{rat}_ti_comparison_filtered.png")
        
        # 3. Plot VTE correlations
        vte_corr_file = os.path.join(rat_dir, "vte_uncertainty_correlations.csv")
        if os.path.exists(vte_corr_file):
            df = pd.read_csv(vte_corr_file)
            # Filter for overall correlations and excluded models
            summary_df = df[(df['scope'] == 'overall') & (~df['model'].isin(exclude_models))]
            
            if len(summary_df) > 0:
                summary_df = summary_df.sort_values('correlation', ascending=False)
                plot_vte_correlations_filtered(summary_df, rat_dir, f"{rat}_vte_correlation_filtered.png")

def plot_model_comparison_filtered(df, output_dir, filename):
    """
    Create bar plot comparing model performance with filtered data
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Sort models by performance
    sorted_df = df.sort_values("mean_match_rate", ascending=False)
    
    # Create bar plot with error bars
    bar_plot = sns.barplot(
        x="model", 
        y="mean_match_rate", 
        data=sorted_df,
        palette="viridis"
    )
    
    # Add error bars manually if std_error column exists
    if 'std_error' in sorted_df.columns:
        for i, row in sorted_df.reset_index(drop=True).iterrows():
            bar_plot.errorbar(
                x=i, 
                y=row["mean_match_rate"],
                yerr=row["std_error"],
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
    plt.ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_ti_comparison_filtered(summary_df, output_dir, filename):
    """
    Create bar plot comparing model TI performance with filtered data
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create bar plot
    bar_plot = sns.barplot(
        x="model", 
        y="error", 
        data=summary_df,
        palette="viridis"
    )
    
    # Add error bars if available
    if 'error_sem' in summary_df.columns:
        for i, row in summary_df.reset_index(drop=True).iterrows():
            if not pd.isna(row.get("error_sem", np.nan)):
                bar_plot.errorbar(
                    x=i, 
                    y=row["error"],
                    yerr=row["error_sem"],
                    color='black',
                    capsize=5,
                    fmt='none'
                )
    
    # Customize plot
    plt.title("Model Comparison: Transitive Inference Performance", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Mean Absolute Error", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_vte_correlations_filtered(summary_df, output_dir, filename):
    """
    Create bar plot comparing VTE-uncertainty correlations with filtered data
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create bar plot
    bar_plot = sns.barplot(
        x="model", 
        y="correlation", 
        data=summary_df,
        palette="viridis"
    )
    
    # Add significance markers if available
    if 'significant' in summary_df.columns:
        for i, row in summary_df.reset_index(drop=True).iterrows():
            if row.get("significant", False):
                bar_height = row["correlation"]
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
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def aggregate_ti_results(comparison_output_path, exclude_models=["ValueTransfer"]):
    """
    Aggregate TI results across all rats, excluding specified models
    
    Parameters:
        - comparison_output_path: Base directory containing results for all rats
        - exclude_models: List of model names to exclude
        
    Returns:
        - combined_df: DataFrame with combined TI results
    """
    # Find all TI comparison CSV files
    ti_files = glob.glob(os.path.join(comparison_output_path, "*", "ti_comparison.csv"))
    
    # Initialize list to store DataFrames
    all_ti_dfs = []
    
    # Process each file
    for file_path in ti_files:
        # Extract rat ID from path
        rat = os.path.basename(os.path.dirname(file_path))
        
        # Load TI data
        try:
            df = pd.read_csv(file_path)
            df['rat'] = rat  # Add rat identifier
            all_ti_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Combine all DataFrames
    if all_ti_dfs:
        combined_df = pd.concat(all_ti_dfs, ignore_index=True)
        
        # Filter out excluded models
        combined_df = combined_df[~combined_df['model'].isin(exclude_models)]
        
        # Save combined results
        combined_df.to_csv(os.path.join(comparison_output_path, "combined_ti_comparison.csv"), index=False)
        
        return combined_df
    else:
        print("No TI comparison files found")
        return None

def plot_combined_ti_results(combined_df, output_path):
    """
    Create visualizations of combined TI results across rats
    
    Parameters:
        - combined_df: DataFrame with combined TI results
        - output_path: Directory to save plots
    """
    
    # Filter for summary rows (where stim1 == -1, these contain mean error per model per rat)
    summary_df = combined_df[combined_df['stim1'] == -1].copy()
    
    if len(summary_df) == 0:
        print("No TI summary data found for plotting")
        return
    
    # Convert error to match rate (assuming error is normalized between 0-1)
    # Match rate = 1 - error, so lower error = higher match rate
    summary_df['ti_match_rate'] = 1 - summary_df['error']
    
    # Calculate aggregated statistics by model
    model_stats = summary_df.groupby('model').agg({
        'ti_match_rate': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten MultiIndex columns
    model_stats.columns = ['model', 'mean_rate', 'std_rate', 'count']
    
    # Calculate standard error
    model_stats['sem_rate'] = model_stats['std_rate'] / np.sqrt(model_stats['count'])
    
    # Sort by mean rate (descending - higher is better)
    model_stats = model_stats.sort_values('mean_rate', ascending=False)
    
    # Create boxplot with stripplot overlay (same style as performance_distribution_filtered)
    summary_df_pct = summary_df.copy()
    summary_df_pct['ti_match_rate'] = summary_df_pct['ti_match_rate'] * 100
    
    plt.figure(figsize=(12, 8))
    
    boxplot = sns.boxplot(
        x='model', 
        y='ti_match_rate', 
        data=summary_df_pct,
        palette="viridis_r",
        order=model_stats['model']  # Use same order as calculated above
    )
    
    # Add individual points
    sns.stripplot(
        x='model', 
        y='ti_match_rate', 
        data=summary_df_pct,
        color='black',
        alpha=0.5,
        jitter=True,
        order=model_stats['model']  # Use same order as boxplot
    )
    
    # Add chance level line (50%)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot (same style as performance_distribution_filtered)
    plt.title("TI Match Rate Distribution Across Rats", fontsize=30)
    plt.xlabel("Model", fontsize=24)
    plt.ylabel("TI Match Rate (%)", fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ti_match_rate_distribution_filtered.png"), dpi=300)
    plt.close()
    
    # Also create a bar plot for summary statistics
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Convert to percentages
    model_stats_pct = model_stats.copy()
    model_stats_pct['mean_rate'] = model_stats_pct['mean_rate'] * 100
    model_stats_pct['sem_rate'] = model_stats_pct['sem_rate'] * 100
    
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
            fmt='none'
        )
    
    # Add chance level line (50%)
    plt.axhline(y=50, color='k', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot
    plt.title("TI Match Rate Across All Rats", fontsize=24)
    plt.xlabel("Model", fontsize=20)
    plt.ylabel("Mean TI Match Rate (%)", fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(40, 100)  # Same range as performance plots
    
    # Add value labels on bars
    for i, bar in enumerate(bar_plot.patches):
        bar_value = model_stats_pct.iloc[i]["mean_rate"]
        bar_plot.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{bar_value:.1f}%",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "combined_ti_match_rate_filtered.png"), dpi=300)
    plt.close()
    
    # Create heatmap showing TI match rate by model and rat
    pivot_df = summary_df.pivot(index='rat', columns='model', values='ti_match_rate') * 100
    
    plt.figure(figsize=(14, 10))
    
    # Sort models by average performance
    model_order = model_stats['model'].tolist()
    pivot_df = pivot_df[model_order]
    
    ax = plt.subplot(111)
    heatmap = sns.heatmap(
        pivot_df,
        annot=False,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={'label': 'TI Match Rate (%)'},
        ax=ax
    )
    
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    plt.title("TI Match Rate by Model and Rat", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ti_match_rate_heatmap_filtered.png"), dpi=300)
    plt.close()

def load_and_plot_combined_results(comparison_output_path, exclude_models=["ValueTransfer"]):
    """
    Load and plot combined results across all rats, excluding specified models
    """
    
    # 1. Load and plot combined performance
    combined_perf_file = os.path.join(comparison_output_path, "combined_performance.csv")
    if os.path.exists(combined_perf_file):
        print("Plotting combined performance results...")
        df = pd.read_csv(combined_perf_file)
        # Filter out excluded models
        df_filtered = df[~df['model'].isin(exclude_models)]
        
        if len(df_filtered) > 0:
            plot_combined_performance_filtered(df_filtered, comparison_output_path)
    
    # 2. Load and plot combined TI results
    print("Aggregating and plotting combined TI results...")
    combined_ti_df = aggregate_ti_results(comparison_output_path, exclude_models)
    if combined_ti_df is not None:
        plot_combined_ti_results(combined_ti_df, comparison_output_path)
    
    # 3. Load and plot combined VTE correlations
    combined_vte_file = os.path.join(comparison_output_path, "combined_vte_correlations.csv")
    if os.path.exists(combined_vte_file):
        print("Plotting combined VTE correlation results...")
        df = pd.read_csv(combined_vte_file)
        # Filter out excluded models
        df_filtered = df[~df['model'].isin(exclude_models)]
        
        if len(df_filtered) > 0:
            plot_combined_vte_filtered(df_filtered, comparison_output_path)

def plot_combined_performance_filtered(combined_df, output_path):
    """
    Create visualizations of combined results across rats (filtered)
    """
    
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
    
    # Convert to percentages
    model_stats_pct = model_stats.copy()
    model_stats_pct['mean_rate'] = model_stats_pct['mean_rate'] * 100
    model_stats_pct['sem_rate'] = model_stats_pct['sem_rate'] * 100
    
    # Create bar plot
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
            fmt='none'
        )
    
    # Add chance level line (50%)
    plt.axhline(y=50, color='k', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot
    plt.title("Model Comparison Across All Rats", fontsize=24)
    plt.xlabel("Model", fontsize=20)
    plt.ylabel("Mean Match Rate (%)", fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(40, 100)
    
    # Add value labels on bars
    for i, bar in enumerate(bar_plot.patches):
        bar_value = model_stats_pct.iloc[i]["mean_rate"]
        bar_plot.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{bar_value:.1f}%",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "combined_performance_filtered.png"), dpi=300)
    plt.close()
    
    # Create boxplot
    combined_df_pct = combined_df.copy()
    combined_df_pct['mean_match_rate'] = combined_df_pct['mean_match_rate'] * 100
    
    plt.figure(figsize=(12, 8))
    
    boxplot = sns.boxplot(
        x='model', 
        y='mean_match_rate', 
        data=combined_df_pct,
        palette="viridis_r",
        order=model_stats['model']
    )
    
    # Add individual points
    sns.stripplot(
        x='model', 
        y='mean_match_rate', 
        data=combined_df_pct,
        color='black',
        alpha=0.5,
        jitter=True,
        order=model_stats['model']
    )
    
    # Add chance level line
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Chance Level')
    
    # Customize plot
    plt.title("Model Performance Distribution Across Rats", fontsize=30)
    plt.xlabel("Model", fontsize=24)
    plt.ylabel("Match Rate (%)", fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "performance_distribution_filtered.png"), dpi=300)
    plt.close()
    
    # Create heatmap
    pivot_df = combined_df.pivot(index='rat', columns='model', values='mean_match_rate') * 100
    
    plt.figure(figsize=(14, 10))
    
    # Sort models by average performance
    model_order = model_stats['model'].tolist()
    pivot_df = pivot_df[model_order]
    
    ax = plt.subplot(111)
    heatmap = sns.heatmap(
        pivot_df,
        annot=False,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={'label': 'Match Rate (%)'},
        ax=ax
    )
    
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    plt.title("Performance by Model and Rat", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "performance_heatmap_filtered.png"), dpi=300)
    plt.close()

def plot_combined_vte_filtered(combined_df, output_path):
    """
    Create visualizations of combined VTE correlation results (filtered)
    """
    
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
    
    # Sort by absolute mean correlation
    model_stats['abs_corr'] = abs(model_stats['mean_corr'])
    model_stats = model_stats.sort_values('abs_corr', ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    bar_plot = sns.barplot(
        x='model', 
        y='mean_corr', 
        data=model_stats,
        palette="viridis_r"
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
    plt.savefig(os.path.join(output_path, "combined_vte_correlation_filtered.png"), dpi=300)
    plt.close()
    
    # Create boxplot
    plt.figure(figsize=(12, 8))
    
    boxplot = sns.boxplot(
        x='model', 
        y='correlation', 
        data=overall_df,
        palette="viridis_r",
        order=model_stats['model']
    )
    
    # Add individual points
    sns.stripplot(
        x='model', 
        y='correlation', 
        data=overall_df,
        color='black',
        alpha=0.5,
        jitter=True,
        order=model_stats['model']
    )
    
    # Add zero line
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # Customize plot
    plt.title("VTE-Uncertainty Correlation Distribution Across Rats", fontsize=30)
    plt.xlabel("Model", fontsize=24)
    plt.ylabel("Correlation (r)", fontsize=24)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "vte_correlation_distribution_filtered.png"), dpi=300)
    plt.close()

def main():
    """
    Main function to load saved data and create filtered plots
    """
    # Set the path to your model comparison results
    comparison_output_path = paths.model_comparison  # Adjust this path as needed
    
    print("Loading saved model comparison data and creating filtered plots...")
    print(f"Looking for data in: {comparison_output_path}")
    
    # Models to exclude from plots
    exclude_models = ["ValueTransfer"]
    
    # Plot individual rat results
    print("\nCreating filtered plots for individual rats...")
    load_and_plot_individual_results(comparison_output_path, exclude_models)
    
    # Plot combined results
    print("\nCreating filtered plots for combined results...")
    load_and_plot_combined_results(comparison_output_path, exclude_models)
    
    print(f"\nFiltered plots saved to: {comparison_output_path}")
    print("Plot files created:")
    print("- Individual rats: *_model_comparison_filtered.png, *_ti_comparison_filtered.png, *_vte_correlation_filtered.png")
    print("- Combined Performance: combined_performance_filtered.png, performance_distribution_filtered.png, performance_heatmap_filtered.png")
    print("- Combined TI: ti_match_rate_distribution_filtered.png, combined_ti_match_rate_filtered.png, ti_match_rate_heatmap_filtered.png")
    print("- Combined VTE: combined_vte_correlation_filtered.png, vte_correlation_distribution_filtered.png")

if __name__ == "__main__":
    main()