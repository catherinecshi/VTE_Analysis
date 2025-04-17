import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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

# 1. Function to create a plot comparing correlations across different models
def plot_model_comparison(df, save_path=None):
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
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

# 2. Function to visualize Betasort model correlations by uncertainty type
def plot_betasort_uncertainty_types(df, save_path=None):
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
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
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
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

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
        
    plt.show()

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
        
    plt.show()

# 5. Function to create box plots of correlations by model
def plot_correlation_boxplots(df, save_path=None):
    """
    Create box plots showing the distribution of correlations by model
    """
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create the box plot
    ax = sns.boxplot(
        x='model',
        y='correlation',
        data=df,
        palette='viridis'
    )
    
    # Add a swarm plot for individual data points
    sns.swarmplot(
        x='model',
        y='correlation',
        data=df,
        color='black',
        alpha=0.5,
        size=4
    )
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Enhance the plot
    plt.title('Distribution of Correlations by Model', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Correlation', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for easier reading
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

# Main execution function
def main(file_path):
    """
    Main function to generate all plots
    """
    # Load the data
    df = load_data(file_path)
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Models: {', '.join(df['model'].unique())}")
    print(f"Uncertainty types: {', '.join(df['uncertainty_type'].unique())}")
    
    # Create output directory for plots if it doesn't exist
    import os
    output_dir = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison", 'correlation_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save all plots
    plot_model_comparison(df, os.path.join(output_dir, '1_model_comparison.png'))
    plot_betasort_uncertainty_types(df, os.path.join(output_dir, '2_betasort_uncertainty.png'))
    plot_betasort_significance(df, os.path.join(output_dir, '3_betasort_significance.png'))
    plot_correlation_heatmap(df, os.path.join(output_dir, '4_correlation_heatmap.png'))
    plot_correlation_boxplots(df, os.path.join(output_dir, '5_correlation_boxplots.png'))
    
    print(f"All plots have been generated and saved to the '{output_dir}' directory")

# Run the script
if __name__ == "__main__":
    # Replace this with your actual file path
    file_path = os.path.join(helper.BASE_PATH, "processed_data", "model_comparison", "combined_vte_correlations.csv")
    main(file_path)