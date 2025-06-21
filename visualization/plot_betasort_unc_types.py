import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from config.paths import paths

base_dir = paths.model_comparison

# Function to find all vte_uncertainty_data.csv files
def find_csv_files():
    """Find all vte_uncertainty_data.csv files in the directory structure."""
    pattern = os.path.join(base_dir, "*", "vte_uncertainty_data.csv")
    return glob.glob(pattern)

# Function to load and combine data from all rats
def load_all_data(file_paths):
    """Load and combine data from all rats, adding a rat_id column."""
    all_data = []
    
    for file_path in file_paths:
        # Extract rat_id from the path
        rat_id = os.path.basename(os.path.dirname(file_path))
        
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Add rat_id column
            df['rat_id'] = rat_id
            
            # Append to our list
            all_data.append(df)
            
            print(f"Loaded data for rat {rat_id}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading data for rat {rat_id}: {e}")
    
    # Combine all data frames
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined data: {len(combined_data)} rows from {len(all_data)} rats")
        return combined_data
    else:
        raise ValueError("No data could be loaded. Check file paths and formats.")

# New function to calculate correlation for each rat
def calculate_per_rat_correlations(data):
    """
    Calculate Pearson correlations between each uncertainty type and VTE occurrence
    for each rat individually.
    Returns a DataFrame with correlations per rat.
    """
    uncertainty_columns = ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']
    rat_correlations = []
    
    for rat_id in data['rat_id'].unique():
        rat_data = data[data['rat_id'] == rat_id]
        
        # Skip rats with too few data points or no variation in VTE occurrence
        if len(rat_data) < 5 or rat_data['vte_occurred'].nunique() < 2:
            print(f"Skipping rat {rat_id} for correlation calculation (insufficient data variation)")
            continue
        
        for col in uncertainty_columns:
            try:
                # Calculate Pearson correlation and p-value for this rat
                corr, p_value = pearsonr(rat_data[col], rat_data['vte_occurred'])
                rat_correlations.append({
                    'rat_id': rat_id,
                    'uncertainty_type': col,
                    'correlation': corr,
                    'p_value': p_value
                })
            except Exception as e:
                print(f"Error calculating correlation for rat {rat_id}, column {col}: {e}")
    
    return pd.DataFrame(rat_correlations)

# Function to perform correlation analysis
def analyze_correlations(data):
    """
    Calculate Pearson correlations between each uncertainty type and VTE occurrence.
    Returns a DataFrame with correlation coefficients, p-values, and standard errors.
    """
    uncertainty_columns = ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']
    results = []
    
    # Calculate overall correlations (across all rats combined)
    for col in uncertainty_columns:
        # Calculate Pearson correlation and p-value
        corr, p_value = pearsonr(data[col], data['vte_occurred'])
        
        # Initialize result dictionary
        result = {
            'uncertainty_type': col,
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05,  # Flag for significance at α=0.05
            'sem': None  # Placeholder for standard error
        }
        
        results.append(result)
    
    # Calculate correlations for each rat separately
    rat_correlations = calculate_per_rat_correlations(data)
    
    # Calculate SEM for each uncertainty type from the per-rat correlations
    if not rat_correlations.empty:
        for col in uncertainty_columns:
            col_correlations = rat_correlations[rat_correlations['uncertainty_type'] == col]['correlation']
            if len(col_correlations) > 1:  # Need at least 2 rats for SEM
                sem = col_correlations.std() / np.sqrt(len(col_correlations))
                # Update the corresponding result entry with SEM
                for result in results:
                    if result['uncertainty_type'] == col:
                        result['sem'] = sem
                        # Store the number of rats for reference
                        result['n_rats'] = len(col_correlations)
    
    return pd.DataFrame(results)

# Function to create correlation plots
def plot_correlations(data, results):
    """Create visualization for the correlation analysis."""
    # Set up the figure for all plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    uncertainty_columns = ['stim1_uncertainty', 'stim2_uncertainty', 'pair_roc_uncertainty']
    
    # Create a scatter plot for each uncertainty type
    for i, col in enumerate(uncertainty_columns):
        ax = axes[i]
        
        # Get correlation and p-value for this uncertainty type
        corr = results.loc[results['uncertainty_type'] == col, 'correlation'].values[0]
        p_val = results.loc[results['uncertainty_type'] == col, 'p_value'].values[0]
        
        # Create a more descriptive title for display
        display_name = col.replace('_', ' ').title()
        display_name = display_name.replace('Stim1', 'Stimulus 1')
        display_name = display_name.replace('Stim2', 'Stimulus 2')
        display_name = display_name.replace('Pair Roc', 'Pair ROC')
        
        # Plot with jitter for better visualization of binary outcome
        x = data[col]
        y = data['vte_occurred'] + np.random.normal(0, 0.05, size=len(data))
        
        ax.scatter(x, y, alpha=0.3)
        
        # Add a regression line
        sns.regplot(x=col, y='vte_occurred', data=data, ax=ax, 
                   scatter=False, line_kws={'color': 'red'})
        
        # Set labels and title
        ax.set_xlabel(display_name)
        ax.set_ylabel('VTE Occurrence')
        ax.set_title(f"{display_name}\nr = {corr:.3f}, p = {p_val:.4f}")
        
        # Set y-axis to show binary nature more clearly
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No VTE', 'VTE'])
    
    plt.tight_layout()
    
    # Create a summary bar plot
    plt.figure(figsize=(10, 6))
    
    num_types = len(results)
    viridis_colors = sns.color_palette("viridis", num_types)
    
    # Extract x-values (labels), y-values (correlations), and error values (SEMs)
    labels = results['uncertainty_type'].apply(lambda x: x.replace('_', ' ').title().replace('Stim1', 'Stimulus 1').replace('Stim2', 'Stimulus 2').replace('Pair Roc', 'Pair ROC'))
    correlations = results['correlation']
    
    # Get SEMs, default to 0 if not available for any uncertainty type
    errors = results['sem'].fillna(0).values
    
    # Create the bars with the viridis colors and error bars
    bars = plt.bar(
        labels,
        correlations,
        color=viridis_colors,
        yerr=errors,  # Add error bars using SEM
        capsize=10,   # Add caps to the error bars
        ecolor='black',  # Color of error bars
        error_kw={'linewidth': 2, 'capthick': 2}  # Style of error bars
    )
    
    # Add labels and title
    plt.xlabel('Uncertainty Type', fontsize=24)
    plt.ylabel('Pearson Correlation Coefficient', fontsize=24)
    plt.title('Correlation Between Uncertainty Types and VTE Occurrence', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.ylim(-0.05, 0.2)  # Set y-axis limits for better visualization
    plt.axhline(y=0, color='black', linestyle='-')  # Add a reference line at y=0
    
    plt.tight_layout()
    plt.savefig('uncertainty_vte_correlations.png')
    plt.show()
    
    return fig

# Main function to run the analysis
def main():
    """Main function to run the analysis pipeline."""
    try:
        # Find CSV files
        csv_files = find_csv_files()
        print(f"Found {len(csv_files)} CSV files")
        
        if not csv_files:
            print("No files found. Please check the path structure.")
            return
        
        # Load and combine data
        all_data = load_all_data(csv_files)
        
        # Basic data inspection
        print("\nData summary:")
        print(all_data.describe())
        print("\nMissing values:")
        print(all_data.isnull().sum())
        
        # Perform correlation analysis
        results = analyze_correlations(all_data)
        
        # Print results
        print("\nCorrelation Results:")
        for _, row in results.iterrows():
            uncertainty_type = row['uncertainty_type'].replace('_', ' ').title()
            uncertainty_type = uncertainty_type.replace('Stim1', 'Stimulus 1')
            uncertainty_type = uncertainty_type.replace('Stim2', 'Stimulus 2')
            uncertainty_type = uncertainty_type.replace('Pair Roc', 'Pair ROC')
            
            print(f"{uncertainty_type}:")
            print(f"  Correlation: {row['correlation']:.3f}")
            print(f"  P-value: {row['p_value']:.4f}")
            print(f"  Significant: {'Yes' if row['significant'] else 'No'}")
            if 'sem' in row and pd.notna(row['sem']):
                print(f"  SEM: {row['sem']:.4f}")
            if 'n_rats' in row and pd.notna(row['n_rats']):
                print(f"  Number of rats: {int(row['n_rats'])}")
            print()
        
        # Create and save plots
        plot_correlations(all_data, results)
        print("Analysis complete. Plots have been displayed and saved.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()