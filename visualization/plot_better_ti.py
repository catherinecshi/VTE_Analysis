import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.paths import paths

def convert_trial_type_to_letters(trial_type_str):
    """Convert trial type from numbers to letters (e.g., '0-1' -> 'A-B')"""
    # Mapping from numbers to letters
    num_to_letter = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E'}
    
    # Split the trial type and convert each number
    parts = trial_type_str.split('-')
    letter_parts = [num_to_letter.get(part, part) for part in parts]
    return '-'.join(letter_parts)

def plot_ti_comparison_enhanced(df, save_path=None):
    """Create bar plot with bigger fonts and letter naming"""
    
    # Convert trial types to letters
    df['trial_type_letters'] = df['trial_type'].apply(convert_trial_type_to_letters)
    
    # Define the desired order
    desired_order = ['A-B', 'B-C', 'C-D', 'D-E', 'B-D', 'A-E']
    
    # Filter to only include the desired trial types and order them
    df_filtered = df[df['trial_type_letters'].isin(desired_order)].copy()
    df_filtered['order'] = df_filtered['trial_type_letters'].map({trial: i for i, trial in enumerate(desired_order)})
    df_filtered = df_filtered.sort_values('order')
    
    # Extract data for plotting
    trial_type_labels = df_filtered['trial_type_letters'].values
    model_test_averages = df_filtered['model_test_performance'].values
    model_regular_averages = df_filtered['model_regular_performance'].values
    rat_averages = df_filtered['rat_performance'].values
    
    # Create the plot with original styling but bigger fonts
    plt.figure(figsize=(15, 8))
    x_pos = np.arange(len(trial_type_labels))
    width = 0.25
    
    # Original bar colors and styling
    bars1 = plt.bar(x_pos - width, model_test_averages, width, label='Model (VTE)', color='skyblue')
    bars2 = plt.bar(x_pos, model_regular_averages, width, label='Model (Uncertainty)', color='lightgreen')
    bars3 = plt.bar(x_pos + width, rat_averages, width, label='Rat', color='lightcoral')
    
    # Bigger fonts but original styling
    plt.xlabel('Trial Types', fontsize=20)
    plt.ylabel('Performance/Accuracy', fontsize=20)
    plt.title('Transitive Inference: Both Models vs Rat Performance', fontsize=24)
    plt.xticks(x_pos, trial_type_labels, fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Add value labels on bars with bigger font
    for i, (test_val, regular_val, rat_val) in enumerate(zip(model_test_averages, model_regular_averages, rat_averages)):
        plt.text(i - width, test_val + 0.02, f'{test_val:.3f}', 
                ha='center', va='bottom', fontsize=10)
        plt.text(i, regular_val + 0.02, f'{regular_val:.3f}', 
                ha='center', va='bottom', fontsize=10)
        plt.text(i + width, rat_val + 0.02, f'{rat_val:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced plot saved to {save_path}")
    
    plt.show()
    
    return df_filtered

def print_enhanced_summary(df_filtered):
    """Print summary statistics with letter naming"""
    print("\n" + "=" * 60)
    print("ENHANCED SUMMARY STATISTICS")
    print("=" * 60)
    
    model_test_averages = df_filtered['model_test_performance'].values
    model_regular_averages = df_filtered['model_regular_performance'].values
    rat_averages = df_filtered['rat_performance'].values
    
    print(f"Trial types analyzed: {', '.join(df_filtered['trial_type_letters'].values)}")
    print(f"Number of trial types: {len(df_filtered)}")
    print()
    print("PERFORMANCE AVERAGES:")
    print(f"  Model (Test):    {np.mean(model_test_averages):.3f} ± {np.std(model_test_averages):.3f}")
    print(f"  Model (Regular): {np.mean(model_regular_averages):.3f} ± {np.std(model_regular_averages):.3f}")
    print(f"  Rat:             {np.mean(rat_averages):.3f} ± {np.std(rat_averages):.3f}")
    print()
    print("DIFFERENCES:")
    test_minus_rat = model_test_averages - rat_averages
    regular_minus_rat = model_regular_averages - rat_averages
    test_minus_regular = model_test_averages - model_regular_averages
    
    print(f"  Test - Rat:      {np.mean(test_minus_rat):.3f} ± {np.std(test_minus_rat):.3f}")
    print(f"  Regular - Rat:   {np.mean(regular_minus_rat):.3f} ± {np.std(regular_minus_rat):.3f}")
    print(f"  Test - Regular:  {np.mean(test_minus_regular):.3f} ± {np.std(test_minus_regular):.3f}")
    print()
    print("CORRELATIONS:")
    if len(model_test_averages) > 1:
        print(f"  Test vs Rat:     {np.corrcoef(model_test_averages, rat_averages)[0,1]:.3f}")
        print(f"  Regular vs Rat:  {np.corrcoef(model_regular_averages, rat_averages)[0,1]:.3f}")
        print(f"  Test vs Regular: {np.corrcoef(model_test_averages, model_regular_averages)[0,1]:.3f}")
    else:
        print("  (Not enough data points for correlation)")
    
    print()
    print("INDIVIDUAL TRIAL TYPE RESULTS:")
    for _, row in df_filtered.iterrows():
        trial = row['trial_type_letters']
        test_perf = row['model_test_performance']
        regular_perf = row['model_regular_performance']
        rat_perf = row['rat_performance']
        print(f"  {trial:4s}: Test={test_perf:.3f}, Regular={regular_perf:.3f}, Rat={rat_perf:.3f}")

def main():
    """Main function to load data and create enhanced plots"""
    # Define paths
    input_dir = os.path.join(paths.betasort_data, "transitive_inference_analysis")
    results_path = os.path.join(input_dir, "ti_comparison_results.csv")
    
    # Check if data exists
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Make sure you've run the original analysis script first.")
        return
    
    # Load the data
    print(f"Loading data from: {results_path}")
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} trial types")
    print(f"Available trial types: {', '.join(df['trial_type'].values)}")
    
    # Create enhanced plot
    output_path = os.path.join(input_dir, "ti_enhanced_comparison.png")
    df_filtered = plot_ti_comparison_enhanced(df, save_path=output_path)
    
    # Print enhanced summary
    print_enhanced_summary(df_filtered)
    
    # Save the filtered results with letter naming
    enhanced_results_path = os.path.join(input_dir, "ti_enhanced_results.csv")
    df_filtered.to_csv(enhanced_results_path, index=False)
    print(f"\nEnhanced results saved to: {enhanced_results_path}")

if __name__ == "__main__":
    main()