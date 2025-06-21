import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

from config.paths import paths
from utilities import logging_utils
from utilities import error_types

# pylint: disable=logging-fstring-interpolation
logger = logging_utils.setup_script_logger()

def find_all_rat_files(vte_path: Path) -> List[Tuple[str, Path]]:
    """
    Find all zIdPhis.csv files in the directory structure.
    
    This function walks through your directory structure looking for:
    paths.vte_values/[rat_id]/zIdPhis.csv
    
    Args:
        vte_path: should be paths object
        
    Returns:
        List of tuples where each tuple contains (rat_id, full_path_to_csv_file)
    """

    rat_files = []
    
    # Check if the main directory exists
    if not vte_path.exists():
        logger.error(f"Error: Could not find directory {vte_path} in find_all_rat_files")
        raise error_types.NoPathError(vte_path, "find_all_rat_files")
    
    # Look through each subdirectory for rat data
    for potential_rat_dir in vte_path.iterdir():
        if potential_rat_dir.is_dir():  # Only look at directories, not files
            rat = potential_rat_dir.name
            csv_file = potential_rat_dir / "zIdPhis.csv"
            
            if csv_file.exists():
                rat_files.append((rat, csv_file))
                logger.info(f"Found data file for rat: {rat}")
            else:
                logger.error(f"Warning: Rat {rat} directory exists but no zIdPhis.csv file found")
    
    logger.info(f"\nTotal rats with data files: {len(rat_files)}")
    return rat_files


def load_single_rat_data(file_path: Path, rat: str) -> pd.DataFrame:
    """
    Load and clean data from a single rat's CSV file.
    
    This function handles the common issues you might encounter:
    - Missing columns
    - Empty or invalid data rows
    - Different data types
    
    Args:
        file_path: Path to the CSV file
        rat: Identifier for this rat (used for tracking and error messages)
        
    Returns:
        Clean DataFrame with rat added, or empty DataFrame if file couldn't be loaded
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(data)} rows for rat {rat}")
        
        # Check for columns needed for analysis
        required_columns = ['zIdPhi', 'Length', 'Correct']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Error: Rat {rat} is missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(data.columns)}")
            return pd.DataFrame()  # Return empty DataFrame to signal failure
        
        # Add the rat identifier so we can track which data came from which rat
        data['rat'] = rat
        
        # Remove any rows where key measurements are missing
        initial_row_count = len(data)
        data = data.dropna(subset=['zIdPhi', 'Length', 'Correct'])
        final_row_count = len(data)
        
        if initial_row_count != final_row_count:
            removed_rows = initial_row_count - final_row_count
            logger.warning(f"Removed {removed_rows} rows with missing data for rat {rat}")
        
        logger.info(f"Final clean dataset for rat {rat}: {final_row_count} trials")
        return data
        
    except Exception as error:
        logger.error(f"Error loading file for rat {rat}: {error}")
        return pd.DataFrame()


def create_scatter_plot_for_single_rat(data: pd.DataFrame, rat: str, save_path: Optional[Path] = None):
    """
    Create a scatter plot showing the relationship between zIdPhi and Length for one rat.
    
    This function creates a visualization that shows:
    - Each trial as a point on the plot
    - Correct trials in green, incorrect trials in red
    - The overall correlation between the two measurements
    - Basic statistics about the rat's performance
    
    Args:
        data: DataFrame containing the rat's trial data
        rat: Identifier for the rat (used in plot title)
        save_path: Optional directory to save the plot image
    """
    plt.figure(figsize=(10, 6))
    
    # This allows us to color-code the points differently
    correct_trials = data[data['Correct'] == True]
    incorrect_trials = data[data['Correct'] == False]
    
    # Create the scatter plot with different colors for correct vs incorrect trials
    # Using alpha (transparency) helps when points overlap
    plt.scatter(correct_trials['zIdPhi'], correct_trials['Length'], 
               color='green', alpha=0.6, label='Correct Trials', s=50)
    plt.scatter(incorrect_trials['zIdPhi'], incorrect_trials['Length'], 
               color='red', alpha=0.6, label='Incorrect Trials', s=50)
    
    # Calculate the correlation coefficient between zIdPhi and Length
    # This gives us a quantitative measure of how strongly related these variables are
    correlation = data['zIdPhi'].corr(data['Length'])
    
    # Set up the plot labels and title
    plt.xlabel('zIdPhi', fontsize=18)
    plt.ylabel('Length', fontsize=18)
    plt.title(f'zIdPhi vs Length - Rat {rat}\nCorrelation: {correlation:.3f}', fontsize=24)
    plt.legend()
    plt.grid(True, alpha=0.3)  # Light grid to help read values
    
    # Add a text box with summary statistics
    # This provides quick insights about the rat's behavior
    total_trials = len(data)
    correct_count = len(correct_trials)
    accuracy_percentage = (correct_count / total_trials) * 100
    
    stats_text = f'Total trials: {total_trials}\n'
    stats_text += f'Correct: {correct_count} ({accuracy_percentage:.1f}%)\n'
    stats_text += f'Incorrect: {len(incorrect_trials)} ({100-accuracy_percentage:.1f}%)'
    
    # Position the text box in the upper left corner
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot if a save path was provided
    if save_path:
        save_path.mkdir(exist_ok=True)
        plot_filename = save_path / f"rat_{rat}_scatter.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot for rat {rat} to {plot_filename}")
    
    plt.show()


def create_combined_scatter_plot(all_rat_data: List[pd.DataFrame], save_path: Optional[Path] = None):
    """
    Create a single scatter plot combining data from all rats.
    
    This gives you a population-level view of the relationship between zIdPhi and Length.
    Individual rats' data points are all shown together, which can reveal
    overall patterns that might not be apparent when looking at rats individually.
    
    Args:
        all_rat_data: List of DataFrames, one for each rat
        save_path: Optional directory to save the plot image
    """
    # Combine all the individual rat datasets into one large dataset
    combined_data = pd.concat(all_rat_data, ignore_index=True)
    logger.info(f"Combined dataset contains {len(combined_data)} trials from {len(all_rat_data)} rats")
    
    # Create a larger figure since we're showing more data points
    plt.figure(figsize=(12, 8))
    
    # Again, separate correct from incorrect trials for color coding
    correct_trials = combined_data[combined_data['Correct'] == True]
    incorrect_trials = combined_data[combined_data['Correct'] == False]
    
    # Create scatter plot - using smaller points and more transparency since we have more data
    plt.scatter(correct_trials['zIdPhi'], correct_trials['Length'], 
               color='green', alpha=0.4, label='Correct Trials', s=30)
    plt.scatter(incorrect_trials['zIdPhi'], incorrect_trials['Length'], 
               color='red', alpha=0.4, label='Incorrect Trials', s=30)
    
    # Calculate overall correlation across all rats
    overall_correlation = combined_data['zIdPhi'].corr(combined_data['Length'])
    
    # Set up plot appearance
    plt.xlabel('zIdPhi', fontsize=18)
    plt.ylabel('Length', fontsize=18)
    plt.title(f'zIdPhi vs Length - All Rats Combined\nOverall Correlation: {overall_correlation:.3f}', fontsize=24)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics for the entire dataset
    total_trials = len(combined_data)
    total_correct = len(correct_trials)
    overall_accuracy = (total_correct / total_trials) * 100
    
    stats_text = f'Total rats: {len(all_rat_data)}\n'
    stats_text += f'Total trials: {total_trials}\n'
    stats_text += f'Overall accuracy: {overall_accuracy:.1f}%\n'
    stats_text += f'Correct trials: {total_correct}\n'
    stats_text += f'Incorrect trials: {len(incorrect_trials)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_path.mkdir(exist_ok=True)
        plot_filename = save_path / "all_rats_combined_scatter.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined plot to {plot_filename}")
    
    plt.show()


def calculate_correlation_statistics(all_rat_data: List[pd.DataFrame]) -> Dict:
    """
    Calculate detailed correlation statistics for each rat and overall.
    
    This function provides quantitative analysis to complement the visual plots.
    Understanding the numbers behind the visualizations helps you make
    more informed conclusions about your data.
    
    Args:
        all_rat_data: List of DataFrames, one for each rat
        
    Returns:
        Dictionary containing correlation statistics
    """
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    
    individual_correlations = []
    individual_stats = []
    
    # Calculate statistics for each individual rat
    for rat_data in all_rat_data:
        rat_id = rat_data['rat_id'].iloc[0]  # Get the rat ID from the data
        
        # Calculate correlation between zIdPhi and Length for this rat
        correlation = rat_data['zIdPhi'].corr(rat_data['Length'])
        
        # Calculate basic behavioral statistics
        total_trials = len(rat_data)
        correct_trials = sum(rat_data['Correct'] == True)
        accuracy = (correct_trials / total_trials) * 100
        
        # Store the statistics
        rat_stats = {
            'rat_id': rat_id,
            'correlation': correlation,
            'total_trials': total_trials,
            'accuracy': accuracy
        }
        individual_stats.append(rat_stats)
        
        # Only include non-missing correlations in our analysis
        if not np.isnan(correlation):
            individual_correlations.append(correlation)
        
        # Print individual rat results
        print(f"Rat {rat_id}:")
        print(f"  Correlation coefficient: {correlation:.3f}")
        print(f"  Number of trials: {total_trials}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print()
    
    # Calculate summary statistics across all rats
    if individual_correlations:
        mean_correlation = np.mean(individual_correlations)
        std_correlation = np.std(individual_correlations)
        min_correlation = min(individual_correlations)
        max_correlation = max(individual_correlations)
        
        print("SUMMARY ACROSS ALL RATS:")
        print(f"  Number of rats with valid correlations: {len(individual_correlations)}")
        print(f"  Mean correlation: {mean_correlation:.3f}")
        print(f"  Standard deviation: {std_correlation:.3f}")
        print(f"  Range: {min_correlation:.3f} to {max_correlation:.3f}")
        
        # Calculate overall correlation using all data combined
        combined_data = pd.concat(all_rat_data, ignore_index=True)
        overall_correlation = combined_data['zIdPhi'].corr(combined_data['Length'])
        print(f"  Overall correlation (all data): {overall_correlation:.3f}")
    else:
        print("No valid correlations could be calculated!")
    
    return {
        'individual_stats': individual_stats,
        'individual_correlations': individual_correlations,
        'summary_stats': {
            'mean': mean_correlation if individual_correlations else None,
            'std': std_correlation if individual_correlations else None,
            'min': min_correlation if individual_correlations else None,
            'max': max_correlation if individual_correlations else None,
            'overall': overall_correlation if individual_correlations else None
        }
    }


def analyze_vte_data(base_path: Union[str, Path], 
                    create_individual_plots: bool = True,
                    create_combined_plot: bool = True,
                    save_plots: bool = False,
                    save_directory: Optional[Union[str, Path]] = None):
    """
    Main function that orchestrates the entire analysis process.
    
    This function ties together all the individual steps:
    1. Finding all the data files
    2. Loading and cleaning the data
    3. Creating visualizations
    4. Calculating statistics
    
    Args:
        base_path: Path to your main data directory (contains vte_values folder)
        create_individual_plots: Whether to make separate plots for each rat
        create_combined_plot: Whether to make one plot with all rats together
        save_plots: Whether to save plot images to disk
        save_directory: Where to save plots (if None, saves to base_path/analysis_results)
    """
    logger.info("Starting VTE Data Analysis")
    logger.info("="*50)
    
    # Step 1: Find all the rat data files
    base_path = Path(base_path)
    rat_files = find_all_rat_files(base_path)
    
    if not rat_files:
        logger.error("No data files found! Please check your file path and directory structure.")
        return
    
    # Step 2: Set up where to save plots if requested
    if save_plots:
        if save_directory is None:
            save_directory = base_path / "analysis_results"
        logger.info(f"Plots will be saved to: {save_directory}")
    
    # Step 3: Load and process each rat's data
    successfully_loaded_data = []
    
    for rat_id, file_path in rat_files:
        logger.info(f"\nProcessing rat {rat_id}...")
        
        # Load the data for this rat
        rat_data = load_single_rat_data(file_path, rat_id)
        
        # Only proceed if we successfully loaded data
        if not rat_data.empty:
            successfully_loaded_data.append(rat_data)
            
            # Create individual plot if requested
            if create_individual_plots:
                create_scatter_plot_for_single_rat(rat_data, rat_id, 
                                                 save_directory if save_plots else None)
        else:
            logger.warning(f"Skipping rat {rat_id} due to data loading issues")
    
    # Step 4: Create combined analysis if we have data from multiple rats
    if len(successfully_loaded_data) > 0:
        logger.info(f"\nSuccessfully loaded data from {len(successfully_loaded_data)} rats")
        
        # Create combined visualization if requested
        if create_combined_plot:
            logger.info("Creating combined plot...")
            create_combined_scatter_plot(successfully_loaded_data, 
                                       save_directory if save_plots else None)
        
        # Calculate and display correlation statistics
        correlation_results = calculate_correlation_statistics(successfully_loaded_data)
        
        logger.info(f"\nAnalysis complete! Processed {len(successfully_loaded_data)} rats successfully.")
        
        return correlation_results
    else:
        logger.warning("No data could be loaded successfully. Please check your files and try again.")
        return None


if __name__ == "__main__":
    results = analyze_vte_data(
        base_path=paths.vte_values,
        create_individual_plots=False,  # Make a separate plot for each rat
        create_combined_plot=True,     # Make one plot combining all rats
        save_plots=True,              # Save the plots as image files
        save_directory=None           # Use default save location
    )
