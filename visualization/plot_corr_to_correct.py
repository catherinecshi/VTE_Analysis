import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pointbiserialr, pearsonr
from typing import Tuple, Dict, Optional

from config.paths import paths
from utilities import logging_utils

logger = logging_utils.setup_script_logger()

def load_all_rat_data(base_path: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load behavioral data from all rats across multiple testing days.
    
    This function navigates a nested directory structure where each rat has
    multiple days of testing data. The structure follows the pattern:
    base_path > {rat} > {day} > zIdPhi_day_{day}.csv
    
    For each rat, data from all available days are combined into a single
    dataframe while preserving day information for temporal analysis.
    
    Args:
        base_path (Path): Path object pointing to the directory containing rat folders
        
    Returns:
        Tuple containing:
            - pd.DataFrame: Combined data from all rats and days with 'Rat' and 'Day' columns
            - Dict[str, pd.DataFrame]: Individual rat data (all days combined) keyed by rat name
    """
    logger.info(f"Starting to load multi-day data from all rats in: {base_path}")
    
    # Initialize storage for our data across all rats and days
    all_data_frames = []
    individual_rat_data = {}
    
    # Find all potential rat directories (any subdirectory could be a rat)
    potential_rat_directories = [directory for directory in base_path.iterdir() if directory.is_dir()]
    
    if not potential_rat_directories:
        logger.warning(f"No subdirectories found in {base_path}")
        return pd.DataFrame(), {}
    
    logger.info(f"Found {len(potential_rat_directories)} potential rat directories: {[d.name for d in potential_rat_directories]}")
    
    # Process each potential rat directory
    rats_with_data = 0
    for rat_directory in potential_rat_directories:
        rat_name = rat_directory.name
        logger.info(f"Processing rat directory: {rat_name}")
        
        # Find all day subdirectories within this rat's folder
        day_directories = [day_dir for day_dir in rat_directory.iterdir() if day_dir.is_dir()]
        
        if not day_directories:
            logger.warning(f"No day subdirectories found for rat {rat_name}")
            continue
        
        logger.info(f"Found {len(day_directories)} day directories for rat {rat_name}: {[d.name for d in day_directories]}")
        
        # Storage for this specific rat's data across all days
        rat_daily_dataframes = []
        days_processed = 0
        
        # Process each day's data for this rat
        for day_directory in day_directories:
            day_name = day_directory.name
            
            # Construct the expected filename pattern: zIdPhi_day_{day}.csv
            # This naming convention helps ensure we're loading the correct files
            expected_filename = f"zIdPhi_day_{day_name}.csv"
            csv_file_path = day_directory / expected_filename
            
            # Check if the expected file exists for this day
            if not csv_file_path.exists():
                logger.warning(f"Expected file {expected_filename} not found in {day_directory}")
                # Also check for alternative naming patterns that might exist
                alternative_files = list(day_directory.glob("*.csv"))
                if alternative_files:
                    logger.info(f"Alternative CSV files found in {day_directory}: {[f.name for f in alternative_files]}")
                continue
            
            try:
                # Load the CSV data for this specific day
                daily_dataframe = pd.read_csv(csv_file_path)
                logger.info(f"Loaded data for rat {rat_name}, day {day_name}: {len(daily_dataframe)} trials")
                
                # Add identifying information to track which rat and day each trial belongs to
                # This metadata is crucial for longitudinal analysis and tracking learning curves
                daily_dataframe['Rat'] = rat_name
                daily_dataframe['Day'] = day_name
                
                # Validate that all required columns are present in this day's data
                # This check helps identify data quality issues early in the pipeline
                required_columns = ['Correct', 'Length', 'zIdPhi', 'VTE']
                missing_columns = [col for col in required_columns if col not in daily_dataframe.columns]
                
                if missing_columns:
                    logger.warning(f"Rat {rat_name}, day {day_name} missing required columns: {missing_columns}")
                    continue
                
                # Convert boolean columns to proper Python boolean types
                # This standardization ensures consistent data types across all days and rats
                daily_dataframe['Correct'] = daily_dataframe['Correct'].astype(bool)
                daily_dataframe['VTE'] = daily_dataframe['VTE'].astype(bool)
                
                # Add this day's data to the rat's collection
                rat_daily_dataframes.append(daily_dataframe)
                days_processed += 1
                
                logger.info(f"Successfully processed rat {rat_name}, day {day_name}")
                
            except Exception as error:
                logger.error(f"Error loading data for rat {rat_name}, day {day_name}: {str(error)}")
                continue
        
        # Combine all days for this rat if we successfully loaded any data
        if rat_daily_dataframes:
            # Concatenate all days into a single dataframe for this rat
            # This creates a comprehensive dataset showing this rat's performance across time
            combined_rat_dataframe = pd.concat(rat_daily_dataframes, ignore_index=True)
            
            # Store the rat's complete dataset
            individual_rat_data[rat_name] = combined_rat_dataframe
            all_data_frames.append(combined_rat_dataframe)
            rats_with_data += 1
            
            logger.info(f"Combined data for rat {rat_name}: {len(combined_rat_dataframe)} total trials across {days_processed} days")
        else:
            logger.warning(f"No valid data found for rat {rat_name}")
    
    # Create the master dataset combining all rats and all days
    if all_data_frames:
        combined_dataframe = pd.concat(all_data_frames, ignore_index=True)
        total_days = combined_dataframe['Day'].nunique() if 'Day' in combined_dataframe.columns else 0
        logger.info(f"Master dataset created: {len(combined_dataframe)} total trials from {rats_with_data} rats across {total_days} unique days")
        return combined_dataframe, individual_rat_data
    else:
        logger.error("No valid data loaded from any rats or days")
        return pd.DataFrame(), {}


def compute_behavioral_correlations(combined_data: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute correlations between correct responses and other behavioral measures.
    
    This function calculates appropriate correlation coefficients based on the
    variable types involved. For binary-continuous relationships, we use point-biserial
    correlation. For binary-binary relationships, we use the phi coefficient.
    
    Args:
        combined_data (pd.DataFrame): Combined behavioral data from all rats
        
    Returns:
        Dict containing correlation results with statistics and metadata for each comparison
    """
    logger.info("Computing behavioral correlations...")
    
    if combined_data.empty:
        logger.error("No data available for correlation analysis")
        return {}
    
    correlation_results = {}
    
    # Remove any rows with missing values to ensure clean correlation calculations
    # This is essential because correlation coefficients cannot handle missing data
    clean_data = combined_data.dropna(subset=['Correct', 'Length', 'zIdPhi', 'VTE'])
    logger.info(f"Using {len(clean_data)} complete cases for correlation analysis")
    
    # Convert boolean Correct responses to numeric for correlation calculation
    # This transformation is necessary for the correlation functions to work properly
    correct_numeric = clean_data['Correct'].astype(int)
    
    # Correct vs Length: Point-biserial correlation
    # Point-biserial is used when one variable is binary and the other is continuous
    try:
        correlation_coeff, p_value = pointbiserialr(correct_numeric, clean_data['Length'])
        correlation_results['Correct_vs_Length'] = {
            'correlation': correlation_coeff,
            'p_value': p_value,
            'type': 'point-biserial',
            'n_observations': len(clean_data),
            'interpretation': _interpret_correlation_strength(abs(correlation_coeff))
        }
        logger.info(f"Correct vs Length: r = {correlation_coeff:.3f}, p = {p_value:.3f}")
    except Exception as error:
        logger.error(f"Error computing Correct vs Length correlation: {error}")
    
    # Correct vs zIdPhi: Point-biserial correlation
    # zIdPhi represents some measure of behavioral certainty or decision-making
    try:
        correlation_coeff, p_value = pointbiserialr(correct_numeric, clean_data['zIdPhi'])
        correlation_results['Correct_vs_zIdPhi'] = {
            'correlation': correlation_coeff,
            'p_value': p_value,
            'type': 'point-biserial',
            'n_observations': len(clean_data),
            'interpretation': _interpret_correlation_strength(abs(correlation_coeff))
        }
        logger.info(f"Correct vs zIdPhi: r = {correlation_coeff:.3f}, p = {p_value:.3f}")
    except Exception as error:
        logger.error(f"Error computing Correct vs zIdPhi correlation: {error}")
    
    # Correct vs VTE: Phi coefficient (correlation between two binary variables)
    # VTE likely represents some binary behavioral measure like hesitation or exploration
    try:
        vte_numeric = clean_data['VTE'].astype(int)
        correlation_coeff, p_value = pearsonr(correct_numeric, vte_numeric)
        correlation_results['Correct_vs_VTE'] = {
            'correlation': correlation_coeff,
            'p_value': p_value,
            'type': 'phi_coefficient',
            'n_observations': len(clean_data),
            'interpretation': _interpret_correlation_strength(abs(correlation_coeff))
        }
        logger.info(f"Correct vs VTE: φ = {correlation_coeff:.3f}, p = {p_value:.3f}")
    except Exception as error:
        logger.error(f"Error computing Correct vs VTE correlation: {error}")
    
    return correlation_results


def _interpret_correlation_strength(absolute_correlation: float) -> str:
    """
    Provide a text interpretation of correlation strength based on common conventions.
    
    This helper function translates numerical correlation values into meaningful
    descriptive terms that are easier to interpret in research contexts.
    
    Args:
        absolute_correlation (float): Absolute value of the correlation coefficient
        
    Returns:
        str: Descriptive interpretation of correlation strength
    """
    if absolute_correlation < 0.1:
        return "negligible"
    elif absolute_correlation < 0.3:
        return "weak"
    elif absolute_correlation < 0.5:
        return "moderate"
    elif absolute_correlation < 0.7:
        return "strong"
    else:
        return "very strong"


def create_correlation_scatter_plots(combined_data: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Create scatter plots showing correlations between correct responses and other variables.
    
    These visualizations help reveal patterns in the data that might not be apparent
    from correlation coefficients alone. Special techniques are used to handle the
    visualization challenges that arise when plotting binary variables.
    
    Args:
        combined_data (pd.DataFrame): Combined behavioral data from all rats
        save_path (Optional[Path]): Directory to save plots. If not provided, displays plots
    """
    logger.info("Creating correlation scatter plots...")
    
    if combined_data.empty:
        logger.error("No data available for plotting")
        return
    
    # Set up the plotting environment with a clean, professional style
    plt.style.use('seaborn-v0_8')
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    figure.suptitle('Behavioral Correlations: Correct Response vs Other Measures', 
                   fontsize=16, fontweight='bold')
    
    # Define consistent colors for correct and incorrect responses
    # Green typically represents success/correct, red represents failure/incorrect
    response_colors = {True: '#2E8B57', False: '#DC143C'}
    
    # Plot 1: Correct vs Length
    # This plot examines whether response accuracy relates to trial duration
    _create_binary_continuous_scatter(axes[0], combined_data, 'Length', 'Correct vs Length', response_colors)
    
    # Plot 2: Correct vs zIdPhi  
    # This examines the relationship between accuracy and the zIdPhi measure
    _create_binary_continuous_scatter(axes[1], combined_data, 'zIdPhi', 'Correct vs zIdPhi', response_colors)
    # Add reference line at zero since zIdPhi can be negative
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Zero reference')
    
    # Plot 3: Correct vs VTE (both binary variables)
    # This requires a special visualization approach since both variables are binary
    _create_binary_binary_plot(axes[2], combined_data, response_colors)
    
    plt.tight_layout()
    
    # Handle saving or displaying the plots
    if save_path is not None:
        output_file = save_path / "correlation_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation plots saved to {output_file}")
    else:
        plt.show()
    
    logger.info("Correlation scatter plots created successfully")


def _create_binary_continuous_scatter(axis, data: pd.DataFrame, x_column: str, title: str, colors: Dict[bool, str]) -> None:
    """
    Create a scatter plot for binary vs continuous variable relationships.
    
    This helper function handles the specific challenges of visualizing relationships
    where one variable is binary. Jittering is applied to separate overlapping points.
    
    Args:
        axis: Matplotlib axis object to plot on
        data (pd.DataFrame): The data to plot
        x_column (str): Name of the continuous variable column
        title (str): Plot title
        colors (Dict[bool, str]): Color mapping for True/False values
    """
    for correct_value in [True, False]:
        # Filter data for each response type
        subset_mask = data['Correct'] == correct_value
        data_subset = data[subset_mask]
        
        # Create y-values for the binary variable with jitter for better visualization
        # Jitter helps separate overlapping points that would otherwise stack
        y_positions = np.array([1 if correct_value else 0] * len(data_subset))
        vertical_jitter = np.random.normal(0, 0.02, len(data_subset))
        
        axis.scatter(data_subset[x_column], y_positions + vertical_jitter, 
                    c=colors[correct_value], alpha=0.6, s=20,
                    label=f'Correct: {correct_value}')
    
    # Configure axis properties for clear interpretation
    axis.set_xlabel(x_column, fontsize=12)
    axis.set_ylabel('Correct Response', fontsize=12)
    axis.set_title(title, fontsize=14, fontweight='bold')
    axis.set_yticks([0, 1])
    axis.set_yticklabels(['False', 'True'])
    axis.legend()
    axis.grid(True, alpha=0.3)


def _create_binary_binary_plot(axis, data: pd.DataFrame, colors: Dict[bool, str]) -> None:
    """
    Create a specialized plot for the relationship between two binary variables.
    
    Since both variables are binary, we use a bubble plot where bubble size
    represents the count of observations in each category combination.
    
    Args:
        axis: Matplotlib axis object to plot on
        data (pd.DataFrame): The data to plot
        colors (Dict[bool, str]): Color mapping for True/False values
    """
    # Create a contingency table to count observations in each category combination
    contingency_table = pd.crosstab(data['VTE'], data['Correct'])
    
    # Define positions and labels for the four possible combinations
    category_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    category_labels = ['VTE:F, Correct:F', 'VTE:F, Correct:T', 'VTE:T, Correct:F', 'VTE:T, Correct:T']
    
    # Extract counts for each combination, handling missing combinations gracefully
    observation_counts = [
        contingency_table.loc[False, False] if (False in contingency_table.index and False in contingency_table.columns) else 0,
        contingency_table.loc[False, True] if (False in contingency_table.index and True in contingency_table.columns) else 0,
        contingency_table.loc[True, False] if (True in contingency_table.index and False in contingency_table.columns) else 0,
        contingency_table.loc[True, True] if (True in contingency_table.index and True in contingency_table.columns) else 0
    ]
    
    # Create bubble plot where size represents frequency of each combination
    max_count = max(observation_counts) if observation_counts else 1
    for position, label, count in zip(category_positions, category_labels, observation_counts):
        # Scale bubble size proportionally to count, with minimum size for visibility
        bubble_size = (count / max_count) * 1000 + 50
        bubble_color = colors[position[1] == 1]  # Color based on correctness
        
        axis.scatter(position[0], position[1], s=bubble_size, c=bubble_color, alpha=0.7)
        # Add count labels inside bubbles for precise information
        axis.annotate(f'{count}', position, ha='center', va='center', 
                     fontsize=10, fontweight='bold')
    
    # Configure axis for clear binary variable interpretation
    axis.set_xlabel('VTE', fontsize=12)
    axis.set_ylabel('Correct Response', fontsize=12)
    axis.set_title('Correct vs VTE\n(Bubble size = count)', fontsize=14, fontweight='bold')
    axis.set_xticks([0, 1])
    axis.set_xticklabels(['False', 'True'])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(['False', 'True'])
    axis.grid(True, alpha=0.3)


def create_individual_rat_plots(rat_data: Dict[str, pd.DataFrame], save_path: Optional[Path] = None) -> None:
    """
    Create separate plots for each individual rat to examine within-subject patterns.
    
    Individual rat analysis is crucial for understanding whether population-level
    patterns hold consistently across subjects or whether there are meaningful
    individual differences that might be obscured in combined analyses.
    
    Args:
        rat_data (Dict[str, pd.DataFrame]): Individual rat data keyed by rat name
        save_path (Optional[Path]): Directory to save plots. If not provided, displays plots
    """
    logger.info("Creating individual rat plots...")
    
    number_of_rats = len(rat_data)
    if number_of_rats == 0:
        logger.error("No rat data available for individual plots")
        return
    
    # Calculate optimal subplot layout to accommodate all rats
    columns = min(3, number_of_rats)  # Maximum 3 columns for readability
    rows = (number_of_rats + columns - 1) // columns  # Ceiling division
    
    figure, axes = plt.subplots(rows, columns, figsize=(6*columns, 4*rows))
    
    # Handle different subplot configurations to ensure consistent axis access
    if number_of_rats == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if columns == 1 else axes
    else:
        axes = axes.flatten()
    
    figure.suptitle('Individual Rat Analysis: Correct vs zIdPhi', fontsize=16, fontweight='bold')
    
    response_colors = {True: '#2E8B57', False: '#DC143C'}
    
    # Create individual plots for each rat
    for rat_index, (rat_name, rat_dataframe) in enumerate(rat_data.items()):
        current_axis = axes[rat_index]
        
        # Plot correct and incorrect responses separately for clarity
        for correct_value in [True, False]:
            response_mask = rat_dataframe['Correct'] == correct_value
            response_subset = rat_dataframe[response_mask]
            
            if len(response_subset) > 0:
                # Apply jitter to separate overlapping points
                y_positions = np.array([1 if correct_value else 0] * len(response_subset))
                vertical_jitter = np.random.normal(0, 0.02, len(response_subset))
                
                current_axis.scatter(response_subset['zIdPhi'], y_positions + vertical_jitter,
                                   c=response_colors[correct_value], alpha=0.7, s=30,
                                   label=f'Correct: {correct_value}')
        
        # Configure individual plot properties
        current_axis.set_xlabel('zIdPhi', fontsize=10)
        current_axis.set_ylabel('Correct', fontsize=10)
        current_axis.set_title(f'Rat: {rat_name}\n(n={len(rat_dataframe)} trials)', fontsize=12)
        current_axis.set_yticks([0, 1])
        current_axis.set_yticklabels(['False', 'True'])
        current_axis.legend(fontsize=8)
        current_axis.grid(True, alpha=0.3)
        current_axis.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Hide any unused subplot areas for cleaner appearance
    for unused_index in range(number_of_rats, len(axes)):
        axes[unused_index].set_visible(False)
    
    plt.tight_layout()
    
    # Handle saving or displaying the plots
    if save_path is not None:
        output_file = save_path / "individual_rat_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Individual rat plots saved to {output_file}")
    else:
        plt.show()


def generate_analysis_summary_report(combined_data: pd.DataFrame, 
                                   rat_data: Dict[str, pd.DataFrame], 
                                   correlations: Dict[str, Dict]) -> str:
    """
    Generate a comprehensive text summary of the analysis results.
    
    This function creates a detailed report that can be saved for research records
    or included in manuscripts. It provides both statistical results and practical
    interpretations of the findings.
    
    Args:
        combined_data (pd.DataFrame): Combined data from all rats
        rat_data (Dict[str, pd.DataFrame]): Individual rat data
        correlations (Dict[str, Dict]): Correlation analysis results
        
    Returns:
        str: Formatted comprehensive analysis report
    """
    if combined_data.empty:
        return "No data available for analysis report generation"
    
    report_sections = []
    report_sections.append("="*60)
    report_sections.append("RAT BEHAVIORAL DATA CORRELATION ANALYSIS REPORT")
    report_sections.append("="*60)
    report_sections.append("")
    
    # Dataset overview section
    report_sections.append("DATASET SUMMARY:")
    report_sections.append(f"  Total number of rats analyzed: {len(rat_data)}")
    report_sections.append(f"  Total number of behavioral trials: {len(combined_data)}")
    report_sections.append(f"  Rats included in analysis: {', '.join(rat_data.keys())}")
    report_sections.append("")
    
    # Individual rat performance metrics
    report_sections.append("INDIVIDUAL RAT PERFORMANCE STATISTICS:")
    for rat_name, rat_dataframe in rat_data.items():
        # Calculate key behavioral metrics for each rat
        accuracy_rate = rat_dataframe['Correct'].mean()
        vte_occurrence_rate = rat_dataframe['VTE'].mean()
        average_trial_length = rat_dataframe['Length'].mean()
        average_zidphi = rat_dataframe['zIdPhi'].mean()
        
        report_sections.append(f"  {rat_name}:")
        report_sections.append(f"    Number of trials completed: {len(rat_dataframe)}")
        report_sections.append(f"    Overall accuracy rate: {accuracy_rate:.1%}")
        report_sections.append(f"    VTE occurrence rate: {vte_occurrence_rate:.1%}")
        report_sections.append(f"    Average trial length: {average_trial_length:.2f}")
        report_sections.append(f"    Average zIdPhi value: {average_zidphi:.3f}")
        report_sections.append("")
    
    # Statistical correlation results
    report_sections.append("CORRELATION ANALYSIS RESULTS:")
    for comparison_name, statistical_results in correlations.items():
        report_sections.append(f"  {comparison_name.replace('_', ' ')}:")
        report_sections.append(f"    Correlation coefficient: {statistical_results['correlation']:.3f}")
        report_sections.append(f"    Statistical significance (p-value): {statistical_results['p_value']:.3f}")
        report_sections.append(f"    Correlation type: {statistical_results['type']}")
        report_sections.append(f"    Sample size: {statistical_results['n_observations']}")
        
        # Provide practical interpretation of results
        significance_status = "statistically significant" if statistical_results['p_value'] < 0.05 else "not statistically significant"
        correlation_strength = statistical_results['interpretation']
        
        report_sections.append(f"    Interpretation: {correlation_strength} correlation, {significance_status}")
        
        # Add effect size interpretation for significant results
        if statistical_results['p_value'] < 0.05:
            effect_magnitude = abs(statistical_results['correlation'])
            if effect_magnitude > 0.5:
                practical_significance = "This represents a practically meaningful relationship"
            elif effect_magnitude > 0.3:
                practical_significance = "This represents a moderate practical effect"
            else:
                practical_significance = "This represents a small but detectable effect"
            report_sections.append(f"    Practical significance: {practical_significance}")
        
        report_sections.append("")
    
    report_sections.append("="*60)
    
    return "\n".join(report_sections)


def analyze_rat_behavioral_correlations(vte_values_path: Path, output_directory: Optional[Path] = None, plot_all_rats: bool = False) -> None:
    """
    Execute the complete behavioral correlation analysis pipeline.
    
    This function orchestrates the entire analysis workflow, from data loading
    through final report generation. It serves as the main entry point for
    running the complete analysis on your rat behavioral data.
    
    Args:
        vte_values_path (Path): Path object pointing to the directory containing rat subdirectories
        output_directory (Optional[Path]): Directory to save analysis outputs. 
                                         If not provided, creates 'analysis_output' in parent directory
    """
    logger.info("Initiating comprehensive rat behavioral correlation analysis...")
    
    # Set up output directory structure
    if output_directory is None:
        analysis_output_path = vte_values_path.parent / "analysis_output"
    else:
        analysis_output_path = output_directory
    
    analysis_output_path.mkdir(exist_ok=True)
    logger.info(f"Analysis outputs will be saved to: {analysis_output_path}")
    
    # Execute the data loading phase
    combined_behavioral_data, individual_rat_data = load_all_rat_data(vte_values_path)
    
    if combined_behavioral_data.empty:
        logger.error("No valid behavioral data found. Analysis cannot proceed.")
        return
    
    # Execute the statistical analysis phase
    correlation_results = compute_behavioral_correlations(combined_behavioral_data)
    
    # Execute the visualization phase
    create_correlation_scatter_plots(combined_behavioral_data, save_path=analysis_output_path)
    
    if plot_all_rats:
        create_individual_rat_plots(individual_rat_data, save_path=analysis_output_path)
    
    # Execute the reporting phase
    comprehensive_report = generate_analysis_summary_report(
        combined_behavioral_data, individual_rat_data, correlation_results
    )
    
    # Save the final report
    report_output_file = analysis_output_path / "behavioral_analysis_report.txt"
    with open(report_output_file, 'w') as report_file:
        report_file.write(comprehensive_report)
    
    logger.info(f"Complete analysis finished successfully! Full report saved to {report_output_file}")
    
    # Display the report for immediate review
    print("\n" + comprehensive_report)


# Example usage demonstration
if __name__ == "__main__":
    
    # Execute the complete analysis pipeline
    analyze_rat_behavioral_correlations(paths.vte_values)
    
    logger.info("Behavioral correlation analysis pipeline completed successfully!")