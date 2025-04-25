"""
Script to generate all figures from performance analysis data using matplotlib directly
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from datetime import datetime

# Import modules from src package
from src import data_processing
from src import performance_analysis
from src import helper

# Set up logging
logging.basicConfig(filename=f"figure_generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    format='%(asctime)s %(message)s',
                    filemode="w")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Define paths
BASE_PATH = "/Users/catpillow/Documents/VTE_Analysis"
SAVE_PATH = os.path.join(BASE_PATH, "data/VTE_Data")
FIGURES_PATH = os.path.join(BASE_PATH, "figures")

# Create figures directory if it doesn't exist
os.makedirs(FIGURES_PATH, exist_ok=True)

def run_anova_analysis(all_days_until_criteria, figures_path):
    """
    Run ANOVA analysis on days until criteria data to determine if there are significant
    differences in learning rates across different trial types.
    
    Args:
        all_days_until_criteria (dict): {rat: {trial_type: day}}
        figures_path (str): Path to save figures
    
    Returns:
        tuple: (f_statistic, p_value, df)
    """
    import scipy.stats as stats
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Reorganize data into a format suitable for ANOVA
    data = []
    for rat, criteria_days in all_days_until_criteria.items():
        for trial_type, days in criteria_days.items():
            data.append({
                'rat': rat,
                'trial_type': trial_type,
                'days_until_criteria': days
            })
    
    df = pd.DataFrame(data)
    
    # Create a summary of the data
    summary = df.groupby('trial_type')['days_until_criteria'].agg(['mean', 'std', 'count'])
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    print("\nSummary Statistics by Trial Type:")
    print(summary)
    
    # Print observations counts
    total_observations = len(df)
    print(f"\nObservation Counts:")
    print(f"Total observations: {total_observations}")
    print("Observations per trial type:")
    for trial_type in sorted(df['trial_type'].unique()):
        count = len(df[df['trial_type'] == trial_type])
        print(f"  Type {trial_type}: {count} observations")
    
    # Save the reorganized data for future reference
    df.to_csv(os.path.join(os.path.dirname(figures_path), "days_until_criteria_by_trial.csv"), index=False)
    
    # Organize data by trial type for ANOVA
    trial_data = {}
    for trial_type in sorted(df['trial_type'].unique()):
        trial_data[trial_type] = df[df['trial_type'] == trial_type]['days_until_criteria'].values
    
    # Run one-way ANOVA
    groups = list(trial_data.values())
    f_statistic, p_value = stats.f_oneway(*groups)
    
    # Calculate degrees of freedom
    k = len(groups)  # Number of groups (trial types)
    n = sum(len(group) for group in groups)  # Total number of observations
    df_between = k - 1  # Between groups degrees of freedom
    df_within = n - k   # Within groups degrees of freedom
    df_total = n - 1    # Total degrees of freedom
    
    print(f"\nDegrees of Freedom:")
    print(f"Number of groups (k): {k}")
    print(f"Between groups df: {df_between}")
    print(f"Within groups df: {df_within}")
    print(f"Total df: {df_total}")
    
    print(f"\nANOVA Results:")
    print(f"F-statistic ({df_between}, {df_within}): {f_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("The differences between trial types are statistically significant.")
        
        # Run post-hoc tests if ANOVA is significant
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        # Prepare data for Tukey's test
        tukey_data = []
        tukey_groups = []
        for group_idx, group_data in enumerate(groups):
            tukey_data.extend(group_data)
            tukey_groups.extend([f"Type {sorted(trial_data.keys())[group_idx]}"] * len(group_data))
        
        # Perform Tukey's test
        tukey_results = pairwise_tukeyhsd(tukey_data, tukey_groups, alpha=0.05)
        print("\nTukey HSD Results:")
        print(tukey_results)
    else:
        print("The differences between trial types are not statistically significant.")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Box plot
    positions = np.arange(1, len(trial_data) + 1)
    box = plt.boxplot([trial_data[t] for t in sorted(trial_data.keys())], 
                      positions=positions, 
                      patch_artist=True)
    
    # Add individual data points (jittered for visibility)
    for i, (trial_type, data) in enumerate(sorted(trial_data.items())):
        # Generate random x-coordinates around position i+1
        x = np.random.normal(i+1, 0.05, size=len(data))
        plt.scatter(x, data, alpha=0.6, color=f'C{i}')
    
    # Add color to the boxplots
    for patch, color in zip(box['boxes'], [f'C{i}' for i in range(len(trial_data))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    
    # Add mean and standard error
    for i, (trial_type, data) in enumerate(sorted(trial_data.items())):
        if len(data) > 0:
            mean = np.mean(data)
            std = np.std(data)
            sem = std / np.sqrt(len(data))
            plt.errorbar(i+1, mean, yerr=sem, fmt='ks', markersize=8, capsize=10, elinewidth=2)
    
    # Add sample size to x-axis labels
    x_labels = []
    for t in sorted(trial_data.keys()):
        count = len(trial_data[t])
        x_labels.append(f"Type {t}\n(n={count})")
    
    # Set labels and title
    plt.xticks(positions, x_labels)
    plt.xlabel('Trial Type')
    plt.ylabel('Number of Days Until Criteria')
    if p_value < 0.05:
        plt.title(f'Days Until Criteria by Trial Type (ANOVA: F({df_between},{df_within})={f_statistic:.2f}, p={p_value:.4f}*)')
    else:
        plt.title(f'Days Until Criteria by Trial Type (ANOVA: F({df_between},{df_within})={f_statistic:.2f}, p={p_value:.4f})')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(figures_path, "days_until_criteria_anova.png"))
    plt.close()
    
    return f_statistic, p_value, df

def generate_all_figures():
    """
    Generate all figures from the performance analysis data
    """
    print("Starting figure generation...")
    
    # Load data structure using the function from data_processing
    DATA_STRUCTURE = data_processing.load_data_structure(SAVE_PATH)
    
    # 1. Generate performance over sessions plot for each rat
    """print("Generating rat performance over sessions plots...")
    for rat_id in DATA_STRUCTURE:
        if rat_id == ".DS_Store" or "pkl" in rat_id:
            continue
            
        try:
            # Get performance data for this rat
            rat_perf_df = performance_analysis.rat_performance_over_sessions(DATA_STRUCTURE, rat_id)
            
            # Convert DataFrame to dictionary format for easier plotting
            performance_by_type = {}
            days = sorted(rat_perf_df['day'].unique())
            
            # Compile performance data by trial type
            for _, row in rat_perf_df.iterrows():
                trial_type = row['trial_type']
                day = row['day']
                performance = (row['correct_trials'] / row['total_trials']) * 100 if row['total_trials'] else 0
                
                if trial_type not in performance_by_type:
                    performance_by_type[trial_type] = [None] * len(days)
                
                day_index = days.index(day)
                performance_by_type[trial_type][day_index] = performance
            
            # Create plot
            plt.figure(figsize=(10, 6))
            for trial_type, performances in performance_by_type.items():
                # Filter out None values
                valid_indices = [i for i, perf in enumerate(performances) if perf is not None]
                valid_days = [days[i] for i in valid_indices]
                valid_performances = [performances[i] for i in valid_indices]
                
                plt.plot(valid_days, valid_performances, label=f"Type {trial_type}", marker="o")
            
            plt.xlabel("Day")
            plt.ylabel("Performance (%)")
            plt.title(f"Rat {rat_id} Performance Over Sessions")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 100)
            plt.tight_layout()
            
            plt.savefig(os.path.join(FIGURES_PATH, f"{rat_id}_performance_over_sessions.png"))
            plt.close()
            print(f"Generated performance over sessions plot for {rat_id}")
        except Exception as e:
            logger.error(f"Error generating performance over sessions plot for {rat_id}: {e}")
    
    # 2. Generate individual rat trial accuracy plots for each day
    print("Generating individual rat trial accuracy plots...")
    for rat_id in DATA_STRUCTURE:
        if rat_id == ".DS_Store" or "pkl" in rat_id:
            continue
            
        for day in DATA_STRUCTURE[rat_id]:
            if day == ".DS_Store" or "pkl" in day:
                continue
            
            try:
                # Skip if statescript log doesn't exist
                if "stateScriptLog" not in DATA_STRUCTURE[rat_id][day] or DATA_STRUCTURE[rat_id][day]["stateScriptLog"] is None:
                    continue
                
                # Set the current rat and day for the helper module
                helper.update_rat(rat_id)
                helper.update_day(day)
                
                # Get trial accuracy data
                ss_data = DATA_STRUCTURE[rat_id][day]["stateScriptLog"]
                trial_types_set, total_trials, correct_trials = performance_analysis.trial_accuracy(ss_data)
                
                # Convert set to sorted list
                trial_types = sorted(trial_types_set)
                
                # Skip if there's not enough data
                if len(total_trials) < 1:
                    continue
                
                # Calculate percentage correct
                percentage_correct = (correct_trials / total_trials) * 100
                
                # Adjust trial types if needed
                length = len(total_trials)
                trial_types_display = [f"Type {t}" for t in trial_types[:length]]
                
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.bar(trial_types_display, percentage_correct, color="darkred")
                plt.title(f"Trial Accuracy for {rat_id} on {day}", fontsize=20)
                plt.ylabel("Performance (%)", fontsize=15)
                plt.xlabel("Trial Types", fontsize=15)
                plt.ylim(0, 100)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(os.path.join(FIGURES_PATH, f"{rat_id}_{day}_trial_accuracy.png"))
                plt.close()
                print(f"Generated trial accuracy plot for {rat_id} on {day}")
            except Exception as e:
                logger.error(f"Error generating trial accuracy plot for {rat_id} on {day}: {e}")"""
    
    # 3. Generate all rats performance data
    print("Generating all rats performance data...")
    all_rats_performances = performance_analysis.create_all_rats_performance(data_structure=DATA_STRUCTURE)
    #performance_analysis.plot_rat_performance(all_rats_performances)
    
    # 4. Generate performance changes plots
    print("Generating performance changes plots...")
    perf_changes = performance_analysis.create_all_perf_changes(all_rats_performances)
    
    # Create a plot for overall performance changes
    try:
        plt.figure(figsize=(12, 8))
        for rat, group in perf_changes.groupby('rat'):
            plt.plot(group['day'], group['perf_change'] * 100, marker='o', linestyle='-', label=rat)
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Day')
        plt.ylabel('Performance Change (%)')
        plt.title('Day-to-Day Performance Changes for All Rats')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_PATH, "all_rats_performance_changes.png"))
        plt.close()
        print("Generated performance changes plot")
    except Exception as e:
        print(f"Error generating performance changes plot: {e}")
    
    # 5. Generate performance changes by trial type
    print("Generating performance changes by trial plots...")
    perf_changes_by_trial = performance_analysis.create_all_perf_changes_by_trials(all_rats_performances)
    
    try:
        # Create box plot of performance changes by trial type
        plt.figure(figsize=(12, 8))
        
        # Group data by trial type
        data_by_trial = []
        labels = []
        for trial_type, group in perf_changes_by_trial.groupby('trial_type'):
            data_by_trial.append(group['perf_change'].values)
            labels.append(f"Type {trial_type}")
        
        # Create box plot
        plt.boxplot(data_by_trial, labels=labels, patch_artist=True)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Trial Type')
        plt.ylabel('Performance Change (%)')
        plt.title('Changes in Performance Across Trial Types')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_PATH, "performance_changes_by_trial_boxplot.png"))
        plt.close()
        
        # Create line plot of mean performance changes by trial type and day
        plt.figure(figsize=(12, 8))
        for trial_type, group in perf_changes_by_trial.groupby('trial_type'):
            means = group.groupby('day')['perf_change'].mean()
            plt.plot(means.index, means.values, marker='o', linestyle='-', label=f"Type {trial_type}")
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Day')
        plt.ylabel('Mean Performance Change (%)')
        plt.title('Mean Performance Changes by Trial Type Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_PATH, "performance_changes_by_trial_lineplot.png"))
        plt.close()
        print("Generated performance changes by trial plots")
    except Exception as e:
        print(f"Error generating performance changes by trial plots: {e}")
    
    # 6. Plot days until criteria
    print("Plotting days until criteria...")
    all_days_until_criteria = performance_analysis.days_until_criteria(all_rats_performances)
    performance_analysis.plot_days_until_criteria(all_days_until_criteria)
    
    f_stat, p_val, anova_df = run_anova_analysis(all_days_until_criteria, FIGURES_PATH)
    print(f_stat, p_val)

    # Extract all day values from the nested dictionary
    all_days = [days for rat_data in all_days_until_criteria.values() 
                for days in rat_data.values()]

    # Calculate and print statistics
    if all_days:
        avg_days = sum(all_days) / len(all_days)
        min_days = min(all_days)
        max_days = max(all_days)
        
        print(f"Days until criteria - Average: {avg_days:.2f}, Minimum: {min_days}, Maximum: {max_days}")
    else:
        print("No data available to calculate days until criteria statistics.")
    
    try:
        # Extract trial types from first rat
        for rat_id, criteria_days in all_days_until_criteria.items():
            trial_types = list(criteria_days.keys())
            break
        
        # Creating dictionaries to hold data for plotting
        trial_data = {trial_type: [] for trial_type in trial_types}
        
        for rat_id, criteria_days in all_days_until_criteria.items():
            for trial_type in trial_types:
                if trial_type in criteria_days:
                    trial_data[trial_type].append(criteria_days[trial_type])
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create positions for x-axis
        positions = np.arange(len(trial_types))
        
        # Plot individual data points
        for i, trial_type in enumerate(trial_types):
            # Generate random x-coordinates around position i
            x = np.random.normal(i, 0.05, size=len(trial_data[trial_type]))
            plt.scatter(x, trial_data[trial_type], alpha=0.6, color=f'C{i}')
        
        # Create boxplots
        box = plt.boxplot([trial_data[trial_type] for trial_type in trial_types], 
                     positions=positions, 
                     patch_artist=True)
        
        # Add color to the boxplots
        for patch, color in zip(box['boxes'], [f'C{i}' for i in range(len(trial_types))]):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        
        # Add mean and standard error
        for i, trial_type in enumerate(trial_types):
            data = trial_data[trial_type]
            if data:
                mean = np.mean(data)
                std = np.std(data)
                sem = std / np.sqrt(len(data))
                plt.errorbar(i, mean, yerr=sem, fmt='ks', markersize=8, capsize=10, elinewidth=3)
        
        # Set labels and title
        plt.xticks(positions, [f"Type {t}" for t in trial_types])
        plt.xlabel('Trial Type')
        plt.ylabel('Number of Days Until Criteria')
        plt.title('Days Until Criteria by Trial Type (75%)')
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(FIGURES_PATH, "days_until_criteria.png"))
        plt.close()
        print("Generated days until criteria plot")
    except Exception as e:
        print(f"Error generating days until criteria plot: {e}")
    
    # 7. Generate days since new arm data and create learning during volatility plots
    print("Generating days since new arm data...")
    days_since_new_arm = performance_analysis.get_days_since_new_arm(SAVE_PATH, DATA_STRUCTURE)
    
    # 8. Create learning during volatility plots
    print("Creating learning during volatility plots...")
    try:
        # Convert trials_available to proper format
        days_since_new_arm["trials_available"] = days_since_new_arm["trials_available"].apply(lambda x: [int(y) for y in x])
        days_since_new_arm = days_since_new_arm.astype({"rat": "str",
                                                      "day": "int", 
                                                      "arm_added": "bool", 
                                                      "days_since_new_arm": "int"})
        perf_changes_by_trial = perf_changes_by_trial.astype({"rat": "str",
                                                             "day": "int", 
                                                             "trial_type": "int", 
                                                             "perf_change": "float"})

        # Create an array of just the corresponding trial type change
        rat_df = days_since_new_arm.groupby("rat")
        learning_during_volatility = []
        for rat, rat_group in rat_df:
            sorted_by_day_df = rat_group.sort_values(by="day")
            for i, row in sorted_by_day_df.iterrows():
                day = row["day"]
                number_of_days_since = row["days_since_new_arm"]
                try:
                    highest_trial_available = max(row["trials_available"])
                except ValueError as e:
                    logger.error(f"Value error {e} with {rat} on {day}")
                    continue

                corresponding_row = perf_changes_by_trial[
                    (perf_changes_by_trial["rat"] == rat) & 
                    (perf_changes_by_trial["day"] == day) & 
                    (perf_changes_by_trial["trial_type"] == highest_trial_available)
                ]
                                                  
                if corresponding_row.empty:
                    logger.error(f"Error for {rat} on {day} - no corresponding perf change")
                    continue

                corresponding_perf_change = corresponding_row["perf_change"].iloc[0]
                learning_during_volatility.append({
                    "rat": rat, 
                    "day": day, 
                    "trial_type": highest_trial_available,
                    "days_since_new_arm": number_of_days_since,
                    "perf_change": corresponding_perf_change
                })

        learning_during_volatility_df = pd.DataFrame(learning_during_volatility)
        learning_during_volatility_df.to_csv(os.path.join(BASE_PATH, "processed_data", "learning_during_volatility.csv"))

        # Calculate SEM and mean for each group of days_since_new_arm
        grouped = learning_during_volatility_df.groupby("days_since_new_arm")["perf_change"]
        means = grouped.mean()
        sems = grouped.apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
        
        # Create boxplot
        plt.figure(figsize=(12, 8))
        days = sorted(learning_during_volatility_df["days_since_new_arm"].unique())
        days = [d for d in days if d <= 15]  # Limit to first 15 days
        
        data_by_day = [learning_during_volatility_df[learning_during_volatility_df["days_since_new_arm"] == day]["perf_change"].values 
                      for day in days]
        
        plt.boxplot(data_by_day, positions=days, patch_artist=True)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Number of Days Since New Arm Added")
        plt.ylabel("Change in Performance since Last Session (%)")
        plt.title("Learning during Volatility")
        plt.grid(True, axis='y', alpha=0.3)
        plt.xlim(0, 15)
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_PATH, "learning_volatility_boxplot.png"))
        plt.close()
        
        # Create histogram as a heatmap
        plt.figure(figsize=(12, 8))
        
        # Create 2D histogram
        h = plt.hist2d(learning_during_volatility_df["days_since_new_arm"], 
                     learning_during_volatility_df["perf_change"],
                     bins=[15, 20], 
                     range=[[0, 15], [-40, 40]],
                     cmap='viridis')
        
        plt.colorbar(h[3], label="Count")
        plt.xlabel("Number of Days Since New Arm Added")
        plt.ylabel("Change in Performance since Last Session (%)")
        plt.title("Learning during Volatility - Distribution")
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_PATH, "learning_volatility_histogram.png"))
        plt.close()
        
        # Create line plot
        plt.figure(figsize=(12, 8))
        
        # Get data for plot
        x_values = means.index.values
        y_values = means.values
        error = sems.values
        
        # Filter to only include days 0-8
        mask = x_values <= 8
        x_values = x_values[mask]
        y_values = y_values[mask]
        error = error[mask]
        
        # Plot
        plt.errorbar(x_values, y_values, yerr=error, fmt='-o', capsize=5, linewidth=2, markersize=8)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel("Number of Days Since New Arm Added")
        plt.ylabel("Change in Performance\nsince Last Session (%)")
        plt.title("Learning during Volatility")
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 8)
        plt.ylim(-10, 20)
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_PATH, "learning_volatility_lineplot.png"))
        plt.close()
        
        print("Generated learning during volatility plots")
    except Exception as e:
        print(f"Error generating learning during volatility plots: {e}")
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    generate_all_figures()