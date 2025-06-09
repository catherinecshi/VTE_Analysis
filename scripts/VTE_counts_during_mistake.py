import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import helper
from analysis import statistics

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")

# for each trajectory, what is the closest mistake trial before and after the trial
until_next_mistake = {} # {traj_id: # trials until next mistake}
since_last_mistake = {} # {traj_id: # trials since last mistake}
for rat in os.listdir(base_path):
    if ".DS_Store" in rat or ".csv" in rat:
        continue
    
    rat_path = os.path.join(base_path, rat)
    for day in os.listdir(rat_path):
        if ".DS_Store" in day:
            continue
        
        day_path = os.path.join(rat_path, day)
        for root, _, files in os.walk(day_path):
            for file in files:
                if "trajectories" not in file:
                    continue

                file_path = os.path.join(root, file)
                trajectory_csv = pd.read_csv(file_path)
                
                trial_count = 0 # count # trials so i can exclude sessions with <5 trials later
                mistakes = [] # gets all the trial numbers that were mistake trials
                for index, row in trajectory_csv.iterrows():
                    traj_id = row["ID"]
                    parts = traj_id.split("_")
                    traj_number = parts[2]
                    is_correct = row["Correct"]
                    trial_count += 1
                    
                    if not is_correct:
                        mistakes.append(int(traj_number))
                
                if trial_count <= 10: # too few trials for good data analysis
                    continue
                
                if not mistakes: # sessions with no mistakes
                    continue
                else: # get number of trials before and after mistakes for each traj_id
                    for index, row in trajectory_csv.iterrows():
                        traj_id = row["ID"]
                        parts = traj_id.split("_")
                        traj_number = parts[2]
                        larger_numbers = [num for num in mistakes if num > int(traj_number)] # closest mistake AFTER current trial
                        smaller_numbers = [num for num in mistakes if num < int(traj_number)] # closest mistake BEFORE current trial
                        
                        if int(traj_number) in mistakes:
                            until_next_mistake[traj_id] = 0
                        else:
                            if larger_numbers:
                                closest_number = min(larger_numbers)
                                if (closest_number - int(traj_number)) == 0:
                                    print(traj_id)
                                until_next_mistake[traj_id] = closest_number - int(traj_number)
                            
                            if smaller_numbers:
                                closest_number = max(smaller_numbers)
                                if (closest_number - int(traj_number)) == 0:
                                    print(traj_id)
                                since_last_mistake[traj_id] = closest_number - int(traj_number) # these will be negative values

all_trial_to_vtes = {} # {rat:{day:{trial # : % vte}}}
for rat in os.listdir(base_path):
    rat_path = os.path.join(base_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file)
            zIdPhi_csv = pd.read_csv(file_path)
            
            mean_zIdPhi = np.mean(zIdPhi_csv["zIdPhi"])
            std_zIdPhi = np.std(zIdPhi_csv["zIdPhi"])
            VTE_threshold = mean_zIdPhi + (std_zIdPhi * 1.5)
            
            # go through each day independently
            grouped_by_day = zIdPhi_csv.groupby("Day")
            for day, group in grouped_by_day:
                trues = {}
                falses = {}
                for index, row in group.iterrows():
                    traj_id = row["ID"]
                    is_VTE = row["zIdPhi"] > VTE_threshold
                    
                    try:
                        trials_until_mistake = until_next_mistake[traj_id]
                    except KeyError:
                        pass
                    else:
                        if is_VTE:
                            try:
                                trues[trials_until_mistake] += 1
                            except KeyError:
                                trues[trials_until_mistake] = 1
                        else:
                            try:
                                falses[trials_until_mistake] += 1
                            except KeyError:
                                falses[trials_until_mistake] = 1
                    
                    try:
                        trials_since_mistake = since_last_mistake[traj_id]
                    except KeyError:
                        pass
                    else:
                        if is_VTE:
                            try:
                                trues[trials_since_mistake] += 1
                            except KeyError:
                                trues[trials_since_mistake] = 1
                        else:
                            try:
                                falses[trials_since_mistake] += 1
                            except KeyError:
                                falses[trials_since_mistake] = 1

                mistake_trials_to_vtes = {} # trial # to mistake : vte proportion
                for trials_to_mistake, number_of_trials in falses.items():
                    if -2 > trials_to_mistake or 2 < trials_to_mistake: # only include -2 to 2
                        continue
                    
                    try:
                        proportion = trues[trials_to_mistake] / (trues[trials_to_mistake] + number_of_trials)
                    except KeyError:
                        mistake_trials_to_vtes[trials_to_mistake] = 0.0
                    else:
                        mistake_trials_to_vtes[trials_to_mistake] = proportion
                
                rat_day = rat + str(day)
                all_trial_to_vtes[rat_day] = mistake_trials_to_vtes
    
# Create lists to hold VTE proportions for each trial number relative to mistakes
trial_groups = {-2: [], -1: [], 0: [], 1: [], 2: []}

# Go through each rat/day combination
for rat_day, mistake_trials_to_vtes in all_trial_to_vtes.items():
    # For each trial number, append the VTE proportion to the appropriate group
    for trial_num, vte_prop in mistake_trials_to_vtes.items():
        if trial_num in trial_groups:  # Only include trials -2 to 2
            trial_groups[trial_num].append(vte_prop)

# Convert to format needed for ANOVA function
data_groups = [trial_groups[i] for i in sorted(trial_groups.keys())]
group_labels = [str(i) for i in sorted(trial_groups.keys())]

def plot_one_way_anova_bar(data_groups, group_labels=None, title=None, xlabel=None, ylabel=None):
    """
    Creates a bar plot with error bars and significance indicators
    
    Parameters:
        - data_groups (list of arrays): arrays of data for each group
        - group_labels (str list): (optional) labels for each group
        - title (str): (optional)
        - xlabel (str): (optional)
        - ylabel (str): (optional)
    
    Returns:
        - fig: plt figure object
        - ax: plt axis object
        - stats_results (dict): results from one_way_anova
    """
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    stats_results = statistics.one_way_anova(data_groups)
    
    # means & SEM
    means = [np.mean(group) * 100 for group in data_groups]
    sems = [stats.sem(group) * 100 for group in data_groups]
    
    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data_groups))
    
    ax.bar(x, means, yerr=sems, capsize=5)
    
    # Add significance bars if ANOVA is significant
    if stats_results["p_value"] < 0.05:
        # Perform pairwise t-tests with Bonferroni correction
        significant_pairs = []
        p_values = {}
        
        for i in range(len(data_groups)):
            for j in range(i+1, len(data_groups)):
                t_stat, p_val = stats.ttest_ind(
                    data_groups[i], 
                    data_groups[j], 
                    equal_var=False  # Use Welch's t-test
                )
                
                # Apply Bonferroni correction for multiple comparisons
                n_comparisons = len(data_groups) * (len(data_groups) - 1) / 2
                corrected_p = p_val * n_comparisons
                corrected_p = min(corrected_p, 1.0)  # Cap at 1.0
                
                if corrected_p < 0.05:
                    significant_pairs.append((i, j))
                    p_values[(i, j)] = corrected_p
        
        # Add significance bars and stars
        max_height = max(means) + max(sems) if sems else max(means)
        step = max_height * 0.1
        
        for i, (group1, group2) in enumerate(significant_pairs):
            # Calculate positions
            bar_height = max_height + step * (i + 1)
            
            # Draw the bar
            x1, x2 = group1, group2
            ax.plot([x1, x1, x2, x2], 
                   [bar_height, bar_height + step/2, bar_height + step/2, bar_height],
                   color='black', linewidth=1)
            
            # Add star
            p_value = p_values[(group1, group2)]
            if p_value < 0.001:
                star = '***'
            elif p_value < 0.01:
                star = '**'
            else:
                star = '*'
            
            ax.text((x1 + x2) / 2, bar_height + step/2, star, 
                   ha='center', va='bottom', fontsize=12)
            
        # Adjust the y-axis to accommodate the significance bars
        y_max = ax.get_ylim()[1]
        required_y_max = max_height + step * (len(significant_pairs) + 1)
        if required_y_max > y_max:
            ax.set_ylim(top=required_y_max)
    
    # add plot elements
    if group_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
    
    if title:
        ax.set_title(title, fontsize=30)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=24)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=24)
    
    # add anova results to plot
    anova_text = f"One-way ANOVA: \nF = {stats_results['f_stat']:.3f}\np = {stats_results['p_value']:.2e}"
    print(anova_text)
    #ax.text(0.95, 0.95, anova_text, transform=ax.transAxes, verticalalignment="top",
           # horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig, ax, stats_results

# Run ANOVA and create plot
fig, ax, stats_results = plot_one_way_anova_bar(
    data_groups,
    group_labels=group_labels,
    title="VTE Relative to Mistake Trials",
    xlabel="Trials from Mistake",
    ylabel="VTE Percentage (%)"
)
plt.show()
print(stats_results)

def print_bonferroni_results(data_groups, group_labels=None):
    """
    Perform pairwise t-tests with Bonferroni correction and print results
    
    Parameters:
        - data_groups (list of arrays): arrays of data for each group
        - group_labels (str list): (optional) labels for each group
    """
    from scipy import stats
    import numpy as np
    
    if group_labels is None:
        group_labels = [str(i) for i in range(len(data_groups))]
    
    # Calculate number of comparisons for Bonferroni correction
    n_comparisons = len(data_groups) * (len(data_groups) - 1) // 2
    
    print("\n===== Bonferroni Post-Hoc Analysis =====")
    print(f"Number of comparisons: {n_comparisons}")
    print("Alpha level after Bonferroni correction: {:.5f}".format(0.05/n_comparisons))
    print("\nSignificant pairwise comparisons:")
    
    # Perform all pairwise comparisons
    found_significant = False
    for i in range(len(data_groups)):
        for j in range(i+1, len(data_groups)):
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(
                data_groups[i],
                data_groups[j],
                equal_var=False  # Welch's t-test for unequal variances
            )
            
            # Apply Bonferroni correction
            corrected_p = p_val * n_comparisons
            corrected_p = min(corrected_p, 1.0)  # Cap at 1.0
            
            # Determine significance level
            if corrected_p < 0.05:
                found_significant = True
                significance = ""
                if corrected_p < 0.001:
                    significance = "***"
                elif corrected_p < 0.01:
                    significance = "**"
                else:
                    significance = "*"
                
                # Format means for comparison
                mean_i = np.mean(data_groups[i]) * 100
                mean_j = np.mean(data_groups[j]) * 100
                
                # Print the result
                print(f"  {group_labels[i]} vs {group_labels[j]}: t = {t_stat:.3f}, p = {p_val:.5f}, " +
                      f"corrected p = {corrected_p:.5f} {significance}")
                print(f"    Mean {group_labels[i]}: {mean_i:.2f}%, Mean {group_labels[j]}: {mean_j:.2f}%")
    
    if not found_significant:
        print("  No significant pairwise comparisons found.")
    
    print("\nSignificance levels:")
    print("  * p < 0.05")
    print("  ** p < 0.01")
    print("  *** p < 0.001")

# Print the number of observations in each group
for trial_num in sorted(trial_groups.keys()):
    print(f"Group {trial_num}: {len(trial_groups[trial_num])} observations")

# Calculate total observations and degrees of freedom
total_observations = sum(len(group) for group in trial_groups.values())
num_groups = len(trial_groups)
between_df = num_groups - 1
within_df = total_observations - num_groups

print(f"\nTotal observations across all groups: {total_observations}")
print(f"Number of groups: {num_groups}")
print(f"Between-groups degrees of freedom: {between_df}")
print(f"Within-groups degrees of freedom: {within_df}")
print(f"F-statistic should be reported as: F({between_df}, {within_df})")

print_bonferroni_results(data_groups, group_labels)