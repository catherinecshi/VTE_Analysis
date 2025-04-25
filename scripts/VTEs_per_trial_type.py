import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import MultiComparison

from src import helper
from src import plotting

vte_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values")

VTE_trials = {} # {trial_type: # VTEs}
all_trials = {} # {trial_type: # trials}
for rat in os.listdir(vte_path):
    if ".DS_Store" in rat:
        continue
    
    rat_path = os.path.join(vte_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file)
            zIdPhi_csv = pd.read_csv(file_path)
            
            zIdPhi_mean = zIdPhi_csv["zIdPhi"].mean()
            zIdPhi_std = zIdPhi_csv["zIdPhi"].std()
            VTE_threshold = zIdPhi_mean + (zIdPhi_std * 1.5)
            
            for index, row in zIdPhi_csv.iterrows():
                zIdPhi = row["zIdPhi"]
                trial_type = row["Trial_Type"]
                is_VTE = zIdPhi > VTE_threshold
                
                if is_VTE: # add to VTE counts for trial type
                    if trial_type in VTE_trials:
                        VTE_trials[trial_type] += 1
                    else:
                        VTE_trials[trial_type] = 1
                    
                # count total number of trials
                if trial_type in all_trials:
                    all_trials[trial_type] += 1
                else:
                    all_trials[trial_type] = 1

VTEs_for_trial_type = {} # {trial_type: proportion VTEs}
for trial_type in VTE_trials.keys():
    if trial_type == 5:
        continue
    
    number_VTEs = VTE_trials[trial_type]
    number_trials = all_trials[trial_type]
    
    VTE_proportion = number_VTEs / number_trials
    VTEs_for_trial_type[trial_type] = VTE_proportion

x_ticks = ["AB", "BC", "CD", "DE"] # trial types
data = VTEs_for_trial_type.values() # number of VTEs
plotting.create_bar_plot(data, x_ticks, title="VTE Percentage for Trial Type", xlabel="Trial Types", ylabel="Percentage of VTEs")

trial_types = ["AB", "BC", "CD", "DE"]
anova_data = []
anova_groups = []

# Calculate raw counts for chi-square test
for i, trial_type in enumerate([1, 2, 3, 4]):  # Numeric codes for trial types
    vte_count = VTE_trials.get(trial_type, 0)
    total_count = all_trials.get(trial_type, 0)
    non_vte_count = total_count - vte_count
    
    # Create an array with 1s for VTE trials and 0s for non-VTE trials
    group_data = np.concatenate([np.ones(vte_count), np.zeros(non_vte_count)])
    anova_data.extend(group_data)
    anova_groups.extend([trial_types[i]] * total_count)

# Convert to arrays
anova_data = np.array(anova_data)
anova_groups = np.array(anova_groups)

# Perform one-way ANOVA
ab_data = anova_data[anova_groups == "AB"]
bc_data = anova_data[anova_groups == "BC"]
cd_data = anova_data[anova_groups == "CD"]
de_data = anova_data[anova_groups == "DE"]

# Print counts for each group and total
print("\nObservation Counts for ANOVA:")
print(f"AB group: {len(ab_data)} observations")
print(f"BC group: {len(bc_data)} observations")
print(f"CD group: {len(cd_data)} observations")
print(f"DE group: {len(de_data)} observations")
total_observations = len(anova_data)
print(f"Total observations: {total_observations}")
print(f"Degrees of freedom: {len(trial_types) - 1}, {total_observations - len(trial_types)}")

f_stat, p_value = f_oneway(ab_data, bc_data, cd_data, de_data)
print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.4f}")

# Set up data for pairwise comparisons
group_data = {
    "AB": ab_data,
    "BC": bc_data,
    "CD": cd_data,
    "DE": de_data
}

# Define pairwise comparisons
pair_names = [("AB", "BC"), ("BC", "CD"), ("CD", "DE"), ("AB", "CD"), ("BC", "DE"), ("AB", "DE")]

# Number of comparisons for Bonferroni correction
num_comparisons = len(pair_names)

# Perform pairwise t-tests with manual Bonferroni correction
p_values = []
for pair in pair_names:
    group1, group2 = pair
    t_stat, p_raw = ttest_ind(group_data[group1], group_data[group2], equal_var=False)
    
    # Apply Bonferroni correction
    p_bonferroni = min(p_raw * num_comparisons, 1.0)
    p_values.append(p_bonferroni)
    
    print(f"Comparing {group1} vs {group2}: t={t_stat:.4f}, raw p={p_raw:.4f}, adj p={p_bonferroni:.4f}, significant={p_bonferroni < 0.05}")

def create_enhanced_bar_plot(data, x_ticks, errors=None, xlim=None, ylim=None, p_values=None, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    
    # Convert proportions to percentages
    data_percentages = [d * 100 for d in data]
    if errors:
        errors_percentages = [e * 100 for e in errors]
    
    if errors:
        bars = plt.bar(x_ticks, data_percentages, yerr=errors_percentages, capsize=5)
    else:
        bars = plt.bar(x_ticks, data_percentages)
    
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    if xlim:
        plt.xlim(xlim)
    
    if ylim:
        # Convert proportion limits to percentage limits
        plt.ylim((ylim[0] * 100, ylim[1] * 100))
    
    plt.ylim(0, 11)
        
    if p_values:
        bar_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        bar_heights = [bar.get_height() for bar in bars]
        max_height = max(bar_heights)
        
        pair_indices = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
        pair_names = ["AB-BC", "BC-CD", "CD-DE", "AB-CD", "BC-DE", "AB-DE"]
        
        height_offset = 0.05
        for i, (idx1, idx2) in enumerate(pair_indices):
            if p_values[i] < 0.05:
                # Determine significance marker
                if p_values[i] < 0.001:
                    sig_marker = '***'
                elif p_values[i] < 0.01:
                    sig_marker = '**'
                else:
                    sig_marker = '*'
                
                # Set height for this comparison line
                line_height = max_height * (1.05 + (i * height_offset))
                
                # Draw the line and marker
                plt.plot([bar_positions[idx1], bar_positions[idx2]], [line_height, line_height], color='black')
                #plt.text((bar_positions[idx1] + bar_positions[idx2]) / 2, line_height + (max_height * 0.02), 
                         #sig_marker, ha='center', va='bottom', fontsize=24)
    
    plt.tight_layout()
    plt.show()

# Create the plot with error bars and significance indicators
create_enhanced_bar_plot(
    data=proportions,
    x_ticks=trial_types,
    errors=standard_errors,
    title="VTE Percentage for Trial Type",
    xlabel="Trial Types",
    ylabel="Percentage of VTEs",
    p_values=p_values
)