import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import helper
from src import plotting
from src import statistics
from src import data_processing
from src import performance_analysis

# Load data structure
data_path = os.path.join(helper.BASE_PATH, "data", "VTE_Data")
data_structure = data_processing.load_data_structure(data_path)

# Get days since new arm added
days_since_new_arm = performance_analysis.get_days_since_new_arm(data_path, data_structure)
days_since_new_arm["trials_available"] = days_since_new_arm["trials_available"].apply(lambda x: [int (y) for y in x])
days_since_new_arm = days_since_new_arm.astype({"rat": "str", "day": "int", "arm_added": "bool", "days_since_new_arm": "int"})

# Collect VTE data
vtes_during_volatility = []
vte_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values")
for rat in os.listdir(vte_path):
    if ".DS" in rat:
        continue

    rat_path = os.path.join(vte_path, rat)
    for root, _, files in os.walk(rat_path):
        for f in files:
            if "zIdPhi" not in f:
                continue
            
            file_path = os.path.join(root, f)
            zIdPhi_df = pd.read_csv(file_path)
            
            grouped_by_day = zIdPhi_df.groupby(by="Day")
            for day, day_group in grouped_by_day:
                vte_trials = day_group[day_group["zIdPhi"] >= 1.5]
                non_vte_trials = day_group[day_group["zIdPhi"] < 1.5]
                no_vtes = len(vte_trials)
                no_non_vtes = len(non_vte_trials)
                perc_vtes = (no_vtes / (no_vtes + no_non_vtes)) * 100
                
                match = re.search(r"\d+", day)
                if match:
                    day_number = int(match.group())
                    
                corresponding_row = days_since_new_arm[(days_since_new_arm["rat"] == rat) &
                                                       (days_since_new_arm["day"] == day_number)]
                
                if not corresponding_row.empty:
                    no_days = corresponding_row["days_since_new_arm"].values[0]
                
                vtes_during_volatility.append({"rat": rat, "day": day,
                                              "perc_vtes": perc_vtes, "no_days": no_days})
                
# Create DataFrame and save to CSV
vtes_during_volatility_df = pd.DataFrame(vtes_during_volatility)
vtes_during_volatility_df.to_csv(os.path.join(vte_path, "vtes_during_volatility.csv"))

def prepare_vte_data_for_anova(vtes_volatility, min_day=0, max_day=5):
    """
    Prepares VTE data for one-way ANOVA analysis by grouping percentage of VTEs by number of days.
    Filters data to include only specified range of days.
    
    Parameters:
        vtes_during_volatility (list): List of dictionaries containing VTE data
        min_day (int): Minimum number of days to include (inclusive)
        max_day (int): Maximum number of days to include (inclusive)
        
    Returns:
        tuple: (data_groups, group_labels)
        - data_groups: List of arrays containing perc_vtes for each no_days group
        - group_labels: List of labels for each group
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(vtes_volatility)
    
    # Filter for specified day range
    df = df[(df['no_days'] >= min_day) & (df['no_days'] <= max_day)]
    
    # Group by number of days
    grouped_data = df.groupby('no_days')['perc_vtes'].apply(list).to_dict()
    
    # Sort by number of days
    sorted_days = sorted(grouped_data.keys())
    
    # Create data groups and labels
    data_groups = [grouped_data[day] for day in sorted_days]
    group_labels = [f'Day {day}' for day in sorted_days]
    
    return data_groups, group_labels

def analyze_vte_anova(vtes_volatility, min_day=0, max_day=5):
    """
    Performs one-way ANOVA analysis on VTE data and creates a visualization.
    
    Parameters:
        vtes_during_volatility (list): List of dictionaries containing VTE data
        min_day (int): Minimum number of days to include (inclusive)
        max_day (int): Maximum number of days to include (inclusive)
    
    Returns:
        tuple: (fig, ax, stats_results)
    """
    # Prepare data for ANOVA
    data_groups, group_labels = prepare_vte_data_for_anova(
        vtes_volatility,
        min_day=min_day,
        max_day=max_day
    )
    
    # Run ANOVA and create plot
    fig, ax, stats_results = statistics.plot_one_way_anova_line(
        data_groups,
        group_labels=group_labels,
        title=f'VTE Percentage by Days Since New Arm',
        xlabel='Days Since New Arm',
        ylabel='VTE Percentage (%)'
    )
    
    return fig, ax, stats_results

# Calculate the mean and SEM for each day
mean_perc_vtes = vtes_during_volatility_df.groupby("no_days")["perc_vtes"].mean()
sem_perc_vtes = vtes_during_volatility_df.groupby("no_days")["perc_vtes"].sem()

# Define fictional expected data
x_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
exp_mean_perc_vtes = np.array([8, 10, 9, 8.5, 7.9, 7.5, 7, 6.5, 6, 5.7, 5.5, 5.3, 5.1, 4.7, 4.5, 4.3])
exp_sem_perc_vtes = np.array([0.5, 0.8, 0.7, 0.4, 0.36, 0.3, 0.28, 0.26, 0.25, 0.22, 0.22, 0.21, 0.20, 0.18, 0.18, 0.2])

# Updated function to plot both sets of data
def create_combined_line_plot(x1, y1, sem1, x2, y2, sem2, xlim=None, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    
    # Plot first dataset
    plt.errorbar(x1, y1, yerr=sem1, fmt="-o", capsize=5, label="Actual VTEs")
    
    # Plot second dataset
    plt.errorbar(x2, y2, yerr=sem2, fmt="-s", capsize=5, label="Expected VTEs")
    
    if xlim:
        plt.xlim(0, xlim)
    
    # Set plot properties
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(False)
    plt.show()

# Call the function to plot both datasets
create_combined_line_plot(mean_perc_vtes.index, mean_perc_vtes, sem_perc_vtes,
                          x_values, exp_mean_perc_vtes, exp_sem_perc_vtes,
                          xlim=8,
                          title="VTEs during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="% VTE Trials")


plotting.create_line_plot(mean_perc_vtes.index, mean_perc_vtes, sem_perc_vtes,
                          xlim=(0, 5), title="VTEs during Volatility",
                          xlabel="Number of Days Since New Arm Added",
                          ylabel="% VTE Trials")

fig, ax, stats_results = analyze_vte_anova(vtes_during_volatility, min_day=0, max_day=5)
plt.show()