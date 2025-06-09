import os
import re
import numpy as np
import pandas as pd

from analysis import performance_analysis
from visualization import generic_plots

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")

VTE_trajectories = []
VTE_infos = []
for rat in os.listdir(base_path):
    if ".DS_Store" in rat:
        continue
    
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
            
            # sort by trajectory ID
            zIdPhi_csv[["rat", "day", "traj_ID"]] = zIdPhi_csv["ID"].str.extract(r"(\w+)_Day(\d+)_([\d]+)")
            zIdPhi_csv["traj_ID"] = zIdPhi_csv["traj_ID"].astype(int)
            
            zIdPhi_sorted = zIdPhi_csv.sort_values(by=["day", "traj_ID"])
            
            # go through each day independently
            grouped_by_day = zIdPhi_sorted.groupby("Day")
            for day, group in grouped_by_day:
                VTE_count = {}
                total_count = {}
                for index, row in group.iterrows():
                    traj_id = row["ID"]
                    traj_number = traj_id.split("_")[2]
                    trial_type = row["Trial_Type"]
                    is_VTE = row["zIdPhi"] > VTE_threshold
                    
                    if is_VTE:
                        VTE_trajectories.append(traj_id)
                        
                        # make sure the trial type exists in the dictionary
                        if trial_type not in VTE_count:
                            VTE_count[trial_type] = 1
                        else:
                            VTE_count[trial_type] += 1
                    
                    # get total counts for proportionality
                    if trial_type not in total_count:
                        total_count[trial_type] = 1
                    else:
                        total_count[trial_type] += 1
                
                # get day number
                match = re.search(r"\d+", day)
                if match:
                    day_number = int(match.group())
                else:
                    print(f"no day number found for {rat} on {day}")
                    continue
                
                for trial_type, trial_counts in total_count.items():
                    if trial_type in VTE_count:
                        vte_counts = VTE_count[trial_type]
                    else:
                        vte_counts = 0
                    vte_proportion = vte_counts / trial_counts
                    VTE_info = {"rat": rat, "day": day_number, "trial_type": trial_type, "VTEs": vte_proportion}
                    VTE_infos.append(VTE_info)

VTE_df = pd.DataFrame(VTE_infos)
all_rats_performances = performance_analysis.get_all_rats_performance()
performance_changes = performance_analysis.save_all_perf_changes(all_rats_performances)
absolute_changes = performance_changes.copy()
absolute_changes["perf_change"] = absolute_changes["perf_change"].abs()

VTEs_vs_learning = pd.merge(VTE_df, performance_changes)
absolute_VTEs_vs_learning = pd.merge(VTE_df, absolute_changes)

# setup for populational scatter plots
grouped_by_trial_type = VTEs_vs_learning.groupby("trial_type")
x_vals = []
y_vals = []
labels = []
for trial_type, trial_group in grouped_by_trial_type:
    vtes = trial_group["VTEs"].tolist()
    perf_changes = trial_group["perf_change"].tolist()
    
    x_vals.append(vtes)
    y_vals.append(perf_changes)
    labels.append(trial_type)

# plot general change
generic_plots.create_populational_scatter_plot(x_vals, y_vals, title="Change in Performance against VTEs",
                                          xlabel="Change in Performance", ylabel="VTE Proportion", labels=labels)

# setup for populational scatter plots
grouped_by_trial_type = absolute_VTEs_vs_learning.groupby("trial_type")
x_vals = []
y_vals = []
labels = []
for trial_type, trial_group in grouped_by_trial_type:
    vtes = trial_group["VTEs"].tolist()
    perf_changes = trial_group["perf_change"].tolist()
    
    x_vals.append(vtes)
    y_vals.append(perf_changes)
    labels.append(trial_type)

# plot general change
generic_plots.create_populational_scatter_plot(x_vals, y_vals, title="Absolute Change in Performance against VTEs",
                                          xlabel="Absolute Change in Performance", ylabel="VTE Proportion", labels=labels)