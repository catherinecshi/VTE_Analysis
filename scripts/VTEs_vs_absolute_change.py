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
                VTE_count = 0
                for index, row in group.iterrows():
                    traj_id = row["ID"]
                    traj_number = traj_id.split("_")[2]
                    is_VTE = row["zIdPhi"] > VTE_threshold
                    
                    if is_VTE:
                        VTE_trajectories.append(traj_id)
                        VTE_count += 1
                
                if VTE_count > 20:
                    print(rat, day)
                    continue
                
                match = re.search(r"\d+", day)
                if match:
                    day_number = int(match.group())
                else:
                    print(f"no day number found for {rat} on {day}")
                    continue
                VTE_info = {"rat": rat, "day": day_number, "VTEs": VTE_count}
                VTE_infos.append(VTE_info)

VTE_df = pd.DataFrame(VTE_infos)
     
all_rats_performances = performance_analysis.get_all_rats_performance()
performance_changes = performance_analysis.save_all_perf_changes(all_rats_performances)
absolute_changes = performance_changes.copy()
absolute_changes["perf_change"] = absolute_changes["perf_change"].abs()

VTEs_vs_learning = pd.merge(VTE_df, performance_changes)
absolute_VTEs_vs_learning = pd.merge(VTE_df, absolute_changes)

# plot general change
generic_plots.create_box_and_whisker_plot(VTEs_vs_learning, "VTEs", "perf_change",
                                          title="Change in Performance against VTEs",
                                          xlabel="VTE Count", ylabel="Change in Performance")

generic_plots.create_scatter_plot(VTEs_vs_learning["VTEs"], VTEs_vs_learning["perf_change"],
                                  "Change in Performance against VTEs", "VTE Count",
                                  "Change in Performance")

# plot absolute change
generic_plots.create_box_and_whisker_plot(absolute_VTEs_vs_learning, "VTEs", "perf_change",
                                          title="Absolute Change in Performance against VTEs",
                                          xlabel="VTE Count", ylabel="Absolute Change in Performance")

generic_plots.create_scatter_plot(absolute_VTEs_vs_learning["VTEs"], VTEs_vs_learning["perf_change"],
                                  "Absolute Change in Performance against VTEs", "VTE Count",
                                  "Absolute Change in Performance")