import os
import numpy as np
import pandas as pd

from src import helper
from src import plotting

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")

trial_performance = {}
for rat in os.listdir(base_path):
    if ".DS_Store" in rat:
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
                
                for index, row in trajectory_csv.iterrows():
                    traj_id = row["ID"]
                    is_correct = row["Correct"]
                    
                    if is_correct:
                        trial_performance[traj_id] = True
                    else:
                        trial_performance[traj_id] = False

trues = []
falses = []
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
            
            # sort by trajectory ID
            zIdPhi_csv[["rat", "day", "traj_ID"]] = zIdPhi_csv["ID"].str.extract(r"(\w+)_Day(\d+)_([\d]+)")
            zIdPhi_csv["traj_ID"] = zIdPhi_csv["traj_ID"].astype(int)
            
            zIdPhi_sorted = zIdPhi_csv.sort_values(by=["day", "traj_ID"])
            
            # go through each day independently
            grouped_by_day = zIdPhi_sorted.groupby("Day")
            for day, group in grouped_by_day:
                last_mistake = None
                since_last_mistake = None
                for index, row in group.iterrows():
                    traj_id = row["ID"]
                    parts = traj_id.split("_")
                    traj_number = parts[2]
                    is_VTE = row["zIdPhi"] > VTE_threshold

                    if traj_id in trial_performance: # get performance
                        is_correct = trial_performance[traj_id]
                    else:
                        print(f"traj id {traj_id} not found in zIdPhi file")
                        continue
                    
                    # check for number of trials since last mistake
                    if last_mistake is not None:
                        since_last_mistake = int(traj_number) - last_mistake
                    
                    if not is_correct: # update
                        last_mistake = int(traj_number)
                    
                    if is_VTE and since_last_mistake is not None:
                        trues.append(since_last_mistake)
                    elif not is_VTE and since_last_mistake is not None:
                        falses.append(since_last_mistake)
            
trues_count_no_trials = {value: trues.count(value) for value in set(trues)} # how many trues for specific # trials
falses_count_no_trials = {value: falses.count(value) for value in set(falses)}

VTE_vs_last_mistake = {}
for no_trials in trues_count_no_trials.keys():
    if no_trials in falses_count_no_trials:
        trues_count = trues_count_no_trials[no_trials]
        falses_count = falses_count_no_trials[no_trials]
        
        # make sure there is enough trials for statistical analysis
        if trues_count + falses_count < 10:
            continue
        elif no_trials > 40:
            continue
        
        VTE_proportion = trues_count / (trues_count + falses_count)
        VTE_vs_last_mistake[no_trials] = VTE_proportion
    else:
        VTE_proportion = 1
        VTE_vs_last_mistake[no_trials] = VTE_proportion

        print(f"something weird, all trials are VTEs for {no_trials}")

x_ticks = VTE_vs_last_mistake.keys()
data = VTE_vs_last_mistake.values()
plotting.create_bar_plot(data, x_ticks, "VTEs after Mistakes",
                         "Number of Trials Prior a Mistake Occurred", "Proportion of VTE Trials")
