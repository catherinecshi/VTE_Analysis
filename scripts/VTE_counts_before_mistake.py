import os
import numpy as np
import pandas as pd

from src import helper
from src import plotting

base_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Values")

until_next_mistake = {} # {traj_id: # trials until next mistake}
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
                trajectory_reversed = trajectory_csv.iloc[::-1] # reverse for before mistake
                
                last_mistake = None
                trials_until_mistake = None
                for index, row in trajectory_reversed.iterrows():
                    traj_id = row["ID"]
                    parts = traj_id.split("_")
                    traj_number = parts[2]
                    is_correct = row["Correct"]
                    
                    if last_mistake is not None: 
                        # get the number of trials until the next mistake
                        trials_until_mistake = last_mistake - int(traj_number) # temp variable
                    
                    if not is_correct:
                        last_mistake = int(traj_number)
                        until_next_mistake[traj_id] = 0

                    if trials_until_mistake is not None:
                        until_next_mistake[traj_id] = trials_until_mistake


trues = [] # trials until mistake for VTE trials
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
            
            # go through each day independently
            grouped_by_day = zIdPhi_csv.groupby("Day")
            for day, group in grouped_by_day:
                last_mistake = None
                since_last_mistake = None
                for index, row in group.iterrows():
                    traj_id = row["ID"]
                    is_VTE = row["zIdPhi"] > VTE_threshold
                    
                    try:
                        trials_until_mistake = until_next_mistake[traj_id]
                    except KeyError:
                        print(f"{traj_id} not found in trials_until_mistake")
                    
                    if is_VTE:
                        trues.append(trials_until_mistake)
                    else:
                        falses.append(trials_until_mistake)
            
trues_count_no_trials = {value: trues.count(value) for value in set(trues)} # how many trues for specific # trials
falses_count_no_trials = {value: falses.count(value) for value in set(falses)}

VTE_vs_last_mistake = {} # {number of trials until mistake: VTE_proportion}
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
plotting.create_bar_plot(data, x_ticks, "VTEs before Mistakes",
                         "Number of Trials Before a Mistake will Occur", "Proportion of VTE Trials")
