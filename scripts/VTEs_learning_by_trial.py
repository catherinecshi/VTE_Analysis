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

# for analysis of trials after VTE Trials
VTEs_correct = 0
VTEs_incorrect = 0
non_VTEs_correct = 0
non_VTEs_incorrect = 0
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
            VTE_threshold = mean_zIdPhi + std_zIdPhi
            
            # sort by trajectory ID
            zIdPhi_csv[["rat", "day", "traj_ID"]] = zIdPhi_csv["ID"].str.extract(r"(\w+)_Day(\d+)_([\d]+)")
            zIdPhi_csv["traj_ID"] = zIdPhi_csv["traj_ID"].astype(int)
            
            zIdPhi_sorted = zIdPhi_csv.sort_values(by=["day", "traj_ID"])
            
            # go through each day independently
            grouped_by_day = zIdPhi_sorted.groupby("Day")
            for day, group in grouped_by_day:
                for index, row in group.iterrows():
                    traj_id = row["ID"]
                    traj_number = traj_id.split("_")[2]
                    is_VTE = row["zIdPhi"] > VTE_threshold
                    
                    pos = group.index.get_loc(index)
                    
                    try:
                        if pos + 1 < len(group):
                            next_row = group.iloc[pos + 1]
                        else:
                            continue
                        next_traj_id = next_row["ID"]
                        next_traj_number = next_traj_id.split("_")[2]
                        if not(int(next_traj_number) == int(traj_number) + 1): # they are not consecutive
                            print(f"cannot find next trajectory for {traj_id}")
                            raise KeyError
                        
                        next_performance = trial_performance[next_traj_id]
                    except Exception as e:
                        print(f"Error {e} for {traj_id}")
                        continue
                    
                    if next_performance is True:
                        if is_VTE:
                            VTEs_correct += 1
                        else:
                            non_VTEs_correct += 1
                    else:
                        if is_VTE:
                            VTEs_incorrect += 1
                        else:
                            non_VTEs_incorrect += 1

VTEs_proportion = VTEs_correct / (VTEs_correct + VTEs_incorrect)
non_VTEs_proportion = non_VTEs_correct / (non_VTEs_correct + non_VTEs_incorrect)
data = [VTEs_proportion, non_VTEs_proportion]
x_ticks = ["VTEs", "non-VTEs"]

plotting.create_bar_plot(data, x_ticks, "VTE Influence on Next Trial Accuracy",
                         "VTEs/Non-VTEs", "Likelihood of Getting Next Trial Correct")

# For analysis of trials before VTE trials
VTEs_correct = 0
VTEs_incorrect = 0
non_VTEs_correct = 0
non_VTEs_incorrect = 0
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
            VTE_threshold = mean_zIdPhi + std_zIdPhi
            
            # sort by trajectory ID
            zIdPhi_csv[["rat", "day", "traj_ID"]] = zIdPhi_csv["ID"].str.extract(r"(\w+)_Day(\d+)_([\d]+)")
            zIdPhi_csv["traj_ID"] = zIdPhi_csv["traj_ID"].astype(int)
            
            zIdPhi_sorted = zIdPhi_csv.sort_values(by=["day", "traj_ID"])
            
            # go through each day independently
            grouped_by_day = zIdPhi_sorted.groupby("Day")
            for day, group in grouped_by_day:
                for index, row in group.iterrows():
                    traj_id = row["ID"]
                    traj_number = traj_id.split("_")[2]
                    is_VTE = row["zIdPhi"] > VTE_threshold
                    
                    pos = group.index.get_loc(index)
                    
                    try:
                        if pos - 1 < len(group):
                            next_row = group.iloc[pos - 1]
                        else:
                            continue
                        next_traj_id = next_row["ID"]
                        next_traj_number = next_traj_id.split("_")[2]
                        if not(int(next_traj_number) == int(traj_number) - 1): # they are not consecutive
                            print(f"cannot find next trajectory for {traj_id}")
                            raise KeyError
                        
                        next_performance = trial_performance[next_traj_id]
                    except Exception as e:
                        print(f"Error {e} for {traj_id}")
                        continue
                    
                    if next_performance is True:
                        if is_VTE:
                            VTEs_correct += 1
                        else:
                            non_VTEs_correct += 1
                    else:
                        if is_VTE:
                            VTEs_incorrect += 1
                        else:
                            non_VTEs_incorrect += 1

VTEs_proportion = VTEs_correct / (VTEs_correct + VTEs_incorrect)
non_VTEs_proportion = non_VTEs_correct / (non_VTEs_correct + non_VTEs_incorrect)
data = [VTEs_proportion, non_VTEs_proportion]
x_ticks = ["VTEs", "non-VTEs"]

plotting.create_bar_plot(data, x_ticks, "Previous Trial Performance Influence on VTE Probability",
                         "VTEs/Non-VTEs", "Likelihood of Previous Trial Being Correct")