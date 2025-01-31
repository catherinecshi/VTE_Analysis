import os
import math
import starbars
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

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
                
                mistakes = []
                for index, row in trajectory_csv.iterrows():
                    traj_id = row["ID"]
                    parts = traj_id.split("_")
                    traj_number = parts[2]
                    is_correct = row["Correct"]
                    
                    if not is_correct:
                        mistakes.append(int(traj_number))
                
                if not mistakes:
                    continue
                else:
                    for index, row in trajectory_csv.iterrows():
                        traj_id = row["ID"]
                        parts = traj_id.split("_")
                        traj_number = parts[2]
                        larger_numbers = [num for num in mistakes if num > int(traj_number)]
                        
                        if int(traj_number) in mistakes:
                            until_next_mistake[traj_id] = 0
                        elif larger_numbers:
                            closest_number = min(larger_numbers)
                            if (closest_number - int(traj_number)) == 0:
                                print(traj_id)
                            until_next_mistake[traj_id] = closest_number - int(traj_number)


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
                        #print(f"{traj_id} not found in trials_until_mistake")
                        continue
                    
                    if is_VTE:
                        trues.append(trials_until_mistake)
                    else:
                        falses.append(trials_until_mistake)
            
trues_count_no_trials_before = {-value: trues.count(value) for value in set(trues)} # how many trues for specific # trials
falses_count_no_trials_before = {-value: falses.count(value) for value in set(falses)}

x_ticks_many = []
data_many = []
for key in trues_count_no_trials_before.keys():
    x_ticks_many.append(key)
    data_many.append(trues_count_no_trials_before[key] / (trues_count_no_trials_before[key] + falses_count_no_trials_before[key]))

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
                        if is_VTE:
                            trues.append(0)
                        else:
                            falses.append(0)
                        
                        continue
                    
                    if is_VTE and since_last_mistake is not None:
                        trues.append(since_last_mistake)
                    elif not is_VTE and since_last_mistake is not None:
                        falses.append(since_last_mistake)
            
trues_count_no_trials_next = {value: trues.count(value) for value in set(trues)} # how many trues for specific # trials
falses_count_no_trials_next = {value: falses.count(value) for value in set(falses)}

for key in trues_count_no_trials_next.keys():
    if key == 0:
        print("skipping 0")
        continue

    x_ticks_many.append(key)
    data_many.append(trues_count_no_trials_next[key] / (trues_count_no_trials_next[key] + falses_count_no_trials_next[key]))
    
errors = []
for i, x_tick in enumerate(x_ticks_many):
    if x_tick < 0:  # for trials before mistake
        vte_count = trues_count_no_trials_before[x_tick]
        total_count = trues_count_no_trials_before[x_tick] + falses_count_no_trials_before[x_tick]
    else:  # for trials after mistake
        vte_count = trues_count_no_trials_next[x_tick]
        total_count = trues_count_no_trials_next[x_tick] + falses_count_no_trials_next[x_tick]
    
    sem = helper.get_sem(vte_count, total_count)
    errors.append(sem)

plotting.create_bar_plot(data_many, x_ticks_many, errors=errors, xlim=(-2.5, 2.5), ylim=(0, 0.1),
                         title="VTE Percentage for Number of Trials Around a Mistake",
                         xlabel="Number of Trials Around Mistake", ylabel="Percentage of VTEs (%)")