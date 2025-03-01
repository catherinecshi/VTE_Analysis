import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import helper
from src import statistics

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

# Run ANOVA and create plot
fig, ax, stats_results = statistics.plot_one_way_anova(
    data_groups,
    group_labels=group_labels,
    title="VTE Relative to Mistake Trials",
    xlabel="Trials from Mistake",
    ylabel="VTE Percentage (%)"
)
plt.show()