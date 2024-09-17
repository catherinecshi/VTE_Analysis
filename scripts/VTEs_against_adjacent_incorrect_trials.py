# not done right now

import os
import pandas as pd

from src import helper
from src import plotting

# pylint: disable=consider-using-dict-items

vte_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values")

vte_proportions = [] # y values
wrong_proportions = [] # x values
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
            
            vte_mean = zIdPhi_csv["zIdPhi"].mean()
            vte_std = zIdPhi_csv["zIdPhi"].std()
            vte_threshold = vte_mean + (vte_std * 1.5)
            
            vte_counts = {} # {day + trial type: vte_trials}
            correct_counts = {} # {day + trial type: correct_trials}
            total_trials = {} # {day + trial type: total_trials}
            for index, row in zIdPhi_csv.iterrows():
                zIdPhi = row["zIdPhi"]
                trial_type = row["Trial_Type"]
                choice = row["Choice"]
                day = row["Day"]
                day_and_trial_type = day + "_" + str(trial_type)
                
                is_VTE = zIdPhi > vte_threshold

                # getting whether trial is correct from choice & trial type
                is_correct = helper.choice_to_correctness(trial_type, choice)
                
                # add to counts
                if day_and_trial_type in vte_counts and is_VTE:
                    vte_counts[day_and_trial_type] += 1
                elif is_VTE:
                    vte_counts[day_and_trial_type] = 1
                
                if day_and_trial_type in correct_counts and is_correct:
                    correct_counts[day_and_trial_type] += 1
                elif is_correct:
                    correct_counts[day_and_trial_type] = 1
                
                if day_and_trial_type in total_trials:
                    total_trials[day_and_trial_type] += 1
                else:
                    total_trials[day_and_trial_type] = 1
            
            # get the values for x and y
            for day_and_trial_type in total_trials.keys():
                total_trial_count = total_trials[day_and_trial_type]
                
                # exclude days and trials where there were too few samples
                if total_trial_count < 5:
                    continue
                
                # separate thing incase there were no vtes in the day
                if day_and_trial_type in vte_counts:
                    vte_trial_count = vte_counts[day_and_trial_type]
                else:
                    vte_trial_count = 0
                
                # same with correct trials
                if day_and_trial_type in correct_counts:
                    correct_trial_count = correct_counts[day_and_trial_type]
                else:
                    correct_trial_count = 0
                
                vte_proportion = vte_trial_count / total_trial_count
                correct_proportion = correct_trial_count / total_trial_count
                wrong_proportion = 1 - correct_proportion

                vte_proportions.append(vte_proportion)
                wrong_proportions.append(wrong_proportion)
            
plotting.create_scatter_plot(wrong_proportions, vte_proportions, "VTEs with Incorrect Trials",
                             "Proportion of Incorrect Trials per Day per Trial Type",
                             "VTE Proportion")
                