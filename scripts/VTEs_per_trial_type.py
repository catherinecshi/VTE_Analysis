import os
import pandas as pd

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
    number_VTEs = VTE_trials[trial_type]
    number_trials = all_trials[trial_type]
    
    VTE_proportion = number_VTEs / number_trials
    VTEs_for_trial_type[trial_type] = VTE_proportion

x_ticks = ["AB", "BC", "CD", "DE", "EF"] # trial types
data = VTEs_for_trial_type.values() # number of VTEs
plotting.create_bar_plot(data, x_ticks, "VTE Proportion for Trial Type", "Trial Types", "Proportion of VTEs")