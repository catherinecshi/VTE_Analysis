import os
import numpy as np
import pandas as pd

from src import helper
from src import plotting
from src import performance_analysis

# get proportion of VTEs for each trial type for each rat
proportion_of_VTEs: dict[str, dict] = {}
vte_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values")
for rat in os.listdir(vte_path):
    if ".DS_Store" in rat:
        continue
    
    trial_type_to_VTEs: dict[int, int] = {}
    trial_type_to_non_VTEs: dict[int, int] = {}
    rat_path = os.path.join(vte_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file) # should be the zIdPhi file now
            zIdPhi_csv = pd.read_csv(file_path)
            
            mean_zIdPhi = np.mean(zIdPhi_csv["zIdPhi"])
            std_zIdPhi = np.std(zIdPhi_csv["zIdPhi"])
            VTE_threshold = mean_zIdPhi + (std_zIdPhi * 1.5)
            
            for index, row in zIdPhi_csv.iterrows():
                zIdPhi = row["zIdPhi"]
                trial_type = row["Trial_Type"]
                
                is_VTE = zIdPhi > VTE_threshold
                
                if is_VTE:
                    if trial_type in trial_type_to_VTEs:
                        trial_type_to_VTEs[trial_type] += 1
                    else:
                        trial_type_to_VTEs[trial_type] = 1
                else:
                    if trial_type in trial_type_to_non_VTEs:
                        trial_type_to_non_VTEs[trial_type] += 1
                    else:
                        trial_type_to_non_VTEs[trial_type] = 1
    
    proportion_of_VTEs_rat: dict[int, float] = {}
    for trial_type, number_of_VTEs in trial_type_to_VTEs.items():
        number_of_non_VTEs = trial_type_to_non_VTEs[trial_type]
        
        VTE_proportion = number_of_VTEs / (number_of_VTEs + number_of_non_VTEs)
        proportion_of_VTEs_rat[trial_type] = VTE_proportion
    
    proportion_of_VTEs[rat] = proportion_of_VTEs_rat

# get number of days until criteria for each trial type for each rat
all_rats_performances = performance_analysis.create_all_rats_performance()
days_until_criteria = performance_analysis.days_until_criteria(all_rats_performances)

# match days until criteria to VTEs
x = [] # days until criteria
y = [] # VTE proportion
for rat, day_items in days_until_criteria.items():
    try:
        proportion_of_VTEs_rat = proportion_of_VTEs[rat]
    except KeyError:
        print(f"can't find {rat} in proportion of VTEs")
    
    trial_types = day_items.keys()
    for trial_type in trial_types:
        days = days_until_criteria[rat][trial_type]
        VTE_proportion = proportion_of_VTEs_rat[trial_type]
        
        x.append(days)
        y.append(VTE_proportion)

plotting.create_scatter_plot(x, y, "VTEs vs Days Until Criteria", 
                             "Number of Days until Criteria was Reached", "VTEs for Trial Type")