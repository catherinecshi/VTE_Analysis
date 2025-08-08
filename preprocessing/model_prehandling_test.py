import os
import re
import pandas as pd
import numpy as np
from scipy import stats

from config import settings
from config.paths import paths
from utilities import conversion_utils

base_path = paths.vte_values / "inferenceTesting"
model_path = paths.preprocessed_data_model / "inferenceTesting"

def extract_day_number(folder_name):
    match = re.search(r'Day(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return float('inf')  # Put folders without proper naming at the end

def extract_trajectory_number(traj_id):
    parts = traj_id.split('_')
    if len(parts) > 0:
        # try to convert the last part to an integer
        try:
            return int(parts[-1])
        except ValueError:
            print("no trajectory number??")
            return float('inf') # put it at the end if conversion fails
    print("no trajectory number?")
    return float('inf') # if no parts, put at the end

# make a better file for the data for legeibility
for rat in os.listdir(base_path):
    if ".DS_Store" in rat or ".csv" in rat:
        continue
    
    settings.update_rat(rat)
    rat_df = []
    rat_path = os.path.join(base_path, rat)
    
    # Collect all data first
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "_zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file)
            trajectory_csv = pd.read_csv(file_path)
            
            for index, row in trajectory_csv.iterrows():
                # get other element
                trial_type = row["Trial Type"]
                is_correct = row["Correct"]
                first_element, second_element = conversion_utils.type_to_elements(trial_type, is_correct)
                
                # now convert them both to indices
                first_index = conversion_utils.letter_to_indices(first_element)
                second_index = conversion_utils.letter_to_indices(second_element)
                
                # and convert is correct to 1s and 0s
                if is_correct:
                    correct = 1
                else:
                    correct = 0
                
                # get everything else for the df
                traj_id = row["ID"]
                zlength = row["zLength"]
                is_VTE = row["VTE"]
                
                rat_df.append({"ID": traj_id,
                                "first": first_index, 
                                "second": second_index, 
                                "correct": correct,
                                "VTE": is_VTE,
                                "zlength": zlength})
    
    # sort by trajectory
    rat_df.sort(key=lambda x: extract_trajectory_number(x["ID"]))
    
    # Replace ID with just the trajectory number
    for item in rat_df:
        item["ID"] = extract_trajectory_number(item["ID"])
    
    # Apply transformations if we have data
    if rat_df:
        # Extract all zLength values for this rat
        zlength_values = np.array([item["zlength"] for item in rat_df])
        
        # Calculate per-rat statistics
        zlength_min = np.min(zlength_values)
        zlength_max = np.max(zlength_values)
        zlength_mean = np.mean(zlength_values)
        zlength_std = np.std(zlength_values)
        
        # Apply transformations to each data point
        for item in rat_df:
            zlength = item["zlength"]
            
            # Min-max scaling: (x - min) / (max - min)
            if zlength_max != zlength_min:
                item["zlength_minmax"] = (zlength - zlength_min) / (zlength_max - zlength_min)
            else:
                item["zlength_minmax"] = 0.5  # If all values are the same
            
            # Normal CDF: Φ((x - μ) / σ)
            if zlength_std > 0:
                item["zlength_norm_cdf"] = stats.norm.cdf((zlength - zlength_mean) / zlength_std)
            else:
                item["zlength_norm_cdf"] = 0.5  # If no variance
            
            # Sigmoid: 1 / (1 + exp(-(x - μ) / σ))
            if zlength_std > 0:
                item["zlength_sigmoid"] = 1 / (1 + np.exp(-(zlength - zlength_mean) / zlength_std))
            else:
                item["zlength_sigmoid"] = 0.5  # If no variance
            
            # Remove original zlength
            del item["zlength"]
    
    save_dir = os.path.join(model_path, rat)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_path = os.path.join(save_dir, f"{rat}.csv")
    save_df = pd.DataFrame(rat_df)
    save_df.to_csv(save_path)