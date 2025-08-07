import os
import re
import pandas as pd

from config.paths import paths
from utilities import conversion_utils

base_path = paths.vte_values
model_path = paths.preprocessed_data_model

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
    
    rat_df = []
    rat_path = os.path.join(base_path, rat)
    
    day_folders = [day for day in os.listdir(rat_path) if ".DS_Store" not in day]
    days = sorted(day_folders, key=extract_day_number)
    
    for day in days:
        day_df = []
        day_path = os.path.join(rat_path, day)
        for root, _, files in os.walk(day_path):
            for file in files:
                if "zIdPhi_day_" not in file:
                    continue

                file_path = os.path.join(root, file)
                trajectory_csv = pd.read_csv(file_path)
                
                for index, row in trajectory_csv.iterrows():
                    # get other element
                    first_element = row["Choice"]
                    is_correct = row["Correct"]
                    second_element = conversion_utils.get_other_element(first_element, is_correct)
                    
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
                    day_number = extract_day_number(day)
                    
                    day_df.append({"ID": traj_id,
                                   "Day": day_number, 
                                   "first": first_index, 
                                   "second": second_index, 
                                   "correct": correct,
                                   "VTE": is_VTE,
                                   "zlength": zlength})
        
        
        # sort day df by trajectory
        day_df.sort(key=lambda x: extract_trajectory_number(x["ID"]))
        
        # 2. Replace ID with just the trajectory number
        for item in day_df:
            item["ID"] = extract_trajectory_number(item["ID"])
        
        save_dir = os.path.join(model_path, rat)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        
        save_path = os.path.join(save_dir, f"{day}.csv")
        save_df = pd.DataFrame(day_df)
        save_df.to_csv(save_path)
        
        rat_df.extend(day_df)
    
    if rat_df:
        combined_rat_df = pd.DataFrame(rat_df)
        combined_save_path = os.path.join(save_dir, f"{rat}_all_days.csv")
        combined_rat_df.to_csv(combined_save_path)