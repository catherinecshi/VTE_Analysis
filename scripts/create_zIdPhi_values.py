"""
Main Script to call to get zIdPhi values for data. Requires create_IdPhi_values to have been ran before.
zIdPhi values are created separate of IdPhi values to ensure zscoring across sessions, not within sessions

Procedure:
1. gets the values created from create_IdPhi_values
2. then zscores across sessions for a rat to get zIdPhi values
"""

import os
import logging
import pandas as pd
from datetime import datetime
from scipy.stats import zscore

from src import data_processing

### LOGGING
logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# makes a new log everytime the code runs by checking the time
log_file = datetime.now().strftime("/Users/catpillow/Documents/VTE_Analysis/doc/create_zIdPhi_values_log_%Y%m%d_%H%M%S.txt")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

base_path = "/Users/catpillow/Documents/VTE_Analysis"
dlc_path = os.path.join(base_path, "processed_data", "cleaned_dlc")
data_path = os.path.join(base_path, "data", "VTE_Data")
data_structure = data_processing.load_data_structure(data_path)

vte_path = os.path.join(base_path, "processed_data", "VTE_data")
values_path = os.path.join(base_path, "processed_data", "VTE_values")

IdPhis_across_days = {} # this is so it can be zscored altogether
IdPhis_in_a_day = 0
for rat in os.listdir(values_path):
    if ".DS" in rat or ".csv" in rat:
        continue
    
    rat_path = os.path.join(values_path, rat)
    before_zscore_df = pd.DataFrame()
    for day in os.listdir(rat_path):
        day_path = os.path.join(rat_path, day)
        if ".DS" in day:
            continue
        
        for root, _, files in os.walk(day_path):
            for file in files:
                if "trajectories.csv" not in file:
                    continue
                
                file_path = os.path.join(root, file)
                traj_info = pd.read_csv(file_path, header=0)
                
                if all(col in traj_info.columns for col in ["IdPhi", "Choice", "ID", "Trial Type", "Length"]):
                    rows = {"ID": traj_info["ID"], "Day": day, "Choice": traj_info["Choice"], 
                            "Trial_Type": traj_info["Trial Type"], "IdPhi": traj_info["IdPhi"], "Length": traj_info["Length"]}
                    day_df = pd.DataFrame(rows)
                    before_zscore_df = pd.concat([before_zscore_df, day_df], ignore_index=True)
                else:
                    logging.warning(f"missing columns in {rat} on {day}")
    
    try:
        grouped_by_choice = before_zscore_df.groupby(by="Choice")
    except Exception:
        logging.error(f"error with groupby for {rat}")
    else:
        logging.info("grouping successful")
        
    zscored_df = pd.DataFrame()
    for choice, choice_group in grouped_by_choice:
        if len(choice_group) > 1:
            zIdPhis = zscore(choice_group["IdPhi"])
            choice_group["zIdPhi"] = zIdPhis
            zscored_df = pd.concat([zscored_df, choice_group], ignore_index=True)
        else:
            logging.warning(f"Skipping choice {choice} for {rat} - insufficient samples ({len(choice_group)})")
            continue

    df_path = os.path.join(rat_path, "zIdPhis.csv")
    zscored_df.to_csv(df_path, index=False)
