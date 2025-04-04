"""
Main Script to call to get zIdPhi values for data. Requires create_IdPhi_values to have been ran before.
zIdPhi values are created separate of IdPhi values to ensure zscoring across sessions, not within sessions
Procedure:
1. gets the values created from create_IdPhi_values
2. then zscores across sessions for a rat to get zIdPhi values
3. saves individual day files with VTE classification
"""
import os
import logging
import numpy as np
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

for rat in os.listdir(values_path):
    if ".DS" in rat or ".csv" in rat:
        continue
    
    rat_path = os.path.join(values_path, rat)
    before_zscore_df = pd.DataFrame()
    day_dataframes = {}  # Dictionary to store data for each day
    
    # First, collect all data across days for z-scoring
    for day in os.listdir(rat_path):
        day_path = os.path.join(rat_path, day)
        if ".DS" in day:
            continue
        
        day_df = pd.DataFrame()  # Initialize dataframe for this day
        
        for root, _, files in os.walk(day_path):
            for file in files:
                if "trajectories.csv" not in file:
                    continue
                
                file_path = os.path.join(root, file)
                traj_info = pd.read_csv(file_path, header=0)
                
                if all(col in traj_info.columns for col in ["IdPhi", "Choice", "ID", "Trial Type", "Length"]):
                    rows = {"ID": traj_info["ID"], "Day": day, "Choice": traj_info["Choice"], "Correct": traj_info["Correct"],
                            "Trial_Type": traj_info["Trial Type"], "IdPhi": traj_info["IdPhi"], "Length": traj_info["Length"]}
                    temp_df = pd.DataFrame(rows)
                    day_df = pd.concat([day_df, temp_df], ignore_index=True)
                    before_zscore_df = pd.concat([before_zscore_df, temp_df], ignore_index=True)
                else:
                    print(f"missing columns in {rat} on {day}")
        
        # Store the day's data for later use
        if not day_df.empty:
            day_dataframes[day] = day_df
    
    # Z-score across all days grouped by choice
    try:
        grouped_by_choice = before_zscore_df.groupby(by="Choice")
    except Exception as e:
        print(f"error with groupby for {rat} - {e}")
        continue
    
    # Create z-scored dataframe for all days combined
    zscored_df = pd.DataFrame()
    for choice, choice_group in grouped_by_choice:
        if len(choice_group) > 1:
            zIdPhis = zscore(choice_group["IdPhi"])
            choice_group["zIdPhi"] = zIdPhis
            zscored_df = pd.concat([zscored_df, choice_group], ignore_index=True)
        else:
            print(f"Skipping choice {choice} for {rat} - insufficient samples ({len(choice_group)})")
            continue
    
    # Save combined z-scored data for the rat
    df_path = os.path.join(rat_path, "zIdPhis.csv")
    zscored_df.to_csv(df_path, index=False)
    
    # Calculate threshold for VTE detection
    mean_zidphi = np.mean(zscored_df["zIdPhi"])
    std_zidphi = np.std(zscored_df["zIdPhi"])
    threshold = mean_zidphi + 1.5 * std_zidphi
    print(f"Threshold for {rat}: {threshold}")
    
    # Add VTE column to combined dataframe
    zscored_df["VTE"] = zscored_df["zIdPhi"] > threshold
    
    # Now create and save individual day files with VTE column
    for day, day_df in day_dataframes.items():
        # Get just the data for this day from the z-scored dataframe
        day_zscored = zscored_df[zscored_df["Day"] == day].copy()
        
        if not day_zscored.empty:
            # Save day-specific file with z-scores and VTE classification
            day_dir = os.path.join(rat_path, day)
            os.makedirs(day_dir, exist_ok=True)
            day_file_path = os.path.join(day_dir, f"zIdPhi_day_{day}.csv")
            day_zscored.to_csv(day_file_path, index=False)
        else:
            print(f"No valid z-scored data for {rat}, day {day}")
