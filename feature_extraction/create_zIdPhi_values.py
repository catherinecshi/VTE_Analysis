"""
Main Script to call to get zIdPhi values for data. Requires create_IdPhi_values to have been ran before.
zIdPhi values are created separate of IdPhi values to ensure zscoring across sessions, not within sessions
Procedure:
1. gets the values created from create_IdPhi_values
2. then zscores across sessions for a rat to get zIdPhi values
3. saves individual day files with VTE classification
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import zscore

from config.paths import paths
from preprocessing import data_processing
from utilities import logging_utils

### LOGGING
logger = logging_utils.setup_script_logger()

base_path = "/Users/catpillow/Documents/VTE_Analysis"
dlc_path = paths.cleaned_dlc
data_path = paths.vte_data
data_structure = data_processing.load_data_structure(data_path)
vte_path = paths.vertice_data

for rat_dir in paths.vte_values.iterdir():
    rat = rat_dir.name
    if ".DS" in rat or ".csv" in rat:
        continue
    
    rat_path = paths.vte_values / rat
    before_zscore_df = pd.DataFrame()
    day_dataframes = {}  # Dictionary to store data for each day
    
    # First, collect all data across days for z-scoring
    for day_dir in rat_path.iterdir():
        day = day_dir.name
        day_path = rat_path / day
        if ".DS" in day:
            continue
        
        day_df = pd.DataFrame()  # Initialize dataframe for this day
        
        for root, _, files in os.walk(day_path):
            for file in files:
                if not file.endswith(f"{rat}_{day}_trajectories.csv"):
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
                    logger.error(f"missing columns in {rat} on {day}")
        
        # Store the day's data for later use
        if not day_df.empty:
            day_dataframes[day] = day_df
    
    # exclude trajectories that are too long
    try:
        valid_length_df = before_zscore_df[before_zscore_df["Length"] <= 4].copy()
        excluded_length_df = before_zscore_df[before_zscore_df["Length"] > 4].copy()
    except KeyError:
        print(rat)
        print("key error")
        continue
    
    # Z-score across all days grouped by choice
    try:
        grouped_by_choice = valid_length_df.groupby(by="Choice")
    except Exception as e:
        logger.error(f"error with groupby for {rat} - {e}")
        continue
    
    # Z-score within each choice group and assign to a new column
    valid_length_df_copy = valid_length_df.copy()
    valid_length_df_copy['zIdPhi'] = valid_length_df_copy.groupby('Choice')['IdPhi'].transform(
        lambda x: zscore(x) if len(x) > 1 else np.nan
    )

    # Remove rows where z-scoring couldn't be performed (groups with <= 1 sample)
    #zscored_valid_df = valid_length_df_copy.dropna(subset=['zIdPhi'])
    
    # for trajectories that are too long, assign a 0 zidphi
    if not excluded_length_df.empty:
        excluded_length_df["zIdPhi"] = 0
    zscored_df = pd.concat([valid_length_df_copy, excluded_length_df], ignore_index=True)
    
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
