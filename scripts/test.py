import os
import numpy as np
import pandas as pd

from src import helper
from src import plotting
from src import data_processing
from src import creating_zones
from src import performance_analysis


base_path = "/Users/catpillow/Documents/VTE_Analysis"
data_path = os.path.join(base_path, "data", "VTE_Data")
data_structure = data_processing.load_data_structure(data_path)
dlc_path = os.path.join(base_path, "processed_data", "dlc_data")
vte_path = os.path.join(base_path, "processed_data", "VTE_data")
values_path = os.path.join(base_path, "processed_data", "VTE_values")

for rat in os.listdir(vte_path):
    rat_path = os.path.join(vte_path, rat)
    if not os.path.isdir(rat_path):
        continue # skip files
    
    if not "TH510" in rat:
        continue
    
    for root, dirs, files in os.walk(rat_path):
        for f in files:
            parts = f.split("_")
            rat = parts[0]
            day = parts[1]
            count = 0
            
            try:
                helper.update_rat(rat)
                helper.update_day(day)
                DLC_df, SS_log, timestamps, trial_starts = helper.initial_processing(data_structure, rat, day)
                x = DLC_df["x"]
                y = DLC_df["y"]
                
                # check if timestamps is ascending bc annoying trodes code & files
                if not np.all(timestamps[:-1] <= timestamps[1:]):
                    raise helper.CorruptionError(rat, day, timestamps)
                
                # define zones
                centre_hull = creating_zones.get_centre_hull(DLC_df)

                _, _, performance = performance_analysis.get_session_performance(SS_log) # a list of whether the trials resulted in a correct or incorrect choice
                same_len = helper.check_equal_length(performance, list(trial_starts.keys())) # check if there are the same number of trials for perf and trial_starts

                
                for i, (trial_start, trial_type) in enumerate(trial_starts.items()): # where trial type is a string of a number corresponding to trial type
                    # cut out the trajectory for each trial
                    trajectory_x, trajectory_y = helper.get_trajectory(DLC_df, trial_start, timestamps, centre_hull)
                    plotting.plot_trajectory_animation(x, y, trajectory_x, trajectory_y, title=f"{rat} on {day} - {count}")
                    count += 1
            except Exception as error:
                print(f"error in rat_VTE_over_session - {error} on day {day} for {rat}")
            