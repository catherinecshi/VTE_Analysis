import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import zscore

from src import helper
from src import data_processing
from src import trajectory_analysis

### LOGGING
logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# makes a new log everytime the code runs by checking the time
log_file = datetime.now().strftime("/Users/catpillow/Documents/VTE_Analysis/doc/VTE_processing_log_%Y%m%d_%H%M%S.txt")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

# days that were kinda messed up in terms of dlc
SKIP_DAYS = [("BP06", "Day10"), ("BP10", "Day43"), ("BP10", "Day21"), ("BP10", "Day27"),
             ("BP10", "Day20"), ("BP10", "Day39"), ("BP10", "Day49"), ("BP10", "Day18"),
             ("BP10", "Day25"), ("BP10", "Day23"), ("BP10", "Day19"), ("BP09", "Day21"),
             ("BP09", "Day13"), ("BP09", "Day29"), ("BP09", "Day8"), ("BP09", "Day7"),
             ("BP09", "Day6"), ("BP09", "Day15"), ("BP09", "Day31"), ("BP09", "Day9"),
             ("BP09", "Day20"), ("BP09", "Day1"), ("BP09", "Day3"), ("BP09", "Day2"),
             ("BP09", "Day11"), ("BP09", "Day16"), ("BP09", "Day5"), ("BP07", "Day12"),
             ("BP07", "Day7"), ("BP22", "Day31"), ("BP22", "Day47"), ("BP15", "Day15"),
             ("BP15", "Day10"), ("TH405", "Day3"), ("BP21", "Day21"), ("BP21", "Day8"),
             ("BP19", "Day28"), ("BP19", "Day17"), ("BP10", "Day37"), ("BP10", "Day26"),
             ("BP10", "Day6"), ("BP10", "Day18"), ("BP10", "Day46"), ("BP10", "Day23"),
             ("BP11", "Day6"), ("BP11", "Day1"), ("BP11", "Day4")]

base_path = "/Users/catpillow/Documents/VTE_Analysis"
dlc_path = os.path.join(base_path, "processed_data", "cleaned_dlc")
data_path = os.path.join(base_path, "data", "VTE_Data")
data_structure = data_processing.load_data_structure(data_path)

vte_path = os.path.join(base_path, "processed_data", "VTE_data")

for rat in os.listdir(vte_path):
    rat_path = os.path.join(vte_path, rat)
    if not os.path.isdir(rat_path):
        continue # skip files

    for root, dirs, files in os.walk(rat_path):
        for file in files:
            parts = file.split("_")
            rat = parts[0]
            day = parts[1]
            
            rat_day = (rat, day)
            if rat_day in SKIP_DAYS:
                continue
            
            try:
                save_path = os.path.join(base_path, "processed_data", "VTE_values", rat, day)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                _, _ = trajectory_analysis.quantify_VTE(data_structure, rat, day, save=save_path)
            except Exception as error:
                logging.error(f"error in rat_VTE_over_session - {error} on day {day} for {rat}")


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
