"""
Main script for making IdPhi values. Automatically saves within the quantify_VTE function
"""

import os
import logging
from datetime import datetime

from src import data_processing
from src import trajectory_analysis

### LOGGING
logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# makes a new log everytime the code runs by checking the time
log_file = datetime.now().strftime("/Users/catpillow/Documents/VTE_Analysis/doc/create_IdPhi_values_log_%Y%m%d_%H%M%S.txt")
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
    if not os.path.isdir(rat_path) or not "BP06" in rat:
        continue # skip files

    for root, dirs, files in os.walk(rat_path):
        for file in files:
            parts = file.split("_")
            rat = parts[0]
            day = parts[1]
            
            if not "8" in day:
                continue
            
            rat_day = (rat, day)
            if rat_day in SKIP_DAYS:
                continue
            
            try:
                save_path = os.path.join(base_path, "processed_data", "VTE_values", rat, day)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                _, _ = trajectory_analysis.quantify_VTE(data_structure, rat, day, save=save_path)
            except Exception as error:
                print(f"error in rat_VTE_over_session - {error} on day {day} for {rat}")