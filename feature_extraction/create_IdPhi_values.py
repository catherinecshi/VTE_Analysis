"""
Main script for making IdPhi values. Automatically saves within the quantify_VTE function
"""
import os

from config import settings
from config.paths import paths
from preprocessing import data_processing
from feature_extraction import trajectory_analysis
from utilities import logging_utils

# pylint: disable=logging-fstring-interpolation, broad-exception-caught
logger = logging_utils.setup_logger("create_IdPhi_values")

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
             ("BP11", "Day6"), ("BP11", "Day1"), ("BP11", "Day4"), ("BP08", "Day25")]

base_path = "/Users/catpillow/Documents/VTE_Analysis"
dlc_path = os.path.join(base_path, "processed_data", "cleaned_dlc")
data_structure = data_processing.load_data_structure(paths.vte_data)

for rat_dir in paths.vte_data.iterdir():
    settings.update_rat(rat_dir.name)
    logger.info(f"creating IdPhi Values for {rat_dir.name}")
    
    if not rat_dir.is_dir():
        logger.info(f"skipping {rat_dir.name}")
        continue

    for root, dirs, files in os.walk(rat_dir):
        for file in files:
            parts = file.split("_")
            rat = parts[0]
            day = parts[1]
            
            settings.update_day(day)
            logger.info(day)
            
            rat_day = (rat, day)
            if rat_day in SKIP_DAYS:
                logger.info(f"skipping {rat_day}")
                continue
            
            try:
                save_path = os.path.join(base_path, "processed_data", "VTE_values", rat, day)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                _ = trajectory_analysis.quantify_VTE(data_structure, rat, day, save=save_path)
                logger.info(f"successfully created idphi for {rat} on {day}")
            except Exception as error:
                logger.error(f"error in rat_VTE_over_session - {error} on day {day} for {rat}")