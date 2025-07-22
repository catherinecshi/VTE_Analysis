"""
Main script for making IdPhi values for test days. Automatically saves within the quantify_VTE function
"""
import os
import pandas as pd
import numpy as np

from config import settings
from config.paths import paths, remote
from preprocessing import data_processing
from feature_extraction import trajectory_analysis
from utilities import logging_utils

# pylint: disable=logging-fstring-interpolation, broad-exception-caught
logger = logging_utils.setup_script_logger()

def create_test_data_structure():
    """
    Create a minimal data structure for inferenceTesting data that works with quantify_VTE
    
    Returns:
        dict: Data structure compatible with quantify_VTE function
    """
    data_structure = {}
    
    # Path to cleaned test coordinates  
    cleaned_dlc_path = paths.cleaned_dlc / "inferenceTesting"
    
    # Iterate through available test coordinate files
    for coord_file in cleaned_dlc_path.glob("*_test_coordinates.csv"):
        # Extract rat name from filename
        rat = coord_file.stem.replace("_test_coordinates", "")
        
        # Skip if not a valid rat directory
        rat_dir = paths.vte_data / rat
        if not rat_dir.exists():
            logger.warning(f"no rat directory found for {rat}")
            continue
        
        settings.update_rat(rat)
        logger.info(f"loading test data for {rat}")
        
        # Load statescript and timestamps from Day1 (since test data is Day1)
        day_folder = rat_dir / "inferenceTesting" / "Day1"
        if not day_folder.exists():
            logger.warning(f"no Day1 folder found for {rat}")
            continue
        
        try:
            # Load statescript
            ss_files = list(day_folder.glob("*stateScriptLog*"))
            if not ss_files:
                # Check what files are actually there
                all_files = list(day_folder.glob("*"))
                print(f"All files in {day_folder}: {[f.name for f in all_files]}")
                # Check for files with similar names
                similar_files = [f for f in all_files if 'state' in f.name.lower()]
                print(f"Files containing 'state': {[f.name for f in similar_files]}")
            ss_log = data_processing.process_statescript_log(ss_files[0])
            
            # Load timestamps
            ts_files = list(day_folder.glob("*videoTimeStamps*"))
            if not ts_files:
                logger.warning(f"no timestamps found for {rat}")
                continue
            
            if ts_files[0].suffix == '.npy':
                timestamps = np.load(ts_files[0])
            else:
                timestamps = data_processing.process_timestamps_data(ts_files[0])
                
            logger.info(f"loaded SS and timestamps for {rat}")
            
        except Exception as e:
            logger.error(f"error loading SS/timestamps for {rat}: {e}")
            continue
        
        # Create temporary coordinate file in expected location for load_specific_files
        rat_dlc_dir = paths.cleaned_dlc / rat
        if not rat_dlc_dir.exists():
            os.makedirs(rat_dlc_dir)
        
        temp_coord_file = rat_dlc_dir / "Day1_coordinates.csv"
        
        # Copy test coordinates to expected location
        try:
            test_coords = pd.read_csv(coord_file)
            test_coords.to_csv(temp_coord_file, index=False)
            logger.info(f"copied test coordinates for {rat}")
        except Exception as e:
            logger.error(f"error copying test coordinates for {rat}: {e}")
            continue
        
        # Create data structure entry
        data_structure[rat] = {
            "Day1": {
                settings.SS: ss_log,
                settings.TIMESTAMPS: timestamps
            }
        }
    
    logger.info(f"created test data structure for {len(data_structure)} rats")
    return data_structure

# Create the test data structure
data_structure = create_test_data_structure()

# Process each rat
for rat in data_structure.keys():
    settings.update_rat(rat)
    logger.info(f"creating IdPhi Values for {rat}")
    #if not "TH510" in rat:
       # continue
    
    # For test data, we only have Day1
    day = "Day1"
    settings.update_day(day)
    logger.info(f"processing {day}")

    try:
        save_path = paths.vte_values / "inferenceTesting" / rat
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        _ = trajectory_analysis.quantify_VTE(data_structure, rat, day, save=save_path)
        logger.info(f"successfully created idphi for {rat} on {day}")
    except KeyError as ke:
        logger.error(f"key error {ke} on {day} for {rat}")
    except Exception as error:
        logger.error(f"error in rat_VTE_over_session - {error} on day {day} for {rat}")
    
    # Clean up temporary coordinate file
    try:
        temp_coord_file = paths.cleaned_dlc / rat / "Day1_coordinates.csv"
        if temp_coord_file.exists():
            temp_coord_file.unlink()
    except Exception as e:
        logger.warning(f"error cleaning up temp file for {rat}: {e}")