"""
used to retrieve data from a remote path, usually citadel, and saved locally
modify 'if __name__...' for choosing whether to retrieve dlc or ss/timestamps data
modify MODULE for getting data from different regimes
BP07's training data is in initialTraining so i had to make a separate function for her

currently excluding most of sleep stuff
"""

import os
import re
import logging
import shutil
import subprocess

from config import settings
from config.paths import remote, paths
from utilities import logging_utils

# pylint: disable=logging-fstring-interpolation
logger = logging_utils.setup_script_logger()

REMOTE_NAME = remote.remote_name
REMOTE_PATH = remote.remote_path
LOCAL_PATH = paths.vte_data

EXCLUDED_FOLDERS = remote.excluded_folders
MODULE = remote.module
IMPLANTED_RATS = settings.IMPLANTED_RATS

def sync_files(remote_folder, local_folder, include_patterns, exclude_patterns):
    """
    copies desired files into local path from folders indicated
    only copies files approved by included_patterns

    Args:
        remote_folder (str arr): folders that are approved to copy files from
        local_folder (str): folders that will be made in the new location
        include_patterns (str arr): patterns that are desired for copying
        exclude_patterns (str arr): patterns that are desired to be ignored
    """
    try:
        for pattern in include_patterns:
            command = ["rclone", "copy", f"{REMOTE_NAME}:{REMOTE_PATH}/{remote_folder}",
                        f"{local_folder}", "--include", pattern]
            
            for exclude_pattern in exclude_patterns:
                command.extend(["--exclude", exclude_pattern])
            
            subprocess.run(
                command,
                timeout=30, # time out in seconds so it doesn't wait for forever
                check=True, # raise error if command fails
                capture_output=True, # capture output
                text=True # decode bytes to str
            )
    except subprocess.TimeoutExpired:
        logging.error("The command took too long - terminated.")
        logging.error(f"Local folder is - {local_folder}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}")
        logging.error(f"Output: {e.output}")
        logging.error(f"remote folder - {remote_folder}, local folder - {local_folder}")

def get_day_folders():
    """
    retrieves the folders within which I want to copy files with INCLUDE_PATTERN patterns
    excludes:
        - postSleep/preSleep
        - folders that don't have 'Day'

    Returns:
        (set): array of directories (of {animal_ID}/{MODULE}/{day}) that i want to copy
    """
    
    try:
        result = subprocess.run(["rclone", "lsl", f"{REMOTE_NAME}:{REMOTE_PATH}"],
                            stdout=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Called process error for {REMOTE_NAME}:{REMOTE_PATH} - {e}")
    
    day_folders = set()
    
    for line in result.stdout.splitlines():
        path = line.split()[-1]
        parts = path.split("/")
        
        if any(pattern in part for pattern in EXCLUDED_FOLDERS for part in parts):
            continue # skip if any part of path matches excluded_folders

        if any(rat in parts[0] for rat in IMPLANTED_RATS):
            implanted = True
        else:
            implanted = False
        
        if len(parts) > 3 and parts[1] == MODULE and ("Day" in parts[-1]) and not implanted:
            day_folder = "/".join(parts[:3])  # Extract {animal_ID}/{MODULE}/{day}/
            day_folders.add(day_folder)
        elif len(parts) > 3 and parts[1] == MODULE and ("Day" in parts[-1]): # implanted rats
            day_folder = "/".join(parts[:4]) # Extract {animal_ID}/{MODULE}/{day}/{track_folder}
            day_folders.add(day_folder)
    
    return day_folders

def check_for_aberrent_folders(excluded_patterns):
    for rat_folder in os.listdir(LOCAL_PATH):
        rat_path = os.path.join(LOCAL_PATH, rat_folder, MODULE)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            logging.info(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        day_numbers = []
        for day_folder in os.listdir(rat_path):
            # for checking if any days are missing
            match = re.match(r'Day(\d+)', day_folder)
            if match:
                day_numbers.append(int(match.group(1)))
            
            day_path = os.path.join(rat_path, day_folder)
            # skip over .DS_Store
            if not os.path.isdir(day_path):
                logging.info(f"Skipping over non-directory folder: {day_path}")
                continue
            
            # remove dirs and files with excluded patterns
            for root, dirs, files in os.walk(day_path):
                for directory in dirs:
                    current_dir_path = os.path.join(root, directory)
                    if any(excluded_pattern in directory for excluded_pattern in excluded_patterns):
                        logging.info(f'deleting {current_dir_path}')
                        shutil.rmtree(current_dir_path)
                    
                    # also had a problem where folders for things like ss logs or video time stamps 
                    # were being made, so fixing that here
                    if "stateScriptLog" in directory or ("videoTimeStamps" in directory and "HWSync" not in directory):
                        for file in os.listdir(current_dir_path):
                            file_path = os.path.join(root, directory, file)
                            if os.path.isfile(file_path):
                                possible_folder_path = os.path.join(root, file)
                                if os.path.exists(possible_folder_path):
                                    new_path = os.path.join(root, "folder")
                                    os.rename(possible_folder_path, new_path)
                                    shutil.move(os.path.join(new_path, file), root)
                                    current_dir_path = new_path
                                    os.rmdir(current_dir_path)
                                    break
                                else:
                                    shutil.move(file_path, root)
                        
                        # delete directory after moving file
                        if os.path.exists(current_dir_path):
                            os.rmdir(current_dir_path)
                
                for file in files:
                    if any(excluded_pattern in file for excluded_pattern in excluded_patterns):
                        current_file_path = os.path.join(root, file)
                        logging.info(f"deleting {current_file_path}")
                        os.remove(current_file_path)
        
        last_day = max(day_numbers)
        
        # check for missing days
        missing_days = []
        for day in range(1, last_day + 1):
            if day not in day_numbers:
                missing_days.append(day)
        
        if len(missing_days) > 0:
            logging.warning(f"missing days for {rat_folder} for days {missing_days}")
                        
def dlc_files_in_folder():
    for rat_folder in os.listdir(LOCAL_PATH):
        if not any(rat in rat_folder for rat in IMPLANTED_RATS):
            continue # this is only for implanted rats bc track folders
        
        # skip over .DS_Store
        rat_path = os.path.join(LOCAL_PATH, rat_folder, MODULE)
        if not os.path.isdir(rat_path):
            logging.info(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        for day_folder in os.listdir(rat_path):
            day_path = os.path.join(rat_path, day_folder)
            # skip over .DS_Store
            if not os.path.isdir(day_path):
                logging.info(f"Skipping over non-directory folder: {day_path}")
                continue
            
            dlc_path = None
            dir_path = None
            for item in os.listdir(day_path):
                item_path = os.path.join(day_path, item)
                if os.path.isfile(item_path) and '.csv' in item:
                    dlc_path = item_path
                elif os.path.isdir(item_path) and 'track' in item:
                    dir_path = item_path
                else:
                    logging.warning(f"weird non-dlc & track folder found - {item_path}")
            
            if dlc_path is not None and dir_path is not None:
                try:
                    shutil.move(dlc_path, dir_path)
                except shutil.Error as se:
                    logging.debug(f"{se} for {dlc_path}")
            else:
                logging.debug(f"path doesn\'t exist. dlc - {dlc_path}. dir - {dir_path}")

def remove_excluded_patterns(excluded_patterns):
    for root, _, files in os.walk(LOCAL_PATH):
        for f in files:
            file_path = os.path.join(root, f)
            if any(excluded_pattern in f for excluded_pattern in excluded_patterns):
                os.remove(file_path)

def dlc():
    """
    copies the dlc files from remote to local document
    """
    
    try:
        result = subprocess.run(["rclone", "lsl", f"{REMOTE_NAME}:{REMOTE_PATH}/TI_DLC_tracked_all"],
                            stdout=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Called process error for {REMOTE_NAME}:{REMOTE_PATH}: {e}")
    
    for line in result.stdout.splitlines():
        path = line.split()[-1]
        parts = path.split("/")
        
        if len(parts) < 2:
            continue # usually just the .DS_Store
        if "sleep" in parts[-2] or not ".csv" in path: # dont need sleep dlc or any non csv files rn
            continue

        # extract rat ID & day #
        ratID = parts[-1].split("_")[0]
        day_match = re.search(r'Day\d+', parts[-1])
        if not day_match:
            continue
        day = day_match.group()
        
        # make folders if it doesn't already exist
        local_folder_path = os.path.join(LOCAL_PATH, ratID, MODULE, day)
        os.makedirs(local_folder_path, exist_ok=True)
        
        # copy files
        path = os.path.join("/TI_DLC_tracked_all", path)
        include_patterns = ["*.csv"]
        excluded_patterns = ["*lickTraining*", "*inferenceTesting*", "*reverse*", "*moveHome*", "*freeForage*"]
        sync_files(path, local_folder_path, include_patterns=include_patterns, exclude_patterns=excluded_patterns)
    
    # make sure everything is fine
    excluded_patterns = ["lickTraining", "inferenceTesting", "inferenceTest", "reverse", "moveHome", "freeForage"]
    remove_excluded_patterns(excluded_patterns)
    dlc_files_in_folder()

def main(include_patterns):
    """
    1. gets desired folders from get_day_folders
    2. get animal ID & day folders
    3. make directory for /animal ID/inferenceTraining/day
    4. copies desired files from folder to directory
    """
    
    day_folders = get_day_folders()
    
    for folder in day_folders:
        # extract rat ID & skip implanted rats
        animal_id = folder.split("/")[0]
        
        # skip BP07
        if "BP07" in animal_id:
            continue
        
        if any(rat in animal_id for rat in IMPLANTED_RATS):
            day_folder_path = "/".join(folder.split("/")[2:4])
            
            # save & copy
            local_folder_path = os.path.join(LOCAL_PATH, animal_id, MODULE, day_folder_path)
            os.makedirs(local_folder_path, exist_ok=True)
            
            excluded_patterns = ["*.dat", "*.mda", "*mountain*", "*.json", "*.mat", "*.txt", "*postSleep*", "*preSleep*", "*.h264", "*geometry*", "*HWSync*"]
            sync_files(folder, local_folder_path, include_patterns=include_patterns, exclude_patterns=excluded_patterns)
        else: # for unimplanted rats - don't have to worry about track folders
            day_folder_path = "/".join(folder.split("/")[2:3])
            
            # save & copy
            local_folder_path = os.path.join(LOCAL_PATH, animal_id, MODULE, day_folder_path)
            os.makedirs(local_folder_path, exist_ok=True)
            
            excluded_patterns = ["*.h264", "*geometry*", "*HWSync*"]
            sync_files(folder, local_folder_path, include_patterns=include_patterns, exclude_patterns=excluded_patterns)
        
    # check and make sure everything is standardised and normal
    excluded_patterns = [".dat", ".mda", "mountain", ".json", ".mat", ".txt", "postSleep", "preSleep", "Logs", ".timestampoffset",
                         ".DIO", "log", "msort", "LFP", "spikes", "reTrain", ".h264", "geometry", "HWSync", "midSleep"]
    check_for_aberrent_folders(excluded_patterns)
    
def BP07(include_patterns):
    try:
        result = subprocess.run(["rclone", "lsl", f"{REMOTE_NAME}:{REMOTE_PATH}/BP07/initialTraining"],
                            stdout=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Called process error on {REMOTE_NAME}:{REMOTE_PATH}: {e}")
    
    day_folders = set()
    for line in result.stdout.splitlines():
        path = line.split()[-1]
        parts = path.split("/")
        
        if "Day" not in path:
            continue
        
        match = re.search(r"Day(\d+)(?:_)?", parts[-2], re.IGNORECASE)
        if match:
            day_number = match.group(1)
            day = "Day" + day_number
            day_folders.add(day)
            logging.info(f"BP07 day folders added {day}")

    for day in day_folders:
        # save & copy
        local_folder_path = os.path.join(LOCAL_PATH, "BP07", MODULE, day)
        os.makedirs(local_folder_path, exist_ok=True)
        
        excluded_patterns = ["*.h264", "*geometry*", "*HWSync*"]
        remote_folder_path = os.path.join("BP07", "initialTraining", day)
        sync_files(remote_folder_path, local_folder_path, include_patterns=include_patterns, exclude_patterns=excluded_patterns)
        
    # check and make sure everything is standardised and normal
    excluded_patterns = [".dat", ".mda", "mountain", ".json", ".mat", ".txt", "postSleep", "preSleep", ".h264", "geometry", "HWSync"]
    check_for_aberrent_folders(excluded_patterns)
    
def copy_timestamps():
    timestamps_path = paths.timestamps
    if not os.path.exists(timestamps_path):
        os.makedirs(timestamps_path)
    
    rclone_command = [
        "rclone", "copy",
        f"{REMOTE_NAME}:{REMOTE_PATH}",
        f"{timestamps_path}",
        "--include", "*/initialTraining*/**/*.videoTimeStamps",
        "--exclude", "*preSleep*",
        "--exclude", "*postSleep*",
        "--exclude", "*midSleep*"
    ]
    
    result = subprocess.run(rclone_command, capture_output=True, text=True, check=True)
    
    if result.returncode != 0:
        logger.error(f"Error: result return code is not 0 but instead {result.stderr}")
    
    for root, dirs, _ in os.walk(timestamps_path):
        for directory in dirs:
            if "postSleep" in directory or "preSleep" in directory or "midSleep" in directory:
                dir_path = os.path.join(root, directory)
                shutil.rmtree(dir_path)

def copy_statescripts():
    statescript_path = paths.statescripts
    if not os.path.exists(statescript_path):
        os.makedirs(statescript_path)
    
    rclone_command = [
        "rclone", "copy",
        f"{REMOTE_NAME}:{REMOTE_PATH}",
        f"{statescript_path}",
        "--include", "*/inferenceTraining*/**/*.stateScriptLog",
        "--exclude", "*preSleep*",
        "--exclude", "*postSleep*",
        "--exclude", "*midSleep*"
    ]

    result = subprocess.run(rclone_command, capture_output=True, text=True, check=True)
    
    if result.returncode != 0:
        logger.error(f"Error: result return code not 0 but instead {result.stderr}")
    
    for root, dirs, _ in os.walk(statescript_path):
        for directory in dirs:
            if "postSleep" in directory or "preSleep" in directory or "midSleep" in directory:
                dir_path = os.path.join(root, directory)
                shutil.rmtree(dir_path)

if __name__ == "__main__":
    # if getting stuff from the main folders
    included_patterns = remote.included_patterns
    main(include_patterns=included_patterns) # for retrieving data where there is one in each day folder
    #BP07(included_patterns)
    
    # if getting dlc stuff from one folder
    #dlc()
    #copy_timestamps()
    #copy_statescripts()