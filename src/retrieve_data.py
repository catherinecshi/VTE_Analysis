"""
used to retrieve data from a remote path, usually citadel, and saved locally
modify 'if __name__...' for choosing whether to retrieve dlc or ss/timestamps data
modify MODULE for getting data from different regimes

currently excluding most of sleep stuff
"""

import os
import re
import logging
import subprocess

REMOTE_NAME = "VTE"
REMOTE_PATH = "data/Projects/bp_inference/Transitive"
LOCAL_PATH = "/Users/catpillow/Documents/VTE Analysis/VTE_Data"

EXCLUDED_FOLDERS = ["preSleep", "postSleep"]
MODULE = "inferenceTraining"

# pylint
# pylint: disable=logging-fstring-interpolation

def sync_files(remote_folder, local_folder, include_patterns):
    """
    copies desired files into local path from folders indicated
    only copies files approved by INCLUDE_PATTERNS

    Args:
        remote_folder (str arr): folders that are approved to copy files from
        local_folder (str): folders that will be made in the new location
    """
    
    try:
        for pattern in include_patterns:
            subprocess.run(
                ["rclone", "copy", f"{REMOTE_NAME}:{REMOTE_PATH}/{remote_folder}",
                f"{local_folder}", "--include", pattern],
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
        print(f'Error: {e}')
    
    day_folders = set()
    
    for line in result.stdout.splitlines():
        path = line.split()[-1]
        parts = path.split('/')
        
        if len(parts) > 3 and parts[1] == MODULE and ('Day' in parts[-1]):
            if any(pattern in part for part in parts for pattern in EXCLUDED_FOLDERS): # skip excluded folders
                continue
            elif 'track' in parts[3] and 'geometry' not in path and 'rec' not in path:
                day_folder = '/'.join(parts[:4]) # Extract {animal_ID}/{MODULE}/{day}/{track folder name}
            else:
                day_folder = '/'.join(parts[:3])  # Extract {animal_ID}/{MODULE}/{day}/
            
            day_folders.add(day_folder)
    
    return day_folders

def dlc():
    """
    copies the dlc files from remote to local document
    """
    
    try:
        result = subprocess.run(["rclone", "lsl", f"{REMOTE_NAME}:{REMOTE_PATH}/TI_DLC_tracked_all"],
                            stdout=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')
    
    for line in result.stdout.splitlines():
        path = line.split()[-1]
        parts = path.split('/')
        
        if len(parts) < 2:
            continue # usually just the .DS_Store
        if 'sleep' in parts[-2] or not '.csv' in path: # dont need sleep dlc or any non csv files rn
            continue

        # extract rat ID & day #
        ratID = parts[-1].split('_')[0]
        day = re.search(r'Day\d+', parts[-1]).group()
        
        # make folders if it doesn't already exist
        local_folder_path = os.path.join(LOCAL_PATH, ratID, MODULE, day)
        os.makedirs(local_folder_path, exist_ok=True)
        
        # copy files
        path = os.path.join("/TI_DLC_tracked_all", path)
        include_patterns = ["*.csv"]
        sync_files(path, local_folder_path, include_patterns=include_patterns)

def main(include_patterns):
    """
    1. gets desired folders from get_day_folders
    2. get animal ID & day folders
    3. make directory for /animal ID/inferenceTraining/day
    4. copies desired files from folder to directory
    """
    
    day_folders = get_day_folders()
    
    for folder in day_folders:
        # extract rat ID & day #
        animal_id = folder.split('/')[0]
        day_folder_path = '/'.join(folder.split('/')[2:3]) # exclude track folder name when creating local folders
        
        # save & copy
        local_folder_path = os.path.join(LOCAL_PATH, animal_id, MODULE, day_folder_path)
        os.makedirs(local_folder_path, exist_ok=True)
        
        sync_files(folder, local_folder_path, include_patterns=include_patterns)

if __name__ == "__main__":
    # if getting stuff from the main folders
    #include_patterns = ["*.stateScriptLog", "*.videoTimeStamps"]
    #main(include_patterns=include_patterns) # for retrieving data where there is one in each day folder
    
    # if getting dlc stuff from one folder
    dlc()
    
