import os
import re
import fnmatch
import subprocess

REMOTE_NAME = "VTE"
REMOTE_PATH = "data/Projects/bp_inference/Transitive"
LOCAL_PATH = "/Users/catpillow/Documents/VTE Analysis/VTE_Data"

EXCLUDED_FOLDERS = ["*preSleep*", "*postSleep*"]
MODULE = 'inferenceTraining'

def sync_files(remote_folder, local_folder, include_patterns):
    """
    copies desired files into local path from folders indicated
    only copies files approved by INCLUDE_PATTERNS

    Args:
        remote_folder (str arr): folders that are approved 
        local_folder (str): should be the same as remote_folders, but for making instead
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
        print("The command took too long - terminated.")
        print(f"Local folder is - {local_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Output: {e.output}")

def get_day_folders():
    """
    retrieves the folders within which I want to copy files with INCLUDE_PATTERN patterns
    excludes:
        - postSleep/preSleep
        - anything that is not inferenceTraining
        - folders that don't have 'Day'

    Returns:
        (str array): array of directories (of {animal_ID}/inferenceTraining/{day}) that i want to copy
    
    Notes:
        - currently will have duplicates of day_folders, but hopefully only one made at the end of the day
    """
    
    result = subprocess.run(["rclone", "lsl", f"{REMOTE_NAME}:{REMOTE_PATH}"],
                            stdout=subprocess.PIPE, text=True)
    
    day_folders = set()
    for line in result.stdout.splitlines():
        path = line.split()[-1]
        parts = path.split('/')
        
        if len(parts) > 3 and parts[1] == MODULE and ('Day' in parts[-1]):
            if any(fnmatch.fnmatch(parts[2], pattern) or fnmatch.fnmatch(parts[3], pattern) for pattern in EXCLUDED_FOLDERS): # skip excluded folders
                continue
            
            day_folder = '/'.join(parts[:3])  # Extract {animal_ID}/inference_Training/{day}/
            day_folders.add(day_folder)
    
    return day_folders

def dlc():
    """
    copies the dlc files from remote to local document
    """
    
    result = subprocess.run(["rclone", "lsl", f"{REMOTE_NAME}:{REMOTE_PATH}/TI_DLC_tracked_all"],
                            stdout=subprocess.PIPE, text=True)
    
    for line in result.stdout.splitlines():
        path = line.split()[-1]
        parts = path.split('/')
        
        if 'sleep' in parts[-2] or not '.csv' in parts[-1]: # dont need sleep dlc or any non csv files
            continue

        # extract rat ID & day #
        ratID = parts[-1].split('_')[0]
        day = re.search(r'Day\d+', parts[-1]).group()
        
        # save & copy
        local_folder_path = os.path.join(LOCAL_PATH, ratID, MODULE, day)
        os.makedirs(local_folder_path, exist_ok=True)
        
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
        day_folder_path = '/'.join(folder.split('/')[2:])
        
        # save & copy
        local_folder_path = os.path.join(LOCAL_PATH, animal_id, MODULE, day_folder_path)
        os.makedirs(local_folder_path, exist_ok=True)
        
        sync_files(folder, local_folder_path, include_patterns=include_patterns)

if __name__ == "__main__":
    # if getting stuff from the main folders
    include_patterns = ["*.stateScriptLog", "*.videoTimeStamps"]
    main(include_patterns=include_patterns) # for retrieving data where there is one in each day folder
    
    # if getting dlc stuff from one folder
    dlc()
    
