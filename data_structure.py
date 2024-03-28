import os
import fnmatch
import pandas as pd
import numpy as np

import readCameraModuleTimeStamps

def process_dlc_data(file_path):
    df = pd.read_csv(file_path, header=[1, 2])
    df.set_index(('bodyparts', 'coords'), inplace=True)
    return df

def process_loaded_dlc_data(file_path):
    df = pd.read_csv(file_path, header=[0, 1])
    return df

def process_timestamps_data(file_path):
    timestamps = readCameraModuleTimeStamps.read_timestamps(file_path)
    return timestamps

def process_statescript_log(file_path):
    with open(file_path) as file:
        content = file.read()
    
    return content

def create_main_data_structure(base_path): # creates a nested dictionary
    data_structure = {}

    for rat_folder in os.listdir(base_path): # loop for each rat
        rat_path = os.path.join(base_path, rat_folder, 'inferenceTraining')
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            print(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        # skip over empty folders
        day_folders = os.listdir(rat_path)
        if not day_folders: # if folder is empty
            print(f"{rat_path} is empty")
            continue
            
        data_structure[rat_folder] = {} # first nest in dictionary
        
        for day_folder in os.listdir(rat_path): # loop for each day (in each rat folder)
            day_path = os.path.join(rat_path, day_folder)
            dlc_data = None
            ss_data = None
            timestamps_data = None
            track_folder_found = False # check if there are any track folders in this day folder
            
            # for checking if a videoframerate, ss log & dlc csv file has been found for each day for each rat
            ss = False
            dlc = False
            timestamps = False
        
            for root, dirs, files in os.walk(day_path): # look at all the files in the day folder
                # check if there are any track folders bc only implanted rats have it, which changes the naming conventions for folders
                for dir_name in dirs:
                    if fnmatch.fnmatch(dir_name.lower(), '*track*'):
                        track_folder_found = True
                
                for f in files:
                    f_actual = f
                    f = f.lower() # there were some problems with cases
                    
                    # storing the DLC csv
                    if fnmatch.fnmatch(f, '*dlc*.csv'): 
                        # checks if there is a duplication
                        if dlc == True:
                            print("More than one DLC file found")
                        else:
                            dlc = True
                        
                        file_path = os.path.join(root, f)
                        dlc_data = process_dlc_data(file_path)
                        
                        print(file_path)
                    
                    # handle fnmatch differently depending on whether track folder was found
                    if track_folder_found:
                        # storing the statescript log
                        if fnmatch.fnmatch(f, '*track*.statescriptlog'):
                            # checks if there is a duplication
                            if ss == True:
                                print("More than one SS Log found")
                            else:
                                ss = True
                            
                            file_path = os.path.join(root, f)
                            ss_data = process_statescript_log(file_path)
                            
                            print(file_path)
                        
                        if fnmatch.fnmatch(f, '*track*.videotimestamps'):
                            if timestamps == True:
                                print("More than one .videotimestamps found")
                            else:
                                timestamps = True

                            file_path = os.path.join(root, f_actual)
                            timestamps_data = process_timestamps_data(file_path)
                    else:
                        # storing the statescript log
                        if fnmatch.fnmatch(f, '*.statescriptlog'):
                            # checks if there is a duplication
                            if ss == True:
                                print("More than one SS Log found")
                            else:
                                ss = True
                            
                            file_path = os.path.join(root, f)
                            ss_data = process_statescript_log(file_path)
                            
                            print(file_path)
                        
                        if fnmatch.fnmatch(f, '*.videotimestamps'):
                            if timestamps == True:
                                print("More than one .videotimestamps found")
                            else:
                                timestamps = True

                            file_path = os.path.join(root, f_actual)
                            timestamps_data = process_timestamps_data(file_path)
            
            # add to dictionary
            if ss and dlc and timestamps:
                data_structure[rat_folder][day_folder] = {
                    "DLC_tracking": dlc_data,
                    "stateScriptLog": ss_data,
                    "videoTimeStamps": timestamps_data
                }
            elif ss and (not dlc) and (not timestamps):
                data_structure[rat_folder][day_folder] = {
                    "stateScriptLog": ss_data
                }
                print(f"only ss found for rat {rat_folder} for {day_folder}")
            elif (not ss) and (not dlc) and (not timestamps):
                print(f"No timestamps, stateScriptLog or DLC file found for rat {rat_folder} for {day_folder}")
            elif (not ss) or (not dlc) or (not timestamps):
                print(f"File missing for rat {rat_folder} for {day_folder}")

    return data_structure

def save_data_structure(data_structure, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for rat, days in data_structure.items():
        rat_path = os.path.join(save_path, rat)
        if not os.path.exists(rat_path):
            os.makedirs(rat_path)
        
        
        for day, data in days.items():
            day_path = os.path.join(rat_path, day)
            if not os.path.exists(day_path):
                os.makedirs(day_path)

            if "DLC_tracking" in data:
                dlc_path = os.path.join(day_path, f"{day}_DLC_tracking.csv")
                data["DLC_tracking"].to_csv(dlc_path, header = True, index = False)
            
            if "stateScriptLog" in data:
                ss_path = os.path.join(day_path, f"{day}_stateScriptLog.txt")
                with open(ss_path, 'w') as file:
                    file.write(data["stateScriptLog"])
            
            if "videoTimeStamps" in data:
                timestamps_path = os.path.join(day_path, f"{day}_videoTimeStamps.npy")
                np.save(timestamps_path, data["videoTimeStamps"])

def load_data_structure(save_path): # this function assumes no errors bc they would've been caught before saving
    data_structure = {}

    for rat_folder in os.listdir(save_path): # loop for each rat
        rat_path = os.path.join(save_path, rat_folder)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            print(f"Skipping over non-directory folder: {rat_path}")
            continue
            
        data_structure[rat_folder] = {} # first nest in dictionary
        
        for day_folder in os.listdir(rat_path): # loop for each day (in each rat folder)
            day_path = os.path.join(rat_path, day_folder)
            dlc_data = None
            ss_data = None
            timestamps_data = None
        
            for root, dirs, files in os.walk(day_path): # look at all the files in the day folder
                for f in files:
                    f = f.lower()
                    
                    # storing the DLC csv
                    if fnmatch.fnmatch(f, '*dlc*.csv'): 
                        file_path = os.path.join(root, f)
                        dlc_data = process_loaded_dlc_data(file_path)
                    
                    # storing the statescript log
                    if fnmatch.fnmatch(f, '*statescriptlog*'):
                        file_path = os.path.join(root, f)
                        ss_data = process_statescript_log(file_path)
                    
                    if fnmatch.fnmatch(f, "*videotimestamps*"):
                        file_path = os.path.join(root, f)
                        timestamps_data = np.load(file_path)
            
            # add to dictionary
            data_structure[rat_folder][day_folder] = {
                "DLC_tracking": dlc_data,
                "stateScriptLog": ss_data,
                "videoTimeStamps": timestamps_data
            }

    return data_structure
