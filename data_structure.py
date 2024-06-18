import os
import re
import fnmatch
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

import readCameraModuleTimeStamps

def process_dlc_data(file_path):
    """
    USE WHEN COMPLETELY UNPROCESSED, like when main df hasn't been created yet
    reads csv files where first and second row and headers and should be ignored in the data processing
    index of df is the column corresponding to bodyparts & coords in header - inplace means df is modified, and not that a new one is created
    """
    df = pd.read_csv(file_path, header=[1, 2])
    df.set_index(('bodyparts', 'coords'), inplace=True)
    return df

def process_loaded_dlc_data(file_path):
    """
    USE WHEN PROCESSED ALREADY, like when it has already been processed by the function above
    """
    df = pd.read_csv(file_path, header=[0, 1])
    return df

def process_timestamps_data(file_path):
    """
    uses script provided by statescript to figure out timestamps of each dlc coordinate
    """
    timestamps = readCameraModuleTimeStamps.read_timestamps(file_path)
    return timestamps

def process_statescript_log(file_path):
    """
    returns a string type containing all of the ss logs
    """
    with open(file_path) as file:
        content = file.read()
    
    return content

def create_main_data_structure(base_path): 
    """ creates a nested dictionary with parsed ss logs, dlc data & timestamps
    uses the folder structure during the original winter assignments, so does not work for every folder structure
    also, anticipates not every day

    Args:
        base_path (str): folder path containing all the rat folders

    Returns:
        dict: {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
        
    Procedure
    1. iterates over rat and day folders
        - initializes an entry in the dict for each rat
        - skips empty or non-folders
    2. checks for DLC, statescript logs, and video timestamps
        - processes & storeseach accordingly
    3. organises all into a nested dictionary
    4. logs messages for missing or duplicate files
    """
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
    """saves the dictionary data structure created by create_main_data_structure to a directory

    Args:
        data_structure (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
        save_path (str): path to directory the data structure would be saved as a folder. if it doesn't exist yet, it'll be created
    """
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
    """loads the dictionary data structure created by create_main_data_structure from a directory it was saved in

    Args:
        save_path (str): path to directory the data structure would be saved as a folder. if it doesn't exist yet, it'll be created
    
    Returns:
        (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
    """
    
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

def filter_dataframe(df, track_part = 'greenLED', std_multiplier = 7, eps = 70, min_samples = 40, distance_threshold = 190, start_index = None): # currently keeps original indices
    """
    Filters dataframes. Check to make sure it's working properly. Generally, more than 100 filtered out points is bad
    Keeps the original indices of the DataFrame

    Args:
        df (pandas.DataFrame): the data frame to be filtered
        track_part (str, optional): part of rat to be used for their position. Defaults to 'greenLED'.
        std_multiplier (int, optional): multiplier for std to define threshold beyond which jumps are excluded. Defaults to 7.
        eps (int, optional): maximum distance between two samples for one to be considered as in the neighbourhood of another for DBCSCAN. Defaults to 70.
        min_samples (int, optional): number of samples in a neighbourhood for a point to be considered a core point for DBSCAN. Defaults to 40.
        distance_threshold (int, optional): distance threshold for identifying jumps in tracking data. Defaults to 190.
        start_index (int, optional): index from which to start filtering. Defaults to None.

    Returns:
        x & y : panda.Series : filtered and interpolated coordinates for x and y
    
    Procedure:
    1. filters based on the likelihood values
    2. filters out points before start_index if provided
    3. DBSCAN
    4. filters out based on std thresholds
    5. filters based on jumps
    6. interpolate
    """
    
    # modify a copy instead of the original
    # also filter based on likelihood values
    likely_data = df[df[(track_part, 'likelihood')] > 0.999].copy()
    
    # filter out points before the rat has started its first trial
    if start_index:
        likely_data = likely_data[likely_data.index >= start_index]
    
    # DBSCAN Cluster analysis
    coordinates = likely_data[[track_part]].copy()[[(track_part, 'x'), (track_part, 'y')]]
    coordinates.dropna(inplace = True) # don't drop nan for dbscan
    
    clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(coordinates)
    labels = clustering.labels_
    #noise_points_count = (labels == -1).sum() # so ik how many points were filtered out
    #print(f"DBSCAN Filtered out {noise_points_count}")

    filtered_indices = labels != -1 # filter out noise
    filtered_data = likely_data[filtered_indices].copy()
    
    # calculate thresholds
    diff_x = df[(track_part, 'x')].diff().abs()
    diff_y = df[(track_part, 'y')].diff().abs()
    threshold_x = diff_x.std() * std_multiplier
    threshold_y = diff_y.std() * std_multiplier
    
    # calculate diff between current point and last non-jump point
    last_valid_index = 0
    jump_indices = [] # just to see how many points are jumped over
    
    for i in range(1, len(filtered_data)):
        diff_x = abs(filtered_data.iloc[i][(track_part, 'x')] - filtered_data.iloc[last_valid_index][(track_part, 'x')])
        diff_y = abs(filtered_data.iloc[i][(track_part, 'y')] - filtered_data.iloc[last_valid_index][(track_part, 'y')])
        #distance = np.sqrt(diff_x**2 + diff_y**2) # euclidean distance
        
        # check for jumps
        if diff_x > threshold_x or diff_y > threshold_y:
            # mark as NaN
            filtered_data.at[filtered_data.index[i], (track_part, 'x')] = np.nan
            filtered_data.at[filtered_data.index[i], (track_part, 'y')] = np.nan
            jump_indices.append(i)
        else:
            # udpate last valid index
            last_valid_index = i
    
    # interpolating
    filtered_data[(track_part, 'x')].interpolate(inplace = True)
    filtered_data[(track_part, 'y')].interpolate(inplace = True)
    
    print(f"number of points filtered out - {len(jump_indices)}")
    
    # final coordinate points
    x = filtered_data[(track_part, 'x')]
    y = filtered_data[(track_part, 'y')]
    
    return x, y 


### For rats where all DLC files are in one folder ------------
def save_DLC(base_path, save_path):
    """ 
    this is a function for that folder full of BP13's DLC files, where it wasn't sorted at all
    saves it according to day

    Args:
        base_path (str): the folder all the DLC files are in
        save_path (str): the folder to save into
    """
    
    rat = 'BP13'
    save_path = os.path.join(save_path, rat)
    
    for root, dirs, files in os.walk(base_path):
        for f in files:
            f = f.lower() # there were some problems with cases
            day = None
            
            # storing the DLC csv
            if fnmatch.fnmatch(f, '*dlc*.csv'): 
                file_path = os.path.join(root, f)
                dlc_data = process_dlc_data(file_path)
                
                # get the day
                match = re.search(r'Day(\d+)(?:_)?', f, re.IGNORECASE)
                if match:
                    day_number = match.group(1)
                    day = 'Day' + day_number
                
                # save
                dlc_path = os.path.join(save_path, day, f"{day}_DLC_tracking.csv")
                if fnmatch.fnmatch(f, '*_2.*'):
                    # this is when there are two files for the same day for whatever reason
                    dlc_path = os.path.join(save_path, day, f"{day}_DLC_tracking_2.csv")
                    dlc_data.to_csv(dlc_path, header = True, index = False)
                elif os.path.exists(dlc_path): # when the file already exists
                    continue
                else:
                    dlc_data.to_csv(dlc_path, header = True, index = False)

def save_timestamps(base_path, save_path):
    """this is also for BP13 where all the files are in one folder
    same as above but for timestmaps instead of DLC

    Args:
        base_path (_type_): _description_
        save_path (_type_): _description_
    """
    
    rat = 'BP13'
    save_path = os.path.join(save_path, rat)
    
    for root, dirs, files in os.walk(base_path):
        for f in files:
            f = f.lower() # there were some problems with cases
            day = None
            
            # storing the DLC csv
            if fnmatch.fnmatch(f, '*.videotimestamps'): 
                file_path = os.path.join(root, f)
                timestamps_data = process_timestamps_data(file_path)
                
                # get the day
                match = re.search(r'Day(\d+)(?:_)?', f, re.IGNORECASE)
                if match:
                    day_number = match.group(1)
                    day = 'Day' + day_number
                
                # save
                timestamps_path = os.path.join(save_path, day, f"{day}_videoTimeStamps.npy")
                if fnmatch.fnmatch(f, '*_2.*'):
                    timestamps_path = os.path.join(save_path, day, f"{day}_videoTimeStamps_2.npy")
                    np.save(timestamps_path, timestamps_data)
                elif os.path.exists(timestamps_path):
                    continue
                else:
                    np.save(timestamps_path, timestamps_data)
                    
def load_one_rat(base_path):
    """loads dictionary for just one rat

    Args:
        base_path (str): where the raw data is

    Returns:
        dict: {day: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}
    """
    
    data_structure = {}
        
    for day_folder in os.listdir(base_path): # loop for each day (in each rat folder)
        day_path = os.path.join(base_path, day_folder)
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
        data_structure[day_folder] = {
            "DLC_tracking": dlc_data,
            "stateScriptLog": ss_data,
            "videoTimeStamps": timestamps_data
        }

    return data_structure