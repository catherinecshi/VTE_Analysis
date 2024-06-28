"""
converting things from computer to be able to be processed by the rest of the code
"""

import os
import re
import fnmatch
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

import readCameraModuleTimeStamps

### LOGGING
logging.basicConfig(filename='data_processing_log.txt',
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG



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

def check_and_process_file(file_path, process_function, data_type, found_flag):
    """
    processes file and checks if there's a duplicate

    Args:
        file_path (str): file path of dlc file
        process_function (func): function corresponding to the type of data it is
        data_type (str): a string corresponding to the tyep of data it is
        found_flag (bool): whether there were previously a file of the same data type for the same day and rat already

    Returns:
        Any: the processed data - types depend on the type of data
        (bool): same as found_flag, but now returns True so indicate there's already been a file corresponding to data type for day and rat
    """
    
    if found_flag: # duplicate found
        logging.warning(f"More than one {data_type} file found: {file_path}")
        data = process_function(file_path)
        return data, found_flag
    else:
        found_flag = True
        try:
            data = process_function(file_path)
        except Exception as e:
            logging.error(f"error {e} for {file_path} data {data_type}")
            return None
            
        return data, found_flag

def create_dictionary_for_rat(rat_path, rat_folder):
    """
    makes a dictionary containing the SS log, dlc file & timestamps

    Args:
        rat_path (str): file path for where the day folders can be found
        rat_folder (str): rat id essentially

    Returns:
        (dict): {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}
    """
    
    rat_structure = {}
    
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
                    result = check_and_process_file(os.path.join(root, f), process_dlc_data, "DLC", dlc)
                    if result: # to make sure there aren't unpacking errors if none type is returned
                        temp_dlc_data, dlc = result
                        
                        if dlc_data is None: # if dlc_data is currently empty
                            dlc_data = temp_dlc_data
                        elif isinstance(dlc_data, list): # if dlc_data is already a list with two files
                            dlc_data.append(temp_dlc_data)
                        else: # dlc_data isn't empty, but there aren't multiple items in the list yet
                            dlc_data = [dlc_data, temp_dlc_data]
                    else:
                        dlc = False
                
                # handle fnmatch differently depending on whether track folder was found
                if track_folder_found:
                    # storing the statescript log
                    if fnmatch.fnmatch(f, '*track*.statescriptlog'):
                        result = check_and_process_file(os.path.join(root, f), process_statescript_log, "SS", ss)
                        if result:
                            temp_ss_data, ss = result
                            
                            if ss_data is None:
                                ss_data = temp_ss_data
                            elif isinstance(ss_data, list):
                                ss_data.append(temp_ss_data)
                            else:
                                ss_data = [ss_data, temp_ss_data]
                                
                        else:
                            ss = False
                    
                    if fnmatch.fnmatch(f, '*track*.videotimestamps'):
                        result = check_and_process_file(os.path.join(root, f_actual), process_timestamps_data, "timestamps", timestamps)
                        if result:
                            temp_timestamps_data, timestamps = result
                            
                            if timestamps_data is None:
                                timestamps_data = temp_timestamps_data
                            elif isinstance(timestamps_data, list):
                                timestamps_data.append(temp_timestamps_data)
                            else:
                                timestamps_data = [timestamps_data, temp_timestamps_data]
                                
                        else:
                            timestamps = False
                        
                else:
                    # storing the statescript log
                    if fnmatch.fnmatch(f, '*.statescriptlog'):
                        result = check_and_process_file(os.path.join(root, f), process_statescript_log, "SS", ss)
                        if result:
                            temp_ss_data, ss = result
                            
                            if ss_data is None:
                                ss_data = temp_ss_data
                            elif isinstance(ss_data, list):
                                ss_data.append(temp_ss_data)
                            else:
                                ss_data = [ss_data, temp_ss_data]
                                
                        else:
                            ss = False
                    
                    if fnmatch.fnmatch(f, '*.videotimestamps'):
                        result = check_and_process_file(os.path.join(root, f_actual), process_timestamps_data, "timestamps", timestamps)
                        if result:
                            temp_timestamps_data, timestamps = result
                            
                            if timestamps_data is None:
                                timestamps_data = temp_timestamps_data
                            elif isinstance(timestamps_data, list):
                                timestamps_data.append(temp_timestamps_data)
                            else:
                                timestamps_data = [timestamps_data, temp_timestamps_data]
                                
                        else:
                            timestamps = False
        
        # add to dictionary
        if dlc_data is None or ss_data is None or timestamps_data is None: # check for NoneTypes
            logging.warning(f"File missing for rat {rat_folder} for {day_folder} - statescript: {ss}; dlc: {dlc}; timestamps: {timestamps}")
        elif ss and dlc and timestamps: # dict
            rat_structure[day_folder] = {
                "DLC_tracking": dlc_data,
                "stateScriptLog": ss_data,
                "videoTimeStamps": timestamps_data
            }
        elif (not ss) and (not dlc) and (not timestamps):
            logging.warning(f"No timestamps, stateScriptLog or DLC file found for rat {rat_folder} for {day_folder}")
        elif (not ss) or (not dlc) or (not timestamps):
            logging.warning(f"File missing for rat {rat_folder} for {day_folder} - statescript: {ss}; dlc: {dlc}; timestamps: {timestamps}")

    return rat_structure


def create_main_data_structure(base_path, module): 
    """ creates a nested dictionary with parsed ss logs, dlc data & timestamps
    currently skips pre/post sleep

    Args:
        base_path (str): folder path containing all the rat folders
        module (str): task type. things like 'inferenceTraining' or 'moveHome'

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
        rat_path = os.path.join(base_path, rat_folder, module)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            logging.info(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        # check if implanted rat since folder system is a little different
        implant = False
        
        if any('Sleep' in folder for folder in os.listdir(rat_path)):
            implant = True
        
        # skip over empty folders
        day_folders = os.listdir(rat_path)
        if not day_folders: # if folder is empty
            logging.warning(f"{rat_path} is empty")
            continue
        
        if implant:
            track_folder = None
            
            for folder in day_folders:
                if "track" in folder:
                    track_folder = folder
                    break
            
            rat_path = os.path.join(base_path, rat_folder, module, track_folder) # so only the track folder & not the post/pre sleep is taken

        rat_structure = create_dictionary_for_rat(rat_path, rat_folder)
            
        data_structure[rat_folder] = rat_structure # first nest in dictionary
        
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

            if "DLC_tracking" in data and data["DLC_tracking"] is not None:
                if isinstance(data["DLC_tracking"], list):
                    # if data is list, iterate over each dlc file and make new csv file for each
                    for i, dlc_data in enumerate(data["DLC_tracking"]):
                        dlc_path = os.path.join(day_path, f"{day}_DLC_tracking_{i}.csv")
                        dlc_data.to_csv(dlc_path, header = True, index = False)
                else: # if data is not list, just save as csv file
                    dlc_path = os.path.join(day_path, f"{day}_DLC_tracking.csv")
                    data["DLC_tracking"].to_csv(dlc_path, header = True, index = False)
            
            if "stateScriptLog" in data and data["stateScriptLog"] is not None:
                if isinstance(data["stateScriptLog"], list):
                    for i, ss_data in enumerate(data["stateScriptLog"]):
                        ss_path = os.path.join(day_path, f"{day}_stateScriptLog_{i}.txt")
                        
                        with open(ss_path, 'w') as file:
                            file.write(ss_data)
                else:
                    ss_path = os.path.join(day_path, f"{day}_stateScriptLog.txt")
                    with open(ss_path, 'w') as file:
                        file.write(data["stateScriptLog"])
            
            if "videoTimeStamps" in data and data["videoTimeStamps"] is not None:
                if isinstance(data["videoTimeStamps"], list):
                    for i, timestamps_data in enumerate(data["videoTimeStamps"]):
                        timestamps_path = os.path.join(day_path, f"{day}_videoTimeStamps_{i}.npy")
                        np.save(timestamps_path, timestamps_data)
                else:
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