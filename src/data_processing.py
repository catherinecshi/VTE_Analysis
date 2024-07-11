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

from src import readCameraModuleTimeStamps
from src import helper_functions

### LOGGING
logging.basicConfig(filename='data_processing_log.txt',
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG


### PYLINT
# pylint: disable=logging-fstring-interpolation, broad-exception-caught

IMPLANTED_RATS = ["BP06", "BP12", "BP13", "TH405", "TH508", "BP20", "TH510", "TH605"]


def process_dlc_data(file_path):
    """
    USE WHEN COMPLETELY UNPROCESSED, like when main df hasn't been created yet
    reads csv files where first and second row and headers
    index of df is the column corresponding to bodyparts & coords in header
    """
    df = pd.read_csv(file_path, header=[1, 2])
    df.set_index(('bodyparts', 'coords'), inplace=True) # inplace -> modifies df, not making new one
    return df

def process_loaded_dlc_data(file_path):
    """USE WHEN PROCESSED ALREADY, like when it has already been processed by the function above"""
    df = pd.read_csv(file_path, header=[0, 1])
    return df

def process_timestamps_data(file_path):
    """uses script provided by statescript to figure out timestamps of each dlc coordinate"""
    timestamps = readCameraModuleTimeStamps.read_timestamps(file_path)
    return timestamps

def process_statescript_log(file_path):
    """
    returns a string type containing all of the ss logs
    """
    with open(file_path, encoding='utf-8') as file:
        content = file.read()
    
    return content

def check_and_process_file(file_path, process_function, data_type, found_flag):
    """
    processes file and checks if there's a duplicate

    Args:
        file_path (str): file path of dlc file
        process_function (func): function corresponding to the type of data it is
        data_type (str): a string corresponding to the tyep of data it is
        found_flag (bool): flag for if there has already been a file for this day & rat

    Returns:
        (Any): the processed data - types depend on the type of data
        (bool): same as found_flag, but now returns True for rat & day accounted for
    """
    if found_flag: # duplicate found
        logging.warning(f"More than one {data_type} file found: {file_path}")
        data = process_function(file_path)
    else:
        found_flag = True
        try:
            data = process_function(file_path)
        except Exception as e:
            logging.error(f"error {e} for {file_path} data {data_type}")
            return None
    return data, found_flag

def convert_all_timestamps(base_path):
    for rat_folder in os.listdir(base_path):
        rat_path = os.path.join(base_path, rat_folder, "inferenceTraining")
        if not os.path.isdir(rat_path):
            # skip .DS_Store
            logging.info(f"skipping {rat_folder}")
            continue
        
        for day_folder in os.listdir(rat_path):
            day_path = os.path.join(rat_path, day_folder)
            for root, _, files in os.walk(day_path):
                for f in files:
                    if ".videoTimeStamps" in f and '.npy' not in f:
                        try:
                            original_ts_path = os.path.join(root, f)
                            timestamps = process_timestamps_data(original_ts_path)
                            new_ts_name = f + ".npy"
                            new_ts_path = os.path.join(root, new_ts_name)
                            np.save(new_ts_path, timestamps)
                            
                            if timestamps is None:
                                logging.error(f"failed to process for {rat_folder} on {day_folder}")
                        except Exception as e:
                            logging.error(f'error {e} for {rat_folder} on {day_folder}')
                        else:
                            os.remove(original_ts_path)
                        
def create_dictionary_for_rat(rat_path, rat_folder):
    """
    makes a dictionary containing the SS log, dlc file & timestamps

    Args:
        rat_path (str): file path for where the day folders can be found
        rat_folder (str): rat id essentially

    Returns:
        (dict): {day_folder: 
                {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}
    """
    rat_structure = {}
    for day_folder in os.listdir(rat_path):
        day_path = os.path.join(rat_path, day_folder)
        dlc_data = None
        ss_data = None
        timestamps_data = None
        track_folder_found = False # check if there are any track folders in this day folder
        
        # flags for if timestamps, ss log & dlc csv file has already been found for rat/day
        ss = False
        dlc = False
        timestamps = False
    
        for root, dirs, files in os.walk(day_path):
            # check if there are any track folders bc only implanted rats have it
            # if so, it changes the naming conventions for folders
            for dir_name in dirs:
                if fnmatch.fnmatch(dir_name.lower(), '*track*'):
                    track_folder_found = True
            for f in files:
                f_actual = f
                f = f.lower() # here bc there were some problems with cases
                if fnmatch.fnmatch(f, '*dlc*.csv'):
                    result = check_and_process_file(os.path.join(root, f),
                                                    process_dlc_data,
                                                    "DLC",
                                                    dlc)
                    if result is not None: # to make sure there aren't unpacking errors if none type is returned
                        temp_dlc_data, dlc = result
                        if dlc_data is None:
                            dlc_data = temp_dlc_data
                        elif isinstance(dlc_data, list): # already 2+ files at least
                            dlc_data.append(temp_dlc_data)
                        else: # dlc_data isn't empty, but there aren't 2+ files in the list yet
                            dlc_data = [dlc_data, temp_dlc_data]
                    else:
                        dlc = False
                
                # handle fnmatch differently depending on whether track folder was found
                if track_folder_found is not None:
                    # storing the statescript log
                    if fnmatch.fnmatch(f, '*track*.statescriptlog'):
                        result = check_and_process_file(os.path.join(root, f), 
                                                        process_statescript_log, 
                                                        "SS", 
                                                        ss)
                        if result is not None:
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
                        result = check_and_process_file(os.path.join(root, f_actual), 
                                                        process_timestamps_data, 
                                                        "timestamps", 
                                                        timestamps)
                        if result is not None:
                            temp_timestamps_data, timestamps = result
                            if timestamps_data is None:
                                timestamps_data = temp_timestamps_data
                            elif isinstance(timestamps_data, list):
                                timestamps_data.append(temp_timestamps_data)
                            else:
                                timestamps_data = [timestamps_data, temp_timestamps_data]  
                        else:
                            timestamps = False
                else: # if track folder wasn't found
                    if fnmatch.fnmatch(f, '*.statescriptlog'):
                        result = check_and_process_file(os.path.join(root, f), 
                                                        process_statescript_log, 
                                                        "SS", 
                                                        ss)
                        if result is not None:
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
                        result = check_and_process_file(os.path.join(root, f_actual), 
                                                        process_timestamps_data, 
                                                        "timestamps", 
                                                        timestamps)
                        if result is not None:
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
            logging.warning(
                f"File missing for rat {rat_folder} for {day_folder} - "
                f"statescript: {ss}; dlc: {dlc}; timestamps: {timestamps}"
                )
        elif ss and dlc and timestamps: # dict
            rat_structure[day_folder] = {
                "DLC_tracking": dlc_data,
                "stateScriptLog": ss_data,
                "videoTimeStamps": timestamps_data
            }
        elif (not ss) and (not dlc) and (not timestamps):
            logging.warning(
                f"No timestamps, stateScriptLog or DLC file found "
                f"for rat {rat_folder} for {day_folder}"
            )
        elif (not ss) or (not dlc) or (not timestamps):
            logging.warning(
                f"File missing for rat {rat_folder} for {day_folder} - "
                "statescript: {ss}; dlc: {dlc}; timestamps: {timestamps}"
            )

    return rat_structure

def create_main_data_structure(base_path, module):
    """ creates a nested dictionary with parsed ss logs, dlc data & timestamps
    currently skips pre/post sleep

    Args:
        base_path (str): folder path containing all the rat folders
        module (str): task type. things like 'inferenceTraining' or 'moveHome'

    Returns:
        dict: {rat_folder: {day_folder: 
              {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
        
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
            # so only the track folder & not the post/pre sleep is taken
            rat_path = os.path.join(base_path, rat_folder, module, track_folder)

        rat_structure = create_dictionary_for_rat(rat_path, rat_folder)
        data_structure[rat_folder] = rat_structure # first nest in dictionary
        
    return data_structure

def save_data_structure(data_structure, save_path):
    """saves the dictionary data structure created by create_main_data_structure to a directory

    Args:
        data_structure (dict): {rat_folder: {day_folder: 
                               {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
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
                        
                        with open(ss_path, 'w', encoding='utf-8') as file:
                            file.write(ss_data)
                else:
                    ss_path = os.path.join(day_path, f"{day}_stateScriptLog.txt")
                    with open(ss_path, 'w', encoding='utf-8') as file:
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
        
            for root, _, files in os.walk(day_path): # look at all the files in the day folder
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
                        try:
                            timestamps_data = np.load(file_path)
                        except Exception as e:
                            print(f'error {e} for {rat_folder} on {day_folder}')
            
            # add to dictionary
            data_structure[rat_folder][day_folder] = {
                "DLC_tracking": dlc_data,
                "stateScriptLog": ss_data,
                "videoTimeStamps": timestamps_data
            }

    return data_structure

def get_first_time(content):
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line and stripped_line[0].isdigit():
            match = re.match(r'^(\d+)\s', stripped_line)
            if match:
                first_time = match.group(1)
                return first_time

def get_last_time(content):
    last_line = None
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line and stripped_line[0].isdigit():
            match = re.match(r'^(\d+)\s', stripped_line)
            if match:
                last_line = match.group(1)
    return last_line

def get_time_diff(content_1, content_2):
    """returns time diff between two ss logs. returns None if no/negative time diff"""
    
    first_end = get_last_time(content_1)
    second_start = get_first_time(content_2)
    
    if second_start > first_end:
        diff = helper_functions.get_time(second_start - first_end) # convert to seconds
        return diff
    elif second_start == first_end:
        return 0
    else:
        return None
    
def concat_ss(ss_0, ss_1):
    # cut off the comments of the second statescript and start from the first trial
    content_1 = process_statescript_log(ss_0)
    filtered_lines = [line for line in content_1.splitlines() if not line.startswith('#')]
    
    # concat them
    content_0 = process_statescript_log(ss_1)
    concatenated_content = content_0 + '\n' + filtered_lines
    
    return concatenated_content

def concat_dlc(dlc_0, dlc_1, day, rat):
    df1 = pd.read_csv(dlc_0, header=[0, 1])
    df2 = pd.read_csv(dlc_1, skiprows=[0, 1]) # skip the header
    
    # get the index at which they split
    split_index = df1.index[-1]
    logging.info(f'concatenating {rat} for {day}'
                    f'last index for first dlc at {split_index}')
    
    # update dlc
    new_dlc = pd.concat([df1, df2], ignore_index=True) # ignore index ensures the indices are continuous
    return new_dlc

def determine_timestamps_diff(timestamps_0, timestamps_1, rat, day, time_diff=None):
    if timestamps_1[0] == 0 or timestamps_1[0] == 1:
        logging.info(f'second timestamps file reset for {rat} on {day}')
        if time_diff is not None:
            new_timestamps_1 = timestamps_1 + timestamps_0[-1] + time_diff
        else:
            new_timestamps_1 = timestamps_1 + timestamps_0[-1] # so the timestamps continue
        return  np.append(timestamps_0, new_timestamps_1)
    elif timestamps_1[0] > timestamps_0[-1]:
        timestamps_time_diff = timestamps_1[0] - timestamps_0[-1]
        logging.info(f'second timestamps file is {timestamps_time_diff} ahead of first file for {rat} on {day}')
        # check if the time diff is similar between timestamps and ss log
        logging.info(f'timestamps_time_diff - ss_time_diff = {timestamps_time_diff - time_diff}')
        return np.append(timestamps_0, timestamps_1)
    else:
        logging.info(f'second timestamps file start at {timestamps_1[0]} for rat {rat} on {day}')
        # check if time diff match up with ss time diff
        if time_diff is not None:
            new_timestamps_1 = timestamps_1 + timestamps_0[-1] + time_diff
        else:
            new_timestamps_1 = timestamps_1 + timestamps_0[-1]
        return np.append(timestamps_0, new_timestamps_1)

def concat_timestamps(timestamps_0, timestamps_1, day, rat, time_diff=None):
    if time_diff is None:
        logging.info(f'second ss file reset for {rat} on {day}')
        return determine_timestamps_diff(timestamps_0, timestamps_1, rat, day)
    elif time_diff == 0:
        logging.info(f'second ss file started where first left for {rat} on {day}')
        return determine_timestamps_diff(timestamps_0, timestamps_1, rat, day)
    else:
        logging.info(f'second ss file is {time_diff} ahead of first for {rat} on {day}')
        return determine_timestamps_diff(timestamps_0, timestamps_1, rat, day)

def concat_duplicates(save_path):
    for rat_folder in os.listdir(save_path): # loop for each rat
        rat_path = os.path.join(save_path, rat_folder, "inferenceTraining")
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            logging.info(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        for day_folder in os.listdir(rat_path): # loop for each day (in each rat folder)
            day_path = os.path.join(rat_path, day_folder)
            
            dlc_0 = None # will always represent the cumulative concatenated file at the end of each loop of day
            dlc_1 = None
            dlc_2 = None
            ss_0 = None
            ss_1 = None
            ss_2 = None
            timestamps_0 = None
            timestamps_1 = None
            timestamps_2 = None
            track_folder = None
            
            for root, dirs, files in os.walk(day_path):
                for f in files:
                    file_path = os.path.join(root, f)
                    f = f.lower()
                    if fnmatch.fnmatch(f, "*.csv"):
                        dlc_0 = f
                    elif fnmatch.fnmatch(f, "*_0.txt"):
                        ss_0 = f
                    elif fnmatch.fnmatch(f, "*_0.npy"):
                        ts_path = os.path.join(root, f)
                        timestamps_0 = np.load(ts_path)
                    
                    if fnmatch.fnmatch(f, "*_1.csv"):
                        dlc_1 = f
                    elif fnmatch.fnmatch(f, "*_1.txt"):
                        ss_1 = f
                    elif fnmatch.fnmatch(f, "*_1.npy"):
                        ts_path = os.path.join(root, f)
                        timestamps_1 = np.load(ts_path)
                    
                    if fnmatch.fnmatch(f, "*_2.csv"):
                        dlc_2 = f
                    elif fnmatch.fnmatch(f, "*_2.txt"):
                        ss_2 = f
                    elif fnmatch.fnmatch(f, "*_2.npy*"):
                        ts_path = os.path.join(root, f)
                        timestamps_2 = np.load(ts_path)

                # check if there is a track folder (for implanted rats)
                for directory in dirs:
                    if directory is not None:
                        track_folder = directory
                
            # concatenate everything
            success = True
            if track_folder is not None:
                dlc_path = os.path.join(day_path, track_folder, f"{day_folder}_concat_DLC_tracking.csv")
                ss_path = os.path.join(day_path, track_folder, f"{day_folder}_concat_stateScriptLog.txt")
                timestamps_path = os.path.join(day_path, track_folder, f"{day_folder}_concat_videoTimeStamps.npy")
            else:
                dlc_path = os.path.join(day_path, f"{day_folder}_concat_DLC_tracking.csv")
                ss_path = os.path.join(day_path, f"{day_folder}_concat_stateScriptLog.txt")
                timestamps_path = os.path.join(day_path, f"{day_folder}_concat_videoTimeStamps.npy")
            
            try:
                if dlc_2 is not None:
                    new_dlc = concat_dlc(dlc_0, dlc_1, day_folder, rat_folder)
                    new_new_dlc = concat_dlc(new_dlc, dlc_2, day_folder, rat_folder)
                    new_new_dlc.to_csv(dlc_path) # save to same folder
                elif dlc_1 is not None:
                    new_dlc = concat_dlc(dlc_0, dlc_1, day_folder, rat_folder)
                    new_dlc.to_csv(dlc_path) # save to same folder
                
                if ss_2 is not None: # timestamps is in this bc it requires ss time diff to work
                    new_ss = concat_ss(ss_0, ss_1)
                    new_new_ss = concat_ss(new_ss, ss_2)
                    with open(ss_path, "w", encoding="utf-8") as file:
                        file.write(new_new_ss)
                        
                    # (optional) return ss time diffs
                    time_diff_0 = get_time_diff(ss_0, ss_1)
                    time_diff_1 = get_time_diff(ss_1, ss_2)
                    if timestamps_2 is not None:
                        new_timestamps = concat_timestamps(timestamps_0, timestamps_1, day_folder, rat_folder, time_diff_0)
                        new_new_timestamps = concat_timestamps(new_timestamps, timestamps_2, day_folder, rat_folder, time_diff_1)
                        np.save(timestamps_path, new_new_timestamps)
                    elif timestamps_1 is not None:
                        new_timestamps = concat_timestamps(timestamps_0, timestamps_1, day_folder, rat_folder, time_diff_0)
                        np.save(timestamps_path, new_timestamps)
                    else:
                        logging.warning(f"3 statescripts but 1 or no timestamps for {rat_folder} on {day_folder}")
                elif ss_1 is not None:
                    new_ss = concat_ss(ss_0, ss_1)
                    with open(ss_path, "w", encoding="utf-8") as file:
                        file.write(new_ss)
                        
                    # (optional) return ss time diffs 
                    time_diff = get_time_diff(ss_0, ss_1)
                    if timestamps_1 is not None:
                        new_timestamps = concat_timestamps(timestamps_0, timestamps_1, day_folder, rat_folder, time_diff)
                        np.save(timestamps_path, new_timestamps)
                    else:
                        logging.warning(f"2 statescripts but 1 or no timestamps for {rat_folder} on {day_folder}")
                else:
                    new_timestamps = concat_timestamps(timestamps_0, timestamps_1, day_folder, rat_folder)
                    np.save(timestamps_path, new_timestamps)
            except Exception as e:
                logging.error(f"concatenation failed for {rat_folder} on {day_folder} because error {e}")
                success = False
            
            if success: # delete old files
                for root, _, files in os.walk(day_path):
                    for f in files:
                        file_path = os.path.join(root, f)
                        if fnmatch.fnmatch(f, "*_0*"):
                            os.remove(file_path)
                        elif fnmatch.fnmatch(f, "*_1*"):
                            os.remove(file_path)
                        elif fnmatch.fnmatch(f, "*_2*"):
                            os.remove(file_path)

def filter_dataframe(df, track_part = "greenLED", std_multiplier = 7, eps = 70, min_samples = 40, distance_threshold = 190, start_index = None): # currently keeps original indices
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
    likely_data = df[df[(track_part, "likelihood")] > 0.999].copy()
    
    # filter out points before the rat has started its first trial
    if start_index:
        likely_data = likely_data[likely_data.index >= start_index]
    
    # DBSCAN Cluster analysis
    coordinates = likely_data[[track_part]].copy()[[(track_part, "x"), (track_part, "y")]]
    coordinates.dropna(inplace = True) # don't drop nan for dbscan
    
    clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(coordinates)
    labels = clustering.labels_
    #noise_points_count = (labels == -1).sum() # so ik how many points were filtered out
    #print(f"DBSCAN Filtered out {noise_points_count}")

    filtered_indices = labels != -1 # filter out noise
    filtered_data = likely_data[filtered_indices].copy()
    
    # calculate thresholds
    diff_x = df[(track_part, "x")].diff().abs()
    diff_y = df[(track_part, "y")].diff().abs()
    threshold_x = diff_x.std() * std_multiplier
    threshold_y = diff_y.std() * std_multiplier
    
    # calculate diff between current point and last non-jump point
    last_valid_index = 0
    jump_indices = [] # just to see how many points are jumped over
    
    for i in range(1, len(filtered_data)):
        diff_x = abs(filtered_data.iloc[i][(track_part, "x")] - filtered_data.iloc[last_valid_index][(track_part, "x")])
        diff_y = abs(filtered_data.iloc[i][(track_part, "y")] - filtered_data.iloc[last_valid_index][(track_part, "y")])
        #distance = np.sqrt(diff_x**2 + diff_y**2) # euclidean distance
        
        # check for jumps
        if diff_x > threshold_x or diff_y > threshold_y:
            # mark as NaN
            filtered_data.at[filtered_data.index[i], (track_part, "x")] = np.nan
            filtered_data.at[filtered_data.index[i], (track_part, "y")] = np.nan
            jump_indices.append(i)
        else:
            # udpate last valid index
            last_valid_index = i
    
    # interpolating
    filtered_data[(track_part, "x")].interpolate(inplace = True)
    filtered_data[(track_part, "y")].interpolate(inplace = True)
    
    print(f"number of points filtered out - {len(jump_indices)}")
    
    # final coordinate points
    x = filtered_data[(track_part, "x")]
    y = filtered_data[(track_part, "y")]
    
    return x, y 

save_path = '/Users/catpillow/Documents/VTE_Analysis/data/VTE_Data'
#timestamps = readCameraModuleTimeStamps.read_timestamps(timestamps_path)
#convert_all_timestamps(save_path)
concat_duplicates(save_path)