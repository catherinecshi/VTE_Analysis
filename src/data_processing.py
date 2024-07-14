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
CURRENT_RAT = ""
CURRENT_DAY = ""

def update_rat(rat):
    global CURRENT_RAT
    CURRENT_RAT = rat

def update_day(day):
    global CURRENT_DAY
    CURRENT_DAY = day

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
    with open(file_path, encoding="utf-8") as file:
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
                                logging.error(f"failed to process ts for {rat_folder} on {day_folder}")
                        except Exception as e:
                            logging.error(f'error {e} for {rat_folder} on {day_folder}')
                        else:
                            os.remove(original_ts_path)

def convert_all_statescripts(base_path):
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
                    if ".stateScriptLog" in f and ".txt" not in f:
                        try:
                            original_ss_path = os.path.join(root, f)
                            ss_log = process_statescript_log(original_ss_path)
                            new_ss_name = f + ".txt"
                            new_ss_path = os.path.join(root, new_ss_name)
                            with open(new_ss_path, "w", encoding="utf-8") as file:
                                file.write(ss_log)
                            
                            if ss_log is None:
                                logging.error(f"failed to process ss for {rat_folder} on {day_folder}")
                        except Exception as e:
                            logging.error(f"error {e} for {rat_folder} on {day_folder}")
                        else:
                            os.remove(original_ss_path)
                        
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

def get_time_diff(ss_1, ss_2):
    """
    returns time diff between two ss logs if second ss start is ahead of the end of first ss
    returns 0 if ss log is continuous (first ss end = second ss start)
    returns None if start of second ss is behind end of first ss
    """
    content_1 = process_statescript_log(ss_1)
    content_2 = process_statescript_log(ss_2)
    
    first_start = helper_functions.get_first_time(content_1)
    first_end = helper_functions.get_last_time(content_1)
    second_start = helper_functions.get_first_time(content_2)
    second_end = helper_functions.get_last_time(content_2)
    
    diff = second_start - first_end
    ss_diff_info = (first_start, first_end, second_start, second_end, diff)
    
    if second_start > first_end:
        logging.info(f"statescript files have a gap with length {diff}")
        return diff, ss_diff_info
    elif second_start == first_end:
        logging.info("statescript files are continuous")
        return 0, ss_diff_info
    else:
        logging.info(f"first statescript file starts at {first_start}"
                     f"first statescript file ends at {first_end}"
                     f"second statescript file starts at {second_start}")
        return None, ss_diff_info
    
def concat_ss(ss_1, ss_2):
    """concatenates two statescript logs - doesn't include comments of second ss log"""
    # cut off the comments of the second statescript and start from the first trial
    content_1 = process_statescript_log(ss_1)
    filtered_lines = [line for line in content_1.splitlines() if not line.startswith('#')]
    filtered_content = "\n".join(filtered_lines)
    
    # concat them
    content_0 = process_statescript_log(ss_2)
    concatenated_content = content_0 + "\n" + filtered_content
    
    logging.info(f"concatenating {CURRENT_RAT} for {CURRENT_DAY} - SS")
    
    return concatenated_content

def concat_dlc(dlc_1, dlc_2):
    csv_1 = pd.read_csv(dlc_1, header=[0, 1])
    csv_2 = pd.read_csv(dlc_2, skiprows=[0, 1]) # skip the header
    
    # get the index at which they split
    last_index = csv_1.index[-1]
    first_index = csv_2.index[0]
    logging.info(f"concatenating {CURRENT_RAT} for {CURRENT_DAY} - DLC"
                 f"last index for first dlc at {last_index}"
                 f"first index for second dlc at {first_index}")
    
    # update dlc
    new_dlc = pd.concat([csv_1, csv_2], ignore_index=True) # ignore index ensures the indices are continuous
    dlc_diff = first_index - last_index
    dlc_diff_info = (csv_1.index[0], csv_1.index[-1], csv_2.index[0], csv_2.index[-1], dlc_diff)
    return new_dlc, dlc_diff_info

def concat_timestamps(timestamps_1, timestamps_2, ss_time_diff=None):
    ts_1 = np.load(timestamps_1)
    ts_2 = np.load(timestamps_2)
    timestamps_time_diff = ts_2[0] - ts_1[-1]
    timestamps_diff_info = (ts_1[0], ts_1[-1], ts_2[0], ts_2[-1], timestamps_time_diff)
    
    if ts_2[0] == ts_1[0] or ts_2[0] < ts_1[0]:
        logging.info(f"concatenating {CURRENT_RAT} for {CURRENT_DAY} - timestamps reset"
                     f"first timestamps started at {ts_1[0]} and ends at {ts_1[-1]}"
                     f"second timestamps started at {ts_2[0]} and ends at {ts_2[-1]}")
        new_timestamps_1 = ts_2 + ts_1[-1] # so the timestamps continue
        new_timestamps = np.append(ts_1, new_timestamps_1)
        return new_timestamps, timestamps_diff_info
    elif ts_2[0] > ts_1[-1]:
        logging.info(f"concatenating {CURRENT_RAT} for {CURRENT_DAY} - timestamps jump"
                     f"second timestamps file is {timestamps_time_diff} ahead of first file")
                     # check if the time diff is similar between timestamps and ss log
        new_timestamps = np.append(ts_1, ts_2)
        return new_timestamps, timestamps_diff_info
    else:
        logging.info(f"concatenating {CURRENT_RAT} for {CURRENT_DAY} - timestamps reset"
                     f"first timestamps started at {ts_1[0]} and ends at {ts_1[-1]}"
                     f"second timestamps started at {ts_2[0]} and ends at {ts_2[-1]}")
        new_timestamps_1 = ts_2 + ts_1[-1] # so the timestamps continue
        new_timestamps = np.append(ts_1, new_timestamps_1)
        return new_timestamps, timestamps_diff_info

def make_concat_file_names(path_name):
    file_name = path_name.split("/")
    parts = re.split(r"Day\d+", file_name[-1])
    day_substring = re.search(r"Day\d+", file_name[-1])
    if parts is not None and len(parts) > 1 and day_substring is not None:
        day_substring = day_substring.group()
        new_file_name = parts[0] + day_substring + "_concat" + parts[1]
        return new_file_name
    else:
        return None

def find_duplicates(day_path):
    duplicate_files = {"dlc_1": "", "dlc_2": "", "dlc_3": "",
                        "ss_1": "", "ss_2": "", "ss_3": "",
                        "timestamps_1": "", "timestamps_2": "", "timestamps_3": ""}
    
    for root, _, files in os.walk(day_path):
        for f in files:
            if ".csv" in f and "_2_track" not in f and "_3_track" not in f:
                duplicate_files["dlc_1"] = os.path.join(root, f)
            elif ".stateScriptLog" in f and "_2.stateScriptLog" not in f and "_3.stateScriptLog" not in f:
                duplicate_files["ss_1"] = os.path.join(root, f)
            elif ".videoTimeStamps" in f and "_2.1.videoTimeStamps" not in f and "_3.1.videoTimeStamps" not in f:
                duplicate_files["timestamps_1"] = os.path.join(root, f)
            
            if fnmatch.fnmatch(f, "*_2_track*.csv"):
                duplicate_files["dlc_2"] = os.path.join(root, f)
            elif fnmatch.fnmatch(f, "*_2.stateScriptLog*"):
                duplicate_files["ss_2"] = os.path.join(root, f)
            elif fnmatch.fnmatch(f, "*_2.1.videoTimeStamps*"):
                duplicate_files["timestamps_2"] = os.path.join(root, f)
            
            if fnmatch.fnmatch(f, "*_3_track*.csv"):
                duplicate_files["dlc_3"] = os.path.join(root, f)
            elif fnmatch.fnmatch(f, "*_3.stateScriptLog*"):
                duplicate_files["ss_3"] = os.path.join(root, f)
            elif fnmatch.fnmatch(f, "*_3.1.videoTimeStamps*"):
                duplicate_files["timestamps_3"] = os.path.join(root, f)
    
    return duplicate_files

def save_concats(duplicate_files, dlc_path, ss_path, timestamps_path):
    dlc_diff_info_1 = None # reset so things don't accidentally get saved
    ss_diff_info_1 = None
    ts_diff_info_1 = None
    dlc_diff_info_2 = None
    ss_diff_info_2 = None
    ts_diff_info_2 = None

    if duplicate_files["dlc_3"] != "":
        new_dlc, dlc_diff_info_1 = concat_dlc(duplicate_files["dlc_1"], duplicate_files["dlc_2"])
        new_new_dlc, dlc_diff_info_2 = concat_dlc(new_dlc, duplicate_files["dlc_3"])
        new_new_dlc.to_csv(dlc_path)
    elif duplicate_files["dlc_2"] != "":
        new_dlc, dlc_diff_info_1 = concat_dlc(duplicate_files["dlc_1"], duplicate_files["dlc_2"])
        new_dlc.to_csv(dlc_path)
    
    if duplicate_files["ss_3"] != "": 
        new_ss = concat_ss(duplicate_files["ss_1"], duplicate_files["ss_2"])
        new_new_ss = concat_ss(new_ss, duplicate_files["ss_3"])
        with open(ss_path, "w", encoding="utf-8") as file:
            file.write(new_new_ss)
        time_diff_1, ss_diff_info_1 = get_time_diff(duplicate_files["ss_1"], duplicate_files["ss_2"]) # return ss time diffs for timestamps
        time_diff_2, ss_diff_info_2 = get_time_diff(duplicate_files["ss_2"], duplicate_files["ss_3"])
    elif duplicate_files["ss_2"] != "":
        new_ss = concat_ss(duplicate_files["ss_1"], duplicate_files["ss_2"])
        with open(ss_path, "w", encoding="utf-8") as file:
            file.write(new_ss)
        time_diff_1, ss_diff_info_1 = get_time_diff(duplicate_files["ss_1"], duplicate_files["ss_2"]) # return ss time diffs for timestamps
        
    if duplicate_files["timestamps_3"] != "":
        if time_diff_1 is not None and time_diff_2 is not None:
            new_timestamps, ts_diff_info_1 = concat_timestamps(duplicate_files["timestamps_1"], duplicate_files["timestamps_2"], time_diff_1)
            new_new_timestamps, ts_diff_info_2 = concat_timestamps(new_timestamps, duplicate_files["timestamps_3"], time_diff_2)
            np.save(timestamps_path, new_new_timestamps)
        elif time_diff_1 is not None:
            new_timestamps, ts_diff_info_1 = concat_timestamps(duplicate_files["timestamps_1"], duplicate_files["timestamps_2"], time_diff_1)
            new_new_timestamps, ts_diff_info_2 = concat_timestamps(new_timestamps, duplicate_files["timestamps_3"])
            np.save(timestamps_path, new_timestamps)
    elif duplicate_files["timestamps_2"] != "" and time_diff_1 is not None:
        new_timestamps, ts_diff_info_1 = concat_timestamps(duplicate_files["timestamps_1"], duplicate_files["timestamps_2"], time_diff_1)
        np.save(timestamps_path, new_timestamps)
    elif duplicate_files["timestamps_2"] != "":
        new_timestamps, ts_diff_info_1 = concat_timestamps(duplicate_files["timestamps_1"], duplicate_files["timestamps_2"])
        np.save(timestamps_path, new_timestamps)
    
    return dlc_diff_info_1, dlc_diff_info_2, ss_diff_info_1, ss_diff_info_2, ts_diff_info_1, ts_diff_info_2

def save_diff_info(dlc_diff_info, ss_diff_info, ts_diff_info):
    if dlc_diff_info is not None:
        dlc_first_start, dlc_first_end, dlc_second_start, dlc_second_end, dlc_diff = dlc_diff_info
    else:
        dlc_first_start, dlc_first_end, dlc_second_start, dlc_second_end, dlc_diff = None, None, None, None, None
    if ss_diff_info is not None:
        ss_first_start, ss_first_end, ss_second_start, ss_second_end, ss_diff = ss_diff_info
    else:
        ss_first_start, ss_first_end, ss_second_start, ss_second_end, ss_diff = None, None, None, None, None
    if ts_diff_info is not None:
        ts_first_start, ts_first_end, ts_second_start, ts_second_end, ts_diff = ts_diff_info
    else:
        ts_first_start, ts_first_end, ts_second_start, ts_second_end, ts_diff = None, None, None, None, None
    
    diff_info = {"rat": CURRENT_RAT, "day": CURRENT_DAY,
                    "dlc_first_start": dlc_first_start, "dlc_first_end": dlc_first_end,
                    "dlc_second_start": dlc_second_start, "dlc_second_end": dlc_second_end, "dlc_diff": dlc_diff,
                    "ss_first_start": ss_first_start, "ss_first_end": ss_first_end,
                    "ss_second_start": ss_second_start, "ss_second_end": ss_second_end, "ss_diff": ss_diff,
                    "ts_first_start": ts_first_start, "ts_first_end": ts_first_end,
                    "ts_second_start": ts_second_start, "ts_second_end": ts_second_end, "ts_diff": ts_diff}
    return diff_info

def concat_duplicates(save_path):
    diff_info = []
    
    for rat_folder in os.listdir(save_path): # loop for each rat
        rat_path = os.path.join(save_path, rat_folder, "inferenceTraining")
        update_rat(rat_folder)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            logging.info(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        for day_folder in os.listdir(rat_path): # loop for each day (in each rat folder)
            day_path = os.path.join(rat_path, day_folder)
            update_day(day_folder)
            
            duplicate_files = find_duplicates(day_path)
            if any([duplicate_files["dlc_2"] == "",
                      duplicate_files["ss_2"] == "",
                      duplicate_files["timestamps_2"] == ""]):
                continue # skip folders with no duplicates
            logging.info(f"{CURRENT_RAT} on {CURRENT_DAY} has a duplicate"
                         f"{duplicate_files}")
            
            track_folder = None
            for _, dirs, _ in os.walk(day_path):
                # check if there is a track folder (for implanted rats)
                for directory in dirs:
                    if directory is not None:
                        track_folder = directory
                
            # new file names
            if track_folder is not None:
                dlc_file_name = make_concat_file_names(duplicate_files["dlc_1"])
                if dlc_file_name is not None:
                    dlc_path = os.path.join(day_path, track_folder, make_concat_file_names(duplicate_files["dlc_1"]))
                
                ss_file_name = make_concat_file_names(duplicate_files["ss_1"])
                if ss_file_name is not None:
                    ss_path = os.path.join(day_path, track_folder, make_concat_file_names(duplicate_files["ss_1"]))
                
                timestamps_file_name = make_concat_file_names(duplicate_files["timestamps_1"])
                if timestamps_file_name is not None:
                    timestamps_path = os.path.join(day_path, track_folder, make_concat_file_names(duplicate_files["timestamps_1"]))
            else:
                dlc_file_name = make_concat_file_names(duplicate_files["dlc_1"])
                if dlc_file_name is not None:
                    dlc_path = os.path.join(day_path, make_concat_file_names(duplicate_files["dlc_1"]))
                
                ss_file_name = make_concat_file_names(duplicate_files["ss_1"])
                if ss_file_name is not None:
                    ss_path = os.path.join(day_path, make_concat_file_names(duplicate_files["ss_1"]))
                
                timestamps_file_name = make_concat_file_names(duplicate_files["timestamps_1"])
                if timestamps_file_name is not None:
                    timestamps_path = os.path.join(day_path, make_concat_file_names(duplicate_files["timestamps_1"]))
            
            # concatenate everything
            try:
                dlc_diff_info_1, dlc_diff_info_2, ss_diff_info_1, ss_diff_info_2, ts_diff_info_1, ts_diff_info_2 = save_concats(duplicate_files, dlc_path, ss_path, timestamps_path)
            except Exception as e:
                logging.error(f"concatenation failed for {rat_folder} on {day_folder} because error {e}")
            else:
                current_diff_info = save_diff_info(dlc_diff_info_1, ss_diff_info_1, ts_diff_info_1)
                if current_diff_info is not None:
                    diff_info.append(current_diff_info)
                
                current_diff_info = save_diff_info(dlc_diff_info_2, ss_diff_info_2, ts_diff_info_2)
                if current_diff_info is not None:
                    diff_info.append(current_diff_info)
                
                for file_type, file_path in duplicate_files.items():
                    if (("dlc" in file_type and duplicate_files["dlc_2"] == "") or
                       ("ss" in file_type and duplicate_files["ss_2"] == "") or
                       ("timestamps" in file_type and duplicate_files["timestamps_2"] == "")):
                        continue # sometimes there are uneven amounts of duplicates across the three file types

                    if file_path != "":
                        path_parts = file_path.split("/")
                        extension = path_parts[-1].split(".")[-1]
                        new_file_name = "old_" + file_type + "." + extension
                        new_file_path = os.path.join("/".join(path_parts[:-1]), new_file_name)
                        os.rename(file_path, new_file_path)

    diff_info_df = pd.DataFrame(diff_info)
    diff_info_path = os.path.join(save_path, "diff_info.csv")
    diff_info_df.to_csv(diff_info_path)

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
