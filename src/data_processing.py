"""
converting things from computer to be able to be processed by the rest of the code
"""

import os
import re
import fnmatch
import logging
import shutil
from datetime import datetime

import pandas as pd
import numpy as np

from src import readCameraModuleTimeStamps
from src import helper

### LOGGING
logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# makes a new log everytime the code runs by checking the time
log_file = datetime.now().strftime("/Users/catpillow/Documents/VTE_Analysis/doc/data_processing_log_%Y%m%d_%H%M%S.txt")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

### PYLINT
# pylint: disable=logging-fstring-interpolation, broad-exception-caught, trailing-whitespace

IMPLANTED_RATS = ["BP06", "BP07", "BP12", "BP13", "TH405", "TH508", "BP20", "TH510", "TH605"]
MODULE = "inferenceTraining"


### SINGLE FILE PROCESSING ---------
def process_dlc_data(file_path):
    """processes normal dlc by skipping first row & appointing second + third row as headers
    processes concat dlc by appointing first and second row as headers, and first column as index"""
    try:
        if "concat" in file_path:
            header_df = pd.read_csv(file_path, nrows=2, header=None)
            dtype_dict = {col: float for col in range(0, len(header_df.columns))} # save memory
            data_df = pd.read_csv(file_path, skiprows=2, dtype=dtype_dict, header=None)
            
            # assign header
            headers = pd.MultiIndex.from_arrays(header_df.values)
            data_df.columns = headers
            
            data_df.drop(data_df.columns[0], axis=1, inplace=True)
        else:
            header_df = pd.read_csv(file_path, skiprows=[0], header=None, nrows=2)
            dtype_dict = {col: float for col in range(0, len(header_df.columns))}
            data_df = pd.read_csv(file_path, skiprows=3, dtype=dtype_dict, header=None)
            
            # assign header
            headers = pd.MultiIndex.from_arrays(header_df.values)
            data_df.columns = headers
    except ValueError as e:
        print(e, file_path)
        return None
    
    return data_df

def process_timestamps_data(file_path):
    """uses script provided by statescript to figure out timestamps of each dlc coordinate"""
    timestamps = readCameraModuleTimeStamps.read_timestamps_new(file_path)
    return timestamps

def process_statescript_log(file_path):
    """returns a string type containing all of the ss logs"""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    
    return content


### PRE-PROCESSING -----------
def convert_all_timestamps(base_path):
    """converts all timestamps into .npy arrays in base_path"""
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
                    if ".videoTimeStamps" in f and ".npy" not in f:
                        try:
                            original_ts_path = os.path.join(root, f)
                            timestamps = process_timestamps_data(original_ts_path)
                            new_ts_name = f + ".npy"
                            new_ts_path = os.path.join(root, new_ts_name)
                            np.save(new_ts_path, timestamps)
                            
                            if timestamps is None:
                                logging.error(f"failed to process ts for {rat_folder} on {day_folder}")
                        except Exception as e:
                            print(f'error {e} for {rat_folder} on {day_folder}')
                        else:
                            os.remove(original_ts_path)

def convert_all_statescripts(base_path):
    """converts all statescripts into txt files in base_path"""
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

def initial_to_inference(base_path):
    """this is just bc i'm using initialTraining for BP07 instead of inferenceTraining & annoying naming issues"""
    for day_folder in os.listdir(base_path): # base path should only lead to BP07
        day_path = os.path.join(base_path, day_folder)
        for root, _, files in os.walk(day_path):
            for f in files:
                if "initialTraining" in f:
                    try:
                        old_path = os.path.join(root, f)
                        new_path = re.sub("initialTraining", "inferenceTraining", old_path)
                    except Exception as e:
                        logging.warning(f"error {e} for {day_folder} for initial to inference")
                    else:
                        os.rename(old_path, new_path)
                

### CREATE MAIN DATA STRUCTURE -----------
def load_data_structure(save_path):
    """loads the dictionary data structure created by create_main_data_structure from a directory it was saved in

    Args:
        save_path (str): path to directory the data structure would be saved as a folder. if it doesn't exist yet, it'll be created
    
    Returns:
        (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
    """
    
    data_structure = {}

    for rat_folder in os.listdir(save_path):
        rat_path = os.path.join(save_path, rat_folder, MODULE)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path) or "DS_Store" in rat_folder:
            logging.info(f"Skipping over non-directory folder: {rat_path}")
            continue
            
        data_structure[rat_folder] = {} # first nest in dictionary
        
        for day_folder in os.listdir(rat_path):
            day_path = os.path.join(rat_path, day_folder)
            if "DS_Store" in day_folder:
                logging.info(f"Skipping over DS_Store: {day_path}")
                continue
            
            dlc_data = None
            ss_data = None
            timestamps_data = None
        
            for root, _, files in os.walk(day_path):
                for f in files:
                    f = f.lower()
                    
                    # skip over old files
                    if "old_" in f:
                        continue
                    
                    # storing the DLC csv
                    if fnmatch.fnmatch(f, "*dlc*.csv"):
                        file_path = os.path.join(root, f)
                        dlc_data = process_dlc_data(file_path)
                    
                    # storing the statescript log
                    if fnmatch.fnmatch(f, "*statescriptlog*"):
                        file_path = os.path.join(root, f)
                        ss_data = process_statescript_log(file_path)
                    
                    # storing the video timestamps array
                    if fnmatch.fnmatch(f, "*videotimestamps*"):
                        file_path = os.path.join(root, f)
                        try:
                            timestamps_data = np.load(file_path)
                        except Exception as e: # bc trodes code keeps on giving me errors :(
                            print(f'error {e} for {rat_folder} on {day_folder}')
            
            # add to dictionary
            data_structure[rat_folder][day_folder] = {
                "DLC_tracking": dlc_data,
                "stateScriptLog": ss_data,
                "videoTimeStamps": timestamps_data
            }
            
            # log missing data
            if dlc_data is None:
                logging.warning(f"DLC data missing for {rat_folder} on {day_folder}")
            if ss_data is None:
                logging.warning(f"SS data missing for {rat_folder} on {day_folder}")
            if timestamps_data is None:
                logging.warning(f"TS data missing for {rat_folder} on {day_folder}")
    
    timestamps_path = "/Users/catpillow/Documents/VTE_Analysis/data/timestamps"
    for rat in os.listdir(timestamps_path):
        if ".DS" in rat:
            continue

        rat_path = os.path.join(timestamps_path, rat, "inferenceTraining")
        for day in os.listdir(rat_path):
            if ".DS" in day:
                continue

            day_path = os.path.join(rat_path, day)
            for root, _, files in os.walk(day_path):
                for file in files:
                    if ".videoTimeStamps" not in file:
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        data_structure[rat][day]["videoTimeStamps"] = np.load(file_path)
                    except KeyError as ke:
                        print(f"key error for {rat} on {day}")
                    except ValueError as ve:
                        print(f"value error for {rat} on {day}")
    
    statescript_path = "/Users/catpillow/Documents/VTE_Analysis/data/statescripts"
    for rat in os.listdir(statescript_path):
        if ".DS" in rat:
            continue

        rat_path = os.path.join(statescript_path, rat, "inferenceTraining")
        for day in os.listdir(rat_path):
            if ".DS" in day:
                continue

            day_path = os.path.join(rat_path, day)
            for root, _, files in os.walk(day_path):
                for file in files:
                    if ".stateScriptLog" not in file:
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        data_structure[rat][day]["stateScriptLog"] = process_statescript_log(file_path)
                    except KeyError as ke:
                        print(f"key error for {rat} on {day}")

    return data_structure


### CONCATENATION METHODS ------------
def get_time_diff(ss_1, ss_2):
    """
    returns time diff between two ss logs if second ss start is ahead of the end of first ss
    returns 0 if ss log is continuous (first ss end = second ss start)
    returns None if start of second ss is behind end of first ss
    """
    content_1 = process_statescript_log(ss_1)
    content_2 = process_statescript_log(ss_2)
    
    first_start = helper.get_first_time(content_1)
    first_end = helper.get_last_time(content_1)
    second_start = helper.get_first_time(content_2)
    second_end = helper.get_last_time(content_2)
    
    diff = second_start - first_end
    ss_diff_info = (first_start, first_end, second_start, second_end, diff)
    
    if second_start > first_end:
        logging.info(f"statescript files have a gap with length {diff}")
        return ss_diff_info
    elif second_start == first_end:
        logging.info("statescript files are continuous")
        return ss_diff_info
    else:
        logging.info(f"first statescript file starts at {first_start}"
                     f"first statescript file ends at {first_end}"
                     f"second statescript file starts at {second_start}")
        return ss_diff_info

def concat_ss(ss_1, ss_2):
    """concatenates two statescript logs - doesn't include comments of second ss log"""
    # cut off the comments of the second statescript and start from the first trial
    content_1 = process_statescript_log(ss_1)
    filtered_lines = [line for line in content_1.splitlines() if not line.startswith('#')]
    filtered_content = "\n".join(filtered_lines)
    
    # concat them
    content_0 = process_statescript_log(ss_2)
    concatenated_content = content_0 + "\n" + filtered_content
    
    logging.info(f"concatenating {helper.CURRENT_RAT} for {helper.CURRENT_DAY} - SS")
    
    return concatenated_content

def concat_dlc(dlc_1, dlc_2):
    """
    concatenates two dlcs, ends with the body part only in first row for first row headers

    Args:
        dlc_1 (str): file path to first dlc csv
        dlc_2 (str): file path to second dlc csv

    Returns:
        pd.DataFrame: dataframe of concatenated dlcs - first column are og coords
        tuples: (dlc_1[0], dlc_1[-1], dlc_2[0], dlc_2[-1], 
                 diff in coords between end of first and start of second)
    """
    df_1 = pd.read_csv(dlc_1, skiprows=[0], header=[0, 1])
    df_2 = pd.read_csv(dlc_2, skiprows=[0], header=[0, 1])
    df_1.columns = df_1.columns.to_flat_index()
    df_2.columns = df_2.columns.to_flat_index()
    
    # get the index at which they split
    last_index = df_1.index[-1]
    first_index = df_2.index[0]
    logging.info(f"concatenating {helper.CURRENT_RAT} for {helper.CURRENT_DAY} - DLC"
                 f"last index for first dlc at {last_index}"
                 f"first index for second dlc at {first_index}")
    
    # update dlc
    new_dlc = pd.concat([df_1, df_2], ignore_index=True) # ignore index ensures the indices are continuous
    new_dlc.columns = pd.MultiIndex.from_tuples(new_dlc.columns)
    
    dlc_diff = first_index - last_index
    dlc_diff_info = (df_1.index[0], df_1.index[-1], df_2.index[0], df_2.index[-1], dlc_diff)
    return new_dlc, dlc_diff_info

def concat_timestamps(timestamps_1, timestamps_2):
    """
    concats timestamps files that are already numpy arrays
    adds last timestamp of first file to all numbers in second file

    Args:
        timestamps_1 (np.Array): first timestamps file
        timestamps_2 (np.Array): second timestamps file

    Returns:
        np.Array: concatenated array of timestamps
        tuple: (ts_1[0], ts_1[-1], ts_2[0], ts_2[-1],
                diff between end of first and start of second file)
    """
    ts_1 = np.load(timestamps_1)
    ts_2 = np.load(timestamps_2)
    timestamps_time_diff = ts_2[0] - ts_1[-1]
    timestamps_diff_info = (ts_1[0], ts_1[-1], ts_2[0], ts_2[-1], timestamps_time_diff)
    
    if ts_2[0] == ts_1[0] or ts_2[0] < ts_1[0]:
        logging.info(f"concatenating {helper.CURRENT_RAT} for {helper.CURRENT_DAY} - timestamps reset"
                     f"first timestamps started at {ts_1[0]} and ends at {ts_1[-1]}"
                     f"second timestamps started at {ts_2[0]} and ends at {ts_2[-1]}")
        new_timestamps_1 = ts_2 + ts_1[-1] # so the timestamps continue
        new_timestamps = np.append(ts_1, new_timestamps_1)
        return new_timestamps, timestamps_diff_info
    elif ts_2[0] > ts_1[-1]:
        logging.info(f"concatenating {helper.CURRENT_RAT} for {helper.CURRENT_DAY} - timestamps jump"
                     f"second timestamps file is {timestamps_time_diff} ahead of first file")
                     # check if the time diff is similar between timestamps and ss log
        new_timestamps = np.append(ts_1, ts_2)
        return new_timestamps, timestamps_diff_info
    else:
        logging.info(f"concatenating {helper.CURRENT_RAT} for {helper.CURRENT_DAY} - timestamps reset"
                     f"first timestamps started at {ts_1[0]} and ends at {ts_1[-1]}"
                     f"second timestamps started at {ts_2[0]} and ends at {ts_2[-1]}")
        new_timestamps_1 = ts_2 + ts_1[-1] # so the timestamps continue
        new_timestamps = np.append(ts_1, new_timestamps_1)
        return new_timestamps, timestamps_diff_info

def make_concat_file_names(path_name):
    """takes path name and returns new path name with concat after day"""
    
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
    """returns dictionary of filepaths of duplicates in day_path"""
    
    duplicate_files = {"dlc_1": "", "dlc_2": "",
                        "ss_1": "", "ss_2": "",
                        "timestamps_1": "", "timestamps_2": ""}
    
    for root, _, files in os.walk(day_path):
        for f in files:
            if "old_" in f:
                continue
            
            if ".csv" in f and "_2_track" not in f and "_2." not in f:
                duplicate_files["dlc_1"] = os.path.join(root, f)
            elif ".stateScriptLog" in f and "_2.stateScriptLog" not in f:
                duplicate_files["ss_1"] = os.path.join(root, f)
            elif ".videoTimeStamps" in f and "_2.1.videoTimeStamps" not in f:
                duplicate_files["timestamps_1"] = os.path.join(root, f)
            
            if fnmatch.fnmatch(f, "*_2_track*.csv") or fnmatch.fnmatch(f, "*_2.*csv"):
                duplicate_files["dlc_2"] = os.path.join(root, f)
            elif fnmatch.fnmatch(f, "*_2.stateScriptLog*"):
                duplicate_files["ss_2"] = os.path.join(root, f)
            elif fnmatch.fnmatch(f, "*_2.1.videoTimeStamps*"):
                duplicate_files["timestamps_2"] = os.path.join(root, f)
    
    return duplicate_files

def find_duplicates_implanted(day_path):
    """same as find_duplicate, but with some filename diff bc implanted rats"""
    
    duplicate_files = {"dlc_1": "", "dlc_2": "",
                        "ss_1": "", "ss_2": "",
                        "timestamps_1": "", "timestamps_2": ""}
    
    for root, _, files in os.walk(day_path):
        for f in files:
            if "old_" in f:
                continue
            
            if ".csv" in f and "_track_2" not in f:
                duplicate_files["dlc_1"] = os.path.join(root, f)
            elif ".stateScriptLog" in f and "track_2.stateScriptLog" not in f:
                duplicate_files["ss_1"] = os.path.join(root, f)
            elif ".videoTimeStamps" in f and "track_2.1.videoTimeStamps" not in f:
                duplicate_files["timestamps_1"] = os.path.join(root, f)
            
            if fnmatch.fnmatch(f, "*_track_2*.csv") or fnmatch.fnmatch(f, "*_2_track*.csv"):
                duplicate_files["dlc_2"] = os.path.join(root, f)
            elif fnmatch.fnmatch(f, "*_track_2.stateScriptLog*"):
                duplicate_files["ss_2"] = os.path.join(root, f)
            elif fnmatch.fnmatch(f, "*_track_2.1.videoTimeStamps*") or fnmatch.fnmatch(f, "*_track_2.videoTimeStamps*"):
                duplicate_files["timestamps_2"] = os.path.join(root, f)
    
    return duplicate_files

def save_concats(duplicate_files, dlc_path, ss_path, timestamps_path):
    """saves concatenated files and gets the information about the diff between duplicates

    Args:
        duplicate_files (dict): dict of duplicate files as found by find_duplicates
        dlc_path (str): new file path name for concat dlc files
        ss_path (str): new file path name for concat ss files
        timestamps_path (str): new file path name for concat ts files

    Returns:
        (tuples): all three are tuples with format:
                  (start of first, end of first, start of second, end of second,
                   difference between end of first and start of second)
    """
    
    dlc_diff_info = None # reset so things don't accidentally get saved
    ss_diff_info = None
    ts_diff_info = None

    if duplicate_files["dlc_2"] != "":
        new_dlc, dlc_diff_info = concat_dlc(duplicate_files["dlc_1"], duplicate_files["dlc_2"])
        new_dlc.to_csv(dlc_path)

    if duplicate_files["ss_2"] != "":
        new_ss = concat_ss(duplicate_files["ss_1"], duplicate_files["ss_2"])
        with open(ss_path, "w", encoding="utf-8") as file:
            file.write(new_ss)
        ss_diff_info = get_time_diff(duplicate_files["ss_1"], duplicate_files["ss_2"]) # return ss time diffs for timestamps
        
    if duplicate_files["timestamps_2"] != "":
        new_timestamps, ts_diff_info = concat_timestamps(duplicate_files["timestamps_1"], duplicate_files["timestamps_2"])
        np.save(timestamps_path, new_timestamps)
    
    return dlc_diff_info, ss_diff_info, ts_diff_info

def create_diff_info_dict(dlc_diff_info, ss_diff_info, ts_diff_info):
    """creates a big dictionary for all the diff info"""
    
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
    
    diff_info = {"rat": helper.CURRENT_RAT, "day": helper.CURRENT_DAY,
                    "dlc_first_start": dlc_first_start, "dlc_first_end": dlc_first_end,
                    "dlc_second_start": dlc_second_start, "dlc_second_end": dlc_second_end, "dlc_diff": dlc_diff,
                    "ss_first_start": ss_first_start, "ss_first_end": ss_first_end,
                    "ss_second_start": ss_second_start, "ss_second_end": ss_second_end, "ss_diff": ss_diff,
                    "ts_first_start": ts_first_start, "ts_first_end": ts_first_end,
                    "ts_second_start": ts_second_start, "ts_second_end": ts_second_end, "ts_diff": ts_diff}
    return diff_info

def concat_duplicates(save_path):
    """main method to call to concatenate all the duplicate files in save_path
    
    Procedure:
        1. loops through each rat & day and find duplicates for that day (find_duplicates)
            - skip rats without any duplicates
        2. get the dir name of track folder if present (for implanted rats)
        3. use that dir name to make file path names for each data type (make_concat_file_names)
        4. make and save the concat file (save_concats)
        5. get the information about the concatenated files (create_diff_info_dict)
        6. rename the old files used for concatenation
            - if an implanted rat, move into another folder just for old files
        7. save the info about conatenated files into csv

    Args:
        save_path (str): path that all the data can be found in

    Raises:
        helper_functions.ExpectationError: raises error when problem with making diff info dict
    """
    
    diff_info = []
    
    for rat_folder in os.listdir(save_path): # loop for each rat
        rat_path = os.path.join(save_path, rat_folder, "inferenceTraining")
        helper.update_rat(rat_folder)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            logging.info(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        for day_folder in os.listdir(rat_path): # loop for each day (in each rat folder)
            day_path = os.path.join(rat_path, day_folder)
            helper.update_day(day_folder)
            
            if any(rat_folder in rat for rat in IMPLANTED_RATS):
                duplicate_files = find_duplicates_implanted(day_path) # naming convention is diff for implanted rats
            else:
                duplicate_files = find_duplicates(day_path)
            if duplicate_files["dlc_2"] == "" and \
                duplicate_files["ss_2"] == "" and \
                duplicate_files["timestamps_2"] == "":
                continue # skip folders with no duplicates
            logging.info(f"{helper.CURRENT_RAT} on {helper.CURRENT_DAY} has a duplicate"
                         f"{duplicate_files}")
            
            track_folder = None
            track_folder_2 = None
            for _, dirs, _ in os.walk(day_path):
                # check if there is a track folder (for implanted rats)
                for directory in dirs:
                    if directory is not None and "_track_2" not in directory:
                        track_folder = directory
                    elif directory is not None and "_track_2" in directory:
                        track_folder_2 = directory
                
            # new file names
            if track_folder is not None:
                dlc_file_name = make_concat_file_names(duplicate_files["dlc_1"])
                if dlc_file_name is not None:
                    dlc_path = os.path.join(day_path,
                                            track_folder,
                                            make_concat_file_names(duplicate_files["dlc_1"]))
                
                ss_file_name = make_concat_file_names(duplicate_files["ss_1"])
                if ss_file_name is not None:
                    ss_path = os.path.join(day_path,
                                           track_folder,
                                           make_concat_file_names(duplicate_files["ss_1"]))
                
                timestamps_file_name = make_concat_file_names(duplicate_files["timestamps_1"])
                if timestamps_file_name is not None:
                    timestamps_path = os.path.join(day_path,
                                                   track_folder,
                                                   make_concat_file_names(duplicate_files["timestamps_1"]))
            else:
                dlc_file_name = make_concat_file_names(duplicate_files["dlc_1"])
                if dlc_file_name is not None:
                    dlc_path = os.path.join(day_path,
                                            make_concat_file_names(duplicate_files["dlc_1"]))
                
                ss_file_name = make_concat_file_names(duplicate_files["ss_1"])
                if ss_file_name is not None:
                    ss_path = os.path.join(day_path,
                                           make_concat_file_names(duplicate_files["ss_1"]))
                
                timestamps_file_name = make_concat_file_names(duplicate_files["timestamps_1"])
                if timestamps_file_name is not None:
                    timestamps_path = os.path.join(day_path,
                                                   make_concat_file_names(duplicate_files["timestamps_1"]))
            
            # concatenate everything
            try:
                dlc_diff_info, ss_diff_info, ts_diff_info = save_concats(duplicate_files, dlc_path, ss_path, timestamps_path)
            except Exception as e:
                logging.error(f"concatenation failed for {rat_folder} on {day_folder} because error {e}")
            else:
                try:
                    current_diff_info = create_diff_info_dict(dlc_diff_info, ss_diff_info, ts_diff_info)
                except Exception as e:
                    logging.critical(f"{helper.CURRENT_RAT} problem on {helper.CURRENT_DAY} with error {e}")
                    raise helper.ExpectationError("saving diff info", f"{e}")
                
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
                        
                        if track_folder_2 is not None:
                            change_folder_path = os.path.join(day_path, track_folder_2, new_file_name)
                            shutil.move(file_path, change_folder_path)
                        else:
                            new_file_path = os.path.join("/".join(path_parts[:-1]), new_file_name)
                            os.rename(file_path, new_file_path)
                    
                if track_folder_2 is not None:
                    new_track_folder = "old_folder"
                    track_folder_path = os.path.join(day_path, track_folder_2)
                    new_track_folder_path = os.path.join(day_path, new_track_folder)
                    os.rename(track_folder_path, new_track_folder_path)

    diff_info_df = pd.DataFrame(diff_info)
    diff_info_path = os.path.join(save_path, "diff_info.csv")
    diff_info_df.to_csv(diff_info_path)

