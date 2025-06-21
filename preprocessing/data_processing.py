"""
converting things from computer to be able to be processed by the rest of the code
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict

from config import settings
from config.paths import paths, remote
from preprocessing import readCameraModuleTimeStamps
from utilities import time_utils
from utilities import error_types
from utilities import logging_utils

# pylint: disable=logging-fstring-interpolation, broad-exception-caught, trailing-whitespace
logger = logging_utils.get_module_logger("data_processing")

# ==============================================================================
# PROCESS SINGLE FILES
# ==============================================================================
def process_dlc_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    processes normal dlc by skipping first row & appointing second + third row as headers
    processes concat dlc by appointing first and second row as headers, and first column as index
    """
    file_path = str(file_path)
    
    try:
        if "concat" in file_path:
            header_df = pd.read_csv(file_path, nrows=2, header=None)
            dtype_dict = {col: float for col in range(0, len(header_df.columns))} # save memory
            data_df = pd.read_csv(file_path, skiprows=2, dtype=dtype_dict, header=None)
            
            # assign header
            headers = pd.MultiIndex.from_arrays(header_df.values)
            data_df.columns = headers
            
            data_df.drop(data_df.columns[0], axis=1, inplace=True)
            logger.info(f"successfully processed concat dlc for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
        else:
            header_df = pd.read_csv(file_path, skiprows=[0], header=None, nrows=2)
            dtype_dict = {col: float for col in range(0, len(header_df.columns))}
            data_df = pd.read_csv(file_path, skiprows=3, dtype=dtype_dict, header=None)
            
            # assign header
            headers = pd.MultiIndex.from_arrays(header_df.values)
            data_df.columns = headers
            
            logger.info(f"successfully processed dlc for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
    except ValueError as e:
        logger.error(f"value error {e} happened for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
    except Exception as e:
        logger.error(f"unexpected error when processing dlc for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}: {e}")
    
    return data_df

def process_timestamps_data(file_path: Union[str, Path]) -> np.ndarray:
    """uses script provided by statescript to figure out timestamps of each dlc coordinate"""
    timestamps = readCameraModuleTimeStamps.read_timestamps_new(file_path)
    return timestamps

def process_statescript_log(file_path: Union[str, Path]) -> str:
    """returns a string type containing all of the ss logs"""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    
    return content


# ==============================================================================
# PRE-PROCESSING FILES
# ==============================================================================

def convert_all_timestamps(base_path: Union[str, Path]):
    """converts all timestamps into .npy arrays in base_path"""
    base_path = Path(base_path) # if base path is str
    
    pattern = f"*/{remote.module}/**/*videoTimeStamps*"
    timestamps_files = [f for f in base_path.glob(pattern) if not f.name.endswith(".npy")]
    logger.info(f"{len(timestamps_files)} timestamps files to convert")
    
    for original_ts_path in timestamps_files:
        try:
            # update settings
            relative_path = original_ts_path.relative_to(base_path)
            rat_folder = relative_path.parts[0]
            day_folder = relative_path.parts[2]
            settings.update_rat(rat_folder)
            settings.update_day(day_folder)
            logger.info(f"converting timestamps for {rat_folder} on {day_folder}")
            
            # process
            timestamps = process_timestamps_data(original_ts_path)
            
            if timestamps is None:
                logger.error(f"timestamps is None for {rat_folder} on {day_folder}")
                raise error_types.UnexpectedNoneError("convert_all_timestamps", "timestamps")
            
            # create new .npy file path
            new_ts_path = original_ts_path.with_suffix(original_ts_path.suffix + ".npy")
            np.save(new_ts_path, timestamps)
        except (IOError, OSError) as e:
            logger.error(f"File operation error for {rat_folder} on {day_folder}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {rat_folder} on {day_folder}: {e}")
        else:
            # remove original if save was successful
            original_ts_path.unlink()
            logger.info(f"successfully converted {original_ts_path} to {new_ts_path}")

def convert_all_statescripts(base_path: Union[str, Path]):
    """converts all statescripts into txt files in base_path"""
    base_path = Path(base_path) # if base path is str
    
    pattern = f"*/{remote.module}/**/*stateScriptLog*"
    ss_files = [f for f in base_path.glob(pattern) if not f.name.endswith(".txt")]
    logger.info(f"{len(ss_files)} ss files to convert")
    
    for original_ss_path in ss_files:
        try:
            # update settings
            relative_path = original_ss_path.relative_to(base_path)
            rat_folder = relative_path.parts[0]
            day_folder = relative_path.parts[2]
            settings.update_rat(rat_folder)
            settings.update_day(day_folder)
            logger.info(f"converting timestamps for {rat_folder} on {day_folder}")
            
            # process
            ss = process_statescript_log(original_ss_path)
            
            if ss is None:
                logger.error(f"ss is None for {rat_folder} on {day_folder}")
                raise error_types.UnexpectedNoneError("convert_all_statescripts", "ss")
            
            # create new .npy file path
            new_ss_path = original_ss_path.with_suffix(original_ss_path.suffix + ".txt")
            with open(new_ss_path, "w", encoding="utf-8") as file:
                file.write(ss)
                
        except (IOError, OSError) as e:
            logger.error(f"File operation error for {rat_folder} on {day_folder}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {rat_folder} on {day_folder}: {e}")
        else:
            # remove original if save was successful
            original_ss_path.unlink()
            logger.info(f"successfully converted {original_ss_path} to {new_ss_path}")

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
                        logger.warning(f"error {e} for {day_folder} for initial to inference")
                    else:
                        os.rename(old_path, new_path)
                

# ==============================================================================
# CONCATENATION
# ==============================================================================

def get_ss_time_diff(ss_1: Union[str, Path], ss_2: Union[str, Path]) -> Optional[tuple]:
    """
    returns info about time diff between two ss logs
    """
    content_1 = process_statescript_log(ss_1)
    content_2 = process_statescript_log(ss_2)
    
    try:
        ss_diff_info = time_utils.get_time_diff(content_1, content_2)
    except error_types.UnexpectedNoneError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"unexpected error for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}: {e}")
    else:
        first_start, first_end, second_start, second_end, diff = ss_diff_info
        
        if second_start > first_end:
            logger.info(f"statescript files have a gap with length {diff}")
            return ss_diff_info
        elif second_start == first_end:
            logger.info("statescript files are continuous")
            return ss_diff_info
        else:
            logger.info(f"first statescript file starts at {first_start}"
                        f"first statescript file ends at {first_end}"
                        f"second statescript file starts at {second_start}"
                        f"second statescript file ends at {second_end}")
            return ss_diff_info

def concat_ss(ss_1: Union[str, Path], ss_2: Union[str, Path]) -> str:
    """concatenates two statescript logs - doesn't include comments of second ss log"""
    # cut off the comments of the second statescript and start from the first trial
    content_1 = process_statescript_log(ss_1)
    filtered_lines = [line for line in content_1.splitlines() if not line.startswith('#')]
    filtered_content = "\n".join(filtered_lines)
    
    # concat them
    content_0 = process_statescript_log(ss_2)
    concatenated_content = content_0 + "\n" + filtered_content
    
    logger.info(f"successfully concatenated {settings.CURRENT_RAT} for {settings.CURRENT_DAY} - SS")
    return concatenated_content

def concat_dlc(dlc_1: Union[str, Path], dlc_2: Union[str, Path]) -> tuple[pd.DataFrame, tuple]:
    """
    concatenates two dlcs, ends with the body part only in first row for first row headers

    Parameters:
    - dlc_1: file path to first dlc csv
    - dlc_2: file path to second dlc csv

    Returns:
    - pd.DataFrame: dataframe of concatenated dlcs - first column are og coords
    - tuples: (dlc_1[0], dlc_1[-1], dlc_2[0], dlc_2[-1], 
               diff in coords between end of first and start of second)
    """
    df_1 = pd.read_csv(dlc_1, skiprows=[0], header=[0, 1])
    df_2 = pd.read_csv(dlc_2, skiprows=[0], header=[0, 1])
    df_1.columns = df_1.columns.to_flat_index()
    df_2.columns = df_2.columns.to_flat_index()
    
    # get the index at which they split
    last_index = df_1.index[-1]
    first_index = df_2.index[0]
    logger.info(f"concatenating {settings.CURRENT_RAT} for {settings.CURRENT_DAY} - DLC"
                 f"last index for first dlc at {last_index}"
                 f"first index for second dlc at {first_index}")
    
    # update dlc
    new_dlc = pd.concat([df_1, df_2], ignore_index=True) # ignore index ensures the indices are continuous
    new_dlc.columns = pd.MultiIndex.from_tuples(new_dlc.columns)
    
    dlc_diff = first_index - last_index
    dlc_diff_info = (df_1.index[0], df_1.index[-1], df_2.index[0], df_2.index[-1], dlc_diff)
    return new_dlc, dlc_diff_info

def concat_timestamps(timestamps_1: Union[str, Path], timestamps_2: Union[str, Path]) -> tuple[np.ndarray, tuple]:
    """
    concats timestamps files that are already numpy arrays
    adds last timestamp of first file to all numbers in second file

    Args:
        timestamps_1: first timestamps file path
        timestamps_2: second timestamps file path

    Returns:
        np.ndarray: concatenated array of timestamps
        tuple: (ts_1[0], ts_1[-1], ts_2[0], ts_2[-1],
                diff between end of first and start of second file)
    """
    ts_1 = np.load(timestamps_1, allow_pickle=True)
    ts_2 = np.load(timestamps_2, allow_pickle=True)
    timestamps_time_diff = ts_2[0] - ts_1[-1]
    timestamps_diff_info = (ts_1[0], ts_1[-1], ts_2[0], ts_2[-1], timestamps_time_diff)
    
    if ts_2[0] == ts_1[0] or ts_2[0] < ts_1[0]:
        logger.info(f"concatenating {settings.CURRENT_RAT} for {settings.CURRENT_DAY} - timestamps reset"
                     f"first timestamps started at {ts_1[0]} and ends at {ts_1[-1]}"
                     f"second timestamps started at {ts_2[0]} and ends at {ts_2[-1]}")
        new_timestamps_1 = ts_2 + ts_1[-1] # so the timestamps continue
        new_timestamps = np.append(ts_1, new_timestamps_1)
        return new_timestamps, timestamps_diff_info
    elif ts_2[0] > ts_1[-1]:
        logger.info(f"concatenating {settings.CURRENT_RAT} for {settings.CURRENT_DAY} - timestamps jump"
                     f"second timestamps file is {timestamps_time_diff} ahead of first file")
                     # check if the time diff is similar between timestamps and ss log
        new_timestamps = np.append(ts_1, ts_2)
        return new_timestamps, timestamps_diff_info
    else:
        logger.info(f"concatenating {settings.CURRENT_RAT} for {settings.CURRENT_DAY} - timestamps reset"
                     f"first timestamps started at {ts_1[0]} and ends at {ts_1[-1]}"
                     f"second timestamps started at {ts_2[0]} and ends at {ts_2[-1]}")
        new_timestamps_1 = ts_2 + ts_1[-1] # so the timestamps continue
        new_timestamps = np.append(ts_1, new_timestamps_1)
        return new_timestamps, timestamps_diff_info

def make_concat_file_names(path_name: Optional[Union[str, Path]]) -> Optional[Path]:
    """takes path name and returns new path name with concat after day"""
    if path_name is None:
        return None
    
    path_name = Path(path_name)
    
    file_name = path_name.name
    parts = re.split(r"(Day\d+)", file_name)
    
    if parts is None:
        logger.error(f"parts is None in concat file name for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
        raise error_types.UnexpectedNoneError("make_concat_file_name", "parts")
    
    # should theoretically only have [0] -> before, [1] -> day pattern, [2] -> after
    if len(parts) == 3:
        prefix, day_part, suffix = parts
        new_file_name = prefix + day_part + "_concat" + suffix
        
        return path_name.parent / new_file_name
    else:
        # Return None if no Day pattern was found
        return None
    
def find_duplicates(day_path: Union[str, Path]) -> Dict:
    """
    returns dictionary of filepaths (Path) of duplicates in day_path
    - dlc_1, dlc_2, ss_1, ss_2, timestamps_1, timestamps_2 as keys
    """
    day_path = Path(day_path)
    
    duplicate_files: Dict[str, Optional[Path]] = {
        "dlc_1": None, "dlc_2": None,
        "ss_1": None, "ss_2": None,
        "timestamps_1": None, "timestamps_2": None
    }
    
    for file_path in day_path.rglob("*"):
        # skip directories
        if not file_path.is_file():
            continue

        f = file_path.name
        
        if "old_" in f:
            continue

        if f.endswith(".csv") and "_2_track" not in f and "_2." not in f:
            duplicate_files["dlc_1"] = file_path
        elif (f.endswith(".csv") and ("_2_track" in f or "_2." in f)):
            duplicate_files["dlc_2"] = file_path
        elif (f.endswith(".stateScriptLog.txt") and not f.endswith("_2.stateScriptLog.txt")):
            duplicate_files["ss_1"] = file_path
        elif "_2.stateScriptLog" in f:
            duplicate_files["ss_2"] = file_path
        elif (f.endswith(".videoTimeStamps.npy") and not f.endswith("_2.1.videoTimeStamps.npy")):
            duplicate_files["timestamps_1"] = file_path
        elif "_2.1.videoTimeStamps" in f:
            duplicate_files["timestamps_2"] = file_path
        
    return duplicate_files

def find_duplicates_implanted(day_path: Union[str, Path]) -> Dict:
    """same as find_duplicate, but with some filename diff bc implanted rats"""
    
    day_path = Path(day_path)
    
    duplicate_files: Dict[str, Optional[Path]] = {
        "dlc_1": None, "dlc_2": None,
        "ss_1": None, "ss_2": None,
        "timestamps_1": None, "timestamps_2": None
    }
    
    for file_path in day_path.rglob("*"):
        if not file_path.is_file():
            continue

        f = file_path.name
        
        if "old_" in f:
            continue

        if f.endswith(".csv") and "_track_2" not in f:
            duplicate_files["dlc_1"] = file_path
        elif (f.endswith(".csv") and ("_track_2" in f or "_2_track" in f)):
            duplicate_files["dlc_2"] = file_path
        elif (f.endswith(".stateScriptLog.txt") and not f.endswith("track_2.stateScriptLog.txt")):
            duplicate_files["ss_1"] = file_path
        elif "_track_2.stateScriptLog" in f:
            duplicate_files["ss_2"] = file_path
        elif (f.endswith(".videoTimeStamps.npy") and not f.endswith("track_2.1.videoTimeStamps.npy")):
            duplicate_files["timestamps_1"] = file_path
        elif "track_2.1.videoTimeStamps" in f or "_track_2.videoTimeStamps" in f:
            duplicate_files["timestamps_2"] = file_path
        
    return duplicate_files

def save_concats(
        duplicate_files: Dict, 
        dlc_path: Union[str, Path], 
        ss_path: Union[str, Path], 
        timestamps_path: Union[str, Path]
    ) -> tuple[Optional[tuple], Optional[tuple], Optional[tuple]]:
    """saves concatenated files and gets the information about the diff between duplicates

    Parameters:
    - duplicate_files: dict of duplicate files as found by find_duplicates
    - dlc_path: new file path name for concat dlc files
    - ss_path: new file path name for concat ss files
    - timestamps_path: new file path name for concat ts files

    Returns:
    - (tuples): all three are tuples with format:
                (start of first, end of first, start of second, end of second,
                difference between end of first and start of second)
    """
    dlc_path = Path(dlc_path)
    ss_path = Path(ss_path) 
    timestamps_path = Path(timestamps_path)
    
    dlc_diff_info = None
    ss_diff_info = None
    ts_diff_info = None

    # Process DLC files if duplicate
    if duplicate_files["dlc_2"] is not None:
        logger.info(f"Concatenating DLC files for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
        
        try:
            # Get the concatenated data and diff information
            new_dlc, dlc_diff_info = concat_dlc(duplicate_files["dlc_1"], duplicate_files["dlc_2"])

            dlc_path.parent.mkdir(parents=True, exist_ok=True)
            
            new_dlc.to_csv(dlc_path)
            logger.info(f"Successfully saved concatenated DLC file to {dlc_path}")
            
        except Exception as e:
            logger.error(f"Failed to concatenate DLC files: {e}")
            raise

    if duplicate_files["ss_2"] is not None:
        logger.info(f"Concatenating StateScript files for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
        
        try:
            new_ss = concat_ss(duplicate_files["ss_1"], duplicate_files["ss_2"])
            
            ss_path.parent.mkdir(parents=True, exist_ok=True)
            
            with ss_path.open("w", encoding="utf-8") as file:
                file.write(new_ss)
            
            # Get timing difference information for analysis
            ss_diff_info = get_ss_time_diff(duplicate_files["ss_1"], duplicate_files["ss_2"])
            logger.info(f"Successfully saved concatenated StateScript file to {ss_path}")
            
        except Exception as e:
            logger.error(f"Failed to concatenate StateScript files: {e}")
            raise
        
    if duplicate_files["timestamps_2"] is not None and timestamps_path is not None:
        logger.info(f"Concatenating timestamp files for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
        
        try:
            new_timestamps, ts_diff_info = concat_timestamps(
                duplicate_files["timestamps_1"], 
                duplicate_files["timestamps_2"]
            )
            
            timestamps_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(str(timestamps_path), new_timestamps)
            logger.info(f"Successfully saved concatenated timestamp file to {timestamps_path}")
            
        except Exception as e:
            logger.error(f"Failed to concatenate timestamp files: {e}")
            raise
    
    return dlc_diff_info, ss_diff_info, ts_diff_info

def create_diff_info_dict(dlc_diff_info: Optional[tuple], ss_diff_info: Optional[tuple], ts_diff_info: Optional[tuple]) -> Dict:
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
    
    diff_info = {"rat": settings.CURRENT_RAT, "day": settings.CURRENT_DAY,
                    "dlc_first_start": dlc_first_start, "dlc_first_end": dlc_first_end,
                    "dlc_second_start": dlc_second_start, "dlc_second_end": dlc_second_end, "dlc_diff": dlc_diff,
                    "ss_first_start": ss_first_start, "ss_first_end": ss_first_end,
                    "ss_second_start": ss_second_start, "ss_second_end": ss_second_end, "ss_diff": ss_diff,
                    "ts_first_start": ts_first_start, "ts_first_end": ts_first_end,
                    "ts_second_start": ts_second_start, "ts_second_end": ts_second_end, "ts_diff": ts_diff}
    return diff_info

def concat_duplicates(save_path: Union[str, Path]):
    """
    main method to call to concatenate all the duplicate files in save_path
    
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

    Parameters:
    - save_path: path that all the data can be found in
    """
    save_path = Path(save_path)
    diff_info = []
    
    for rat_folder_path in save_path.iterdir():
        if not rat_folder_path.is_dir():
            logger.info(f"Skipping over non-directory folder: {rat_folder_path}")
            continue

        rat_folder = rat_folder_path.name
        settings.update_rat(rat_folder)
        rat_path = rat_folder_path / remote.module

        for day_folder_path in rat_path.iterdir():
            if not day_folder_path.is_dir():
                logger.info(f"Skipping over non-directory folder: {day_folder_path}")
                continue

            day_folder = day_folder_path.name
            settings.update_day(day_folder)

            # find duplicate files
            if any(rat_folder in rat for rat in settings.IMPLANTED_RATS):
                duplicate_files = find_duplicates_implanted(day_folder_path) # naming convention is diff for implanted rats
            else:
                duplicate_files = find_duplicates(day_folder_path)
            
            if duplicate_files is None:
                logger.error(f"{settings.CURRENT_RAT} on {settings.CURRENT_DAY} ")
                continue  
            elif duplicate_files["dlc_2"] is None and \
                duplicate_files["ss_2"] is None and \
                duplicate_files["timestamps_2"] is None:
                continue # skip folders with no duplicates
            
            logger.info(f"{settings.CURRENT_RAT} on {settings.CURRENT_DAY} has a duplicate"
                         f"{duplicate_files}")
            
            # extra pathing for implanted rats
            track_folder = None
            track_folder_2 = None
            for item in day_folder_path.iterdir():
                # check if there is a track folder (for implanted rats)
                if item.is_dir():
                    directory = item.name
                    if "_track_2" not in directory:
                        track_folder = directory
                    elif "_track_2" in directory:
                        track_folder_2 = directory
                
            # new file names
            if track_folder is not None:
                # For rats with track folders, place concat files in the track folder
                track_folder_path = day_folder_path / track_folder
                
                dlc_file_name = make_concat_file_names(duplicate_files["dlc_1"])
                dlc_path = track_folder_path / dlc_file_name.name if dlc_file_name else None
                
                ss_file_name = make_concat_file_names(duplicate_files["ss_1"])
                ss_path = track_folder_path / ss_file_name.name if ss_file_name else None
                
                timestamps_file_name = make_concat_file_names(duplicate_files["timestamps_1"])
                timestamps_path = track_folder_path / timestamps_file_name.name if timestamps_file_name else None
            else:
                # For rats without track folders, place concat files directly in day folder
                dlc_file_name = make_concat_file_names(duplicate_files["dlc_1"])
                dlc_path = day_folder_path / dlc_file_name.name if dlc_file_name else None
                
                ss_file_name = make_concat_file_names(duplicate_files["ss_1"])
                ss_path = day_folder_path / ss_file_name.name if ss_file_name else None
                
                timestamps_file_name = make_concat_file_names(duplicate_files["timestamps_1"])
                timestamps_path = day_folder_path / timestamps_file_name.name if timestamps_file_name else None
            
            # concatenate everything
            try:
                if dlc_path is None:
                    raise error_types.UnexpectedNoneError("concat_duplicates", "dlc_path")
                elif ss_path is None:
                    raise error_types.UnexpectedNoneError("concat_duplicates", "ss_path")
                elif timestamps_path is None:
                    raise error_types.UnexpectedNoneError("concat_duplicates", "timestamps_path")
                
                dlc_diff_info, ss_diff_info, ts_diff_info = save_concats(duplicate_files, dlc_path, ss_path, timestamps_path)
            except Exception as e:
                logger.error(f"concatenation failed for {rat_folder} on {day_folder} because error {e}")
                continue
            
            # consolidate info about the concatenations
            try:
                current_diff_info = create_diff_info_dict(dlc_diff_info, ss_diff_info, ts_diff_info)
            except Exception as e:
                logger.critical(f"{settings.CURRENT_RAT} problem on {settings.CURRENT_DAY} with error {e}")
                raise error_types.ExpectationError("saving diff info", f"{e}")
            
            if current_diff_info is not None:
                diff_info.append(current_diff_info)
                
            # rename old files
            for file_type, file_path in duplicate_files.items():
                if (("dlc" in file_type and duplicate_files["dlc_2"] is None) or
                    ("ss" in file_type and duplicate_files["ss_2"] is None) or
                    ("timestamps" in file_type and duplicate_files["timestamps_2"] is None)):
                    continue # sometimes there are uneven amounts of duplicates across the three file types

                if file_path is not None:
                    extension = file_path.suffix
                    new_file_name = f"old_{file_type}{extension}"
                    
                    if track_folder_2 is not None:
                        change_folder_path = day_folder_path / track_folder_2 / new_file_name
                        file_path.rename(change_folder_path)
                    else:
                        new_file_path = file_path.parent / new_file_name
                        file_path.rename(new_file_path)
                
            if track_folder_2 is not None:
                track_folder_path = day_folder_path / track_folder_2
                new_track_folder_path = day_folder_path / "old_folder"
                track_folder_path.rename(new_track_folder_path)

    diff_info_df = pd.DataFrame(diff_info)
    diff_info_path = paths.base / "diff_info_2.csv"
    diff_info_df.to_csv(diff_info_path)



# ==============================================================================
# LOAD MAIN DATA STRUCTURE
# ==============================================================================
def _load_dlc_data_structure(save_path: Path) -> Dict:
    """
    loads dlc tracking data in data structure from saved location
    
    Parameters:
    - save_path: main path with rat folders with dlc data
    
    Returns:
    - Dict: data structure with dlc data loaded
    """
    
    data_structure = {}
    
    for rat_folder_path in save_path.iterdir():
        if not rat_folder_path.is_dir() or rat_folder_path.name.startswith(".") or ".DS" in rat_folder_path.name:
            logger.debug(f"Skipping non-directory path: {rat_folder_path}")
            continue

        rat = rat_folder_path.name
        settings.update_rat(rat)
        rat_path = rat_folder_path / remote.module

        if not rat_path.is_dir():
            logger.warning(f"No {remote.module} directory found for {rat}")
            continue

        logger.info(f"processing dlc data for rat {rat}")
        data_structure[rat] = {}
        
        for day_folder_path in rat_path.iterdir():
            if not day_folder_path.is_dir() or day_folder_path.name.startswith(".") or ".DS" in day_folder_path.name:
                logger.info(f"skipping non-directory path: {day_folder_path}")
                continue

            day = day_folder_path.name
            settings.update_day(day)
            logger.info(f"processing dlc data for {day}")
            
            data_structure[rat][day] = {
                settings.DLC: None,
                settings.SS: None,
                settings.TIMESTAMPS: None
            }
            
            # look for dlc file
            dlc_files = list(day_folder_path.rglob("*DLC*.csv"))
            dlc_files = [f for f in dlc_files if "old_" not in f.name.lower()]
            
            if not dlc_files:
                logger.warning(f"No dlc files found for {rat} on {day}")
                continue

            dlc_file = dlc_files[0]
            logger.info(f"loading dlc data from {dlc_file}")
            
            dlc_data = process_dlc_data(dlc_file)
            data_structure[rat][day][settings.DLC] = dlc_data
    
    return data_structure

def _load_timestamps_data_structure(data_structure: Dict, save_path: Path):
    """
    adds timestamps data to data structure
    
    Parameters:
    - data_structure: data structure with dlc data already loaded in
    - save_path: path to find timestamps
    """
    logger.info("loading timestamps data")
    
    for rat_folder_path in save_path.iterdir():
        if not rat_folder_path.is_dir() or rat_folder_path.name.startswith("."):
            logger.debug(f"Skipping non-directory path: {rat_folder_path}")
            continue

        rat = rat_folder_path.name
        settings.update_rat(rat)
        rat_path = rat_folder_path / remote.module

        if not rat_path.is_dir():
            logger.warning(f"No {remote.module} directory found for {rat}")
            continue

        logger.info(f"processing ts data for rat {rat}")
        for day_folder_path in rat_path.iterdir():
            if not day_folder_path.is_dir() or day_folder_path.name.startswith("."):
                logger.info(f"skipping non-directory path: {day_folder_path}")
                continue

            day = day_folder_path.name
            settings.update_day(day)
            logger.info(f"processing timestamps data for {day}")
            
            # look for ts file
            ts_files = list(day_folder_path.rglob("*.videoTimeStamps.npy"))
            ts_files = [f for f in ts_files if "old_" not in f.name.lower()]
            
            if not ts_files:
                logger.warning(f"No ts files found for {rat} on {day}")
                continue

            ts_file = ts_files[0]
            logger.info(f"loading ts data from {ts_file}")
            
            timestamps = np.load(ts_file)
            data_structure[rat][day][settings.TIMESTAMPS] = timestamps
            logger.info(f"successfully loaded timestamps for {rat} on {day}")

def _load_ss_data_structure(data_structure: Dict, save_path: Path):
    """
    adds statescript data to existing data structure
    
    Parameters:
    - data_structure: existing data structure with presumably dlc & ts data loaded
    - save_path: path where statescript files are stored
    """
    logger.info("loading ss data")
    
    for rat_folder_path in save_path.iterdir():
        if not rat_folder_path.is_dir() or rat_folder_path.name.startswith("."):
            logger.debug(f"Skipping non-directory path: {rat_folder_path}")
            continue

        rat = rat_folder_path.name
        settings.update_rat(rat)
        rat_path = rat_folder_path / remote.module

        if not rat_path.is_dir():
            logger.warning(f"No {remote.module} directory found for {rat}")
            continue

        logger.info(f"processing ss data for rat {rat}")
        for day_folder_path in rat_path.iterdir():
            if not day_folder_path.is_dir() or day_folder_path.name.startswith("."):
                logger.info(f"skipping non-directory path: {day_folder_path}")
                continue

            day = day_folder_path.name
            settings.update_day(day)
            logger.info(f"processing ss data for {day}")
            
            # look for ss file
            ss_files = list(day_folder_path.rglob("*.stateScriptLog.txt"))
            ss_files = [f for f in ss_files if "old_" not in f.name.lower()]
            
            if not ss_files:
                logger.warning(f"No ss files found for {rat} on {day}")
                continue

            ss_file = ss_files[0]
            logger.info(f"loading ss data from {ss_file}")
            
            ss_data = process_statescript_log(ss_file)
            data_structure[rat][day][settings.SS] = ss_data
            logger.info(f"successfully loaded ss for {rat} on {day}")

def _log_data_structure_summary(data_structure: Dict):
    """
    Logs a summary of what data was successfully loaded.
    
    This function provides useful feedback about the data loading process,
    helping users understand what data is available and identify any gaps.
    
    Parameters:
    - data_structure: The loaded data structure to summarize
    """
    
    total_rats = len(data_structure)
    total_days = sum(len(days) for days in data_structure.values())
    
    dlc_count = 0
    timestamps_count = 0
    statescript_count = 0
    
    missing_data = []
    
    for rat_name, days in data_structure.items():
        for day_name, data in days.items():
            if data["DLC_tracking"] is not None:
                dlc_count += 1
            else:
                missing_data.append(f"{rat_name}/{day_name}: missing DLC data")
                
            if data["videoTimeStamps"] is not None:
                timestamps_count += 1
            else:
                missing_data.append(f"{rat_name}/{day_name}: missing timestamps")
                
            if data["stateScriptLog"] is not None:
                statescript_count += 1
            else:
                missing_data.append(f"{rat_name}/{day_name}: missing statescript")
    
    logger.info("Data structure loading complete:")
    logger.info(f"  Total rats: {total_rats}")
    logger.info(f"  Total rat-day combinations: {total_days}")
    logger.info(f"  DLC data loaded: {dlc_count}/{total_days}")
    logger.info(f"  Timestamps loaded: {timestamps_count}/{total_days}")
    logger.info(f"  Statescripts loaded: {statescript_count}/{total_days}")

def load_data_structure(save_path: Optional[Union[str, Path]] = paths.vte_data) -> Dict:
    """
    loads the dictionary data structure created by create_main_data_structure from a directory it was saved in

    Parameters:
    - save_path: path to directory the data structure would be saved as a folder. if it doesn't exist yet, it'll be created
    
    Returns:
    - (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
    """
    logger.info(f"loading data structure from {str(save_path)}")
    if save_path is None:
        save_path = paths.vte_data
    
    save_path = Path(save_path)
    data_structure = {}
    
    try:
        data_structure = _load_dlc_data_structure(save_path)
        _load_timestamps_data_structure(data_structure, save_path)
        _load_ss_data_structure(data_structure, save_path)
        
        if data_structure is None:
            logger.critical("data structure is None")
            raise error_types.UnexpectedNoneError("load_data_structure", "data_structure")
        
        _log_data_structure_summary(data_structure)
    except KeyError as e:
        logger.error(f"key not found with {settings.CURRENT_RAT} on {settings.CURRENT_DAY}: {e}")
    except ValueError as e:
        logger.error(f"value error with {settings.CURRENT_RAT} on {settings.CURRENT_DAY}: {e}")
    except Exception as e:
        logger.error(f"unexpected error with {settings.CURRENT_RAT} on {settings.CURRENT_DAY}: {e}")
    else:
        logger.info("successfully loaded data structure")
    
    return data_structure


# ==============================================================================
# LOAD SPECIFIC FILES
# ==============================================================================

def load_specific_files(data_structure: Dict, rat: str, day: str) -> tuple[pd.DataFrame, str, np.ndarray, dict]:
    """load dlc, ss, ts & trial_starts for specific rat and day"""
    
    SS_log = data_structure[rat][day][settings.SS]
    timestamps = data_structure[rat][day][settings.TIMESTAMPS]
    trial_starts = time_utils.get_video_trial_starts(timestamps, SS_log)
    
    dlc_path = Path(paths.cleaned_dlc) / rat
    file_path = None
    
    # This assumes files are directly in the dlc_path directory
    for file_path_obj in dlc_path.iterdir():
        if file_path_obj.is_file():  # Only process files, not subdirectories
            file_name = file_path_obj.name
            parts = file_name.split("_")
            day_from_file = parts[0]
            
            # Check our conditions
            if day == day_from_file and "coordinates" in parts[1]:
                file_path = file_path_obj
                break
    
    if file_path is not None:
        df = pd.read_csv(file_path)
        return df, SS_log, timestamps, trial_starts
    else:
        logger.error(f"cannot find dlc file path for {rat} on {day}")
        raise error_types.UnexpectedNoneError("load_specific_files", "file_path")
