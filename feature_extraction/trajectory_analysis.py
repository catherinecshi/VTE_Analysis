"""
calculates the VTE zIdPhi values for one day for one rat
main function to call is quantify_VTE()
"""

import os
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from scipy.spatial.qhull import ConvexHull

from config import settings
from config.paths import paths
from utilities import math_utils
from utilities import spatial_utils
from utilities import logging_utils
from utilities import conversion_utils
from debugging import error_types
from utilities import file_manipulation
from preprocessing import data_processing
from preprocessing import process_statescript
from visualization import trajectory_plots

# pylint: disable=broad-exception-caught, invalid-name, logging-fstring-interpolation, global-statement
logger = logging_utils.get_module_logger("trajectory_analysis")

# ==============================================================================
# CREATE TRAJECTORY
# ==============================================================================

def find_starting_index(df: pd.DataFrame, start_time: float) -> tuple[int, float, float, float]:
    """
    find starting point of trajectory
    
    Parameters:
    - df: DataFrame with trajectory data
    - start_time: Time to start looking for trajectory
    
    Returns:
    - tuple: (index, x_value, y_value, actual_time)
    
    Raises:
    - ValueError: if no suitable starting point is found
    """
    
    corresponding_row = df[df["times"] == start_time]
    
    if not corresponding_row.empty: # exact match
        index = corresponding_row.index[0]
        return (index,
                corresponding_row["x"].values[0],
                corresponding_row["y"].values[0],
                corresponding_row["times"].values[0])
    
    # no exact match - find closest neighbor
    index = bisect.bisect_right(df["times"].values, start_time)
    if index >= len(df):
        raise ValueError(f"Cannot find starting point after time {start_time} for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
    
    row = df.iloc[index]
    return index, row["x"], row["y"], row["times"]

def extract_trajectory_points(
        df: pd.DataFrame, 
        start_index: int, 
        end_time: float, 
        hull: ConvexHull
    ) -> tuple[list, list, Optional[float], Optional[float]]:
    """
    extract continuous set of points within convex hull
    
    Parameters:
    - df: DataFrame with trajectory data
    - start_index: Index to start extracting from
    - end_time: start of next trajectory
    - hull: ConvexHull for centre zone
    
    Returns:
    - tuple: (x_points, y_points, start_time, end_time)
    """
    
    trajectory_x = []
    trajectory_y = []
    trajectory_start_time = None
    trajectory_end_time = None
    past_inside = False

    current_index = start_index
    count = 0
    
    while current_index < len(df) and count < 5000:
        row = df.iloc[current_index]
        current_time = row["times"]
        
        if current_time >= end_time:
            break

        x_val = row["x"] if not isinstance(row["x"], list) else row["x"][0]
        y_val = row["y"] if not isinstance(row["y"], list) else row["y"][1]
        
        point = (x_val, y_val)
        inside = spatial_utils.is_point_in_hull(point, hull)
        
        if inside:
            if not past_inside:
                past_inside = True
                trajectory_start_time = current_time
            
            trajectory_x.append(x_val)
            trajectory_y.append(y_val)
        elif past_inside:
            # we've exited the hull, so trajectory is complete
            trajectory_end_time = current_time
            break

        current_index += 1
        count += 1
    
    return trajectory_x, trajectory_y, trajectory_start_time, trajectory_end_time

def calculate_trajectory_duration(start_time: Optional[float], end_time: Optional[float]) -> Optional[float]:
    """
    calculates the time in seconds of a trajectory
    
    Parameters:
    - start_time: When trajectory started
    - end_time: When trajectory ended
    
    Returns:
    - Duration in seconds, or None
    """
    if start_time is not None and end_time is not None:
        return end_time - start_time
    return None

def get_trajectory(
        df: pd.DataFrame, 
        start: float, 
        end: float, 
        hull: ConvexHull, 
        repeats: int = 0,
        min_length: int = 8
    ) -> tuple[Optional[list], Optional[list], Optional[float], int]:
    """
    gets all the x and y points within a trajectory given the start point and hull within which the trajectory is
    If trajectory is too short, tries to find a new trajectory starting from the end of the previous one
    
    Parameters:
    - df: contains x and y coordinates and times for each coordinate point
    - start: time corresponding to start of trajectory search
    - end: time corresponding to start of the next trajectory
    - hull: hull within which trajectory is
    - repeats: number of repeats of trajectory already
    - min_length: minimum number of points required for a valid trajectory
    
    Returns:
    - (float array): all x points for trajectory
    - (float array): all y points for trajectory
    - float: the amount of time in seconds spent on the trajectory
    - int: number of repeats of trajectory already
    """
    try:
        current_start_time = start
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            # find starting point
            start_index, _, _, _ = find_starting_index(df, current_start_time)
            
            trajectory_x, trajectory_y, start_time, end_time = extract_trajectory_points(df, start_index, end, hull)
            
            duration = calculate_trajectory_duration(start_time, end_time)
            
            # Check if trajectory exists
            if not trajectory_x or not trajectory_y:
                logger.error(f"no trajectory found for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} for {settings.CURRENT_TRIAL}")
                return None, None, None, repeats + 1
            
            # Check if trajectory meets minimum length requirement
            if len(trajectory_x) >= min_length:
                # Trajectory is long enough, return it
                return trajectory_x, trajectory_y, duration, repeats
            
            # Trajectory is too short, try to find a new starting point
            retry_count += 1
            if retry_count <= max_retries and end_time is not None:
                # Start next attempt from where this trajectory ended + 1
                current_start_time = end_time + 1
                
                # Make sure we haven't gone past the end time
                if current_start_time >= end:
                    logger.warning(f"reached end of time window while retrying trajectory for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} for {settings.CURRENT_TRIAL}")
                    break
                
                logger.debug(f"trajectory too short ({len(trajectory_x)} points), retrying from time {current_start_time} (attempt {retry_count}/{max_retries})")
            else:
                logger.error(f"no end time for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} ??? {settings.CURRENT_TRIAL}")
                break
        
        # If we've exhausted retries, return the last trajectory even if it's short
        logger.warning(f"returning short trajectory ({len(trajectory_x)} points) after {retry_count} retries for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} for {settings.CURRENT_TRIAL}")
        return trajectory_x, trajectory_y, duration, repeats
        
    except ValueError as e:
        logger.error(f"value error {e} finding trajectory for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} for {settings.CURRENT_TRIAL}")
        return None, None, None, repeats + 1
    except Exception as e:
        logger.error(f"unexpected error {e} finding trajectory for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} for {settings.CURRENT_TRIAL}")
        return None, None, None, repeats + 1

# ==============================================================================
# DATA VALIDATION
# ==============================================================================

def check_timestamps(timestamps: np.ndarray):
    """check that timestamps are proceeding how time usually does"""
    
    not_ascending_count = 0
    stagnant_count = 0
    for i, timestamp in enumerate(timestamps):
        if i > 0 and i + 1 < len(timestamps):
            if timestamp > timestamps[i + 1]:
                not_ascending_count += 1
                logger.error(f"timestamps not ascending for {settings.CURRENT_TRIAL} for {timestamp}")
            elif i > 0 and timestamp == timestamps[i + 1]:
                stagnant_count += 1
                logger.warning(f"stagnant at {timestamp} {timestamps[i + 1]}")
            else:
                stagnant_count = 0
        else:
            continue
        
        if not_ascending_count > 5 or stagnant_count > 150:
            raise error_types.CorruptionError(timestamps, "check_timestamps")
        
def check_trial_data(performance: list, trial_starts: dict) -> bool:
    """makes sure dlc and timestamps are consistent"""
    return math_utils.check_equal_length(performance, list(trial_starts.keys()))

def trajectory_present(dlc: pd.DataFrame, start: float, end: float) -> bool:
    trajectory_df = dlc[(dlc["times"] >= start) & (dlc["times"] <= end)]
    if trajectory_df.empty:
        # sometimes there are gaps in dlc, so skip these trajectories
        return False
    else:
        return True

def should_exclude_trajectory(trajectory_x: list, trajectory_y: list, traj_len: float,
                              last_trajectory_x: Optional[list]) -> tuple[bool, str]:
    """return True if trajectory should be skipped"""
    
    # not enough points
    if len(trajectory_x) < 5:
        return True, "<5 Points"
    
    # if too long or too short
    try:
        if traj_len is not None:
            if traj_len < 0.2:
                return True, "Too Short"
            
            if traj_len > 4:
                return False, "Too Long"
    except TypeError:
        # can't determine length??
        return True, "Length unknown"
    
    # see if it's a duplicate
    if last_trajectory_x is not None and last_trajectory_x == trajectory_x:
        return True, "Repeat"
    
    # if the rat is just staying in one place
    staying_x = math_utils.check_difference(trajectory_x, threshold=10)
    staying_y = math_utils.check_difference(trajectory_y, threshold=10)
    
    if not staying_x or not staying_y:
        print(staying_x, staying_y)
        return False, "Staying"
    
    return False, ""

# ==============================================================================
# SETUP DATA
# ==============================================================================

def get_centre_hull() -> ConvexHull:
    """
    gets centre convex hull that was created from zone_creation.py
    assumes that current rat and day has been continually getting updated in settings
    
    Returns:
    - scipy.spatial.ConvexHull: convex hull corresponding to the centre zone
    """
    # path to load from
    #hull_path = paths.hull_data / "inferenceTesting" / f"{settings.CURRENT_RAT}_hull_test.npy"
    hull_path = paths.hull_data / f"{settings.CURRENT_RAT}_{settings.CURRENT_DAY}_hull.npy"
    
    # load hull
    densest_cluster_points = np.load(hull_path)
    hull = ConvexHull(densest_cluster_points)
    
    return hull

def setup_analysis_environment(data_structure: dict, rat: str, day: str) -> tuple:
    """load in the necessary data and setup settings"""
    
    settings.update_rat(rat)
    settings.update_day(day)
    logger.info(f"creating IdPhi values for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
    
    # load and validate data
    dlc, ss, ts, trial_starts = data_processing.load_specific_files(data_structure, rat, day)
    check_timestamps(ts)
    centre_hull = get_centre_hull()
    
    # setup file paths
    excluded_path = paths.processed / "excluded_trajectories" / f"{rat}_excluded_trajectories.csv"
    
    return dlc, ss, ts, trial_starts, excluded_path, centre_hull

# ==============================================================================
# PROCESS DATA
# ==============================================================================

def process_single_trial(trial_info: dict, dlc: pd.DataFrame, centre_hull: ConvexHull,
                         performance: list) -> tuple[Optional[dict], Optional[int]]:
    """
    process a single trial and trajectory
    
    Parameters:
    - trial_info: includes start, end, index (of trial), trial type, and id of trial
    - dlc: processed dlc data
    - centre_hull: convex hull of centre zone
    - performance: which trials the rat got correct
    
    Returns:
    - dict: (trajectory_x, trajectory_y, traj_len, choice, trial_type, traj_id, performance)
    - int: repeats/errors
    """
    trial_start = trial_info["start_time"]
    trial_end = trial_info["end_time"]
    trial_index = trial_info["index"]
    trial_type = trial_info["type"]
    traj_id = trial_info["id"]
    
    settings.update_trial(traj_id)
    
    # check if trajectory is present to save time
    traj_present = trajectory_present(dlc, trial_start, trial_end)
    if not traj_present:
        return None, None
    
    trajectory_x, trajectory_y, traj_len, repeats = get_trajectory(
        dlc, trial_start, trial_end, centre_hull, trial_info.get("repeats", 0)
    )
    
    if not trajectory_x or not trajectory_y:
        return None, repeats
    
    # determine what the rat chose
    try:
        choice = conversion_utils.type_to_choice(trial_type, performance[trial_index])
    except IndexError:
        return None, repeats
    
    return {
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
        "traj_len": traj_len,
        "choice": choice,
        "trial_type": trial_type,
        "traj_id": traj_id,
        "is_correct": performance[trial_index]
    }, repeats

def save_trajectory_results(store_data: list, save_path: str, repeats: int) -> bool:
    """validates results and saves them if cool"""
    # see if there were too many processing errors
    if repeats > 10:
        logger.error(f"more than 10 repeats for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
        return False
    
    # save
    df = pd.DataFrame(store_data)
    file_path = os.path.join(save_path, f"{settings.CURRENT_RAT}_{settings.CURRENT_DAY}_trajectories.csv")
    df.to_csv(file_path)
    
    return True

def quantify_VTE(data_structure, rat, day, save = None):
    """
    gets relevant VTE values for one rat for a specific day

    Args:
        data_structure (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
        ratID (str): current rat being processed
        day (str): current day being processed
        save (str, optional): filepath if saving is desired. Defaults to None.

    Raises:
        helper_functions.LengthMismatchError: if the number of trials are different in statescript vs dlc

    Returns:
        zIdPhi_values (dict): {choice: zIdPhi}
        IdPhi_values (dict): {choice: IdPhi}
        trajectories (dict): {choice: (trajectory_x, trajectory_y)}
        
    Procedure:
        1. get necessary components like coords, trial start times, and performance for session
            - check if trial_starts and performance have the same number of trials
        2. cut out the trajectory from the coordinates
            - do so by using the start of the trial, and including the coordinates that are part of the first consecutive string in centre zone
        3. calculate the IdPhi value of that trajectory
        4. determine where the rat ended up going by looking at trial type and whether it got it correct
            - sort all values according to where the rat ended up going
        5. calculate zIdPhi values by zscoring across choice arm
    """
    dlc, ss, ts, trial_starts, excluded_path, centre_hull = setup_analysis_environment(data_structure, rat, day)
    
    _, _, performance = process_statescript.get_session_performance(ss) # a list of whether the trials resulted in a correct or incorrect choice
    data_is_consistent = check_trial_data(performance, trial_starts) # check if there are the same number of trials for perf and trial_starts
    
    if not data_is_consistent:
        logger.error(f"different number of trials in performance and trial_starts for {rat} on {day}")
    
    # store IdPhi and trajectory values
    trajectories = {}
    store_data = []
    repeats = 0
    last_trajectory_x = None
    
    trial_start_keys = list(trial_starts.keys())
    for i, trial_start in enumerate(trial_start_keys): # where trial type is a string of a number corresponding to trial type
        trial_info = {
            "start_time": trial_start,
            "end_time": trial_start_keys[i + 1] if i + 1 < len(trial_start_keys) else ts[-1],
            "index": i,
            "type": trial_starts[trial_start],
            "id": f"{rat}_{day}_{i + 1}",
            "repeats": repeats
        }
        
        trial_results, new_repeats = process_single_trial(
            trial_info, dlc, centre_hull, performance
        )
        if new_repeats is not None:
            repeats = new_repeats
        
        # skip if processing failed
        if trial_results is None:
            logger.error(f"unexpected none for trial_results for {rat} on {day} for time {trial_start}")
            continue
        
        # check if trajectory should be excluded
        should_exclude, exclusion_reason = should_exclude_trajectory(
            trial_results["trajectory_x"],
            trial_results["trajectory_y"],
            trial_results["traj_len"],
            last_trajectory_x
        )
        
        if should_exclude:
            skip_row = {
                "ID": trial_results["traj_id"],
                "X Values": trial_results["trajectory_x"],
                "Y Values": trial_results["trajectory_y"],
                "Correct": trial_results["is_correct"],
                "Choice": trial_results["choice"],
                "Trial Type": trial_results["trial_type"],
                "Length": trial_results["traj_len"],
                "Reason": exclusion_reason
            }
            file_manipulation.add_row_to_csv(excluded_path, skip_row)
            continue
        elif exclusion_reason != "": # not an error, record down but don't not analyse the trajectory
            skip_row = {
                "ID": trial_results["traj_id"],
                "X Values": trial_results["trajectory_x"],
                "Y Values": trial_results["trajectory_y"],
                "Correct": trial_results["is_correct"],
                "Choice": trial_results["choice"],
                "Trial Type": trial_results["trial_type"],
                "Length": trial_results["traj_len"],
                "Reason": exclusion_reason
            }
            file_manipulation.add_row_to_csv(excluded_path, skip_row)
        
        last_trajectory_x = trial_results["trajectory_x"] # update
        
        # convert to cm and create idphi values
        trajectory_x, trajectory_y = conversion_utils.convert_pixels_to_cm(trial_results["trajectory_x"], trial_results["trajectory_y"])
        IdPhi = math_utils.calculate_IdPhi(trajectory_x, trajectory_y)
        #plotting.plot_trajectory_animation((DLC_df["x"] / 5), (DLC_df["y"] / 5), trajectory_x, trajectory_y, title=traj_id)
        
        # store each trajectory for plotting later
        choice = trial_results["choice"]
        if choice in trajectories:
            trajectories[choice].append((trajectory_x, trajectory_y))
        else:
            trajectories[choice] = [(trajectory_x, trajectory_y)]
        
        new_row = {"ID": trial_results["traj_id"],
                   "X Values": trajectory_x,
                   "Y Values": trajectory_y,
                   "Correct": trial_results["is_correct"],
                   "Choice": choice,
                   "Trial Type": trial_results["trial_type"],
                   "IdPhi": IdPhi,
                   "Length": trial_results["traj_len"]}
        store_data.append(new_row)
        
        # plot and save if desired
        if save is not None:
            trajectory = (trajectory_x, trajectory_y)
            trajectory_plots.plot_trajectory(dlc["x"], dlc["y"], trajectory, title=trial_results["traj_id"], save=save, traj_id=trial_results["traj_id"])
    
    # check if there are too many repeats
    if save is not None:
        save_trajectory_results(store_data, save, repeats)
    
    plt.close()
    return trajectories