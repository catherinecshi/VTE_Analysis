"""
utility functions for getting time and speed values
uses statescript and dlc times
"""

import re
import bisect
import numpy as np
from typing import Union

from config import settings
from utilities import error_types

# ==============================================================================
# SPEED
# ==============================================================================

def get_speed(x, y, framerate):
    """gets change in position over time"""
    
    diff_x = x.diff().fillna(0)
    diff_y = y.diff().fillna(0)
    diff_per_frame = np.sqrt(diff_x**2 + diff_y**2)
    return diff_per_frame * framerate

def get_speed_session(data_structure, rat_ID, day):
    """gets the average speed of the rat for the entire session"""
    DLC_data = data_structure[rat_ID][day][settings.DLC]
    
    track_part = "haunch" # assumed to be the best for tracking animal speed
    x, y = data_structure.filter_dataframe(DLC_data, track_part)
    
    # get framerate
    framerate = settings.FRAMERATE
    
    speed = get_speed(x, y, framerate)
    
    return speed

# ==============================================================================
# STATESCRIPT TIME VALUES
# ==============================================================================

def statescript_to_seconds(statescript_time: Union[str, int]):
    """converts statescript time (first integer) to seconds (from start of recording)"""
    if isinstance(statescript_time, str):
        time_passed = int(statescript_time) / 1000
    else:
        time_passed = statescript_time / 1000 # turning it from ms to seconds
    return time_passed

def get_first_time(content: str):
    """returns first time in recording in seconds (from start of recording)"""
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line and stripped_line[0].isdigit():
            match = re.match(r'^(\d+)\s', stripped_line)
            if match:
                first_line = match.group(1)
                first_time = statescript_to_seconds(first_line)
                return first_time

def get_last_time(content: str):
    """returns last time in recording in seconds (from start of recording)"""
    last_line = None
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line and stripped_line[0].isdigit():
            match = re.match(r'^(\d+)\s', stripped_line)
            if match:
                last_line = match.group(1)
    
    if last_line is None:
        raise ValueError(f"{settings.CURRENT_RAT} {settings.CURRENT_DAY} no last time")
    else:
        last_time = statescript_to_seconds(last_line)
        return last_time

def get_trial_starts(content: str):
    """
    gets the time associated with each of the trial start times from a statescript log
    trial start times being when the time associated with "New Trial" appearing

    Args:
        content (str): statescript log

    Returns:
        dict: {trial_starts (str): trial_type (str)}
    """
    
    lines = content.splitlines()
    
    start_of_trial = False # know when the last line was the start of new trial
    trial_starts = []
    trial_info = {} # returned dict

    for line in lines:
        if line.startswith('#') or line == "": # skip the starting comments & empty comments
            continue
        
        elif not line[0].isdigit(): # check if the first char is a number - skip if not
            # hopefully this takes cares of the weird errors wth pressng summary after new trial showed
            continue
        
        elif start_of_trial and "trialType" in line: # store trial type
            parts = line.split()
            trial_type = parts[3]
            trial_info[trial_starts[-1]] = trial_type # assumes this will always come after New Trial'
            start_of_trial = False
                  
        elif 'New Trial' in line: # indicate start of a new trial
            start_of_trial = True
            
            # store the time during this event
            parts = line.split()
            trial_start = parts[0]
            trial_info[trial_start] = None
            trial_starts.append(trial_start)
    
    return trial_info

def get_time_until_choice(content: str):
    """calculates the time the rat takes until its first choice in the session"""
    lines = content.splitlines()
    
    new_trial = False # to check if a trial is in session
    time = []
    arm = set()
    lick_count = 0
    not_first = False
    time_home = None
    time_arm = None
    
    for line in lines:
        if '#' in line: # skip starting comments
            continue
        elif 'New Trial' in line:
            new_trial = True
        elif all(char.isdigit() or char.isspace() for char in line) and new_trial: # a normal licking line
            parts = line.split()
            current_arm = int(parts[1])
            current_time = int(parts[0])
            
            # check when rat left home
            if current_arm > 1 and not not_first: # licked at arm not at home
                time_home = time[-1] # haven't appended to time yet so this is last line's time
                time_arm = current_time
            
            # check for how many licks
            if current_arm > 1:
                if current_arm in arm: # not a new arm
                    lick_count += 1
                else:
                    lick_count = 1 # new arm so reset lick count
            
            # check if the rat changed its choice without licking enough at one arm
            if current_arm > 1 and len(arm) > 3:
                time_arm = current_time
            
            time.append(current_time)
            arm.add(current_arm)

        if lick_count == 3: # enough licks for a choice
            break

    # calculating the difference between the time
    if time_arm is None:
        raise error_types.UnexpectedNoneError("get_time_until_choice", "time_arm")
    elif time_home is None:
        raise error_types.UnexpectedNoneError("get_time_until_choice", "time_home")
    else:
        time_arm_seconds = statescript_to_seconds(time_arm)
        time_home_seconds = statescript_to_seconds(time_home)
        time_diff = time_arm_seconds - time_home_seconds
        
        return time_diff

def get_time_diff(content_1: str, content_2: str):
    """
    returns info about time diff between two statescript logs
    all times are seconds since start of recording
    (though "start of recording" of second file could be the start of second or first)
    
    Returns:
        (first_start, first_end, second_start, second_end, diff)
        diff = second_start - first_end
    """
    first_start = get_first_time(content_1)
    first_end = get_last_time(content_1)
    second_start = get_first_time(content_2)
    second_end = get_last_time(content_2)
    
    if first_start is None or first_end is None or second_start is None or second_end is None:
        raise error_types.UnexpectedNoneError("get_time_diff", (first_start, first_end, second_start, second_end))
    
    diff = second_start - first_end
    ss_diff_info = (first_start, first_end, second_start, second_end, diff)
    
    return ss_diff_info

# ==============================================================================
# TIMESTAMPS TIME VALUES
# ==============================================================================

def get_video_trial_starts(timestamps: np.ndarray, SS_log: str) -> dict: # gets indices for x/y where trials start & corresponding trial type
    """
    gets the trial start times according to the corresponding index for dlc dataframe

    Args:
        timestamps: the times for each dlc frame
        SS_log: the statescript log

    Returns:
        dict: {trial type: trial starts} where trial types are the numbers corresponding to the trial type
    """
    
    trial_info = get_trial_starts(SS_log) # where trial_info is {trial_starts: trial_type}
    trial_starts = list(trial_info.keys())
    
    video_starts = ss_trial_starts_to_video(timestamps, trial_starts) # this should be the indices for x/y where trials start
    
    # change trial_info such that the key is video_starts instead of trial_starts
    video_trial_info = {}
    
    if len(video_starts) == len(trial_starts):
        for index, video_start in enumerate(video_starts):
            if not isinstance(video_start, (float, int, np.number)):
                video_start = video_start[0]
            
            original_start_time = trial_starts[index]
            trial_type = trial_info.get(original_start_time) # get trial type for trial start time
            video_trial_info[video_start] = trial_type
    
    return video_trial_info

# ==============================================================================
# CONVERSIONS
# ==============================================================================

def ss_trial_starts_to_video(timestamps: np.ndarray, SS_times: list[str]) -> list[float]:
    """
    converts statescript trial starts to dlc video trial starts

    Args:
        timestamps: the timestamps associated with each dlc frame
        SS_times: the list of times from statescript of when trials start

    Returns:
        list: the indices corresponding to where in the filtered dataframe will have trial starts
        
    Procedure:
        1. Loop through each trial start time in SS_times
            - trial start times being when "New Trial" appears
        2. Check if the trial start time matches with a number in timestamps
            - doesn't always happen because mismatch between ECU and MCU time
            - if there is a match, add that time to trial_starts
        3. If there isn't a perfect match
            - check for the closest number, then add that to trial_starts
            - skip 0
    """
    
    trial_starts = []
    
    for time in SS_times:
        # ensure consistent data types
        time = float(int(time) / 1000)
        
        if time in timestamps: # if there is a perfect match between MCU & ECU and the time is in timestamps
            index, = np.where(timestamps == time)
            trial_starts.append(index)
        else: # if there isn't a perfect match
            # index where time is inserted into timestamps
            idx = bisect.bisect_left(timestamps, time)
            if idx < len(timestamps):
                trial_starts.append(timestamps[idx])
    
    return trial_starts # with this, each trial_start is the index of the time when trial starts in relation to timestamps