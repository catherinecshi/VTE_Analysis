"""
general auxillary functions for multiple purposes:
    - custom errors
        - LengthMismatchError
        - ExpectationError
    - get values
        - get_time
        - get_framerate
        - get_speed
        - get_speed_session
        - get_time_until_choice
        - get_ss_trial_starts
        - get_video_trial_starts
        - get_trajectory
    - conversions
        - round_to_sig_figs
        - ss_trials_to_video
        - string_to_int_trial_type
    - if point is in zone
        - is_point_in_ellipse
        - is_point_in_hull
    - checking things are normal
        - check_timestamps
        - check_equal_length
        - start_check
        - trial_type_equivalency
        - check_difference
    - file manipulation
        - add_row_to_csv
    - startup
        - initial_processing
"""

import os
import re
import csv
import math
import bisect
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
from scipy.spatial.qhull import Delaunay

### LOGGING
logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# makes a new log everytime the code runs by checking the time
log_file = datetime.now().strftime("/Users/catpillow/Documents/VTE_Analysis/doc/helper_log_%Y%m%d_%H%M%S.txt")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

# pylint: disable=global-statement, logging-fstring-interpolation

### Useful globals
BASE_PATH = "/Users/catpillow/Documents/VTE_Analysis"
CURRENT_RAT = ""
CURRENT_DAY = ""
IMPLANTED_RATS = ["TH508", "TH510", "TH605"]

### UPDATE GLOBALS ----------
def update_rat(rat):
    """use when iterating over rats or days, and debug messages need to know rat & days"""
    global CURRENT_RAT
    CURRENT_RAT = rat

def update_day(day):
    """use when iterating over rats or days, and debug messages need to know rat & days"""
    global CURRENT_DAY
    CURRENT_DAY = day



### ERRORS ------------
class LengthMismatchError(Exception):
    """Exception raised for errors where two things should be the same length but are not"""
    
    def __init__(self, first_length, second_length):
        self.first_length = first_length
        self.second_length = second_length
        self.message = f"Mismatch of the two lengths. {first_length} vs {second_length}"
        
        super().__init__(self.message)

class ExpectationError(Exception):
    """Exception raised for errors where something was expected but not there. usually at the end of if else or match case statements"""
    
    def __init__(self, expected, given):
        self.expected = expected
        self.given = given
        self.message = f"Expected {expected} but got {given}"
        
        super().__init__(self.message)

class CorruptionError(Exception):
    def __init__(self, ratID, day, file):
        self.ratID = ratID
        self.day = day
        self.file = file
        self.message = f"error with {ratID} on {day}, for file {file}"
        
        super().__init__(self.message)


### GET VALUE --------------
def get_time(statescript_time):
    # converts statescript time to seconds since start of recording
    if isinstance(statescript_time, str):
        time_passed = int(statescript_time) / 1000
    else:
        time_passed = statescript_time / 1000 # turning it from ms to seconds
    return time_passed

def get_framerate():
    # returns framerate, should usually be the same, or check_timestamps will catch it
    framerate = 0.03 # seconds
    
    return framerate

def get_speed(x, y, framerate):
    diff_x = x.diff().fillna(0)
    diff_y = y.diff().fillna(0)
    
    displacement_per_frame = np.sqrt(diff_x**2 + diff_y**2)
    
    speed = displacement_per_frame * framerate
    
    return speed

def get_speed_session(data_structure, ratID, day):
    # gets the average speed of the rat for the entire session
    DLC_data = data_structure[ratID][day]['DLC_tracking']
    #SS_data = data_structure[ratID][day]['stateScriptLog']
    
    track_part = 'haunch' # assumed to be the best for tracking animal speed
    x, y = data_structure.filter_dataframe(DLC_data, track_part)
    
    # get framerate
    framerate = get_framerate()
    
    # calculate speed
    diff_x = x.diff().fillna(0)
    diff_y = y.diff().fillna(0)
    
    displacement_per_frame = np.sqrt(diff_x**2 + diff_y**2)
    
    speed = displacement_per_frame * framerate
    
    return speed

def get_first_time(content):
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line and stripped_line[0].isdigit():
            match = re.match(r'^(\d+)\s', stripped_line)
            if match:
                first_line = match.group(1)
                first_time = get_time(first_line)
                return first_time

def get_last_time(content):
    last_line = None
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line and stripped_line[0].isdigit():
            match = re.match(r'^(\d+)\s', stripped_line)
            if match:
                last_line = match.group(1)
    last_time = get_time(last_line)
    return last_time

def get_time_until_choice(data_structure, ratID, day):
    content = data_structure[ratID][day]['stateScriptLog']
    
    # calculates the time the rat takes until its first choice in the session
    lines = content.splitlines()
    
    # some variables
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
    time_arm_seconds = get_time(time_arm)
    time_home_seconds = get_time(time_home)
    time_diff = time_arm_seconds - time_home_seconds
    
    return time_diff

def get_ss_trial_starts(SS):
    """
    gets the time associated with each of the trial start times from a statescript log
    trial start times being when the time associated with "New Trial" appearing

    Args:
        SS_df (str): statescript log

    Returns:
        dict: {trial_starts: trial_type}
    """
    
    lines = SS.splitlines()
    
    # storage variables
    start_of_trial = False # know when the last line was the start of new trial
    trial_starts = []
    trial_info = {} # store trial start times and trial types
    
    # get the trial start times from SS
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

def get_video_trial_starts(timestamps, SS_df): # gets indices for x/y where trials start & corresponding trial type
    """
    gets the trial start times according to the corresponding index for dlc dataframe

    Args:
        timestamps (np int array): the times for each dlc frame
        SS_df (str): the statescript log

    Returns:
        dict: {trial type: trial starts} where trial types are the numbers corresponding to the trial type
    """
    
    trial_info = get_ss_trial_starts(SS_df) # where trial_info is {trial_starts: trial_type}
    trial_starts = list(trial_info.keys())
    
    video_starts = ss_trial_starts_to_video(timestamps, trial_starts) # this should be the indices for x/y where trials start
    
    # change trial_info such that the key is video_starts instead of trial_starts
    video_trial_info = {}
    
    if len(video_starts) == len(trial_starts):
        for index, video_start in enumerate(video_starts):
            if not isinstance(video_start, (float, int, np.int64, np.float64)):
                video_start = video_start[0]
            
            original_start_time = trial_starts[index]
            trial_type = trial_info.get(original_start_time) # get trial type for trial start time
            video_trial_info[video_start] = trial_type
    
    return video_trial_info



### CONVERSIONS ----------
def round_to_sig_figs(num, sig_figs = 3):
    """
    round a number to a specific number of significant figures

    Args:
        num (float): number to be rounded
        sig_figs (int): the number of significant figures desired. Defaults to 3

    Returns:
        float: the rounded number
    """
    
    if num == 0:
        return 0
    else:
        return round(num, sig_figs - int(math.floor(math.log10(abs(num)))) - 1)

def ss_trial_starts_to_video(timestamps, SS_times):
    """
    converts statescript trial starts to dlc video trial starts

    Args:
        timestamps (np int array): the timestamps associated with each dlc frame
        SS_times (str): the list of times from statescript of when trials start

    Returns:
        int array: the indices corresponding to where in the filtered dataframe will have trial starts
        
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

def string_to_int_trial_types(string_trial):
    if string_trial == 'AB':
        return 1
    elif string_trial == 'BC':
        return 2
    elif string_trial == 'CD':
        return 3
    elif string_trial == 'DE':
        return 4
    elif string_trial == 'EF':
        return 5
    elif string_trial == 'BD':
        return 6
    elif string_trial == 'CE':
        return 7
    elif string_trial == 'BE':
        return 8
    elif string_trial == 'AC':
        return 9
    elif string_trial == 'DF':
        return 10
    else:
        logging.warning(f"no string trial - {string_trial}")
        return None

def choice_to_correctness(trial_type, choice):
    """takes trial type and choice and returns whether it was a correct trial"""
    if trial_type == 1:
        if choice == "A":
            correct = True
        else:
            correct = False
    elif trial_type == 2:
        if choice == "B":
            correct = True
        else:
            correct = False
    elif trial_type == 3:
        if choice == "C":
            correct = True
        else:
            correct = False
    elif trial_type == 4:
        if choice == "D":
            correct = True
        else:
            correct = False
    elif trial_type == 5:
        if choice == "E":
            correct = True
        else:
            correct = False
    
    return correct


### IF POINT IN SPACE --------------
def is_point_in_ellipse(x, y, ellipse_params):
    center, width, height, angle = ellipse_params['center'], ellipse_params['width'], ellipse_params['height'], ellipse_params['angle']
    
    # Convert angle from degrees to radians for np.cos and np.sin
    theta = np.radians(angle)
    
    # Translate point to origin based on ellipse center
    x_translated = x - center[0]
    y_translated = y - center[1]
    
    # Rotate point by -theta to align with ellipse axes
    x_rotated = x_translated * np.cos(-theta) - y_translated * np.sin(-theta)
    y_rotated = x_translated * np.sin(-theta) + y_translated * np.cos(-theta)
    
    # Check if rotated point is inside the ellipse
    if (x_rotated**2 / (width/2)**2) + (y_rotated**2 / (height/2)**2) <= 1:
        return True  # Inside the ellipse
    else:
        return False  # Outside the ellipse

def is_point_in_hull(point, hull):
    # delaunay triangulation of hull points
    del_tri = Delaunay(hull.points[hull.vertices])
    
    # check if point is inside the hull
    return del_tri.find_simplex(point) >= 0



### CHECKING THINGS ARE NORMAL ------------
def check_timestamps(df, timestamps):
    # first check - makes sure there is around 0.03s between each frame
    time_off = np.zeros(len(timestamps)) # records indices where time isn't ~0.03s between frames
    index_off = 0
    
    for index, _ in enumerate(timestamps): #loops through timestamps
        if index == 0:
            continue
    
        # calculate diff in seconds between frames
        current_time = timestamps[index]
        past_time = timestamps[index - 1]
        time_diff = current_time - past_time
        
        # make sure it's around the 0.03 range
        if time_diff > 0.05 or time_diff < 0.01:
            time_off[index] = time_diff # time is off here
            
            if index_off < 5:
                index_off += 1
        else:
            continue
    
    # second check - make sure x and timestamps are the same length
    if not(len(df) == len(timestamps)):
        diff = len(df) - len(timestamps)
        
        # it seems like most of them differ by 1, where df = timestamps - 1, so i'm doing a rough subtraction here
        if diff == -1:
            timestamps = timestamps[:-1].copy()
    
    return timestamps

def check_equal_length(a, b):
    """
    checks if two arrays/dictionaries are the same length (same number of elements or key-value pairs)

    Args:
        a (array or dict): first thing being compared
        b (array or dict): second thing being compared
        
    Returns:
        (bool): true if the two things are the same length
    """
    
    #same_len = len(a) == len(b)
    
    # have some lenience if it's just like one off
    same_len = -2 < (len(a) - len(b)) < 2

    return same_len
    
def trial_type_equivalency(trial_type_i, trial_type_j):
    string_trial = None
    int_trial = None

    if isinstance(trial_type_i, str):
        string_trial = trial_type_i
    elif isinstance(trial_type_i, int):
        int_trial = trial_type_i
    else:
        logging.warning(f'trial type error with {trial_type_i}')
        return None
    
    if isinstance(trial_type_j, str) and string_trial is not None:
        string_trial = trial_type_j
    elif isinstance(trial_type_j, str):
        logging.info(f'two string trials - {trial_type_i}, {trial_type_j}')
        return trial_type_i is trial_type_j
    elif isinstance(trial_type_j, int) and int_trial is not None:
        int_trial = trial_type_j
    elif isinstance(trial_type_j, int):
        logging.info(f'two int trial types - {trial_type_i}, {trial_type_j}')
        return trial_type_i == trial_type_j
    else:
        logging.warning(f'trial type error with {trial_type_j}')
    
    if string_trial == 'AB' and int_trial == 1:
        return True
    elif string_trial == 'BC' and int_trial == 2:
        return True
    elif string_trial == 'CD' and int_trial == 3:
        return True
    elif string_trial == 'DE' and int_trial == 4:
        return True
    elif string_trial == 'EF' and int_trial == 5:
        return True
    elif string_trial == 'BD' and int_trial == 6:
        return True
    elif string_trial == 'CE' and int_trial == 7:
        return True
    elif string_trial == 'BE' and int_trial == 8:
        return True
    elif string_trial == 'AC' and int_trial == 9:
        return True
    elif string_trial == 'DF' and int_trial == 10:
        return True
    else:
        return False

def check_difference(list, threshold):
    """check if values in a list are within a certain range of each other

    Args:
        list (int/float list): list of values with the dfferences to check
        threshold (int): acceptable range for values within each other

    Returns:
        bool: returns True if there are points that are more than threshold away from each other
    """
    
    for a, b in combinations(list, 2):
        if abs(a - b) > threshold:
            return True

    return False # if none are found


### FILE MANIPULATION ---------
def add_row_to_csv(file_path, row_data, headers=None):
    """appends a single row to a csv file
       creates the files with the headers provided if it doesn't exist already

    Args:
        file_path (str): file path of the csv file
        row_data (dict): dictionary with {column header: value}
        headers (str list, optional): list of headers - shoudl correspond to dict. Defaults to None.
    """
    # infer headers if not provided
    if headers is None:
        headers = list(row_data.keys())
    
    # check if the file exists
    file_exists = False
    try:
        with open(file_path, "r"):
            file_exists = True
    except FileNotFoundError:
        pass
    
    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # write headers if file is being crated
        if not file_exists and headers:
            writer.writeheader()
        
        writer.writerow(row_data)


### STARTUP -------
def initial_processing(data_structure, rat, day):
    SS_log = data_structure[rat][day]["stateScriptLog"]
    timestamps = data_structure[rat][day]["videoTimeStamps"]
    
    trial_starts = get_video_trial_starts(timestamps, SS_log)
    
    dlc_path = os.path.join(BASE_PATH, "processed_data", "cleaned_dlc", rat)
    file_path = None
    for _, _, files in os.walk(dlc_path):
        for file in files:
            parts = file.split("_")
            day_from_file = parts[0]
            
            if day == day_from_file and "coordinates" in parts[1]:
                file_path = os.path.join(dlc_path, file)
                break
    
    df = pd.read_csv(file_path)
    
    return df, SS_log, timestamps, trial_starts
