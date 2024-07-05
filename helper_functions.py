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
    - if point is in zone
        - is_point_in_ellipse
        - is_point_in_hull
    - checking things are normal
        - check_timestamps
        - check_equal_length
        - start_check
    - startup
        - initial_processing
"""

import math
import bisect
import numpy as np
from scipy.spatial import Delaunay

import data_processing

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
    SS_data = data_structure[ratID][day]['stateScriptLog']
    
    track_part = 'haunch' # assumed to be the best for tracking animal speed
    x, y = data_structure.filter_dataframe(DLC_data, track_part)
    
    # get framerate
    framerate = get_framerate(SS_data, x)
    
    # calculate speed
    diff_x = x.diff().fillna(0)
    diff_y = y.diff().fillna(0)
    
    displacement_per_frame = np.sqrt(diff_x**2 + diff_y**2)
    
    speed = displacement_per_frame * framerate
    
    return speed

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
    time_arm_seconds = get_time(content, time_arm)
    time_home_seconds = get_time(content, time_home)
    time_diff = time_arm_seconds - time_home_seconds
    
    return time_diff

def get_ss_trial_starts(SS_df):
    """
    gets the time associated with each of the trial start times from a statescript log
    trial start times being when the time associated with "New Trial" appearing

    Args:
        SS_df (str): statescript log

    Returns:
        dict: {trial_starts: trial_type}
    """
    
    lines = SS_df.splitlines()
    
    # storage variables
    start_of_trial = False # know when the last line was the start of new trial
    trial_starts = []
    trial_info = {} # store trial start times and trial types
    
    # get the trial start times from SS
    for line in lines:
        if line.startswith('#'): # skip the starting comments
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
            original_start_time = trial_starts[index]
            trial_type = trial_info.get(original_start_time) # get trial type for trial start time
            video_trial_info[video_start] = trial_type
    
    return video_trial_info

def get_trajectory(x, y, start, timestamps, hull):
    """
    gets all the x and y points within a trajectory given the start point and hull within which the trajectory is
    
    Args:
        x (int array): x coordinates from which to cut trajectory out of
        y (int array): y coordinates from which to cut trajectory out of
        start (int): index of dataframe corresponding to start of trajectory
        timestamps (int array): the times for each frame of the dataframe
        hull (scipy.spatial ConvexHull): hull within which trajectory is
    
    Returns:
        (int array): all x points for trajectory
        (int array): all y points for trajectory
    """
    
    # cut out the trajectory for each trial
    # look through points starting at trial start time to see when it goes into different hulls
    past_inside = False # this checks if any point has ever been inside hull for this iteration of loop
    trajectory_x = []
    trajectory_y = []
    
    for index in range(start, len(timestamps)): # x has been filtered so is not an appropriate length now
        # getting x and y
        if index == start:
            print(index)
        
        if index in x.index and index in y.index:
            x_val = x.loc[index] # loc is based on actual index, iloc is based on position
            y_val = y.loc[index]
        elif index == start: # elif instead of else so it doesn't spam this message
            print(f'trial started and cannot find x and y values - {start}')
            continue
        else:
            continue
        
        point = (x_val, y_val)
        inside = is_point_in_hull(point, hull) # check if still inside desired hull
        
        if inside:
            past_inside = True
            trajectory_x.append(x_val)
            trajectory_y.append(y_val)
        else:
            if past_inside:
                break # ok so now it has exited the centre hull
        
        """if index < trial_start + 1000:
            trajectory_x.append(x_val)
            trajectory_y.append(y_val)
        else:
            break"""
    
    return trajectory_x, trajectory_y



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
            index = timestamps.index(time)
            trial_starts.append(index)
        else: # if there isn't a perfect match
            #print(f"Imperfect match between ECU and MCU at {time}")
            
            # index where time is inserted into timestamps
            idx = bisect.bisect_left(timestamps, time)
            
            # check neighbours for closest time
            if idx == 0: # if time < any available timestamp, so = 0
                trial_starts.append(0)
            elif idx == len(timestamps): # if time > any available timestamp, so = len(timestamps)
                trial_starts.append(len(timestamps))
            else:
                before = timestamps[idx - 1]
                after = timestamps[idx]
                closest_time = before if (time - before) <= (after - time) else after
                index = np.where(timestamps == closest_time)[0][0]
                trial_starts.append(index)
    
    return trial_starts # with this, each trial_start is the index of the time when trial starts in relation to timestamps



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
    #print(timestamps)
    
    for index, time in enumerate(timestamps): #loops through timestamps
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
        if diff == 1:
            timestamps.pop()
    
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
    
    same_len = a.count == b.count

    return same_len
    
def start_check(DLC_df, timestamps):
    check_timestamps(DLC_df, timestamps)



### STARTUP -------
def initial_processing(data_structure, ratID, day):
    """
    does all the typical processing needed at the start of a cycle

    Args:
        data_structure (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
        ratID (str): the ID of the rat currently being processed
        day (str): the day currently being processed

    Returns:
        x (int array): x coordinates of rat position
        y (int array): y coordinates of rat position
        DLC_df (pd dataframe): dataframe containing rat position & likelihood of various body parts
        SS_log (str): statescript log
        timestamps (np int array): the time associated with each dlc coordinate
        trial_starts (dict): {trial_start: trial_type} where trial_start is the time when trials start
    """
    
    DLC_df = data_structure[ratID][day]['DLC_tracking']
    SS_log = data_structure[ratID][day]['stateScriptLog']
    timestamps = data_structure[ratID][day]['videoTimeStamps']
    
    start_check(DLC_df, timestamps) # initial check of everything
    
    # get trial start times + trial type
    trial_starts = get_video_trial_starts(timestamps, SS_log)
    
    # get x and y coordinates
    first_trial_start = next(iter(trial_starts)) # get the first trial start time to pass into filtering
    x, y = data_processing.filter_dataframe(DLC_df, start_index=first_trial_start)
    
    return x, y, DLC_df, SS_log, timestamps, trial_starts
