"""
check over data and file integrity before processing
"""

import numpy as np

from config import settings
from utilities import error_types

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

def trial_type_equivalency(trial_type_i, trial_type_j):
    """check if a string and int trial type are the same trial type"""
    
    string_trial = None
    int_trial = None

    if isinstance(trial_type_i, str):
        string_trial = trial_type_i
    elif isinstance(trial_type_i, int):
        int_trial = trial_type_i
    else:
        raise error_types.ExpectationError(trial_type_i, "trial type int or str")
    
    if isinstance(trial_type_j, str) and string_trial is not None:
        string_trial = trial_type_j
    elif isinstance(trial_type_j, str): # both are strings
        return trial_type_i is trial_type_j
    elif isinstance(trial_type_j, int) and int_trial is not None:
        int_trial = trial_type_j
    elif isinstance(trial_type_j, int): # both are int
        return trial_type_i == trial_type_j
    else:
        raise error_types.ExpectationError(trial_type_j, "str or int trial type")
    
    if string_trial is None:
        raise error_types.UnexpectedNoneError("trial_type_equivalency", "string_trial")
    elif int_trial is None:
        raise error_types.UnexpectedNoneError("trial_type_equivalency", "int_trial")
    else:
        return settings.TRIAL_TYPE_MAPPINGS.get(string_trial) == int_trial
    