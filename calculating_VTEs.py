import math
import bisect
import numpy as np

import plotting
import creating_zones
import helper_functions

### GET INFORMATION -----------
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

def get_trial_start_times(timestamps, SS_df): # gets indices for x/y where trials start & corresponding trial type
    """
    gets the trial start times according to the corresponding index for dlc dataframe

    Args:
        timestamps (np int array): the times for each dlc frame
        SS_df (str): the statescript log

    Returns:
        dict: {trial type: trial starts}
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



### CALCULATING HEAD VELOCITY VALUES ----------------
def derivative(values, sr, d, m): # assumes each value is separated by regular time intervals -> sr
    """
    Estimates the derivative for a sequence of values sampled at regular intervals

    Args:
        values (numpy int array): 1D array of data points
        sr (float): sampling rate - time interval between consecutive data points
        d (float): threshold for quality of linear approximation
        m (int): maximum window length for differentiation

    Returns:
        numpy.ndarray: 1D array of estimated slopes (derivatives) with the same length as input 'values'
        
    Procedure:
        1. Initialise v_est with zeroes
            - store estimated slopes
        2. for each value in values, adjust window length in an infinite loop
            - make window 1 bigger until window > m (max length) or value index is bigger than window length
                - then make window 1 smaller & break loop
            - take the difference between the current value of values and the first value of window
            - if window > 1, check how well slope fits values in window
                - compute c for y = mx + c
                - for each point in window
                    - calculate deviation (delta)
                    - if delta > 2 * d, can_increase_window = False, window - 1, and go back to the last slope
            - assign calculated slope to v_est[i]
    """
    
    v_est = np.zeros_like(values) # initialise slope array with zeroes / velocity estimates
    #print(values)
    
    # start from second element for differentiation
    for i in range(1, len(values)):
        window_len = 0
        can_increase_window = True

        while True: # infinite loop
            window_len += 1
            
            if window_len > m or i - window_len < 0: # reached end of window / safety check
                window_len -= 1
                break
            
            # calculate slope from values[i] to values[i - window_len]
            slope_ = v_est[i - 1] # save previous slope / i changed from original code to be v_est[i - 1] instead of v_est[i]
            slope = (values[i] - values[i - window_len]) / (window_len * sr)
            
            if window_len > 1:
                #print("window_len > 1")
                # y = mx + c where c -> y-intercept, values[i] -> y, slope -> m, i * sr -> x (time at point i)
                c = values[i] - slope * i * sr

                # check every point
                for j in range(1, window_len):
                    # diff between actual point and position calculated by model at every point up to i
                    delta = values[i - j] - (c + slope * (i - j) * sr)
                    
                    # use delta to assess quality of linear approximation -> 2 * d is threshold
                    if abs(delta) > 2 * d: # if model too far from actuality, excludes the problematic point in model
                        can_increase_window = False
                        window_len -= 1
                        slope = slope_
                        #print("model too far from actual results")
                        break
            
            if not can_increase_window:
                break # exit while loop if window cannot be increased
        
        v_est[i] = slope
    
    return v_est

def calculate_IdPhi(trajectory_x, trajectory_y):
    """
    calculating IdPhi value given trajectory

    Args:
        trajectory_x (np int array): x values for trajectory
        trajectory_y (np int array): y values for trajectory

    Returns:
        float: numerical integration of change of angular velocity values (IdPhi)
    """
    
    # parameters - need to change
    sr = 0.03 # sampling rate
    d = 0.05 # position noise boundary
    m = 20 # window size
    
    # derivatives
    dx = derivative(trajectory_x, sr, d, m)
    dy = derivative(trajectory_y, sr, d, m)
    
    # calculate + unwrap angular velocity
    Phi = np.arctan2(dy, dx)
    Phi = np.unwrap(Phi)
    dPhi = derivative(Phi, sr, d, m)
    
    # integrate change in angular velocity
    IdPhi = np.trapz(np.abs(dPhi))
    
    return IdPhi



### TRAJECTORY ANALYSIS ------------
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
        inside = helper_functions.is_point_in_hull(point, hull) # check if still inside desired hull
        
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



def quantify_VTE(data_structure, ratID, day, save = False):
    """DLC_df = data_structure[ratID][day]['DLC_tracking']
    SS_df = data_structure[ratID][day]['stateScriptLog']
    timestamps = data_structure[ratID][day]['videoTimeStamps']"""
    
    DLC_df = data_structure[day]['DLC_tracking']
    SS_df = data_structure[day]['stateScriptLog']
    timestamps = data_structure[day]['videoTimeStamps']
    
    # check timestamps
    helper_functions.check_timestamps(DLC_df, timestamps)

    # get trial start times + trial type
    trial_starts = get_trial_start_times(timestamps, SS_df)
    
    # get x and y coordinates
    first_trial_start = next(iter(trial_starts)) # get the first trial start time to pass into filtering
    x, y = data_structure.filter_dataframe(DLC_df, start_index=first_trial_start)
    
    # define zones
    centre_hull = creating_zones.get_centre_zone(x, y, ratID, day, save)
    
    # calculate IdPhi for each trial
    IdPhi_values = {}
    
    # store trajectories for plotting later
    trajectories = {}
    
    for trial_start, trial_type in trial_starts.items(): # where trial type is a string of a number corresponding to trial type
        # cut out the trajectory for each trial
        # look through points starting at trial start time to see when it goes into different hulls
        past_inside = False # this checks if any point has ever been inside hull for this iteration of loop
        trajectory_x = []
        trajectory_y = []
        
        trial_start = math.floor(trial_start) # round down so it can be used as an index
        
        for index in range(trial_start, len(timestamps)): # x has been filtered so is not an appropriate length now
            # getting x and y
            if index == trial_start:
                print(index)
            
            if index in x.index and index in y.index:
                x_val = x.loc[index] # loc is based on actual index, iloc is based on position
                y_val = y.loc[index]
            elif index == trial_start:
                print(f'trial started and cannot find x and y values - {trial_start}')
                continue
            else:
                continue
            
            """point = (x_val, y_val)
            inside = check_if_inside(point, centre_hull)
            
            if inside:
                past_inside = True
                trajectory_x.append(x_val)
                trajectory_y.append(y_val)
            else:
                if past_inside:
                    break # ok so now it has exited the centre hull"""
            
            if index < trial_start + 1000:
                trajectory_x.append(x_val)
                trajectory_y.append(y_val)
            else:
                break
        
        # calculate Idphi of this trajectory
        IdPhi = calculate_IdPhi(trajectory_x, trajectory_y)
        #plot_animation(x, y, trajectory_x = trajectory_x, trajectory_y= trajectory_y)
        
        # store IdPhi according to trial type
        if trial_type not in IdPhi_values:
            IdPhi_values[trial_type] = []
        IdPhi_values[trial_type].append(IdPhi)
        
        # store each trajectory for plotting latter
        if trial_type in trajectories:
            trajectories[trial_type].append((trajectory_x, trajectory_y))
        else:
            trajectories[trial_type] = [(trajectory_x, trajectory_y)]
    
    # calculate zIdPhi according to trial types
    zIdPhi_values = {}
    highest_zIdPhi = None
    highest_trajectories = None
    lowest_zIdPhi = None
    lowest_trajectories = None
    
    # this z scores according to trial type
    for trial_type, IdPhis in IdPhi_values.items():
        zIdPhis = zscore(IdPhis)
        zIdPhi_values[trial_type] = zIdPhis
        
        for i, zIdPhi in enumerate(zIdPhis): # this is to get the highest and lowest zidphi for plotting vte/non
            if highest_zIdPhi:
                if zIdPhi > highest_zIdPhi:
                    highest_zIdPhi = zIdPhi
                    highest_trajectories = trajectories[trial_type][i]
            else:
                highest_zIdPhi = zIdPhi
                highest_trajectories = trajectories[trial_type][i]
            
            if lowest_zIdPhi:
                if zIdPhi < lowest_zIdPhi and len(trajectories[trial_type][i]) > 2:
                    lowest_zIdPhi = zIdPhi
                    lowest_trajectories = trajectories[trial_type][i]
            else:
                lowest_zIdPhi = zIdPhi
                lowest_trajectories = trajectories[trial_type][i]
    
    highest_trajectory_x, highest_trajectory_y = highest_trajectories
    lowest_trajectory_x, lowest_trajectory_y = lowest_trajectories
    
    plot_zIdPhi(zIdPhi_values)
    plot_animation(x, y, highest_trajectory_x, highest_trajectory_y, highest = 2, zIdPhi=highest_zIdPhi)
    plot_animation(x, y, lowest_trajectory_x, lowest_trajectory_y, highest = 1, zIdPhi=lowest_zIdPhi)
    
    return zIdPhi_values, IdPhi_values, trajectories