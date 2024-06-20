"""
calculates the VTE zIdPhi values for one day for one rat
main function to call is quantify_VTE()
"""

import os
import pickle
import numpy as np
from scipy.stats import zscore

import plotting
import creating_zones
import helper_functions
import performance_analysis

### AUXILIARY FUNCTIONS ------
def type_to_choice(trial_type, correct, ratID, day):
    """
    gets the arm the rat went down when given trial_type and whether the rat got the choice correct

    Args:
        trial_type (int): the number corresponding to the trial type, as shown at the start of statescript comments
        correct (str): either "correct" for correct choice, or "Wrong" for incorrect choice. Note the capitalisation
        ratID (str): the current rat
        day (str): the current day being analysed

    Raises:
        helper_functions.ExpectationError: if trial_type is a number it shouldn't be
        helper_functions.ExpectationError: if trial_type is a number it shouldn't be
        helper_functions.ExpectationError: if correct doesn't correspond to either "correct" or "Wrong"

    Returns:
        (str): the letter corresponding to which arm the rat went down
    """
    
    choice = None
    
    if "correct" in correct:
        match trial_type:
            case 1:
                choice = "A"
            case 2:
                choice = "B"
            case 3:
                choice = "C"
            case 4:
                choice = "D"
            case 5:
                choice = "E"
            case 6:
                choice = "B"
            case 7:
                choice = "C"
            case 8:
                choice = "B"
            case 9:
                choice = "A"
            case 10:
                choice = "D"
            case _:
                print(f"Error for {ratID} on {day}")
                raise helper_functions.ExpectationError("number from 1 - 10", trial_type)
    elif "Wrong" in correct:
        match trial_type:
            case 1:
                choice = "B"
            case 2:
                choice = "C"
            case 3:
                choice = "D"
            case 4:
                choice = "E"
            case 5:
                choice = "F"
            case 6:
                choice = "D"
            case 7:
                choice = "E"
            case 8:
                choice = "E"
            case 9:
                choice = "C"
            case 10:
                choice = "F"
            case _:
                print(f"Error for {ratID} on {day}")
                raise helper_functions.ExpectationError("number from 1 - 10", trial_type)
    else:
        print(f"Error for {ratID} on {day}")
        raise helper_functions.ExpectationError("correct or Wrong", correct)

    return choice



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

def calculate_zIdPhi(IdPhi_values, trajectories = None, x = None, y = None):
    """
    calculates the zIdPhi values when given the IdPhi values, and zscores according to which arm the rat went down
    takes trajectories as well for visualising purposes

    Args:
        IdPhi_values (dict): {choice: IdPhi} where choice is where the rat went down, and IdPhi is the head velocity value
        trajectories (dict): {choice: (trajectory_x, trajectory_y)}

    Returns:
        (dict): {choice: zIdPhi}
    """
    
    # calculate zIdPhi according to trial types
    zIdPhi_values = {}
    highest_zIdPhi = None
    highest_trajectories = None
    lowest_zIdPhi = None
    lowest_trajectories = None
    
    # this z scores according to choice arm
    for choice, IdPhis in IdPhi_values.items():
        zIdPhis = zscore(IdPhis) # zscored within the sample of same choices within a session
        zIdPhi_values[choice] = zIdPhis
        
        if trajectories:
            for i, zIdPhi in enumerate(zIdPhis): # this is to get the highest and lowest zidphi for plotting vte/non
                if highest_zIdPhi:
                    if zIdPhi > highest_zIdPhi:
                        highest_zIdPhi = zIdPhi
                        highest_trajectories = trajectories[choice][i]
                else:
                    highest_zIdPhi = zIdPhi
                    highest_trajectories = trajectories[choice][i]
                
                if lowest_zIdPhi:
                    if zIdPhi < lowest_zIdPhi and len(trajectories[choice][i]) > 2:
                        lowest_zIdPhi = zIdPhi
                        lowest_trajectories = trajectories[choice][i]
                else:
                    lowest_zIdPhi = zIdPhi
                    lowest_trajectories = trajectories[choice][i]
    
    if trajectories and x and y:
        plotting.plot_zIdPhi(zIdPhi_values)
        
        highest_trajectory_x, highest_trajectory_y = highest_trajectories
        lowest_trajectory_x, lowest_trajectory_y = lowest_trajectories
        
        plotting.plot_trajectory_animation(x, y, highest_trajectory_x, highest_trajectory_y, title = "Highest zIdPhi Trajectory", label = highest_zIdPhi)
        plotting.plot_trajectory_animation(x, y, lowest_trajectory_x, lowest_trajectory_y, title = "Lowest zIdPhi Trajectory", label = lowest_zIdPhi)
    
    return zIdPhi_values



### MAIN FUNCTIONS -----------
def quantify_VTE(data_structure, ratID, day, save = False):
    """
    gets relevant VTE values for one rat for a specific day

    Args:
        data_structure (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
        ratID (str): current rat being processed
        day (str): current day being processed
        save (bool, optional): whether plots should be saved. Defaults to False.

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
    
    x, y, _, SS_log, timestamps, trial_starts = helper_functions.initial_processing(data_structure, ratID, day)
    
    # define zones
    centre_hull = creating_zones.get_centre_zone(x, y, ratID, day, save)
    
    # store IdPhi and trajectory values
    IdPhi_values = {}
    trajectories = {}
    
    performance = performance_analysis.trial_perf_for_session(SS_log) # a list of whether the trials resulted in a correct or incorrect choice
    same_len = helper_functions(performance, trial_starts) # check if there are the same number of trials for perf and trial_starts
    
    # raise error if there isn't the same number of trials for performance and trial_starts
    if not same_len:
        print(f"Mismatch for {ratID} on {day} for performance vs trial_starts")
        raise helper_functions.LengthMismatchError(performance.count, trial_starts.count)
    
    for i, (trial_start, trial_type) in enumerate(trial_starts.items()): # where trial type is a string of a number corresponding to trial type
        # cut out the trajectory for each trial
        trajectory_x, trajectory_y = helper_functions.get_trajectory(x, y, trial_start, timestamps, centre_hull)
        
        # calculate Idphi of this trajectory
        IdPhi = calculate_IdPhi(trajectory_x, trajectory_y)
        #plot_animation(x, y, trajectory_x = trajectory_x, trajectory_y= trajectory_y)
        
        # get the choice arm from the trial type and performance
        choice = type_to_choice(trial_type, performance[i])
        
        # store IdPhi according to which arm the rat went down
        if choice not in IdPhi_values:
            IdPhi_values[choice] = []
            
        IdPhi_values[choice].append(IdPhi)
        
        # store each trajectory for plotting later
        if choice in trajectories:
            trajectories[choice].append((trajectory_x, trajectory_y))
        else:
            trajectories[choice] = [(trajectory_x, trajectory_y)]
    
    zIdPhi_values = calculate_zIdPhi(IdPhi_values, trajectories)
    
    return zIdPhi_values, IdPhi_values, trajectories

def rat_VTE_over_sessions(data_structure, ratID):
    """
    iterates over each day for one rat, then save zIdPhi, IdPhi and trajectories using pickle
    saves three file per day (zIdPhi values, IdPhi values, trajectories)

    Args:
        data_structure (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
        ratID (str): rat
    
    Raises:
        Exception: if something bad happens for a day
    """
    
    rat_path = f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/{ratID}'
    
    for day in data_structure:
        try:
            zIdPhi, IdPhi, trajectories = quantify_VTE(data_structure, ratID, day, save = False)
            zIdPhi_path = os.path.join(rat_path, day, 'zIdPhi.npy')
            IdPhi_path = os.path.join(rat_path, day, 'IdPhi.npy')
            trajectories_path = os.path.join(rat_path, day, 'trajectories.npy')
            # save 
            with open(zIdPhi_path, 'wb') as fp:
                pickle.dump(zIdPhi, fp)
            
            with open(IdPhi_path, 'wb') as fp:
                pickle.dump(IdPhi, fp)
            
            with open(trajectories_path, 'wb') as fp:
                pickle.dump(trajectories, fp)
        except Exception as error:
            print(f'error in rat_VTE_over_session - {error} on day {day}')
            
def zIdPhis_across_sessions(base_path):
    """
    zscores the zIdPhis across an multiple sessions instead of just one session
    increases sample such that what counts as a VTE should be more accurate, given camera is constant

    Args:
        base_path (str): file path where IdPhi values were saved, presumably from rat_VTE_over_sessions

    Raises:
        helper_functions.ExpectationError: if more than 1 IdPhi values file is found in a day

    Returns:
        (dict): {choice: zIdPhi_values}
    """
    
    IdPhis_across_days = {} # this is so it can be zscored altogether
    days = []
    IdPhis_in_a_day = 0
    
    for day_folder in os.listdir(base_path):
        day_path = os.path.join(base_path, day_folder)
        if os.path.isdir(day_path):
            days.append(day_folder)
        
        for root, dirs, files in os.walk(day_path):
            for f in files:
                file_path = os.path.join(root, f)
                if 'IdPhi' in f and 'z' not in f:
                    with open(file_path, 'rb') as fp:
                        IdPhi_values = pickle.load(fp)
                    
                    for (choice, IdPhis) in IdPhi_values:
                        if not IdPhis_across_days[choice]:
                            IdPhis_across_days[choice] = []
                        
                        IdPhis_across_days[choice].append(IdPhis)
                        IdPhis_in_a_day += 1

                    if IdPhis_in_a_day > 1:
                        print(f"Error on day {day_folder}")
                        raise helper_functions.ExpectationError("only 1 IdPhi file in a day", "more than 1")
        
        IdPhis_in_a_day = 0
    
    zIdPhi_values = calculate_zIdPhi(IdPhis_across_days)
    
    return zIdPhi_values