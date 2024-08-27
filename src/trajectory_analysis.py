"""
calculates the VTE zIdPhi values for one day for one rat
main function to call is quantify_VTE()
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from datetime import datetime

from src import plotting
from src import creating_zones
from src import helper
from src import performance_analysis

### LOGGING
logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

# makes a new log everytime the code runs by checking the time
log_file = datetime.now().strftime("/Users/catpillow/Documents/VTE_Analysis/doc/calculating_VTEs_log_%Y%m%d_%H%M%S.txt")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

# pylint: disable=broad-exception-caught, invalid-name, logging-fstring-interpolation

### AUXILIARY FUNCTIONS ------
def type_to_choice(trial_type, correct):
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
    
    if isinstance(trial_type, str):
        trial_type = int(trial_type)
    
    if correct is True:
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
                print(f"Error for {helper.CURRENT_RAT} on {helper.CURRENT_DAY}")
                raise helper.ExpectationError("number from 1 - 10", trial_type)
    else:
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
                print(f"Error for {helper.CURRENT_RAT} on {helper.CURRENT_DAY}")
                raise helper.ExpectationError("number from 1 - 10", trial_type)

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
                        logging.info("model too far from actual results for "
                                     f"{helper.CURRENT_RAT} on {helper.CURRENT_DAY}")
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

def calculate_zIdPhi(IdPhi_values, trajectories=None, x=None, y=None):
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
def quantify_VTE(data_structure, rat_ID, day, save = None):
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
    helper.update_rat(rat_ID)
    helper.update_day(day)
    DLC_df, SS_log, timestamps, trial_starts = helper.initial_processing(data_structure, rat_ID, day)
    
    # check if timestamps is ascending bc annoying trodes code & files
    if not np.all(timestamps[:-1] <= timestamps[1:]):
        raise helper.CorruptionError(rat_ID, day, timestamps)
    
    # define zones
    centre_hull = creating_zones.get_centre_hull(DLC_df)
    
    # store IdPhi and trajectory values
    IdPhi_values = {}
    trajectories = {}
    store_data = []
    count = 0
    
    _, _, performance = performance_analysis.get_session_performance(SS_log) # a list of whether the trials resulted in a correct or incorrect choice
    same_len = helper.check_equal_length(performance, list(trial_starts.keys())) # check if there are the same number of trials for perf and trial_starts
    
    for i, (trial_start, trial_type) in enumerate(trial_starts.items()): # where trial type is a string of a number corresponding to trial type
        count += 1
        traj_id = rat_ID + "_" + day + "_" + str(count)
        
        # cut out the trajectory for each trial
        trajectory_x, trajectory_y, traj_len = helper.get_trajectory(DLC_df, trial_start, centre_hull, traj_id)
        if not trajectory_x: # empty list, happens for the last trajectory
            continue
        
        # calculate Idphi of this trajectory
        IdPhi = calculate_IdPhi(trajectory_x, trajectory_y)
        #plotting.plot_trajectory_animation(DLC_df["x"], DLC_df["y"], trajectory_x, trajectory_y, title=traj_id)
        
        # get the choice arm from the trial type and performance
        if same_len or len(performance) < i:
            choice = type_to_choice(trial_type, performance[i])
        elif len(performance) >= i:
            logging.debug(f"performance not capturing every trial for {rat_ID} on {day}")
            continue
        
        # store IdPhi according to which arm the rat went down
        if choice not in IdPhi_values:
            IdPhi_values[choice] = []
            
        IdPhi_values[choice].append(IdPhi)
        
        # store each trajectory for plotting later
        if choice in trajectories:
            trajectories[choice].append((trajectory_x, trajectory_y))
        else:
            trajectories[choice] = [(trajectory_x, trajectory_y)]
        
        # store each trajectory for later
        new_row = {"ID": traj_id, "X Values": trajectory_x, "Y Values": trajectory_y, "Correct": performance[i],
                   "Choice": choice, "Trial Type": trial_type, "IdPhi": IdPhi, "Length": traj_len}
        store_data.append(new_row)
        
        # plot and save if desired
        if save is not None:
            trajectory = (trajectory_x, trajectory_y)
            plotting.plot_trajectory(DLC_df["x"], DLC_df["y"], trajectory, title=f"{rat_ID}_{day}_{traj_id}", save=save, traj_id=traj_id)
    
    df = pd.DataFrame(store_data)
    file_path = os.path.join(save, "trajectories.csv")
    df.to_csv(file_path)
    plt.close()
    
    return IdPhi_values, trajectories