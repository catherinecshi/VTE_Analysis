import os
import re
import numpy as np
import pandas as pd
from typing import Dict

from config import settings
from config.paths import paths, remote
from utilities import logging_utils
from debugging import error_types
from preprocessing import data_processing
from preprocessing import process_statescript
from visualization import performance_plots

# pylint: disable=logging-fstring-interpolation, consider-using-enumerate, broad-exception-caught
logger = logging_utils.get_module_logger("performance_analysis")

# ==============================================================================
# GET TRIALS
# ==============================================================================

def get_trials_available(save_path: str, data_structure: Dict) -> pd.DataFrame:
    """
    returns a dataframe with structure
    {'rat': ratID, 'day': day, 'trials_available': which trials were available that day}

    Args:
        save_path (str): path where data is stored
        data_structure (dict): dictionary with all of the data

    Returns:
        pd.DataFrame: dataframe with structure outlined above
    """
    
    trials_available = []
    for rat in os.listdir(save_path):
        settings.update_rat(rat)
        rat_path = os.path.join(save_path, rat, remote.module)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            logger.info(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        # sort through the days so it's stored in ascending order
        days = os.listdir(rat_path)
        if days is not None or days != "":
            sorted_days = sorted(days, key=lambda x: int(re.sub(r"\D", "", x)) if re.sub(r"\D", "", x) else 100)
        else:
            logger.debug(f"no days for {rat}, {rat_path}")
            continue
        
        for day in sorted_days:
            # check for DS_Store
            if not os.path.isdir(os.path.join(rat_path, day)):
                logger.info(f"skipping over non-directory folder: {day}")
                continue
            
            settings.update_day(day)
            match = re.search(r"\d+", day)
            if match:
                day_number = int(match.group())
            else:
                raise error_types.NoMatchError("integers", day, "get_trials_available")

            if day_number == 100:
                logger.error(f"day with no number for {rat}")
            
            try:
                content = data_structure[rat][day][settings.SS]
                trials_in_day = process_statescript.get_trials_for_session(content)
                trials_available.append({"rat": rat, "day": day_number, "trials_available": trials_in_day})
            except Exception as e:
                logger.error(f"error {e} for rat {rat} on {day} in get_trials_available")
    
    df = pd.DataFrame(trials_available)
    return df

# ==============================================================================
# GET PERFORMANCE
# ==============================================================================

def one_rat_performance(data_structure, rat_ID):
    """
    analyses performance of one rat over days

    Args:
        data_structure (dict): {rat_folder: {day_folder: {"DLC_tracking":dlc_data, "stateScriptLog": ss_data, "timestamps": timestamps_data}}}
        ratID (str): rat
        should_save (bool, optional): if True, then save to a file. Defaults to False.
        save_path (str, optional): path to be saved to if should_save is true. Defaults to None.

    Returns:
        pd.DataFrame: {'rat', 'day', 'trial_type', 'total_trials', 'correct_trials'}
    """
    if data_structure is None:
        data_structure = data_processing.load_data_structure()
    
    if rat_ID not in data_structure:
        raise error_types.ExpectationError("ratID", rat_ID)
    
    rat_performance = pd.DataFrame(columns=["rat", "day", "trial_type", "total_trials", "correct_trials"])
    for day_folder, contents in data_structure[rat_ID].items():
        if day_folder == ".DS_Store" or "pkl" in day_folder: # skip .DS_Store
            continue
        
        settings.update_day(day_folder)
        
        ss_data = contents["stateScriptLog"]
        if ss_data is None:
            logger.error(f"ss_data is None Type for {rat_ID} on {day_folder}")
            raise error_types.UnexpectedNoneError("one_rat_performance", "ss_data")
        
        try:
            trial_types_set, total_trials, correct_trials = process_statescript.trial_accuracy(ss_data)
        except Exception as e:
            logger.error(f"error during trial_accuracy {e} for {rat_ID} on {day_folder}")
            continue

        trial_types = sorted(trial_types_set)

        # exclude days where there were errors and didn't even have AB and BC
        # and days i'm too lazy to decode right now
        length = len(total_trials)
        if length < 2 or length > 5:
            logger.error(f"total trial length is is {length} for {rat_ID} on {day_folder}")
            logger.error(f"total trials - {total_trials}")
            logger.error(f"correct trials - {correct_trials}")
            continue
        
        for i in range(len(total_trials)):
            try:
                match = re.search(r"\d+", day_folder) # gets first int in the string
                if match:
                    day_int = int(match.group())
                    rat_performance.loc[len(rat_performance)] = [rat_ID, day_int, trial_types[i], total_trials[i], correct_trials[i]]
            except IndexError as error: # happens if lengths are incongruent
                logger.error(f"Error for rat {rat_ID} for {day_folder}: {error}")
                logger.error(f"trial types - {trial_types}")
                logger.error(f"total trials - {total_trials}")
                logger.error(f"correct trials - {correct_trials}")
    
    return rat_performance # this returns df for one rat

def get_all_rats_performance(data_structure=None, save_path=None, save=True):
    """
    loops through all the rats to create csv files from ss data

    Args:
        data_structure (dict): all the data

    Returns:
        pd.DataFrames: {'rat', 'day', 'trial_type', 'total_trials', 'correct_trials'}
                        where trial_type are the corresponding numbers
    """
    if save_path is None:
        save_path = paths.performance / "rat_performance.csv"
    
    if save:
        dataframes = []
        
        if data_structure is None:
            data_structure = data_processing.load_data_structure()
        
        for rat_ID in data_structure:
            try:
                settings.update_rat(rat_ID)
                rat_performance = one_rat_performance(data_structure, rat_ID)
                dataframes.append(rat_performance)
            except Exception as e:
                logger.error(f"error {e} for {rat_ID} and maybe {settings.CURRENT_DAY} when getting performance")
        all_rats_performances = pd.concat(dataframes, ignore_index=True)
        
        # save dataframe
        all_rats_performances.to_csv(save_path)
    else:
        all_rats_performances = pd.read_csv(save_path)
    
    return all_rats_performances

# ==============================================================================
# GET CHANGE IN PERFORMANCE
# ==============================================================================

def get_change_in_performance(rat_performance: dict, criterias=None) -> tuple[pd.DataFrame, list]:
    """
    calculates the change in performance of each consecutive day

    1. loops through days, where one is the current day and one is the day prior
        skips day 1 bc no prior days
    2. loops through trial types for current and past day
        checks for matches between trial types
    3. if match, then calculate the performance of current and past day
        then get the change in performance between the two
        
        if this is only done until criteria, then check the day to make sure the current day's change in performnace
        should be kept

    Args:
        rat_performance (dictionary): {day: trial_type, total_trials, correct_trials}
        criterias (dictionary, optional): {trial_type: criteria_day}. Defaults to None.

    Returns:
        performance_changes (pd.DataFrame): {'day': day, 'trial_type': trial_type, 'perf_change': change}
    
    Notes:
        - currently calculates separate changes in performance for each trial type
        - the day in the dataframe corresponds to the latter day being compared
    """
    
    sorted_days = sorted(rat_performance.keys(), key = lambda x: int(x[3:])) # sort by after 'day'
    past_day_perf = None # storage
    performance_changes = [] # array for differences in performances
    avg_changes = []
    
    # loop through days
    for day in sorted_days:
        all_performances_changes_for_day = []
        performance_for_day = rat_performance[day]
        
        # first day doesn't count as a change in performance
        if day is "Day1":
            past_day_perf = performance_for_day
            continue
        
        for trial_type_i, total_trials_i, correct_trials_i in performance_for_day:
            if past_day_perf is None:
                logger.error(f"unexpected none for past_day_perf in change_in_performance_trial_type for {settings.CURRENT_RAT} on {day}")
                continue
            
            for trial_type_j, total_trials_j, correct_trials_j in past_day_perf:
                if trial_type_i not in trial_type_j: # skip through things that are not the same trial type
                    continue
                
                # if it's here there's a match between trial type
                trial_type = trial_type_j # save the trial type
                
                if criterias: # if criterias exist
                    # convert day into an integer
                    match = re.search(r"\d+", day) # gets first int in the string
                    criteria_day = 0  # Initialize with default value
                    if match:
                        day_int = int(match.group())
                        criteria_day = criterias[trial_type]  # Set actual value
                        if "CD" in trial_type or "DE" in trial_type:
                            criteria_day += day_int
                    
                    if criteria_day == day_int:
                        # calculate performance change
                        past_performance = correct_trials_j / total_trials_j
                        current_performance = correct_trials_i / total_trials_i
                        change = current_performance - past_performance # change in performance
                        
                        performance_changes.append({"day": day, "trial_type": trial_type, "perf_change": change})
                    elif criteria_day < day_int and criteria_day != 0:
                        continue # skip if criteria has already been reached
                    else: # this would be if the current day is lower than the criteria
                        # calculate performance change
                        past_performance = correct_trials_j / total_trials_j
                        current_performance = correct_trials_i / total_trials_i
                        change = current_performance - past_performance # change in performance
                        
                        performance_changes.append({"day": day, "trial_type": trial_type, "perf_change": change})
                else:
                    # calculate performance change
                    past_performance = correct_trials_j / total_trials_j
                    current_performance = correct_trials_i / total_trials_i
                    change = current_performance - past_performance # change in performance
                    
                    performance_changes.append({"day": day, "trial_type": trial_type, "perf_change": change})
                    all_performances_changes_for_day.append(change)
                logger.info(f"performance change - {change} on day {day} for trial type {trial_type}")
                logger.info(f"past performance - {past_performance}, current - {current_performance}")
        
        avg_change = np.mean(all_performances_changes_for_day)
        avg_changes.append(avg_change)
        past_day_perf = performance_for_day # update past day
    
    changes_df = pd.DataFrame(performance_changes)
    
    return changes_df, avg_changes

def save_all_perf_changes(all_rats_performances):
    """
    calculates the change in performance across days for all rats. first day returns NaN

    Args:
        all_rats_performances (pd.DataFrame): {'rat', 'day', 'trial_type', 'total_trials', 'correct_trials'}

    Returns:
        pd.DataFrame: {'rat', 'day', 'perf_change'}
    """
    rat_data = all_rats_performances.groupby('rat')
    
    all_perf_changes = []
    all_rat_day_pairs = []
    for rat, rat_group in rat_data:
        sorted_rat_data = rat_group.sort_values(by="day")
        day_data = sorted_rat_data.groupby("day")
        
        rat_performance = []
        for day, day_group in day_data:
            day_performance = day_group["correct_trials"] / day_group["total_trials"]
            day_performance = day_performance.mean() # get mean to average across trial types
            rat_performance.append(day_performance)
            all_rat_day_pairs.append((rat, day))
        
        rat_performance_series = pd.Series(rat_performance)
        perf_changes = rat_performance_series.diff()
        all_perf_changes.append(perf_changes)
    perf_changes_series = pd.concat(all_perf_changes, ignore_index=True)
    
    # create new dataframe
    perf_changes_df = pd.DataFrame(all_rat_day_pairs, columns=["rat", "day"])
    perf_changes_df["perf_change"] = perf_changes_series.reset_index(drop=True)
    
    # save dataframe
    save_path = paths.performance / "performance_changes.csv"
    perf_changes_df.to_csv(save_path)
    
    return perf_changes_df

def save_all_perf_changes_trials(all_rats_performances):
    """
    calculates the change in performance across days for all rats according to trial type
    first day returns NaN

    Args:
        all_rats_performances (pd.DataFrame): {'rat', 'day', 'trial_type', 'total_trials', 'correct_trials'}

    Returns:
        pd.DataFrame: {'rat', 'day', 'trial_type', 'perf_change'}
    """
    rat_data = all_rats_performances.groupby("rat")
    
    all_rat_perf = []
    for rat, rat_group in rat_data:
        sorted_rat_data = rat_group.sort_values(by="day")
        trial_data = sorted_rat_data.groupby("trial_type")
        
        for trial_type, trial_group in trial_data:
            trial_performance = (trial_group["correct_trials"] / trial_group["total_trials"]) * 100
            perf_change_in_trial = trial_performance.diff()
            for i, (_, row) in enumerate(trial_group.iterrows()):
                if i == 0:
                    all_rat_perf.append({"rat": rat, "day":row["day"], "trial_type":trial_type, "perf_change": trial_performance.iloc[0] - 50})
                else:
                    all_rat_perf.append({"rat": rat, "day":row["day"], "trial_type":trial_type, "perf_change":perf_change_in_trial.iloc[i]})
            

    # save and return dataframe
    perf_changes_df = pd.DataFrame(all_rat_perf)
    save_path = paths.performance / "performance_changes_by_trial.csv"
    perf_changes_df.to_csv(save_path)
    return perf_changes_df

# ==============================================================================
# GET DAYS UNTIL CRITERIA
# ==============================================================================

def days_until_criteria(all_rats_performances):
    """counts the days until a rat hits criteria for a specific trial type

    Args:
        all_rats_performances (pd.DataFrames): {'rat', 'day', 'trial_type', 'total_trials', 'correct_trials'}
                                               where trial_type are the corresponding numbers

    Returns:
        dict: {rat: {trial_type: day}} where day is the day in which the rat reached criteria
    """
    
    all_days_until_criteria = {}
    
    for rat, rat_data in all_rats_performances.groupby("rat"):
        # skip BP06 bc its recording didn't start until day 7
        if rat == "BP06":
            continue
        
        # each trial type
        trial_learned: dict[int, int] = {}
        trial_starts = np.zeros(5)
        day_learned: dict[int, int] = {}
        
        sorted_days = rat_data.sort_values(by=["day"])
        for _, row in sorted_days.iterrows():
            day = row["day"]
            trial_type = row["trial_type"]
            total_trials = row["total_trials"]
            correct_trials = row["correct_trials"]
            performance = correct_trials / total_trials
            
            # get when trial types are first introduced
            if trial_type == 1 and trial_starts[0] == 0: # AB
                trial_starts[0] = day
                if day != 1:
                    print(f"{rat}'s AB started on day {day}")
            elif trial_type == 2 and trial_starts[1] == 0: # BC
                trial_starts[1] = day
                if day != 1:
                    print(f"{rat}'s BC started on day {day}")
            elif trial_type == 3 and trial_starts[2] == 0: # CD
                trial_starts[2] = day
            elif trial_type == 4 and trial_starts[3] == 0: # DE
                trial_starts[3] = day
            elif trial_type == 5 and trial_starts[4] == 0: # EF
                trial_starts[4] = day
            
            # get when performance crosses threshold
            if performance >= 0.75:
                if trial_type in trial_learned:
                    if trial_learned[trial_type] == 0:
                        trial_learned[trial_type] += 1
                    elif trial_learned[trial_type] == 1: # second >0.75 in a row
                        if trial_type in day_learned:
                            continue
                        
                        day -= (trial_starts[trial_type - 1] - 1)
                        
                        if day < 2: # something has gone wrong
                            print(f"{rat} on day{day} for {trial_type} has day < 2")
                        
                        day_learned[trial_type] = day
                        trial_learned[trial_type] += 1
                    elif trial_learned[trial_type] < 0 or trial_learned[trial_type] > 3:
                        print(f"trial learned is not 0 or 1 {trial_learned[trial_type]}")
                else:
                    trial_learned[trial_type] = 1
            else: # reset bc criteria needs in a row
                trial_learned[trial_type] = 0
        
        all_days_until_criteria[rat] = day_learned
    
    return all_days_until_criteria # returns {ratID: {trial_type:day}} where day is day it was learned

def perf_until_critera(all_rats_performances):
    # gets the performance array of how the rats did before hitting criteria, and plots it
    all_rats_criteria_days = days_until_criteria(all_rats_performances)
    all_rats_changes = {}
    
    for rat_ID, criteria_days in all_rats_criteria_days.items():
        rat_performance = all_rats_performances[rat_ID]
        
        performance_changes = get_change_in_performance(rat_performance, criteria_days)
        all_rats_changes[rat_ID] = performance_changes
    
    performance_plots.plot_all_rat_perf_changes(all_rats_changes, criterias=True)

def get_days_since_new_arm():
    """
    Calculate days since new arm was added for each rat based on Trial_Type changes in zIdPhi.csv files.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['rat', 'day', 'arm_added', 'days_since_new_arm', 'trials_available']
    """
    results = []
    vte_path = paths.vte_values
    
    for rat in os.listdir(vte_path):
        if ".DS" in rat or "inferenceTesting" in rat:
            continue
        
        rat_path = os.path.join(vte_path, rat)
        
        # Find zIdPhi.csv file for this rat (exact filename match only)
        zidphi_file = None
        for root, _, files in os.walk(rat_path):
            for f in files:
                if f == "zIdPhis.csv":  # Exact match only, ignore other files with zIdPhi in name
                    zidphi_file = os.path.join(root, f)
                    break
            if zidphi_file:
                break
        
        if not zidphi_file:
            print(f"Warning: No zIdPhis.csv file found for rat {rat}")
            continue
            
        try:
            # Load the CSV file
            df = pd.read_csv(zidphi_file)
            
            # Extract day numbers from Day column (e.g., "Day12" -> 12)
            df['day_num'] = df['Day'].str.extract(r'Day(\d+)').astype(int)
            
            # Group by day and get unique trial types for each day
            daily_data = []
            for day_num in sorted(df['day_num'].unique()):
                day_data = df[df['day_num'] == day_num]
                unique_trials = sorted(day_data['Trial_Type'].unique())
                daily_data.append({
                    'day_num': day_num,
                    'day_name': day_data['Day'].iloc[0],
                    'trials_available': unique_trials
                })
            
            # Process each day to determine when new arms are added
            previous_trial_count = 0
            days_since_new_arm = 0
            all_previous_trials = set()  # Track all trial types seen so far
            
            skip_rat = False
            
            for i, day_info in enumerate(daily_data):
                current_trials = set(day_info['trials_available'])
                current_trial_count = len(current_trials)
                
                # Check for error condition: new trial types that were seen before but not in previous days
                if i > 0:  # Skip first day
                    new_trials = current_trials - all_previous_trials
                    if new_trials:
                        # If there are completely new trial types, the count should increase
                        if current_trial_count == previous_trial_count:
                            raise ValueError(f"Error: Rat {rat} on {day_info['day_name']} has new trial types {new_trials} but same total count as previous day. This shouldn't happen.")
                
                # Determine if new arm was added
                arm_added = False
                if i == 0:  # First day - treat as new arm introduction
                    arm_added = True
                    days_since_new_arm = 0
                elif current_trial_count > previous_trial_count:
                    # More trial types than previous day = new arm added
                    arm_added = True
                    days_since_new_arm = 0
                else:
                    # Same number of trial types as previous day
                    days_since_new_arm += 1
                
                # Store result
                results.append({
                    'rat': rat,
                    'day': day_info['day_num'],
                    'arm_added': arm_added,
                    'days_since_new_arm': days_since_new_arm,
                    'trials_available': day_info['trials_available']
                })
                
                # Update tracking variables for next iteration
                previous_trial_count = current_trial_count
                all_previous_trials.update(current_trials)
            
            if skip_rat:
                # Remove any results added for this rat before the error
                results = [r for r in results if r['rat'] != rat]
                
        except Exception as e:
            print(f"Error processing rat {rat}: {e}")
            # Remove any results added for this rat before the error
            results = [r for r in results if r['rat'] != rat]
            continue
    
    return pd.DataFrame(results)
