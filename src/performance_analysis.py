import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from datetime import datetime

from src import helper

### LOGGING
logger = logging.getLogger() # creating logging object
logger.setLevel(logging.DEBUG) # setting threshold to DEBUG

log_file = datetime.now().strftime("/Users/catpillow/Documents/VTE_Analysis/doc/performance_analysis_log_%Y%m%d_%H%M%S.txt")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

### PYLINT
# pylint: disable=logging-fstring-interpolation, consider-using-enumerate

MODULE = "inferenceTraining"
BASE_PATH = "/Users/catpillow/Documents/VTE_Analysis"

# DATA ANALYSIS -------------
def get_session_performance(content):
    """this function analyses one statescript log to retrieve performance
    
    This analysis is done through by getting trial starts, trial types and using the middle number of lines
    1. The analysis starts when the first 'New Trial' is detected in a line. this signifies the start of new trial
    2. After 'New Trial' has been detected, the code then looks for an instance of 'trialType'
        this can be used to determine the trial type associated with the current trial
        
        some errors when experimenter hits summary right after 'New Trial', but should be fixed by ignoring
        lines that don't start with numbers - since doing so brings up an error
    3. For every line of just numbers and spaces, it splits the line into 3 parts
        the first part is the time
        the second part is the indication of which arm the rat is at
        the third part is the gate configuration, and not used in this analysis
        
        every unique value in the second parts (parts[1]) within a trial is stored into a set, middle_numbers
        since correct trials involve the rat going to 2 arms and incorrect trials - 1, the # of unique values in 
        middle_numbers can be used to determine whether the rat was correct or not
            rats will have '0' and '1' in the middle_numbers always, and so the maths come out to
            - 4 unique values in middle numbers -> correct
            - 3 unique values -> incorrect
            
        some errors prior due to rats peeing
            - this was solved by checking which unique values in set align with pre-determined possible arm values
        some errors with statescript log randomly registering arm when rat is not on arm
            - right now this is solved very roughly by just taking unique values in middle numbers and checking
            which ones align wiht possible arm values, then determining correctness based on those values that align
    4. 'New Trial' detected again
        previous trial ends and new trial begins. end_of_trial activated, which allows for trial analysis to see
        whether the rat was correct in the trial
    5. When the SS log reaches the last line, it automatically triggers end of trial
        right now i don't think the way i'm dealing with is right - the end of trial still valuates into
        correct or incorrect. will fix later
    6. Since I initialise the two returned arrays with np.zero, i remove the zeroes
    7. after removing zeros, I might've removed trials where the rat just didn't get any trials correct.
        I check for this by seeing if the length of correct trials is lower, and if it is, something should be changed

    Args:
        content (text file): state script log
        printTrue (boolean): prints stuff that are useful when activated

    Returns:
        final_total_trials (numpy array): array of total trials for each trial type; 0 index -> 'AB' etc
        final_correct_trials (numpy array): array of correct trials for each trial type
        each_trial_perf (bool list): list of trues and falses depending on whether the rat performed correctly for trial
    """
    
    lines = content.splitlines()
    
    # initialise variables
    middle_numbers = set() # array to hold the middle numbers
    end_of_trial = False # checks if middle_numbers should be reset
    error_trial = None
    current_trial = None
    last_line = len(lines) - 2 # this is for the last trial of a session bc no 'New Trial' to signal end of trial
    no_trials = 0 # count number of trials that has passed
    each_trial_perf = [] # store bools here
    
    # stored for graphing
    total_trials = np.zeros(10) # start out with the total number of possible trial types
    correct_trials = np.zeros(10)
    
    # to check if a rat peed
    possible_middle_numbers = {"0", "1", "2", "4", "8", "16", "32", "64", "128", "256"}
    
    for index, line in enumerate(lines):
        parts = line.split()
        correct = None
        
        if line.startswith("#") or line.strip() == "": # skip the starting comments or gaps from concat
            continue
        elif not line[0].isdigit(): # check if the first char is a number - skip if not
            # hopefully this takes cares of the weird errors wth pressng summary after new trial showed
            continue
        elif all(char.isdigit() or char.isspace() for char in line): # a normal licking line
            # check the middle numbers to determine arms that has been ran to
            if len(parts) == 3:
                middle_numbers.add(parts[1])  
        elif "New Trial" in line and "reset" not in line: # indicate start of a new trial
            end_of_trial = True
        elif end_of_trial and "trialType" in line: # this excludes 'trialType' from summaries
            try: # there were some weird errors with pressing summary right after new trial has started
                current_trial = int(line[-1]) - 1 # last char (line[-1]) is the trial type
                
                # had a weird instance of trialtype = 0, skipping that
                # 0 = trialtype 1 bc minus 1 earlier
                if current_trial < 0:
                    print(f"current trial = {current_trial} time - {parts[0]}" 
                          f"for {helper.CURRENT_RAT} on {helper.CURRENT_DAY}")
                    end_of_trial = False
                    continue
                if current_trial > 4:
                    print(f"current trial larger than 4, time - {parts[0]}")
                    
                end_of_trial = False # reset
            except Exception as e:
                print("weird error", e)
                continue
        
        if "Trial correct" in line or "Wrong choice" in line or "Error. Return home" in line:
            if "correct" in line:
                correct = True
            elif "Wrong" in line or "Error" in line:
                correct = False
            
        if index == last_line:
            end_of_trial = True
        
        # analysis when a trial has ended
        if end_of_trial and middle_numbers:
            if current_trial is None: # so it doesn't add to all the trials
                continue
            
            if len(middle_numbers) > 4: # if a rat pees in one well it might come out to be 
                # check if something dodgy is going on or if the rat peed
                if middle_numbers - possible_middle_numbers: # there is a value inside middle_numbers that's not arm
                    logging.info(f"{helper.CURRENT_RAT} peed on {helper.CURRENT_DAY} at {parts[0]}")
                    middle_numbers = middle_numbers.intersection(possible_middle_numbers)
            
            if len(middle_numbers) == 3:
                error_trial = True
            elif len(middle_numbers) == 4:
                error_trial = False
            elif len(middle_numbers) > 4:
                # if it is still above 4, something's wrong, but focus on resetting the problem for now
                error_trial = False
                logging.warning(f"something weird - middle_numbers > 4 at {parts[0]}"
                                f"happening for {helper.CURRENT_RAT} on {helper.CURRENT_DAY}")
            else:
                continue # this usually happens at the start when the rat first licks for a session or at the end
            
            # check if middle numbers align with prints of correct/wrong
            if correct is not None:
                if correct != error_trial and helper.CURRENT_RAT != "BP06":
                    logging.warning("middle numbers doesn't align with logs of correctness"
                                    f"{helper.CURRENT_RAT} on {helper.CURRENT_DAY} at {parts[0]}")
                    error_trial = correct
                elif correct != error_trial:
                    # for BP06, since his ss looks so diff from others
                    logging.warning(f"middle number is different from log correctness"
                                    f"{helper.CURRENT_RAT} on {helper.CURRENT_DAY} at {parts[0]}")
            
            # add to total trials
            total_trials[current_trial] += 1
            
            # add to correct trials if correct
            if not error_trial:
                each_trial_perf.append(True)
                correct_trials[current_trial] += 1
            else:
                each_trial_perf.append(False)
            
            middle_numbers = set() # reset
            no_trials += 1
    
    # removing the zeroes in the trial count arrays
    total_mask = total_trials != 0
    final_total_trials = total_trials[total_mask]
    
    correct_mask = correct_trials != 0
    final_correct_trials = correct_trials[correct_mask]
    
    # if # correct trials < total, there might've been a case where a rat got none of a specific trial correct
    # add 0s for that if it is the case
    if len(final_correct_trials) < len(final_total_trials):
        for i in range(len(final_total_trials)):
            # check if there is a zero in correct trials for lenth of total
            if correct_trials[i] == 0:
                final_correct_trials = np.insert(final_correct_trials, i, 0)
            
            # check if there are still any completely incorrects
            if len(final_correct_trials) < len(final_total_trials):
                continue
            else:
                break

    return final_total_trials, final_correct_trials, each_trial_perf

def get_trials_for_session(content):
    numbers = set()
    lines = content.splitlines()
    
    # trial type found
    for line in lines:
        if "#" in line:
            continue
        
        match = re.search(r"trialType = ([0-9]+)", line) # look for the number after trial types
        
        if match: # if trialType is found
            number = int(match.group(1))
            numbers.add(number)

            if number == 0:
                print(f"trial type is 0 for {helper.CURRENT_RAT} on {helper.CURRENT_DAY} for {line}")
    
    return numbers # where this only contains the numbers corresponding to trial types that are available

def trial_accuracy(content):
    # just adding session_performance and get_trial_types together
    total_trials, correct_trials, _ = get_session_performance(content)
    trial_types = get_trials_for_session(content)
    
    return trial_types, total_trials, correct_trials

def change_in_performance_trial_type(rat_performance, criterias=None):
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
    past_day_perf = None
    performance_changes = [] # array for differences in performances
    avg_changes = []
    
    # loop through days
    for day in sorted_days:
        all_performances_changes_for_day = []
        performance_for_day = rat_performance[day]
        
        # first day doesn't count as a change in performance
        if day == "Day1":
            past_day_perf = performance_for_day
            continue
        
        for trial_type_i, total_trials_i, correct_trials_i in performance_for_day:
            for trial_type_j, total_trials_j, correct_trials_j in past_day_perf:
                if trial_type_i not in trial_type_j: # skip through things that are not the same trial type
                    continue
                
                # if it's here there's a match between trial type
                trial_type = trial_type_j # save the trial type
                
                if criterias: # if criterias exist
                    # convert day into an integer
                    match = re.search(r"\d+", day) # gets first int in the string
                    if match:
                        day_int = int(match.group())
                        if "CD" in trial_type or "DE" in trial_type:
                            criteria_day = criterias[trial_type] + day_int
                        else:
                            criteria_day = criterias[trial_type]
                    
                    if criteria_day == day_int:
                        # calculate performance change
                        past_performance = correct_trials_j / total_trials_j
                        current_performance = correct_trials_i / total_trials_i
                        change = current_performance - past_performance # change in performance
                        
                        performance_changes.append({"day": day, "trial_type": trial_type, "perf_change": change})
                    elif criteria_day < day_int:
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
                #print(f'performance change - {change} on day {day} for trial type {trial_type}')
                #print(f"past performance - {past_performance}, current - {current_performance}")
        
        avg_change = np.mean(all_performances_changes_for_day)
        avg_changes.append(avg_change)
        past_day_perf = performance_for_day # update past day
    
    changes_df = pd.DataFrame(performance_changes)
        
    return changes_df, avg_changes # returns pd.DataFrame with {'day': day, 'trial_type': trial_type, 'perf_change': change}

def get_trials_available(save_path, data_structure):
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
        rat_path = os.path.join(save_path, rat, MODULE)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            print(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        # sort through the days so it's stored in ascending order
        days = os.listdir(rat_path)
        if days is not None or days != "":
            sorted_days = sorted(days, key=lambda x: int(re.sub(r"\D", "", x)) if re.sub(r"\D", "", x) else 100)
        else:
            logging.debug(f"no days for {rat}, {rat_path}")
            continue
        
        for day in sorted_days:
            # check for DS_Store
            if not os.path.isdir(os.path.join(rat_path, day)):
                continue
                
            match = re.search(r"\d+", day)
            if match:
                day_number = int(match.group())

            if day_number == 100:
                logging.error(f"day with no number for {rat}")
            
            try:
                content = data_structure[rat][day]["stateScriptLog"]
                trials_in_day = get_trials_for_session(content)
                trials_available.append({"rat": rat, "day": day_number, "trials_available": trials_in_day})
            except Exception as e:
                print(f"error {e} for rat {rat} on {day}")
    
    df = pd.DataFrame(trials_available)
    return df



# PLOTTING --------------
def plot_rat_performance(rat_performance):
    """
    Plots the performance of rats over multiple sessions
    calculates the performance as percentage of correct trials for each trial type per day
    plots performances over time, with days on the x axis and performance % on the y axis, and each line is a different trial type

    Args:
        rat_performance (dict): {days: {trial_type (str), total_trials (numpy array), correct_trials (numpy array)}}
    """
    
    performance_by_type = {}
    
    # sort days to follow chronological order
    sorted_days = sorted(rat_performance.keys(), key=lambda x: int(x[3:]))
    
    # compile perf data
    for day in sorted_days:
        for trial_type, total_trials, correct_trials in rat_performance[day]:
            performance = (correct_trials / total_trials) * 100 if total_trials else 0
            
            if trial_type not in performance_by_type:
                performance_by_type[trial_type] = [None] * len(sorted_days)
            
            day_index = sorted_days.index(day)
            performance_by_type[trial_type][day_index] = performance
    
    # plot
    plt.figure(figsize=(10, 6))
    for trial_type, performances in performance_by_type.items():
        adjusted_days = [sorted_days[i] for i, perf in enumerate(performances) if perf is not None]
        adjusted_performances = [perf for perf in performances if perf is not None]
        
        plt.plot(adjusted_days, adjusted_performances, label=trial_type, marker="o")
        
    plt.xlabel("Days")
    plt.ylabel("Performance (%)")
    plt.title("Rat Performance Over Sessions")
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_trial_accuracy(total_trials, correct_trials, trial_types):
    """
    Plots the performance of one rat for one day for every trial type

    Args:
        total_trials (numpy array): total number of trials for that trial type
        correct_trials (numpy array): the number of correct trials for that trial type
        trial_types (str array): the corresponding trial types
    """
    
    percentage_correct = (correct_trials / total_trials) * 100
    
    # adjusting trial types
    length = len(total_trials)
    trial_types = trial_types[:length]
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(trial_types, percentage_correct, color="red")
    plt.title("Trial Accuracy", fontsize=20)
    plt.ylabel("Performance (%)", fontsize=15)
    plt.xlabel("Trial Types", fontsize=15)
    plt.ylim(top=100, bottom=0)
    plt.xticks(trial_types)
    plt.show()

def plot_rat_perf_changes(rat_performance): # probably doesn't work anymore bc changes to change_in_perf_trial_type?
    """
    calculates then plots the changes in rat performance across different trial types

    Args:
        rat_performance (dict): {day: trial_type, total_trials, correct_trials}
    """
    
    performance_changes, _ = change_in_performance_trial_type(rat_performance)
    
    # extract trial_types
    trial_types = performance_changes.keys()
    
    # plotting 
    _, ax = plt.subplots()
    x_ticks = np.arange(len(trial_types))
    colormap = plt.cm.get_cmap("Pastel1", len(trial_types))
    
    # plot for each trial type
    for i, trial_type in enumerate(trial_types):
        # scatter plot
        y = performance_changes[trial_type]
        y = [item * 100 for item in y] # turn into percentage
        color = colormap(i / (len(trial_types) - 1)) if len(trial_types) > 1 else colormap(0.0)
        ax.scatter([i] * len(y), y, color=color, label=trial_type)
        
        # plot mean & std
        mean = np.mean(y)
        sem = scipy.stats.sem(y)
        
        ax.scatter(i, mean, color="black", s=100, zorder=5)
        ax.errorbar(i, mean, yerr=sem, fmt="o", color="black", ecolor="black", elinewidth=2, capsize=1)
        
        # display mean value slightly above the mean
        offset = sem + 1
        ax.text(i, mean + offset, f"{mean:.2f}", ha="center", va="bottom")
    
    # set up labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(trial_types)
    ax.set_xlabel("Trial Type")
    ax.set_ylabel("Difference in Performance in Consecutive Days (%)")
    ax.set_title("Change in Performance across trial types")
    
    ax.legend()
    plt.show()

def plot_all_rat_perf_changes(all_rats_changes, criterias = False): # currently only works with dict, but all_rats_changes has been changed to a df
    """
    plots all perf changes of all rats across different trial types

    Args:
        all_rats_changes (dict): {ratID: {trial types: performance change array}}
        criterias (bool, optional): if true, excluding data for days after criteria is reached. Defaults to False.
    
    Procedure:
    1. extract trial types from first rat's performance changes
        - I'm assuming trial types would be same for all rats here
    2. set up colormap
    3. set up figure and axis
    4. iterates over performance changes for each rat, skipping over those without data
        - displays data in percentages
    5. calculates and plots means adn stdev
    6. set up labels, titles and displays the plot
    
    """
    
    # arrays for calculating means and std later
    total = {}
    
    # extract trial_types
    for rat_ID, perf_changes_dict in all_rats_changes.items():
        trial_types = perf_changes_dict.keys()
        break
    
    # for sorting out colors
    num_rats = len(all_rats_changes)
    colormap = plt.cm.get_cmap("Pastel1", num_rats)
    
    _, ax = plt.subplots()
    x_ticks = np.arange(len(trial_types))
    
    for i, (rat_ID, perf_changes_dict) in enumerate(all_rats_changes.items()):
        # exclude rats that didn't make it to DE for now
        if len(perf_changes_dict) < 4:
            continue
        
        color = colormap(i / (num_rats - 1)) if num_rats > 1 else colormap(0.0)
        
        for j, trial_type in enumerate(trial_types):
            # plotting
            y = perf_changes_dict[trial_type]
            if y: # check that there is data
                #if any(y_val > 0.6 and y_val < -0.6 for y_val in y):
                    #print(f'past 0.6 {ratID} for {trial_type}')
                    #continue
                y = [item * 100 for item in y] # turn into percentage
                ax.scatter([j] * len(y), y, color=color, label=rat_ID if j == 0 else "")
            else:
                print(f"there is no data in y for {rat_ID} for {trial_type}")
            
            # add to an array for averages and std
            if trial_type not in total:
                total[trial_type] = []
            
            total[trial_type].extend(y)
    
    for i, trial_type in enumerate(trial_types):
        mean = np.mean(total[trial_type])
        std = np.std(total[trial_type])
        ax.scatter(i, mean, color="black", s=100, zorder=5)
        ax.errorbar(i, mean, yerr=std, fmt="o", color="black", ecolor="black", elinewidth=2, capsize=1)
        
        # display mean value slightly above the mean
        offset = std + 1
        ax.text(i, mean + offset, f"{mean:.2f}", ha="center", va="bottom")
    
    # set up graph
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(trial_types)
    ax.set_xlabel("Trial Type")
    ax.set_ylabel("Difference in Performance of Consecutive Days (%)")
    
    # check for title
    if criterias:
        ax.set_title("Change in Performance Until Reaching Criteria")
    else:
        ax.set_title("Change in Performance Across Trial Types")
    
    #ax.legend(title = 'Rat ID')
    plt.show()

def plot_days_until_criteria(all_days_until_criteria):
    # arrays for calculating means and std later
    total = {}
    
    # extract trial_types
    for rats, days_dict in all_days_until_criteria.items():
        trial_types = days_dict.keys()
        break
    
    # for sorting out colors
    num_rats = len(all_days_until_criteria)
    colormap = plt.cm.get_cmap("Pastel1", num_rats)
    
    _, ax = plt.subplots()
    x_ticks = np.arange(len(trial_types))
    
    for i, (rats, days_dict) in enumerate(all_days_until_criteria.items()):
        # exclude rats that didn't make it to DE for now
        if len(days_dict) < 4:
            continue
        
        color = colormap(i / (num_rats - 1)) if num_rats > 1 else colormap(0.0)
        
        for j, trial_type in enumerate(trial_types):
            if trial_type == 5:
                continue
            
            try:
                y = days_dict[trial_type]
            except KeyError:
                continue
            
            ax.scatter(j, y, color=color, label=rats if j == 0 else"")
            
            # add to an array for averages and std
            if trial_type not in total:
                total[trial_type] = []
            
            total[trial_type].append(y)
    
    for i, trial_type in enumerate(trial_types):
        #y = sum[trial_type]
        #ax.scatter([j] * len(y), y, color=color, label=ratID if j == 0 else "") # plot
        
        if trial_type == 5:
            continue
        
        mean = np.mean(total[trial_type])
        std = np.std(total[trial_type])
        ax.scatter(i, mean, color="black", s=100, zorder=5)
        ax.errorbar(i, mean, yerr=std, fmt="o", color="black", ecolor="black", elinewidth=2, capsize=1)
    
    # set up graph
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(trial_types)
    ax.set_xlabel("Trial Type")
    ax.set_ylabel("Number of Days Until Criteria")
    ax.set_title("Days until Criteria by Trial Type (75%)")
    #ax.legend(title="Rat ID")
    plt.show()



# TRAVERSAL FUNCTIONS ----------
def rat_performance_one_session(data_structure, rat_ID, day):
    # plots trial accuracy for one rat and one day
    ss_data = data_structure[rat_ID][day]["stateScriptLog"]
    trial_types, total_trials, correct_trials = trial_accuracy(ss_data)
    
    plot_trial_accuracy(total_trials, correct_trials, trial_types)

def rat_performance_over_sessions(data_structure, rat_ID):
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
    
    if rat_ID not in data_structure:
        raise helper.ExpectationError("ratID", rat_ID)
    
    rat_performance = pd.DataFrame(columns=["rat", "day", "trial_type", "total_trials", "correct_trials"])
    for day_folder, contents in data_structure[rat_ID].items():
        if day_folder == ".DS_Store" or "pkl" in day_folder: # skip .DS_Store
            continue
        
        helper.update_day(day_folder)
        
        ss_data = contents["stateScriptLog"]
        if ss_data is None:
            print(f"ss_data is None Type for {rat_ID} on {day_folder}")
            continue
        
        try:
            trial_types_set, total_trials, correct_trials = trial_accuracy(ss_data)
        except Exception as e:
            print(f"error {e} for {rat_ID} on {day_folder}")
        trial_types = sorted(trial_types_set)

        # exclude days where there were errors and didn't even have AB and BC
        # and days i'm too lazy to decode right now
        length = len(total_trials)
        if length < 2 or length > 5:
            logging.error(f"total trial length is is {length} for {rat_ID} on {day_folder}")
            logging.error(f"total trials - {total_trials}")
            logging.error(f"correct trials - {correct_trials}")
            continue
        
        for i in range(len(total_trials)):
            try:
                match = re.search(r"\d+", day_folder) # gets first int in the string
                if match:
                    day_int = int(match.group())
                rat_performance.loc[len(rat_performance)] = [rat_ID, day_int, trial_types[i], total_trials[i], correct_trials[i]]
            except IndexError as error: # happens if lengths are incongruent
                logging.error(f"Error for rat {rat_ID} for {day_folder}: {error}")
                logging.error(f"trial types - {trial_types}")
                logging.error(f"total trials - {total_trials}")
                logging.error(f"correct trials - {correct_trials}")
    
    #plot_rat_performance(rat_performance)
    return rat_performance # this returns df for one rat

def create_all_rats_performance(data_structure=None):
    """
    loops through all the rats to create csv files from ss data

    Args:
        data_structure (dict): all the data
        save_path (str, optional): save path if desired. Defaults to None.

    Returns:
        pd.DataFrames: {'rat', 'day', 'trial_type', 'total_trials', 'correct_trials'}
                        where trial_type are the corresponding numbers
    """
    save_path = os.path.join(BASE_PATH, "processed_data", "rat_performance.csv")
    if data_structure is not None:
        dataframes = []
        
        for rat_ID in data_structure:
            helper.update_rat(rat_ID)
            rat_performance = rat_performance_over_sessions(data_structure, rat_ID)
            dataframes.append(rat_performance)
        all_rats_performances = pd.concat(dataframes, ignore_index=True)
        
        # save dataframe
        all_rats_performances.to_csv(save_path)
    else:
        all_rats_performances = pd.read_csv(save_path)
    
    return all_rats_performances

def create_all_perf_changes(all_rats_performances):
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
    save_path = os.path.join(BASE_PATH, "processed_data", "performance_changes.csv")
    perf_changes_df.to_csv(save_path)
    
    return perf_changes_df

def create_all_perf_changes_by_trials(all_rats_performances):
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
    save_path = os.path.join(BASE_PATH, "processed_data", "performance_changes_by_trial.csv")
    perf_changes_df.to_csv(save_path)
    return perf_changes_df

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
        
        performance_changes = change_in_performance_trial_type(rat_performance, criteria_days)
        all_rats_changes[rat_ID] = performance_changes
    
    plot_all_rat_perf_changes(all_rats_changes, criterias = True)

def get_days_since_new_arm(save_path, data_structure):
    """
    returns a DataFrame including the number of days since new arm was introduced

    Args:
        save_path (str): where the files are saved
        data_structure (dict): dictionary with all the data

    Returns:
        pd.DataFrame: {'rat': rat, 'day': day, 'trials_available': trials for the day, 
                       'arm_added': bool for whether new arm was added, 'days_since_new_arm': self-explanatory}
    """
    
    trials_available = get_trials_available(save_path, data_structure)
    new_df_rows = []
    
    rats = trials_available.groupby("rat")
    for rat, group in rats:
        previous_number_of_trials = 0
        days_since_new_arm = 0
        decrease_present = None # get the number of trials available right before decrease
        
        sorted_by_day = group.sort_values(by="day") # continue by day
        for _, row in sorted_by_day.iterrows():
            day = row["day"]
            trials_for_day = row["trials_available"]
            
            number_of_trials = len(trials_for_day)
            if previous_number_of_trials == 0: # first day
                arm_added = True
                days_since_new_arm = 0
            elif previous_number_of_trials < number_of_trials: # arm added
                arm_added = True
                days_since_new_arm = 0
            elif previous_number_of_trials == number_of_trials:
                arm_added = False
                days_since_new_arm += 1
            else: # decrease in trials, something wacky going on
                arm_added = False
                days_since_new_arm += 1
                decrease_present = previous_number_of_trials # for checking if # trials increase in future
                logging.warning(f"decrease in # trials for {rat} on {day}")
            
            if decrease_present is not None:
                if number_of_trials > decrease_present: # if rat gets a trial never experienced before
                    decrease_present = None
            else:
                previous_number_of_trials = number_of_trials
                
            new_df_rows.append({"rat": rat,
                                "day": day,
                                "trials_available": trials_for_day,
                                "arm_added": arm_added,
                                "days_since_new_arm": days_since_new_arm})
        
    new_arms_df = pd.DataFrame(new_df_rows)
    path = os.path.join(BASE_PATH, "processed_data", "days_since_new_arm.csv")
    new_arms_df.to_csv(path)
    
    return new_arms_df
