import os
import re
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from datetime import datetime

from src import helper_functions

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
CURRENT_RAT = ""
CURRENT_DAY = ""
BASE_PATH = "/Users/catpillow/Documents/VTE_Analysis"

def update_rat(rat):
    global CURRENT_RAT
    CURRENT_RAT = rat

def update_day(day):
    global CURRENT_DAY
    CURRENT_DAY = day

# SAVING & LOADING -----------
def pickle_rat_performance(rat_performance, save_path):
    """this function saves rat_performance, a dictionary of {ratID: performance} into a pickle file"""
    # save in folder
    pickle_path = save_path + "/rat_performance.pkl"
    with open(pickle_path, "wb") as fp:
        pickle.dump(rat_performance, fp)

def unpickle_rat_performance(save_path):
    """this function loads the pickle file for rat performance, a dictionary of {ratID: performance}"""
    
    all_rats_performances = {}
        
    for rat_folder in os.listdir(save_path): # loop for each rat
        rat_path = os.path.join(save_path, rat_folder)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            print(f"Skipping over non-directory folder: {rat_path}")
            continue
            
        pickle_path = rat_path + "/rat_performance.pkl"
        
        with open(pickle_path, "rb") as fp:
            rat_performance = pickle.load(fp)
        
        all_rats_performances[rat_folder] = rat_performance
    
    return all_rats_performances


# DATA ANALYSIS -------------
def trial_perf_for_session(content):
    """
    generates an ordered list of whether each trial resulted in a correct or incorrect choice

    Args:
        content (str): statescript log

    Returns:
        str array: either 'wrong' for incorrect choices or 'correct' for correct choices. should be in order
    """
    
    lines = content.splitlines()
    
    performance = [] # storing whether the trial resulted in a correct or incorrect choice
    
    for line in lines:
        if "#" in line: # skip initial comments
            continue
        
        if "Wrong" in line:
            performance.append("wrong")
        elif "correct" in line:
            performance.append("correct")
    
    return performance
        
def session_performance(content):
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
    """
    
    lines = content.splitlines()
    
    # temporary variables
    middle_numbers = set() # array to hold the middle numbers
    end_of_trial = False # checks if middle_numbers should be reset
    error_trial = None
    current_trial = None
    last_line = len(lines) - 2 # this is for the last trial of a session bc no 'New Trial' to signal end of trial
    no_trials = 0 # count number of trials that has passed
    
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
                    print(f"current trial = {current_trial} time - {parts[0]} for {CURRENT_RAT} on {CURRENT_DAY}")
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
                    logging.info(f"{CURRENT_RAT} peed on {CURRENT_DAY} at {parts[0]}")
                    middle_numbers = middle_numbers.intersection(possible_middle_numbers)
            
            if len(middle_numbers) == 3:
                error_trial = True
            elif len(middle_numbers) == 4:
                error_trial = False
            elif len(middle_numbers) > 4:
                # if it is still above 4, something's wrong, but focus on resetting the problem for now
                error_trial = False
                logging.warning(f"something weird - middle_numbers > 4 at {parts[0]}"
                                f"happening for {CURRENT_RAT} on {CURRENT_DAY}")
            else:
                continue # this usually happens at the start when the rat first licks for a session or at the end
            
            # check if middle numbers align with prints of correct/wrong
            if correct is not None:
                if correct != error_trial and CURRENT_RAT != "BP06":
                    logging.warning("middle numbers doesn't align with logs of correctness"
                                    f"{CURRENT_RAT} on {CURRENT_DAY} at {parts[0]}")
                    error_trial = correct
                elif correct != error_trial:
                    # for BP06, since his ss looks so diff from others
                    logging.warning(f"middle number is different from log correctness"
                                    f"{CURRENT_RAT} on {CURRENT_DAY} at {parts[0]}")
            
            # add to total trials
            total_trials[current_trial] += 1
            
            # add to correct trials if correct
            if not error_trial:
                correct_trials[current_trial] += 1
            
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

    return final_total_trials, final_correct_trials

def trial_accuracy(content):
    # just adding session_performance and get_trial_types together
    total_trials, correct_trials = session_performance(content)
    trial_types = get_trials_for_session(content)
    
    return trial_types, total_trials, correct_trials

def create_all_perf_dictionary(all_rat_performances):
    """gets avg and std, makes a dictionary of performance instead of total/correct trials
        doesn't take into account trial type
        This is for plotting showing ABC, ABCD, ABCDE
        ONLY WORKS IF 9 DAYS
    
    1. loops through rats
    2. loops through days in order
    3. calculate performance for each trial type, stores it
    4. gets avg & std

    Args:
        all_rat_performances (dictionary): {ratID: {day: performance}} where performance is an array with
                                            trial types, total trials, and correct trials for each trial type

    Returns:
        all_performances (dictionary): {ratID: performance} where performance is overall performance per day
        avg (float array): average for each day
        std (float array): std for each day
    """
    
    all_performances = {}
    totals = [[] for _ in range(9)] # currently assuming ABCDE
    
    # loop over rats - creating dictionary of all_performances
    for rat_ID, rat_performance in all_rat_performances.items():
        # loop over days
        sorted_days = sorted(rat_performance.keys(), key = lambda x: int(x[3:])) # sort by after 'day'
        rats_performance = []
        num_days = 0
        print(f'rat - {rat_ID}')
        
        for day in sorted_days:
            performance_for_day = rat_performance[day] # get performance for day
            
            # loop over each trial type - right now just taking sum w/o caring about trial type
            total_trials_in_day = 0
            correct_trials_in_day = 0
            
            for _, total_trials, correct_trials in performance_for_day:
                total_trials_in_day += total_trials
                correct_trials_in_day += correct_trials
            
            performance = correct_trials_in_day / total_trials_in_day # calculate sum perf in day
            rats_performance.append(performance)
            
            # get totals for calculating avg
            totals[num_days].append(performance)
            num_days += 1
        
        all_performances[rat_ID] = rats_performance
        
        print(f'num days - {num_days}')

    # calculating avg
    avg = [np.mean(day_totals) if day_totals else 0 for day_totals in totals]
    sem = [scipy.stats.sem(day_totals) if day_totals else 0 for day_totals in totals]
    
    return all_performances, avg, sem

def create_trial_type_dictionary(all_rat_performances):
    """gets avg and std, makes a dictionary of performance instead of total/correct trials
        doesn't take into account trial type
        This is for plotting showing AB, BC, CD, De
        ONLY WORKS IF 9 DAYS
    
    1. loops through rats
    2. loops through days in order
    3. adds each performance of a day into the array for the trial type
        done based on days, such that it takes AB & BC from days 1-3, CD from days 4-6 and DE from days 7-9
    4. gets avg & std

    Args:
        all_rat_performances (_dictionary_): _{ratID: {day: performance}} where performance is an array with
                                            trial types, total trials, and correct trials for each trial type_

    Returns:
        all_performances (_dictionary_): _{ratID: performance} where performance is overall performance per day_
        avg (_float array_): average for each day
        std (_float array_): std for each day
    """
    
    all_performances = {}
    totals = [[] for _ in range(12)] # currently assuming ABCDE
    
    # loop over rats - creating dictionary of all_performances
    for ratID, rat_performance in all_rat_performances.items():
        # loop over days
        sorted_days = sorted(rat_performance.keys(), key = lambda x: int(x[3:])) # sort by after 'day'
        rats_performance = []
        alt_rats_performance = [] # for BC
        num_bins = 0 # for trial type, this is more of a representative of bin number
        num_days = 0
        
        for day in sorted_days:
            performance_for_day = rat_performance[day] # get performance for day
            
            # loop over each trial type - right now just taking sum w/o caring about trial type
            total_trials_in_day = 0
            correct_trials_in_day = 0
            alt_total_trials_in_day = 0 # just bc AB & BC both collected during first 3 days
            alt_correct_trials_in_day = 0 # bc AB & BC
            
            # add if it is the correct trial type for the day it is
            for trial_type, total_trials, correct_trials in performance_for_day:
                if 'AB' in trial_type and num_days < 3:
                    total_trials_in_day += total_trials
                    correct_trials_in_day += correct_trials
                elif 'BC' in trial_type and num_days < 3:
                    alt_total_trials_in_day += total_trials
                    alt_correct_trials_in_day += correct_trials
                elif 'CD' in trial_type and num_days < 6 and num_days > 2:
                    total_trials_in_day += total_trials
                    correct_trials_in_day += correct_trials
                elif 'DE' in trial_type and num_days < 9 and num_days > 5:
                    total_trials_in_day += total_trials
                    correct_trials_in_day += correct_trials
                else:
                    logging.error('trial types not found')
            
            performance = correct_trials_in_day / total_trials_in_day # calculate sum perf in day
            rats_performance.append(performance)
            
            # get totals for calculating avg
            totals[num_bins].append(performance)
            num_bins += 1
            
            # for BC
            if num_bins == 3:
                performance = alt_correct_trials_in_day / alt_total_trials_in_day
                alt_rats_performance.append(performance)
                totals[num_bins + 2].append(performance)
                rats_performance.extend(alt_rats_performance)
                num_bins += 3
            elif alt_total_trials_in_day != 0:
                performance = alt_correct_trials_in_day / alt_total_trials_in_day
                alt_rats_performance.append(performance)
                totals[num_bins + 2].append(performance)
            
            num_days += 1
        
        all_performances[ratID] = rats_performance

    # calculating avg
    avg = [np.mean(day_totals) if day_totals else 0 for day_totals in totals]
    std = [np.std(day_totals) if day_totals else 0 for day_totals in totals]
    
    return all_performances, avg, std

def change_in_performance_trial_type(rat_performance, criterias = None):
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
        if day == 'Day1':
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
                    match = re.search(r'\d+', day) # gets first int in the string
                    if match:
                        day_int = int(match.group())
                        if 'CD' in trial_type or 'DE' in trial_type:
                            criteria_day = criterias[trial_type] + day_int
                        else:
                            criteria_day = criterias[trial_type]
                    
                    if criteria_day == day_int:
                        # calculate performance change
                        past_performance = correct_trials_j / total_trials_j
                        current_performance = correct_trials_i / total_trials_i
                        change = current_performance - past_performance # change in performance
                        
                        performance_changes.append({'day': day, 'trial_type': trial_type, 'perf_change': change})
                    elif criteria_day < day_int:
                        continue # skip if criteria has already been reached
                    else: # this would be if the current day is lower than the criteria
                        # calculate performance change
                        past_performance = correct_trials_j / total_trials_j
                        current_performance = correct_trials_i / total_trials_i
                        change = current_performance - past_performance # change in performance
                        
                        performance_changes.append({'day': day, 'trial_type': trial_type, 'perf_change': change})
                else:
                    # calculate performance change
                    past_performance = correct_trials_j / total_trials_j
                    current_performance = correct_trials_i / total_trials_i
                    change = current_performance - past_performance # change in performance
                    
                    performance_changes.append({'day': day, 'trial_type': trial_type, 'perf_change': change})
                    all_performances_changes_for_day.append(change)
                #print(f'performance change - {change} on day {day} for trial type {trial_type}')
                #print(f"past performance - {past_performance}, current - {current_performance}")
        
        avg_change = np.mean(all_performances_changes_for_day)
        avg_changes.append(avg_change)
        past_day_perf = performance_for_day # update past day
    
    changes_df = pd.DataFrame(performance_changes)
        
    return changes_df, avg_changes # returns pd.DataFrame with {'day': day, 'trial_type': trial_type, 'perf_change': change}

def get_trials_for_session(content):
    numbers = set()
    lines = content.splitlines()
    
    # trial type found
    for line in lines:
        if '#' in line:
            continue
        
        match = re.search(r'trialType = ([0-9]+)', line) # look for the number after trial types
        
        if match: # if trialType is found
            number = int(match.group(1))
            numbers.add(number)

            if number == 0:
                print(f"trial type is 0 for {CURRENT_RAT} on {CURRENT_DAY} for {line}")
    
    return numbers # where this only contains the numbers corresponding to trial types that are available
    


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
    sorted_days = sorted(rat_performance.keys(), key = lambda x: int(x[3:]))
    
    # compile perf data
    for day in sorted_days:
        for trial_type, total_trials, correct_trials in rat_performance[day]:
            performance = (correct_trials / total_trials) * 100 if total_trials else 0
            
            if trial_type not in performance_by_type:
                performance_by_type[trial_type] = [None] * len(sorted_days)
            
            day_index = sorted_days.index(day)
            performance_by_type[trial_type][day_index] = performance
    
    # plot
    plt.figure(figsize = (10, 6))
    for trial_type, performances in performance_by_type.items():
        adjusted_days = [sorted_days[i] for i, perf in enumerate(performances) if perf is not None]
        adjusted_performances = [perf for perf in performances if perf is not None]
        
        plt.plot(adjusted_days, adjusted_performances, label = trial_type, marker = 'o')
        
    plt.xlabel('Days')
    plt.ylabel('Performance (%)')
    plt.title('Rat Performance Over Sessions')
    plt.legend()
    plt.xticks(rotation = 45)
    
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
    print(percentage_correct)
    
    # adjusting trial types
    length = len(total_trials)
    trial_types = trial_types[:length]
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(trial_types, percentage_correct, color='red')
    plt.title('Trial Accuracy', fontsize = 20)
    plt.ylabel('Performance (%)', fontsize = 15)
    plt.xlabel('Trial Types', fontsize = 15)
    plt.ylim(top=100, bottom=0)
    plt.xticks(trial_types)
    plt.show()

def plot_all_rat_performances(all_rat_performances, plot_trial_types = False): # currently only works with ABCDE
    """
    plots the performance of all rats over multiple days

    Args:
        all_rat_performances (dict): {rat: performance arrays (either 9 or 12 long)}
        plot_trial_types (bool, optional): if True, then plot the performance for AB, BC, CD, DE instead of ABC, ABCD, ABCDE. Defaults to False.
    
    Procedure:
    1. checks what type of plot it is
        - either 12 bins for AB, BC, CD, DE
        - or 9 bins for ABC, ABCD, ABCDE
    2. create time array representing each day
    3. initialises figure size and plots
        - including setting up colormap for colours
    4. iterates over each rat's performance data, considering triplets together (for each trial type)
        - doing so in triplets such that the triplet's lines are connected, but separated from other triplets
        - then separately plot all the averages and standard deviations in a similar fashion
    5. set up x ticks and bin names, such that on the bottom the order of triplets is displayed and on the top the trial type is
    6. set up color for the background so a different color is used for each trial type
    7. label axes and title
    8. save plot as 'PerformanceFigure.png' and display the plot
    """
    
    # check what type of plot this is
    if plot_trial_types: # this still assumes ABCDE
        bins = 12
        all_performances, averages, std = create_trial_type_dictionary(all_rat_performances)
    else:
        bins = 9
        all_performances, averages, std = create_all_perf_dictionary(all_rat_performances) # store with {ratID:performance array}- should have 9 arrays per rat for each day
    
    # x bins
    time = np.arange(1, bins + 1, 1) # for each day
    
    # figure
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    
    # Setup for colors
    colormap = plt.cm.get_cmap('Set1')
    colormap_background = plt.cm.get_cmap('Pastel2')
    
    # plot pairs
    num_rat = 0
    
    for rat, performances in all_performances.items():
        # check that it is 9/12 items
        if len(performances) != bins:
            continue
        
        # create triplets for ABC/D/E or AB/BC/CD/DE
        three_days_perf = [] # depending on whether trial type or not, this can also be three_bin_perf technically
        count = 0
        
        for index, performance in enumerate(performances):
            if count < 2:
                three_days_perf.append(performance)
                count += 1
            else:
                three_days_perf.append(performance)
                print('inside else statement in all-perfor')
                print(time[(index - 2):(index + 1)])
                
                # plot
                ax1.plot(time[(index - 2):(index + 1)], three_days_perf,
                        label=rat, linestyle='-', marker=None, markersize=5,
                        color=colormap(num_rat % colormap.N), alpha=0.3)
                
                # reset
                count = 0
                three_days_perf = []
        
        num_rat += 1 # this is just for consistent coloring
        
    # plot average + std
    count = 0
    three_day_avg = []
    
    for index, average in enumerate(averages):
        if count < 2:
            three_day_avg.append(average)
            count += 1
        else:
            three_day_avg.append(average)
            
            ax1.errorbar(time[(index - 2):(index + 1)], three_day_avg,
                        yerr=std[(index - 2):(index + 1)], linestyle='-', marker='o', markersize=5, color='black', alpha=1,
                        capsize=5, ecolor='gray', elinewidth=2)

            # reset
            count = 0
            three_day_avg = []

    # Setup x ticks
    x_ticks = time
    x_labels = ["Day 1", "Middle", "Criteria"]
    x_labels_ABCDE = x_labels * int(bins / 3)

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels_ABCDE)
    ax1.tick_params(axis='x', labelsize=12)

    ax1.set_xlim(0.5, bins + 0.5)
    
    # Setup bins
    ax2 = ax1.twiny()

    if plot_trial_types:
        bin_ticks = [2, 5, 8, 11]
        bin_labels = ["AB", "BC", "CD", "DE"]
    else:
        bin_ticks = [2, 5, 8]
        bin_labels = ["ABC", "ABCD", "ABCDE"]

    ax2.set_xticks(bin_ticks)
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlim(ax1.get_xlim())
    ax2.tick_params(axis='x', labelsize=12)

    #chance level line
    plt.axhline(y = 0.5, color='blue', linestyle='dotted')
    
    # ylim
    plt.ylim(0, 1)

    #setup for colours
    for i in range(bins):
        plt.axvspan(i*3 + 0.5, (i*3)+3.5, facecolor = colormap_background(i % colormap_background.N), alpha = 0.3)

    #labels
    ax1.set_title("Performance in Transitive Inference (TI) Task", fontsize=15, weight='bold')
    ax1.set_ylabel("Proportion of Correct Trials", fontsize=15)
    plt.tick_params(axis='y', labelsize=12)

    #show figure
    plt.tight_layout()
    plt.savefig('PerformanceFigure.png', dpi=300)
    plt.show()
 
def plot_rat_perf_changes(rat_performance):
    """
    calculates then plots the changes in rat performance across different trial types

    Args:
        rat_performance (dict): {day: trial_type, total_trials, correct_trials}
    """
    
    performance_changes, _ = change_in_performance_trial_type(rat_performance)
    #print(performance_changes)
    
    # extract trial_types
    trial_types = performance_changes.keys()
    
    # plotting 
    fig, ax = plt.subplots()
    x_ticks = np.arange(len(trial_types))
    colormap = plt.cm.get_cmap('Pastel1', len(trial_types))
    
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
        
        ax.scatter(i, mean, color = 'black', s = 100, zorder = 5)
        ax.errorbar(i, mean, yerr = sem, fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 2, capsize = 1)
        
        # display mean value slightly above the mean
        offset = sem + 1
        ax.text(i, mean + offset, f'{mean:.2f}', ha = 'center', va = 'bottom')
    
    # set up labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(trial_types)
    ax.set_xlabel('Trial Type')
    ax.set_ylabel('Difference in Performance in Consecutive Days (%)')
    ax.set_title('Change in Performance across trial types')
    
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
    sum = {}
    
    # extract trial_types
    for ratID, perf_changes_dict in all_rats_changes.items():
        trial_types = perf_changes_dict.keys()
        break
    
    # for sorting out colors
    num_rats = len(all_rats_changes)
    colormap = plt.cm.get_cmap('Pastel1', num_rats)
    
    _, ax = plt.subplots()
    x_ticks = np.arange(len(trial_types))
    
    for i, (ratID, perf_changes_dict) in enumerate(all_rats_changes.items()):
        #print(ratID)
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
                ax.scatter([j] * len(y), y, color=color, label=ratID if j == 0 else "")
            else:
                print(f'there is no data in y for {ratID} for {trial_type}')
            
            # add to an array for averages and std
            #print(trial_type)
            if trial_type not in sum:
                sum[trial_type] = []
            
            sum[trial_type].extend(y)
    
    for i, trial_type in enumerate(trial_types):
        mean = np.mean(sum[trial_type])
        std = np.std(sum[trial_type])
        ax.scatter(i, mean, color = 'black', s = 100, zorder = 5)
        ax.errorbar(i, mean, yerr = std, fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 2, capsize = 1)
        
        # display mean value slightly above the mean
        offset = std + 1
        ax.text(i, mean + offset, f'{mean:.2f}', ha = 'center', va = 'bottom')
    
    # set up graph
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(trial_types)
    ax.set_xlabel('Trial Type')
    ax.set_ylabel('Difference in Performance of Consecutive Days (%)')
    
    # check for title
    if criterias:
        ax.set_title('Change in Performance Until Reaching Criteria')
    else:
        ax.set_title('Change in Performance Across Trial Types')
    
    #ax.legend(title = 'Rat ID')
    plt.show()
    
def plot_days_until_criteria(all_days_until_criteria):
    # arrays for calculating means and std later
    sum = {}
    
    # extract trial_types
    for ratID, days_dict in all_days_until_criteria.items():
        trial_types = days_dict.keys()
        break
    
    # for sorting out colors
    num_rats = len(all_days_until_criteria)
    colormap = plt.cm.get_cmap('Pastel1', num_rats)
    
    fig, ax = plt.subplots()
    x_ticks = np.arange(len(trial_types))
    
    for i, (ratID, days_dict) in enumerate(all_days_until_criteria.items()):
        #print(ratID)
        # exclude rats that didn't make it to DE for now
        if len(days_dict) < 4:
            continue
        
        color = colormap(i / (num_rats - 1)) if num_rats > 1 else colormap(0.0)
        
        for j, trial_type in enumerate(trial_types):
            #print(trial_type)
            # plotting
            y = days_dict[trial_type]
            #print(y)
            
            ax.scatter(j, y, color=color, label=ratID if j == 0 else"")
            
            # add to an array for averages and std
            #print(trial_type)
            if trial_type not in sum:
                sum[trial_type] = []
            
            sum[trial_type].append(y)
    
    for i, trial_type in enumerate(trial_types):
        #y = sum[trial_type]
        #ax.scatter([j] * len(y), y, color=color, label=ratID if j == 0 else "") # plot
        
        mean = np.mean(sum[trial_type])
        std = np.std(sum[trial_type])
        ax.scatter(i, mean, color = 'black', s = 100, zorder = 5)
        ax.errorbar(i, mean, yerr = std, fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 2, capsize = 1)
    
    # set up graph
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(trial_types)
    ax.set_xlabel('Trial Type')
    ax.set_ylabel('Number of Days Until Criteria')
    ax.set_title('Days until Criteria by Trial Type (80%)')
    ax.legend(title = 'Rat ID')
    plt.show()



# TRAVERSAL FUNCTIONS ----------
def rat_performance_one_session(data_structure, ratID, day):
    # plots trial accuracy for one rat and one day
    ss_data = data_structure[ratID][day]['stateScriptLog']
    trial_types, total_trials, correct_trials = trial_accuracy(ss_data)
    
    plot_trial_accuracy(total_trials, correct_trials, trial_types)

def rat_performance_over_sessions(data_structure, ratID):
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
    
    if ratID not in data_structure:
        raise helper_functions.ExpectationError("ratID", ratID)
    
    rat_performance = pd.DataFrame(columns=["rat", "day", "trial_type", "total_trials", "correct_trials"])
    for day_folder, contents in data_structure[ratID].items():
        if day_folder == ".DS_Store" or "pkl" in day_folder: # skip .DS_Store
            continue
        
        update_day(day_folder)
        
        ss_data = contents["stateScriptLog"]
        if ss_data is None:
            print(f"ss_data is None Type for {ratID} on {day_folder}")
            continue
        
        try:
            trial_types_set, total_trials, correct_trials = trial_accuracy(ss_data)
        except Exception as e:
            print(f"error {e} for {ratID} on {day_folder}")
        trial_types = sorted(trial_types_set)

        # exclude days where there were errors and didn't even have AB and BC
        # and days i'm too lazy to decode right now
        length = len(total_trials)
        if length < 2 or length > 5:
            logging.error(f'total trial length is is {length} for {ratID} on {day_folder}')
            logging.error(f'total trials - {total_trials}')
            logging.error(f'correct trials - {correct_trials}')
            continue
        
        for i in range(len(total_trials)):
            try:
                match = re.search(r'\d+', day_folder) # gets first int in the string
                if match:
                    day_int = int(match.group())
                rat_performance.loc[len(rat_performance)] = [ratID, day_int, trial_types[i], total_trials[i], correct_trials[i]]
            except IndexError as error: # happens if lengths are incongruent
                logging.error(f'Error for rat {ratID} for {day_folder}: {error}')
                logging.error(f'trial types - {trial_types}')
                logging.error(f'total trials - {total_trials}')
                logging.error(f'correct trials - {correct_trials}')
    
    #plot_rat_performance(rat_performance)
    return rat_performance # this returns df for one rat

def create_all_rats_performance(data_structure):
    """
    loops through all the rats to create pickle files from ss data

    Args:
        data_structure (dict): all the data
        save_path (str, optional): save path if desired. Defaults to None.

    Returns:
        pd.DataFrames: {'rat', 'day', 'trial_type', 'total_trials', 'correct_trials'}
                        where trial_type are the corresponding numbers
    """
    dataframes = []
    
    for ratID in data_structure:
        update_rat(ratID)
        rat_performance = rat_performance_over_sessions(data_structure, ratID)
        dataframes.append(rat_performance)
    all_rats_performances = pd.concat(dataframes, ignore_index=True)
    
    # save dataframe
    save_path = os.path.join(BASE_PATH, "processed_data", "rat_performance.csv")
    all_rats_performances.to_csv(save_path)
    
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
        sorted_rat_data = rat_group.sort_values(by='day')
        day_data = sorted_rat_data.groupby('day')
        
        rat_performance = []
        for day, day_group in day_data:
            day_performance = day_group['correct_trials'] / day_group['total_trials']
            day_performance = day_performance.mean() # get mean to average across trial types
            rat_performance.append(day_performance)
            all_rat_day_pairs.append((rat, day))
        
        rat_performance_series = pd.Series(rat_performance)
        perf_changes = rat_performance_series.diff()
        all_perf_changes.append(perf_changes)
    perf_changes_series = pd.concat(all_perf_changes, ignore_index=True)
    
    # create new dataframe
    perf_changes_df = pd.DataFrame(all_rat_day_pairs, columns=['rat', 'day'])
    perf_changes_df['perf_change'] = perf_changes_series.reset_index(drop=True)
    
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
            trial_performance = trial_group["correct_trials"] / trial_group["total_trials"]
            perf_change_in_trial = trial_performance.diff()
            for i, (_, row) in enumerate(trial_group.iterrows()):
                if i == 0:
                    all_rat_perf.append({"rat": rat, "day":row["day"], "trial_type":trial_type, "perf_change": trial_performance.iloc[0] - 0.5})
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
        all_rats_performances (dict): {rat: performance array}

    Returns:
        dict: {rat: {trial_type: day}} where day is the day in which the rat reached criteria
    """
    
    all_days_until_criteria = {}
    
    for ratID, rat_performance in all_rats_performances.items():
        sorted_days = sorted(rat_performance.keys(), key = lambda x: int(x[3:])) # sort by after 'day'
        
        # each trial type
        AB_learned = 0
        BC_learned = 0
        CD_learned = 0
        DE_learned = 0
        CD_start = None
        DE_start = None
        day_learned = {}
        
        for day in sorted_days:
            performance_for_day = rat_performance[day]
            
            match = re.search(r'\d+', day) # gets first int in the string
            if match:
                day_int = int(match.group())
            
            for trial_type, total_trials, correct_trials in performance_for_day:
                # find day where trial_types get introduced
                if 'CD' in trial_type and not CD_start: # if CD_start is still None & in trialtype
                    CD_start = day_int
                elif 'DE' in trial_type and not DE_start:
                    DE_start = day_int
                
                performance = correct_trials / total_trials
                
                if performance >= 0.8: # reached criteria
                    if AB_learned < 2 and 'AB' in trial_type and total_trials > 6.0: # haven't reached criteria yet but got >0.8
                        # set total trials above 5 because oto few trials probably shouldn't be counted - happend w bp19
                        AB_learned += 1
                    
                    if BC_learned < 2 and 'BC' in trial_type and total_trials > 6.0:
                        BC_learned += 1

                    if CD_learned < 2 and 'CD' in trial_type:
                        CD_learned += 1
                        
                    if DE_learned < 2 and 'DE' in trial_type:
                        DE_learned += 1
                    
                    if AB_learned == 2: # reached criteria
                        AB_learned += 1 # so this if statement doesn't get repeatedly triggered
                        day_learned[trial_type] = day_int
                    
                    if BC_learned == 2:
                        BC_learned += 1
                        day_learned[trial_type] = day_int
                        
                    if CD_learned == 2:
                        CD_learned += 1
                        day_diff = day_int - CD_start # minus the day the rat first started learning a pair
                        day_learned[trial_type] = day_diff
                    
                    if DE_learned == 2:
                        DE_learned += 1
                        day_diff = day_int - DE_start
                        day_learned[trial_type] = day_diff
                else: # failed
                    if AB_learned < 2 and AB_learned > 0 and 'AB' in trial_type: # haven't reached criteria yet but got <0.8
                        AB_learned -= 1
                    
                    if BC_learned < 2 and BC_learned > 0 and 'BC' in trial_type:
                        BC_learned -= 1

                    if CD_learned < 2 and CD_learned > 0 and 'CD' in trial_type:
                        CD_learned -= 1
                        
                    if DE_learned < 2 and DE_learned > 0 and 'DE' in trial_type:
                        DE_learned -= 1
        
        if AB_learned and BC_learned and CD_learned and DE_learned:
            all_days_until_criteria[ratID] = day_learned
        else:
            print(f'not all pairs learned - {ratID} {AB_learned} {BC_learned} {CD_learned} {DE_learned}')

    #plot_days_until_criteria(all_days_until_criteria)
    
    return all_days_until_criteria # returns {ratID: {trial_type:day}} where day is day it was learned

def perf_until_critera(all_rats_performances):
    # gets the performance array of how the rats did before hitting criteria, and plots it
    all_rats_criteria_days = days_until_criteria(all_rats_performances)
    all_rats_changes = {}
    
    for ratID, criteria_days in all_rats_criteria_days.items():
        rat_performance = all_rats_performances[ratID]
        
        performance_changes = change_in_performance_trial_type(rat_performance, criteria_days)
        all_rats_changes[ratID] = performance_changes
    
    plot_all_rat_perf_changes(all_rats_changes, criterias = True)
    
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