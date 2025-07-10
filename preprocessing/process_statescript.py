import re
import numpy as np

from config import settings
from utilities import logging_utils

# pylint: disable=broad-exception-caught
logger = logging_utils.get_module_logger("process_statescript")

def get_session_performance(content: str) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """
    this function analyses one statescript log to retrieve performance
    
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

    Parameters:
    - content: state script log

    Returns:
    - numpy array: array of total trials for each trial type; 0 index -> 'AB' etc
    - numpy array: array of correct trials for each trial type
    - bool list: list of trues and falses depending on whether the rat performed correctly for trial
    """
    
    lines = content.splitlines()
    
    # initialise variables
    middle_numbers = set() # array to hold the middle numbers
    end_of_trial = False # checks if middle_numbers should be reset
    error_trial = None
    current_trial = None
    last_line = len(lines) - 3 # this is for the last trial of a session bc no 'New Trial' to signal end of trial
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
                current_trial = int(line.split()[-1]) - 1
                
                # had a weird instance of trialtype = 0, skipping that
                # 0 = trialtype 1 bc minus 1 earlier
                if current_trial < 0:
                    logger.debug(f"current trial = {current_trial} time - {parts[0]}" 
                                 f"for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
                    end_of_trial = False
                    continue
                if current_trial > 4:
                    logger.debug(f"current trial = {current_trial} larger than 4, time - {parts[0]}"
                                  f"for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
                    
                end_of_trial = False # reset
            except Exception as e:
                logger.error(f"error when trying to get current trial type after end of trial with: {e}"
                             f"with {settings.CURRENT_RAT} on {settings.CURRENT_DAY} at {parts[0]}")
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
            
            # check if something dodgy is going on or if the rat peed
            if middle_numbers - possible_middle_numbers: # there is a value inside middle_numbers that's not arm
                logger.info(f"{settings.CURRENT_RAT} peed on {settings.CURRENT_DAY} at {parts[0]}")
                middle_numbers = middle_numbers.intersection(possible_middle_numbers)
            
            if len(middle_numbers) == 3:
                error_trial = True
            elif len(middle_numbers) == 4:
                error_trial = False
            elif len(middle_numbers) > 4:
                # if it is still above 4, something's wrong, but focus on resetting the problem for now
                error_trial = False
                logger.warning(f"something weird - middle_numbers > 4 at {parts[0]}"
                                f"happening for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
            else:
                continue # this usually happens at the start when the rat first licks for a session or at the end
            
            # check if middle numbers align with prints of correct/wrong
            if correct is not None:
                if correct != error_trial and settings.CURRENT_RAT != "BP06":
                    logger.warning("middle numbers doesn't align with logs of correctness"
                                    f"{settings.CURRENT_RAT} on {settings.CURRENT_DAY} at {parts[0]}")
                    error_trial = correct
                elif correct != error_trial:
                    # for BP06, since his ss looks so diff from others
                    logger.warning(f"middle number is different from log correctness"
                                    f"{settings.CURRENT_RAT} on {settings.CURRENT_DAY} at {parts[0]}")
            
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
            settings.update_trial(str(no_trials))
    
    # do analysis for the last trial
    if middle_numbers - possible_middle_numbers: # there is a value inside middle_numbers that's not arm
        logger.info(f"{settings.CURRENT_RAT} peed on {settings.CURRENT_DAY} at {parts[0]}")
        middle_numbers = middle_numbers.intersection(possible_middle_numbers)
    
    if len(middle_numbers) == 3:
        error_trial = True
    elif len(middle_numbers) == 4:
        error_trial = False
    elif len(middle_numbers) > 4:
        # if it is still above 4, something's wrong, but focus on resetting the problem for now
        error_trial = False
        logger.warning(f"something weird - middle_numbers > 4 at {parts[0]}"
                        f"happening for {settings.CURRENT_RAT} on {settings.CURRENT_DAY}")
    
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

def get_trials_for_session(content: str) -> set:
    """gets the possible trial types in a session"""
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
                logger.error(f"trial type is 0 for {settings.CURRENT_RAT} on {settings.CURRENT_DAY} for {line}")
    
    return numbers # where this only contains the numbers corresponding to trial types that are available

def trial_accuracy(content: str) -> tuple[set, np.ndarray, np.ndarray]:
    """returns trial types, total number of trials, and correct number of trials"""
    total_trials, correct_trials, _ = get_session_performance(content)
    trial_types = get_trials_for_session(content)
    
    return trial_types, total_trials, correct_trials
