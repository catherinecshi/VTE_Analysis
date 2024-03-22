import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

def save_rat_performance(rat_performance, save_path):
    # save in folder
    pickle_path = save_path + '/rat_performance.pkl'
    with open(pickle_path, 'wb') as fp:
        pickle.dump(rat_performance, fp)

def trial_analysis(content):
    lines = content.splitlines()
    
    # temporary variables
    middle_numbers = set() # array to hold the middle numbers
    end_of_trial = False # checks if middle_numbers should be reset
    error_trial = None
    current_trial = None
    last_line = len(lines) - 1 # this is for the last trial of a session bc no 'New Trial' to signal end of trial
    no_trials = 0 # count number of trials that has passed
    
    # stored for graphing
    total_trials = np.zeros(10) # start out with the total number of possible trial types
    correct_trials = np.zeros(10)
    
    for index, line in enumerate(lines):
        if line.startswith('#'): # skip the starting comments
            continue
        
        elif all(char.isdigit() or char.isspace() for char in line): # a normal licking line
            # check the middle numbers to determine arms that has been ran to
            parts = line.split()
            if len(parts) == 3:
                middle_numbers.add(parts[1])
            else:
                print('All number line has ' + str(len(parts)) + ' integers')
                print(parts)
                
        elif 'New Trial' in line and 'reset' not in line: # indicate start of a new trial
            end_of_trial = True
            
        elif end_of_trial and 'trialType' in line: # this excludes 'trialType' from summaries
            current_trial = int(line[-1]) - 1 # last char (line[-1]) is the trial type
            end_of_trial = False # reset
            
        elif index == last_line:
            end_of_trial = True
        
        # analysis when a trial has ended
        if end_of_trial and middle_numbers: 
            if len(middle_numbers) == 3:
                error_trial = True
            elif len(middle_numbers) == 4:
                error_trial = False
            else:
                print('middle_numbers has ' + str(len(middle_numbers)) + 'integers')
                print(middle_numbers)
                continue # this usually happens at the start when the rat first licks for a session
            
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
    
    return final_total_trials, final_correct_trials

def get_trial_types(content):
    trial_types = np.empty(10, dtype=object)
    lines = content.splitlines()
    
    for line in lines:
        if '#' not in line: # i only want to look at the starting comments
            break # might want to handle this better in the future
        
        if 'iTrialType' in line and 'Num' in line and '%' in line: # select the lines I want
            parts = line.split('%')
            number_part = parts[0]
            letter_pair = parts[1].strip()
            
            # store the trial 
            match = re.search(r'iTrialType(\d+)', number_part)
            if match:
                number = int(match.group(1)) - 1 # minus one for the index
            
            # store into array
            trial_types[number] = letter_pair
                    
    # remove any excess pairs
    trial_mask = trial_types != None
    final_trial_types = trial_types[trial_mask]
    
    return final_trial_types  

def trial_accuracy(content):
    total_trials, correct_trials = trial_analysis(content)
    trial_types = get_trial_types(content)
    #create_trial_accuracy(total_trials, correct_trials, trial_types)
    
    return trial_types, total_trials, correct_trials

def plot_rat_performance(rat_performance):
    performance_by_type = {}
    
    # sort days to follow chronological order
    sorted_days = sorted(rat_performance.keys(), key = lambda x: int(x[3:]))
    
    # compile perf data
    for day in sorted_days:
        print(day)
        for trial_type, total_trials, correct_trials in rat_performance[day]:
            performance = (correct_trials / total_trials) * 100 if total_trials else 0
            
            if trial_type not in performance_by_type:
                performance_by_type[trial_type] = [None] * len(sorted_days)
            
            day_index = sorted_days.index(day)
            print(trial_type)
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
    percentage_correct = (correct_trials / total_trials) * 100
    print(percentage_correct)
    
    # adjusting trial types
    length = len(total_trials)
    trial_types = trial_types[:length]
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(trial_types, percentage_correct, color='red')
    plt.title('Trial Accuracy')
    plt.ylabel('Performance (%)')
    plt.xlabel('Trial Types')
    plt.ylim(top=100, bottom=0)
    plt.xticks(trial_types)
    plt.show()

def rat_performance_one_session(data_structure, ratID, day):
    ss_data = data_structure[ratID][day]['stateScriptLog']
    trial_types, total_trials, correct_trials = trial_accuracy(ss_data)
    
    plot_trial_accuracy(total_trials, correct_trials, trial_types)

def rat_performance_over_sessions(data_structure, ratID, should_save = False, save_path = None):
    if ratID in data_structure:
        rat_performance = {} # dict for all results of rats
        all_trials = None
        
        for day_folder, contents in data_structure[ratID].items():
            if day_folder != '.DS_Store': # skip .DS_Store
                ss_data = contents["stateScriptLog"]
                trial_types, total_trials, correct_trials = trial_accuracy(ss_data) # get the trial info for this day for this rat
                
                # check if trial types is empty; doesn't work if first day is the fucked up one
                if len(trial_types) == 0 or not all(trial.isalpha() for trial in trial_types):
                    trial_types = all_trials
                elif all_trials is None:
                    all_trials = trial_types
                    
                # make sure they're the same length
                length = len(total_trials)
                trial_types = trial_types[:length]
                
                # add to dictionary
                rat_performance_for_day = []
                
                for i in range(len(total_trials)):
                    rat_performance_for_day.append([trial_types[i], total_trials[i], correct_trials[i]])

                rat_performance[day_folder] = rat_performance_for_day
    
    if should_save:
        file_path = save_path + '/' + ratID
        save_rat_performance(rat_performance, file_path)
    
    return rat_performance

def create_all_rats_performance(data_structure, save_path):
    for ratID in data_structure:
        rat_performance_over_sessions(data_structure, ratID, should_save = True, save_path = save_path)

    
    