import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

# SAVING & LOADING -----------
def save_rat_performance(rat_performance, save_path):
    # save in folder
    pickle_path = save_path + '/rat_performance.pkl'
    with open(pickle_path, 'wb') as fp:
        pickle.dump(rat_performance, fp)

def load_rat_performance(save_path):
    all_rats_performances = {}
        
    for rat_folder in os.listdir(save_path): # loop for each rat
        rat_path = os.path.join(save_path, rat_folder)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            print(f"Skipping over non-directory folder: {rat_path}")
            continue
            
        pickle_path = rat_path + '/rat_performance.pkl'
        
        with open(pickle_path, 'rb') as fp:
            rat_performance = pickle.load(fp)
        
        all_rats_performances[rat_folder] = rat_performance
    
    return all_rats_performances

def create_all_rats_performance(data_structure, save_path):
    for ratID in data_structure:
        rat_performance_over_sessions(data_structure, ratID, should_save = True, save_path = save_path)


# DATA ANALYSIS -------------
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

def create_all_perf_dictionary(all_rat_performances):
    all_performances = {}
    totals = [[] for _ in range(9)] # currently assuming ABCDE
    
    # loop over rats - creating dictionary of all_performances
    for ratID, rat_performance in all_rat_performances.items():
        # loop over days
        sorted_days = sorted(rat_performance.keys(), key = lambda x: int(x[3:])) # sort by after 'day'
        rats_performance = []
        num_days = 0
        
        for day in sorted_days:
            performance_for_day = rat_performance[day] # get performance for day
            
            # loop over each trial type - right now just taking sum w/o caring about trial type
            total_trials_in_day = 0
            correct_trials_in_day = 0
            
            for trial_type, total_trials, correct_trials in performance_for_day:
                total_trials_in_day += total_trials
                correct_trials_in_day += correct_trials
            
            performance = correct_trials_in_day / total_trials_in_day # calculate sum perf in day
            rats_performance.append(performance)
            
            # get totals for calculating avg
            totals[num_days].append(performance)
            num_days += 1
        
        all_performances[ratID] = rats_performance

    # calculating avg
    avg = [np.mean(day_totals) if day_totals else 0 for day_totals in totals]
    std = [np.std(day_totals) if day_totals else 0 for day_totals in totals]
    
    return all_performances, avg, std

def create_trial_type_dictionary(all_rat_performances):
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
                    'trial types not found'
            
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


# PLOTTING --------------
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

def plot_all_rat_performances(all_rat_performances, plot_trial_types = False): # currently only works with ABCDE
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
    

# TRAVERSAL FUNCTIONS ----------
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
