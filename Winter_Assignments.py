import os
import fnmatch
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# DATA STRUCTURE METHODS ---------
def process_dlc_data(file_path):
    df = pd.read_csv(file_path, header=[1, 2])
    df.set_index(('bodyparts', 'coords'), inplace=True)
    return df

def process_loaded_dlc_data(file_path):
    df = pd.read_csv(file_path, header=[0, 1])
    return df

def process_statescript_log(file_path):
    with open(file_path) as file:
        content = file.read()
    
    return content

def create_main_data_structure(base_path): # creates a nested dictionary
    data_structure = {}

    for rat_folder in os.listdir(base_path): # loop for each rat
        rat_path = os.path.join(base_path, rat_folder, 'inferenceTraining')
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            print(f"Skipping over non-directory folder: {rat_path}")
            continue
        
        # skip over empty folders
        day_folders = os.listdir(rat_path)
        if not day_folders: # if folder is empty
            print(f"{rat_path} is empty")
            continue
            
        data_structure[rat_folder] = {} # first nest in dictionary
        
        for day_folder in os.listdir(rat_path): # loop for each day (in each rat folder)
            day_path = os.path.join(rat_path, day_folder)
            dlc_data = None
            ss_data = None
            
            # for checking if a ss log & dlc csv file has been found for each day for each rat
            ss = False
            dlc = False
        
            for root, dirs, files in os.walk(day_path): # look at all the files in the day folder
                for f in files:
                    f = f.lower() # there were some problems with cases
                    
                    # storing the DLC csv
                    if fnmatch.fnmatch(f, '*dlc*.csv'): 
                        # checks if there is a duplication
                        if dlc == True:
                            print("More than one DLC file found")
                        else:
                            dlc = True
                        
                        file_path = os.path.join(root, f)
                        dlc_data = process_dlc_data(file_path)
                        
                        print(file_path)
                    
                    # storing the statescript log
                    if fnmatch.fnmatch(f, '*track*.statescriptlog'):
                        # checks if there is a duplication
                        if ss == True:
                            print("More than one SS Log found")
                        else:
                            ss = True
                        
                        file_path = os.path.join(root, f)
                        ss_data = process_statescript_log(file_path)
                        
                        print(file_path)
            
            # add to dictionary
            if ss and dlc:
                data_structure[rat_folder][day_folder] = {
                    "DLC_tracking": dlc_data,
                    "stateScriptLog": ss_data
                }
            elif (not ss) and (not dlc):
                print(f"No stateScriptLog or DLC file found for rat {rat_folder} for {day_folder}")
            elif not ss:
                print(f"No stateScriptLog found for rat {rat_folder} for {day_folder}")
            elif not dlc:
                print(f"No DLC .csv found for rat {rat_folder} for {day_folder}")

    return data_structure

def save_data_structure(data_structure, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for rat, days in data_structure.items():
        rat_path = os.path.join(save_path, rat)
        if not os.path.exists(rat_path):
            os.makedirs(rat_path)
        
        
        for day, data in days.items():
            day_path = os.path.join(rat_path, day)
            if not os.path.exists(day_path):
                os.makedirs(day_path)

            if "DLC_tracking" in data:
                dlc_path = os.path.join(day_path, f"{day}_DLC_tracking.csv")
                data["DLC_tracking"].to_csv(dlc_path, header = True, index = False)
            
            if "stateScriptLog" in data:
                ss_path = os.path.join(day_path, f"{day}_stateScriptLog.txt")
                with open(ss_path, 'w') as file:
                    file.write(data["stateScriptLog"])

def load_data_structure(save_path): # this function assumes no errors bc they would've been caught before saving
    data_structure = {}

    for rat_folder in os.listdir(save_path): # loop for each rat
        rat_path = os.path.join(save_path, rat_folder)
        
        # skip over .DS_Store
        if not os.path.isdir(rat_path):
            print(f"Skipping over non-directory folder: {rat_path}")
            continue
            
        data_structure[rat_folder] = {} # first nest in dictionary
        
        for day_folder in os.listdir(rat_path): # loop for each day (in each rat folder)
            day_path = os.path.join(rat_path, day_folder)
            dlc_data = None
            ss_data = None
        
            for root, dirs, files in os.walk(day_path): # look at all the files in the day folder
                for f in files:
                    f = f.lower()
                    
                    # storing the DLC csv
                    if fnmatch.fnmatch(f, '*dlc*.csv'): 
                        file_path = os.path.join(root, f)
                        dlc_data = process_loaded_dlc_data(file_path)
                    
                    # storing the statescript log
                    if fnmatch.fnmatch(f, '*statescriptlog*'):
                        file_path = os.path.join(root, f)
                        ss_data = process_statescript_log(file_path)
            
            # add to dictionary
            data_structure[rat_folder][day_folder] = {
                "DLC_tracking": dlc_data,
                "stateScriptLog": ss_data
            }

    return data_structure


# DATA MANIPULATION/ANALYSIS METHODS ---------
def filter_dataframe(df, track_part = 'greenLED', std_multiplier = 3):
    # modify a copy instead of the original
    # also filter based on likelihood values
    filtered_data = df[df[(track_part, 'likelihood')] > 0.9].copy()
    
    
    # diff between consecutive frames
    filtered_data['diff_x'] = filtered_data[(track_part, 'x')].diff().abs()
    filtered_data['diff_y'] = filtered_data[(track_part, 'y')].diff().abs()
    
    # determining position threshold
    threshold_x = filtered_data['diff_x'].std() * std_multiplier
    threshold_y = filtered_data['diff_y'].std() * std_multiplier
    
    # filtering out jumps in coordinates
    jumps = (filtered_data['diff_x'] > threshold_x) | (filtered_data['diff_y'] > threshold_y)
    filtered_data.loc[jumps, [(track_part, 'x'), (track_part, 'y')]] = None
    
    # interpolating
    filtered_data[(track_part, 'x')].interpolate(inplace = True)
    filtered_data[(track_part, 'y')].interpolate(inplace = True)
    
    
    # final coordinate points
    x = filtered_data[(track_part, 'x')]
    y = filtered_data[(track_part, 'y')]
    
    return x, y

def get_time(content, statescript_time):
    lines = content.splitlines()
    
    starting_time = None # store the starting time
    
    for line in lines:
        if '#' in line:
            continue # skip the starting comments
        elif all(char.isspace() or char.isdigit()for char in line):
            parts = line.split()
            starting_time = int(parts[0])
            break
    
    # calculating real time passed since start of session
    time_passed = statescript_time - starting_time
    time_passed = time_passed / 1000 # turning it from ms to seconds
    
    return time_passed
            
def calculate_framerate(content, x):
    # get the last line in the statescript log that only has numbers
    last_line = None
    
    lines = content.splitlines()
    for line in lines:
        if all(char.isdigit() or char.isspace() for char in line):
            last_line = line
    
    # get the time value from the line
    time = None
    
    for index, char in enumerate(last_line):
        if char.isspace():
            time = last_line[:index]
            break
    
    # turn from string to integer
    time = int(time)
    
    # turn into seconds
    get_time(content, time)
    
    # calculate framerate
    framerate = time / len(x)
    
    return framerate

def calculate_speed(x, y, framerate):
    diff_x = x.diff().fillna(0)
    diff_y = y.diff().fillna(0)
    
    displacement_per_frame = np.sqrt(diff_x**2 + diff_y**2)
    
    speed = displacement_per_frame * framerate
    
    return speed

def trial_analysis(content):
    lines = content.splitlines()
    
    # temporary variables
    middle_numbers = set() # array to hold the middle numbers
    end_of_trial = False # checks if middle_numbers should be reset
    error_trial = None
    current_trial = None
    last_line = len(lines) - 1 # this is for the last trial of a session bc no 'New Trial' to signal end of trial
    
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
                
        elif 'New Trial' in line: # indicate start of a new trial
            end_of_trial = True
            
        elif end_of_trial and 'trialType' in line: # this excludes 'trialType' from summaries
            current_trial = int(line[-1]) - 1 # last char is the trial type
            end_of_trial = False # reset
            
        elif index == last_line:
            end_of_trial = True
        
        # analysis when a trial has ended
        if end_of_trial and middle_numbers: # excludes first trial (bc it would be empty at the start)
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
            break
        
        if 'iTrialType' and 'Num' and '%' in line: # select the lines I want
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
 
def time_until_choice(content): # currently only for the first choice
    lines = content.splitlines()
    
    # some variables
    new_trial = False # to check if a trial is in session
    time = []
    arm = set()
    lick_count = 0
    not_first = False
    time_home = None
    time_arm = None
    
    for line in lines:
        if '#' in line: # skip starting comments
            continue
        elif 'New Trial' in line:
            new_trial = True
        elif all(char.isdigit() or char.isspace() for char in line) and new_trial: # a normal licking line
            parts = line.split()
            current_arm = int(parts[1])
            current_time = int(parts[0])
            
            # check when rat left home
            if current_arm > 1 and not not_first: # licked at arm not at home
                time_home = time[-1] # haven't appended to time yet so this is last line's time
                time_arm = current_time
            
            # check for how many licks
            if current_arm > 1:
                if current_arm in arm: # not a new arm
                    lick_count += 1
                else:
                    lick_count = 1 # new arm so reset lick count
            
            # check if the rat changed its choice without licking enough at one arm
            if current_arm > 1 and len(arm) > 3:
                time_arm = current_time
            
            time.append(current_time)
            arm.add(current_arm)

        if lick_count == 3: # enough licks for a choice
            break

    # calculating the difference between the time
    time_arm_seconds = get_time(content, time_arm)
    time_home_seconds = get_time(content, time_home)
    time_diff = time_arm_seconds - time_home_seconds
    
    return time_diff
     
            
# PLOTTING METHODS --------
def create_scatter_plot(x, y):
    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, c = 'green', alpha = 0.6)
    plt.title('Tracking Data')
    plt.grid(True)
    plt.show()

def create_occupancy_map(x, y, framerate, bin_size = 15):
    # determine size of occupancy map
    x_max = x.max()
    y_max = y.max()
    num_bins_x = int(np.ceil(x_max / bin_size))
    num_bins_y = int(np.ceil(y_max / bin_size))
    
    # empty grid
    occupancy_grid = np.zeros((num_bins_x, num_bins_y))
    
    # bin data points
    for i in range(len(x)):
        bin_x = int(x.iloc[i] // bin_size)
        bin_y = int(y.iloc[i] // bin_size)
        
        occupancy_grid[bin_x, bin_y] += 1
    
    occupancy_grid = occupancy_grid / framerate
    
    # plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(occupancy_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Time spent')
    plt.title('Occupancy Map')
    plt.show()

def create_trial_accuracy(total_trials, correct_trials, trial_types):
    percentage_correct = (correct_trials / total_trials) * 100
    print(percentage_correct)
    
    # adjusting trial types
    length = len(total_trials)
    trial_types = trial_types[:length]
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(trial_types, percentage_correct, color='blue')
    plt.title('Trial Accuracy')
    plt.xticks(trial_types)
    plt.show()


# CENTRAL METHODS (traversing) -----------
def scatter_plot(data_structure, ratID, day):
    DLC = data_structure[ratID][day]['DLC_tracking']
    
    # get coordinates
    x, y = filter_dataframe(DLC)
    
    # plot
    create_scatter_plot(x, y)

def occupancy_map(data_structure, ratID, day):
    DLC_df = data_structure[ratID][day]['DLC_tracking']
    SS_df = data_structure[ratID][day]['stateScriptLog']
    
    # get x and y coordinates
    x, y = filter_dataframe(DLC_df)
    
    # get framerate
    framerate = calculate_framerate(SS_df, x)
    
    # make map
    create_occupancy_map(x, y, framerate)

def speed(data_structure, ratID, day):
    DLC_data = data_structure[ratID][day]['DLC_tracking']
    SS_data = data_structure[ratID][day]['stateScriptLog']
    
    track_part = 'haunch' # assumed to be the best for tracking animal speed
    x, y = filter_dataframe(DLC_data, track_part)
    
    # get framerate
    framerate = calculate_framerate(SS_data, x)
    
    # calculate speed
    speed = calculate_speed(x, y, framerate)
    
    return speed

def trial_accuracy(data_structure, ratID, day):
    content = data_structure[ratID][day]['stateScriptLog']
    
    total_trials, correct_trials = trial_analysis(content)
    trial_types = get_trial_types(content)
    create_trial_accuracy(total_trials, correct_trials, trial_types)

def time_until_first_choice(data_structure, ratID, day):
    content = data_structure[ratID][day]['stateScriptLog']
    
    time = time_until_choice(content)
    print(time)

# ASSIGNMENT 1 --------
# creating the main data structure
#base_path = '/Users/catpillow/Downloads/Data 2'
#main_data_structure = create_main_data_structure(base_path)

# saving
save_path = '/Users/catpillow/Downloads/VTE_Data'
#save_data_structure(main_data_structure, save_path)

# loading
loaded_data_structure = load_data_structure(save_path)

# ASSIGNMENT 2 ---------
# example
ratID = 'BP06'
day = 'Day7'

# plot positioning for greenLED
#scatter_plot(loaded_data_structure, ratID, day)

# occupancy map
#occupancy_map(loaded_data_structure, ratID, day)

# calculate speed
#speed(loaded_data_structure, ratID, day)

# ASSIGNMENT 3 ---------
#trial_accuracy(loaded_data_structure, ratID, day)

# ASSIGNMENT 4 ---------
time_until_first_choice(loaded_data_structure, ratID, day)

# ASSIGNMENT 5 --------