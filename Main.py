# packages
import os
import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy.stats import zscore

# other helper files
import data_structure
import performance_analysis


# VTE METHODS -------------

def calculate_trajectory(x, y, window_size = 100):
    # Assuming x and y are your coordinates
    x_median = np.median(x)
    y_median = np.median(y)

    # Define a window size based on your observations of the maze's layout
    window_size = 100  # This is an example size; adjust based on your specific maze dimensions

    # Define the choice point window around the medians
    window_bounds = {
        'xmin': x_median - window_size / 2,
        'xmax': x_median + window_size / 2,
        'ymin': y_median - window_size / 2,
        'ymax': y_median + window_size / 2
    }

    # Plot to verify the window
    plt.scatter(x, y, alpha=0.5)  # Plot all points
    plt.gca().add_patch(plt.Rectangle((window_bounds['xmin'], window_bounds['ymin']), window_size, window_size, linewidth=1, edgecolor='r', facecolor='none'))
    plt.axvline(x=x_median, color='k', linestyle='--')
    plt.axhline(y=y_median, color='k', linestyle='--')
    plt.title('Estimated Choice Point Area')
    plt.show()


#def learning_rates_vs_VTEs():
    

# PLOTTING METHODS --------

def plot_animation(x, y, trajectory_x = None, trajectory_y = None, interval = 20, highest = 0, zIdPhi = None):
    if not trajectory_x: # this is for when you want to plot the entire trajectory throughout the trial
        trajectory_x = x
    
    if not trajectory_y:
        trajectory_y = y
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha = 0.2) # plot the totality first
    line, = ax.plot([], [], 'bo-', linewidth = 2) # line plot
    
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    
    def init():
        # initialises animation with empty line plot
        line.set_data([], [])
        return line,
    
    def update(frame):
        # updates line plot for each frame
        # frame -> the current frame number
        
        x_val = trajectory_x[:frame] # get x up to the current frame
        y_val = trajectory_y[:frame]
        line.set_data(x_val, y_val)
        return line,

    # create animation
    ani = FuncAnimation(fig, update, frames = len(x), init_func = init, blit = True, interval = interval)
    
    # set title accordingly
    if highest == 1: # is highest
        plt.title('VTE Trial')
    elif highest == 2: # is lowest
        plt.title('Non-VTE Trial')
    
    #plt.show()
    ani.save('/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13')

def plot_zIdPhi(zIdPhi_values, day = None):
    # Collect all zIdPhi values from all trial types
    all_zIdPhis = []
    for zIdPhis in zIdPhi_values.values():
        all_zIdPhis.extend(zIdPhis)
    
    # Convert to a NumPy array for statistical calculations
    all_zIdPhis = np.array(all_zIdPhis)
    
    # Create a single plot
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(all_zIdPhis, bins=30, alpha=0.7, label='All Trial Types')
    
    # Calculate and plot the mean and standard deviation lines
    mean = np.mean(all_zIdPhis)
    std = np.std(all_zIdPhis)
    
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=2, label='+1 STD')
    plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=2, label='-1 STD')
    
    # Set the title and labels
    plt.title('Combined IdPhi Distribution Across All Trial Types')
    plt.xlabel('zIdPhi')
    plt.ylabel('Frequency')
    
    # Show the legend
    plt.legend()
    
    plt.tight_layout()
    #plt.show()
    if day:
        plt.savefig(f'/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13/{day}')

def plot_trajectory(x, y, zIdPhi, trajectories, highest):
    
    # get trajectory points
    trajectory_x, trajectory_y = trajectories
    
    # plot the normal points
    plt.figure(figsize = (10, 6))
    plt.plot(x, y, color='green', alpha=0.4)
    
    # plot the trajectory
    plt.plot(trajectory_x, trajectory_y, color = 'red', alpha = 0.8, label = zIdPhi)
    
    # display plot
    if highest:
        plt.title('VTE Trial trajectory')
    else:
        plt.title('Non-VTE Trial Trajectory')
        
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()


# CENTRAL METHODS (traversing) ----------

def test(data_structure, ratID, day):
    DLC_df = data_structure[ratID][day]['DLC_tracking']
    
    # get x and y coordinates
    x, y = data_structure.filter_dataframe(DLC_df)
    
    get_centre_zone(x, y)

def rat_VTE_over_sessions(data_structure, ratID):
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
            print(f'error - {error} on day {day}')
            


### VTE ANALYSIS --------
def compare_zIdPhis(base_path):
    IdPhis_across_days = [] # this is so it can be zscored altogether
    
    days = []
    zIdPhi_means = []
    zIdPhi_stds = []
    IdPhi_means = []
    IdPhi_stds = []
    vte_trials = []
    
    for day_folder in os.listdir(base_path):
        day_path = os.path.join(base_path, day_folder)
        if os.path.isdir(day_path):
            days.append(day_folder)
        
        for root, dirs, files in os.walk(day_path):
            for f in files:
                file_path = os.path.join(root, f)
                if 'zIdPhi' in f:
                    with open(file_path, 'rb') as fp:
                        zIdPhis = pickle.load(fp)
                    
                    # flatten from dict to array
                    all_zIdPhis = [zIdPhi for zIdPhi_vals in zIdPhis.values() for zIdPhi in zIdPhi_vals]
                    
                    # mean & std stored across days
                    zIdPhis_mean = np.mean(all_zIdPhis)
                    zIdPhis_std = np.std(all_zIdPhis)
                    
                    # append into array
                    zIdPhi_means.append(zIdPhis_mean)
                    zIdPhi_stds.append(zIdPhis_std)
                    
                    # check how many vte trials
                    cutoff = zIdPhis_mean + zIdPhis_std
                    vte_trials.append(sum(zIdPhi > cutoff for zIdPhi in all_zIdPhis))
                
                if 'IdPhi' in f and 'z' not in f:
                    with open(file_path, 'rb') as fp:
                        IdPhis = pickle.load(fp)
                    
                    # flatten
                    all_IdPhis = [IdPhi for IdPhi_vals in IdPhis.values() for IdPhi in IdPhi_vals]
                
                    IdPhis_mean = np.mean(all_IdPhis)
                    IdPhis_std = np.std(all_IdPhis)
                    
                    IdPhi_means.append(IdPhis_mean)
                    IdPhi_stds.append(IdPhis_std)
                    
                    # zscore later perhaps
                    # IdPhis_across_days.append(all_IdPhis)
    
    """print(len(zIdPhi_means))
    print(len(vte_trials))
    print(len(IdPhi_means))
    print(len(days))"""
    
    df = pd.DataFrame({
        'Day': day,
        'zIdPhi Mean': zIdPhi_means,
        'zIdPhi Std': zIdPhi_stds,
        'IdPhi Mean': IdPhi_means,
        'IdPhi Std': IdPhi_stds,
        'VTE Trials': vte_trials
    })
    
    # sort according to day number
    df['sort_key'] = df['Day'].apply(lambda x: int(x[3:])) 
    df_sorted = df.sort_values(by = 'sort_key')
    df_sorted = df_sorted.drop(columns = ['sort_key']) # drop now that it's sorted
    
    """# calculate difference between values for consecutive days
    comparison_cols = df.columns.drop('Day')
    diffs = df[comparison_cols].diff()
    # diffs['Day'] = df['Day'] # add days back in if desired"""
    
    # save dataframe
    dataframe_path = os.path.join(base_path, 'zIdPhis_and_IdPhis')
    df.to_csv(dataframe_path)
    
    """# save differences in a separate numpy file
    IdPhi_mean_diffs = diffs['IdPhi Mean'].to_numpy()
    zIdPhi_mean_diffs = diffs['zIdPhi Mean'].to_numpy()
    
    idphi_diffs_path = os.path.join(base_path, 'IdPhi_Mean_Diffs')
    zidphi_diffs_path = os.path.join(base_path, 'zIdPhi_Mean_Diffs')
    
    np.save(idphi_diffs_path, IdPhi_mean_diffs)
    np.save(zidphi_diffs_path, zIdPhi_mean_diffs)
    
    print(f'idphi - {IdPhi_mean_diffs}')
    print(f'zidphi - {zIdPhi_mean_diffs}')"""
    
    # return sorted vte trials according to day
    vte_trials_sorted = df_sorted['VTE Trials']
    
    return vte_trials_sorted
    

# ASSIGNMENT 1 --------
# creating the main data structure
#base_path = '/Users/catpillow/Documents/VTE Analysis/Data_draft'
#main_data_structure = data_structure.create_main_data_structure(base_path)

# saving
#base_path = '/Users/catpillow/Downloads/BP13_timestamps'
#save_path = '/Users/catpillow/Documents/VTE Analysis/VTE_Data' # this is just SS (added BP13 DLC & timestamps)
save_path_BP = '/Users/catpillow/Documents/VTE Analysis/VTE_Data/BP13'
#save_path = '/Users/catpillow/Downloads/VTE_Data'
#data_structure.save_data_structure(main_data_structure, save_path)
#data_structure.save_DLC(base_path, save_path)
#data_structure.save_timestamps(base_path, save_path)

# loading
#loaded_data_structure = data_structure.load_data_structure(save_path)
BP13_data = data_structure.load_one_rat(save_path_BP)

# ASSIGNMENT 2 ---------
# example
ratID = 'BP13'
day = 'Day8'

# plot positioning for greenLED
#scatter_plot(loaded_data_structure, ratID, day)

# occupancy map
#occupancy_map(loaded_data_structure, ratID, day)

# calculate speed
#speed(loaded_data_structure, ratID, day)

# ASSIGNMENT 3 ---------
#performance_analysis.rat_performance_one_session(loaded_data_structure, ratID, day)

# ASSIGNMENT 4 ---------
#time_until_first_choice(loaded_data_structure, ratID, day)

# VTEs --------
#DLC = loaded_data_structure[ratID][day]['DLC_tracking']
    
# get coordinates
#x, y = filter_dataframe(DLC)

#plot_animation(x, y)
zIdPhi, IdPhi, trajectories = quantify_VTE(BP13_data, ratID, day, save = False)
#rat_VTE_over_sessions(BP13_data, ratID)
#print(f"zIdPhi - {zIdPhi}")
#print(f"IdPhi - {IdPhi}")
#test(loaded_data_structure, ratID, day)

#rat_VTE_over_sessions(loaded_data_structure, ratID)

# LEARNING RATES --------
#rat_performance = performance_analysis.rat_performance_over_sessions(loaded_data_structure, ratID)
#performance_analysis.create_all_rats_performance(loaded_data_structure, save_path = save_path)
#all_rats_performances = performance_analysis.load_rat_performance(save_path)
#performance_analysis.plot_all_rat_performances(all_rats_performances)
#perf_changes = None
"""for rat, rat_performance in all_rats_performances.items():
    if rat == ratID:
        performance_analysis.plot_rat_perf_changes(rat_performance)"""
        #perf_changes, avg_changes = performance_analysis.change_in_performance(rat_performance)
#performance_analysis.all_rats_perf_changes(all_rats_performances)

#performance_analysis.days_until_criteria(all_rats_performances)
#performance_analysis.perf_until_critera(all_rats_performances)


# COMPARISON ---------------
"""vte_trials = compare_zIdPhis(save_path_BP)

print(len(vte_trials))
print(len(avg_changes))
create_scatter_plot(vte_trials, avg_changes)"""