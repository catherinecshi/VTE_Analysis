import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from scipy import stats

from analysis import performance_analysis

def plot_rat_performance(rat_performance: pd.DataFrame):
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

def plot_trial_accuracy(total_trials: np.ndarray, correct_trials: np.ndarray, trial_types: Union[list[str], set[str]]):
    """
    Plots the performance of one rat for one day for every trial type

    Args:
        total_trials: total number of trials for that trial type
        correct_trials: the number of correct trials for that trial type
        trial_types: the corresponding trial types
    """
    if isinstance(trial_types, set):
        trial_types = list(trial_types)
    
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
    
    performance_changes = performance_analysis.save_all_perf_changes_trials(rat_performance)
    
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
        sem = stats.sem(y)
        
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