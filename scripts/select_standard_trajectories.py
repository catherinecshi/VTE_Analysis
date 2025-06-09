import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

from models import helper
from utilities import math_utils

is_good = False
def plot_trajectory_animation(x, y, trajectory_x, trajectory_y, interval=20, traj_id="", label=None):
    """
    creates and displays an animation of a trajectory over the backdrop of x and y coordinates

    Args:
        x (int array): x coordinates
        y (int array): y coordinates
        trajectory_x (int array, optional): x coordinates of trajectory. Defaults to None.
        trajectory_y (int array, optional): y coordinates of trajectory. Defaults to None.
        interval (int, optional): time interval between frames in milliseconds. Defaults to 20.
        title (str, optional): title. Defaults to "".
        label (Any, optional): label. Defaults to None.
        save (str, optional): file path if saving is desired. Defaults to None.
    """
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.2) # plot the totality first
    line, = ax.plot([], [], "bo-", linewidth=2) # line plot
    
    ax.set_xlim(np.min(x) - 50, np.max(x) + 50)
    ax.set_ylim(np.min(y) - 50, np.max(y) + 50)
    
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
    
    def good_trajectory(event):
        global is_good
        is_good = True
        plt.close()
    
    def bad_trajectory(event):
        global is_good
        is_VTE = False
        plt.close()

    # create animation
    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=interval)
    
    # set title
    plt.title(traj_id)
    
    if label:
        if isinstance(label, float):
            rounded_label = math_utils.round_to_sig_figs(label, 3)
        else:
            rounded_label = label
        legend_elements = [Line2D([0], [0], color="blue", lw=2, label=rounded_label)]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))
    
    axes1 = plt.axes((0.8, 0.05, 0.1, 0.075))
    btn1 = Button(axes1, "good")
    btn1.on_clicked(good_trajectory)
    
    axes2 = plt.axes((0.2, 0.05, 0.1, 0.075))
    btn2 = Button(axes2, "bad")
    btn2.on_clicked(bad_trajectory)

    plt.show()
    
def repeat_or_not():
    _, ax = plt.subplots()
    ax.set_visible(False) # make plot area empty
    
    def repeat(event):
        global is_good
        is_good = False
        plt.close()
    
    def no_repeat(event):
        global is_good
        is_good = True
        plt.close()
        
    axes1 = plt.axes((0.8, 0.05, 0.1, 0.075))
    btn1 = Button(axes1, "repeat")
    btn1.on_clicked(repeat)
    
    axes2 = plt.axes((0.2, 0.05, 0.1, 0.075))
    btn2 = Button(axes2, "no repeat")
    btn2.on_clicked(no_repeat)
    
    plt.show()

dlc_path = os.path.join(helper.BASE_PATH, "processed_data", "cleaned_dlc")
vte_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values")
standard_traj_path = os.path.join(helper.BASE_PATH, "processed_data", "standard_trajectories")
for rat in os.listdir(vte_path):
    if ".DS_Store" in rat or "BP06" in rat or "zIdPhi" in rat:
        continue
    
    # store the standard trajectories
    rat_standard_trajectories = {}
    print(f"currently iterating through {rat}")

    rat_path = os.path.join(vte_path, rat)
    for day in sorted(os.listdir(rat_path), key=lambda x: int(re.search(r"\d+", x).group() if re.search(r"\d+", x) else 0)):
        if ".DS_Store" in day or "Day" not in day:
            continue
        
        day_standard_trajectories = {} # {day: {choice: (x_vals, y_vals)}}
        trajectories_not_found = {} # {day: {choice: (all_x, all_y)}} so i have the backdrop to plot against later
        repeat_day_choices = {} # {day: [choice]} # so i know to skip when checking past trajectories
        print(f"currently iterating through {day}")
        
        day_path = os.path.join(rat_path, day)
        for root, _, files in os.walk(day_path):
            for file in files:
                if "trajectories.csv" not in file:
                    continue
                trajectories_csv = pd.read_csv(os.path.join(root, file))
                
                # find a standard trajectory for each choice
                try:
                    choice_groups = trajectories_csv.groupby(by="Choice")
                except KeyError:
                    print(f"{rat} on {day} does not have Choice in trajectory file")
                    continue
                    
                # get all of the x and y values for this session
                coordinates_path = os.path.join(dlc_path, rat, f"{day}_coordinates.csv")
                coordinates_csv = pd.read_csv(coordinates_path)
                all_x = coordinates_csv["x"]
                all_y = coordinates_csv["y"]

                # go through each of the choice groups
                for choice, choice_group in choice_groups:
                    choice_empty = True # to keep track of any choices that didn't have any good trajectories
                    print("there are currently the following days in rat_standard_trajectories")
                    print(f"{rat_standard_trajectories.keys()}")
                    
                    # check if the previous days' trajectories will work
                    for iterate_day in sorted(list(rat_standard_trajectories.keys()), key=lambda x: int(re.search(r"\d+", x).group() if re.search(r"\d+", x) else 0)):
                        print(f"there are currently the following choices in rat_standard_trajectories[{iterate_day}]")
                        print(f"{rat_standard_trajectories[iterate_day].keys()}")
                        if iterate_day == day:
                            break
                        
                        # check if this is a repeat
                        if iterate_day in repeat_day_choices:
                            if choice in repeat_day_choices[iterate_day]:
                                continue
                        
                        if choice in rat_standard_trajectories[iterate_day]:
                            old_traj = rat_standard_trajectories[iterate_day][choice]
                            old_x, old_y = old_traj
                        else:
                            continue # first time seeing this choice
                        
                        is_good = False
                        title = f"{rat}_{day}_{choice} with the trajectory from {iterate_day}"
                        plot_trajectory_animation(all_x, all_y, old_x, old_y, traj_id=title, label=choice)
                        
                        # see if old trajectories work out
                        if is_good:
                            day_standard_trajectories[choice] = (old_x, old_y)
                            choice_empty = False
                            
                            # make sure i skip repeats
                            if day in repeat_day_choices:
                                repeat_day_choices[day].append(choice)
                            else:
                                repeat_day_choices[day] = [choice]
                            break
                    
                    # if an appropraite trajecotry was found from old ones, skip future code
                    if not choice_empty:
                        continue
                    
                    for index, row in choice_group.iterrows():
                        x_vals = row["X Values"]
                        y_vals = row["Y Values"]
                        x_vals = ast.literal_eval(x_vals)
                        y_vals = ast.literal_eval(y_vals)
                        traj_id = row["ID"]

                        is_good = False # reset before each time
                        plot_trajectory_animation(all_x, all_y, x_vals, y_vals, traj_id=traj_id, label=choice)
                        
                        # save into the dictionary if trajectory looks good
                        if is_good:
                            day_standard_trajectories[choice] = (x_vals, y_vals)
                            choice_empty = False
                            break
                        else:
                            # this is so that i can know where it starts and ends later
                            day_standard_trajectories[choice] = (x_vals, y_vals)
                    
                    if choice_empty:
                        # ask if user wants a repeat incase there actually was a good trajectory they missed
                        is_good = False
                        repeat_or_not()
                        
                        if not is_good:
                            for index, row in choice_group.iterrows():
                                x_vals = row["X Values"]
                                y_vals = row["Y Values"]
                                x_vals = ast.literal_eval(x_vals)
                                y_vals = ast.literal_eval(y_vals)
                                traj_id = row["ID"]

                                is_good = False # reset before each time
                                plot_trajectory_animation(all_x, all_y, x_vals, y_vals, traj_id=traj_id, label=choice)
                                
                                # save into the dictionary if trajectory looks good
                                if is_good:
                                    day_standard_trajectories[choice] = (x_vals, y_vals)
                                    choice_empty = False
                                    break
                                else:
                                    # this is so that i can know where it starts and ends later
                                    day_standard_trajectories[choice] = (x_vals, y_vals)
                        else:
                            # save so that i can plot out potentially better trajectories later
                            trajectories_not_found[day] = {choice: (all_x, all_y)}
        
        print(f"adding to rat standard trajectories with {day}")
        # put all the trajectories into rat_standard_trajectories
        rat_standard_trajectories[day] = day_standard_trajectories
    
    # after going through all the days, go through any choices that didn't have a trajectory and try out new ones
    for day, day_group in trajectories_not_found.items():
        for choice, all_vals in day_group.items():
            # first plot the inappropriate choice so i know where it starts and ends
            all_x, all_y = all_vals
            trajectory = rat_standard_trajectories[day][choice]
            traj_x, traj_y = trajectory
            
            label = f"{day} {choice} original trajectory"
            plot_trajectory_animation(all_x, all_y, traj_x, traj_y, label=label)
            
            # now plot the trajectories for other days to see if there is another that will fit
            for other_day, other_day_group in rat_standard_trajectories.items():
                # skip the day that i'm currently looking at
                if other_day == day:
                    continue
                
                for other_choice, other_vals in other_day_group.items():
                    if other_choice != choice: # skip other choices
                        continue

                    # so now it's another day with the same choice
                    new_x, new_y = other_vals
                    
                    label = f"{other_day} on {day} for {choice}"
                    is_good = False # reset before each trajectory
                    plot_trajectory_animation(all_x, all_y, new_x, new_y, label=label)
                    
                    # now see if it's a fit
                    if is_good:
                        rat_standard_trajectories[day][choice] = (new_x, new_y)
                        break
                break
    
    # save rat_standard_trajectories
    rows = []
    for day, day_standard_trajectories in rat_standard_trajectories.items():
        for choice, (trajectory_x, trajectory_y) in day_standard_trajectories.items():
            # Append a row as a dictionary
            rows.append({
                "day": day,
                "choice": choice,
                "x": trajectory_x,
                "y": trajectory_y
            })

    # Create DataFrame from rows
    rat_standard_trajectories_df = pd.DataFrame(rows)
    
    save_path = os.path.join(standard_traj_path, f"{rat}_standard_trajectories.csv")
    rat_standard_trajectories_df.to_csv(save_path)
