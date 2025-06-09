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

is_VTE = None

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
    
    def is_VTE(event):
        global is_VTE
        is_VTE = True
        print("VTE")
        plt.close()
    
    def not_VTE(event):
        global is_VTE
        is_VTE = False
        print("Not VTE")
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
    btn1 = Button(axes1, "VTE")
    btn1.on_clicked(is_VTE)
    
    axes2 = plt.axes((0.2, 0.05, 0.1, 0.075))
    btn2 = Button(axes2, "not VTE")
    btn2.on_clicked(not_VTE)

    plt.show()

# show a couple of trajectories for each choice for each rat
# first get zIdPhi values
vte_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values")
zIdPhis = {} # {traj_id: zIdPhi}
for rat in os.listdir(vte_path):
    if ".DS_Store" in rat:
        continue

    rat_path = os.path.join(vte_path, rat)
    for root, _, files in os.walk(rat_path):
        for file in files:
            if "zIdPhi" not in file:
                continue

            file_path = os.path.join(root, file)
            zIdPhi_csv = pd.read_csv(file_path).dropna() # skip trajectories with nan values
            
            for index, row in zIdPhi_csv.iterrows():
                traj_id = row["ID"]
                zIdPhi = row["zIdPhi"]
                zIdPhis[traj_id] = zIdPhi

# then get all the x and y values for the scatter plots
dlc_path = os.path.join(helper.BASE_PATH, "processed_data", "cleaned_dlc")

for rat in os.listdir(vte_path):
    if ".DS_Store" in rat or "BP06" in rat or "BP08" in rat or "BP07" in rat or "BP09" in rat or "TH510" in rat or "TH605" in rat or "BP13" in rat or "BP15" in rat or "BP22" in rat:
        continue
    
    # list to see when new trial types are added so all the trial types will be seen and judged
    trial_types = set()
    trial_count = {}
    VTE_data = pd.DataFrame(columns=["ID", "X", "Y", "IdPhi", "zIdPhi", "VTE"])

    rat_path = os.path.join(vte_path, rat)
    for day in sorted(os.listdir(rat_path), key=lambda x: int(re.search(r"\d+", x).group() if re.search(r"\d+", x) else 0)):
        if ".DS_Store" in day:
            continue

        day_path = os.path.join(rat_path, day)
        for root, _, files in os.walk(day_path):
            for file in files:
                if "trajectories.csv" not in file:
                    continue

                file_path = os.path.join(root, file)
                trajectories_csv = pd.read_csv(file_path)
                
                # get the x and y values for scatter plotting
                coordinates_file_name = day + "_" + "coordinates.csv"
                coordinates_path = os.path.join(dlc_path, rat, coordinates_file_name)
                coordinates_csv = pd.read_csv(coordinates_path)
                all_x_vals = coordinates_csv["x"]
                all_y_vals = coordinates_csv["y"]
                
                # choices that haven't been added to the set yet
                try:
                    new_choices = trajectories_csv[~trajectories_csv["Choice"].isin(trial_types)]
                except KeyError:
                    print(f"{rat} on {day} do not have 'Choice' in trajectories csv")
                    continue
                
                if new_choices.empty:
                    continue # skip if there are no new choices
                elif not new_choices.empty: # new arm introduced
                    # go through each new choice by itself
                    unique_choices = new_choices["Choice"].unique()
                    for choice in unique_choices:
                        print(choice)
                        
                        choice_rows = new_choices[new_choices["Choice"] == choice]
                        
                        for i in range(len(choice_rows)):
                            if i > 5:
                                continue
                            
                            row = choice_rows.iloc[i]
                            traj_id = row["ID"]
                            try:
                                zIdPhi = zIdPhis[traj_id]
                            except KeyError:
                                print(f"{traj_id} not in zIdPhis")
                                continue
                            
                            if zIdPhi == 0:
                                continue
                            
                            # count the number of times each choice has appeared is tracked
                            if choice in trial_count:
                                if trial_count[choice] > 25:
                                    break
                                else:
                                    trial_count[choice] += 1
                            else:
                                trial_count[choice] = 1
                            
                            traj_id = row["ID"]
                            zIdPhi = zIdPhis[traj_id]
                            IdPhi = row["IdPhi"]
                            x_vals = row["X Values"]
                            y_vals = row["Y Values"]
                            length = row["Length"]
                            trial_type = row["Trial Type"]
                            
                            rounded_zIdPhi = math_utils.round_to_sig_figs(zIdPhi)
                            rounded_IdPhi = math_utils.round_to_sig_figs(IdPhi)
                            label = "zIdPhi: " + str(rounded_zIdPhi) + ", IdPhi: " + str(rounded_IdPhi)
                            
                            # make sure the typing is correct
                            x_vals = ast.literal_eval(x_vals)
                            y_vals = ast.literal_eval(y_vals)
                            
                            is_VTE = None
                            plot_trajectory_animation(all_x_vals, all_y_vals, x_vals, y_vals,
                                                        traj_id=traj_id, label=label)
                            
                            if is_VTE is not None:
                                new_row = pd.DataFrame({"ID": traj_id, "X": [str(x_vals)], "Y": [str(y_vals)],
                                                        "Choice": choice, "Trial Type": trial_type, "Length": length,
                                                        "IdPhi": IdPhi, "zIdPhi": zIdPhi, "VTE": is_VTE})
                                VTE_data = pd.concat([VTE_data, new_row], ignore_index=True)
                            else:
                                print(f"is_VTE is none for {traj_id}")
                
    vte_data_path = os.path.join(helper.BASE_PATH, "processed_data", "manual_VTE", f"{rat}_VTEs.csv")
    VTE_data.to_csv(vte_data_path)