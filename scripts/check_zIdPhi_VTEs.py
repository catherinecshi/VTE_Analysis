import os
import ast
import numpy as np
import pandas as pd

from src import helper
from src import plotting

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
            zIdPhi_csv = pd.read_csv(file_path)
            
            for index, row in zIdPhi_csv.iterrows():
                traj_id = row["ID"]
                zIdPhi = row["zIdPhi"]
                zIdPhis[traj_id] = zIdPhi

# then get all the x and y values for the scatter plots
dlc_path = os.path.join(helper.BASE_PATH, "processed_data", "cleaned_dlc")

for rat in os.listdir(vte_path):
    if ".DS_Store" in rat:
        continue

    rat_path = os.path.join(vte_path, rat)
    for day in os.listdir(rat_path):
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
                
                # now plot each trajectory
                for index, row in trajectories_csv.iterrows():
                    traj_id = row["ID"]
                    zIdPhi = zIdPhis[traj_id]
                    IdPhi = row["IdPhi"]
                    x_vals = row["X Values"]
                    y_vals = row["Y Values"]
                    
                    rounded_zIdPhi = helper.round_to_sig_figs(zIdPhi)
                    rounded_IdPhi = helper.round_to_sig_figs(IdPhi)
                    label = "zIdPhi: " + str(rounded_zIdPhi) + ", IdPhi: " + str(rounded_IdPhi)
                    
                    # make sure the typing is correct
                    x_vals = ast.literal_eval(x_vals)
                    y_vals = ast.literal_eval(y_vals)
                    
                    plotting.plot_trajectory_animation(all_x_vals, all_y_vals, x_vals, y_vals,
                                                       title=traj_id, label=label)