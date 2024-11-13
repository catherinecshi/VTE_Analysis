"""
generates jpgs and/or gifs for trajectories i have manually sorted VTEs on.
shows idphi, zidphi, traj_id and choice with trajectory
"""

import os
import ast
import pandas as pd

from src import helper
from src import plotting

vte_path = os.path.join(helper.BASE_PATH, "processed_data", "manual_VTE")
values_path = os.path.join(helper.BASE_PATH, "processed_data", "IdPhi_zIdPhi_values")
for root, _, files in os.walk(vte_path):
    for file in files:
        if "VTEs" not in file:
            continue
        
        parts = file.split("_")
        rat = parts[0]

        vte_pd = pd.read_csv(os.path.join(root, file))
        for index, row in vte_pd.iterrows():
            traj_id = row["ID"]
            sections = traj_id.split("_")
            day = sections[1]
            
            dlc_path = os.path.join(helper.BASE_PATH, "processed_data", "cleaned_dlc", rat, f"{day}_coordinates.csv")
            dlc_pd = pd.read_csv(dlc_path)
            all_x = dlc_pd["x"]
            all_y = dlc_pd["y"]
            
            choice = row["Choice"]
            IdPhi = row["IdPhi"]
            zIdPhi = row["zIdPhi"]
            IdPhi = helper.round_to_sig_figs(IdPhi)
            zIdPhi = helper.round_to_sig_figs(zIdPhi)
            
            x_vals = row["X"]
            y_vals = row["Y"]
            x_vals = ast.literal_eval(x_vals)
            y_vals = ast.literal_eval(y_vals)
            
            label = "IdPhi: " + str(IdPhi) + " zIdPhi: " + str(zIdPhi) + " for choice: " + choice

            jpg_path = os.path.join(values_path, "JPGs")
            gif_path = os.path.join(values_path, "GIFs")
            plotting.plot_trajectory(all_x, all_y,(x_vals, y_vals), title=label, traj_id=traj_id, save=jpg_path)
            #plotting.plot_trajectory_animation(all_x, all_y, x_vals, y_vals, traj_id=traj_id, title=label, save=gif_path)