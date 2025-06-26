import os
import ast
import pandas as pd

from config.paths import paths
from utilities import math_utils
from utilities import conversion_utils
from visualization import trajectory_plots

rat = "BP06"
day = "Day7"

dlc_path = paths.cleaned_dlc / rat / f"{day}_coordinates.csv"
dlc_pd = pd.read_csv(dlc_path)
base_x = dlc_pd["x"]
base_y = dlc_pd["y"]
all_x, all_y = conversion_utils.convert_pixels_to_cm(base_x, base_y)

zidphi_path = paths.vte_values / rat / "zIdPhis.csv"
zidphi_pd = pd.read_csv(zidphi_path)
zidphis = {}
for index, row in zidphi_pd.iterrows():
    Id = row["ID"]
    if day not in Id:
        continue
    zidphi_val = row["zIdPhi"]
    zidphis[Id] = zidphi_val

print(zidphis)
vte_path = paths.vte_values / rat / day / f"{rat}_{day}_trajectories.csv"
vte_pd = pd.read_csv(vte_path)
for index, row in vte_pd.iterrows():
    traj_id = row["ID"]
    
    choice = row["Choice"]
    IdPhi = row["IdPhi"]
    IdPhi = math_utils.round_to_sig_figs(IdPhi)
    
    x_vals = row["X Values"]
    y_vals = row["Y Values"]
    x_vals = ast.literal_eval(x_vals)
    y_vals = ast.literal_eval(y_vals)
    
    try:
        zIdPhi = zidphis[traj_id]
    except KeyError:
        print(traj_id)
        zIdPhi = "NaN"
        label = "IdPhi: " + str(IdPhi) + " zIdPhi: " + zIdPhi + " for choice: " + choice
    else:
        zIdPhi = math_utils.round_to_sig_figs(zIdPhi)
        label = "IdPhi: " + str(IdPhi) + " zIdPhi: " + str(zIdPhi) + " for choice: " + choice
    
    jpg_path = paths.processed / "IdPhi_zIdPhi_values" / f"{rat}_{day}"
    os.makedirs(jpg_path, exist_ok=True) # Create directory if it doesn't exist
    
    trajectory_plots.plot_trajectory(all_x, all_y,(x_vals, y_vals), title=label, traj_id=traj_id, save=jpg_path)
    #trajectory_plots.plot_trajectory_animation(all_x, all_y, x_vals, y_vals, traj_id=traj_id)