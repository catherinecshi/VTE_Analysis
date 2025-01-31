import os
import ast
import pandas as pd

from src import helper
from src import plotting

rat = "TH605"
day = "Day11"

dlc_path = os.path.join(helper.BASE_PATH, "processed_data", "cleaned_dlc", rat, f"{day}_coordinates.csv")
dlc_pd = pd.read_csv(dlc_path)
all_x = dlc_pd["x"]
all_y = dlc_pd["y"]

zidphi_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values", rat, "zIdPhis.csv")
zidphi_pd = pd.read_csv(zidphi_path)
zidphis = {}
for index, row in zidphi_pd.iterrows():
    Id = row["ID"]
    if day not in Id:
        continue
    zidphi_val = row["zIdPhi"]
    zidphis[Id] = zidphi_val

print(zidphis)
vte_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_values", rat, day, "trajectories.csv")
vte_pd = pd.read_csv(vte_path)
for index, row in vte_pd.iterrows():
    traj_id = row["ID"]
    
    choice = row["Choice"]
    IdPhi = row["IdPhi"]
    IdPhi = helper.round_to_sig_figs(IdPhi)
    
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
        zIdPhi = helper.round_to_sig_figs(zIdPhi)
        label = "IdPhi: " + str(IdPhi) + " zIdPhi: " + str(zIdPhi) + " for choice: " + choice
    
    jpg_path = os.path.join(helper.BASE_PATH, "processed_data", "IdPhi_zIdPhi_values", f"{rat}_{day}")
    os.makedirs(jpg_path, exist_ok=True) # Create directory if it doesn't exist
    
    plotting.plot_trajectory(all_x, all_y,(x_vals, y_vals), title=label, traj_id=traj_id, save=jpg_path)