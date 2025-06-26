import os
import numpy as np
import pandas as pd
from numpy.linalg import norm

from config.paths import paths

standard_trajectories_path = paths.standard_trajectories
all_trajectories_path = paths.vte_values

for rat in os.listdir(all_trajectories_path):
    if ".DS_Store" in rat:
        continue

    rat_path = os.path.join(all_trajectories_path, rat)
    for day in os.listdir(rat_path):
        if ".DS_Store" in rat:
            continue

        trajectories_path = os.path.join(rat_path, day, "trajectories.csv")
        trajectories_pd = pd.read_csv(trajectories_path)
        
        # find the standard_trajectories for the day
        rat_standard_trajectories_path = os.path.join(standard_trajectories_path, f"{rat}_standard_trajectories.csv")
        rat_standard_trajectories_pd = pd.read_csv(rat_standard_trajectories_path)
        
        for index, row in trajectories_pd.iterrows():
            traj_id = row["ID"]
            choice = row["Choice"]
            trial_type = row["Trial Type"]
            x_vals = row["X"]
            y_vals = row["Y"]
            
            # retrieve standard deviation info
            standard_row = rat_standard_trajectories_pd.loc[(rat_standard_trajectories_pd["day"] == day) & (rat_standard_trajectories_pd["choice"] == choice)]
            
            # check if it's empty
            if standard_row.empty:
                print(f"{rat} on {day} for {choice} doesn't have a standard deviation")
                continue
            
            standard_x_vals = standard_row["x"]
            standard_y_vals = standard_row["y"]

            # create line segments between each standard trajectory point and point + 1, then get distance of each
            # x_val and y_val to those line segments
            smallest_distances = []
            for i, x_val in enumerate(x_vals):
                y_val = y_vals[i] # corresponding y value
                point = np.array([x_val, y_val])
                smallest_d = 0.0
                
                # creating line segments
                for j, standard_x in enumerate(standard_x_vals):
                    if i == 0:
                        continue # skip the first one since only one point
                    
                    current_point = np.array([standard_x, standard_y_vals[j]])
                    previous_point = np.array([standard_x_vals[j - 1], standard_y_vals[j - 1]])
                    
                    # calculating perpendicular distance between point and line
                    d = np.cross(previous_point - current_point, point - current_point) / norm(previous_point - current_point)
                    
                    # check if this is the smallest distance encountered thus far
                    if d < smallest_d:
                        smallest_d = d
                
                # store the smallest d
                smallest_distances.append(smallest_d)
            
            # check if the distances cross the threshold
            