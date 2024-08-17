import os
import numpy as np
import pandas as pd
from scipy.stats import zscore

from src import helper
from src import data_processing
from src import creating_zones
from src import calculating_VTEs

### CREATING CENTRE HULL
base_path = "/Users/catpillow/Documents/VTE_Analysis"
dlc_path = os.path.join(base_path, "processed_data", "dlc_data")

for rat in os.listdir(dlc_path):
    rat_path = os.path.join(dlc_path, rat)
    helper.update_rat(rat)
    if "BP11" in rat:
        continue
    
    for root, _, files in os.walk(rat_path):
        for f in files:
            if not "coordinates" in f:
                continue
            
            file_path = os.path.join(root, f)
            parts = f.split("_")
            day = parts[0]
            helper.update_day(day)
            
            df = pd.read_csv(file_path)
            x = df["x"]
            y = df["y"]
            
            if x.empty or y.empty:
                print(f"x or y is empty for {rat} on {day}")
                continue
            
            save_hull_path = os.path.join(helper.BASE_PATH, "processed_data", "VTE_Data", rat)
            if not os.path.exists(save_hull_path):
                os.makedirs(save_hull_path)
            hull_path = os.path.join(save_hull_path, f"{helper.CURRENT_RAT}_{helper.CURRENT_DAY}_hull_vertices.npy")
            
            lines = creating_zones.create_lines(x, y)
            
            coverages, starts, ends = creating_zones.calculate_line_coverages(x, y, lines)
            
            avg = np.mean(coverages)
            std = np.std(coverages)
            std_up = avg + std
            
            updated_lines, updated_starts, updated_ends = creating_zones.make_new_lines(lines, coverages, starts, ends, threshold=std_up)
            
            intersections = creating_zones.find_intersections(updated_lines, updated_starts, updated_ends)
            intersection_points = np.array(intersections) # np array for DBSCAN to work
            
            if intersection_points.size == 0:
                print(f"intersection empty for {rat} on {day}")
                continue
            
            hull, densest_cluster_points = creating_zones.make_convex_hull(intersection_points)
            creating_zones.plot_hull_with_intx_points(x, y, intersection_points, densest_cluster_points, hull, save=save_hull_path)
            np.save(hull_path, densest_cluster_points[hull.vertices])



### GETTING zIdPhi VALUES
data_path = os.path.join(base_path, "data", "VTE_Data")
data_structure = data_processing.load_data_structure(data_path)

vte_path = os.path.join(base_path, "processed_data", "VTE_data")
for rat in os.listdir(vte_path):
    rat_path = os.path.join(vte_path, rat)
    if not os.path.isdir(rat_path):
        continue # skip files

    for root, dirs, files in os.walk(rat_path):
        for f in files:
            parts = f.split("_")
            rat = parts[0]
            day = parts[1]
            
            try:
                save_path = os.path.join(base_path, "processed_data", "VTE_values", rat, day)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                _, _ = calculating_VTEs.quantify_VTE(data_structure, rat, day, save=save_path)
            except Exception as error:
                print(f"error in rat_VTE_over_session - {error} on day {day} for {rat}")

values_path = os.path.join(base_path, "processed_data", "VTE_values")

IdPhis_across_days = {} # this is so it can be zscored altogether
IdPhis_in_a_day = 0
for rat in os.listdir(values_path):
    if ".DS" in rat or ".csv" in rat:
        continue
    
    rat_path = os.path.join(values_path, rat)
    big_df = pd.DataFrame()
    for day in os.listdir(rat_path):
        day_path = os.path.join(rat_path, day)
        if ".DS" in day:
            continue
        
        for root, _, files in os.walk(day_path):
            for f in files:
                if "trajectories.csv" not in f:
                    continue
                
                file_path = os.path.join(root, f)
                traj_info = pd.read_csv(file_path, header=0)
                
                if all(col in traj_info.columns for col in ["IdPhi", "Choice", "ID", "Trial Type"]):
                    IdPhi_values = traj_info["IdPhi"]
                    choices = traj_info["Choice"]
                    traj_ids = traj_info["ID"]
                    trial_type = traj_info["Trial Type"]
                    rows = {"ID": traj_ids, "Day": day, "Choice": choices, "Trial_Type": trial_type, "IdPhi": IdPhi_values}
                    df = pd.DataFrame(rows)
                    big_df = pd.concat([big_df, df], ignore_index=True)
                else:
                    print(f"missing columns in {rat} on {day}")
    
    try:
        grouped_by_choice = big_df.groupby(by="Choice")
    except Exception:
        print(f"error with groupby for {rat}")
    else:
        print("grouping successful")
        
    many_z_df = pd.DataFrame()
    for choice, choice_group in grouped_by_choice:
        zIdPhis = zscore(choice_group["IdPhi"])
        choice_group["zIdPhi"] = zIdPhis
        many_z_df = pd.concat([many_z_df, choice_group], ignore_index=True)
    
    big_z_df_path = os.path.join(rat_path, "zIdPhis.csv")
    many_z_df.to_csv(big_z_df_path, index=False)
