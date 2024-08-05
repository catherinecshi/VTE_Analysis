import os
import gc
import datetime
import numpy as np
import pandas as pd
#import dask.dataframe as dd
from sklearn.cluster import DBSCAN
from scipy.stats import zscore

from src import data_processing
from src import helper
from src import plotting
from src import calculating_VTEs
from src import creating_zones

def filter_dataframe(df, tracking="greenLED", std_multiplier=7, eps=70, min_samples=40, max_interpolation_distance=100, start_index=None): # currently keeps original indices
    """
    Filters dataframes. Check to make sure it's working properly. Generally, more than 100 filtered out points is bad
    Keeps the original indices of the DataFrame

    Args:
        df (pandas.DataFrame): the data frame to be filtered
        track_part (str, optional): part of rat to be used for their position. Defaults to 'greenLED'.
        std_multiplier (int, optional): multiplier for std to define threshold beyond which jumps are excluded. 
                                        Defaults to 7.
        eps (int, optional): maximum distance between two samples for one to be considered as in the 
                             neighbourhood of another for DBCSCAN. Defaults to 70.
        min_samples (int, optional): number of samples in a neighbourhood for a point to be 
                                     considered a core point for DBSCAN. Defaults to 40.
        distance_threshold (int, optional): distance threshold for identifying jumps in tracking data. Defaults to 190.
        start_index (int, optional): index from which to start filtering. Defaults to None.

    Returns:
        x & y : panda.Series : filtered and interpolated coordinates for x and y
    
    Procedure:
    1. filters based on the likelihood values
    2. filters out points before start_index if provided
    3. DBSCAN
    4. filters out based on std thresholds
    5. filters based on jumps
    6. interpolate
    """
    
    # modify a copy instead of the original
    # also filter based on likelihood values
    likely_data = df[df[(tracking, "likelihood")] > 0.95].copy()
    unlikely_data = len(df) - len(likely_data)
    
    # filter out points before the rat has started its first trial
    if start_index:
        likely_data = likely_data[likely_data.index >= start_index]
    
    if likely_data.empty:
        print(f"{helper.CURRENT_RAT} on {helper.CURRENT_DAY} empty after filtering by likelihood")
        return None, None, None
    
    # DBSCAN Cluster analysis
    coordinates = likely_data[[tracking]].copy()[[(tracking, "x"), (tracking, "y")]]
    coordinates.dropna(inplace = True) # don't drop nan for dbscan
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_
    noise_points_count = (labels == -1).sum() # so ik how many points were filtered out

    filtered_indices = labels != -1 # filter out noise
    filtered_data = likely_data[filtered_indices].copy()
    
    # calculate thresholds
    diff_x = df[(tracking, "x")].diff().abs()
    diff_y = df[(tracking, "y")].diff().abs()
    threshold_x = diff_x.std() * std_multiplier
    threshold_y = diff_y.std() * std_multiplier
    
    # calculate diff between current point and last non-jump point
    last_valid_index = 0
    jump_indices = [] # just to see how many points are jumped over
    
    for i in range(1, len(filtered_data)):
        diff_x = abs(filtered_data.iloc[i][(tracking, "x")] - filtered_data.iloc[last_valid_index][(tracking, "x")])
        diff_y = abs(filtered_data.iloc[i][(tracking, "y")] - filtered_data.iloc[last_valid_index][(tracking, "y")])
        
        # check for jumps
        if diff_x > threshold_x or diff_y > threshold_y:
            # mark as NaN
            filtered_data.at[filtered_data.index[i], (tracking, "x")] = np.nan
            filtered_data.at[filtered_data.index[i], (tracking, "y")] = np.nan
            jump_indices.append(i)
        else:
            # udpate last valid index
            last_valid_index = i
    
    # interpolating
    for axis in ["x", "y"]:
        valid_points = filtered_data[filtered_data[(tracking, axis)].notna()]
        for i in range(len(valid_points) - 1):
            start_idx = valid_points.index[i]
            end_idx = valid_points.index[i + 1]
            if end_idx - start_idx > 1:  # There are NaNs between these points
                if abs(filtered_data.at[end_idx, (tracking, axis)] - \
                   filtered_data.at[start_idx, (tracking, axis)]) <= max_interpolation_distance:
                    filtered_data.loc[start_idx:end_idx, (tracking, axis)] = \
                    filtered_data.loc[start_idx:end_idx, (tracking, axis)].interpolate()

    # final coordinate points
    x_coords = filtered_data[(tracking, "x")]
    y_coords = filtered_data[(tracking, "y")]
    try:
        time = filtered_data[("time", "time")]
    except Exception:
        print(helper.CURRENT_RAT, helper.CURRENT_DAY)
    
    # check how many points have been filtered out
    #total_filtered_out = len(df) - len(x_coords)
    #points_filtered = {"total": total_filtered_out, "likelihood": unlikely_data, "start": start_index,
                       #"DBSCAN": noise_points_count, "jumps": len(jump_indices)}
    
    return x_coords, y_coords, time#, points_filtered

def smooth_points(points, span=3):
    new_points = []
    for i, point in enumerate(points):
        if i < span:
            new_points.append(point)
            continue
            
        if i > len(points) - span:
            new_points.append(point)
            continue
        
        averaged_across = []
        for j in range(i - span, i + span):
            averaged_across.append(points.iloc[j])
        
        new_points.append(np.mean(averaged_across))
    
    return new_points

def check_too_filtered(df, coords_x, coords_y, tracking, save=None):
    unlikely_data = df[df[(tracking, "likelihood")] < 0.95].copy()
    unlikely_x = unlikely_data[(tracking, "x")]
    unlikely_y = unlikely_data[(tracking, "y")]
    
    plotting.create_populational_scatter_plot(coords_x, coords_y, unlikely_x, unlikely_y, save=save)

data_path = os.path.join(helper.BASE_PATH, "data", "VTE_Data")
IMPLANTED_RATS = ["BP06", "BP07", "BP12", "BP13", "TH405", "TH508", "BP20", "TH510", "TH605", "TH608"]

data_structure = data_processing.load_data_structure(data_path)

for rat, day_group in data_structure.items():
    filtered_info = []
    
    if "BP06" in rat or "TH608" in rat or "BP08" in rat:
        continue
    
    # determine which part to track rats on
    if any(rat == rat_ID for rat_ID in IMPLANTED_RATS):
        track_part = "greenLED"
    else:
        track_part = "nose"
    
    helper.update_rat(rat)
    
    # make folder if it doesn't exist for rat already
    rat_folder = os.path.join(helper.BASE_PATH, "processed_data", "dlc_data", rat)
    if not os.path.exists(rat_folder):
        os.mkdir(rat_folder)
    
    for day, data in day_group.items():
        DLC_df = data_structure[rat][day]["DLC_tracking"]
        SS_log = data_structure[rat][day]["stateScriptLog"]
        timestamps = data_structure[rat][day]["videoTimeStamps"]
        
        helper.update_day(day)
        
        # skip if no DLC or empty DLC
        if DLC_df is None or DLC_df.empty:
            print(f"no or empty DLC for {rat} on {day}")
            continue
        
        # skip if already made
        save_path_coords = os.path.join(rat_folder, f"{day}_coordinates.csv")
        #if os.path.exists(save_path_coords):
            #continue
        """
        if os.path.exists(save_path_coords):
            two_days_ago = datetime.datetime.now() - datetime.timedelta(days=2)
            stat = os.stat(save_path_coords)
            try:
                creation_time = stat.st_birthtime
            except AttributeError:
                creation_time = stat.st_mtime
                print(f"Attribution error for {rat} on {day}")
            
            creation_time_dt = datetime.datetime.fromtimestamp(creation_time)
            if creation_time_dt < two_days_ago:
                continue
        """
        if timestamps is not None:
            timestamps = helper.check_timestamps(DLC_df, timestamps) # initial check of everything
            try:
                DLC_df[("time", "time")] = timestamps
            except ValueError:
                print(rat, day)
                print(f"{len(DLC_df)}, {len(timestamps)}")
                continue
        
        if timestamps is not None and SS_log is not None:
            try:
                trial_starts = helper.get_video_trial_starts(timestamps, SS_log)
            except Exception as e:
                print(f"{e} for {rat} on {day}")
                #x, y, times, points_filtered_out = filter_dataframe(DLC_df, tracking=track_part)
                x, y, times = filter_dataframe(DLC_df, tracking=track_part)
            else:
                try:
                    first_trial_start = next(iter(trial_starts)) # get the first trial start time to pass into filtering
                except StopIteration as si:
                    print(f"{si} for {rat} on {day}")
                    #x, y, times, points_filtered_out = filter_dataframe(DLC_df, tracking=track_part)
                    x, y, times = filter_dataframe(DLC_df, tracking=track_part)
                else:
                    #x, y, times, points_filtered_out = filter_dataframe(DLC_df, tracking=track_part, start_index=first_trial_start)
                    x, y, times = filter_dataframe(DLC_df, tracking=track_part, start_index=first_trial_start)
        else:
            #x, y, times, points_filtered_out = filter_dataframe(DLC_df, tracking=track_part)
            x, y, times = filter_dataframe(DLC_df, tracking=track_part)
        
        # check for NoneTypes
        if x is None or y is None:
            continue
        
        new_x = smooth_points(x)
        new_y = smooth_points(y)

        coords_df = pd.DataFrame({"x": new_x, "y": new_y, "times": times})
        coords_df.to_csv(save_path_coords, index=False)
        
        #check_too_filtered(DLC_df, x, y, track_part, save=rat_folder)
        """
        save_filter_info = {"rat": rat, "day": day, "total": points_filtered_out["total"], 
                            "likelihood": points_filtered_out["likelihood"], "start": points_filtered_out["start"],
                            "DBSCAN": points_filtered_out["DBSCAN"], "jumps": points_filtered_out["jumps"]}
        filtered_info.append(save_filter_info)
        """
        # garbage disposal
        del DLC_df, SS_log, timestamps, x, y, new_x, new_y, coords_df#, save_filter_info, points_filtered_out
        gc.collect()

    """
    save_path_filter = os.path.join(helper.BASE_PATH, "processed_data", f"{rat}_filtered_points_info.csv")
    filtered_info_df = pd.DataFrame(filtered_info)
    filtered_info_df.to_csv(save_path_filter)
    """
    
    del filtered_info#, filtered_info_df
    gc.collect()

print("DLC finished")


base_path = "/Users/catpillow/Documents/VTE_Analysis"
"""dlc_path = os.path.join(base_path, "processed_data", "dlc_data")

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
            
            if os.path.exists(hull_path):
                modification_time = os.path.getmtime(hull_path)
                modification_datetime = datetime.datetime.fromtimestamp(modification_time)
                if modification_datetime > datetime.datetime.now() - datetime.timedelta(days=2):
                    print(f"{hull_path} exists and was modified within the last 2 days. Skipping...")
                    continue
            
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
            creating_zones.plot_hull(x, y, intersection_points, densest_cluster_points, hull, save=save_hull_path)
            np.save(hull_path, densest_cluster_points[hull.vertices])"""

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
            
            if not any(rat in process_rat for process_rat in helper.IMPLANTED_RATS):
                continue
            
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
    
    #big_z_df = pd.DataFrame(many_z_df)
    big_z_df_path = os.path.join(rat_path, "zIdPhis.csv")
    many_z_df.to_csv(big_z_df_path, index=False)
